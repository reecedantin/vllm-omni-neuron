# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 DiT on CPU: the prefix/target split must reproduce upstream's joint forward.

Runs on the random-weight tiny checkpoint (``qwen_image_tiny.py``), fp32. Covers text-to-image
(text prefix, right-padded to a bucket) and image-conditioned generation (condition-image
blocks inside the prefix, block-causal), for two denoising steps against upstream's KV cache.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from .test_qwen_image_tiny import build

    return build(str(tmp_path_factory.mktemp("qwen21tiny")))


def _upstream(tiny_dir):
    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    return QwenImage21Transformer2DModel.from_pretrained(
        tiny_dir, subfolder="transformer", torch_dtype=torch.float32
    ).eval()


def _ours(tiny_dir):
    from vllm_omni_neuron.diffusion.models.qwen_image.transformer_qwenimage21 import (
        NeuronQwenImage21Transformer,
        QwenImage21DiTConfig,
    )

    m = NeuronQwenImage21Transformer(
        QwenImage21DiTConfig.from_model_dir(tiny_dir), dtype=torch.float32
    )
    m.load_weights(tiny_dir, "cpu")
    return m


def _case(cond: bool, n_text: int = 13, pad_text: int = 3):
    """VLM-side inputs: text embeddings (with right padding), img_mask incl. target slots."""
    torch.manual_seed(0)
    th, tw = 4, 6  # target latent grid (tokens)
    shapes, mask, cond_lat = [], [], None
    if cond:
        ch, cw = 4, 4  # one condition image: 16 tokens = 4 VLM slots, after 5 text tokens
        mask = [False] * 5 + [True] * (ch * cw // 4) + [False] * (n_text - 5)
        shapes.append((1, ch, cw))
        cond_lat = torch.randn(1, ch * cw, 16)
    else:
        mask = [False] * n_text
    shapes.append((1, th, tw))
    s_text = len(mask) + pad_text  # text embeddings include right padding
    emb = torch.randn(1, s_text, 128)
    valid = torch.ones(1, s_text, dtype=torch.bool)
    valid[:, s_text - pad_text :] = False
    emb[~valid] = 0
    vlm_mask = torch.tensor(mask + [False] * pad_text + [True] * (th * tw // 4))
    return emb, valid, vlm_mask, shapes, cond_lat, th * tw


@pytest.mark.parametrize("cond", [False, True])
def test_prefix_target_matches_upstream(vllm_single_rank, tiny_dir, cond):
    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.transformer_qwenimage21 import (
        QwenImage21KVCache,
    )
    from vllm_omni_neuron.diffusion.models.qwen_image.transformer_qwenimage21 import (
        assemble_prefix_inputs,
        build_layout,
    )

    emb, valid, vlm_mask, shapes, cond_lat, n_tgt = _case(cond)
    up, ours = _upstream(tiny_dir), _ours(tiny_dir)
    cache = QwenImage21KVCache(ours.cfg.num_layers)
    lay = build_layout(ours.cfg, vlm_mask, valid, shapes, bucket=64)
    txt, img = assemble_prefix_inputs(lay, emb, cond_lat, ours.cfg.in_channels, torch.float32)
    with torch.no_grad():
        kv = ours.forward_prefix(txt, img, lay.is_img, lay.cos_p, lay.sin_p, lay.prefix_bias)
        for step, (t, mode) in enumerate(((0.83, "extract"), (0.41, "cached"))):
            lat = torch.randn(1, n_tgt, 16)
            hs = lat if cond_lat is None else torch.cat([cond_lat, lat], dim=1)
            ref = up(
                hidden_states=hs,
                encoder_hidden_states=emb,
                timestep=torch.tensor([t]),
                img_shapes=[shapes],
                img_mask=vlm_mask[None],
                encoder_hidden_states_mask=valid,
                kv_cache=cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0][:, -n_tgt:]
            out = ours.forward_target(
                lat, torch.tensor([t]), lay.cos_t, lay.sin_t, lay.target_bias, *kv
            )
            assert _rel(out, ref) < 1e-4, (cond, step, _rel(out, ref))


def test_no_cache_equivalence(vllm_single_rank, tiny_dir):
    """Upstream without the KV cache (one joint forward) equals ours too."""
    from vllm_omni_neuron.diffusion.models.qwen_image.transformer_qwenimage21 import (
        assemble_prefix_inputs,
        build_layout,
    )

    emb, valid, vlm_mask, shapes, cond_lat, n_tgt = _case(True)
    up, ours = _upstream(tiny_dir), _ours(tiny_dir)
    lay = build_layout(ours.cfg, vlm_mask, valid, shapes, bucket=48)
    txt, img = assemble_prefix_inputs(lay, emb, cond_lat, ours.cfg.in_channels, torch.float32)
    lat = torch.randn(1, n_tgt, 16)
    with torch.no_grad():
        ref = up(
            hidden_states=torch.cat([cond_lat, lat], 1),
            encoder_hidden_states=emb,
            timestep=torch.tensor([0.5]),
            img_shapes=[shapes],
            img_mask=vlm_mask[None],
            encoder_hidden_states_mask=valid,
            return_dict=False,
        )[0][:, -n_tgt:]
        kv = ours.forward_prefix(txt, img, lay.is_img, lay.cos_p, lay.sin_p, lay.prefix_bias)
        out = ours.forward_target(
            lat, torch.tensor([0.5]), lay.cos_t, lay.sin_t, lay.target_bias, *kv
        )
    assert _rel(out, ref) < 1e-4, _rel(out, ref)


def test_attention_qblock_policy(monkeypatch):
    """Query-row blocking is exact on CPU and is used only up to ATTN_QBLOCK_MAX_ROWS rows (the
    blocked graph miscompiles on device at 2048x2048, see common.py)."""
    from vllm_omni_neuron.diffusion.models.qwen_image import common

    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, n, 16, generator=g) for n in (96, 112, 112))
    bias = torch.zeros(1, 1, 1, 112)
    bias[..., 100:] = common.MASK_VALUE
    full = common.torch_attention(q, k, v, 0.25, bias, qblock=0)
    assert _rel(common.torch_attention(q, k, v, 0.25, bias, qblock=32), full) < 1e-6

    calls = []
    real_matmul = torch.matmul
    monkeypatch.setattr(
        common.torch, "matmul", lambda a, b: calls.append(a.shape[-2]) or real_matmul(a, b)
    )
    monkeypatch.setattr(common, "ATTN_QBLOCK", 32)
    monkeypatch.setattr(common, "ATTN_QBLOCK_MAX_ROWS", 96)
    common.torch_attention(q, k, v, 0.25, bias)  # 96 rows: blocked
    assert calls[0] == 32
    calls.clear()
    monkeypatch.setattr(common, "ATTN_QBLOCK_MAX_ROWS", 64)
    common.torch_attention(q, k, v, 0.25, bias)  # 96 > 64 rows: one unblocked pass
    assert calls[0] == 96
