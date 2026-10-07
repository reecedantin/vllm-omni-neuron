# SPDX-License-Identifier: Apache-2.0
"""Shrunk random-weight pi0.52 checkpoint (structure test) + its CPU unit tests.

The checkpoint has the real ``lerobot/pi052_base`` structure: the same ``config.json`` /
``model.safetensors`` / ``policy_{pre,post}processor.json`` layout, the same parameter names
(``model.paligemma_with_expert...``), the same layer types (SigLIP, PaliGemma Gemma LM, AdaRMS
Gemma action expert, flow head) and attention layout (8 heads / 1 KV head), the full PaliGemma
vocabulary so the real tokenizer works, and 224 px / 3 cameras / 200 tokens / 50x32 chunks. Only
the widths and the depths shrink (``variant_dims`` in ``config.json``). It says nothing about
quality: it exists to prove loading, name mapping, compile and a device forward in minutes.

Generate one::

    python test/unit/test_pi0_tiny.py --out /path/to/pi052-tiny [--ref /path/to/pi052-base]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pytest
import torch

TINY_DIMS = {
    "vision": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    },
    "paligemma": {
        "width": 64,
        "depth": 2,
        "mlp_dim": 128,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 16,
    },
    "action_expert": {
        "width": 32,
        "depth": 2,
        "mlp_dim": 64,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 16,
    },
}

_DEFAULT_CONFIG = {
    "type": "pi052",
    "n_obs_steps": 1,
    "input_features": {
        **{
            f"observation.images.{c}": {"type": "VISUAL", "shape": [3, 224, 224]}
            for c in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "dtype": "float32",
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "num_inference_steps": 10,
    "min_period": 0.004,
    "max_period": 4.0,
    "use_relative_actions": False,
    "image_resolution": [224, 224],
    "empty_cameras": 0,
    "tokenizer_max_length": 200,
}

# Real lerobot/pi052_base checkpoint dir, for the key-layout test; skipped when unset.
REF_CHECKPOINT = os.environ.get("PI052_WEIGHTS", "")


def make_tiny_checkpoint(
    out_dir: str, ref_dir: str | None = None, seed: int = 0, dims: dict | None = None
) -> str:
    """Write a tiny pi0.52 checkpoint to ``out_dir``; copies the real config/processor files from
    ``ref_dir`` when given (so the layout matches byte for byte apart from ``variant_dims``)."""
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.modeling_pi05 import (
        Pi05ForActionPrediction,
    )

    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(_DEFAULT_CONFIG)
    if ref_dir and os.path.exists(os.path.join(ref_dir, "config.json")):
        with open(os.path.join(ref_dir, "config.json")) as f:
            cfg = json.load(f)
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            if os.path.exists(os.path.join(ref_dir, name)):
                shutil.copy(os.path.join(ref_dir, name), os.path.join(out_dir, name))
    cfg["variant_dims"] = dims or TINY_DIMS
    cfg["device"] = "cpu"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    torch.manual_seed(seed)
    model = Pi05ForActionPrediction(Pi05Config.from_model_config(cfg))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and ("norm" in name and "dense" not in name):
                p.normal_(0.0, 0.1)  # (1 + w) / LayerNorm weights near their identity
                if "layer_norm" in name or "post_layernorm" in name:
                    p.add_(1.0 if name.endswith("weight") else 0.0)
            else:
                p.normal_(0.0, 0.02 if p.ndim > 1 else 0.01)
            if ".dense." in name:  # AdaRMS: non-zero so the gates are open and t matters
                p.normal_(0.0, 0.05)
    vt = "paligemma_with_expert.paligemma.model.vision_tower."

    def disk_name(k: str) -> str:  # LeRobot's on-disk SigLIP nesting (transformers <= 5.3)
        if k.startswith(vt) and not k.startswith(vt + "vision_model."):
            k = vt + "vision_model." + k[len(vt) :]
        return f"model.{k}"

    state = {
        disk_name(k): v.contiguous()
        for k, v in model.state_dict().items()
        if "rotary_emb" not in k and not k.endswith("position_ids")
    }
    # LeRobot stores PaliGemma's tied embedding under both names.
    emb = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state["model.paligemma_with_expert.paligemma.lm_head.weight"] = state[emb].clone()
    import safetensors.torch

    safetensors.torch.save_file(
        state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"}
    )
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "Random-weight shrunk pi0.52 structure-test checkpoint (not a model). "
            "Generated by test/unit/test_pi0_tiny.py.\n"
        )
    return out_dir


def _safetensors_keys(path: str) -> dict:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


# ---------------------------------------------------------------------------------------------
# CPU unit tests
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    ref = REF_CHECKPOINT if os.path.isdir(REF_CHECKPOINT) else None
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("pi052_tiny")), ref)


def _inputs(cfg, seed=0, n_real_tokens=37, n_real_cams=2):
    g = torch.Generator().manual_seed(seed)
    r = cfg.image_resolution[0]
    images = [torch.rand(1, 3, r, r, generator=g) * 2 - 1 for _ in range(cfg.max_cameras)]
    masks = [torch.tensor([i < n_real_cams]) for i in range(cfg.max_cameras)]
    tokens = torch.zeros(1, cfg.tokenizer_max_length, dtype=torch.long)
    tokens[0, :n_real_tokens] = torch.randint(2, 250000, (n_real_tokens,), generator=g)
    tmask = torch.zeros(1, cfg.tokenizer_max_length, dtype=torch.bool)
    tmask[0, :n_real_tokens] = True
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return images, masks, tokens, tmask, noise


def test_tiny_key_layout_matches_real(tiny_dir):
    if not os.path.isdir(REF_CHECKPOINT):
        pytest.skip("real pi052 checkpoint not present")
    import re

    tiny = _safetensors_keys(os.path.join(tiny_dir, "model.safetensors"))
    real = _safetensors_keys(os.path.join(REF_CHECKPOINT, "model.safetensors"))

    def keep(k):  # real keys of the layers the tiny model has
        m = re.search(r"\.layers\.(\d+)\.", k)
        return m is None or int(m.group(1)) < 2

    assert set(tiny) == {k for k in real if keep(k)}
    assert all(tiny[k]["dtype"] == real[k]["dtype"] for k in tiny)


def test_tiny_loads_strict(tiny_dir):
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    assert cfg.policy_type == "pi052"
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)  # strict: raises on any missing / extra tensor


@pytest.mark.parametrize("n_real_cams", [3, 1])
def test_graphs_match_upstream_fp32(tiny_dir, n_real_cams):
    """The fixed-shape graphs reproduce the vendored upstream model exactly in fp32."""
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg, n_real_cams=n_real_cams)
    with torch.no_grad():
        ref = m.ref.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
        out = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
    rel = ((out - ref).norm() / ref.norm()).item()
    assert out.shape == (1, cfg.chunk_size, cfg.max_action_dim)
    assert rel < 1e-5, rel


def test_graphs_bf16_close(tiny_dir):
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m32 = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m32.load_checkpoint(tiny_dir)
    m16 = NeuronPi05ActionModel(cfg, dtype=torch.bfloat16)
    m16.load_checkpoint(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg)
    with torch.no_grad():
        a = m32.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
        b = m16.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
    rel = ((a - b).norm() / a.norm()).item()
    assert rel < 0.05, rel


def test_prefix_kv_ignores_padding(tiny_dir):
    """Changing padded token ids / a padded camera's pixels must not change the actions."""
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg, n_real_cams=2)
    with torch.no_grad():
        a = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
        tokens2 = tokens.clone()
        tokens2[0, 100:] = 12345
        images2 = list(images)
        images2[2] = torch.zeros_like(images[2])
        b = m.sample_actions(images2, masks, tokens2, tmask, noise=noise.clone())
    assert torch.allclose(a, b, atol=1e-6, rtol=0)


def test_subtask_generation_deterministic_and_text_graph_matches(tiny_dir):
    """pi0.52 greedy subtask decode runs, is deterministic, and the text-prefix graph reproduces
    the vendored upstream LM forward (full prefix -> lm_head) that LeRobot's select_message uses.

    Uses a real PaliGemma tokenizer when present; the tiny random weights make the text itself
    gibberish — the test checks the decode LOOP (indexing the last real text position past the
    image tokens, growing the sequence, EOS stop) and graph↔reference numerics, not the words."""
    import os

    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    tok = AutoTokenizer.from_pretrained(tok_dir, padding_side="right")

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    m.enable_subtask_generation(tok)
    images, masks, tokens, tmask, noise = _inputs(cfg, n_real_cams=cfg.max_cameras)

    out1 = m.generate_subtask(images, masks, "pick up the cube", max_new_tokens=8)
    out2 = m.generate_subtask(images, masks, "pick up the cube", max_new_tokens=8)
    assert out1 == out2  # greedy decode is deterministic

    # The text-prefix graph's last-text-position logits must equal running the vendored upstream
    # model's full LM forward + norm + lm_head over the identical prefix.
    import torch as _t

    from vllm_omni_neuron.diffusion.models.pi0.subtask import format_subtask_prompt

    ids = tok(format_subtask_prompt("pick up the cube"), add_special_tokens=True)["input_ids"]
    n = len(ids)
    pix = _t.stack(images, dim=1)
    pix = pix.reshape(len(images) * pix.shape[0], *pix.shape[2:]).float()
    iv = _t.stack([mm.float() for mm in masks], dim=1)
    logits = m.text_prefix(pix, iv, _t.tensor([ids]), _t.ones(1, n))
    n_image = logits.shape[1] - n
    ours = logits[0, n_image + n - 1].float()

    # vendored upstream reference: embed prefix, full forward (returns post-norm hidden), lm_head.
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.modeling_pi05 import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    ref = m.ref
    with _t.no_grad():
        pe, pp, pa = ref.embed_prefix(
            images, [mm.bool() for mm in masks], _t.tensor([ids]), _t.ones(1, n, dtype=_t.bool)
        )
        att = make_att_2d_masks(pp, pa)
        pos = _t.cumsum(pp, 1) - 1
        (vlm, _), _ = ref.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(att),
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[pe, None],
            use_cache=False,
        )
        lm = ref.paligemma_with_expert.paligemma.lm_head
        ref_logits = lm(vlm[:, -1:].to(lm.weight.dtype))[0, -1].float()
    rel = (ours - ref_logits).norm() / ref_logits.norm().clamp_min(1e-9)
    assert int(ours.argmax()) == int(ref_logits.argmax()), (
        int(ours.argmax()),
        int(ref_logits.argmax()),
    )
    assert rel < 1e-4, rel


def test_subtask_image_cache_matches_uncached(tiny_dir):
    """The image-prefix-cached decode path (vision tower run once per request) must produce the
    identical subtask as the uncached path (vision tower rerun every token)."""
    import os

    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    tok = AutoTokenizer.from_pretrained(tok_dir, padding_side="right")
    cfg = Pi05Config.from_pretrained(tiny_dir)
    images, masks, _, _, _ = _inputs(cfg, n_real_cams=cfg.max_cameras)

    m_un = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m_un.load_checkpoint(tiny_dir)
    m_un.enable_subtask_generation(tok, cache_image_prefix=False)
    out_uncached = m_un.generate_subtask(images, masks, "pick up the cube", max_new_tokens=8)

    m_c = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m_c.load_checkpoint(tiny_dir)
    m_c.enable_subtask_generation(tok, cache_image_prefix=True)
    assert m_c.embed_images_graph is not None and m_c.text_prefix_cached is not None
    out_cached = m_c.generate_subtask(images, masks, "pick up the cube", max_new_tokens=8)

    assert out_cached == out_uncached


def _upstream_prefix_lm_greedy(ref, images, masks, ids, n_new, fast_skip, special_ids):
    """LeRobot ``select_message(use_kv_cache=False)`` on the vendored upstream model: full
    recompute per token, generated tokens appended with att_mask 1 (causal among themselves,
    prompt never sees them). Returns (ids, per-step logits)."""
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.modeling_pi05 import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )
    from vllm_omni_neuron.diffusion.models.pi0.subtask import _FAST_ACTION_VOCAB_SIZE

    n = len(ids)
    pe, pp, pa = ref.embed_prefix(
        images, [mm.bool() for mm in masks], torch.tensor([ids]), torch.ones(1, n, dtype=torch.bool)
    )
    lm = ref.paligemma_with_expert.paligemma.lm_head
    one = torch.ones(1, 1, dtype=torch.bool)
    out, all_logits = [], []
    for step in range(n_new):
        (vlm, _), _ = ref.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pp, pa)),
            position_ids=torch.cumsum(pp, 1) - 1,
            past_key_values=None,
            inputs_embeds=[pe, None],
            use_cache=False,
        )
        lg = lm(vlm[:, -1:].to(lm.weight.dtype))[0, -1].float()
        all_logits.append(lg.clone())
        v = lg.shape[-1]
        lo = v - 1 - fast_skip - (_FAST_ACTION_VOCAB_SIZE - 1)
        if 0 < lo < 256000:
            lg[lo:256000] = float("-inf")
        lg[256000:257024] = float("-inf")
        for sid in special_ids:
            if sid < v:
                lg[sid] = float("-inf")  # min_new_tokens == n_new: fixed-length decode
        tok = int(lg.argmax())
        out.append(tok)
        new = ref.paligemma_with_expert.embed_language_tokens(torch.tensor([[tok]]))
        pe = torch.cat([pe, new.to(pe.dtype)], 1)
        pp, pa = torch.cat([pp, one], 1), torch.cat([pa, one], 1)
    return out, all_logits


@pytest.mark.parametrize("n_real_cams", [3, 2])
def test_subtask_kv_cache_matches_upstream_prefix_lm(tiny_dir, n_real_cams):
    """Prefill-once + KV-cached decode (shared StaticKVCache / decode_step) reproduces upstream's
    full-recompute greedy decode token for token, with a padded camera slot too (cache
    compaction). Fixed-length (EOS masked) so every decode step is exercised."""
    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config
    from vllm_omni_neuron.diffusion.models.pi0.subtask import format_subtask_prompt

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    tok = AutoTokenizer.from_pretrained(tok_dir, padding_side="right")
    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    m.enable_subtask_generation(tok)
    assert m.subtask_decode is not None and m._subtask_cache_cfg.max_len % 128 == 0
    images, masks, _, _, _ = _inputs(cfg, n_real_cams=n_real_cams)
    task, n_new = "pick up the cube", 12
    _, kv_ids = m.generate_subtask(
        images, masks, task, max_new_tokens=n_new, min_new_tokens=n_new, return_ids=True
    )
    ids = tok(format_subtask_prompt(task), add_special_tokens=True)["input_ids"]
    with torch.no_grad():
        ref_ids, _ = _upstream_prefix_lm_greedy(
            m.ref,
            images,
            masks,
            ids,
            n_new,
            m.subtask_gen.fast_skip_tokens,
            m.subtask_gen.special_ids,
        )
    assert kv_ids == ref_ids
    # The re-prefill fallback applies the same prefix-LM mask.
    _, re_ids = m.generate_subtask(
        images,
        masks,
        task,
        max_new_tokens=n_new,
        min_new_tokens=n_new,
        return_ids=True,
        kv_cache=False,
    )
    assert re_ids == ref_ids


def test_subtask_kv_decode_step_logits_match_reprefill(tiny_dir):
    """Per-step logits of the KV-cached decode equal a full prefix-LM recompute (rel < 1e-4)."""
    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.attention.decode_attention import StaticKVCache
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config
    from vllm_omni_neuron.diffusion.models.pi0.subtask import format_subtask_prompt

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    tok = AutoTokenizer.from_pretrained(tok_dir, padding_side="right")
    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    m.enable_subtask_generation(tok)
    images, masks, _, _, _ = _inputs(cfg, n_real_cams=2)
    ids = tok(format_subtask_prompt("open the drawer"), add_special_tokens=True)["input_ids"]
    forced = [1000, 2000, 3000, 4000]
    with torch.no_grad():
        # Reference: upstream logits after feeding the forced tokens (teacher forcing).
        from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.modeling_pi05 import (
            make_att_2d_masks,
            prepare_attention_masks_4d,
        )

        ref = m.ref
        pe, pp, pa = ref.embed_prefix(
            images,
            [mm.bool() for mm in masks],
            torch.tensor([ids]),
            torch.ones(1, len(ids), dtype=torch.bool),
        )
        lm = ref.paligemma_with_expert.paligemma.lm_head
        new = ref.paligemma_with_expert.embed_language_tokens(torch.tensor([forced]))
        pe = torch.cat([pe, new.to(pe.dtype)], 1)
        ones = torch.ones(1, len(forced), dtype=torch.bool)
        pp, pa = torch.cat([pp, ones], 1), torch.cat([pa, ones], 1)
        (vlm, _), _ = ref.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pp, pa)),
            position_ids=torch.cumsum(pp, 1) - 1,
            past_key_values=None,
            inputs_embeds=[pe, None],
            use_cache=False,
        )
        ref_logits = lm(vlm[0, -len(forced) - 1 :].to(lm.weight.dtype)).float()

        # Ours: prefill, then decode the forced tokens one at a time.
        sg = m.subtask_gen
        pix = torch.stack(images, 1).reshape(len(images), *images[0].shape[1:])
        img_emb = m.embed_images_graph(pix)
        n_img = img_emb.shape[1]
        iv = torch.stack([mm.float() for mm in masks], 1)
        keep = [c * n_img + j for c in range(m.num_cameras) if iv[0, c] > 0.5 for j in range(n_img)]
        keep += [m.num_cameras * n_img + j for j in range(len(ids))]
        bucket = sg.buckets[0]
        cc = m._subtask_cache_cfg
        sel = torch.zeros(cc.max_len, m.num_cameras * n_img + bucket)
        sel[torch.arange(len(keep)), torch.tensor(keep)] = 1.0
        last = torch.zeros(1, sel.shape[1])
        last[0, keep[-1]] = 1.0
        tokens = torch.zeros(1, bucket, dtype=torch.long)
        tokens[0, : len(ids)] = torch.tensor(ids)
        valid = (torch.arange(bucket) < len(ids)).float()[None]
        bias = torch.zeros(1, lm.weight.shape[0])
        out = m.subtask_prefill(img_emb, iv, tokens, valid, sel, last, bias)
        L = (len(out) - 2) // 2
        caches = []
        for i in range(L):
            c = StaticKVCache(cc, "cpu")
            c.k, c.v = out[2 + i], out[2 + L + i]
            c.pos = torch.tensor([len(keep)], dtype=torch.int32)
            caches.append(c)
        ours = [out[0][0]]
        for t in forced:
            ours.append(m.subtask_decode(torch.tensor([[t]]), bias, caches)[0][0])
    for a, b in zip(ours, ref_logits):
        rel = (a - b).norm() / b.norm()
        assert rel < 1e-4, rel


def test_lerobot_config_resolves_to_pipeline(tiny_dir, tmp_path):
    """vllm-omni 0.24 compat: a LeRobot config.json (``type`` only, no model_type/architectures)
    resolves to the registered pipeline instead of failing with 'model_index.json not found'."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    import vllm_omni_neuron.diffusion.models.pi0  # noqa: F401  installs the shim

    c = OmniDiffusionConfig(model=tiny_dir)
    c.enrich_config()
    assert c.model_class_name == "Pi05Pipeline"

    other = tmp_path / "not_lerobot"
    other.mkdir()
    (other / "config.json").write_text('{"type": "something_else"}')
    with pytest.raises((OSError, ValueError)):
        OmniDiffusionConfig(model=str(other)).enrich_config()


def test_pipeline_end_to_end_cpu(tiny_dir):
    """The serving pipeline registers, builds from a checkpoint dir, and turns a raw robot
    observation into a finite ``(chunk_size, action_dim)`` action chunk on CPU — pi0.52 running
    its subtask-generation step first."""
    import dataclasses
    import os
    import types

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    import vllm_omni_neuron

    vllm_omni_neuron._register_pipelines()  # must not raise
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline

    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype, od.model_config = tiny_dir, "float32", {}
    od.model_config = {"tokenizer": tok_dir}
    pipe = NeuronPi05Pipeline(od_config=od)
    assert pipe.config.policy_type == "pi052"

    g = torch.Generator().manual_seed(0)
    cams = [k for k in pipe.config.input_features if "image" in k]
    obs = {
        "prompt": "pick up the red cube",
        **{k: (torch.rand(224, 224, 3, generator=g)).numpy() for k in cams},
        "state": (torch.rand(pipe.config.state_dim, generator=g) * 2 - 1).numpy(),
    }
    sp = types.SimpleNamespace(
        extra_args={"robot_obs": obs},
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(1),
    )
    out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
    assert out.error is None, out.error
    a = out.output["actions"]
    assert a.shape == (pipe.config.chunk_size, pipe.config.action_dim)
    assert bool(np.isfinite(a).all())


@pytest.mark.parametrize("mode", ["unrolled", "host"])
def test_denoise_modes_match_default(tiny_dir, mode):
    """One device-resident graph per Euler step (default), the loop unrolled into one graph, and
    the host loop give the same actions (fp32)."""
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg)
    with torch.no_grad():
        a = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=5)
        m.denoise_loop.mode = mode
        b = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=5)
        ref = m.ref.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=5)
    assert torch.allclose(a, b, atol=1e-6, rtol=0), (a - b).abs().max()
    assert ((a - ref).norm() / ref.norm()).item() < 1e-5


def _tiny_with_subtask(tiny_dir):
    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(tiny_dir)
    m.enable_subtask_generation(AutoTokenizer.from_pretrained(tok_dir, padding_side="right"))
    return cfg, m


def test_prefix_from_image_embedding_matches(tiny_dir):
    """pi0.52's action prefix over the subtask decode's image embedding (no second SigLIP pass)
    equals the full prefix graph."""
    cfg, m = _tiny_with_subtask(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg, n_real_cams=2)
    with torch.no_grad():
        a = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
        emb = m.encode_images(images)
        b = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), img_emb=emb)
    assert torch.allclose(a, b, atol=1e-6, rtol=0), (a - b).abs().max()


@pytest.mark.parametrize("adarms_tables", [True, False])
def test_adarms_tables_match_per_step_dense(tiny_dir, adarms_tables):
    """Precomputed AdaRMS modulation tables (default) == dense(cond) inside every step == upstream."""
    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    cfg = Pi05Config.from_pretrained(tiny_dir)
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32, adarms_tables=adarms_tables)
    m.load_checkpoint(tiny_dir)
    images, masks, tokens, tmask, noise = _inputs(cfg)
    with torch.no_grad():
        a = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=4)
        ref = m.ref.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=4)
    assert ((a - ref).norm() / ref.norm()).item() < 1e-5


@pytest.mark.parametrize("sync_every", [1, 3])
def test_subtask_in_graph_greedy_matches_host_greedy(tiny_dir, sync_every):
    """The device-resident decode (argmax + suppression in the graph, tokens fed back on the
    device, EOS read every ``sync_every`` steps) picks the same ids as the host greedy loop, with
    natural stopping and at fixed length; a reused image embedding changes nothing."""
    cfg, m = _tiny_with_subtask(tiny_dir)
    images, masks, _, _, _ = _inputs(cfg, n_real_cams=3)
    for task in ("pick up the cube", "open the drawer"):
        for n, forced in ((10, False), (7, True)):
            kw = dict(max_new_tokens=n, min_new_tokens=n if forced else 0, return_ids=True)
            _, host = m.generate_subtask(images, masks, task, step_hook=lambda s, lg: None, **kw)
            _, dev = m.generate_subtask(images, masks, task, sync_every=sync_every, **kw)
            emb = m.encode_images(images)
            _, dev2 = m.generate_subtask(images, masks, task, img_emb=emb, **kw)
            assert dev == host == dev2, (task, n, dev, host)
            if forced:
                assert len(dev) == n


def test_raw_byte_cameras_match_arrays(tiny_dir):
    """Cameras (and noise) sent as raw bytes decode to the same request: identical actions."""
    import dataclasses
    import types

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline
    from vllm_omni_neuron.diffusion.models.pi0.request import encode_camera

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype = tiny_dir, "float32"
    od.model_config = {"tokenizer": tok_dir}
    pipe = NeuronPi05Pipeline(od_config=od)
    rng = np.random.default_rng(0)
    cams = [k for k in pipe.config.input_features if "image" in k]
    frames = {k: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for k in cams}
    state = rng.uniform(-1, 1, pipe.config.state_dim).astype(np.float32)
    noise = rng.standard_normal((1, pipe.config.chunk_size, pipe.config.max_action_dim))
    noise = noise.astype(np.float32)

    def run(obs, nz):
        sp = types.SimpleNamespace(
            extra_args={"robot_obs": obs, "noise": nz}, num_inference_steps=3, generator=None
        )
        out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
        assert out.error is None, out.error
        return out.output["actions"]

    a = run({"prompt": "pick up the cube", **frames, "state": state}, noise)
    b = run(
        {
            "prompt": "pick up the cube",
            **{k: encode_camera(v) for k, v in frames.items()},
            "state": state,
        },
        {"data": noise.tobytes(), "shape": list(noise.shape), "dtype": "float32"},
    )
    np.testing.assert_array_equal(a, b)


def test_min_subtask_tokens_and_stats_digest(tiny_dir, tmp_path, monkeypatch):
    """``min_new_subtask_tokens`` = ``max_new_subtask_tokens`` runs exactly that many decode
    steps (the fixed-length benchmark), and the stats record carries the rank, the step count
    and digests of the action chunk and subtask ids (the all-rank agreement check)."""
    import dataclasses
    import hashlib
    import json
    import types

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline
    from vllm_omni_neuron.diffusion.models.pi0 import request as req_mod

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    stats = tmp_path / "stats.jsonl"
    monkeypatch.setattr(req_mod, "STATS_FILE", str(stats))
    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype = tiny_dir, "float32"
    od.model_config = {
        "tokenizer": tok_dir,
        "min_new_subtask_tokens": 5,
        "max_new_subtask_tokens": 5,
    }
    pipe = NeuronPi05Pipeline(od_config=od)
    assert pipe.host_threads == 0  # default: leave the worker's thread setting alone
    rng = np.random.default_rng(0)
    cams = [k for k in pipe.config.input_features if "image" in k]
    obs = {
        "prompt": "pick up the cube",
        # one camera left out: masked, as LeRobot does for a camera the robot lacks
        **{k: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for k in cams[:-1]},
        "state": rng.uniform(-1, 1, pipe.config.state_dim).astype(np.float32),
    }
    sp = types.SimpleNamespace(extra_args={"robot_obs": obs}, num_inference_steps=2, generator=None)
    out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
    assert out.error is None, out.error
    assert len(pipe.last_subtask_ids) == 5
    assert pipe.tokenizer.eos_token_id not in pipe.last_subtask_ids
    rec = json.loads(stats.read_text().splitlines()[-1])
    assert rec["subtask_steps"] == 5 and rec["tp_rank"] == 0
    arr = np.ascontiguousarray(np.asarray(out.output["actions"], dtype=np.float32))
    assert rec["actions_sha256"] == hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    ids = json.dumps(pipe.last_subtask_ids).encode()
    assert rec["subtask_ids_sha256"] == hashlib.sha256(ids).hexdigest()[:16]


def _tp_worker(rank, size, tiny_dir, tok_dir, port, out_path):
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size
    )
    try:
        from transformers import AutoTokenizer

        cfg = Pi05Config.from_pretrained(tiny_dir)
        m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
        m.load_checkpoint(tiny_dir)
        if size > 1:
            m.shard_tp(rank, size, dist.group.WORLD)
        m.enable_subtask_generation(AutoTokenizer.from_pretrained(tok_dir, padding_side="right"))
        images, masks, tokens, tmask, noise = _inputs(cfg, n_real_cams=2)
        seen = []
        with torch.no_grad():
            acts = m.sample_actions(images, masks, tokens, tmask, noise=noise.clone())
            _, ids = m.generate_subtask(
                images,
                masks,
                "pick up the cube",
                max_new_tokens=6,
                min_new_tokens=6,
                return_ids=True,
            )
            m.generate_subtask(
                images,
                masks,
                "pick up the cube",
                max_new_tokens=6,
                min_new_tokens=6,
                step_hook=lambda s, lg: seen.append(lg),
            )
        if rank == 0:
            torch.save({"actions": acts, "ids": ids, "logits": torch.stack(seen)}, out_path)
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_lm_matches_single_rank(tiny_dir, tmp_path):
    """TP=2 over the PaliGemma LM (query heads / MLP / vocabulary split, all-reduces on CPU gloo)
    reproduces TP=1: actions, the greedy subtask ids and every decode step's logits."""
    import socket

    import torch.multiprocessing as mp

    tok_dir = os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    if not os.path.isdir(tok_dir):
        pytest.skip("PaliGemma tokenizer not present (set WEIGHTS)")
    res = {}
    for size in (1, 2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        out = str(tmp_path / f"tp{size}.pt")
        mp.spawn(_tp_worker, args=(size, tiny_dir, tok_dir, port, out), nprocs=size, join=True)
        res[size] = torch.load(out)
    a, b = res[2], res[1]
    assert ((a["actions"] - b["actions"]).norm() / b["actions"].norm()).item() < 1e-5
    assert a["ids"] == b["ids"]
    rel = (a["logits"] - b["logits"]).norm(dim=-1) / b["logits"].norm(dim=-1)
    assert rel.max().item() < 1e-5, rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=REF_CHECKPOINT if os.path.isdir(REF_CHECKPOINT) else None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_tiny_checkpoint(a.out, a.ref, a.seed))
