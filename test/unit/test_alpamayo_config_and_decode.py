# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Alpamayo decode-attention seam (``decode_backend.py``) and config resolution.

No real weights needed: these check the seam's own math (prefill+decode reconstructs full-sequence
attention exactly) and that ``AlpamayoConfig`` resolves the two checkpoints' different config.json
shapes to the same fields. Real-checkpoint config tests are skipped if the weights are not present
(set ``$WEIGHTS`` to the directory holding the checkpoints).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.alpamayo.decode_backend import (
    DecodeConfig,
    cross_step,
    decode_step,
    prefill,
)
from vllm_omni_neuron.diffusion.models.alpamayo.prep import (
    decode_masks,
    expert_inputs,
    prefill_bias,
)

WEIGHTS = os.environ.get("WEIGHTS", "")  # directory holding alpamayo-1.5-10b / alpamayo2-super


def _reference(q, k, v, cfg: DecodeConfig):
    n = cfg.num_kv_groups
    if n > 1:
        b, h, s, d = k.shape
        k = k[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)
        v = v[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True, scale=cfg.scale
    ).to(q.dtype)


@pytest.mark.parametrize("kv_heads", [2, 8])  # MQA/GQA-ish and the real GQA ratio (q_heads=8)
@pytest.mark.parametrize("bucket", [5, 7])  # exact-fit prompt, and a right-padded bucket
def test_static_prefill_then_decode_matches_full_causal_attention(kv_heads, bucket):
    """Fixed-length cache, full-cache reads + additive masks + one-hot writes reproduce plain causal
    attention over the real tokens -- including when the bucket's padding K/V sit in slots the
    decode later overwrites."""
    torch.manual_seed(0)
    cfg = DecodeConfig(q_heads=8, kv_heads=kv_heads, head_dim=16, max_len=32, dtype=torch.float32)
    s_prompt, s_new = 5, 3
    s = s_prompt + s_new
    q = torch.randn(1, cfg.q_heads, s, cfg.head_dim)
    k = torch.randn(1, cfg.kv_heads, s, cfg.head_dim)
    v = torch.randn(1, cfg.kv_heads, s, cfg.head_dim)
    ref = _reference(q, k, v, cfg)

    def padded(x):  # prompt tokens, then (bucket - s_prompt) garbage padding rows
        return torch.cat(
            [x[:, :, :s_prompt], torch.randn(1, x.shape[1], bucket - s_prompt, x.shape[3]) * 9], 2
        )

    out_p, kc, vc = prefill(cfg, padded(q), padded(k), padded(v), prefill_bias(s_prompt, bucket))
    assert kc.shape[2] == cfg.max_len
    assert torch.allclose(out_p[:, :, :s_prompt], ref[:, :, :s_prompt], atol=1e-5)
    outs = [out_p[:, :, :s_prompt]]
    for t in range(s_prompt, s):
        wm, bias = decode_masks(t, cfg.max_len, "cpu")
        o, kc, vc = decode_step(
            cfg, q[:, :, t : t + 1], k[:, :, t : t + 1], v[:, :, t : t + 1], kc, vc, wm, bias
        )
        outs.append(o)
    got = torch.cat(outs, dim=2)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_cross_step_matches_cropped_cache_attention():
    """The expert's view (full fixed cache + own K/V, masked to ``[0, min(offset, valid))`` + own) equals
    attention over exactly upstream's cropped cache with ``[offset, valid)`` masked."""
    from transformers import Qwen3VLConfig

    torch.manual_seed(2)
    hf = Qwen3VLConfig()
    d = hf.text_config.head_dim
    hk, h, L, n, valid, offset = 2, 4, 20, 3, 11, 9
    kc, vc = torch.randn(1, hk, L, d), torch.randn(1, hk, L, d)
    q, k, v = torch.randn(1, h, n, d), torch.randn(1, hk, n, d), torch.randn(1, hk, n, d)
    _, _, bias = expert_inputs(hf, offset, valid, 0, n, L, torch.float32, "cpu")
    got = cross_step(h // hk, d**-0.5, q, k, v, kc, vc, bias)
    kk = torch.cat([kc[:, :, :valid], k], 2).repeat_interleave(h // hk, 1)
    vv = torch.cat([vc[:, :, :valid], v], 2).repeat_interleave(h // hk, 1)
    m = torch.zeros(1, 1, n, valid + n)
    m[..., offset:valid] = float("-inf")
    want = F.scaled_dot_product_attention(q, kk, vv, attn_mask=m, scale=d**-0.5)
    assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.skipif(
    not os.path.isdir(f"{WEIGHTS}/alpamayo-1.5-10b"), reason="needs the real checkpoint"
)
def test_config_alpamayo1_5_matches_checkpoint_tensor_shapes():
    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    c = AlpamayoConfig.from_model_dir(f"{WEIGHTS}/alpamayo-1.5-10b")
    assert c.variant == "alpamayo1_5"
    tc = c.backbone["text_config"]
    assert (
        tc["hidden_size"],
        tc["num_hidden_layers"],
        tc["num_attention_heads"],
        tc["num_key_value_heads"],
    ) == (4096, 36, 32, 8)
    assert tc["vocab_size"] == 155697  # extended for Alpamayo's trajectory-token vocabulary
    assert c.expert_hidden_size == 2048
    assert c.n_waypoints == 64
    assert c.max_new_tokens == 128


@pytest.mark.skipif(
    not os.path.isdir(f"{WEIGHTS}/alpamayo2-super"), reason="needs the real checkpoint"
)
def test_config_alpamayo2_super_matches_checkpoint_tensor_shapes():
    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    c = AlpamayoConfig.from_model_dir(f"{WEIGHTS}/alpamayo2-super")
    assert c.variant == "alpamayo2_super"
    tc = c.backbone["text_config"]
    assert (
        tc["hidden_size"],
        tc["num_hidden_layers"],
        tc["num_attention_heads"],
        tc["num_key_value_heads"],
    ) == (5120, 64, 64, 8)
    assert c.expert_hidden_size == 1536
    assert c.n_waypoints == 64


QWEN3_VL_8B_CONFIG = os.environ.get(
    "ALPAMAYO_TOKENIZER_DIR", ""
)  # local Qwen3-VL-8B-Instruct tokenizer/config


@pytest.mark.skipif(
    not os.path.isdir(QWEN3_VL_8B_CONFIG),
    reason="needs $ALPAMAYO_TOKENIZER_DIR (Qwen3-VL-8B-Instruct tokenizer)",
)
@pytest.mark.skipif(
    not os.path.isdir(f"{WEIGHTS}/alpamayo-1.5-10b"), reason="needs the real checkpoint"
)
def test_tokenizer_vocab_extension_matches_checkpoint():
    """The public Qwen3-VL-8B-Instruct tokenizer, extended by Alpamayo's own vocabulary-growth code,
    must reproduce the real checkpoint's embed_tokens row count and documented special-token ids
    exactly -- this is the backup path while nvidia/Cosmos-Reason2-8B access is pending."""
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    c = AlpamayoConfig.from_model_dir(f"{WEIGHTS}/alpamayo-1.5-10b")
    tok = c.build_tokenizer()
    assert len(tok) == c.backbone["text_config"]["vocab_size"]
    assert tok.traj_token_start_idx == c.head["traj_token_start_idx"]
    assert tok.traj_token_ids == c.extra["traj_token_ids"]

    import json

    idx = json.load(open(f"{WEIGHTS}/alpamayo-1.5-10b/model.safetensors.index.json"))["weight_map"]
    with safe_open(
        f"{WEIGHTS}/alpamayo-1.5-10b/{idx['vlm.model.language_model.embed_tokens.weight']}", "pt"
    ) as f:
        rows = f.get_slice("vlm.model.language_model.embed_tokens.weight").get_shape()[0]
    assert len(tok) == rows


@pytest.mark.skipif(
    not os.path.isdir(QWEN3_VL_8B_CONFIG),
    reason="needs $ALPAMAYO_TOKENIZER_DIR (Qwen3-VL-8B-Instruct config)",
)
@pytest.mark.skipif(
    not os.path.isdir(f"{WEIGHTS}/alpamayo-1.5-10b"), reason="needs the real checkpoint"
)
def test_backbone_tensor_shapes_match_checkpoint_exhaustively():
    """Every vlm.model.* / vlm.lm_head tensor's shape, derived purely from the public backbone
    config, must match the real checkpoint -- no mismatches anywhere in the 36 text layers,
    27 vision blocks, 3 deepstack mergers, or the embed/merger/patch-embed tensors."""
    import json

    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    model_dir = f"{WEIGHTS}/alpamayo-1.5-10b"
    c = AlpamayoConfig.from_model_dir(model_dir)
    tc, vc = c.backbone["text_config"], c.backbone["vision_config"]
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    files: dict[str, object] = {}

    def shape(k: str):
        f = idx[k]
        if f not in files:
            files[f] = safe_open(f"{model_dir}/{f}", "pt")
        return tuple(files[f].get_slice(k).get_shape())

    H, L = tc["hidden_size"], tc["num_hidden_layers"]
    QH, KVH, HD, IM = (
        tc["num_attention_heads"],
        tc["num_key_value_heads"],
        tc["head_dim"],
        tc["intermediate_size"],
    )
    expected = {}
    for i in range(L):
        p = f"vlm.model.language_model.layers.{i}."
        expected[p + "self_attn.q_proj.weight"] = (QH * HD, H)
        expected[p + "self_attn.k_proj.weight"] = (KVH * HD, H)
        expected[p + "self_attn.v_proj.weight"] = (KVH * HD, H)
        expected[p + "self_attn.o_proj.weight"] = (H, QH * HD)
        expected[p + "mlp.gate_proj.weight"] = (IM, H)
        expected[p + "mlp.up_proj.weight"] = (IM, H)
        expected[p + "mlp.down_proj.weight"] = (H, IM)
    VD, VI = vc["hidden_size"], vc["intermediate_size"]
    for i in range(vc["depth"]):
        p = f"vlm.model.visual.blocks.{i}."
        expected[p + "attn.qkv.weight"] = (3 * VD, VD)
        expected[p + "attn.proj.weight"] = (VD, VD)
        expected[p + "mlp.linear_fc1.weight"] = (VI, VD)
        expected[p + "mlp.linear_fc2.weight"] = (VD, VI)
    expected["vlm.model.visual.patch_embed.proj.weight"] = (
        VD,
        vc["in_channels"],
        vc["temporal_patch_size"],
        vc["patch_size"],
        vc["patch_size"],
    )
    expected["vlm.model.visual.pos_embed.weight"] = (vc["num_position_embeddings"], VD)
    expected["vlm.model.visual.merger.linear_fc2.weight"] = (vc["out_hidden_size"], VD * 4)
    for k, want in expected.items():
        assert k in idx, f"missing from checkpoint: {k}"
        assert shape(k) == want, f"{k}: got {shape(k)}, expected {want}"
    assert len(expected) == 36 * 7 + 27 * 4 + 3
