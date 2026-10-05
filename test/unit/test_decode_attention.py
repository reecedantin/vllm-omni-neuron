# SPDX-License-Identifier: Apache-2.0
"""Shared decode-attention layer for embedded text towers (Cosmos3 Qwen3-VL reasoner, Qwen-Image
2.1 prompt-enhancer, pi0.5.2 text subtask). CPU only -- exercises the torch fallback path (the
kernel path is NC-v3-only; see test/neuron/smoke_platform_trn2.py for the device check)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.attention.decode_attention import (
    DecodeAttentionConfig,
    DecodeWeights,
    StaticKVCache,
    can_use_decode_kernel,
    decode_prefill,
    decode_step,
)


def _rope_tables(max_len: int, head_dim: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_len).float()
    freqs = torch.outer(pos, inv_freq)  # [max_len, head_dim // 2]
    return freqs.cos(), freqs.sin()


# ---------------------------------------------------------------- config / gate


def test_config_derived_properties():
    cfg = DecodeAttentionConfig(q_heads=8, kv_heads=2, head_dim=64, max_len=128)
    assert cfg.hidden_size == 512
    assert cfg.num_kv_groups == 4
    assert cfg.scale == pytest.approx(64**-0.5)
    assert cfg.max_decode_tokens() == 16  # 128 // 8


def test_config_rejects_odd_head_dim_and_bad_gqa_ratio():
    with pytest.raises(ValueError, match="even"):
        DecodeAttentionConfig(q_heads=4, kv_heads=1, head_dim=63, max_len=16)
    with pytest.raises(ValueError, match="multiple"):
        DecodeAttentionConfig(q_heads=5, kv_heads=2, head_dim=64, max_len=16)


@pytest.mark.parametrize(
    "q_heads,head_dim,s_tkg,expect",
    [(8, 64, 1, True), (8, 64, 16, True), (8, 64, 17, False), (3, 50, 1, False)],
)
def test_can_use_decode_kernel_shape_gate(monkeypatch, q_heads, head_dim, s_tkg, expect):
    from vllm_omni_neuron import nc_generation as ncg

    monkeypatch.setenv("VLLM_OMNI_NEURON_CORE_GEN", "3")
    ncg.neuron_core_generation.cache_clear()
    cfg = DecodeAttentionConfig(q_heads=q_heads, kv_heads=1, head_dim=head_dim, max_len=128)
    assert can_use_decode_kernel(cfg, s_tkg) is expect
    ncg.neuron_core_generation.cache_clear()


@pytest.mark.parametrize(
    "max_len,expect", [(128, True), (256, True), (64, False), (32, False), (100, False)]
)
def test_can_use_decode_kernel_needs_max_len_multiple_of_128(monkeypatch, max_len, expect):
    """attention_decode's fused mask-gen path requires s_prior % 128 == 0; this layer's flat cache
    makes s_prior == max_len (regression test for the Trn2 smoke failure traced to max_len=64)."""
    from vllm_omni_neuron import nc_generation as ncg

    monkeypatch.setenv("VLLM_OMNI_NEURON_CORE_GEN", "3")
    ncg.neuron_core_generation.cache_clear()
    cfg = DecodeAttentionConfig(q_heads=8, kv_heads=1, head_dim=64, max_len=max_len)
    assert can_use_decode_kernel(cfg, 1) is expect
    ncg.neuron_core_generation.cache_clear()


def test_decode_prefill_from_hidden_matches_manual_projection():
    """decode_prefill_from_hidden (projection inside the call, so a compiled device region has only
    contiguous graph inputs -- round 11's 'non-contiguous slicing for requested Device Tensor') must
    equal projecting by hand, splitting, calling decode_prefill, and applying W_out."""
    import vllm_omni_neuron.diffusion.attention.decode_attention as da

    torch.manual_seed(0)
    q_heads, kv_heads, d, S = 4, 2, 8, 6
    cfg = da.DecodeAttentionConfig(
        q_heads=q_heads, kv_heads=kv_heads, head_dim=d, max_len=128, dtype=torch.float32
    )
    H = cfg.hidden_size
    w = da.DecodeWeights.from_separate(
        torch.randn(q_heads * d, H),
        torch.randn(kv_heads * d, H),
        torch.randn(kv_heads * d, H),
        torch.randn(H, H),
        q_bias=torch.randn(q_heads * d),
        k_bias=torch.randn(kv_heads * d),
        v_bias=torch.randn(kv_heads * d),
        out_bias=torch.randn(H),
    )
    cos, sin = _rope_tables(S, d)
    hidden = torch.randn(1, S, H)

    cache_a = da.StaticKVCache(cfg, "cpu")
    got = da.decode_prefill_from_hidden(cfg, w, cache_a, hidden, cos=cos, sin=sin)

    cache_b = da.StaticKVCache(cfg, "cpu")
    qkv = hidden @ w.W_qkv + w.bias_qkv
    q_end, k_end = q_heads * d, q_heads * d + kv_heads * d
    q = qkv[..., :q_end].view(1, S, q_heads, d).transpose(1, 2)
    k = qkv[..., q_end:k_end].view(1, S, kv_heads, d).transpose(1, 2)
    v = qkv[..., k_end:].view(1, S, kv_heads, d).transpose(1, 2)
    attn = da.decode_prefill(cfg, cache_b, q, k, v, cos=cos, sin=sin)
    want = attn.transpose(1, 2).reshape(S, q_heads * d) @ w.W_out + w.bias_out

    assert got.shape == (S, H)
    torch.testing.assert_close(got, want)
    K_a, V_a = cache_a.as_4d_cache()
    K_b, V_b = cache_b.as_4d_cache()
    torch.testing.assert_close(K_a, K_b)
    torch.testing.assert_close(V_a, V_b)


def test_kv_new_to_cache_layout_handles_fallback_and_device_layouts():
    """attention_decode returns the new K tokens head-dim-major in two layouts: the torch fallback's
    rank-3 [head_dim, B*kv_heads, S_tkg] and the device kernel's rank-4 [head_dim, B, kv_heads,
    S_tkg] ((128, 1, 2, 1) in smoke round 13, which broke the old rank-3 permute). Both must land
    in the cache layout [1, kv_heads, S_tkg, head_dim] with the right element at each position."""
    from vllm_omni_neuron.diffusion.attention.decode_attention import _kv_new_to_cache_layout

    kv, s, d = 2, 3, 8
    want = torch.arange(kv * s * d, dtype=torch.float32).reshape(1, kv, s, d)
    k3 = want[0].reshape(kv * s, d).t().contiguous()  # [d, kv*S]   -> fallback [d, B*kv, S]
    k3 = k3.reshape(d, kv, s)
    assert _kv_new_to_cache_layout(k3, kv, s, d, head_dim_first=True).equal(want)
    k4 = k3.reshape(d, 1, kv, s)  # device: [d, B, kv, S]
    assert _kv_new_to_cache_layout(k4, kv, s, d, head_dim_first=True).equal(want)
    v4 = want.clone()  # v comes back [B, kv, S, d] already
    assert _kv_new_to_cache_layout(v4, kv, s, d, head_dim_first=False).equal(want)
    with pytest.raises(ValueError, match="unexpected new K/V size"):
        _kv_new_to_cache_layout(torch.zeros(d, kv, s + 1), kv, s, d, head_dim_first=True)


def test_prefill_nki_kernel_is_opt_in_default_off(monkeypatch):
    """decode_prefill's NKI attention_cte path is opt-in, default OFF, even on NC-v3+ -- it is the
    only causal_mask=True call site of this kernel in the plugin and fails to compile under
    neuron_native_lite (round 8's smoke finding; the VAE's own causal_mask=False use of the same
    kernel compiles fine). The torch (SDPA) path is the default so a text tower is usable now."""
    from vllm_omni_neuron.diffusion.attention.decode_attention import (
        PREFILL_NKI_ENV,
        prefill_uses_nki_kernel,
    )

    monkeypatch.delenv(PREFILL_NKI_ENV, raising=False)
    assert prefill_uses_nki_kernel() is False
    monkeypatch.setenv(PREFILL_NKI_ENV, "1")
    assert prefill_uses_nki_kernel() is True
    monkeypatch.setenv(PREFILL_NKI_ENV, "0")
    assert prefill_uses_nki_kernel() is False


def test_decode_prefill_uses_torch_path_by_default_on_cpu(monkeypatch):
    """Even with the gate forced on, decode_prefill never takes the NKI path off a neuron device
    (q.device.type == 'neuron' is also required) -- CPU always uses SDPA regardless of the flag."""
    import vllm_omni_neuron.diffusion.attention.decode_attention as da

    monkeypatch.setenv(da.PREFILL_NKI_ENV, "1")
    torch.manual_seed(0)
    cfg = da.DecodeAttentionConfig(
        q_heads=2, kv_heads=2, head_dim=8, max_len=16, dtype=torch.float32
    )
    cache = da.StaticKVCache(cfg, "cpu")
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    out = da.decode_prefill(cfg, cache, q, k, v)
    assert torch.isfinite(out).all()


def test_decode_step_nki_kernel_is_opt_in_default_off(monkeypatch):
    """decode_step's fused attention_decode kernel path is opt-in, default OFF (round 15: the kernel
    path's decode output was uncorrelated with the CPU oracle, rel 1.03, while the torch path
    matched at 0.0037 with an identical, verified cache). Integrators get the torch path."""
    from vllm_omni_neuron.diffusion.attention.decode_attention import (
        DECODE_STEP_NKI_ENV,
        decode_step_uses_nki_kernel,
    )

    monkeypatch.delenv(DECODE_STEP_NKI_ENV, raising=False)
    assert decode_step_uses_nki_kernel() is False
    monkeypatch.setenv(DECODE_STEP_NKI_ENV, "1")
    assert decode_step_uses_nki_kernel() is True


def test_decode_step_skips_kernel_without_opt_in_even_when_eligible(monkeypatch):
    """With an NC-v3 shape that passes can_use_decode_kernel and a fake 'neuron' device check, the
    kernel import must never happen unless the opt-in env is set: decode_step consults the gate
    BEFORE touching vllm_neuron."""
    import vllm_omni_neuron.diffusion.attention.decode_attention as da
    from vllm_omni_neuron import nc_generation as ncg

    monkeypatch.setenv("VLLM_OMNI_NEURON_CORE_GEN", "3")
    monkeypatch.delenv(da.DECODE_STEP_NKI_ENV, raising=False)
    ncg.neuron_core_generation.cache_clear()
    cfg = da.DecodeAttentionConfig(q_heads=8, kv_heads=2, head_dim=64, max_len=128)
    assert da.can_use_decode_kernel(cfg, 1)
    monkeypatch.setattr(
        da, "can_use_decode_kernel", lambda *_: pytest.fail("kernel gate consulted")
    )
    torch.manual_seed(0)
    w = da.DecodeWeights.from_separate(
        torch.randn(512, 512), torch.randn(128, 512), torch.randn(128, 512), torch.randn(512, 512)
    )
    cache = da.StaticKVCache(cfg, "cpu")
    out = da.decode_step(cfg, w, cache, torch.randn(1, 1, 512))
    assert out.shape == (1, 512) and cache.fill == 1
    ncg.neuron_core_generation.cache_clear()


def test_can_use_decode_kernel_needs_nc_v3(monkeypatch):
    from vllm_omni_neuron import nc_generation as ncg

    monkeypatch.setenv("VLLM_OMNI_NEURON_CORE_GEN", "2")
    ncg.neuron_core_generation.cache_clear()
    cfg = DecodeAttentionConfig(q_heads=8, kv_heads=1, head_dim=64, max_len=32)
    assert not can_use_decode_kernel(cfg, 1)
    ncg.neuron_core_generation.cache_clear()


# ---------------------------------------------------------------- weights packing


def test_decode_weights_from_separate_matches_manual_concat():
    torch.manual_seed(0)
    q_w, k_w, v_w, o_w = (torch.randn(n, 32) for n in (32, 16, 16, 32))
    o_w = torch.randn(32, 32)
    w = DecodeWeights.from_separate(q_w, k_w, v_w, o_w)
    x = torch.randn(2, 5, 32)
    got = x @ w.W_qkv
    want = torch.cat([x @ q_w.t(), x @ k_w.t(), x @ v_w.t()], dim=-1)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(w.W_out, o_w.t())


def test_decode_weights_to_moves_dtype_and_device():
    w = DecodeWeights.from_separate(torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8))
    w16 = w.to(torch.bfloat16)
    assert w16.W_qkv.dtype == torch.bfloat16 and w16.W_out is None


# ---------------------------------------------------------------- StaticKVCache


def test_static_kv_cache_fill_and_overflow():
    cfg = DecodeAttentionConfig(q_heads=4, kv_heads=2, head_dim=8, max_len=10, dtype=torch.float32)
    cache = StaticKVCache(cfg, "cpu")
    assert cache.k.shape == (1, 2, 10, 8) and cache.fill == 0
    cache.write_prefill(torch.ones(1, 2, 6, 8), torch.full((1, 2, 6, 8), 2.0))
    assert cache.fill == 6
    torch.testing.assert_close(cache.k[:, :, :6], torch.ones(1, 2, 6, 8))
    torch.testing.assert_close(cache.k[:, :, 6:], torch.zeros(1, 2, 4, 8))
    cache.write_decode(torch.full((1, 2, 2, 8), 3.0), torch.full((1, 2, 2, 8), 4.0))
    assert cache.fill == 8
    torch.testing.assert_close(cache.k[:, :, 6:8], torch.full((1, 2, 2, 8), 3.0))
    torch.testing.assert_close(cache.v[:, :, 6:8], torch.full((1, 2, 2, 8), 4.0))
    torch.testing.assert_close(cache.k[:, :, 8:], torch.zeros(1, 2, 2, 8))
    # the position is a device tensor (position-static compiled decode); overflow cannot raise in a
    # compiled region, so an overflowing decode write is DROPPED (never wraps) and only advances pos
    before = cache.k.clone()
    cache.write_decode(torch.full((1, 2, 5, 8), 9.0), torch.zeros(1, 2, 5, 8))
    torch.testing.assert_close(cache.k[:, :, :8], before[:, :, :8])
    assert cache.fill == 13 and bool((cache.k[:, :, 8:] == 9.0).all())  # the 2 in-range rows land
    with pytest.raises(ValueError, match="exceeds max_len"):
        StaticKVCache(cfg, "cpu").write_prefill(torch.zeros(1, 2, 11, 8), torch.zeros(1, 2, 11, 8))


def test_static_kv_cache_pos_is_a_tensor_and_write_is_exact_in_bf16():
    """Position-static decode: ``pos`` is an int32 device tensor (a graph input, not a Dynamo
    constant), and the one-hot-matmul write lands bf16 rows bit-exactly."""
    cfg = DecodeAttentionConfig(q_heads=4, kv_heads=2, head_dim=8, max_len=16, dtype=torch.bfloat16)
    cache = StaticKVCache(cfg, "cpu")
    assert cache.pos.dtype == torch.int32 and cache.pos.shape == (1,)
    torch.manual_seed(0)
    k0, v0 = torch.randn(1, 2, 5, 8).to(torch.bfloat16), torch.randn(1, 2, 5, 8).to(torch.bfloat16)
    cache.write_prefill(k0, v0)
    k1, v1 = torch.randn(1, 2, 3, 8).to(torch.bfloat16), torch.randn(1, 2, 3, 8).to(torch.bfloat16)
    cache.write_decode(k1, v1)
    assert torch.equal(cache.k[:, :, :5], k0) and torch.equal(cache.k[:, :, 5:8], k1)
    assert torch.equal(cache.v[:, :, 5:8], v1) and cache.fill == 8
    assert bool((cache.k[:, :, 8:] == 0).all())
    # visible_bias: new token i (of the 3 just written) sees slots <= 5 + i
    bias = cache.visible_bias(3, torch.float32)
    assert bias.shape == (3, 16)
    want = torch.where(torch.arange(16)[None, :] <= (5 + torch.arange(3))[:, None], 0.0, -30000.0)
    torch.testing.assert_close(bias, want)


def test_active_blocks_table_shape():
    cfg = DecodeAttentionConfig(q_heads=6, kv_heads=3, head_dim=8, max_len=10)
    cache = StaticKVCache(cfg, "cpu")
    t = cache.active_blocks_table()
    # per-head entries are the kernel's GLOBAL pool indices b*kv_heads + h for the single block b=0
    assert t.shape == (1, 3, 1) and t.dtype == torch.int32
    assert t.flatten().tolist() == [0, 1, 2]
    t2 = cache.active_blocks_table(per_head=False)
    assert t2.shape == (1, 1) and t2.dtype == torch.int32 and int(t2) == 0


def test_active_blocks_table_matches_vllm_neuron_per_head_convention():
    """The kernel's per-head table rule, as vllm_neuron's own build_per_head_block_table encodes it
    (if that helper is importable here): the 2D single-block table [[0]] expanded to kv_heads."""
    pytest.importorskip("vllm_neuron")
    try:
        from vllm_neuron.model.qwen3_vl.utils.decode_kv import build_per_head_block_table
    except ImportError:  # helper moved / absent in this vllm_neuron: the rule is tested above
        pytest.skip("build_per_head_block_table not available")
    cfg = DecodeAttentionConfig(q_heads=16, kv_heads=2, head_dim=8, max_len=128)
    cache = StaticKVCache(cfg, "cpu")
    expect = build_per_head_block_table(torch.zeros(1, 1, dtype=torch.int32), 2)
    assert torch.equal(cache.active_blocks_table(), expect.to(torch.int32))


# ---------------------------------------------------------------- prefill (torch fallback)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_decode_prefill_matches_sdpa_causal(q_heads, kv_heads):
    torch.manual_seed(0)
    head_dim, s, max_len = 16, 7, 32
    cfg = DecodeAttentionConfig(
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, max_len=max_len, dtype=torch.float32
    )
    cache = StaticKVCache(cfg, "cpu")
    q = torch.randn(1, q_heads, s, head_dim)
    k = torch.randn(1, kv_heads, s, head_dim)
    v = torch.randn(1, kv_heads, s, head_dim)
    out = decode_prefill(cfg, cache, q, k, v)
    k_full = k.repeat_interleave(cfg.num_kv_groups, dim=1)
    v_full = v.repeat_interleave(cfg.num_kv_groups, dim=1)
    ref = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True, scale=cfg.scale)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert cache.fill == s
    torch.testing.assert_close(cache.k[:, :, :s], k)


def test_decode_prefill_applies_rope_before_write():
    torch.manual_seed(0)
    head_dim, s = 8, 4
    cfg = DecodeAttentionConfig(
        q_heads=2, kv_heads=2, head_dim=head_dim, max_len=16, dtype=torch.float32
    )
    cache = StaticKVCache(cfg, "cpu")
    q = torch.randn(1, 2, s, head_dim)
    k = torch.randn(1, 2, s, head_dim)
    v = torch.randn(1, 2, s, head_dim)
    cos, sin = _rope_tables(s, head_dim)
    decode_prefill(cfg, cache, q, k, v, cos=cos, sin=sin)
    assert not torch.allclose(cache.k[:, :, :s], k)  # RoPE changed it
    assert (
        cache.k[:, :, :s].norm(dim=-1).allclose(k.norm(dim=-1), atol=1e-5)
    )  # rotation preserves norm


def test_decode_prefill_rejects_overlong_sequence():
    cfg = DecodeAttentionConfig(q_heads=2, kv_heads=2, head_dim=8, max_len=4, dtype=torch.float32)
    cache = StaticKVCache(cfg, "cpu")
    with pytest.raises(ValueError):
        decode_prefill(
            cfg, cache, torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8)
        )


# ---------------------------------------------------------------- decode (torch fallback)


def _reference_tower(cfg, weights, cos_table, sin_table):
    """A tiny from-scratch reference: project, RoPE, GQA, causal, no kernel involved at all."""

    def run(hidden_all):
        B, S, _ = hidden_all.shape
        qkv = hidden_all @ weights.W_qkv
        d = cfg.head_dim
        q_end, k_end = cfg.q_heads * d, cfg.q_heads * d + cfg.kv_heads * d
        q = qkv[..., :q_end].view(B, S, cfg.q_heads, d).transpose(1, 2)
        k = qkv[..., q_end:k_end].view(B, S, cfg.kv_heads, d).transpose(1, 2)
        v = qkv[..., k_end:].view(B, S, cfg.kv_heads, d).transpose(1, 2)
        cos, sin = cos_table[:S].to(q.dtype), sin_table[:S].to(q.dtype)
        cos_f = torch.cat([cos, cos], dim=-1)[None, None]
        sin_f = torch.cat([sin, sin], dim=-1)[None, None]

        def rotate(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        q = q * cos_f + rotate(q) * sin_f
        k = k * cos_f + rotate(k) * sin_f
        k_full = k.repeat_interleave(cfg.num_kv_groups, dim=1)
        v_full = v.repeat_interleave(cfg.num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True, scale=cfg.scale)
        flat = out.transpose(1, 2).reshape(B * S, cfg.q_heads * d)
        return (flat @ weights.W_out) if weights.W_out is not None else out

    return run


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads,with_out", [(4, 4, True), (4, 2, True), (4, 1, False)])
def test_decode_step_matches_from_scratch_reference(q_heads, kv_heads, with_out, dtype):
    torch.manual_seed(0)
    head_dim, max_len, prompt_len = 16, 64, 6
    cfg = DecodeAttentionConfig(
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, max_len=max_len, dtype=dtype
    )
    hidden_size = cfg.hidden_size
    q_w = torch.randn(q_heads * head_dim, hidden_size)
    k_w = torch.randn(kv_heads * head_dim, hidden_size)
    v_w = torch.randn(kv_heads * head_dim, hidden_size)
    o_w = torch.randn(q_heads * head_dim, hidden_size) if with_out else None
    weights = DecodeWeights.from_separate(q_w, k_w, v_w, o_w).to(dtype)
    cos_table, sin_table = _rope_tables(max_len, head_dim)
    reference = _reference_tower(cfg, weights, cos_table, sin_table)

    full_hidden = torch.randn(1, prompt_len + 3, hidden_size).to(dtype)
    ref_full = reference(full_hidden)

    cache = StaticKVCache(cfg, "cpu")
    prompt_qkv = full_hidden[:, :prompt_len] @ weights.W_qkv
    d = cfg.head_dim
    q_end, k_end = cfg.q_heads * d, cfg.q_heads * d + cfg.kv_heads * d
    q = prompt_qkv[..., :q_end].view(1, prompt_len, q_heads, d).transpose(1, 2)
    k = prompt_qkv[..., q_end:k_end].view(1, prompt_len, kv_heads, d).transpose(1, 2)
    v = prompt_qkv[..., k_end:].view(1, prompt_len, kv_heads, d).transpose(1, 2)
    decode_prefill(
        cfg,
        cache,
        q,
        k,
        v,
        cos=cos_table[:prompt_len].to(dtype),
        sin=sin_table[:prompt_len].to(dtype),
    )
    assert cache.k.dtype == dtype  # the cache never silently upcasts/downcasts past construction

    outs = []
    for t in range(prompt_len, prompt_len + 3):
        tok = full_hidden[:, t : t + 1]
        cos = cos_table[t : t + 1].to(dtype)
        sin = sin_table[t : t + 1].to(dtype)
        out = decode_step(cfg, weights, cache, tok, cos=cos, sin=sin)
        assert out.dtype == dtype
        outs.append(out)
    assert cache.fill == prompt_len + 3

    tol = 1e-4 if dtype == torch.float32 else 3e-2
    if with_out:
        got = torch.cat(outs, dim=0)
        ref_new = ref_full[prompt_len:]
    else:
        got = torch.cat(outs, dim=3).squeeze(0).permute(2, 0, 1)  # [B,heads,d,S] -> [S,heads,d]
        ref_new = ref_full[0, :, prompt_len:, :].transpose(0, 1)  # [heads,S,d] -> [S,heads,d]
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref_new, rtol=tol, atol=tol)
    else:
        # bf16: a handful of elements near zero can have a large relative error while contributing
        # nothing to the overall result; judge by relative L2 norm, as the rest of this codebase does
        # for device-vs-reference bf16 comparisons (see test_wan_vae_platform.py, smoke_platform_trn2.py).
        rel = (got.float() - ref_new.float()).norm() / ref_new.float().norm().clamp_min(1e-12)
        assert rel.item() < tol, rel.item()


def test_decode_step_rejects_mismatched_rope_dtype():
    """Regression test for the Trn2 smoke failure (RuntimeError: Expected self.dtype() ==
    dst.dtype()): an eager cross-dtype .to() on cos/sin inside this layer can fail on a real
    Neuron/XLA device even though the identical cast succeeds on CPU (see decode_attention.py's
    _apply_rope docstring). The fix is a hard contract: decode_step/decode_prefill never cast
    cos/sin themselves -- callers must hand them over already in cfg.dtype, cast on the HOST before
    any device transfer (the production caller, vllm_neuron.model.qwen3_vl, does exactly this).
    A mismatched dtype must raise a clear ValueError here, on CPU, rather than reach the device."""
    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(
        q_heads=4, kv_heads=2, head_dim=16, max_len=32, dtype=torch.bfloat16
    )
    weights = DecodeWeights.from_separate(
        torch.randn(64, 64), torch.randn(32, 64), torch.randn(32, 64), torch.randn(64, 64)
    ).to(torch.bfloat16)
    cache = StaticKVCache(cfg, "cpu")
    hidden = torch.randn(1, 3, 64, dtype=torch.bfloat16)
    cos_table, sin_table = _rope_tables(
        32, 16
    )  # fp32 RoPE tables, as a real caller would hold them

    with pytest.raises(ValueError, match="must already be in"):
        decode_prefill(
            cfg, cache, *_split(cfg, weights, hidden), cos=cos_table[:3], sin=sin_table[:3]
        )

    # correctly cast (on the host, as the contract requires): both phases succeed.
    decode_prefill(
        cfg,
        cache,
        *_split(cfg, weights, hidden),
        cos=cos_table[:3].to(cfg.dtype),
        sin=sin_table[:3].to(cfg.dtype),
    )
    assert cache.k.dtype == torch.bfloat16

    tok = torch.randn(1, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must already be in"):
        decode_step(cfg, weights, cache, tok, cos=cos_table[3:4], sin=sin_table[3:4])

    out = decode_step(
        cfg, weights, cache, tok, cos=cos_table[3:4].to(cfg.dtype), sin=sin_table[3:4].to(cfg.dtype)
    )
    assert out.dtype == torch.bfloat16
    assert cache.k.dtype == torch.bfloat16 and cache.v.dtype == torch.bfloat16  # cache never upcast


def test_decode_prefill_bf16_rope_tensors():
    """Same regression, for the prefill path: cos/sin cast on the host to cfg.dtype before the call
    must succeed; the uncast fp32 table must raise instead of silently casting on-device."""
    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(q_heads=4, kv_heads=1, head_dim=8, max_len=16, dtype=torch.bfloat16)
    cache = StaticKVCache(cfg, "cpu")
    q = torch.randn(1, 4, 5, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 5, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 5, 8, dtype=torch.bfloat16)
    cos_table, sin_table = _rope_tables(16, 8)
    with pytest.raises(ValueError, match="must already be in"):
        decode_prefill(cfg, cache, q, k, v, cos=cos_table[:5], sin=sin_table[:5])
    out = decode_prefill(
        cfg, cache, q, k, v, cos=cos_table[:5].to(cfg.dtype), sin=sin_table[:5].to(cfg.dtype)
    )
    assert out.dtype == torch.bfloat16 and cache.k.dtype == torch.bfloat16


def test_cache_writes_tolerate_noncontiguous_sources():
    """Regression test for the Trn2 smoke failure (RuntimeError: Expected self.is_contiguous()):
    StaticKVCache.write_prefill/write_decode must force contiguity before copy_ -- every real caller
    in this module hands them a .transpose()/.permute() view (q/k/v split from a fused QKV
    projection, or attention_decode's K_out/V_out), which is tolerant on CPU but the Neuron/XLA
    lowering of the equivalent copy is strict about it. Verify directly with a transposed, strided,
    explicitly non-contiguous source."""
    cfg = DecodeAttentionConfig(q_heads=2, kv_heads=2, head_dim=4, max_len=16, dtype=torch.float32)
    cache = StaticKVCache(cfg, "cpu")
    base = torch.randn(1, 5, 2, 4)  # [B, S, kv_heads, D], as a QKV split produces before transpose
    k = base.transpose(1, 2)  # [1, kv_heads, S, D] -- a non-contiguous view
    v = base.clone().transpose(1, 2)
    assert not k.is_contiguous() and not v.is_contiguous()
    cache.write_prefill(k, v)
    torch.testing.assert_close(cache.k[:, :, :5], k)
    torch.testing.assert_close(cache.v[:, :, :5], v)

    tok = base[:, 1:3].transpose(1, 2)
    assert not tok.is_contiguous()
    cache.write_decode(tok, tok)
    torch.testing.assert_close(cache.k[:, :, 5:7], tok)


def test_apply_rope_forces_contiguous_qk():
    """Regression test for the Trn2 smoke failure (round 5: diag_kernel pinned it to decode_prefill
    with q shape [1,16,12,128] stride [30720,128,2560,1], contiguous=False -- a plain
    .transpose(1,2) view of a QKV split, exactly reproduced here). _apply_rope's slice/negate/cat
    (contiguous_layout=True) or stack/flatten (False) chain on a non-contiguous q/k raised
    RuntimeError: Expected self.is_contiguous() on the real device even though CPU tolerates it for
    the identical view. _apply_rope must force q/k contiguous before touching them."""
    from vllm_omni_neuron.diffusion.attention.decode_attention import _apply_rope

    torch.manual_seed(0)
    cos_table, sin_table = _rope_tables(12, 128)
    # the exact shape from the smoke's diag_kernel: [B, S, heads, D] projected then transposed
    qkv_like = torch.randn(1, 12, 16, 128)
    q = qkv_like.transpose(1, 2)  # [1, 16, 12, 128], non-contiguous (matches the smoke's stride)
    k = torch.randn(1, 12, 2, 128).transpose(1, 2)
    assert not q.is_contiguous() and not k.is_contiguous()
    for contiguous_layout in (True, False):
        q_out, k_out = _apply_rope(q, k, cos_table, sin_table, contiguous_layout)
        assert q_out.shape == q.shape and k_out.shape == k.shape
        assert torch.isfinite(q_out).all() and torch.isfinite(k_out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_prefill_accepts_noncontiguous_qkv(dtype):
    """decode_prefill must force q/k/v contiguous at entry (the smoke's actual failing call):
    every real caller passes a QKV-split .transpose(1, 2) view, never a pre-contiguous tensor."""
    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(q_heads=4, kv_heads=2, head_dim=8, max_len=16, dtype=dtype)
    cache = StaticKVCache(cfg, "cpu")
    qkv = torch.randn(1, 6, 4 + 2 + 2, 8, dtype=dtype)  # [B, S, q_heads+2*kv_heads, D]
    q = qkv[:, :, :4].transpose(1, 2)
    k = qkv[:, :, 4:6].transpose(1, 2)
    v = qkv[:, :, 6:8].transpose(1, 2)
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    cos_table, sin_table = _rope_tables(16, 8)
    out = decode_prefill(
        cfg, cache, q, k, v, cos=cos_table[:6].to(dtype), sin=sin_table[:6].to(dtype)
    )
    assert out.shape == q.shape and torch.isfinite(out).all()
    assert cache.fill == 6


def test_decode_step_torch_fallback_accepts_noncontiguous_hidden_and_cache_slice():
    """decode_step's torch fallback must tolerate a non-contiguous K_cache/V_cache slice
    (K_cache[:, :, :pos0] is itself non-contiguous for any pos0 short of the cache's full length)
    concatenated with freshly-split, non-contiguous k/v -- both are the real shapes this path sees."""
    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(q_heads=2, kv_heads=2, head_dim=4, max_len=16, dtype=torch.float32)
    weights = DecodeWeights.from_separate(
        torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8)
    )
    cache = StaticKVCache(cfg, "cpu")
    hidden = torch.randn(1, 3, 8)
    decode_prefill(cfg, cache, *_split(cfg, weights, hidden))
    assert cache.fill == 3  # K_cache[:, :, :3] below is a genuinely partial, non-contiguous slice
    tok = torch.randn(1, 1, 8)
    out = decode_step(cfg, weights, cache, tok)
    assert torch.isfinite(out).all()


def test_prefill_attention_cte_kernel_is_module_scope():
    """Regression test for the Trn2 smoke failure (BackendCompilerFailed: 'failed to legalize
    operation torch.operator that was explicitly marked illegal'): the @nki.jit kernel must be
    defined at MODULE scope, like the VAE module's _vae_attention_kernel (whose wrap_nki/compile
    pattern this mirrors and which does compile cleanly) -- a fresh @nki.jit closure rebuilt inside
    _prefill_attention_cte on every call did not get the same treatment under torch.compile and
    lowered as an opaque illegal op instead of the recognized NKI HOP."""
    import inspect

    import vllm_omni_neuron.diffusion.attention.decode_attention as da

    # @nki.jit wraps the function into its own nki.framework.kernel.Kernel object (so __module__
    # is theirs, not ours) -- what matters is that the module itself holds a stable reference to
    # it, i.e. it is a plain module attribute, not rebuilt inside _prefill_attention_cte on every
    # call (the bug this guards against).
    kernel = getattr(da, "_prefill_attention_cte_kernel", None)
    assert kernel is not None
    assert (
        kernel is da._prefill_attention_cte_kernel
    )  # same object on repeated access -> module-level
    # the caller must not define its own @nki.jit kernel inline -- that is exactly the regression
    # (check for the decorator as a code line, not the docstring's prose mention of it)
    src_lines = [ln.strip() for ln in inspect.getsource(da._prefill_attention_cte).splitlines()]
    assert "@nki.jit" not in src_lines


def test_prefill_attention_cte_receives_contiguous_inputs():
    """The NKI prefill kernel path only runs on NC-v3+ device (not exercised on CPU), so guard the
    call site structurally: q[0]*scale, k_full[0], v_full[0] (produced by repeat_interleave +
    indexing, non-contiguous in general) must each be wrapped in .contiguous() before the call."""
    import inspect

    import vllm_omni_neuron.diffusion.attention.decode_attention as da

    src = inspect.getsource(da.decode_prefill)
    start = src.index("_prefill_attention_cte(")
    end = src.index(")[None]", start) + len(")[None]")
    call_expr = src[start:end]
    assert call_expr.count(".contiguous()") == 3, call_expr


def test_decode_step_multi_token_block_is_causal_among_new_tokens():
    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(q_heads=2, kv_heads=2, head_dim=8, max_len=32, dtype=torch.float32)
    weights = DecodeWeights.from_separate(
        torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16)
    )
    cache = StaticKVCache(cfg, "cpu")
    hidden = torch.randn(1, 5, 16)
    decode_prefill(cfg, cache, *_split(cfg, weights, hidden))
    block = torch.randn(1, 4, 16)
    out_block = decode_step(cfg, weights, cache, block)
    cache2 = StaticKVCache(cfg, "cpu")
    decode_prefill(cfg, cache2, *_split(cfg, weights, hidden))
    outs_one = [decode_step(cfg, weights, cache2, block[:, i : i + 1]) for i in range(4)]
    torch.testing.assert_close(out_block, torch.cat(outs_one, dim=0), rtol=1e-4, atol=1e-4)


def _split(cfg, weights, hidden):
    qkv = hidden @ weights.W_qkv
    d = cfg.head_dim
    q_end, k_end = cfg.q_heads * d, cfg.q_heads * d + cfg.kv_heads * d
    B, S, _ = hidden.shape
    q = qkv[..., :q_end].view(B, S, cfg.q_heads, d).transpose(1, 2)
    k = qkv[..., q_end:k_end].view(B, S, cfg.kv_heads, d).transpose(1, 2)
    v = qkv[..., k_end:].view(B, S, cfg.kv_heads, d).transpose(1, 2)
    return q, k, v


def test_decode_step_compiles_once_for_every_position():
    """The whole point of the tensor position: compiled ``decode_step`` must trace ONE graph and
    reuse it for every decode position (no per-step recompile, no recompile-limit failure under
    ``fullgraph=True`` on the ninth token), and the compiled steps must match the eager ones."""
    import torch._dynamo

    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(q_heads=4, kv_heads=2, head_dim=8, max_len=32, dtype=torch.float32)
    weights = DecodeWeights.from_separate(
        torch.randn(32, 32) * 0.2, torch.randn(16, 32) * 0.2, torch.randn(16, 32) * 0.2
    )
    compiles = []

    def counting_backend(gm, example_inputs):
        compiles.append(gm)
        return gm.forward

    torch._dynamo.reset()
    step_c = torch.compile(decode_step, backend=counting_backend, fullgraph=True, dynamic=False)
    cache_c, cache_e = StaticKVCache(cfg, "cpu"), StaticKVCache(cfg, "cpu")
    toks = torch.randn(12, 1, 1, 32)
    for t in range(12):  # > Dynamo's default recompile limit of 8
        out_c = step_c(cfg, weights, cache_c, toks[t])
        out_e = decode_step(cfg, weights, cache_e, toks[t])
        torch.testing.assert_close(out_c, out_e)
    assert len(compiles) == 1, f"decode_step recompiled: {len(compiles)} graphs for 12 positions"
    assert cache_c.fill == 12 and torch.equal(cache_c.k, cache_e.k)
    torch._dynamo.reset()


def test_decode_step_overflow_is_dropped_not_wrapped():
    """``pos`` is data inside the compiled step, so overflow cannot raise there; the write past
    ``max_len`` must be dropped (no slot matches) and the earlier slots left intact."""
    cfg = DecodeAttentionConfig(q_heads=2, kv_heads=2, head_dim=8, max_len=3, dtype=torch.float32)
    weights = DecodeWeights.from_separate(
        torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16)
    )
    cache = StaticKVCache(cfg, "cpu")
    decode_step(cfg, weights, cache, torch.randn(1, 2, 16))
    decode_step(cfg, weights, cache, torch.randn(1, 1, 16))
    before = cache.k.clone()
    decode_step(cfg, weights, cache, torch.randn(1, 1, 16))
    assert cache.fill == 4 and torch.equal(cache.k, before)
