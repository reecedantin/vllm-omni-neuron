# SPDX-License-Identifier: Apache-2.0
"""Shared block-sparse attention (diffusion/attention/block_sparse.py): plan construction, the fp32
masked reference, and the three executors' torch paths, exact against the reference on CPU; the L3
NKI kernel on the NKI CPU simulator (``NKI_SIMULATOR=1``) at a tiny shape.

Covers both consumers' geometries: SLA (128-row query blocks, 64- or 128-row key blocks, top-15%,
trailing pad rows dropped) and SSTA (384-token tiles, text tiles always kept, zero pad rows that
STAY in the softmax).
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni_neuron.diffusion.attention import block_sparse as BS
from vllm_omni_neuron.diffusion.attention.block_sparse import (
    BlockSparsePlan,
    attend_dense_bias,
    attend_gather_dense,
    attend_index_list,
    block_bias,
    index_list_inputs,
    plan_from_block_mask,
    plan_from_lists,
    reference_attention,
    sla_block_mask,
)

D = 128


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _qkv(h, lq, lk, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(h, lq, D, generator=g)
    k = torch.randn(h, lk, D, generator=g)
    v = torch.randn(h, lk, D, generator=g)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _sla_plan(h=2, nq=4, nk=11, k_block=64, keep=0.3, lk_real=None, seed=1):
    """Random top-k block mask + SLA tail padding (lk_real < nK * k_block)."""
    g = torch.Generator().manual_seed(seed)
    score = torch.randn(h, nq, nk, generator=g)
    topk = max(1, int(keep * nk))
    mask = torch.zeros(h, nq, nk, dtype=torch.bool)
    mask.scatter_(2, torch.topk(score, topk, dim=-1).indices, True)
    # one query block keeps more (ragged counts -> pads)
    mask[0, 1, : topk + 2] = True
    lk = nk * k_block
    lk_real = lk if lk_real is None else lk_real
    valid = torch.arange(lk) < lk_real
    return plan_from_block_mask(mask, 128, k_block, key_valid=valid), lk


# --------------------------------------------------------------------------------- plan + reference
def test_plan_lists_are_sorted_real_first_and_round_trip_to_the_mask():
    plan, _ = _sla_plan()
    assert plan.lists.dtype == torch.int32 and plan.kp == int(plan.counts.max())
    for hh in range(plan.heads):
        for qi in range(plan.n_q_blocks):
            n = int(plan.counts[hh, qi])
            row = plan.lists[hh, qi]
            assert (row[:n] >= 0).all() and (row[n:] == -1).all()
            assert (row[:n].diff() > 0).all()  # ascending ids
    mask = plan.block_mask()
    assert torch.equal(
        plan_from_block_mask(mask, 128, plan.k_block, key_valid=plan.key_valid).lists, plan.lists
    )
    assert torch.equal(
        plan_from_lists(
            plan.lists, plan.counts, 128, plan.k_block, plan.n_k_blocks, plan.key_valid
        ).lists,
        plan.lists,
    )


def test_pad_to_widens_the_lists_with_dummy_pads_only():
    plan, _ = _sla_plan()
    wide = plan_from_block_mask(
        plan.block_mask(), 128, plan.k_block, key_valid=plan.key_valid, pad_to=8
    )
    assert wide.kp % 8 == 0 and wide.kp >= plan.kp
    assert (
        torch.equal(wide.lists[..., : plan.kp], plan.lists)
        and (wide.lists[..., plan.kp :] == -1).all()
    )
    assert torch.equal(wide.block_mask(), plan.block_mask())


def test_token_mask_drops_tail_pad_keys_and_reference_ignores_them():
    plan, lk = _sla_plan(lk_real=11 * 64 - 37)
    q, k, v = _qkv(2, 4 * 128, lk)
    tok = plan.token_mask(q.shape[1], lk)
    assert tok.shape == (2, 512, lk) and not tok[..., lk - 37 :].any()
    out = reference_attention(q, k, v, plan)
    # pad keys get arbitrary values: the output must not depend on them
    k2, v2 = k.clone(), v.clone()
    k2[:, lk - 37 :] = 7.0
    v2[:, lk - 37 :] = -3.0
    torch.testing.assert_close(reference_attention(q, k2, v2, plan), out, rtol=0, atol=0)


def test_sla_block_mask_matches_lightx2v_semantics():
    """Pooled mean-Q . mean-(K - mean K), top max(1, int(0.15 nK)), partial blocks divide by real rows."""
    torch.manual_seed(0)
    h, lq, lk = 2, 3 * 128, 10 * 64
    q, k, v = _qkv(h, lq, lk)
    lq_real, lk_real = lq - 50, lk - 20
    mask = sla_block_mask(q, k, 128, 64, 0.15, lq_real=lq_real, lk_real=lk_real)
    assert mask.shape == (h, 3, 10) and (mask.sum(-1) == 1).all()  # int(0.15*10) = 1
    kf = k.float() - k.float()[:, :lk_real].mean(1, keepdim=True)
    qm = torch.stack([q[:, 0:128].mean(1), q[:, 128:256].mean(1), q[:, 256:lq_real].mean(1)], 1)
    km = torch.stack([kf[:, i * 64 : min((i + 1) * 64, lk_real)].mean(1) for i in range(10)], 1)
    want = torch.zeros_like(mask)
    want.scatter_(2, (qm @ km.transpose(1, 2)).argmax(-1, keepdim=True), True)
    assert torch.equal(mask, want)
    mask20 = sla_block_mask(q, k, 128, 64, 0.25)
    assert (mask20.sum(-1) == 2).all()


# ---------------------------------------------------------------------------------- L1: bias path
@pytest.mark.parametrize("k_block", [64, 128])
def test_block_bias_expansion_equals_the_token_mask(k_block):
    plan, lk = _sla_plan(k_block=k_block, lk_real=11 * k_block - 9)
    lq = 4 * 128
    bias = block_bias(plan, lq, lk)
    assert bias.shape == (2, lq, lk) and bias.dtype == torch.bfloat16
    kept = bias == 0
    torch.testing.assert_close(kept, plan.token_mask(lq, lk))
    assert set(bias.unique().abs().tolist()) <= {0.0, -BS.NEG_BIAS}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_l1_dense_bias_matches_reference(dtype):
    plan, lk = _sla_plan(lk_real=11 * 64 - 37)
    q, k, v = _qkv(2, 4 * 128, lk, dtype=dtype)
    ref = reference_attention(q, k, v, plan)
    out = attend_dense_bias(q, k, v, plan)
    assert out.shape == ref.shape and out.dtype == dtype
    assert _rel(out, ref) < (
        1e-6 if dtype == torch.float32 else 1e-2
    )  # -30000 vs -inf: exp underflows to 0
    assert _rel(attend_dense_bias(q, k, v, plan, q_chunk=256), ref) < 1e-2


# -------------------------------------------------------------------------------- L2: gather path
def test_l2_gather_order_puts_full_blocks_first_partial_last_and_bounds_count_valid_keys():
    plan, lk = _sla_plan(lk_real=11 * 64 - 37)  # block 10 is partial (27 valid rows)
    lists_g, bound_max = BS._lists_gather_order(plan)
    assert lists_g.shape == plan.lists.shape and lists_g.dtype == torch.int32
    for hh in range(plan.heads):
        for qi in range(plan.n_q_blocks):
            n = int(plan.counts[hh, qi])
            row = lists_g[hh, qi].tolist()
            real, pads = row[:n], row[n:]
            assert all(p == plan.n_k_blocks for p in pads)  # dummy id
            assert sorted(real) == sorted(plan.lists[hh, qi, :n].tolist())
            if 10 in real:
                assert real[-1] == 10
                assert int(bound_max[hh, qi]) == (n - 1) * 64 + 27
            else:
                assert int(bound_max[hh, qi]) == n * 64


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_l2_gather_dense_matches_reference(dtype):
    plan, lk = _sla_plan(lk_real=11 * 64 - 37)
    q, k, v = _qkv(2, 4 * 128, lk, dtype=dtype)
    ref = reference_attention(q, k, v, plan)
    out = attend_gather_dense(q, k, v, plan)
    assert out.shape == ref.shape
    assert _rel(out, ref) < (1e-6 if dtype == torch.float32 else 1e-2)


def test_l2_refuses_invalid_keys_that_are_not_a_trailing_run():
    plan, lk = _sla_plan()
    valid = plan.key_valid.clone()
    valid[5] = False  # a hole inside block 0
    bad = BlockSparsePlan(plan.lists, plan.counts, 128, 64, plan.n_k_blocks, valid)
    with pytest.raises(AssertionError, match="trailing"):
        BS._lists_gather_order(bad)


# ----------------------------------------------------------------------------- L3: kernel inputs
def test_l3_inputs_pad_to_pass_width_zero_invalid_keys_and_build_rowstarts():
    plan, lk = _sla_plan(lk_real=11 * 64 - 37)
    q, k, v = _qkv(2, 4 * 128, lk, dtype=torch.bfloat16)
    k3, v_ext, blocks, rowstart = index_list_inputs(plan, k, v, pb=16)
    nk, bs = plan.n_k_blocks, plan.k_block  # 64-row plan blocks stay 64-row kernel blocks
    assert k3.shape == (2, nk + 1, bs, D) and torch.equal(
        k3[:, nk], torch.zeros(2, bs, D, dtype=k.dtype)
    )
    assert torch.equal(k3[:, 10, 27:], torch.zeros(2, bs - 27, D, dtype=k.dtype))  # tail pad zeroed
    assert v_ext.shape == (2, (nk + 1) * bs, D + 32)
    assert torch.equal(v_ext[:, : lk - 37, D], torch.ones(2, lk - 37, dtype=v.dtype))
    assert not v_ext[:, lk - 37 :, D].any()  # pads + dummy invalid
    kp = blocks.shape[2]
    assert kp % 16 == 0 and kp >= plan.kp and blocks.shape == (2, 4, kp, 1)
    assert (blocks[..., plan.kp :, 0] == nk).all()  # pads -> dummy id
    assert rowstart.shape == (2, 4, kp // 16, 128, 1)
    p = torch.arange(128)
    for ps in range(kp // 16):
        want = blocks[0, 1, ps * 16 + (p % 16), 0].long() * bs + p // 16
        assert torch.equal(rowstart[0, 1, ps, :, 0].long(), want)


def test_l3_inputs_expand_384_tiles_into_three_128_row_sub_blocks():
    mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    mask[0, 0, [0, 3]] = True
    mask[0, 1, [4]] = True
    plan = plan_from_block_mask(mask, 384, 384)
    k = torch.randn(1, 5 * 384, D).to(torch.bfloat16)
    k3, v_ext, blocks, rowstart = index_list_inputs(plan, k, k, pb=16)
    assert k3.shape == (1, 16, 128, D)  # 15 sub-blocks + dummy
    assert blocks.shape == (1, 6, 16, 1)  # 2 plan query blocks of 384 rows -> 6 kernel tiles of 128
    assert (
        blocks[0, 0, :6, 0].tolist() == [0, 1, 2, 9, 10, 11] and (blocks[0, 0, 6:, 0] == 15).all()
    )
    assert torch.equal(blocks[0, 0], blocks[0, 1]) and torch.equal(blocks[0, 0], blocks[0, 2])
    assert blocks[0, 3, :3, 0].tolist() == [12, 13, 14] and (blocks[0, 3, 3:, 0] == 15).all()
    torch.testing.assert_close(k3[0].reshape(-1, D)[: 5 * 384], k[0])
    assert v_ext.shape == (1, 16 * 128, D + 32) and (v_ext[0, : 5 * 384, D] == 1).all()


def test_l3_torch_path_equals_reference_and_ssta_zero_pads_stay_in_the_softmax():
    """SSTA geometry: 384-token tiles, text tiles always kept, zero-padded rows with validity 1."""
    h, bs = 1, 384
    n_video, n_text = 5, 2
    nk = n_video + n_text
    lk = nk * bs
    q = torch.randn(h, 3 * 128, D)
    k = torch.randn(h, lk, D)
    v = torch.randn(h, lk, D)
    pad_rows = (torch.arange(lk) >= 4 * bs + 200) & (
        torch.arange(lk) < 5 * bs
    )  # tile 4 half padded
    k[:, pad_rows] = 0
    v[:, pad_rows] = 0
    mask = torch.zeros(h, 1, nk, dtype=torch.bool)
    mask[0, 0, [0, 4, 5, 6]] = True  # two video tiles (one padded) + both text tiles
    plan = plan_from_block_mask(mask, 384, bs)  # key_valid all True: zero pads count
    plan = BlockSparsePlan(plan.lists, plan.counts, 384, bs, nk, plan.key_valid)
    ref = reference_attention(q, k, v, plan)
    # zero keys contribute exp(0 - m): dropping them changes the answer
    plan_drop = BlockSparsePlan(plan.lists, plan.counts, 384, bs, nk, ~pad_rows)
    assert _rel(reference_attention(q, k, v, plan_drop), ref) > 1e-3
    out = attend_index_list(q, k, v, plan)  # torch path off-device == reference
    torch.testing.assert_close(out, ref)
    k3, v_ext, blocks, rowstart = index_list_inputs(plan, k, v)
    assert (v_ext[0, : nk * bs, D][pad_rows] == 1).all() and (v_ext[0, nk * bs :, D] == 0).all()


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
@pytest.mark.parametrize("k_block", [64, 128, 384])
def test_l3_kernel_on_nki_simulator_matches_reference(k_block):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.bsa_index_list import bsa_index_list

    torch.manual_seed(0)
    h, nq = 1, 2
    nk = {64: 20, 128: 9, 384: 4}[k_block]
    lk_real = nk * k_block - 40
    q, k, v = _qkv(h, nq * 128, nk * k_block, dtype=torch.bfloat16)
    mask = torch.zeros(h, nq, nk, dtype=torch.bool)
    mask[0, 0, [0, 2, nk - 1]] = True
    mask[0, 1, [1, 2, 3, nk - 1]] = True
    plan = plan_from_block_mask(mask, 128, k_block, key_valid=torch.arange(nk * k_block) < lk_real)
    k3, v_ext, blocks, rowstart = index_list_inputs(plan, k, v)
    qT = (q.float() * D**-0.5).to(torch.bfloat16).transpose(1, 2).contiguous()
    out = wrap_nki(bsa_index_list)[2](
        qT=qT, k3=k3, v_ext=v_ext, blocks_i32=blocks, rowstart_i32=rowstart, pb=16
    )
    ref = reference_attention(q, k, v, plan)
    assert out.shape == ref.shape
    assert _rel(out, ref) < 2e-2


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
def test_l3_kernel_on_nki_simulator_ssta_geometry_with_odd_tile_count(monkeypatch):
    """384-row query blocks (lists repeated per 128-row tile), 384-row key tiles expanded to
    sub-blocks, zero pad rows kept in the softmax, and an odd kernel tile count (3) padded for the
    LNC2 pair -- the whole L3 path as a model would call it."""
    import vllm_omni_neuron.diffusion.attention.block_sparse as BSm

    torch.manual_seed(0)
    h, nk, bs = 1, 5, 384
    lk = nk * bs
    q, k, v = _qkv(h, 384, lk, dtype=torch.bfloat16)
    pad = (torch.arange(lk) >= 3 * bs + 100) & (torch.arange(lk) < 4 * bs)
    k[:, pad] = 0
    v[:, pad] = 0
    mask = torch.zeros(h, 1, nk, dtype=torch.bool)
    mask[0, 0, [1, 3, 4]] = True
    plan = plan_from_block_mask(mask, 384, bs)
    ref = reference_attention(q, k, v, plan)
    monkeypatch.setattr(BSm, "_kernel_ok", lambda x: True)  # force the kernel path on the simulator
    out = attend_index_list(q, k, v, plan)
    assert out.shape == ref.shape and _rel(out, ref) < 2e-2


# ================================================================ L4: streaming kernel (bsa_stream)
def test_l4_pass_picker_pads_least_with_wide_passes():
    assert BS.stream_pass_blocks(88, 64) == 44  # 2 x 2816 keys, no padding
    assert BS.stream_pass_blocks(44, 128) == 22
    assert BS.stream_pass_blocks(66, 384) == 6  # 11 x 2304
    for kp, kb in [(7, 64), (5, 128), (3, 384), (13, 32)]:
        assert (BS.stream_pass_blocks(kp, kb) * kb) % 128 == 0


def _ssta_case(n_q=3, nk=6, n_text=2, seed=3):
    """384-row tiles, text tiles always kept, a zero-padded video tile (pad rows STAY valid)."""
    g = torch.Generator().manual_seed(seed)
    bs = 384
    lk = nk * bs
    q, k, v = _qkv(2, n_q * bs, lk, seed=seed, dtype=torch.bfloat16)
    pad = (torch.arange(lk) >= 2 * bs + 200) & (torch.arange(lk) < 3 * bs)
    q[:, pad[: n_q * bs]] = 0
    k[:, pad] = 0
    v[:, pad] = 0
    mask = torch.rand(2, n_q, nk, generator=g) < 0.4
    mask[..., nk - n_text :] = True
    mask[..., 2] = True  # the zero-padded tile is listed everywhere
    mask[0, 1, :] = True  # one full list -> ragged counts, pads elsewhere
    return q, k, v, plan_from_block_mask(mask, bs, bs), pad


@pytest.mark.parametrize("k_block", [64, 128])
@pytest.mark.parametrize("pb", [None, 2])
def test_l4_operands_emulate_to_reference_sla(k_block, pb):
    q, k, v = _qkv(2, 3 * 128, 11 * k_block, dtype=torch.bfloat16)
    plan, lk = _sla_plan(h=2, nq=3, nk=11, k_block=k_block, lk_real=11 * k_block - 37)
    ops = BS.stream_inputs(plan, q, k, v, pb=pb)
    n = ops.pop("n_real_items")
    assert ops["lists"].shape[0] % 2 == 0 and ops["lists"].shape[1] % ops["pb"] == 0
    out = BS.stream_emulate(**ops)[: n * 128].reshape(2, 3 * 128, D)
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    assert _rel(out, ref) < 5e-3  # bf16 rounding of the pre-scaled q only


def test_l4_operands_emulate_to_reference_ssta_keeps_zero_pads_in_softmax():
    q, k, v, plan, pad = _ssta_case()
    ops = BS.stream_inputs(plan, q, k, v, pb=2)
    n = ops.pop("n_real_items")
    out = BS.stream_emulate(**ops)[: n * 384].reshape(q.shape)
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    assert _rel(out, ref) < 5e-3
    # had the zero pad rows been dropped instead, the result would differ well above bf16 level
    plan_drop = BlockSparsePlan(plan.lists, plan.counts, 384, 384, plan.n_k_blocks, ~pad)
    ref_drop = reference_attention(q.float(), k.float(), v.float(), plan_drop)
    assert _rel(ref_drop, ref) > 10 * _rel(out, ref)


def _sim_kernel(ops):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.bsa_stream import bsa_stream

    return wrap_nki(bsa_stream)[2](**ops)


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
@pytest.mark.parametrize("k_block,pb", [(64, 4), (64, 6), (128, 2), (128, 1)])
def test_l4_kernel_on_nki_simulator_sla(k_block, pb):
    """Tail pad keys dropped, ragged lists (pads masked by the bound), several passes per item, an
    odd item count (3 heads x 3 tiles = 9, padded for the LNC2 pair)."""
    h, nq, nk = 3, 3, 13
    q, k, v = _qkv(h, nq * 128, nk * k_block, seed=5, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(2)
    mask = torch.rand(h, nq, nk, generator=g) < 0.35
    mask[..., nk - 1] = True  # the partial tail block is listed everywhere
    mask[1, 2, :] = True
    plan = plan_from_block_mask(
        mask, 128, k_block, key_valid=torch.arange(nk * k_block) < nk * k_block - 45
    )
    ops = BS.stream_inputs(plan, q, k, v, pb=pb)
    n = ops.pop("n_real_items")
    out = _sim_kernel(ops)[: n * 128].reshape(h, nq * 128, D)
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    assert _rel(out, ref) < 1e-2


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
def test_l4_kernel_on_nki_simulator_ssta_384_query_blocks(monkeypatch):
    """384-row query blocks (three sub-tiles share one K/V fetch), 384-row key tiles in one DMA,
    text tiles always kept, zero pad rows kept in the softmax, via the public executor."""
    import vllm_omni_neuron.diffusion.attention.block_sparse as BSm

    q, k, v, plan, _ = _ssta_case()
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    monkeypatch.setattr(BSm, "_kernel_ok", lambda x: True)
    out = BS.attend_stream(q, k, v, plan, pb=2)
    assert out.shape == ref.shape and _rel(out, ref) < 1e-2


@pytest.mark.parametrize("k_block", [128, 384])
def test_l4_packed_operands_emulate_to_reference(k_block):
    nk = 7 if k_block == 128 else 4
    q, k, v = _qkv(2, 3 * 128, nk * k_block, seed=9, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(4)
    mask = torch.rand(2, 3, nk, generator=g) < 0.5
    mask[..., nk - 1] = True
    plan = plan_from_block_mask(
        mask, 128, k_block, key_valid=torch.arange(nk * k_block) < nk * k_block - 20
    )
    ops = BS.stream_inputs(plan, q, k, v, pb=2, packed=True)
    n = ops.pop("n_real_items")
    assert ops["kT_blk"].shape[1:] == (128, 2 * k_block)
    out = BS.stream_emulate(**ops)[: n * 128].reshape(q.shape)
    assert _rel(out, reference_attention(q.float(), k.float(), v.float(), plan)) < 5e-3


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
@pytest.mark.parametrize("k_block,pb,v_queue", [(128, 4, 1), (128, 3, 0), (128, 2, 1), (128, 3, 2)])
def test_l4_packed_kernel_on_nki_simulator_sla(k_block, pb, v_queue):
    h, nq, nk = 3, 3, 13
    q, k, v = _qkv(h, nq * 128, nk * k_block, seed=5, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(2)
    mask = torch.rand(h, nq, nk, generator=g) < 0.35
    mask[..., nk - 1] = True
    mask[1, 2, :] = True
    plan = plan_from_block_mask(
        mask, 128, k_block, key_valid=torch.arange(nk * k_block) < nk * k_block - 45
    )
    ops = BS.stream_inputs(plan, q, k, v, pb=pb, packed=True)
    n = ops.pop("n_real_items")
    ops["v_queue"] = v_queue
    out = _sim_kernel(ops)[: n * 128].reshape(h, nq * 128, D)
    assert _rel(out, reference_attention(q.float(), k.float(), v.float(), plan)) < 1e-2


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
def test_l4_packed_kernel_on_nki_simulator_ssta(monkeypatch):
    """384-row tiles packed (K^T + three V sub-tiles per partition row), chunks straddling tiles."""
    import vllm_omni_neuron.diffusion.attention.block_sparse as BSm

    q, k, v, plan, _ = _ssta_case()
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    monkeypatch.setattr(BSm, "_kernel_ok", lambda x: True)
    out = BS.attend_stream(q, k, v, plan, pb=2, packed=True, v_queue=1)
    assert out.shape == ref.shape and _rel(out, ref) < 1e-2


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
def test_l4_prebuilt_stream_plan_entry_point(monkeypatch):
    """The model-facing path: StreamPlan built once on the host, reused across calls; SLA tail pad
    rows zeroed through key_valid."""
    import vllm_omni_neuron.diffusion.attention.block_sparse as BSm

    h, nq, nk, kb = 2, 3, 9, 128
    q, k, v = _qkv(h, nq * 128, nk * kb, seed=11, dtype=torch.bfloat16)
    valid = torch.arange(nk * kb) < nk * kb - 50
    g = torch.Generator().manual_seed(6)
    mask = torch.rand(h, nq, nk, generator=g) < 0.4
    mask[..., nk - 1] = True
    plan = plan_from_block_mask(mask, 128, kb, key_valid=valid)
    sp = BS.stream_plan(plan)
    assert sp.packed and sp.n_items % 2 == 0 and sp.n_real_items == h * nq
    monkeypatch.setattr(BSm, "_kernel_ok", lambda x: True)
    ref = reference_attention(q.float(), k.float(), v.float(), plan)
    for _ in range(2):
        out = BS.attend_stream(q, k, v, sp, key_valid=valid)
        assert out.shape == ref.shape and _rel(out, ref) < 1e-2


@pytest.mark.parametrize("case", ["sla", "ssta"])
def test_l4_device_list_builder_matches_host_plan(case):
    """stream_lists (compile-friendly, per-layer) produces operands equivalent to stream_plan: SLA's
    sorted top-k with a partial tail block; SSTA's ragged -1-padded lists with zero pads valid."""
    if case == "sla":
        h, nq, nk, kb, topk, tail = 3, 5, 12, 128, 4, 37
        g = torch.Generator().manual_seed(8)
        idx = torch.rand(h, nq, nk, generator=g).topk(topk, -1).indices.sort(-1).values
        idx[0, 0] = torch.tensor([2, 5, 9, nk - 1])  # the partial tail block selected
        counts = torch.full((h, nq), topk)
        valid = torch.arange(nk * kb) < nk * kb - tail
    else:
        h, nq, nk, kb, tail = 2, 3, 6, 384, 0
        mask = torch.rand(h, nq, nk, generator=torch.Generator().manual_seed(3)) < 0.5
        mask[..., -1] = True
        counts = mask.sum(-1)
        order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)
        idx = torch.where(torch.arange(nk) < counts[..., None], order, torch.full_like(order, -1))
        valid = torch.ones(nk * kb, dtype=torch.bool)
    mask = torch.zeros(h, nq, nk, dtype=torch.bool)
    real = idx >= 0
    hi, qi, _ = torch.nonzero(real, as_tuple=True)
    mask[hi, qi, idx[real]] = True
    plan = plan_from_block_mask(mask, 128, kb, key_valid=valid, kp=idx.shape[-1])
    sp = BS.stream_plan(plan, pb=2)
    lists, bounds = BS.stream_lists(idx, counts, kb, nk, pb=2, n_tail_invalid=tail)
    q, k, v = _qkv(h, nq * 128, nk * kb, seed=1, dtype=torch.bfloat16)
    kT, vb = BS.stream_kv(sp, k, v, valid)
    sp2 = sp.with_lists(lists, bounds)
    kw = dict(q_block=128, pb=2, packed=1, k_block=kb)
    a = BS.stream_emulate(BS.stream_q(sp, q), kT, vb, sp.lists, sp.bounds, **kw)
    b = BS.stream_emulate(BS.stream_q(sp2, q), kT, vb, sp2.lists, sp2.bounds, **kw)
    ref = reference_attention(q.float(), k.float(), v.float(), plan).reshape(-1, D)
    assert _rel(b[: h * nq * 128], ref) < 5e-3 and _rel(a, b) < 1e-5


def _vsa_case(h=2, n_pairs=3, nk=20, k_sel=5, n_prefix=1, seed=12):
    g = torch.Generator().manual_seed(seed)
    sel = torch.zeros(h, 2 * n_pairs, nk, dtype=torch.bool)
    sel[..., :n_prefix] = True
    pick = torch.rand(h, 2 * n_pairs, nk - n_prefix, generator=g).topk(k_sel, -1).indices + n_prefix
    sel.scatter_(2, pick, True)
    q, k, v = _qkv(h, 2 * n_pairs * 64, nk * 64, seed=seed, dtype=torch.bfloat16)
    ref = reference_attention(q.float(), k.float(), v.float(), plan_from_block_mask(sel, 64, 64))
    return sel, q, k, v, ref


def test_l4_half_masks_emulate_to_per_tile_reference():
    sel, q, k, v, ref = _vsa_case()
    plan = BS.pair_union_plan(sel, 64)
    sp = BS.stream_plan(plan, pb=plan.kp + plan.kp % 2)  # one pass: plain softmax per item
    kmask, qind = BS.stream_half_masks(sp, sel)
    kT, vb = BS.stream_kv(sp, k, v)
    qT = BS.stream_q(sp, q)
    # emulate: union attention per half with the additive mask
    d = 128
    out = torch.empty(sp.n_items * 128, d)
    for i in range(sp.n_items):
        ids = sp.lists[i].long()
        kk = kT[ids].float().permute(0, 2, 1).reshape(-1, d)
        vv = vb[ids].float().reshape(-1, d)
        s = qT[:, i * 128 : (i + 1) * 128].float().t() @ kk.t()
        s = s + qind.float().t() @ kmask[i].float()
        s = s.masked_fill(torch.arange(s.shape[1]) >= sp.bounds[i, 0, 0], float("-inf"))
        out[i * 128 : (i + 1) * 128] = torch.softmax(s, -1) @ vv
    assert sp.bounds.shape[2] == 1
    assert _rel(out[: sp.n_real_items * 128].reshape(q.shape), ref) < 5e-3


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
@pytest.mark.parametrize("pb", [2, 4])
def test_l4_half_masks_kernel_on_nki_simulator_vsa_pairs(pb):
    """VSA geometry: pairs of 64-row query tiles over the union of their 64-row key-tile lists, each
    half masked to its own tiles (prefix tiles in every list), several passes, odd item count."""
    sel, q, k, v, ref = _vsa_case(h=3, n_pairs=3)
    sp = BS.stream_plan(BS.pair_union_plan(sel, 64), pb=pb)
    kmask, qind = BS.stream_half_masks(sp, sel)
    kT, vb = BS.stream_kv(sp, k, v)
    ops = dict(
        qT=BS.stream_q(sp, q), kT_blk=kT, v_blk=vb, lists=sp.lists, bounds=sp.bounds,
        kmask=kmask, qind=qind, masked=1, **BS.stream_kernel_kwargs(sp),
    )  # fmt: skip
    out = _sim_kernel(ops)[: sp.n_real_items * 128].reshape(q.shape)
    assert _rel(out, ref) < 1e-2


def _sla_super_case(h=3, nq=3, nk=13, seed=21):
    q, k, v = _qkv(h, nq * 128, nk * 128, seed=seed, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(h, nq, nk, generator=g) < 0.35
    mask[..., nk - 1] = True  # partial tail block (odd nk: its super-block has a padded half)
    valid = torch.arange(nk * 128) < nk * 128 - 45
    return q, k, v, plan_from_block_mask(mask, 128, 128, key_valid=valid)


def test_l4_superblock_plan_covers_selection_and_masks_drop_the_rest():
    q, k, v, fine = _sla_super_case()
    sup = BS.superblock_plan(fine)
    assert sup.k_block == 256 and sup.n_k_blocks == 7
    assert int(sup.counts.sum()) <= int(fine.counts.sum())
    sp = BS.stream_plan(sup, pb=sup.kp)  # one pass: plain softmax per item
    kmask, qind = BS.superblock_masks(sp, fine)
    lk = sup.n_k_blocks * 256
    kp_ = torch.nn.functional.pad
    kT, vb = BS.stream_kv(
        sp, kp_(k, (0, 0, 0, lk - k.shape[1])), kp_(v, (0, 0, 0, lk - v.shape[1])), sup.key_valid
    )
    qT = BS.stream_q(sp, q)
    out = torch.empty(sp.n_items * 128, D)
    for i in range(sp.n_items):  # one-pass-per-item emulation with the additive mask
        ids = sp.lists[i].long()
        kv_ = kT[ids].float()
        kk = kv_[:, :, :256].permute(0, 2, 1).reshape(-1, D)
        vv = kv_[:, :, 256:].reshape(-1, 128, 2, D).permute(0, 2, 1, 3).reshape(-1, D)
        s = qT[:, i * 128 : (i + 1) * 128].float().t() @ kk.t() + kmask[i].float()
        s = s.masked_fill(torch.arange(s.shape[1]) >= sp.bounds[i, 0, 0], float("-inf"))
        out[i * 128 : (i + 1) * 128] = torch.softmax(s, -1) @ vv
    assert sp.bounds.shape[2] == 1
    ref = reference_attention(q.float(), k.float(), v.float(), fine)
    assert _rel(out[: sp.n_real_items * 128].reshape(q.shape), ref) < 5e-3


@pytest.mark.skipif(os.environ.get("NKI_SIMULATOR") != "1", reason="NKI CPU simulator not enabled")
@pytest.mark.parametrize("pb", [2, 3])
def test_l4_superblock_kernel_on_nki_simulator(pb):
    """Turbo-SLA with 256-row super-blocks: one 1 KB-per-partition DMA per listed super-block, the
    unselected 128-row half masked, partial tail dropped by the bound, several passes."""
    q, k, v, fine = _sla_super_case()
    sup = BS.superblock_plan(fine)
    sp = BS.stream_plan(sup, pb=pb)
    kmask, qind = BS.superblock_masks(sp, fine)
    lk = sup.n_k_blocks * 256
    pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, lk - t.shape[1]))  # noqa: E731
    kT, vb = BS.stream_kv(sp, pad(k), pad(v), sup.key_valid)
    ops = dict(
        qT=BS.stream_q(sp, q), kT_blk=kT, v_blk=vb, lists=sp.lists, bounds=sp.bounds,
        kmask=kmask, qind=qind, masked=1, **BS.stream_kernel_kwargs(sp),
    )  # fmt: skip
    out = _sim_kernel(ops)[: sp.n_real_items * 128].reshape(q.shape)
    assert _rel(out, reference_attention(q.float(), k.float(), v.float(), fine)) < 1e-2
