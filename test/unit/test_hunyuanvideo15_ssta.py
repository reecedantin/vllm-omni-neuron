# SPDX-License-Identifier: Apache-2.0
"""SSTA CPU reference (``hunyuanvideo15/ssta.py``) vs upstream HunyuanVideo-1.5's own
``hyvideo/models/transformers/modules/ssta_attention.py``.

The upstream module is loaded from its source file (``HV15_UPSTREAM_SSTA``: the path of
``ssta_attention.py`` in a Tencent HunyuanVideo-1.5 checkout)
with ``flex_block_attn_func`` replaced by its documented semantics: dense softmax attention restricted to
the kept (query block, key block) pairs. Skipped when the upstream source is not present.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch

from vllm_omni_neuron.diffusion.models.hunyuanvideo15 import ssta

UPSTREAM = os.environ.get("HV15_UPSTREAM_SSTA", "")


def _flex_block_attn_func(q, k, v, q_bs, kv_bs, block_mask):
    if block_mask.dim() == 2:
        block_mask = block_mask[None, None]
    if block_mask.dim() == 5:  # upstream's shared-mask path adds a second singleton head dim
        block_mask = block_mask.squeeze(2)
    tok = block_mask.repeat_interleave(q_bs, dim=-2).repeat_interleave(kv_bs, dim=-1)
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    s = s.masked_fill(~tok, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM or not os.path.exists(UPSTREAM):
        pytest.skip(f"upstream SSTA source not found: {UPSTREAM}")
    fake = types.ModuleType("flex_block_attn")
    fake.flex_block_attn_func = _flex_block_attn_func
    sys.modules.setdefault("flex_block_attn", fake)
    spec = importlib.util.spec_from_file_location("upstream_ssta", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _qkv(thw, text_len, heads=2, d=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = thw[0] * thw[1] * thw[2] + text_len
    # smooth-in-space features so tile means carry structure (random iid tokens make every tile mean ~0)
    base = torch.randn(3, 1, heads, s, d, generator=g)
    drift = torch.randn(3, 1, heads, 1, d, generator=g) * 2
    return [b + dr * torch.linspace(-1, 1, s)[None, None, :, None] for b, dr in zip(base, drift)]


CASES = [
    # thw, text_len, n_valid_text, topk     (padding on every axis; t <= 31 -> top-k halves)
    ((7, 10, 13), 300, 200, 4),
    ((13, 17, 20), 450, 400, 8),
    ((37, 8, 8), 384, 384, 6),  # t > 31: no halving, text exactly one tile
    ((13, 17, 20), 900, 100, 8),  # text tiles past the valid ones are dropped
]


@pytest.mark.parametrize("thw,text_len,n_valid,topk", CASES)
@pytest.mark.parametrize("share", [False, True])
def test_mask_and_attention_match_upstream(upstream, thw, text_len, n_valid, topk, share):
    q, k, v = _qkv(thw, text_len)
    text_mask = torch.zeros(1, text_len, dtype=torch.int64)
    text_mask[0, :n_valid] = 1

    ref_out, _ = upstream.ssta_3d_attention(
        q,
        k,
        v,
        thw,
        topk=topk,
        tile_thw=ssta.TILE,
        kernel_thw=list(ssta.WINDOW),
        text_len=text_len,
        sparse_type="ssta",
        threshold=0.0,
        lambda_=0.7,
        pad_type="zero",
        text_mask=text_mask,
        mask_share_within_head=share,
        sampling_type="importance",
        adaptive_pool=None,
    )

    # upstream's caller (attention.py, flex-block-attn) halves top-k for <= 31 latent frames
    eff = ssta.effective_topk(thw, topk)
    out, mask = ssta.ssta_attention(
        q,
        k,
        v,
        thw,
        text_len,
        topk=topk,
        lambda_=0.7,
        n_valid_text=n_valid,
        share_within_head=share,
        return_mask=True,
    )

    # the block mask itself, built by upstream's helpers on the same tiled tensors
    lay = ssta.layout(thw, text_len)
    sv = lay.video_tokens
    qt = ssta.tile_video(q[:, :, :sv], lay)
    kt = ssta.tile_video(k[:, :, :sv], lay)
    up_mask = upstream.create_ssta_3d_mask(
        qt,
        kt,
        canvas_thw=lay.padded,
        topk=eff,
        tile_thw=ssta.TILE,
        kernel_thw=list(ssta.WINDOW),
        text_block_num=lay.text_tiles,
        threshold=0.0,
        lambda_=0.7,
        text_mask=text_mask[0],
        mask_share_within_head=share,
        sampling_type="importance",
    )
    assert torch.equal(mask[0], up_mask), (mask[0] ^ up_mask).sum()

    # tiling order vs upstream's einops tile()
    pq = ssta.pad_video(q[:, :, :sv], lay)
    assert torch.equal(qt, upstream.tile(pq, lay.padded, ssta.TILE))

    if eff == topk:  # upstream's ssta_3d_attention does not halve on its own; compare only then
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("thw,text_len,n_valid,topk", CASES[:2])
def test_attention_matches_upstream_with_halved_topk(upstream, thw, text_len, n_valid, topk):
    """End-to-end attention with the caller's top-k rule applied on the upstream side."""
    q, k, v = _qkv(thw, text_len, seed=1)
    text_mask = torch.zeros(1, text_len, dtype=torch.int64)
    text_mask[0, :n_valid] = 1
    eff = ssta.effective_topk(thw, topk)
    ref_out, ratio = upstream.ssta_3d_attention(
        q,
        k,
        v,
        thw,
        topk=eff,
        tile_thw=ssta.TILE,
        kernel_thw=list(ssta.WINDOW),
        text_len=text_len,
        sparse_type="ssta",
        threshold=0.0,
        lambda_=0.7,
        pad_type="zero",
        text_mask=text_mask,
        mask_share_within_head=False,
        sampling_type="importance",
        adaptive_pool=None,
    )
    out, mask = ssta.ssta_attention(
        q, k, v, thw, text_len, topk=topk, lambda_=0.7, n_valid_text=n_valid, return_mask=True
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
    assert abs(mask.float().mean().item() - ratio) < 1e-9


def test_window_and_index_lists():
    lay = ssta.layout((31, 45, 80), 1985)  # 720p 121 frames, I2V encoder length
    assert lay.padded == (36, 48, 80) and lay.grid == (6, 6, 10) and lay.text_tiles == 6
    w = ssta.window_mask(lay)
    assert int(w.sum(dim=-1).min()) == 27 and int(w.sum(dim=-1).max()) == 27  # window slides inward
    m = torch.zeros(1, 1, lay.n_tiles, lay.n_tiles, dtype=torch.bool)
    m[..., : lay.video_tiles, : lay.video_tiles] = w
    ids, counts = ssta.kept_tile_lists(m)
    assert torch.equal(counts[0, 0, : lay.video_tiles], torch.full((lay.video_tiles,), 27))
    row = ids[0, 0, 0]
    assert torch.equal(row[:27].long(), torch.nonzero(w[0]).flatten())


# ------------------------------------------------------------------------------------------------
# model path (ssta_block_attention): the in-graph selection + list attention vs the reference above
# ------------------------------------------------------------------------------------------------
class _FakeCP:
    """Single-process stand-in for a CP GroupCoordinator: serves the precomputed gathered video K/V and
    records the rank's own text-query output (the text gather is checked by reassembly)."""

    def __init__(self, k_full, v_full, cp):
        self.queue, self.cp, self.text = [k_full, v_full], cp, None
        self.ranks, self.world_size = list(range(cp)), cp  # ascending: no reorder

    def all_gather(self, x, dim):
        if self.queue:
            return self.queue.pop(0)
        self.text = x
        return torch.cat([x] * self.cp, dim=dim)


BLOCK_CASES = [
    # thw, tile, text_len, n_valid, topk, cp   (60 tiles of 8 tokens: window 27 + top-k is sparse)
    ((5, 7, 9), (2, 2, 2), 30, 13, 8, 1),
    ((5, 7, 9), (2, 2, 2), 30, 13, 8, 2),
    ((5, 7, 9), (2, 2, 2), 30, 30, 8, 4),  # all text valid; 4 text tiles over 4 ranks
    ((37, 6, 6), (6, 2, 2), 17, 5, 6, 3),  # t > 31 (no halving); 7x3x3 tiles over 3 ranks
]


@pytest.mark.parametrize("thw,tile,text_len,n_valid,topk,cp", BLOCK_CASES)
def test_block_attention_matches_reference(thw, tile, text_len, n_valid, topk, cp):
    torch.manual_seed(0)
    heads, d = 2, 16
    q, k, v = _qkv(thw, text_len, heads=heads, d=d, seed=3)
    ref = ssta.ssta_attention(
        q, k, v, thw, text_len, topk=topk, lambda_=0.7, n_valid_text=n_valid, tile=tile
    )
    params = ssta.SSTAParams(topk=topk, lambda_=0.7, tile=tile)
    st = ssta.build_static(params, thw, text_len, cp, heads, with_kernel=False)
    sv = st.lay.video_tokens
    tkeep = ssta.text_tile_keep(st, torch.tensor([n_valid]))
    to_bshd = lambda x: x.transpose(1, 2)  # noqa: E731  [B, H, S, D] -> [B, S, H, D]
    enc = [to_bshd(x[:, :, sv:]) for x in (q, k, v)]
    ris = [ssta.rank_inputs(st, r) for r in range(cp)]

    def slots(x, ri):
        y = to_bshd(x[:, :, :sv])[:, ri["vidx"]]
        return y * ri["slot_valid"].to(y.dtype).view(1, -1, 1, 1)

    k_full = torch.cat([slots(k, ri) for ri in ris], dim=1)
    v_full = torch.cat([slots(v, ri) for ri in ris], dim=1)
    outs, texts = [], []
    for ri in ris:
        fake = _FakeCP(k_full, v_full, cp) if cp > 1 else None
        ov, oe = ssta.ssta_block_attention(
            slots(q, ri),
            slots(k, ri),
            slots(v, ri),
            *enc,
            st,
            ri["slot_valid"],
            ri["win"],
            tkeep,
            ri["tq_idx"],
            cp=fake,
        )
        outs.append(ov)
        texts.append(fake.text.transpose(0, 1)[None] if cp > 1 else oe)  # [1, rows, H, D]
    video = torch.cat(outs, dim=1)[:, ris[0]["inv"]].transpose(1, 2)  # raster [B, H, Sv, D]
    torch.testing.assert_close(video, ref[:, :, :sv], rtol=1e-5, atol=1e-5)
    text = torch.cat(texts, dim=1)[:, :text_len].transpose(1, 2)
    kept = min(
        int(tkeep.sum()) * st.lay.tile_tokens, text_len
    )  # rows of dropped text tiles are never read
    torch.testing.assert_close(text[:, :, :kept], ref[:, :, sv : sv + kept], rtol=1e-5, atol=1e-5)


def test_select_lists_matches_block_mask():
    """The static-shape selection (masked-max top-k + cumsum compaction) gives upstream's block mask."""
    thw, tile, text_len, n_valid = (5, 7, 9), (2, 2, 2), 30, 13
    q, k, _ = _qkv(thw, text_len, heads=2, d=16, seed=5)
    lay = ssta.layout(thw, text_len, tile)
    sv = lay.video_tokens
    qt = torch.cat([ssta.tile_video(q[:, :, :sv], lay), ssta.pad_text(q[:, :, sv:], lay)], dim=2)
    kt = torch.cat([ssta.tile_video(k[:, :, :sv], lay), ssta.pad_text(k[:, :, sv:], lay)], dim=2)
    want = ssta.ssta_block_mask(qt, kt, lay, 8, 0.7, n_valid)[0, :, : lay.video_tiles]  # [H, Nv, N]
    st = ssta.build_static(
        ssta.SSTAParams(topk=8, tile=tile), thw, text_len, 1, 2, with_kernel=False
    )
    qm, km = ssta.tile_means(qt, lay)[0], ssta.tile_means(kt, lay)[0]
    ids, counts = ssta.select_lists(
        qm,
        km,
        ssta.window_mask(lay),
        ssta.text_tile_keep(st, torch.tensor([n_valid]))[0],
        st.topk,
        0.7,
        st.kmax,
    )
    got = torch.zeros_like(want)
    for h in range(2):
        for i in range(lay.video_tiles):
            row = ids[h, i, : counts[h, i]].long()
            assert torch.all(row[1:] > row[:-1]) and torch.all(ids[h, i, counts[h, i] :] == -1)
            got[h, i, row] = True
    assert torch.equal(got, want)
    assert int(counts.max()) <= st.kmax < lay.n_tiles


@pytest.mark.parametrize("cp", [1, 2])
def test_dense_tiles_baseline_is_all_tiles_kept(cp):
    """``dense=True`` (the dense-kernel baseline) equals SSTA with every video tile selected."""
    thw, tile, text_len, n_valid = (5, 7, 9), (2, 2, 2), 30, 13
    heads, d = 2, 16
    q, k, v = _qkv(thw, text_len, heads=heads, d=d, seed=7)
    ref = ssta.ssta_attention(q, k, v, thw, text_len, topk=10_000, n_valid_text=n_valid, tile=tile)
    st = ssta.build_static(
        ssta.SSTAParams(tile=tile, dense=True), thw, text_len, cp, heads, with_kernel=False
    )
    sv = st.lay.video_tokens
    tkeep = ssta.text_tile_keep(st, torch.tensor([n_valid]))
    enc = [x[:, :, sv:].transpose(1, 2) for x in (q, k, v)]
    ris = [ssta.rank_inputs(st, r) for r in range(cp)]

    def slots(x, ri):
        y = x[:, :, :sv].transpose(1, 2)[:, ri["vidx"]]
        return y * ri["slot_valid"].to(y.dtype).view(1, -1, 1, 1)

    k_full = torch.cat([slots(k, ri) for ri in ris], dim=1)
    v_full = torch.cat([slots(v, ri) for ri in ris], dim=1)
    outs = []
    for ri in ris:
        fake = _FakeCP(k_full, v_full, cp) if cp > 1 else None
        ov, _ = ssta.ssta_block_attention(
            slots(q, ri),
            slots(k, ri),
            slots(v, ri),
            *enc,
            st,
            ri["slot_valid"],
            ri["win"],
            tkeep,
            ri["tq_idx"],
            cp=fake,
        )
        outs.append(ov)
    video = torch.cat(outs, dim=1)[:, ris[0]["inv"]].transpose(1, 2)
    torch.testing.assert_close(video, ref[:, :, :sv], rtol=1e-5, atol=1e-5)
