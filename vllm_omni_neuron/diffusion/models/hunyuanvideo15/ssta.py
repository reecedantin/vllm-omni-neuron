# SPDX-License-Identifier: Apache-2.0
"""SSTA (selective sliding-tile attention) of HunyuanVideo-1.5 ``*_distilled_sparse`` checkpoints:
CPU reference of the block selection and of the attention it defines.

Mirrors upstream ``hyvideo/models/transformers/modules/ssta_attention.py`` (``ssta_3d_attention`` with
``sparse_type="ssta"``, ``sampling_type="importance"``) plus the caller's rules in
``hyvideo/models/transformers/modules/attention.py`` (``attn_mode == "flex-block-attn"``):

1. **Tiling.** The video tokens ``[t, h, w]`` are padded per axis to multiples of ``tile = (6, 8, 8)``
   (``pad_type="zero"``: zero q/k/v rows) and permuted into tile-major order: 384-token tiles in
   ``(n_t, n_h, n_w)`` raster order. The text tokens follow, padded to a multiple of 384 by repeating
   the last text token.
2. **Window (STA).** Video query tile ``i`` keeps every video key tile inside a ``3x3x3``-tile window
   centred on ``clip(i, k//2, n-1-k//2)`` per axis (the window slides inward at the borders, so it is
   always ``min(3, n)`` tiles per axis). Every query tile keeps every text tile and every text tile
   keeps every key tile.
3. **Importance top-k (MoBA).** Per head: tile means of q and k (padding zeros included), L2-normalised,
   ``sim = (q.k + 1) / 2``, ``red_j = nanmean_{i != j} (k_i.k_j + 1) / 2``,
   ``score = lambda * sim - (1 - lambda) * red``; each video query tile keeps its ``topk`` best video key
   tiles (``topk`` halves when ``t <= 31`` latent frames, e.g. 121 frames). Union with (2).
4. **Text mask.** With ``attn_use_text_mask``: text tiles past ``ceil(n_valid_text / 384)`` are dropped
   as keys and as queries, except each attends itself. Inside the kept text tiles, padded text tokens
   remain visible (block granularity, as upstream).
5. **Attention.** Full softmax attention restricted to the kept (query tile, key tile) pairs, scale
   ``1/sqrt(128)``; zero-padded video keys stay in the softmax (score 0), exactly as the kernel does.

Everything here is host/CPU torch; the device kernel consumes ``block_mask`` (or the equivalent
per-query-tile index lists from :func:`kept_tile_lists`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TILE = (6, 8, 8)
WINDOW = (3, 3, 3)


@dataclass(frozen=True)
class SSTALayout:
    thw: tuple[int, int, int]  # video latent grid (post-patch)
    padded: tuple[int, int, int]  # padded to multiples of the tile
    tile: tuple[int, int, int]
    grid: tuple[int, int, int]  # tiles per axis
    text_len: int
    text_tiles: int

    @property
    def tile_tokens(self) -> int:
        return self.tile[0] * self.tile[1] * self.tile[2]

    @property
    def video_tiles(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def n_tiles(self) -> int:
        return self.video_tiles + self.text_tiles

    @property
    def video_tokens(self) -> int:
        return self.thw[0] * self.thw[1] * self.thw[2]

    @property
    def padded_video_tokens(self) -> int:
        return self.video_tiles * self.tile_tokens


def layout(thw, text_len: int, tile=TILE) -> SSTALayout:
    t, h, w = thw
    if t == 1:  # upstream image mode: 1x16x24 tiles (384 tokens), no window
        tile = (1, 16, 24)
    padded = tuple(-(-n // s) * s for n, s in zip(thw, tile))
    grid = tuple(p // s for p, s in zip(padded, tile))
    bs = tile[0] * tile[1] * tile[2]
    return SSTALayout(tuple(thw), padded, tuple(tile), grid, text_len, math.ceil(text_len / bs))


def effective_topk(thw, topk: int) -> int:
    """Upstream halves top-k for clips of at most 31 latent frames (<= 121 frames at 24 fps)."""
    return topk // 2 if 1 < thw[0] <= 31 else topk


def effective_window(thw, window=WINDOW):
    return (1, 1, 1) if thw[0] == 1 else tuple(window)


# ------------------------------------------------------------------------------------------------
# token layout
# ------------------------------------------------------------------------------------------------
def pad_video(x: torch.Tensor, lay: SSTALayout) -> torch.Tensor:
    """``[B, H, t*h*w, D]`` -> ``[B, H, T*Hp*Wp, D]`` with zero rows at the far end of each axis."""
    b, nh, _, d = x.shape
    t, h, w = lay.thw
    T, Hp, Wp = lay.padded
    out = x.new_zeros(b, nh, T, Hp, Wp, d)
    out[:, :, :t, :h, :w] = x.reshape(b, nh, t, h, w, d)
    return out.reshape(b, nh, T * Hp * Wp, d)


def tile_order(lay: SSTALayout) -> torch.Tensor:
    """Index ``idx`` such that ``x_tiled = x_padded[..., idx, :]`` (tile-major order)."""
    T, Hp, Wp = lay.padded
    tt, th, tw = lay.tile
    nt, nh, nw = lay.grid
    pos = torch.arange(T * Hp * Wp).reshape(nt, tt, nh, th, nw, tw)
    return pos.permute(0, 2, 4, 1, 3, 5).reshape(-1)


def tile_video(x: torch.Tensor, lay: SSTALayout) -> torch.Tensor:
    return pad_video(x, lay)[:, :, tile_order(lay)]


def untile_video(x: torch.Tensor, lay: SSTALayout) -> torch.Tensor:
    b, nh, _, d = x.shape
    out = torch.empty_like(x)
    out[:, :, tile_order(lay)] = x
    t, h, w = lay.thw
    T, Hp, Wp = lay.padded
    return out.reshape(b, nh, T, Hp, Wp, d)[:, :, :t, :h, :w].reshape(b, nh, t * h * w, d)


def pad_text(x: torch.Tensor, lay: SSTALayout) -> torch.Tensor:
    n = lay.text_tiles * lay.tile_tokens - lay.text_len
    if n <= 0:
        return x
    return torch.cat([x, x[:, :, -1:].expand(-1, -1, n, -1)], dim=2)


# ------------------------------------------------------------------------------------------------
# masks
# ------------------------------------------------------------------------------------------------
def window_mask(lay: SSTALayout, window=WINDOW) -> torch.Tensor:
    """``[Nv, Nv]`` bool: video query tile -> video key tile inside the (border-clipped) window."""
    window = effective_window(lay.thw, window)
    g = torch.tensor(lay.grid)
    ids = torch.arange(lay.video_tiles)
    coord = torch.stack(
        [ids // (g[1] * g[2]), (ids % (g[1] * g[2])) // g[2], ids % g[2]], dim=-1
    )  # [N, 3]
    half = torch.tensor(window) // 2
    lo, hi = half, g - 1 - half
    # numpy clip semantics (min first, then max): a grid smaller than the window clips to hi
    centre = torch.minimum(torch.maximum(coord, lo), hi)
    return ((centre[:, None, :] - coord[None, :, :]).abs() <= half).all(dim=-1)


def tile_means(x_tiled: torch.Tensor, lay: SSTALayout) -> torch.Tensor:
    b, nh, _, d = x_tiled.shape
    return (
        x_tiled[:, :, : lay.padded_video_tokens]
        .reshape(b, nh, lay.video_tiles, lay.tile_tokens, d)
        .mean(dim=-2)
    )


def importance_topk(
    q_means: torch.Tensor, k_means: torch.Tensor, topk: int, lambda_: float
) -> torch.Tensor:
    """``[B, H, N, D]`` tile means -> ``[B, H, N, topk]`` selected key tiles (fp32 math, as upstream)."""
    q = q_means.float()
    k = k_means.float()
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    sim = (torch.einsum("bhsd,bhkd->bhsk", q, k) + 1.0) / 2.0
    uniq = (torch.einsum("bhsd,bhkd->bhsk", k, k) + 1.0) / 2.0
    n = k.shape[2]
    eye = torch.eye(n, dtype=torch.bool)
    red = uniq.masked_fill(eye, 0.0).sum(dim=-2, keepdim=True) / max(
        n - 1, 1
    )  # nanmean off-diagonal
    score = lambda_ * sim - (1 - lambda_) * red
    return score.topk(k=min(topk, n), dim=-1, sorted=False).indices


def ssta_block_mask(
    q_tiled,
    k_tiled,
    lay: SSTALayout,
    topk: int,
    lambda_: float,
    n_valid_text: int | None,
    window=WINDOW,
    share_within_head: bool = False,
) -> torch.Tensor:
    """``[B, H, N, N]`` bool block mask over ``[video tiles | text tiles]`` (H = 1 if shared)."""
    b, nh = q_tiled.shape[:2]
    nv, n = lay.video_tiles, lay.n_tiles
    qm, km = tile_means(q_tiled, lay).float(), tile_means(k_tiled, lay).float()
    if share_within_head:
        qm, km = qm.mean(dim=1, keepdim=True), km.mean(dim=1, keepdim=True)
        nh = 1
    sel = importance_topk(qm, km, effective_topk(lay.thw, topk), lambda_)
    mask = torch.zeros(b, nh, n, n, dtype=torch.bool)
    mask[:, :, :nv, :nv].scatter_(-1, sel, True)
    mask[:, :, :nv, :nv] |= window_mask(lay, window)
    if lay.text_tiles:
        mask[:, :, :, nv:] = True
        mask[:, :, nv:, :] = True
    if n_valid_text is not None and lay.text_tiles:
        keep = max(math.ceil(n_valid_text / lay.tile_tokens), 1)
        cut = nv + keep
        mask[:, :, cut:, :] = False
        mask[:, :, :, cut:] = False
        idx = torch.arange(cut, n)
        mask[:, :, idx, idx] = True
    return mask


def kept_tile_lists(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., N, N]`` -> (``[..., N, Kmax]`` int32 key-tile ids padded with -1, ``[..., N]`` counts):
    the per-query-tile index lists a block-sparse kernel walks with indirect DMA."""
    counts = mask.sum(dim=-1)
    kmax = int(counts.max())
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)[..., :kmax]
    ids = torch.where(torch.arange(kmax) < counts[..., None], order, torch.full_like(order, -1))
    return ids.to(torch.int32), counts


# ------------------------------------------------------------------------------------------------
# attention (dense masked reference)
# ------------------------------------------------------------------------------------------------
def ssta_attention(
    q,
    k,
    v,
    thw,
    text_len: int,
    topk: int = 64,
    lambda_: float = 0.7,
    n_valid_text: int | None = None,
    window=WINDOW,
    share_within_head: bool = False,
    return_mask: bool = False,
    tile=TILE,
):
    """``q/k/v`` ``[B, H, S_video + text_len, D]`` (video tokens raster order, then text) -> same shape.

    The reference materialises the token mask; use it on small shapes only."""
    lay = layout(thw, text_len, tile)
    sv = lay.video_tokens
    qt = torch.cat([tile_video(q[:, :, :sv], lay), pad_text(q[:, :, sv:], lay)], dim=2)
    kt = torch.cat([tile_video(k[:, :, :sv], lay), pad_text(k[:, :, sv:], lay)], dim=2)
    vt = torch.cat([tile_video(v[:, :, :sv], lay), pad_text(v[:, :, sv:], lay)], dim=2)
    bm = ssta_block_mask(qt, kt, lay, topk, lambda_, n_valid_text, window, share_within_head)
    bs = lay.tile_tokens
    tok = bm.repeat_interleave(bs, dim=-1).repeat_interleave(bs, dim=-2)
    scores = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~tok, float("-inf"))
    o = torch.matmul(torch.softmax(scores, dim=-1), vt.float()).to(q.dtype)
    pv = lay.padded_video_tokens
    out = torch.cat([untile_video(o[:, :, :pv], lay), o[:, :, pv : pv + text_len]], dim=2)
    return (out, bm) if return_mask else out


# ================================================================================================
# model path: the same SSTA inside the compiled DiT blocks (selection on the device, block-sparse
# kernel; fp32 list attention off the device)
# ================================================================================================
@dataclass(frozen=True)
class SSTAParams:
    """The checkpoint's ``attn_param`` (``attn_mode == "flex-block-attn"``, ``attn_sparse_type == "ssta"``)."""

    topk: int = 64
    lambda_: float = 0.7
    tile: tuple[int, int, int] = TILE
    window: tuple[int, int, int] = WINDOW
    use_text_mask: bool = True
    # baseline: the same tile layout, padding and text rule, but every query attends every (kept-text) key
    # tile through the dense flash kernel (``HV15_ATTN_MODE=dense_tiles``)
    dense: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> SSTAParams | None:
        p = cfg.get("attn_param") or {}
        if cfg.get("attn_mode") != "flex-block-attn" or p.get("attn_sparse_type", "ssta") != "ssta":
            return None
        unsupported = {
            "ssta_sampling_type": (p.get("ssta_sampling_type", "importance"), "importance"),
            "attn_pad_type": (p.get("attn_pad_type", "zero"), "zero"),
            "win_type": (p.get("win_type", "fixed"), "fixed"),
            "attn_mask_share_within_head": (int(p.get("attn_mask_share_within_head", 0)), 0),
            "ssta_adaptive_pool": (p.get("ssta_adaptive_pool"), None),
        }
        for k, (got, want) in unsupported.items():
            if got != want:
                raise NotImplementedError(
                    f"HunyuanVideo-1.5 SSTA on Neuron: {k}={got!r} (supported: {want!r})"
                )
        win = p.get("win_size", [list(WINDOW)])
        return cls(
            topk=int(p.get("ssta_topk", 64)),
            lambda_=float(p.get("ssta_lambda", 0.7)),
            tile=tuple(int(x) for x in p.get("tile_size", TILE)),
            window=tuple(int(x) for x in win[0]),
            use_text_mask=bool(p.get("attn_use_text_mask", 1)),
        )


@dataclass
class SSTAStatic:
    """Per (geometry, encoder length, CP size): the compile-time constants of the sparse blocks. Identical on
    every CP rank (rank-specific data is passed as tensors), so every rank compiles the same graphs."""

    lay: SSTALayout
    window: tuple[int, int, int]
    topk: int  # effective (halved for <= 31 latent frames)
    lambda_: float
    cp: int
    nq: int  # video query tiles per rank
    ntq: int  # text query tiles per rank (ceil(text tiles / CP))
    enc_len: int
    kmax: int  # list width of a video query tile: window + top-k (capped at the video tiles) + text tiles
    sp_video: object = None  # block_sparse.StreamPlan (static geometry; lists replaced per layer)
    sp_text: object = None
    dense: bool = False

    @property
    def nv(self) -> int:
        return self.lay.video_tiles

    @property
    def nt(self) -> int:
        return self.lay.text_tiles

    @property
    def nk(self) -> int:
        return self.lay.n_tiles


def build_static(
    params: SSTAParams, thw, enc_len: int, cp: int, heads: int, with_kernel: bool
) -> SSTAStatic:
    lay = layout(thw, enc_len, params.tile)
    win = effective_window(thw, params.window)
    if lay.video_tiles % cp:
        raise ValueError(
            f"SSTA with CP={cp}: {lay.video_tiles} video tiles must divide over the CP ranks"
        )
    topk = min(effective_topk(thw, params.topk), lay.video_tiles)
    n_win = math.prod(min(w, g) for w, g in zip(win, lay.grid))
    kmax = min(lay.video_tiles, n_win + topk) + lay.text_tiles
    st = SSTAStatic(
        lay,
        win,
        topk,
        params.lambda_,
        cp,
        lay.video_tiles // cp,
        -(-lay.text_tiles // cp),
        enc_len,
        kmax,
        dense=params.dense,
    )
    if with_kernel and not params.dense:
        from vllm_omni_neuron.diffusion.attention import block_sparse as BS

        bs = lay.tile_tokens

        def _sp(nq, kp):
            lists = torch.arange(kp, dtype=torch.int32).expand(heads, nq, kp).contiguous()
            plan = BS.plan_from_lists(
                lists, torch.full((heads, nq), kp, dtype=torch.int32), bs, bs, lay.n_tiles
            )
            return BS.stream_plan(plan)

        st.sp_video = _sp(st.nq, kmax)
        st.sp_text = _sp(st.ntq, lay.n_tiles)
    return st


def rank_inputs(st: SSTAStatic, rank: int) -> dict:
    """Host tensors of one CP rank: ``vidx`` (slot -> raster video token, pad slots -> 0), ``slot_valid``
    (0 on zero-pad slots), ``win`` (window rows of the rank's query tiles), ``tq_idx`` (the rank's text query
    rows in the padded text sequence) and ``inv`` (raster token -> slot in the GLOBAL tile-major order)."""
    lay, bs = st.lay, st.lay.tile_tokens
    t, h, w = lay.thw
    T, Hp, Wp = lay.padded
    pos = tile_order(lay)  # slot -> padded-raster index
    pt, rem = pos // (Hp * Wp), pos % (Hp * Wp)
    ph, pw = rem // Wp, rem % Wp
    real = (pt < t) & (ph < h) & (pw < w)
    raster = torch.where(real, pt * h * w + ph * w + pw, torch.zeros_like(pos))
    inv = torch.empty(t * h * w, dtype=torch.long)
    inv[raster[real]] = torch.nonzero(real).flatten()
    sl = slice(rank * st.nq * bs, (rank + 1) * st.nq * bs)
    tiles = torch.arange(st.ntq * rank, st.ntq * (rank + 1)).clamp(max=st.nt - 1)
    tq_idx = (tiles[:, None] * bs + torch.arange(bs)).reshape(-1)
    win = window_mask(lay, st.window)[rank * st.nq : (rank + 1) * st.nq]
    return dict(
        vidx=raster[sl].contiguous(),
        slot_valid=real[sl].contiguous(),
        win=win.contiguous(),
        tq_idx=tq_idx.contiguous(),
        inv=inv,
    )


def text_tile_keep(
    st: SSTAStatic, n_valid: torch.Tensor, use_text_mask: bool = True
) -> torch.Tensor:
    """``[B]`` valid encoder tokens -> ``[B, n_text_tiles]`` bool: text tiles kept as keys (upstream:
    ``ceil(n_valid / 384)`` tiles with ``attn_use_text_mask``, else all)."""
    nt, bs = st.nt, st.lay.tile_tokens
    if not use_text_mask:
        return torch.ones(n_valid.shape[0], nt, dtype=torch.bool)
    keep = torch.clamp(torch.ceil(n_valid.float() / bs), min=1).long()
    return torch.arange(nt)[None] < keep[:, None]


def select_lists(qm, km, win, tkeep, topk: int, lambda_: float, kmax: int):
    """Importance top-k + window + text tiles -> kernel lists, with static-shape tensor ops only (no
    ``topk`` / sort / scatter, so it stays inside the compiled block graph).

    ``qm`` ``[H, nQ, D]`` / ``km`` ``[H, Nv, D]`` tile means (zero pad rows included), ``win`` ``[nQ, Nv]``
    bool, ``tkeep`` ``[Nt]`` bool. Returns ``ids`` ``[H, nQ, kmax]`` (ascending key tiles over
    ``[video | text]``, ``-1`` pads) and ``counts`` ``[H, nQ]``. The k-th best score is found with ``topk``
    masked-max passes (ties are kept together; exact ties do not occur in practice)."""
    q = qm.float()
    k = km.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    nv = k.shape[1]
    sim = (q @ k.transpose(-1, -2) + 1.0) * 0.5
    uniq = (k @ k.transpose(-1, -2) + 1.0) * 0.5
    off = 1.0 - torch.eye(nv, dtype=uniq.dtype, device=uniq.device)
    red = (uniq * off).sum(dim=-2) / max(nv - 1, 1)  # [H, Nv]
    score = lambda_ * sim - (1.0 - lambda_) * red[:, None, :]
    if topk >= nv:
        sel = torch.ones_like(score, dtype=torch.bool)
    else:
        cur, thr = score, None
        for _ in range(topk):
            thr = cur.amax(dim=-1, keepdim=True)
            cur = torch.where(cur >= thr, torch.full_like(cur, float("-inf")), cur)
        sel = score >= thr
    h, nq = score.shape[:2]
    mask = torch.cat(
        [sel | win[None].to(torch.bool), tkeep.to(torch.bool).view(1, 1, -1).expand(h, nq, -1)], -1
    )
    return compact_lists(mask, kmax)


def compact_lists(mask: torch.Tensor, kmax: int):
    """``[H, nQ, Nk]`` bool -> ascending ids ``[H, nQ, kmax]`` (``-1`` pads) + counts, by a cumulative-sum
    matmul and a one-hot reduction (exact in fp32 for Nk < 2**24)."""
    nk = mask.shape[-1]
    dev = mask.device
    mf = mask.float()
    tri = torch.triu(torch.ones(nk, nk, device=dev))
    cum = mf @ tri  # entries up to and including j
    slot = torch.arange(1, kmax + 1, device=dev, dtype=torch.float32)
    hit = (cum[..., None, :] == slot[:, None]).float() * mf[..., None, :]  # [H, nQ, kmax, Nk]
    ids = hit @ torch.arange(nk, device=dev, dtype=torch.float32)
    counts = mf.sum(dim=-1)
    ids = torch.where(slot - 1 < counts[..., None], ids, torch.full_like(ids, -1.0))
    return ids.to(torch.int32), counts.to(torch.int32)


def list_attention(q, k, v, ids, counts, bs: int, scale: float):
    """fp32 reference of the kernel: ``q`` ``[H, nQ * bs, D]``, ``k``/``v`` ``[H, Nk * bs, D]``; each query tile
    attends the key tiles in its list (host / CPU path; one small attention per (head, query tile))."""
    h, lq, d = q.shape
    out = torch.empty(h, lq, d, dtype=torch.float32)
    kb = k.float().reshape(h, -1, bs, d)
    vb = v.float().reshape(h, -1, bs, d)
    for hi in range(h):
        for qi in range(lq // bs):
            sel = ids[hi, qi, : int(counts[hi, qi])].long()
            kk = kb[hi, sel].reshape(-1, d)
            vv = vb[hi, sel].reshape(-1, d)
            s = (q[hi, qi * bs : (qi + 1) * bs].float() @ kk.t()) * scale
            out[hi, qi * bs : (qi + 1) * bs] = torch.softmax(s, dim=-1) @ vv
    return out.to(q.dtype)


def _pad_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    """``[H, L, D]`` -> ``[H, n, D]`` repeating the last row (upstream's text padding)."""
    if n <= x.shape[1]:
        return x
    return torch.cat([x, x[:, -1:].expand(-1, n - x.shape[1], -1)], dim=1)


def _prefix_attention(q, k, v, n_keys, scale, qblock: int = 2048):
    """fp32 attention over the first ``n_keys`` keys (host path of the dense-tiles baseline)."""
    n = int(n_keys)
    kk, vv = k[:, :n].float(), v[:, :n].float()
    out = torch.cat(
        [
            torch.softmax((q[:, s : s + qblock].float() @ kk.transpose(1, 2)) * scale, -1) @ vv
            for s in range(0, q.shape[1], qblock)
        ],
        dim=1,
    )
    return out.to(q.dtype)


def _bounded_dense(q, k, v, n_keys, scale):
    """Dense flash attention (nkilib ``attention_cte``) over the first ``n_keys`` keys (``bound_max``)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.diffusion.attention import block_sparse as BS

    h, lq, d = q.shape
    qs = (q.float() * scale).to(torch.bfloat16).contiguous()
    bmax = n_keys.to(torch.int32).view(1, 1, 1).expand(h, lq, 1).contiguous()
    bmin = torch.zeros_like(bmax)
    out = wrap_nki(BS._dense_bounded_kernel)[BS.LNC](
        q=qs,
        k=k.to(torch.bfloat16).contiguous(),
        v=v.to(torch.bfloat16).contiguous(),
        bound_min=bmin,
        bound_max=bmax,
    )
    return out.reshape(h, lq, d)


def ssta_block_attention(
    q, k, v, eq, ek, ev, st: SSTAStatic, slot_valid, win, tkeep, tq_idx, cp=None
):
    """SSTA of one dual-stream block on one rank.

    ``q/k/v`` ``[B, L, nh, D]``: this rank's video slots (tile-major, zero-pad slots included) after RoPE;
    ``eq/ek/ev`` ``[B, Ne, nh, D]``: the encoder tokens (replicated). ``slot_valid`` ``[L]``, ``win``
    ``[nQ, Nv]``, ``tkeep`` ``[B, Nt]``, ``tq_idx`` ``[ntq * 384]``; ``cp``: the CP GroupCoordinator (video
    K/V all-gathered, text query tiles split over the ranks and their outputs all-gathered).
    Returns ``(video out [B, L, nh, D], encoder out [B, Ne, nh, D])``."""
    from vllm_omni_neuron.diffusion.attention import block_sparse as BS

    b, _, nh, d = q.shape
    bs, ne = st.lay.tile_tokens, eq.shape[1]
    scale = d**-0.5
    m = slot_valid.to(q.dtype).view(1, -1, 1, 1)
    q, k, v = (
        q * m,
        k * m,
        v * m,
    )  # upstream pad_type="zero": pad slots are zero rows inside attention
    if cp is not None:
        from .transformer import group_all_gather

        k = group_all_gather(cp, k, dim=1)
        v = group_all_gather(cp, v, dim=1)
    nt_rows = st.nt * bs
    kernel = BS._kernel_ok(q)
    ov, oe = [], []
    for bi in range(b):
        qh = q[bi].transpose(0, 1)  # [nh, L, D]
        kv_ = k[bi].transpose(0, 1)
        vv_ = v[bi].transpose(0, 1)
        eqh = _pad_rows(eq[bi].transpose(0, 1), nt_rows)
        ekh = _pad_rows(ek[bi].transpose(0, 1), nt_rows)
        evh = _pad_rows(ev[bi].transpose(0, 1), nt_rows)
        k_all = torch.cat([kv_, ekh], dim=1).contiguous()
        v_all = torch.cat([vv_, evh], dim=1).contiguous()
        tq = eqh[:, tq_idx]
        # text query tiles: every video tile + the kept text tiles (a valid prefix of the key tiles)
        tvalid = torch.cat(
            [torch.ones(st.nv, dtype=torch.bool, device=q.device), tkeep[bi].to(torch.bool)]
        )
        if st.dense:
            n_keys = tvalid.sum() * bs
            if kernel:
                o = _bounded_dense(qh, k_all, v_all, n_keys, scale)
                ot = _bounded_dense(tq, k_all, v_all, n_keys, scale)
            else:
                o = _prefix_attention(qh, k_all, v_all, n_keys, scale)
                ot = _prefix_attention(tq, k_all, v_all, n_keys, scale)
        else:
            qm = qh.reshape(nh, st.nq, bs, d).float().mean(dim=2)
            km = kv_.reshape(nh, st.nv, bs, d).float().mean(dim=2)
            ids, counts = select_lists(qm, km, win, tkeep[bi], st.topk, st.lambda_, st.kmax)
            t_ids = torch.where(
                tvalid,
                torch.arange(st.nk, device=q.device),
                torch.full((st.nk,), -1, device=q.device),
            )
            t_ids = t_ids.view(1, 1, -1).expand(nh, st.ntq, -1).to(torch.int32)
            t_counts = tvalid.sum().to(torch.int32).view(1, 1).expand(nh, st.ntq)
            if kernel:
                sp = st.sp_video
                lists, bounds = BS.stream_lists(ids, counts, bs, st.nk, sp.pb)
                o = BS.attend_stream(qh, k_all, v_all, sp.with_lists(lists, bounds), scale=scale)
                spt = st.sp_text
                tl, tb = BS.stream_lists(t_ids, t_counts, bs, st.nk, spt.pb)
                ot = BS.attend_stream(tq, k_all, v_all, spt.with_lists(tl, tb), scale=scale)
            else:
                o = list_attention(qh, k_all, v_all, ids, counts, bs, scale)
                ot = list_attention(tq, k_all, v_all, t_ids, t_counts, bs, scale)
        if cp is not None:
            ot = group_all_gather(cp, ot, dim=1)
        ov.append(o.transpose(0, 1))
        oe.append(ot[:, :ne].transpose(0, 1))
    return torch.stack(ov).to(q.dtype), torch.stack(oe).to(q.dtype)


__all__ = [
    "SSTALayout",
    "SSTAParams",
    "SSTAStatic",
    "build_static",
    "compact_lists",
    "effective_topk",
    "importance_topk",
    "kept_tile_lists",
    "layout",
    "list_attention",
    "rank_inputs",
    "select_lists",
    "ssta_attention",
    "ssta_block_attention",
    "ssta_block_mask",
    "text_tile_keep",
    "tile_order",
    "tile_video",
    "untile_video",
    "window_mask",
]
