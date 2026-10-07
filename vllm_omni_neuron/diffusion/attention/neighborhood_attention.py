# SPDX-License-Identifier: Apache-2.0
"""Device-legal neighborhood (windowed / NATTEN-style) attention for Swin/NATTEN VAEs.

The FLUX-3-Action video VAE (and any Swin3D / NATTEN VAE, e.g. Wan's) attends over a fixed
``kernel``-sized window around every token. The three obvious formulations all fail ``neuronx-cc``
on NeuronCore-v3 (found while porting the FLUX-3-Action video VAE):

* ``index_select`` gather of each query's window -> ``NCC_EBIR026`` (Vector-DGE index must start at
  partition 0);
* dense ``masked_fill(~band, -inf)`` -> ``NCC_IMPR902`` (mask-propagation internal error);
* dense additive band bias in a query-block loop -> same ``NCC_IMPR902``.

This module avoids all three with **halo tiling**: partition the grid into fixed ``tile``-sized
blocks of queries; each tile attends to a fixed ``tile + 2*(kernel-1)`` key window around it
(halo = ``kernel - 1`` per side so NATTEN's inward border clamp is always covered), zero-padded at
the grid edges. Every tile is therefore an identical small DENSE attention -- one static shape, pure
``matmul`` + ``softmax``, no gather and no ``[N, N]`` mask. The only mask is a small per-tile additive
bias (``[tile_tokens, window_tokens]``), a host-side constant shared by every tile, applied as a
plain fp32 add -- the same trick that made the FLUX-3-Action banded path compile, now with a bounded window
instead of all N keys.

Semantics match ``flux3_action.neighborhood.neighborhood_attention_reference`` exactly (bit-exact on
CPU for the shapes tested): NATTEN stride-1 dilation-1 odd-kernel windows, non-causal axes clamp the
window inward at the borders, causal axes use ``[max(0, i-k+1), i]``; softmax scale ``head_dim**-0.5``;
layout ``[B, *axes, heads, head_dim]`` (heads-last).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

# Cache of per-(axes, kernel, causal, tile) additive bias tensors -- host constants, reused across
# calls and across every tile of a layout (keyed by shape, not by the actual q/k/v).
_BIAS_CACHE: dict = {}


def _axis_window_start(i: Tensor, length: int, kernel: int, causal: bool) -> Tensor:
    """NATTEN per-axis window start for query index ``i`` (same definition as the reference)."""
    if causal:
        return (i - (kernel - 1)).clamp(min=0)
    return (i - kernel // 2).clamp(0, length - kernel)


def _tile_bias(
    axes: tuple[int, ...],
    kernel: tuple[int, ...],
    causal: tuple[bool, ...],
    tile: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """``[n_tiles_total, prod(tile), prod(window)]`` additive bias (0 inside the window, large
    negative outside / off-grid), built on the host. One row block per tile position; the per-axis
    window validity is an outer product, so this is assembled vectorised, not with a Python
    quadruple loop."""
    key = (axes, kernel, causal, tile, str(device), dtype)
    hit = _BIAS_CACHE.get(key)
    if hit is not None:
        return hit

    n_ax = len(axes)
    halo = tuple(k - 1 for k in kernel)
    span = tuple(tile[a] + 2 * halo[a] for a in range(n_ax))
    n_tiles = tuple(math.ceil(axes[a] / tile[a]) for a in range(n_ax))

    # Per-axis validity: [n_tiles_a, tile_a, span_a] bool. query global index g = tile*ti + ql;
    # key global index kg = tile*ti - halo + kp. Valid iff kg in [0,length) and in the query's window.
    per_axis = []
    for a in range(n_ax):
        ti = torch.arange(n_tiles[a])[:, None, None]
        ql = torch.arange(tile[a])[None, :, None]
        kp = torch.arange(span[a])[None, None, :]
        qg = tile[a] * ti + ql  # [n_tiles_a, tile_a, 1]
        kg = tile[a] * ti - halo[a] + kp  # [n_tiles_a, 1, span_a] broadcast
        # Padded queries (qg >= axes[a], cropped from the result) are NOT masked out: they keep the
        # clamped window of the last real row, so no bias row is ever all -30000 -- an all-masked row
        # is a 0/0 hazard for a device softmax lowering (the r16 crop=False variant on Trn2 returned
        # NaN into the VALID region once such rows were kept in the graph), and a uniform-weight row
        # on CPU was only ever a harmless accident of the finite -30000 choice.
        start = _axis_window_start(qg, axes[a], kernel[a], causal[a])
        on_grid_k = (kg >= 0) & (kg < axes[a])
        if causal[a]:
            # causal window is (qg - kernel, qg], i.e. kg in [start, qg] -- NOT [start, start+kernel):
            # when start is clamped up from qg-kernel+1 to 0 (qg < kernel-1), the window is shorter
            # than `kernel` (query 0 sees only itself), so the upper bound is qg, not start+kernel.
            in_window = (kg >= start) & (kg <= qg)
        else:
            in_window = (kg >= start) & (kg < start + kernel[a])
        per_axis.append(on_grid_k & in_window)  # [n_tiles_a, tile_a, span_a] bool

    # Outer-product the per-axis validity over all tiles / query-positions / key-positions.
    # valid[ti_0..ti_{n-1}, ql_0.., kp_0..] = AND_a per_axis[a][ti_a, ql_a, kp_a].
    valid = None
    for a in range(n_ax):
        v = per_axis[a]  # [nt_a, t_a, s_a]
        # reshape to broadcast across the full (tiles, tile-positions, window-positions) grid
        shape = [1] * (3 * n_ax)
        shape[a] = n_tiles[a]
        shape[n_ax + a] = tile[a]
        shape[2 * n_ax + a] = span[a]
        vb = v.reshape(
            *[n_tiles[a] if i == a else 1 for i in range(n_ax)],
            *[tile[a] if i == a else 1 for i in range(n_ax)],
            *[span[a] if i == a else 1 for i in range(n_ax)],
        )
        valid = vb if valid is None else (valid & vb)

    tiles_total = 1
    for nt in n_tiles:
        tiles_total *= nt
    tile_tokens = 1
    for t in tile:
        tile_tokens *= t
    win_tokens = 1
    for sp in span:
        win_tokens *= sp
    valid = valid.reshape(tiles_total, tile_tokens, win_tokens)
    # A causal axis can still leave a padded query with no on-grid key in its window; give such a
    # row (cropped anyway) the whole window rather than an all-masked softmax.
    empty = ~valid.any(dim=-1, keepdim=True)
    valid = valid | empty
    # A moderate, conventional additive-mask magnitude (the FLUX-3-Action VAE's own working _band_bias uses the same
    # -30000.0, not an extreme near-overflow finfo.min value): large enough that softmax zeroes a
    # masked key's weight at any realistic score scale, small enough that fp32 arithmetic around it
    # (the scale multiply, the matmul accumulation, the compiler's own softmax reduction) stays well
    # inside normal range on every backend. An all-masked row (the padding tile rows cropped at the
    # end) then softmaxes to a harmless uniform distribution instead of NaN, on CPU and on device.
    neg = -30000.0
    bias = torch.where(valid, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
    # Stored as [tiles_total, 1, tile_tokens, win_tokens]: the heads axis is broadcast on the HOST
    # layout, so a compiled region adds the bias to [tiles, heads, tt, wt] scores with no in-graph
    # unsqueeze (one less reshape of a graph input for the device compiler to get wrong).
    bias = bias.unsqueeze(1).to(device)
    _BIAS_CACHE[key] = bias
    return bias


def _window_axis(x: Tensor, dim: int, size: int, step: int, n_windows: int) -> Tensor:
    """``Tensor.unfold(dim, size, step)``, built from ``narrow`` + ``stack`` (no native XLA lowering
    for ``unfold`` on this backend; see :func:`neighborhood_attention_tiled`'s comment). Replaces
    axis ``dim`` (length >= ``(n_windows-1)*step + size``) with ``n_windows`` and appends a new
    trailing dim of length ``size``: same output layout as ``unfold``.

    Kept for callers/tests that want ``unfold``'s layout. The hot path uses :func:`_window_axis_split`
    instead, whose in-place ``(n_windows, size)`` layout needs no ``movedim`` afterwards.
    """
    return _window_axis_split(x, dim, size, step, n_windows).movedim(dim + 1, -1).contiguous()


def _window_axis_split(x: Tensor, dim: int, size: int, step: int, n_windows: int) -> Tensor:
    """Replace axis ``dim`` with the adjacent pair ``(n_windows, size)`` -- windows of ``size`` at
    stride ``step``. Non-overlapping (``size == step``, the query tiles) is a pure ``reshape``;
    overlapping (the halo'd key/value windows) is ``n_windows`` contiguous ``narrow`` slices stacked
    along ``dim``. No ``unfold`` (no native lowering here, silently wrong numbers) and no ``movedim``/
    multi-axis ``permute``: on this backend only adjacent-axis transposes are trusted (the three Wan
    VAE ``NCC_IDDT901`` fixes were all non-adjacent permutes; the compiled-but-wrong rel_err 0.43 of
    round 7's neighborhood check is the same class of op, so the whole hot path avoids them).

    This is the ``window_impl="slice"`` formulation. Smoke round 17 on Trn2 found that, correct as
    it is on its own (the windowing stage alone matched CPU in r13-r15), it is wrong once FUSED with
    the attention in one bf16 graph (see :func:`_window_axis_select`), so the hot path defaults to
    the selection-matmul formulation and keeps this one as an option / CPU reference.
    """
    if size == step and x.shape[dim] == n_windows * size:
        shape = list(x.shape)
        return x.reshape(*shape[:dim], n_windows, size, *shape[dim + 1 :])
    windows = [x.narrow(dim, i * step, size).contiguous() for i in range(n_windows)]
    return torch.stack(windows, dim=dim)


def _window_select_matrix(
    length: int, size: int, step: int, n_windows: int, dtype: torch.dtype, device
) -> Tensor:
    """``[n_windows * size, length]`` 0/1 matrix with ``S[n*size + s, n*step + s] = 1``: left-multiplying
    an axis of length ``length`` by it produces the ``n_windows`` overlapping windows of ``size`` at
    stride ``step``, concatenated. Exactly one nonzero per row, so the matmul is EXACT in any dtype
    (each output element is one input element times 1.0, accumulated with zeros).

    Built FUNCTIONALLY (``arange`` + broadcast add + compare + cast; no ``zeros`` + indexed write, no
    integer div/mod) and with NO module-level cache: smoke round 18 on Trn2 found a global dict cache
    consulted inside the traced op makes Dynamo guard on ``key in cache`` -- the first call traces the
    matrix CONSTRUCTION into the graph (an in-graph scatter, 74 s compile) and the second call fails
    the guard and recompiles (``warm_s`` 58-75 s = a full recompile per call). Inside a compiled
    region this version is a handful of static integer ops the compiler folds or runs in microseconds;
    callers who want the matrices as plain graph inputs instead build them once on the host with
    :func:`neighborhood_select_matrices` and pass ``select=``.
    """
    cols = (
        torch.arange(n_windows, device=device)[:, None] * step
        + torch.arange(size, device=device)[None, :]
    ).reshape(-1)  # [n_windows*size]: source index of each window position
    return (torch.arange(length, device=device)[None, :] == cols[:, None]).to(dtype)


def neighborhood_select_matrices(
    axes: Sequence[int],
    kernel: Sequence[int],
    tile: Sequence[int] | None,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, ...]:
    """The per-axis key/value window-selection matrices :func:`neighborhood_attention_tiled` uses
    for one layout (``window_impl="select"``), built on the host in the inputs' ``dtype``: pass them
    as ``select=`` so they enter a compiled region as plain contiguous graph inputs (the same pattern
    as ``bias=``) instead of being constructed inside the trace. ``tile=None`` = :func:`pad_free_tile`."""
    n_ax = len(axes)
    kernel = tuple(kernel)
    tile = tuple(pad_free_tile(axes, kernel)) if tile is None else tuple(tile)
    tile = tuple(min(tile[a], axes[a]) for a in range(n_ax))
    out = []
    for a in range(n_ax):
        halo = kernel[a] - 1
        span = tile[a] + 2 * halo
        n_tiles = math.ceil(axes[a] / tile[a])
        length = n_tiles * tile[a] + 2 * halo  # the padded key axis
        out.append(
            _window_select_matrix(length, span, tile[a], n_tiles, dtype, "cpu")
            .to(device)
            .contiguous()
        )
    return tuple(out)


def _window_axis_select(
    x: Tensor, dim: int, size: int, step: int, n_windows: int, sel: Tensor | None = None
) -> Tensor:
    """Same result as :func:`_window_axis_split`, built as ONE matmul against a 0/1 selection matrix
    instead of ``n_windows`` overlapping ``narrow`` slices + ``stack`` -- the ``window_impl="select"``
    formulation (the default on device).

    Why: smoke rounds 13-17 on Trn2 (the FLUX-3-Action 136x184 grid, window 5x5) found the slice-and-stack
    windowing correct as a graph of its own but WRONG once fused with the attention core in a single
    bf16 graph, in a region that depends on the tile geometry (16x16: the last tile row, rel 0.20;
    8x8: everything but the last tile column, rel 0.19 / NaN) -- while the identical graph fed
    host-windowed operands (r17 ``host_window``, both tilings), fed fp32 inputs (8x8 ``fp32_in``,
    2e-6), or with the final crop done on the host (16x16 ``nocrop``) matched CPU to the bf16 band.
    So the trigger is the compiler's fusion of the overlapping strided bf16 reads with their consumer
    (the ``.float()`` cast / the matmul), not the arithmetic. A selection matmul has no strided
    overlapping read at all: the windowed axis is produced by TensorE from a plain contiguous
    operand, and the result is a fresh contiguous tensor. Cost at that grid: ~25 GFLOP of 0/1
    matmuls per call (microseconds on NC-v3), plus two small constant matrices per layout (``sel``,
    a host-built :func:`neighborhood_select_matrices` entry, or built in-graph when omitted).

    ``x``: any rank, windowed along ``dim``; the matmul contracts ``dim`` with the trailing axes
    flattened, so no transpose is needed for any ``dim`` (``torch.matmul`` broadcasts the selection
    matrix over the leading batch axes). Non-overlapping windows (``size == step``) stay a reshape.
    """
    if size == step and x.shape[dim] == n_windows * size:
        shape = list(x.shape)
        return x.reshape(*shape[:dim], n_windows, size, *shape[dim + 1 :])
    shape = list(x.shape)
    length = shape[dim]
    lead, trail = shape[:dim], shape[dim + 1 :]
    rest = 1
    for t in trail:
        rest *= t
    if sel is None:
        sel = _window_select_matrix(length, size, step, n_windows, x.dtype, x.device)
    x2 = x.reshape(*lead, length, rest)  # contiguous reshape: `dim` becomes the matmul K axis
    out = torch.matmul(sel, x2)  # [.., n_windows*size, rest], sel broadcast over `lead`
    return out.reshape(*lead, n_windows, size, *trail)


def _collapse_tiles_adjacent(x: Tensor, n_ax: int, heads: int, d: int) -> Tensor:
    """``[B, n_0, i_0, n_1, i_1, .., heads, D]`` -> ``[B*prod(n), heads, prod(i), D]`` using ONLY
    reshapes and single adjacent-axis transposes (each followed by ``contiguous()``): bubble each
    inner (window) dim rightwards past the next tile-count dim, merging as it goes, then swap the
    merged window dim with ``heads``."""
    for a in range(1, n_ax):
        # dims: 0=B, 1..a = n_0..n_{a-1}, a+1 = I (inner merged so far), a+2 = n_a, a+3 = i_a
        x = x.transpose(a + 1, a + 2).contiguous()  # -> [.., n_a, I, i_a, ..]
        s = list(x.shape)
        x = x.reshape(*s[: a + 2], s[a + 2] * s[a + 3], *s[a + 4 :])  # merge (I, i_a)
    # [B, n_0..n_{n-1}, I, heads, D] -> [B, n.., heads, I, D]
    x = x.transpose(n_ax + 1, n_ax + 2).contiguous()
    s = list(x.shape)
    return x.reshape(-1, heads, s[-2], d)


def _uncollapse_tiles_adjacent(
    x: Tensor, n_ax: int, b: int, heads: int, d: int, n_tiles: tuple, tile: tuple, axes: tuple
) -> Tensor:
    """Inverse of :func:`_collapse_tiles_adjacent` for the query tiling, then crop the padding:
    ``[B*prod(n), heads, prod(tile), D]`` -> ``[B, *axes, heads, D]``.

    Built from ``reshape`` + ``unbind`` + ``cat``/``stack`` ONLY -- no transposes at all. Smoke
    round 14 on Trn2: with the previous reshape + adjacent-transpose formulation the interior of the
    result matched CPU (rel 0.0036) while the last tile row / column were wrong (0.198 / 0.057), the
    four attention stages each matched CPU over ALL tiles, and this stage compiled alone failed with
    ``NCC_IXRO002 Undefined SB Memloc transpose`` -- i.e. the compiler mis-lowers the transpose chain
    for the partially-padded edge tiles. ``cat``/``stack`` along any dim are the ops this plugin
    trusts on Neuron (the VAE and the windowing stage use them throughout), so the interleave of
    (tile index, in-tile position) is done by concatenating tiles along their spatial axis instead.
    """
    x = x.reshape(b, *n_tiles, heads, *tile, d)
    # layout: [B, n_0..n_{k-1}, heads, t_0..t_{k-1}, D]. Walk axes last-to-first: remove n_a by
    # unbinding it and concatenating the pieces along t_a (which becomes the padded axis H_a).
    for a in reversed(range(n_ax)):
        n_dim = 1 + a  # n_a (n_{a+1}.. already removed)
        t_dim = (
            2 * a + 2
        )  # t_a after n_a is removed: 1 + a (remaining n) + 1 (heads) + a (t_0..t_{a-1})
        pieces = x.unbind(n_dim)
        x = torch.cat(pieces, dim=t_dim)
    # now [B, heads, H_0..H_{k-1}, D] -> heads after the spatial axes
    x = torch.stack(x.unbind(1), dim=1 + n_ax)  # [B, H_0..H_{k-1}, heads, D]
    for a in range(n_ax):
        x = x.narrow(1 + a, 0, axes[a])
    return x.contiguous()


def neighborhood_bias(
    axes: Sequence[int],
    kernel: Sequence[int],
    causal: Sequence[bool] | None,
    tile: Sequence[int],
    device: torch.device | str = "cpu",
) -> Tensor:
    """The ``[n_tiles_total, 1, prod(tile), prod(window)]`` fp32 additive bias (heads axis
    pre-broadcast) :func:`neighborhood_attention_tiled` needs for one grid layout, built on the host. Build it once
    per shape and pass it as ``bias=`` so it enters a compiled region as a plain graph input (a
    contiguous device tensor) rather than being constructed inside the trace."""
    n_ax = len(axes)
    causal_t = tuple(causal) if causal is not None else (False,) * n_ax
    tile_t = tuple(min(tile[a], axes[a]) for a in range(n_ax))
    return _tile_bias(tuple(axes), tuple(kernel), causal_t, tile_t, device, torch.float32)


def pad_free_tile(
    axes: Sequence[int], kernel: Sequence[int], max_tile: int = 16
) -> tuple[int, ...]:
    """Per-axis query-tile size that DIVIDES the axis (no partial last tile, so no padded queries
    and no crop): the largest divisor of ``axes[a]`` that is ``<= max_tile``, or the axis itself when
    it is shorter than ``max_tile``. Falls back to ``max_tile`` (padding) only when the sole divisor
    in range is 1 (a prime axis length), since a 1-wide tile would be absurdly inefficient.

    Why this exists: smoke rounds 13-15 on Trn2 found the fused device graph wrong ONLY on the
    partially padded last tile row / column (rel 0.20 / 0.057 vs 0.0036 in the interior and on the
    first tile), while the same computation split into two graphs (attention, then uncollapse+crop)
    matched CPU everywhere -- i.e. the compiler mis-handles the padded tile when the final crop is in
    the same graph as the attention, independent of how the uncollapse is written (transposes in r13,
    cat/stack in r14: identical numbers). A tiling with no padded tile sidesteps the whole class.
    For FLUX-3-Action's 136x184 grid with ``max_tile=16`` this gives (8, 8): 17x23 tiles of 64
    queries over 16x16=256-key windows -- fewer total MACs than 16x16 tiles (which pad 136->144 and
    184->192 and use 576-key windows), at the cost of smaller matmuls.
    """
    out = []
    for a, n in enumerate(axes):
        n = int(n)
        if n <= max_tile:
            out.append(n)
            continue
        best = 1
        for t in range(min(max_tile, n), 1, -1):
            if n % t == 0:
                best = t
                break
        out.append(best if best > 1 else max_tile)
    return tuple(out)


WINDOW_IMPLS = ("select", "slice")


def neighborhood_attention_tiled(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kernel: Sequence[int],
    causal: Sequence[bool] | None = None,
    tile: Sequence[int] | None = None,
    bias: Tensor | None = None,
    crop: bool = True,
    window_impl: str = "select",
    upcast_first: bool = False,
    select: Sequence[Tensor] | None = None,
) -> Tensor:
    """Halo-tiled neighborhood attention. ``q/k/v [B, *axes, heads, head_dim]`` -> same shape.

    ``kernel`` is the per-axis window (odd), ``causal`` the per-axis causal flag (default all False),
    ``tile`` the per-axis query-tile size (default: :func:`pad_free_tile` -- the largest divisor of
    each axis ``<= 16``, so no axis has a partially padded last tile; see that function for the
    device finding behind it). ``bias`` is the optional precomputed :func:`neighborhood_bias` for
    this layout (built on the host if omitted). ``crop=False`` returns the PADDED grid
    ``[B, *ceil(axes/tile)*tile, heads, D]`` without the final narrow (a diagnostic knob: lets a
    device smoke do the crop outside the compiled graph; identical to ``crop=True`` when no axis is
    padded). ``window_impl`` selects how the overlapping key/value windows are formed:
    ``"select"`` (default) = one 0/1 selection matmul per axis (:func:`_window_axis_select`, slice-
    free -- the formulation that survives fusion with the attention on Trn2), ``"slice"`` =
    ``narrow`` + ``stack`` (:func:`_window_axis_split`; correct alone, wrong once fused on Trn2 in
    bf16). ``select`` is the optional host-built :func:`neighborhood_select_matrices` for this layout
    and the inputs' dtype (graph inputs, like ``bias``; built in-graph from ``arange`` + compare when
    omitted -- a few static integer ops, no cache, one compile). ``upcast_first=True`` casts q/k/v
    to fp32 BEFORE padding/windowing instead of casting the windows afterwards (diagnostic knob;
    smoke r17's ``fp32_in`` variant passed at 8x8 where the bf16 windows failed). NATTEN semantics,
    bit-exact against ``flux3_action.neighborhood.neighborhood_attention_reference`` with every
    setting.
    """
    if window_impl not in WINDOW_IMPLS:
        raise ValueError(f"window_impl must be one of {WINDOW_IMPLS}, got {window_impl!r}")
    n_ax = q.ndim - 3
    kernel = tuple(kernel)
    causal = tuple(causal) if causal is not None else (False,) * n_ax
    if tile is None:
        tile = pad_free_tile(q.shape[1 : 1 + n_ax], kernel)
    tile = tuple(min(tile[a], q.shape[1 + a]) for a in range(n_ax))
    assert len(kernel) == len(causal) == len(tile) == n_ax, "kernel/causal/tile rank mismatch"
    axes = tuple(q.shape[1 : 1 + n_ax])
    for a in range(n_ax):
        if axes[a] < kernel[a]:
            raise ValueError(f"axis {a} length {axes[a]} < kernel {kernel[a]}")

    b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
    halo = tuple(ker - 1 for ker in kernel)
    span = tuple(tile[a] + 2 * halo[a] for a in range(n_ax))
    n_tiles = tuple(math.ceil(axes[a] / tile[a]) for a in range(n_ax))

    out_dtype = q.dtype
    if upcast_first:
        q, k, v = q.float(), k.float(), v.float()
    q_t, k_t, v_t = _window_qkv(
        q,
        k,
        v,
        n_ax,
        axes,
        halo,
        span,
        tile,
        n_tiles,
        heads,
        d,
        window_impl=window_impl,
        select=select,
    )
    # q_t: [B*tiles_total, heads, tile_tokens, D]; k_t/v_t: [..., win_tokens, D]

    if bias is None:
        bias = _tile_bias(axes, kernel, causal, tile, q.device, torch.float32)  # [tiles, tt, wt]
    if bias.dim() == 3:  # a caller-built [tiles, tt, wt] bias: add the heads axis
        bias = bias.unsqueeze(1)
    if b > 1:
        bias = bias.repeat(b, 1, 1, 1)  # [B*tiles_total, 1, tt, wt]
    scale = d**-0.5
    scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale + bias
    out_t = torch.matmul(
        torch.softmax(scores, dim=-1), v_t.float()
    )  # [B*tiles_total, heads, tt, D]

    padded = tuple(n_tiles[a] * tile[a] for a in range(n_ax))
    return _uncollapse_tiles_adjacent(
        out_t, n_ax, b, heads, d, n_tiles, tile, axes if crop else padded
    ).to(out_dtype)


def _window_qkv(
    q,
    k,
    v,
    n_ax,
    axes,
    halo,
    span,
    tile,
    n_tiles,
    heads,
    d,
    window_impl: str = "select",
    select: Sequence[Tensor] | None = None,
):
    """Pad + window + collapse q/k/v into per-tile dense-attention operands (the first stage of
    :func:`neighborhood_attention_tiled`, split out so the device smoke can compile it alone).
    ``window_impl`` / ``select``: see :func:`neighborhood_attention_tiled`."""
    if select is not None and len(select) != n_ax:
        raise ValueError(
            f"select must have one matrix per spatial axis ({n_ax}), got {len(select)}"
        )

    def window_axis(x, dim, size, step, n_windows, a):
        if window_impl == "select":
            sel = None if select is None else select[a]
            return _window_axis_select(x, dim, size, step, n_windows, sel)
        return _window_axis_split(x, dim, size, step, n_windows)

    # Pad K/V by halo (low) + halo + fill (high) on every spatial axis, Q by the trailing fill only.
    kpad_list, qpad_list = [], []
    for a in reversed(range(n_ax)):
        fill = n_tiles[a] * tile[a] - axes[a]
        kpad_list += [halo[a], halo[a] + fill]
        qpad_list += [0, fill]
    kp = F.pad(k, [0, 0, 0, 0, *kpad_list])
    vp = F.pad(v, [0, 0, 0, 0, *kpad_list])
    qp = F.pad(q, [0, 0, 0, 0, *qpad_list])

    # Window each spatial axis IN PLACE: axis a -> (n_tiles_a, tile_a) for queries (a pure reshape,
    # stride == size) and (n_tiles_a, span_a) for keys/values (overlapping: selection matmul, or
    # narrow + stack). Walk the axes from the last to the first so each replaced axis's index stays
    # valid.
    for a in reversed(range(n_ax)):
        dim = 1 + a
        qp = _window_axis_split(qp, dim, tile[a], tile[a], n_tiles[a])  # always a reshape
        kp = window_axis(kp, dim, span[a], tile[a], n_tiles[a], a)
        vp = window_axis(vp, dim, span[a], tile[a], n_tiles[a], a)
    # now [B, n_0, i_0, n_1, i_1, .., heads, D]
    q_t = _collapse_tiles_adjacent(qp, n_ax, heads, d)
    k_t = _collapse_tiles_adjacent(kp, n_ax, heads, d)
    v_t = _collapse_tiles_adjacent(vp, n_ax, heads, d)
    return q_t, k_t, v_t
