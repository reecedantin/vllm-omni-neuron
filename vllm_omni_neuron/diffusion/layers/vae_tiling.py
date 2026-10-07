# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape spatial tiling for device VAE encode/decode, shared across models.

Why this exists (fleet inputs, 2026-10-03):

* **One compiled shape per tile role.** diffusers' ``tiled_decode`` / ``tiled_encode`` let the last
  tile on each axis be *narrower* (``min(total, start + tile)``), so a grid that does not divide
  evenly produces up to four distinct tile shapes -- four NEFFs per decoder specialization, each a
  cold compile of minutes (LTX-2: ~19 min per shape). Here every
  tile is exactly ``tile`` wide: the last tile is *pulled back* to end at the boundary
  (``starts = [0, stride, 2*stride, ..., total - tile]``), so one NEFF serves the whole grid, and a
  total smaller than one tile is zero-padded and the output cropped.
* **Per-core graph limits.** A full-frame graph does not fit one NeuronCore at real resolutions:
  the Wan2.2-5B encoder fails above 192 px, LTX-2 at 512x768
  hits the 10M-instruction limit (``NCC_EVRF007``). A tile that compiles once decodes in well under
  a second warm (LTX-2: 0.46 s vs 330 s CPU).
* **Tile-parallel across TP ranks.** Tiles are independent work items with identical shapes, so
  they deal round-robin across the ranks that would otherwise idle during the VAE stage
  (e.g. 42 tiles over 8 ranks). :func:`run_tiles` does the dealing
  and the gather; the merge stays on rank 0.

The merge (:func:`merge_tiles`) reproduces diffusers' ``blend_v`` / ``blend_h`` linear ramps and
crop-to-stride exactly for evenly spaced tiles (bit-exact against the existing Wan ``tiled_decode``
in ``test/unit/test_vae_tiling.py``) and extends them to the pulled-back last tile: tile ``j``
contributes the output range from where tile ``j-1``'s kept range ends to ``start_j + stride``
(the last tile: to the end), and the first ``blend`` samples of that range are the ramp against tile
``j-1``. Per-axis ``scale`` maps tile coordinates to output coordinates (decode: latent -> pixel,
``spatial_compression_ratio``; encode: pixel -> latent, ``1 / ratio``, expressed as the pair
``(in_scale, out_scale)``).

Usage (any VAE; the callable owns its model, feat caches, device placement)::

    grid = TileGrid.for_axes(total=(h, w), tile=(th, tw), stride=(sh, sw), out_scale=(8, 8))
    tiles = run_tiles(grid, lambda idx, (i, j): decode_one(grid.slice_input(z, i, j)))
    if tiles is not None:                      # rank 0 (or single process)
        out = merge_tiles(tiles, grid)         # [..., H_out, W_out]
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


def tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Start offsets of fixed-width tiles covering ``[0, total)``: ``0, stride, 2*stride, ...`` with
    the last tile pulled back to ``total - tile`` so every tile has the same width (one compiled
    shape). ``total <= tile`` gives ``[0]`` (the caller pads the input to ``tile`` and crops)."""
    if tile <= 0 or stride <= 0:
        raise ValueError(f"tile and stride must be positive, got tile={tile} stride={stride}")
    if stride > tile:
        raise ValueError(f"stride {stride} > tile {tile} would leave gaps")
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile, stride))
    starts.append(total - tile)
    return starts


def kept_ranges(starts: Sequence[int], tile: int, stride: int, total: int) -> list[tuple[int, int]]:
    """For each tile, the ``[begin, end)`` of the output it owns (in tile-input coordinates): from
    where the previous tile's range ended to its own ``start + stride`` (the last: to ``total``).
    For evenly spaced tiles this is exactly diffusers' crop-to-stride."""
    out = []
    prev_end = 0
    for n, s in enumerate(starts):
        end = total if n == len(starts) - 1 else min(s + stride, total)
        out.append((prev_end, end))
        prev_end = end
    return out


@dataclass(frozen=True)
class TileGrid:
    """A fixed-shape tile grid over the trailing spatial axes of a ``[..., *axes]`` tensor.

    ``total``/``tile``/``stride`` are per-axis in INPUT coordinates; ``out_scale`` is the per-axis
    integer factor from input to output coordinates (decode: ``spatial_compression_ratio``; encode:
    use ``in_scale`` instead, see :meth:`for_axes`). ``blend`` (input coords) defaults to
    ``tile - stride``, diffusers' convention.
    """

    total: tuple[int, ...]
    tile: tuple[int, ...]
    stride: tuple[int, ...]
    starts: tuple[tuple[int, ...], ...]
    out_scale: tuple[int, ...]  # output units per input unit (>= 1)
    in_scale: tuple[int, ...]  # input units per output unit (>= 1); exactly one of the two is > 1
    blend: tuple[int, ...]

    @classmethod
    def for_axes(
        cls,
        total: Sequence[int],
        tile: Sequence[int],
        stride: Sequence[int],
        out_scale: Sequence[int] | int = 1,
        in_scale: Sequence[int] | int = 1,
        blend: Sequence[int] | None = None,
    ) -> TileGrid:
        n = len(total)
        if isinstance(out_scale, int):
            out_scale = (out_scale,) * n
        if isinstance(in_scale, int):
            in_scale = (in_scale,) * n
        if not (len(tile) == len(stride) == len(out_scale) == len(in_scale) == n):
            raise ValueError("total/tile/stride/out_scale/in_scale rank mismatch")
        for a in range(n):
            if in_scale[a] > 1 and out_scale[a] > 1:
                raise ValueError("use either out_scale or in_scale per axis, not both")
            if in_scale[a] > 1 and (tile[a] % in_scale[a] or stride[a] % in_scale[a]):
                raise ValueError(
                    f"axis {a}: tile {tile[a]} and stride {stride[a]} must be multiples of "
                    f"in_scale {in_scale[a]} (encode tiles must map to whole latent units)"
                )
        if blend is None:
            blend = tuple(tile[a] - stride[a] for a in range(n))
        starts = tuple(tuple(tile_starts(total[a], tile[a], stride[a])) for a in range(n))
        return cls(
            tuple(total),
            tuple(tile),
            tuple(stride),
            starts,
            tuple(out_scale),
            tuple(in_scale),
            tuple(blend),
        )

    @property
    def ndim(self) -> int:
        return len(self.total)

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of tiles per axis."""
        return tuple(len(s) for s in self.starts)

    @property
    def num_tiles(self) -> int:
        return math.prod(self.shape)

    def indices(self) -> list[tuple[int, ...]]:
        """All tile indices in row-major order (axis 0 slowest)."""
        out: list[tuple[int, ...]] = [()]
        for n in self.shape:
            out = [idx + (i,) for idx in out for i in range(n)]
        return out

    def padded_total(self) -> tuple[int, ...]:
        return tuple(max(self.total[a], self.tile[a]) for a in range(self.ndim))

    def pad_input(self, x: Tensor) -> Tensor:
        """Zero-pad the trailing axes up to one tile where the input is smaller than a tile."""
        pad: list[int] = []
        for a in reversed(range(self.ndim)):
            pad += [0, max(0, self.tile[a] - self.total[a])]
        return torch.nn.functional.pad(x, pad) if any(pad) else x

    def slice_input(self, x: Tensor, idx: Sequence[int]) -> Tensor:
        """The ``idx``-th tile of ``x`` (trailing axes), always exactly ``tile`` wide. ``x`` must
        already be :meth:`pad_input`-ed when smaller than one tile. Uses ``narrow`` + ``contiguous``
        (static bounds, no data-dependent slicing) so the result is a legal compiled-graph input."""
        for a in range(self.ndim):
            dim = x.ndim - self.ndim + a
            x = x.narrow(dim, self.starts[a][idx[a]], self.tile[a])
        return x.contiguous()

    def out_units(self, a: int, v: int) -> int:
        """Convert ``v`` input units on axis ``a`` to output units."""
        return v * self.out_scale[a] // self.in_scale[a]

    def out_total(self) -> tuple[int, ...]:
        return tuple(self.out_units(a, self.total[a]) for a in range(self.ndim))


def _ramp(prev: Tensor, cur: Tensor, n: int, dim: int) -> Tensor:
    """diffusers' linear blend of the last ``n`` of ``prev`` into the first ``n`` of ``cur`` along
    ``dim``: ``prev * (1 - x/n) + cur * (x/n)``, ``x = 0..n-1``."""
    shape = [1] * cur.ndim
    shape[dim] = n
    w = (torch.arange(n, device=cur.device, dtype=cur.dtype) / n).reshape(shape)
    return prev * (1.0 - w) + cur * w


def merge_tiles(tiles: dict[tuple[int, ...], Tensor] | Sequence[Tensor], grid: TileGrid) -> Tensor:
    """Blend and assemble per-tile outputs (``[..., *tile_out]``, output coordinates) into the
    full ``[..., *out_total]`` tensor. diffusers' order per tile: blend against the tile above
    (axis 0), then against the tile to the left (axis 1), each against the *raw* neighbour, then
    crop to the kept range. Equals ``AutoencoderKLWan.tiled_decode``'s merge for evenly spaced
    tiles; the pulled-back last tile keeps only ``[prev_end, total)``."""
    if not isinstance(tiles, dict):
        tiles = dict(zip(grid.indices(), tiles, strict=True))
    nd = grid.ndim
    kept = [
        kept_ranges(grid.starts[a], grid.tile[a], grid.stride[a], grid.total[a]) for a in range(nd)
    ]
    blended: dict[tuple[int, ...], Tensor] = {}
    pieces: dict[tuple[int, ...], Tensor] = {}
    for idx in grid.indices():
        cur = tiles[idx]
        tile_dims = [cur.ndim - nd + a for a in range(nd)]
        for a in range(nd):
            if idx[a] == 0:
                continue
            prev_idx = idx[:a] + (idx[a] - 1,) + idx[a + 1 :]
            prev = tiles[prev_idx]  # raw neighbour, as diffusers does
            s_cur, s_prev = grid.starts[a][idx[a]], grid.starts[a][idx[a] - 1]
            begin = kept[a][idx[a]][0]  # output-owned range start, input coords
            n_blend = min(grid.blend[a], kept[a][idx[a]][1] - begin)
            n = grid.out_units(a, n_blend)
            if n <= 0:
                continue
            off_cur = grid.out_units(a, begin - s_cur)
            off_prev = grid.out_units(a, begin - s_prev)
            d = tile_dims[a]
            prev_part = prev.narrow(d, off_prev, n)
            head = cur.narrow(d, 0, off_cur) if off_cur > 0 else None
            mid = _ramp(prev_part, cur.narrow(d, off_cur, n), n, d)
            tail = cur.narrow(d, off_cur + n, cur.shape[d] - off_cur - n)
            cur = torch.cat([p for p in (head, mid, tail) if p is not None and p.shape[d] > 0], d)
        blended[idx] = cur
        # crop to the kept range
        piece = cur
        for a in range(nd):
            begin, end = kept[a][idx[a]]
            s = grid.starts[a][idx[a]]
            d = tile_dims[a]
            piece = piece.narrow(d, grid.out_units(a, begin - s), grid.out_units(a, end - begin))
        pieces[idx] = piece
    # assemble: concatenate along the last axis first, then outwards
    return _assemble(pieces, grid.shape, (), nd)


def _assemble(pieces, shape, prefix, nd):
    a = len(prefix)
    if a == nd:
        return pieces[prefix]
    parts = [_assemble(pieces, shape, prefix + (i,), nd) for i in range(shape[a])]
    dim = parts[0].ndim - nd + a
    return torch.cat(parts, dim)


def run_tiles(
    grid: TileGrid,
    fn: Callable[[int, tuple[int, ...]], Tensor],
    *,
    group=None,
    world_size: int | None = None,
    rank: int | None = None,
) -> dict[tuple[int, ...], Tensor] | None:
    """Evaluate ``fn(n, idx) -> Tensor`` for every tile, dealt round-robin across the ranks of
    ``group`` (``n % world == rank``), and gather the results to rank 0 over the host (gloo)
    ``group``. Returns the ``{idx: tensor}`` dict on rank 0 (and in a single process); ``None`` on
    the other ranks. ``fn`` must return a CPU tensor (the gather is a host collective) and every
    rank must call this with the same ``grid``.

    This is the tile-parallel split: tiles are value-independent work
    items, so no rank needs anything from another until the final gather, and the merge on rank 0
    is the untouched single-process code. ``world_size``/``rank`` override the ``torch.distributed``
    lookup for tests.
    """
    idxs = grid.indices()
    if world_size is None or rank is None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group)
            rank = torch.distributed.get_rank(group)
        else:
            world_size, rank = 1, 0
    mine = {idx: fn(n, idx) for n, idx in enumerate(idxs) if n % world_size == rank}
    if world_size == 1:
        return mine
    local = {idx: t.detach().cpu() for idx, t in mine.items()}
    gathered: list[dict] | None = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local, gathered, dst=0, group=group)
    if rank != 0:
        return None
    out: dict[tuple[int, ...], Tensor] = {}
    for part in gathered:  # type: ignore[union-attr]
        out.update(part)
    return out
