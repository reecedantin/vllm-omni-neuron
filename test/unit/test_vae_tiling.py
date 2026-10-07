# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape VAE tiling helper: one compiled tile shape, diffusers-exact blend, tile-parallel."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from vllm_omni_neuron.diffusion.layers.vae_tiling import (
    TileGrid,
    kept_ranges,
    merge_tiles,
    run_tiles,
    tile_starts,
)


def _diffusers_tiled_merge(tiles, grid_h, grid_w, stride_h, stride_w, nbh, nbw, out_h, out_w):
    """diffusers AutoencoderKLWan.tiled_decode's merge, verbatim semantics (blend_v then blend_h
    against the RAW neighbours, then crop to stride) -- the reference for the evenly spaced case."""

    def blend_v(a, b, n):
        for y in range(n):
            b[..., y, :] = a[..., -n + y, :] * (1 - y / n) + b[..., y, :] * (y / n)
        return b

    def blend_h(a, b, n):
        for x in range(n):
            b[..., :, x] = a[..., :, -n + x] * (1 - x / n) + b[..., :, x] * (x / n)
        return b

    rows = [[tiles[(i, j)].clone() for j in range(grid_w)] for i in range(grid_h)]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            tile = tile.clone()
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, nbh)
            if j > 0:
                tile = blend_h(row[j - 1], tile, nbw)
            result_row.append(tile[..., :stride_h, :stride_w])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)[..., :out_h, :out_w]


# --- 1D primitives ---


@pytest.mark.parametrize(
    "total,tile,stride,want",
    [
        (32, 8, 6, [0, 6, 12, 18, 24]),  # evenly spaced, last lands exactly
        (
            33,
            8,
            6,
            [0, 6, 12, 18, 24, 25],
        ),  # same tile COUNT as diffusers (0..30); last pulled back
        (8, 8, 6, [0]),
        (5, 8, 6, [0]),  # smaller than a tile: one (padded) tile
        (16, 8, 8, [0, 8]),  # abutting, no overlap
        (17, 8, 8, [0, 8, 9]),
    ],
)
def test_tile_starts(total, tile, stride, want):
    starts = tile_starts(total, tile, stride)
    assert starts == want
    # every tile is exactly `tile` wide and the union covers [0, total)
    covered = set()
    for s in starts:
        assert s + tile >= min(total, tile) and s >= 0
        covered |= set(range(s, s + tile))
    assert covered >= set(range(min(total, tile)))


def test_kept_ranges_partition_the_output():
    starts = tile_starts(33, 8, 6)
    kept = kept_ranges(starts, 8, 6, 33)
    assert kept == [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 33)]
    assert kept[0][0] == 0 and kept[-1][1] == 33
    for (a0, a1), (b0, b1) in zip(kept, kept[1:], strict=False):
        assert a1 == b0  # contiguous, no gaps, no double counting
    # evenly spaced -> exactly crop-to-stride
    assert kept_ranges([0, 6, 12, 18, 24], 8, 6, 32) == [
        (0, 6),
        (6, 12),
        (12, 18),
        (18, 24),
        (24, 32),
    ]


def test_tile_starts_rejects_bad_geometry():
    with pytest.raises(ValueError):
        tile_starts(10, 4, 6)
    with pytest.raises(ValueError):
        tile_starts(10, 0, 1)


# --- merge: diffusers-exact for even grids ---


@pytest.mark.parametrize("scale", [1, 8])
def test_merge_matches_diffusers_order_on_even_grid(scale):
    torch.manual_seed(0)
    th, tw, sh, sw = 8, 8, 6, 4
    grid = TileGrid.for_axes(total=(20, 16), tile=(th, tw), stride=(sh, sw), out_scale=scale)
    assert grid.shape == (3, 3) and grid.starts == ((0, 6, 12), (0, 4, 8))  # 16 = 8 + 2*4: even
    tiles = {idx: torch.randn(1, 3, 2, th * scale, tw * scale) for idx in grid.indices()}
    got = merge_tiles(tiles, grid)
    want = _diffusers_tiled_merge(
        tiles,
        3,
        3,
        sh * scale,
        sw * scale,
        (th - sh) * scale,
        (tw - sw) * scale,
        20 * scale,
        16 * scale,
    )
    assert got.shape == (1, 3, 2, 20 * scale, 16 * scale)
    # diffusers would add a 4th, narrower tile at 18 / 12 and crop every tile to its stride; this
    # helper pulls the last tile back instead, so the two agree exactly up to the last tile's start
    # on each axis (its kept remainder is the one place the schemes differ by construction).
    hs, ws = grid.starts[0][-1] * scale, grid.starts[1][-1] * scale
    torch.testing.assert_close(got[..., :hs, :ws], want[..., :hs, :ws])


def test_merge_accepts_list_in_row_major_order():
    torch.manual_seed(0)
    grid = TileGrid.for_axes(total=(12, 12), tile=(8, 8), stride=(4, 4))
    tiles = {idx: torch.randn(2, 8, 8) for idx in grid.indices()}
    as_list = [tiles[idx] for idx in grid.indices()]
    torch.testing.assert_close(merge_tiles(as_list, grid), merge_tiles(tiles, grid))


# --- pulled-back last tile: tiled == untiled for a local operator ---


def _pointwise_op(x):
    """A per-pixel op: tiled evaluation must reproduce the untiled result EXACTLY (the ramps blend
    identical values), for any grid including pulled-back last tiles and padded inputs."""
    return x * 2.0 + 1.0


def _local_op(x, k=3):
    """A depthwise blur standing in for a conv VAE: like a real VAE its zero-padded tile edges are
    wrong within the kernel halo, so tiled vs untiled is a PSNR question (diffusers' tiling has the
    same seams), not an exactness one."""
    w = torch.ones(x.shape[1], 1, k, k) / (k * k)
    return torch.nn.functional.conv2d(x, w, padding=k // 2, groups=x.shape[1])


def _psnr(a, b):
    mse = (a - b).pow(2).mean().clamp_min(1e-20)
    return (10 * torch.log10((b.max() - b.min()) ** 2 / mse)).item()


@pytest.mark.parametrize("total", [(33, 21), (40, 17), (9, 9), (5, 30)])
def test_tiled_equals_untiled_with_pulled_back_tiles_and_padding(total):
    torch.manual_seed(0)
    x = torch.randn(1, 4, *total)
    tile, stride = (8, 8), (6, 6)
    grid = TileGrid.for_axes(total=total, tile=tile, stride=stride)
    xp = grid.pad_input(x)
    assert xp.shape[-2:] == grid.padded_total()
    tiles = {idx: _pointwise_op(grid.slice_input(xp, idx)) for idx in grid.indices()}
    # single compiled shape: every tile input is exactly `tile`
    assert {tuple(t.shape[-2:]) for t in tiles.values()} == {tile}
    got = merge_tiles(tiles, grid)
    want = _pointwise_op(x)
    assert got.shape == x.shape
    torch.testing.assert_close(got, want)


def test_overlap_blend_hides_seams_of_a_local_op():
    """With overlap >= 2x the halo, the blur's seams blend away: high PSNR vs untiled, and the
    pulled-back last tile is no worse than the regular ones (LTX-2 measured 34->39 dB going from
    overlap 2 to 4 on its real decoder; this toy op is far more forgiving)."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 45, 37)
    grid = TileGrid.for_axes(total=(45, 37), tile=(16, 16), stride=(12, 12))  # overlap 4, halo 1
    tiles = {idx: _local_op(grid.slice_input(x, idx)) for idx in grid.indices()}
    got = merge_tiles(tiles, grid)
    want = _local_op(x)
    assert _psnr(got, want) > 40.0
    # the last (pulled-back) row/column strip is as clean as the first
    assert _psnr(got[..., -12:, :], want[..., -12:, :]) > 40.0


def test_encode_direction_in_scale():
    """Encode tiles are in pixel space and outputs in latent space (in_scale > 1): kept ranges and
    blend lengths must land on whole latent units."""
    torch.manual_seed(0)
    ratio = 8
    total, tile, stride = (176, 120), (64, 64), (48, 48)  # 176 = 2 tiles + pulled back
    grid = TileGrid.for_axes(total=total, tile=tile, stride=stride, in_scale=ratio)
    assert grid.out_total() == (22, 15)
    x = torch.randn(1, 3, *total)

    def enc(t):  # a local "encoder": avg-pool by ratio (exactly tile-separable, halo-free)
        return torch.nn.functional.avg_pool2d(t, ratio)

    tiles = {idx: enc(grid.slice_input(x, idx)) for idx in grid.indices()}
    assert all(t.shape[-2:] == (8, 8) for t in tiles.values())
    got = merge_tiles(tiles, grid)
    torch.testing.assert_close(got, enc(x))
    with pytest.raises(ValueError, match="multiples of in_scale"):
        TileGrid.for_axes(total=total, tile=(60, 64), stride=stride, in_scale=ratio)


def test_slice_input_is_contiguous_and_fixed_shape():
    grid = TileGrid.for_axes(total=(33, 21), tile=(8, 8), stride=(6, 6), out_scale=8)
    z = torch.randn(1, 16, 3, 33, 21)
    for idx in grid.indices():
        t = grid.slice_input(z, idx)
        assert t.is_contiguous() and t.shape == (1, 16, 3, 8, 8)
    assert grid.num_tiles == len(grid.indices()) == 6 * 4


# --- tile-parallel dealing + gather ---


def test_run_tiles_single_process_returns_every_tile():
    grid = TileGrid.for_axes(total=(20, 16), tile=(8, 6), stride=(6, 4))
    seen = []

    def fn(n, idx):
        seen.append((n, idx))
        return torch.full((1, 2, 8, 6), float(n))

    out = run_tiles(grid, fn)
    assert out is not None and set(out) == set(grid.indices())
    assert [n for n, _ in seen] == list(range(grid.num_tiles))


def test_run_tiles_dealing_with_explicit_rank_override(monkeypatch):
    """Round-robin assignment: rank r evaluates exactly the items n with n % world == r, and rank 0
    merges what every rank gathered. The collective is stubbed so no process group is needed."""
    grid = TileGrid.for_axes(total=(20, 16), tile=(8, 6), stride=(6, 4))  # 3 x 4 = 12 tiles
    world = 3
    gathered_calls = []

    def fake_gather_object(obj, object_gather_list=None, dst=0, group=None):
        gathered_calls.append(obj)
        if object_gather_list is not None:  # rank 0: pretend the other ranks sent their shares
            object_gather_list[0] = obj
            for r in range(1, world):
                object_gather_list[r] = {
                    idx: torch.full((1,), float(n))
                    for n, idx in enumerate(grid.indices())
                    if n % world == r
                }

    monkeypatch.setattr(torch.distributed, "gather_object", fake_gather_object)
    for rank in range(world):
        evaluated = []

        def fn(n, idx):
            evaluated.append(n)
            return torch.full((1,), float(n))

        out = run_tiles(grid, fn, world_size=world, rank=rank)
        assert evaluated == [n for n in range(grid.num_tiles) if n % world == rank]
        if rank == 0:
            assert out is not None and set(out) == set(grid.indices())
            assert all(out[idx].item() == n for n, idx in enumerate(grid.indices()))
        else:
            assert out is None
    assert len(gathered_calls) == world


def _worker(rank, world, init_file, result_file):
    import torch
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, merge_tiles, run_tiles

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    x = torch.randn(1, 4, 33, 21)  # identical on every rank (same seed)
    grid = TileGrid.for_axes(total=(33, 21), tile=(8, 8), stride=(6, 6))
    xp = grid.pad_input(x)
    tiles = run_tiles(grid, lambda n, idx: _pointwise_op(grid.slice_input(xp, idx)))
    if rank == 0:
        got = merge_tiles(tiles, grid)
        want = _pointwise_op(x)
        torch.save({"got": got, "want": want}, result_file)
    else:
        assert tiles is None
    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_run_tiles_two_gloo_ranks_bit_identical_to_serial():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        result_file = os.path.join(d, "result.pt")
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(r, 2, init_file, result_file)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(100)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        res = torch.load(result_file)
    torch.testing.assert_close(res["got"], res["want"], atol=1e-6, rtol=1e-6)
