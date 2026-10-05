# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the fixed-shape tiled LTX-2 VAE decode (``models/ltx2/vae_tiling.py``).

With a position-wise "decoder" (nearest upsample), every tile decodes to exactly the matching crop
of the full decode, so a correct tile layout + crop + blend must reproduce the full decode up to
the blend's float rounding (~1e-7), including the pinned last tile and latents smaller than one
tile. The real-VAE comparison (tiled vs untiled) is ``LTX2_VAE_TILING_REAL=<model dir>``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_neuron.diffusion.models.ltx2.vae_tiling import TiledLTX2VideoDecoder, tile_starts


class _PointwiseDecoder(torch.nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, z, temb=None, causal=None):
        x = z[:, :3] * 0.5 + torch.arange(3, dtype=z.dtype).view(1, 3, 1, 1, 1)
        return x.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)


def _fake_vae(scale=4):
    return SimpleNamespace(
        decoder=_PointwiseDecoder(scale), spatial_compression_ratio=scale, dtype=torch.float32
    )


@pytest.mark.parametrize(
    "total,tile,stride,expect",
    [
        (24, 8, 6, [0, 6, 12, 16]),
        (20, 8, 6, [0, 6, 12]),
        (8, 8, 6, [0]),
        (5, 8, 6, [0]),
        (9, 8, 6, [0, 1]),
    ],
)
def test_tile_starts_fixed_size_and_cover(total, tile, stride, expect):
    starts = tile_starts(total, tile, stride)
    assert starts == expect
    assert starts[0] == 0 and starts[-1] + tile >= total
    assert all(b - a <= stride for a, b in zip(starts, starts[1:], strict=False))


@pytest.mark.parametrize("h,w", [(16, 24), (16, 20), (16, 8), (16, 5), (20, 24), (34, 60), (7, 9)])
def test_tiled_matches_full_for_pointwise_decoder(h, w):
    vae = _fake_vae()
    z = torch.randn(1, 4, 3, h, w)
    full = vae.decoder(z)
    shapes = []
    dec = TiledLTX2VideoDecoder(vae, "cpu", tile_h=16, tile_w=8, overlap=2)
    orig = dec._decode_tile

    def spy(zt):
        shapes.append(tuple(zt.shape))
        return orig(zt)

    dec._decode_tile = spy
    out = dec.decode(z)
    assert out.shape == full.shape
    # exact layout/crop; the blend ramp a*(1-w)+a*w only costs float rounding (~1 ulp)
    assert torch.allclose(out, full, rtol=0, atol=1e-6), (out - full).abs().max()
    assert len(set(shapes)) == 1 and shapes[0][-2:] == (16, 8), shapes  # one compiled shape


def test_blend_is_linear_ramp_in_overlap():
    """Constant-per-tile decoder: the overlap ramps linearly from the left tile's value to the
    right tile's (diffusers' blend_h weights), and is flat elsewhere."""
    calls = []

    class _Const(torch.nn.Module):
        def forward(self, z, temb=None, causal=None):
            calls.append(None)
            v = float(len(calls))
            return torch.full((1, 3, z.shape[2], z.shape[3] * 4, z.shape[4] * 4), v)

    vae = SimpleNamespace(decoder=_Const(), spatial_compression_ratio=4, dtype=torch.float32)
    dec = TiledLTX2VideoDecoder(vae, "cpu", tile_h=4, tile_w=8, overlap=2)
    out = dec.decode(torch.zeros(1, 4, 1, 4, 14))  # tiles at 0 and 6 (stride 6, 2 overlap)
    row = out[0, 0, 0, 0]
    assert torch.all(row[:24] == 1.0)
    ramp = row[24:32]  # 2 latent * 4 px
    assert torch.allclose(ramp, 1.0 + torch.arange(8) / 8.0)
    assert torch.all(row[32:] == 2.0)


@pytest.mark.skipif(
    not os.environ.get("LTX2_VAE_TILING_REAL"), reason="set LTX2_VAE_TILING_REAL=<LTX-2.5 dir>"
)
def test_real_vae_tiled_close_to_untiled():
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        os.environ["LTX2_VAE_TILING_REAL"], subfolder="vae", torch_dtype=torch.float32
    ).eval()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, 128, 3, 16, 24, generator=g)
    with torch.no_grad():
        full = vae.decoder(z, None)
        tiled = TiledLTX2VideoDecoder(vae, "cpu").decode(z)
    mse = (
        float(((full.clamp(-1, 1) - tiled.clamp(-1, 1)) ** 2).mean()) / 4.0
    )  # [-1,1] -> [0,1] range
    psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
    assert psnr > 35.0, psnr


def _parallel_worker(rank, world, port, out_path, shape):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    group = SimpleNamespace(
        world_size=world, rank_in_group=rank, ranks=list(range(world)), cpu_group=dist.group.WORLD
    )
    dec = TiledLTX2VideoDecoder(_fake_vae(), "cpu", tile_h=16, tile_w=8, overlap=2, group=group)
    calls = []
    orig = dec._decode_tile

    def spy(zt):
        calls.append(1)
        return orig(zt)

    dec._decode_tile = spy
    if rank == 0:
        z = torch.randn(*shape, generator=torch.Generator().manual_seed(0)).bfloat16()
        torch.save({"z": z, "out": dec.decode(z), "calls": len(calls)}, f"{out_path}.0")
    else:
        dec.serve()
        torch.save({"calls": len(calls)}, f"{out_path}.{rank}")
    dist.destroy_process_group()


@pytest.mark.parametrize("world,h,w", [(2, 16, 24), (3, 20, 24), (4, 16, 8)])
def test_tile_parallel_decode_is_bit_identical(tmp_path, world, h, w):
    """Tiles spread over gloo ranks (rank 0 broadcasts the latent and merges) give exactly the
    single-rank decode, and every tile is decoded exactly once."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "res")
    shape = (1, 4, 3, h, w)
    mp.start_processes(
        _parallel_worker,
        args=(world, port, out, shape),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    r = [torch.load(f"{out}.{i}") for i in range(world)]
    single = TiledLTX2VideoDecoder(_fake_vae(), "cpu", tile_h=16, tile_w=8, overlap=2)
    ref = single.decode(r[0]["z"])
    assert torch.equal(r[0]["out"], ref)
    n_tiles = len(tile_starts(max(h, 16), 16, 14)) * len(tile_starts(max(w, 8), 8, 6))
    assert sum(x["calls"] for x in r) == n_tiles
