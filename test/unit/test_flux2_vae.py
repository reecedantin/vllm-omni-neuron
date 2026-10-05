# SPDX-License-Identifier: Apache-2.0
"""Tiled FLUX.2 VAE decode on CPU (tiny-flux2): whole-image GroupNorm passes, dynamo-traceability of
the pass graphs, and tile-parallel exactness."""

from __future__ import annotations

import math
import os

import torch
import torch.multiprocessing as mp

from .test_flux2_components import _free_port, tiny_flux2  # noqa: F401

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _psnr(a, b):
    a, b = a.float().clamp(-1, 1), b.float().clamp(-1, 1)
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)


def _vae(path, passes=2, untiled_max=0):
    from diffusers import AutoencoderKLFlux2

    os.environ.update(
        FLUX2_VAE_TILE="16",
        FLUX2_VAE_OVERLAP="4",
        FLUX2_VAE_GN_PASSES=str(passes),
        FLUX2_VAE_UNTILED_MAX=str(untiled_max),
    )
    from vllm_omni_neuron.diffusion.models.flux2.vae_flux2 import NeuronFlux2Vae

    v = NeuronFlux2Vae(AutoencoderKLFlux2.from_pretrained(path, subfolder="vae").eval())
    v._use_device = True  # take the device code path (eager on CPU tensors)
    return v


def _z():
    return torch.randn(1, 32, 40, 24, generator=torch.Generator().manual_seed(0))


def test_global_groupnorm_beats_tile_local(tiny_flux2):  # noqa: F811
    # strongly non-stationary latent (brighter/louder bottom-right) so per-tile GN statistics are wrong
    z = _z()
    z[..., 20:, 12:] = z[..., 20:, 12:] * 3 + 2
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    with torch.no_grad():
        ref = _vae(tiny_flux2, untiled_max=10**6).decode(z, return_dict=False)[0]
        local = _vae(tiny_flux2, passes=1).decode(z, return_dict=False)[0]
        glob = _vae(tiny_flux2, passes=2).decode(z, return_dict=False)[0]
    assert glob.shape == ref.shape
    # tiny 16-latent tiles: receptive-field + tile-local attention error dominates what is left; the
    # real-weight number (1024 px, 64-latent tiles) is 37.6 dB vs 24.0 dB for plain tiling
    assert rel(glob, ref) < 0.85 * rel(local, ref), (rel(glob, ref), rel(local, ref))


def test_pass_graphs_trace_fullgraph(tiny_flux2):  # noqa: F811
    """collect/apply mutate a shared context inside the traced region: they must trace as one graph
    each (fullgraph) and reproduce eager exactly."""
    z = _z()
    with torch.no_grad():
        eager = _vae(tiny_flux2, passes=2).decode(z, return_dict=False)[0]
        v = _vae(tiny_flux2, passes=2)
        v.compile("eager")
        compiled = v.decode(z, return_dict=False)[0]
    assert torch.allclose(compiled, eager, atol=1e-5), float((compiled - eager).abs().max())


def _par(rank, world, port, path, out_path, rank_dump=None):
    import torch.distributed as dist

    if rank_dump:
        os.environ["FLUX2_RANK_DUMP"] = rank_dump
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    with torch.no_grad():
        out = _vae(path, passes=2).decode(_z(), return_dict=False)[0]
    if rank == 0:
        torch.save(out, out_path)
    dist.destroy_process_group()


def test_tile_parallel_matches_single(tiny_flux2, tmp_path):  # noqa: F811
    v = _vae(tiny_flux2, passes=2)
    v.tile_parallel = False
    with torch.no_grad():
        single = v.decode(_z(), return_dict=False)[0]
    p, dump = tmp_path / "par.pt", tmp_path / "ranks"
    mp.spawn(_par, args=(3, _free_port(), tiny_flux2, str(p), str(dump)), nprocs=3, join=True)
    assert torch.allclose(torch.load(p), single, atol=1e-6)
    # FLUX2_RANK_DUMP: one file per rank with that rank's VAE-input latents
    dumps = [torch.load(dump / f"rank{r:02d}.pt") for r in range(3)]
    assert [d["rank"] for d in dumps] == [0, 1, 2]
    for d in dumps:
        assert torch.equal(d["latents"], _z().to(v.dtype).float())
