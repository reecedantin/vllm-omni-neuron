# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Neuron HunyuanVideo-1.5 VAE facade (tiny checkpoint)."""

from __future__ import annotations

import os

import pytest
import torch

from .test_hunyuanvideo15_tiny_ckpt import tiny_ckpt  # noqa: F401  (session fixture)


def _upstream_mask(n_frame, n_hw, dtype, device, batch_size=None):
    """diffusers' loop (copied: the class method itself is patched once the facade is imported)."""
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        mask[i, : (i // n_hw + 1) * n_hw] = 0
    return mask if batch_size is None else mask.unsqueeze(0).expand(batch_size, -1, -1)


def test_causal_mask_equals_upstream():
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import MASK_VALUE, causal_frame_mask

    for f, hw in ((1, 4), (3, 6), (5, 2)):
        ref = _upstream_mask(f, hw, torch.float32, "cpu", batch_size=2)
        ours = causal_frame_mask(f, hw, torch.float32, "cpu", batch_size=2)
        assert torch.equal(torch.isinf(ref), ours == MASK_VALUE)
        assert torch.equal(ref[~torch.isinf(ref)], ours[ours != MASK_VALUE])


def test_replicate_pad_equals_fpad():
    import torch.nn.functional as F

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import replicate_pad_3d

    x = torch.randn(1, 3, 4, 5, 6)
    for pad in ((1, 1, 1, 1, 2, 0), (0, 0, 0, 0, 2, 0), (1, 2, 0, 1, 0, 3)):
        assert torch.equal(replicate_pad_3d(x, pad), F.pad(x, pad, mode="replicate"))


@pytest.mark.parametrize(
    "t,chunk,bounds",
    [
        (1, 1, None),
        (2, 1, None),
        (5, 1, None),
        (7, 2, None),
        (6, 4, None),
        (5, 1, [0, 1, 2, 3, 4, 5]),
        (7, 2, [0, 2, 5]),
    ],
)
def test_causal_chunks_equal_whole_clip(tiny_ckpt, t, chunk, bounds):  # noqa: F811
    """Whole-clip trunk + up path in causal chunks with carried conv caches == the whole-clip decoder,
    for the up path as one graph and split into per-block segments."""
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import (
        DecoderTrunk,
        NeuronHunyuanVideo15VAE,
        UpPathChunk,
        UpPathPipeline,
        decode_causal_chunks,
    )

    v = NeuronHunyuanVideo15VAE.from_pretrained(tiny_ckpt, torch_dtype=torch.float32)
    dec = v.vae.decoder
    z = torch.randn(1, 32, t, 3, 4, generator=torch.Generator().manual_seed(t))
    if bounds is None:
        first, rest = UpPathChunk(dec, True), UpPathChunk(dec, False)
    else:
        up = UpPathPipeline(dec, bounds)
        first, rest = up.first, up.rest
    with torch.no_grad():
        want = dec(z)
        got = decode_causal_chunks(DecoderTrunk(dec), first, rest, z, chunk)
    assert got.shape == want.shape == (1, 3, (t - 1) * 4 + 1, 48, 64)
    assert ((got - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("host", ["0", "1"])
def test_decode_matches_diffusers(tiny_ckpt, monkeypatch, host):  # noqa: F811
    from diffusers import AutoencoderKLHunyuanVideo15

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import NeuronHunyuanVideo15VAE

    monkeypatch.setenv("HV15_VAE_HOST", host)
    ref = AutoencoderKLHunyuanVideo15.from_pretrained(
        os.path.join(tiny_ckpt, "vae"), torch_dtype=torch.float32
    ).eval()
    ours = NeuronHunyuanVideo15VAE.from_pretrained(tiny_ckpt, torch_dtype=torch.float32)
    z = torch.randn(1, 32, 3, 4, 5, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = ref.decode(z, return_dict=False)[0]
        got = ours.decode(z, return_dict=False)[0]  # one tile (latent smaller than a tile): exact
    assert got.shape == want.shape == (1, 3, 9, 64, 80)
    assert ((got - want).norm() / want.norm()).item() < 1e-5

    # multi-tile: fixed-shape tiles + blend; random weights -> seams, but shape and finiteness hold
    ours.tile, ours.overlap = 3, 1
    with torch.no_grad():
        tiled = ours.decode(z, return_dict=False)[0]
    assert tiled.shape == want.shape and torch.isfinite(tiled).all()
    assert ((tiled - want).norm() / want.norm()).item() < 0.5


def _sync_worker(rank, world, port, out):
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import sync_latents

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
    )
    try:
        z = torch.randn(1, 4, 3, 5, 6, generator=torch.Generator().manual_seed(0)).to(
            torch.bfloat16
        )
        if rank >= world // 2:  # half the ranks hold other latents
            z = z + 1
        os.environ["HV15_SYNC_LATENTS"] = "0"
        assert sync_latents(z, dist.group.WORLD) is z  # debug-only: off by default
        os.environ["HV15_SYNC_LATENTS"] = "1"
        got = sync_latents(z, dist.group.WORLD)
        assert got.dtype == z.dtype and got.shape == z.shape
        torch.save(got, os.path.join(out, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_sync_latents_debug_decodes_rank0_latents(tmp_path):
    """Debug hook (``HV15_SYNC_LATENTS=1``): every rank ends with rank 0's latents even when its own
    copy diverged; off by default."""
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import sync_latents

    from .test_hunyuanvideo15_pipeline import spawn_with_free_port

    z = torch.randn(1, 4, 3, 5, 6, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    assert sync_latents(z, None) is z  # single process: untouched
    world = 4
    spawn_with_free_port(_sync_worker, lambda port: (world, port, str(tmp_path)), world)
    for r in range(world):
        assert torch.equal(torch.load(tmp_path / f"r{r}.pt"), z)
