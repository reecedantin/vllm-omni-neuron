# SPDX-License-Identifier: Apache-2.0
"""Neuron VAE wrappers reproduce the diffusers VAE decode on CPU (tiling ON, the released default), with the
structure checkpoint.

Device parity (vs the CPU reference, since Neuron VAE decode is not bit-reproducible) is checked by
``test/neuron/test_minimax_h3_dit_device.py``; here only the wrapper plumbing.

Tiling matters here beyond memory: disabling it (tried during the M2 device-decode bisection) measurably streaks
the video at 384x640 (fine vertical texture artifacts in fur/water), reproduced entirely on CPU with no device
involved -- plain fp32 diffusers decode, tiling off, on the SAME latents the device run used. 256p never showed it
because diffusers' tile split returns a single whole-frame tile below ``tile_sample_min_height/width`` (256), so
tiling on/off is a no-op exactly at that canvas. See ``examples/minimax_h3/eval/vae_compare.py`` for the bisection
tool and ``PROGRESS.md`` 2026-10-03 for the measured numbers (striping-ratio metric, same code path, CPU vs device).
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_video_vae_wrapper_matches_reference(h3_tiny):
    """The wrapper enforces the proven FastH3 precision placement: 36 blocks bf16 (fp32 norms inside), but
    proj_in / norm_out / proj_out upcast to fp32. Tiling stays ON (the released default; see module docstring)."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import NeuronMiniMaxH3VideoVAE

    ref32 = AutoencoderKLMiniMaxH3.from_pretrained(
        h3_tiny, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    ref = AutoencoderKLMiniMaxH3.from_pretrained(
        h3_tiny, subfolder="vae", torch_dtype=torch.bfloat16
    ).eval()
    assert (
        ref.decoder.proj_in.weight.dtype == torch.bfloat16
    )  # the plain bf16 load (before the wrapper upcasts)
    z = torch.randn(1, ref.config.latent_channels, 4, 4, 6)  # latent frames x h x w
    with torch.no_grad():
        want32 = ref32.decode(z, return_dict=False)[0]  # tiling on (released default), fp32 oracle
    wrap = NeuronMiniMaxH3VideoVAE(ref, torch.device("cpu"), torch.bfloat16)
    assert wrap.dtype == torch.float32  # decoder input (proj_in) is fp32
    assert wrap.block_dtype == torch.bfloat16  # the 36 blocks stay bf16
    assert ref.decoder.proj_in.weight.dtype == torch.float32  # wrapper upcast it
    assert ref.decoder.proj_out.weight.dtype == torch.float32  # and the pixel-block projection
    assert (
        ref.decoder.transformer_blocks[0].ff.net[0].proj.weight.dtype == torch.bfloat16
    )  # block FF stays bf16
    assert wrap.vae.use_tiling  # NOT disabled -- see module docstring
    with torch.no_grad():
        got = wrap.decode(z, return_dict=False)[0]
    assert got.shape == want32.shape, (got.shape, want32.shape)
    assert _rel(got, want32) < 0.1, _rel(got, want32)  # bf16 blocks vs fp32: a loose bound


def test_disabling_tiling_splits_differently(h3_tiny):
    """Regression guard for the M2 bug (video streaked at 384x640 with tiling off, confirmed on CPU with the real
    weights -- see PROGRESS.md 2026-10-03 and examples/minimax_h3/eval/vae_compare.py). A random tiny checkpoint
    cannot reproduce the artifact itself (it has no learned positional sensitivity to exploit), so this test pins
    the structural fact the bug rests on: above the tile threshold, diffusers actually SPLITS into >1 tile, so
    tiling on/off take genuinely different code paths (full-frame attention vs per-tile)."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

    ref = AutoencoderKLMiniMaxH3.from_pretrained(
        h3_tiny, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    ref.tile_sample_min_height = ref.tile_sample_min_width = 32
    ref.tile_sample_min_overlap_height = ref.tile_sample_min_overlap_width = 16
    starts, lengths, overlaps = ref._split_tiles(
        8 * 16, ref.tile_sample_min_height, ref.tile_sample_min_overlap_height
    )
    assert len(starts) > 1, "fixture must exercise the >1-tile path, or this test proves nothing"


def test_tiling_is_a_noop_below_the_tile_threshold(h3_tiny):
    """At or below tile_sample_min_height/width (256 released default), diffusers' tile split returns a single
    whole-frame tile, so tiling on vs off must be IDENTICAL -- this is why 256p never showed the M2 streaking."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

    ref = AutoencoderKLMiniMaxH3.from_pretrained(
        h3_tiny, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    assert ref._split_tiles(4 * 16, ref.tile_sample_min_height, ref.tile_sample_min_overlap_height)[
        0
    ] == [0]
    z = torch.randn(1, ref.config.latent_channels, 4, 4, 6)
    with torch.no_grad():
        tiled = ref.decode(z, return_dict=False)[0]
    ref.use_tiling = False
    with torch.no_grad():
        untiled = ref.decode(z, return_dict=False)[0]
    assert torch.equal(tiled, untiled)


def _tile_parallel_worker(rank, world, port, weights, out_path):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import NeuronMiniMaxH3VideoVAE

    ctx = set_current_vllm_config(
        VllmConfig()
    )  # keep a reference: the context must outlive this statement
    ctx.__enter__()
    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(world, 1)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        weights, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    vae.tile_sample_min_height = vae.tile_sample_min_width = (
        32  # force several tiles on the tiny canvas
    )
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 16
    wrap = NeuronMiniMaxH3VideoVAE(vae, torch.device("cpu"), torch.float32)
    # 12 latent frames: several temporal chunks with cross-faded frames, and several spatial tiles
    z = torch.randn(
        1, vae.config.latent_channels, 12, 8, 10, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        serial = wrap.decode(z, return_dict=False)[0]
        n_serial = wrap.tile_calls
        tp = get_tp_group()
        os.environ["MINIMAX_H3_VAE_GATHER_DTYPE"] = "float32"
        par = wrap.decode(z, return_dict=False, tp_group=tp)[0]  # batched calls, /dev/shm transport
        batch = wrap.timing.get("batch")
        os.environ["MINIMAX_H3_VAE_SHM"] = "0"
        par_gloo = wrap.decode(z, return_dict=False, tp_group=tp)[0]
        del (
            os.environ["MINIMAX_H3_VAE_SHM"],
            os.environ["MINIMAX_H3_VAE_GATHER_DTYPE"],
        )  # the default: 16-bit
        par16 = wrap.decode(z, return_dict=False, tp_group=tp)[0]
        unit = lambda v: v * 0.25 + 0.5  # noqa: E731  (a per-channel affine map like the pipeline's de-normalise)
        os.environ["MINIMAX_H3_VAE_COMPOSE"] = "0"
        par8 = wrap.decode(z, return_dict=False, tp_group=tp, to_unit=unit)[
            0
        ]  # uint8 tiles, rank-0 blend
        applied = wrap.applied_unit
        del os.environ["MINIMAX_H3_VAE_COMPOSE"]
        comp = wrap.decode(z, return_dict=False, tp_group=tp, to_unit=unit)[
            0
        ]  # every rank blends its frames
        comp_u8 = wrap.output_uint8 and wrap.timing.get("transport") == "uint8-compose"
        plan = wrap._compose_plan(z, n_serial)
        n_chunks = len({c for f in plan["frames"] for c, _, _ in f})
    if rank == 0:
        torch.save(
            {
                "serial": serial,
                "par": par,
                "par_gloo": par_gloo,
                "par16": par16,
                "par8": par8,
                "applied": applied,
                "comp": comp,
                "comp_u8": comp_u8,
                "n_chunks": n_chunks,
                "n_serial": n_serial,
                "batch": batch,
                "n_rank0": wrap.tile_calls,
            },
            out_path,
        )
    else:
        assert par is None and par16 is None and par8 is None and comp is None


def test_tile_parallel_matches_serial(h3_tiny, tmp_path):
    """Tile-parallel decode (work items dealt over 2 gloo ranks as one batched decoder call each, gathered through
    /dev/shm or gloo, blended on rank 0) matches the serial decode: to fp32 rounding with an fp32 gather (batched vs
    single-tile matmuls), within one 8-bit level with the default fp16 gather, and within one level of the 8-bit
    pixels when each rank ships uint8 pixels (``to_unit``)."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "tp.pt"
    mp.spawn(_tile_parallel_worker, args=(2, port, h3_tiny, str(out)), nprocs=2, join=True)
    r = torch.load(out)
    serial = r["serial"]
    peak = serial.abs().max().item()
    assert r["n_serial"] > 2  # the fixture really exercises several work items
    assert r["batch"] == -(-r["n_serial"] // 2)  # rank 0's share
    assert r["n_rank0"] == -(
        -r["batch"] // 8
    )  # in batched calls of at most MINIMAX_H3_VAE_BATCH (8)
    assert torch.equal(r["par"], r["par_gloo"])  # the two transports carry the same bytes
    assert (r["par"] - serial).abs().max().item() < 1e-5 * peak
    err = (r["par16"] - serial).abs().max().item()
    assert err < 2e-3 * peak, err  # fp16 gather: well below one 8-bit output level
    assert r["applied"]
    want = (serial * 0.25 + 0.5).clamp(0, 1)
    err8 = (r["par8"] - want).abs().max().item()
    assert err8 <= 1.0 / 255 + 1e-6, err8  # uint8 tiles: within one output level
    # composed on every rank from the blend weights (diffusers' stitch + cross-fade as a linear map), uint8 out
    assert r["comp_u8"] and r["comp"].dtype == torch.uint8 and r["n_chunks"] > 1
    ref8 = (
        r["par8"] * 255.0
    ).round()  # the rank-0 blend of the same uint8 tiles, rounded the same way
    d = (r["comp"].float() - ref8).abs()
    assert d.max().item() <= 1 and (d > 0).float().mean().item() < 1e-3, (
        d.max(),
        (d > 0).float().mean(),
    )
    assert ((r["comp"].float() / 255.0) - want).abs().max().item() <= 1.5 / 255


def test_audio_vae_wrapper_matches_reference(h3_tiny):
    """Also checks the wrapper forces fp32 regardless of the requested dtype (diffusers' own note: bf16 measures
    ~20 dB quieter on this VAE)."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import NeuronMiniMaxH3AudioVAE

    ref = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        h3_tiny, subfolder="audio_vae", torch_dtype=torch.float32
    ).eval()
    z = torch.randn(2, ref.config.latent_channels, 44)
    with torch.no_grad():
        want = ref.decode(z, return_dict=False)[0]
    wrap = NeuronMiniMaxH3AudioVAE(ref, torch.device("cpu"), torch.bfloat16)  # requested bf16
    assert wrap.dtype == torch.float32  # ignored
    with torch.no_grad():
        got = wrap.decode(z, return_dict=False)[0]
    assert got.shape == want.shape
    snr = 10 * torch.log10(
        want.pow(2).mean() / (got - want).pow(2).mean()
    )  # folded activations: ends ~1e-5
    assert snr > 60, snr


def test_audio_windows_cover_and_match(h3_tiny):
    """Windowed audio decode (one window length, edge windows shifted inward, weight norm folded) stitches back to
    the one-shot decode: every latent frame is kept exactly once, and with enough halo the waveform matches."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import (
        NeuronMiniMaxH3AudioVAE,
        audio_windows,
    )

    for total, chunk, halo in ((207, 26, 16), (44, 26, 16), (30, 8, 4)):
        wins = audio_windows(total, chunk, halo)
        w = min(total, chunk + 2 * halo)
        kept = [f for _, s0, s1 in wins for f in range(s0, s1)]
        assert kept == list(range(total))
        assert all(0 <= a0 <= s0 and s1 <= a0 + w <= total for a0, s0, s1 in wins)
    ref = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        h3_tiny, subfolder="audio_vae", torch_dtype=torch.float32
    ).eval()
    z = torch.randn(2, ref.config.latent_channels, 120, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = ref.decode(z, return_dict=False)[0]
    wrap = NeuronMiniMaxH3AudioVAE(ref, torch.device("cpu"))
    wins = audio_windows(120, 26, 16)
    with torch.no_grad():
        got = torch.cat([wrap.decode_window(z, x, 58) for x in wins], dim=-1)
    assert len(wins) == 5 and got.shape == want.shape
    snr = 10 * torch.log10(want.pow(2).mean() / (got - want).pow(2).mean())
    assert snr > 60, snr


def test_cte_attention_processor_layout(h3_tiny, monkeypatch):
    """The decoder's attention_cte processor, with the kernel gate forced on (on CPU ``_nki_attend`` runs its torch
    fallback in the kernel's d-major layout), reproduces the reference processor: q/k norm, partial RoPE and the
    layout round trip."""
    import vllm_omni_neuron.nc_generation as ncg
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import CTEVideoAttnProcessor

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        h3_tiny, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    z = torch.randn(
        1, vae.config.latent_channels, 2, 4, 4, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        ref = vae.decoder(vae.post_quant_conv(z))
        monkeypatch.setattr(ncg, "use_nki_kernels", lambda t: True)
        for blk in vae.decoder.transformer_blocks:
            blk.attn.set_processor(CTEVideoAttnProcessor())
        out = vae.decoder(vae.post_quant_conv(z))
    assert ((out - ref).norm() / ref.norm()).item() < 1e-5


@pytest.mark.parametrize("rpad", ["0", "1"])
def test_folded_activation_matches(h3_tiny, monkeypatch, rpad):
    """The time-folded alias-free activation equals the original on interior samples and stays close at the ends
    (default F.pad replicate, and the opt-in gather-free ``_rpad``)."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import fold_long_activations

    monkeypatch.setenv(
        "MINIMAX_H3_AUDIO_FOLD_ROWS", "64"
    )  # force folding on the tiny decoder's widths
    monkeypatch.setenv("MINIMAX_H3_AUDIO_RPAD", rpad)
    load = lambda: AutoencoderKLMiniMaxH3Audio.from_pretrained(  # noqa: E731
        h3_tiny, subfolder="audio_vae", torch_dtype=torch.float32
    ).eval()
    ref, fold = load(), load()
    assert fold_long_activations(fold.decoder) > 0
    z = torch.randn(2, ref.config.latent_channels, 60, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = ref.decode(z, return_dict=False)[0]
        got = fold.decode(z, return_dict=False)[0]
    snr = 10 * torch.log10(want.pow(2).mean() / (got - want).pow(2).mean())
    assert got.shape == want.shape and snr > 60, snr


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
def test_video_shm_handoff_roundtrip():
    """The composed clip file handed to the engine as vLLM-Omni's shared-memory handle comes back bit-exact and the
    file is gone afterwards (the engine owns it)."""
    import uuid

    import numpy as np

    ipc = pytest.importorskip("vllm_omni.diffusion.ipc")
    from vllm_omni_neuron.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        NeuronMiniMaxH3Pipeline,
    )

    shape = (1, 3, 5, 16, 24)
    name = f"minimax_h3_vae_test_{uuid.uuid4().hex}_clip.bin"
    want = torch.randint(0, 256, shape, dtype=torch.uint8)
    want.numpy().tofile(f"/dev/shm/{name}")
    video = torch.from_numpy(np.memmap(f"/dev/shm/{name}", dtype=np.uint8, mode="c", shape=shape))
    handle = NeuronMiniMaxH3Pipeline._shm_handle(video, name)
    got = ipc._unpack_if_shm_handle((handle, None))[0]
    assert got.dtype == torch.uint8 and torch.equal(got, want)
    assert not os.path.exists(f"/dev/shm/{name}")
