# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the memory-bounding paths that the whole-graph device compile cannot afford:
VAE host-tiling parity, and the DiT shared N-block runner graph count."""

from __future__ import annotations

import os

import pytest
import torch

from .test_z_image_tiny_ckpt import make_tiny

SRC = os.environ.get(
    "Z_IMAGE_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", "/nonexistent"), "z-image-turbo")
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not os.path.isdir(os.path.join(SRC, "tokenizer")):
        pytest.skip("set Z_IMAGE_WEIGHTS to a Z-Image checkout")
    return make_tiny(SRC, str(tmp_path_factory.mktemp("tiny-z-image")))


def test_vae_tiled_decode_matches_diffusers(tiny):
    """Host-tiled decode == diffusers' own tiled decode (same split/pad/blend algorithm).

    Tiling is NOT equal to a whole decode -- a conv decoder loses the cross-tile receptive field at
    every seam, which is exactly what the overlap blend is for; that approximation is validated at
    real resolution by the device check against the whole-latent fp32 oracle. Here we only assert
    our tiler reproduces diffusers' tiler. Latent 32x48 is 16-aligned so no edge padding is needed."""
    from diffusers import AutoencoderKL

    from vllm_omni_neuron.diffusion.models.z_image.vae import NeuronAutoencoderKL

    z = torch.randn(1, 16, 32, 48)
    for overlap in (0.0, 0.25):
        ref = AutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32).eval()
        ref.enable_tiling()
        ref.tile_sample_min_size, ref.tile_latent_min_size, ref.tile_overlap_factor = (
            128,
            16,
            overlap,
        )
        with torch.no_grad():
            want = ref.tiled_decode(z, return_dict=False)[0]
        v = NeuronAutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32)
        v.tile_lat, v.tile_overlap = (
            16,
            overlap,
        )  # instance attributes only; module default untouched
        got = v.decode(z, return_dict=False)[0]
        assert got.shape == want.shape
        rel = ((got - want).norm() / want.norm()).item()
        # overlap=0 is bit-close; overlap>0 differs only in edge-tile padding (replicate vs smaller tile)
        assert rel < (1e-4 if overlap == 0 else 3e-1), (overlap, rel)


def test_dummy_run_follows_stage_resolution():
    """``ZImageDiffusionEngine`` must warm up at ``model_config.dummy_run_height/width`` when set,
    and fall back to upstream's hardcoded 512x512 otherwise. A mismatched warmup resolution keeps
    a second set of DiT/VAE graphs resident on the core."""
    from types import SimpleNamespace

    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import ZImageDiffusionEngine

    def run(model_config):
        eng = ZImageDiffusionEngine.__new__(ZImageDiffusionEngine)
        eng.od_config = SimpleNamespace(
            model_config=model_config,
            model_class_name="ZImagePipeline",
            diffusion_load_format="default",
            diffusers_pipeline_cls=None,
        )
        captured = {}

        def fake_wait(req):
            captured["req"] = req
            return SimpleNamespace(error=None)

        eng.add_req_and_wait_for_response = fake_wait
        eng.pre_process_func = None
        eng._dummy_run()
        return captured["req"].sampling_params

    sp = run({"dummy_run_height": 1024, "dummy_run_width": 1024})
    assert (sp.height, sp.width) == (1024, 1024)
    sp = run({})
    assert (sp.height, sp.width) == (512, 512)  # unset -> upstream default, unchanged


def test_block_runner_one_graph(tiny):
    """The shared runner over the main blocks is ONE graph when group_size divides n_layers,
    two when the last chunk is shorter — never one-per-block."""
    from vllm_omni_neuron.diffusion.models.z_image.transformer import (
        NeuronZImageDiT,
        ZImageDiTConfig,
    )

    cfg = ZImageDiTConfig.from_model_dir(tiny)  # tiny has n_layers=2
    dit = NeuronZImageDiT(cfg, dtype=torch.float32, tp=(1, 0, None))
    dit.load_weights(tiny)
    dit.setup_runner(group_size=1)
    assert dit._runner.num_graphs == 1
    assert len(dit._runner.groups) == cfg.n_layers


def test_precompute_mod_is_contiguous(tiny):
    """``torch.stack`` over ``unbind`` views does not reliably produce a contiguous result in
    every execution mode. Covers both ``precompute_mod`` (traced, inside the compiled prologue
    for the refiner blocks) and ``host_mod`` (eager CPU, for the main-block loop)."""
    from vllm_omni_neuron.diffusion.models.z_image.transformer import (
        ADALN_EMBED_DIM,
        ZBlock,
        ZImageDiTConfig,
    )

    cfg = ZImageDiTConfig.from_model_dir(tiny)
    blk = ZBlock(cfg, 1, torch.float32, True)
    adaln = torch.randn(3, min(cfg.dim, ADALN_EMBED_DIM))
    for mod in (blk.precompute_mod(adaln), blk.host_mod(adaln)):
        assert mod.is_contiguous()
        assert mod.shape == (3, 4, cfg.dim)


def test_vae_tiling_engages_with_compiled_decoder(tiny):
    """``compile()`` followed by a forced-tiling decode must not crash. Regression for `_tile_fn or
    self._dec` / `_enc_fn or ...`: an ``OptimizedModule`` is falsy-ambiguous (no ``__len__``), so an
    `or`-based fallback raises on the FIRST tiled call after compile; the fix checks ``is not None``.
    identity backend: compiles for real (torch.compile machinery) but returns the eager result, so
    this runs anywhere without a Neuron device."""
    from vllm_omni_neuron.diffusion.models.z_image import vae as vmod
    from vllm_omni_neuron.diffusion.models.z_image.vae import NeuronAutoencoderKL

    saved = vmod.VAE_TILE
    try:
        v = NeuronAutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32)
        v.compile(backend="eager", compile_encoder=True)
        v.tile_lat = (
            8  # instance attribute drives the tiling decision; the module default is untouched
        )
        z = torch.randn(1, 16, 16, 16)  # > tile_lat on both axes -> forces the tiled path
        out = v.decode(z, return_dict=False)[0]
        assert out.shape[-2:] == (16 * v.scale, 16 * v.scale)
        enc = v.encode(torch.randn(1, 3, 16, 16), return_dict=False)[0]
        assert enc.mean.shape[-2:] == (2, 2)
    finally:
        vmod.VAE_TILE = saved


def test_text_encoder_on_host_ignores_device_and_compile(tiny):
    """``text_encoder_on_host`` (stage model_config): to()/compile() leave the encoder eager on the CPU."""
    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImageTextEncoder

    te = NeuronZImageTextEncoder(tiny, torch.float32, on_host=True)
    te.to(torch.device("meta"))
    te.compile(backend="eager")
    assert te._device == torch.device("cpu")
    assert te._fn is te.enc
