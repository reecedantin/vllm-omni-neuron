# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 on CPU (fp32, tiny random-weight checkpoint): text encoder vs HF Qwen3-VL,
VAE decode vs upstream, and the whole Neuron pipeline vs diffusers' ``QwenImage21Pipeline``.

The tiny checkpoint borrows the tokenizer/processor of a real Qwen-Image 2.1 checkout, found via
``QWEN_IMAGE21_WEIGHTS`` (default ``$WEIGHTS/qwen-image-21``); without one these tests skip.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

REAL = os.environ.get(
    "QWEN_IMAGE21_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", ""), "qwen-image-21")
)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    if not os.path.isdir(os.path.join(REAL, "processor")):
        pytest.skip(
            "set QWEN_IMAGE21_WEIGHTS to a Qwen-Image-2.1 checkout (for its processor files)"
        )
    from .test_qwen_image_tiny import build

    return build(str(tmp_path_factory.mktemp("qwen21tiny")), processor_from=REAL)


def _reference_pipeline(tiny_dir):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
    )
    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.pipeline_qwenimage21 import (
        QwenImage21Pipeline,
    )
    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    f32 = torch.float32
    return QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(tiny_dir, subfolder="scheduler"),
        vae=AutoencoderKLQwenImage21.from_pretrained(tiny_dir, subfolder="vae", torch_dtype=f32),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            tiny_dir, subfolder="text_encoder", dtype=f32
        ),
        processor=Qwen3VLProcessor.from_pretrained(tiny_dir, subfolder="processor"),
        transformer=QwenImage21Transformer2DModel.from_pretrained(
            tiny_dir, subfolder="transformer", torch_dtype=f32
        ),
    )


@pytest.fixture(scope="module")
def pipes(vllm_single_rank, tiny_dir):
    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    ours = NeuronQwenImage21Pipeline(model_path=tiny_dir, dtype=torch.float32)
    ours.load_weights()
    return ours, _reference_pipeline(tiny_dir)


def test_prompt_embeds_match(pipes):
    ours, ref = pipes
    prompt = "A red fox reading a newspaper on a park bench, watercolor"
    emb, mask = ours.encode_prompt(prompt)
    ref_emb, ref_mask, ref_img = ref.encode_prompt(prompt, device="cpu")
    assert ref_mask is None  # unpadded single prompt
    assert emb.shape == ref_emb.shape, (emb.shape, ref_emb.shape)
    assert torch.equal(mask, ref_img[0].bool())
    assert _rel(emb, ref_emb) < 1e-4, _rel(emb, ref_emb)


def test_vae_decode_matches(pipes):
    ours, ref = pipes
    z = torch.randn(1, ours.latent_channels, 1, 4, 6)
    with torch.no_grad():
        a = ours.vae.decode(z, return_dict=False)[0]
        b = ref.vae.decode(z, return_dict=False)[0]
    assert a.shape == b.shape == (1, 4, 1, 64, 96)
    assert _rel(a, b) < 1e-5, _rel(a, b)


def test_vae_encode_matches(pipes):
    ours, ref = pipes
    x = torch.rand(1, 4, 1, 64, 64) * 2 - 1
    with torch.no_grad():
        a = ours.vae.encode(x).latent_dist.mode()
        b = ref.vae.encode(x).latent_dist.mode()
    assert _rel(a, b) < 1e-5, _rel(a, b)


def test_vae_tiling_blend_is_exact_for_local_decoders(pipes):
    """With a pointwise 'decoder' every tile agrees on its overlap, so the blend must be exact."""
    ours, _ = pipes
    vae = ours.vae

    def pointwise(z):
        return torch.nn.functional.interpolate(z[:, :4, 0].float(), scale_factor=16)[:, :, None]

    saved = vae._decode_one
    vae._decode_one = pointwise
    try:
        z = torch.randn(1, ours.latent_channels, 1, 37, 23)
        tiled = vae._decode_tiled(z, 16, 4).float()
    finally:
        vae._decode_one = saved
    assert tiled.shape == (1, 4, 1, 37 * 16, 23 * 16)
    assert torch.allclose(tiled, pointwise(z), atol=1e-2), (tiled - pointwise(z)).abs().max()


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_pipeline_matches_diffusers(pipes, cfg_scale):
    ours, ref = pipes
    prompt, neg = "a lighthouse on a cliff at dawn", "blurry"
    h, w, steps, seed = 256, 192, 4, 7
    noise = torch.randn(
        (1, 1, ours.latent_channels, h // 16, w // 16),
        generator=torch.Generator().manual_seed(seed),
    )
    packed = noise.view(1, ours.latent_channels, -1).transpose(1, 2)
    out = ours.generate(
        prompt,
        height=h,
        width=w,
        num_inference_steps=steps,
        latents=packed,
        negative_prompt=neg,
        true_cfg_scale=cfg_scale,
    )
    exp = ref(
        prompt,
        negative_prompt=neg,
        true_cfg_scale=cfg_scale,
        height=h,
        width=w,
        num_inference_steps=steps,
        latents=packed,
        output_type="pt",
    ).images
    # diffusers post-processes to [0, 1]; ours is the raw VAE output in [-1, 1]
    assert out.shape == exp.shape, (out.shape, exp.shape)
    assert _rel(out / 2 + 0.5, exp) < 1e-4, _rel(out / 2 + 0.5, exp)
