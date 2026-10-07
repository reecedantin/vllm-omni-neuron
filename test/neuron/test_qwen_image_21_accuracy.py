# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 accuracy tiers on one NeuronCore (docs/model-dev/accuracy-evaluation-debugging.md).

1. Component three-way: each compiled graph (text encoder, DiT prefix, DiT target, VAE decode)
   as fp32 CPU / bf16 CPU / bf16 Neuron, via ``vllm_neuron.accuracy.testing.assert_close_three_way``.
2. Single-step pipeline: one denoising step on Neuron vs the diffusers ``QwenImage21Pipeline``
   CPU fp32 latent.
3. End-to-end regression: a full multi-step image on Neuron vs the CPU fp32 golden image of the
   same seed, per-image SSIM. Per-step error compounds, so this tier is the one that catches drift.

Runs on the random-weight tiny checkpoint by default (``QWEN_IMAGE21_TINY``, from
``test/unit/test_qwen_image_tiny.py``); skips without a Neuron device or a checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

DEVICE = os.environ.get("QWEN21_ACC_DEVICE", "neuron:0")  # "cpu" = dry run of the test logic


def _neuron_available() -> bool:
    if DEVICE == "cpu":
        return True
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

PROMPT = "A red fox reading a newspaper on a park bench, watercolor"
H = W = 256
SEED = 0


@pytest.fixture(scope="module")
def model_dir():
    path = os.environ.get("QWEN_IMAGE21_TINY", "")
    if not os.path.isdir(os.path.join(path, "transformer")):
        pytest.skip("set QWEN_IMAGE21_TINY to a checkpoint from test/unit/test_qwen_image_tiny.py")
    return path


def _pipes(model_dir):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    out = {}
    for name, dtype in (
        ("cpu32", torch.float32),
        ("cpu16", torch.bfloat16),
        ("dev", torch.bfloat16),
    ):
        p = NeuronQwenImage21Pipeline(model_path=model_dir, dtype=dtype)
        p.load_weights()
        if name == "dev" and DEVICE != "cpu":
            p.to(torch.device(DEVICE))
            p.compile(backend=get_compile_backend_name())
        out[name] = p
    return out


@pytest.fixture(scope="module")
def pipes(vllm_single_rank, model_dir):
    return _pipes(model_dir)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM of two images in [0, 1] (data range 1)."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2)))


# -- tier 1 ------------------------------------------------------------------------------
def test_tier1_text_encoder_three_way(pipes):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    outs = {k: p.encode_prompt(PROMPT)[0].float() for k, p in pipes.items()}
    assert_close_three_way(outs["cpu32"], outs["cpu16"], outs["dev"], name="text_encoder")


def test_tier1_dit_three_way(pipes):
    """DiT prefix + one target call, identical inputs for all three (built from the fp32 run)."""
    from vllm_neuron.accuracy.testing import assert_close_three_way

    p32 = pipes["cpu32"]
    emb, mask = p32.encode_prompt(PROMPT)
    h, w = H // 16, W // 16
    lat = torch.randn(1, h * w, p32.latent_channels, generator=torch.Generator().manual_seed(SEED))
    outs = {}
    for k, p in pipes.items():
        branch = p._prefix_kv(emb.to(p.dtype), mask, h * w // 4, [(1, h, w)])
        outs[k] = p._velocity(lat.to(p.dtype), torch.tensor(500.0), branch).float()
    assert_close_three_way(outs["cpu32"], outs["cpu16"], outs["dev"], name="dit_target")


def test_tier1_vae_three_way(pipes):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    c = pipes["cpu32"].latent_channels
    z = torch.randn(1, c, 1, H // 16, W // 16, generator=torch.Generator().manual_seed(SEED))
    outs = {
        k: p.vae.decode(z.to(p.vae.dtype), return_dict=False)[0].float() for k, p in pipes.items()
    }
    assert_close_three_way(outs["cpu32"], outs["cpu16"], outs["dev"], name="vae_decode")


# -- tier 2 ------------------------------------------------------------------------------
def test_tier2_single_step_vs_diffusers(pipes, model_dir):
    """One denoising step on Neuron vs diffusers' QwenImage21Pipeline (CPU fp32), same noise.

    ``sigmas=[0.5]``: a lone sigma of 1.0 (the default 1-step schedule) makes the scheduler's
    ``shift_terminal`` stretch divide 0 by 0 in diffusers and here alike."""
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
    ref = QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler"),
        vae=AutoencoderKLQwenImage21.from_pretrained(model_dir, subfolder="vae", torch_dtype=f32),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, subfolder="text_encoder", dtype=f32
        ),
        processor=Qwen3VLProcessor.from_pretrained(model_dir, subfolder="processor"),
        transformer=QwenImage21Transformer2DModel.from_pretrained(
            model_dir, subfolder="transformer", torch_dtype=f32
        ),
    )
    c = pipes["cpu32"].latent_channels
    noise = torch.randn((1, 1, c, H // 16, W // 16), generator=torch.Generator().manual_seed(SEED))
    packed = noise.view(1, c, -1).transpose(1, 2)
    exp = ref(
        PROMPT,
        height=H,
        width=W,
        num_inference_steps=1,
        sigmas=[0.5],
        latents=packed,
        output_type="latent",
    ).images
    band = (
        pipes["cpu16"]
        .generate(
            PROMPT,
            height=H,
            width=W,
            num_inference_steps=1,
            sigmas=[0.5],
            latents=packed,
            output_type="latent",
        )
        .float()
    )
    dev = (
        pipes["dev"]
        .generate(
            PROMPT,
            height=H,
            width=W,
            num_inference_steps=1,
            sigmas=[0.5],
            latents=packed,
            output_type="latent",
        )
        .float()
    )
    rel = lambda a: ((a - exp.float()).norm() / exp.float().norm()).item()  # noqa: E731
    assert rel(dev) <= 2.0 * rel(band) + 0.005, (rel(dev), rel(band))


# -- tier 3 ------------------------------------------------------------------------------
def test_tier3_e2e_golden_ssim(pipes):
    """Full 8-step image vs the CPU fp32 golden of the same seed; SSIM must be within the
    CPU-bf16 run's own SSIM minus a small margin, and deterministic across reruns."""
    kw = dict(height=H, width=W, num_inference_steps=8, seed=SEED)
    to01 = lambda x: (x.float().clamp(-1, 1) / 2 + 0.5)[0].permute(1, 2, 0).numpy()  # noqa: E731
    golden = to01(pipes["cpu32"].generate(PROMPT, **kw))
    band = _ssim(to01(pipes["cpu16"].generate(PROMPT, **kw)), golden)
    a = pipes["dev"].generate(PROMPT, **kw)
    b = pipes["dev"].generate(PROMPT, **kw)
    assert torch.equal(a, b), "device e2e is not deterministic"
    dev = _ssim(to01(a), golden)
    assert dev >= min(band - 0.02, 0.98), (dev, band)
