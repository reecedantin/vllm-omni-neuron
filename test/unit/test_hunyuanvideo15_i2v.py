# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Neuron HunyuanVideo-1.5 image-to-video pipeline vs diffusers'
``HunyuanVideo15ImageToVideoPipeline`` on tiny random-weight I2V checkpoints (fp32): SigLIP image tokens,
first-frame VAE condition, the full denoising loop with CFG (guidance 6) and CFG-distilled (guidance 1,
one DiT forward per step), and the registry entry."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_hunyuanvideo15_pipeline import PROMPT, single_rank  # noqa: E402,F401  (module fixture)
from .test_hunyuanvideo15_tiny_ckpt import tiny_i2v_ckpt, tiny_i2v_distilled_ckpt  # noqa: E402,F401

H, W, F, STEPS = 64, 96, 9, 3


def _image():
    from PIL import Image

    g = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 255 / W, yy * 255 / H, 128 + 60 * np.sin(xx / 7.0)], -1) + g.normal(
        0, 10, (H, W, 3)
    )
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _ours(ckpt):
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15 import NeuronHunyuanVideo15I2VPipeline

    od = SimpleNamespace(
        model=ckpt,
        dtype=torch.float32,
        model_config={"vae_dtype": "float32"},
        flow_shift=None,
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = NeuronHunyuanVideo15I2VPipeline(od_config=od)
    pipe.load_weights()
    return pipe


def _run_both(ckpt):
    from diffusers import HunyuanVideo15ImageToVideoPipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ref = HunyuanVideo15ImageToVideoPipeline.from_pretrained(ckpt, torch_dtype=torch.float32)
    # diffusers picks the output size from the image with its resolution buckets (target_size); keep it tiny
    # and serve the same size explicitly (vLLM-Omni requests carry height/width)
    ref.target_size = 64
    image = _image()
    h, w = ref.video_processor.calculate_default_height_width(
        image.size[1], image.size[0], ref.target_size
    )
    ours = _ours(ckpt)
    ours._target_size = ref.target_size
    lat0 = torch.randn(
        1, 32, (F - 1) // 4 + 1, h // 16, w // 16, generator=torch.Generator().manual_seed(7)
    )

    calls = []
    orig = ref.transformer.forward

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    ref.transformer.forward = counting
    with torch.no_grad():
        want = ref(
            image=image,
            prompt=PROMPT,
            num_frames=F,
            num_inference_steps=STEPS,
            latents=lat0.clone(),
            output_type="latent",
        ).frames
    sp = OmniDiffusionSamplingParams(
        num_frames=F, num_inference_steps=STEPS, latents=lat0.clone()
    )  # size from bucket
    req = _Batch({"prompt": PROMPT, "multi_modal_data": {"image": image}}, sp)
    n0 = ours.transformer.stats["calls"]
    with torch.no_grad():
        got = ours.forward(req, output_type="latent").output
    assert (sp.height, sp.width) == (h, w)
    return ref, ours, image, got, want, len(calls), ours.transformer.stats["calls"] - n0


class _Batch:
    """Like the served engine's request batch: ``prompts`` is rebuilt on every access."""

    def __init__(self, prompt, sp):
        self._prompt, self.sampling_params = prompt, sp

    @property
    def prompts(self):
        return [self._prompt]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_i2v_conditioning_matches_diffusers(tiny_i2v_ckpt, single_rank):  # noqa: F811
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    ref = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        tiny_i2v_ckpt, torch_dtype=torch.float32
    )
    ours = _ours(tiny_i2v_ckpt)
    image = _image()
    with torch.no_grad():
        e_ref = ref.encode_image(image, 1, torch.device("cpu"), torch.float32)
        e_ours = ours._get_image_embeds(image, torch.device("cpu"))
        lat = torch.zeros(1, 32, 3, H // 16, W // 16)
        c_ref, m_ref = ref.prepare_cond_latents_and_mask(
            lat, image, 1, H, W, torch.float32, torch.device("cpu")
        )
        c_ours, m_ours = ours.prepare_cond_latents_and_mask(
            lat, image, H, W, torch.float32, torch.device("cpu")
        )
    assert e_ours.shape == (1, 729, 1152)
    torch.testing.assert_close(e_ours.float(), e_ref.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c_ours, c_ref, rtol=1e-5, atol=1e-5)
    assert (
        torch.equal(m_ours, m_ref)
        and c_ours[:, :, 1:].abs().max() == 0
        and c_ours[:, :, 0].abs().max() > 0
    )


def test_i2v_pipeline_matches_diffusers_cfg(tiny_i2v_ckpt, single_rank):  # noqa: F811
    ref, ours, _, got, want, n_ref, n_ours = _run_both(tiny_i2v_ckpt)
    assert ours._default_guidance == 6.0
    rel = _rel(got, want)
    print(f"[i2v-parity cfg] latents rel-L2 {rel:.3e}, DiT calls ref {n_ref} ours {n_ours}")
    assert rel < 1e-5, rel
    assert n_ref == n_ours == 2 * STEPS


def test_i2v_distilled_one_forward_per_step(tiny_i2v_distilled_ckpt, single_rank):  # noqa: F811
    ref, ours, _, got, want, n_ref, n_ours = _run_both(tiny_i2v_distilled_ckpt)
    assert ours._default_guidance == 1.0 and ours.scheduler.config.shift == 7.0
    rel = _rel(got, want)
    print(f"[i2v-parity distilled] latents rel-L2 {rel:.3e}, DiT calls ref {n_ref} ours {n_ours}")
    assert rel < 1e-5, rel
    assert n_ref == n_ours == STEPS


def test_i2v_registered():
    from vllm_omni_neuron.diffusion.models import hunyuanvideo15 as pkg

    arch = {e["model_arch"]: e for e in pkg.PIPELINE_REGISTRY}
    e = arch["HunyuanVideo15ImageToVideoPipeline"]
    for key in ("class_name", "pre_process_func_name", "post_process_func_name"):
        assert hasattr(pkg, e[key]), e[key]
    assert "HunyuanVideo15Pipeline" in arch
