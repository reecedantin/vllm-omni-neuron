# SPDX-License-Identifier: Apache-2.0
"""Device VAE decode parity for the Cosmos3-Edge (Wan2.2-TI2V-5B) VAE on NeuronCore-v2.

Plugin ``NeuronAutoencoderKLWan`` compiled on neuron:0 vs diffusers ``AutoencoderKLWan`` on CPU
(fp32) for one latent frame (T2I) at ``COSMOS3_VAE_TEST_LATENT`` (default 40 -> 640x640).
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

from .test_cosmos3_edge_und_device import _neuron_available  # noqa: F401

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float().clamp(-1, 1) - b.float().clamp(-1, 1)).pow(2).mean().item()
    return 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def test_vae_decode_single_frame(edge_weights):
    from diffusers import AutoencoderKLWan
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    n = int(os.environ.get("COSMOS3_VAE_TEST_LATENT", "40"))
    torch.manual_seed(0)
    z = torch.randn(1, 48, 1, n, n)

    ref_vae = AutoencoderKLWan.from_pretrained(edge_weights, subfolder="vae", torch_dtype=torch.float32).eval()
    t0 = time.time()
    with torch.no_grad():
        ref = ref_vae.decode(z).sample
    cpu_s = time.time() - t0

    dev = torch.device("neuron", 0)
    vae = NeuronAutoencoderKLWan.from_pretrained(edge_weights, subfolder="vae", torch_dtype=torch.bfloat16).eval()
    vae.to(dev)
    vae.compile(
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": "cosmos3_edge_vae",
            "compiler_args": ["--model-type=unet-inference", "--auto-cast=none",
                              "--internal-max-instruction-limit=15000000", "-O1"],
        },
    )
    zd = z.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(zd).sample.cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = vae.decode(zd).sample.cpu()
    warm = time.time() - t0

    psnr = _psnr(out, ref)
    report = {"latent": n, "shape": list(out.shape), "psnr_db": psnr, "first_s": first, "warm_s": warm, "cpu_fp32_s": cpu_s,
              "deterministic": bool(torch.equal(out, out2))}
    with open(os.path.join(os.environ.get("COSMOS3_TEST_OUT", "/tmp"), f"vae_decode_{n}.json"), "w") as f:
        json.dump(report, f, indent=1)
    assert tuple(out.shape) == tuple(ref.shape)
    assert psnr > 40.0, report
    assert report["deterministic"]


def test_vae_encode_on_device(edge_weights):
    """Encoder graph (incl. Wan2.2 patchify) on neuron:0 vs diffusers CPU fp32 latent mean."""
    from diffusers import AutoencoderKLWan
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    hw = os.environ.get("COSMOS3_VAE_TEST_ENC_HW", "640,640")
    frames = int(os.environ.get("COSMOS3_VAE_TEST_ENC_FRAMES", "1"))
    h, w = (int(v) for v in hw.split(","))
    torch.manual_seed(0)
    # smooth synthetic image/video in [-1, 1] (random noise is a pathological VAE input)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
    x = torch.stack([base * (1 - 0.05 * f) for f in range(frames)], dim=1)[None].clamp(-1, 1)

    ref_vae = AutoencoderKLWan.from_pretrained(edge_weights, subfolder="vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = ref_vae.encode(x).latent_dist.mean

    dev = torch.device("neuron", 0)
    vae = NeuronAutoencoderKLWan.from_pretrained(edge_weights, subfolder="vae", torch_dtype=torch.bfloat16).eval()
    vae.to(dev)
    vae.compile(
        backend=get_compile_backend_name(), fullgraph=True, dynamic=False, compile_encoder=True,
        options={"model_name": "cosmos3_edge_vae", "compiler_args": [
            "--model-type=unet-inference", "--auto-cast=none", "--internal-max-instruction-limit=15000000", "-O1"]},
    )
    xd = x.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        h_dev = vae._encode(xd).cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        h_dev2 = vae._encode(xd).cpu()
    warm = time.time() - t0
    mean = h_dev[:, : h_dev.shape[1] // 2].float()
    rel = ((mean - ref).norm() / ref.norm()).item()
    report = {"hw": [h, w], "frames": frames, "latent_shape": list(mean.shape), "rel_err": rel,
              "first_s": first, "warm_s": warm, "deterministic": bool(torch.equal(h_dev, h_dev2))}
    with open(os.path.join(os.environ.get("COSMOS3_TEST_OUT", "/tmp"), f"vae_encode_{h}x{w}x{frames}.json"), "w") as f:
        json.dump(report, f, indent=1)
    assert tuple(mean.shape) == tuple(ref.shape), report
    assert rel < 0.02, report
    assert report["deterministic"], report
