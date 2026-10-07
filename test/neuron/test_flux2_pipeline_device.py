# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-dev accuracy tiers 2 and 3 on Neuron: the served pipeline (real weights), TP=8 unless
``FLUX2_STAGE_CONFIG`` names another stage config.

Multi-rank Neuron execution is only supported inside a vLLM-Omni stage, so each device run goes through
``examples/flux2/run.py`` in a subprocess, and the CPU references run in a CPU-only subprocess.

* **Tier 2, single step:** one denoising step at 256 px. Text encoder plus one DiT call at TP=8, the
  scheduler, and latent unpacking; the VAE is excluded. All three runs share one fp32 initial noise
  tensor (``FLUX2_INIT_LATENTS``). The CPU fp32 / bf16 references are the same pipeline class on CPU
  (``examples/flux2/parity_ref.py``), then ``assert_close_three_way``. Single step because the
  guidance-distilled flow is chaotic: an in-band per-step error compounds past any fixed band over
  several steps.
* **Tier 3a, end to end vs an independent CPU reference:** 512 px, 8 steps, one shared fp32 initial noise
  tensor. The device run (TP=8, tiled device VAE) is compared with the same pipeline run on CPU in fp32
  and in bf16, decoded there with the untiled diffusers VAE. Two metrics, latent rel-L2 and image
  1 - SSIM, are each bounded by 2x the CPU-bf16-vs-fp32 floor + 0.5%.
  COST: the two CPU reference runs (fp32 + bf16 pipeline, 8 steps, on the unsharded 32B DiT) take about
  2 h 20 min and ~250 GB host RAM. Set ``FLUX2_REF_CACHE=<dir>`` to keep them: the references depend only
  on the weights, prompt, size, steps, seed and initial noise (all fixed here), so later runs reuse them
  and the test drops to the ~10 min device run.
* **Tier 3b, end to end at the served setting:** a 1024 px, 50-step generation. Checks on the result:
  * the device VAE decode is >= 35 dB PSNR against the CPU fp32 untiled decode of the same latents;
  * the image is not degenerate;
  * optionally, SSIM >= 0.9 against a golden image from an earlier accepted run (``FLUX2_GOLDEN``), as a
    regression check (same prompt / seed 42 / 1024 px / 50 steps).

Needs ``FLUX2_WEIGHTS``, 8 NeuronCores and ~250 GB host RAM for the CPU references.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from .test_flux2_components_device import _neuron_available

FLUX2_WEIGHTS = os.environ.get("FLUX2_WEIGHTS", "")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "examples", "flux2", "run.py")
REF = os.path.join(ROOT, "examples", "flux2", "parity_ref.py")
# the tiers run the 8-core TP=8 layout by default; FLUX2_STAGE_CONFIG selects another (e.g. flux2_stage.yaml)
STAGE = os.environ.get(
    "FLUX2_STAGE_CONFIG", os.path.join(ROOT, "examples", "flux2", "flux2_stage_tp8.yaml")
)

pytestmark = [
    pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device"),
    pytest.mark.skipif(
        not os.path.isdir(os.path.join(FLUX2_WEIGHTS, "transformer")), reason="set FLUX2_WEIGHTS"
    ),
]


def _cpu_env():
    env = dict(
        os.environ,
        PJRT_DEVICE="CPU",
        VLLM_NEURON_CPU_MODE="1",
        VLLM_NEURON_LIBTORCH_NEURONX_LITE="0",
        NEURON_RT_VISIBLE_CORES="",
    )
    env.pop("VLLM_NEURON_BACKEND", None)
    return env


def _run(cmd, env=None, timeout=7200):
    p = subprocess.run(
        [sys.executable, *cmd],
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


def _init_latents(path, height, width, seed=0):
    with open(os.path.join(FLUX2_WEIGHTS, "transformer", "config.json")) as f:
        c = json.load(f)["in_channels"]
    with open(os.path.join(FLUX2_WEIGHTS, "vae", "config.json")) as f:
        vsf = 2 ** (len(json.load(f)["block_out_channels"]) - 1)
    g = torch.Generator().manual_seed(seed)
    torch.save(
        {"latents": torch.randn(1, c, height // (vsf * 2), width // (vsf * 2), generator=g)}, path
    )


def test_single_step_three_way(tmp_path):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    init, dev, refs = tmp_path / "init.pt", tmp_path / "dev.pt", tmp_path / "refs.pt"
    _init_latents(init, 256, 256)
    env = dict(os.environ, FLUX2_INIT_LATENTS=str(init))
    _run(
        [
            RUN,
            "--stage-config",
            STAGE,
            "--model-path",
            FLUX2_WEIGHTS,
            "--height",
            "256",
            "--width",
            "256",
            "--steps",
            "1",
            "--seed",
            "0",
            "--parity-latents",
            str(dev),
        ],
        env,
    )
    _run(
        [
            REF,
            "--model-path",
            FLUX2_WEIGHTS,
            "--device-latents",
            str(dev),
            "--save-refs",
            str(refs),
        ],
        dict(_cpu_env(), FLUX2_INIT_LATENTS=str(init)),
    )
    r = torch.load(refs)
    assert_close_three_way(r["fp32"], r["bf16"], r["device"], name="flux2_pipeline_1step_256")


def test_end_to_end_vs_cpu_reference_512(tmp_path):
    """Independent end-to-end reference: device vs the same pipeline on CPU fp32, with CPU bf16 as the
    floor. Both latents and decoded images, bar = 2 x floor + 0.5%."""
    init, dump, refs = tmp_path / "init.pt", tmp_path / "dump.pt", tmp_path / "refs.pt"
    lat = tmp_path / "dev_lat.pt"
    _init_latents(init, 512, 512)
    env = dict(os.environ, FLUX2_INIT_LATENTS=str(init), FLUX2_VAE_DUMP=str(dump))
    _run(
        [
            RUN,
            "--stage-config",
            STAGE,
            "--model-path",
            FLUX2_WEIGHTS,
            "--height",
            "512",
            "--width",
            "512",
            "--steps",
            "8",
            "--seed",
            "0",
            "--output",
            str(tmp_path / "dev_512.png"),
        ],
        env,
    )
    d = torch.load(dump)
    torch.save(
        {
            "latents": d["latents"].float(),
            "prompt": _DEFAULT_PROMPT,
            "height": 512,
            "width": 512,
            "steps": 8,
            "guidance_scale": 4.0,
            "seed": 0,
        },
        lat,
    )
    cache = os.environ.get("FLUX2_REF_CACHE")
    cached = os.path.join(cache, "flux2_e2e_ref_512_8step_seed0.pt") if cache else None
    if cached and os.path.isfile(cached):
        refs = cached
    else:
        _run(
            [
                REF,
                "--model-path",
                FLUX2_WEIGHTS,
                "--device-latents",
                str(lat),
                "--save-refs",
                str(refs),
                "--decode",
            ],
            dict(_cpu_env(), FLUX2_INIT_LATENTS=str(init)),
            timeout=14400,
        )
        if cached:
            os.makedirs(cache, exist_ok=True)
            torch.save({k: v for k, v in torch.load(refs).items() if k != "device"}, cached)
    r = torch.load(refs)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    lat_floor, lat_dev = rel(r["bf16"], r["fp32"]), rel(d["latents"], r["fp32"])
    luma = lambda x: _to_gray(x)  # noqa: E731
    img_floor = 1.0 - _ssim(luma(r["bf16_image"]), luma(r["fp32_image"]))
    img_dev = 1.0 - _ssim(luma(d["image"]), luma(r["fp32_image"]))
    report = {
        "latent_rel_floor": lat_floor,
        "latent_rel_dev": lat_dev,
        "latent_bar": 2 * lat_floor + 0.005,
        "image_1mssim_floor": img_floor,
        "image_1mssim_dev": img_dev,
        "image_bar": 2 * img_floor + 0.005,
    }
    print("[e2e-512] " + json.dumps({k: round(v, 5) for k, v in report.items()}))
    assert lat_dev <= report["latent_bar"], report
    assert img_dev <= report["image_bar"], report


def test_end_to_end_1024(tmp_path):
    png, dump = tmp_path / "flux2_1024.png", tmp_path / "vae_dump.pt"
    _run(
        [
            RUN,
            "--stage-config",
            STAGE,
            "--model-path",
            FLUX2_WEIGHTS,
            "--height",
            "1024",
            "--width",
            "1024",
            "--steps",
            "50",
            "--seed",
            "42",
            "--output",
            str(png),
        ],
        dict(os.environ, FLUX2_VAE_DUMP=str(dump)),
    )
    check = (
        "import math, sys, torch\n"
        "from diffusers import AutoencoderKLFlux2\n"
        "d = torch.load(sys.argv[2])\n"
        "v = AutoencoderKLFlux2.from_pretrained(sys.argv[1], subfolder='vae', torch_dtype=torch.float32).eval()\n"
        "with torch.no_grad(): ref = v.decode(d['latents'].float(), return_dict=False)[0]\n"
        "a, b = d['image'].float().clamp(-1, 1), ref.clamp(-1, 1)\n"
        "print('PSNR', 10 * math.log10(4 / float(((a - b) ** 2).mean())), 'STD', float(a.std()))\n"
    )
    out = _run(["-c", check, FLUX2_WEIGHTS, str(dump)], _cpu_env())
    vals = out.split()
    psnr, std = float(vals[vals.index("PSNR") + 1]), float(vals[vals.index("STD") + 1])
    print(f"[e2e-1024] device VAE vs CPU fp32 untiled PSNR {psnr:.2f} dB, image std {std:.3f}")
    assert psnr >= 35.0, f"device VAE vs CPU fp32 untiled: {psnr:.2f} dB"
    assert std > 0.1, f"degenerate image (std {std:.3f})"
    golden = os.environ.get("FLUX2_GOLDEN")
    if golden:  # repeatability vs a previously accepted device image (not a correctness reference)
        ssim = _ssim(_gray(png), _gray(golden))
        print(f"[e2e-1024] SSIM vs golden {ssim:.4f}")
        assert ssim >= 0.9, f"SSIM vs golden {ssim:.3f}"


_DEFAULT_PROMPT = (
    "A cozy reading nook by a rain-streaked window, warm lamp light, a cat asleep on a stack of "
    "books, photorealistic"
)  # examples/flux2/run.py default


def _to_gray(img):
    """[1, 3, H, W] image in [-1, 1] -> [1, 1, H, W] luma in [0, 255] (BT.601)."""
    x = (img.float().clamp(-1, 1) + 1) * 127.5
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


def _gray(path):
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32))[None, None]


def _ssim(a, b, data_range=255.0, win=11, sigma=1.5):
    """Mean SSIM (Wang et al. 2004, Gaussian window), on [1, 1, H, W] grayscale tensors."""
    x = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    f = lambda t: torch.nn.functional.conv2d(t, w)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a**2, f(b * b) - mu_b**2, f(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return float(s.mean())
