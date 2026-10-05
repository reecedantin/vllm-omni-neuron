# SPDX-License-Identifier: Apache-2.0
"""Tier 3 accuracy: an end-to-end Cosmos3 generation scored against a cached golden image.

Runs the full served path (``examples/cosmos3_edge/run.py`` -> vLLM-Omni stage -> Neuron pipeline:
text tokenizer, both CFG branches, the whole 50-step UniPC loop, VAE decode) in a subprocess, then
compares the output PNG with ``COSMOS3_E2E_GOLDEN`` by SSIM. This is a regression check against an
earlier accepted Neuron output, not a comparison with the reference implementation (tiers 1-2 are),
and it is the tier that catches per-step error compounding over the trajectory.

Env:
* ``COSMOS3_QWEN3_WEIGHTS``  Cosmos3-Nano checkout; ``COSMOS3_E2E_GOLDEN`` the golden PNG (same prompt,
  seed, size and step count as below). Both required, else the test skips.
* I2V: ``COSMOS3_E2E_I2V_GOLDEN`` (golden mp4) and ``COSMOS3_E2E_I2V_IMAGE`` (conditioning image);
  ``COSMOS3_E2E_PROFILE=1`` also times a warm repeat.
* ``COSMOS3_E2E_STAGE`` (default the Nano TP=4 Trn2 stage), ``COSMOS3_E2E_SSIM_MIN`` (0.90),
  ``COSMOS3_TEST_OUT`` (where the generated PNG and report go).
* Multi-process stage: ``NEURON_VISIBLE_DEVICES`` is derived from the launch core set when unset
  (vLLM workers refuse ``NEURON_RT_VISIBLE_CORES``); stage ``devices:`` index into it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from .test_cosmos3_edge_und_device import _neuron_available

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.environ.get("COSMOS3_QWEN3_WEIGHTS", "")
GOLDEN = os.environ.get("COSMOS3_E2E_GOLDEN", "")
STAGE = os.environ.get(
    "COSMOS3_E2E_STAGE",
    os.path.join(REPO, "examples", "cosmos3_edge", "cosmos3_nano_stage_trn2_tp4.yaml"),
)
SSIM_MIN = float(os.environ.get("COSMOS3_E2E_SSIM_MIN", "0.90"))
PROMPT = "A red sports car parked on a wet city street at golden hour, photorealistic"
ARGS = ["--mode", "t2i", "--height", "640", "--width", "640", "--steps", "50", "--seed", "1"]


def ssim(a: np.ndarray, b: np.ndarray, window: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM (Wang et al. 2004, Gaussian window, K1=0.01 K2=0.03) over channels of two uint8 HxWxC
    images. Self-contained so the test needs no image-quality package."""
    x = torch.from_numpy(a.astype(np.float64)).permute(2, 0, 1)[None]
    y = torch.from_numpy(b.astype(np.float64)).permute(2, 0, 1)[None]
    c = x.shape[1]
    g = torch.exp(
        -((torch.arange(window, dtype=torch.float64) - window // 2) ** 2) / (2 * sigma**2)
    )
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    k = g.expand(c, 1, window, window)
    conv = lambda z: F.conv2d(z, k, groups=c)  # noqa: E731  (valid region only, like skimage)
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x * x) - mx**2, conv(y * y) - my**2, conv(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def _stage_env() -> dict:
    env = dict(os.environ)
    if not env.get("NEURON_VISIBLE_DEVICES"):
        from .test_cosmos3_edge_pipeline_accuracy import launch_core_list

        env["NEURON_VISIBLE_DEVICES"] = ",".join(str(c) for c in launch_core_list())
    env.pop("NEURON_RT_VISIBLE_CORES", None)
    env.pop("NEURON_RT_NUM_CORES", None)
    env.setdefault("COSMOS3_HANDSHAKE_TIMEOUT_S", "7200")
    return env


def test_ssim_helper_sanity():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.stack([xx * 4, yy * 4, (xx + yy) * 2], axis=-1).astype(np.uint8)  # smooth gradients
    noisy = np.clip(img.astype(np.int16) + rng.integers(-40, 41, img.shape), 0, 255).astype(
        np.uint8
    )
    assert abs(ssim(img, img) - 1.0) < 1e-9
    assert ssim(img, noisy) < 0.9


def test_e2e_t2i_matches_golden():
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "transformer")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a real Cosmos3-Nano checkout")
    if not GOLDEN or not os.path.isfile(GOLDEN):
        pytest.skip("set COSMOS3_E2E_GOLDEN to the golden PNG")
    from PIL import Image

    out_dir = os.environ.get("COSMOS3_TEST_OUT", ".")
    out_png = os.path.join(out_dir, "e2e_t2i.png")
    cmd = [
        sys.executable,
        os.path.join(REPO, "examples", "cosmos3_edge", "run.py"),
        *ARGS,
        "--prompt",
        PROMPT,
        "--model-path",
        WEIGHTS,
        "--stage-config",
        STAGE,
        "--output",
        out_png,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=_stage_env(), capture_output=True, text=True)
    wall = time.time() - t0
    with open(os.path.join(out_dir, "e2e_t2i.log"), "w") as f:
        f.write(proc.stdout + proc.stderr)
    assert proc.returncode == 0, (proc.returncode, (proc.stdout + proc.stderr)[-3000:])
    got = np.asarray(Image.open(out_png).convert("RGB"))
    ref = np.asarray(Image.open(GOLDEN).convert("RGB"))
    assert got.shape == ref.shape, (got.shape, ref.shape)
    diff = np.abs(got.astype(np.int32) - ref.astype(np.int32))
    mse = float((diff.astype(np.float64) ** 2).mean())
    report = {
        "ssim": ssim(got, ref),
        "ssim_min": SSIM_MIN,
        "psnr_db": 10 * np.log10(255.0**2 / mse) if mse else None,
        "mean_abs_diff": float(diff.mean()),
        "wall_s": round(wall, 1),
        "golden": os.path.basename(GOLDEN),
    }
    with open(os.path.join(out_dir, "e2e_t2i.json"), "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    print("[e2e]", json.dumps(report))
    assert report["ssim"] >= SSIM_MIN, report


I2V_GOLDEN = os.environ.get("COSMOS3_E2E_I2V_GOLDEN", "")
I2V_IMAGE = os.environ.get("COSMOS3_E2E_I2V_IMAGE", "")
I2V_ARGS = [
    "--mode",
    "i2v",
    "--height",
    "640",
    "--width",
    "640",
    "--num-frames",
    "9",
    "--steps",
    "35",
    "--seed",
    "1",
]


def _read_video(path: str) -> list[np.ndarray]:
    import av

    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def test_e2e_i2v_matches_golden():
    """I2V end to end (conditioning-frame encode -> 35-step loop -> decode) against a golden video,
    per-frame SSIM. Gate: mean over frames >= ``COSMOS3_E2E_SSIM_MIN``. ``COSMOS3_E2E_PROFILE=1`` adds
    a warm repeat (``--profile``) and records its latency."""
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "transformer")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a real Cosmos3-Nano checkout")
    if not (I2V_GOLDEN and os.path.isfile(I2V_GOLDEN) and I2V_IMAGE and os.path.isfile(I2V_IMAGE)):
        pytest.skip(
            "set COSMOS3_E2E_I2V_GOLDEN (golden mp4) and COSMOS3_E2E_I2V_IMAGE (conditioning image)"
        )
    import re

    out_dir = os.environ.get("COSMOS3_TEST_OUT", ".")
    out_mp4 = os.path.join(out_dir, "e2e_i2v.mp4")
    profile = os.environ.get("COSMOS3_E2E_PROFILE") == "1"
    cmd = [
        sys.executable,
        os.path.join(REPO, "examples", "cosmos3_edge", "run.py"),
        *I2V_ARGS,
        "--image",
        I2V_IMAGE,
        "--prompt",
        PROMPT,
        "--model-path",
        WEIGHTS,
        "--stage-config",
        STAGE,
        "--output",
        out_mp4,
    ]
    if profile:
        cmd.append("--profile")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=_stage_env(), capture_output=True, text=True)
    wall = time.time() - t0
    log = proc.stdout + proc.stderr
    with open(os.path.join(out_dir, "e2e_i2v.log"), "w") as f:
        f.write(log)
    assert proc.returncode == 0, (proc.returncode, log[-3000:])
    got, ref = _read_video(out_mp4), _read_video(I2V_GOLDEN)
    assert len(got) == len(ref) and got[0].shape == ref[0].shape, (
        len(got),
        len(ref),
        got[0].shape,
        ref[0].shape,
    )
    per_frame = [round(ssim(g, r), 4) for g, r in zip(got, ref, strict=True)]
    warm = re.search(r"warm request: ([\d.]+)s", log)
    report = {
        "ssim_mean": float(np.mean(per_frame)),
        "ssim_min_frame": min(per_frame),
        "ssim_per_frame": per_frame,
        "ssim_min": SSIM_MIN,
        "frames": len(got),
        "wall_s": round(wall, 1),
        "warm_s": float(warm.group(1)) if warm else None,
        "golden": os.path.basename(I2V_GOLDEN),
    }
    with open(os.path.join(out_dir, "e2e_i2v.json"), "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    print("[e2e-i2v]", json.dumps(report))
    assert report["ssim_mean"] >= SSIM_MIN, report
