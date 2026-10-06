# SPDX-License-Identifier: Apache-2.0
"""Tier-3 repeatability: end-to-end decoded video vs a reviewed Neuron golden (per-frame SSIM).

docs/model-dev/onboarding-models.md, Step 4 tier 3. A full served generation (text encode, every
denoising step with CFG, VAE decode) is compared frame by frame with a cached, reviewed golden
produced by the same configuration. This is a regression check against an earlier Neuron output,
not a comparison with the reference implementation (that is
``test_wan2_2_ti2v_e2e_reference_accuracy.py``): small per-step errors tiers 1-2 cannot see can
still compound into a collapsed video, which this catches between releases.

    WAN22_MODEL=<checkpoint> WAN22_GOLDEN=<golden frames .npy> \\
        pytest test/neuron/test_wan2_2_ti2v_e2e_accuracy.py

    # create / refresh the golden after reviewing the video by eye:
    WAN22_MODEL=<checkpoint> \\
        python test/neuron/test_wan2_2_ti2v_e2e_accuracy.py --regen <golden.npy>

The golden records its generation settings next to it (``<golden>.json``); the test reruns with
exactly those. Optional ``WAN22_SERVED_ARGS`` sets the parallel layout (default TP4 x CP4 on 16
cores) and ``WAN22_SSIM_MIN`` the per-frame floor (default 0.90; mean must reach 0.95).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys

import numpy as np
import pytest

MODEL = os.environ.get("WAN22_MODEL", "")
GOLDEN = os.environ.get("WAN22_GOLDEN", "")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "examples", "wan2_2", "run_ti2v.py")
DEFAULT_SETTINGS = {
    "prompt": "A fluffy orange cat walking gracefully across a sunny garden path, high quality",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "seed": 42,
    "steps": None,  # None = the checkpoint default (3 for FastWan DMD2, 50 for TI2V-5B)
    "guidance_scale": 5.0,
}
DEFAULT_LAYOUT = "--tp 4 --sp on --cp 4 --vae-pp 16 --devices " + ",".join(
    str(i) for i in range(16)
)


def ssim_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Luma SSIM per frame (8x8 box windows, standard C1/C2), frames in [0, 1], ``[T, H, W, 3]``."""
    import torch
    import torch.nn.functional as F

    weights = torch.tensor([0.299, 0.587, 0.114])

    def luma(x):
        return (torch.from_numpy(np.ascontiguousarray(x)).float() * weights).sum(-1)[:, None]

    x, y = luma(a), luma(b)
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = F.avg_pool2d(x, 8, 1), F.avg_pool2d(y, 8, 1)
    sxx = F.avg_pool2d(x * x, 8, 1) - mu_x**2
    syy = F.avg_pool2d(y * y, 8, 1) - mu_y**2
    sxy = F.avg_pool2d(x * y, 8, 1) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3)).numpy()


def generate(model_path: str, settings: dict, out_stem: str) -> np.ndarray:
    cmd = [
        sys.executable, RUNNER, "--model-path", model_path, "--prompt", settings["prompt"],
        "--height", str(settings["height"]), "--width", str(settings["width"]),
        "--num-frames", str(settings["num_frames"]), "--seed", str(settings["seed"]),
        "--guidance-scale", str(settings["guidance_scale"]), "--cfg-parallel", "1",
        "--save-frames", "--output", out_stem + ".mp4",
        *shlex.split(os.environ.get("WAN22_SERVED_ARGS", DEFAULT_LAYOUT)),
    ]  # fmt: skip
    if settings.get("steps"):
        cmd += ["--steps", str(settings["steps"])]
    env = dict(
        os.environ,
        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
    )
    timeout = int(os.environ.get("WAN22_STEP_TIMEOUT", "1800"))
    # Streamed (not captured) so a stall is visible live; own process group so a timeout also
    # takes down the stage workers instead of leaving them holding cores.
    proc = subprocess.Popen([sys.executable, "-u", *cmd[1:]], env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise AssertionError(f"served run exceeded {timeout} s (output above)") from None
    assert rc == 0, f"served run failed rc={rc} (output above)"
    return np.load(out_stem + ".mp4.npy")


@pytest.mark.skipif(
    not (os.path.isdir(os.path.join(MODEL, "transformer")) and os.path.isfile(GOLDEN)),
    reason="set WAN22_MODEL and WAN22_GOLDEN (create it with --regen)",
)
def test_e2e_matches_golden(tmp_path):
    meta_path = GOLDEN[: -len(".npy")] + ".json" if GOLDEN.endswith(".npy") else GOLDEN + ".json"
    settings = dict(DEFAULT_SETTINGS)
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            settings.update(json.load(f))
    golden = np.load(GOLDEN)
    frames = generate(MODEL, settings, str(tmp_path / "e2e"))
    assert frames.shape == golden.shape, (frames.shape, golden.shape)
    assert np.isfinite(frames).all()
    ssim = ssim_per_frame(frames, golden)
    floor = float(os.environ.get("WAN22_SSIM_MIN", "0.90"))
    worst = int(ssim.argmin())
    print(f"SSIM per frame: mean {ssim.mean():.4f} min {ssim.min():.4f} (frame {worst})")
    assert ssim.min() >= floor, f"frame {int(ssim.argmin())} SSIM {ssim.min():.4f} < {floor}"
    assert ssim.mean() >= 0.95, f"mean SSIM {ssim.mean():.4f} < 0.95"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--regen":
        out = sys.argv[2]
        stem = out[: -len(".npy")] if out.endswith(".npy") else out
        frames = generate(MODEL, DEFAULT_SETTINGS, stem)
        np.save(stem + ".npy", frames)
        with open(stem + ".json", "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)
        print(f"golden: {stem}.npy {frames.shape}; review {stem}.mp4 by eye before relying on it")
    elif len(sys.argv) == 3:
        a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
        s = ssim_per_frame(a, b)
        print(f"SSIM per frame: mean {s.mean():.4f} min {s.min():.4f} (frame {int(s.argmin())})")
    else:
        print(__doc__)
