# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU runs of the Wan2.2 Omni pipelines on tiny checkpoints.

``examples/wan2_2/run_ti2v.py`` in CPU mode (``VLLM_NEURON_CPU_MODE=1``, eager, latent output):
FastWan DMD2 at TP=1 and at TP=2 with sequence parallelism must give the same latents. TP=2
covers ranks that hold no VAE, which still have to size the latents from the VAE config (the
TI2V VAE is 16x spatial).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

from test.unit.test_wan2_2_tiny import make_tiny_checkpoint, require_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "examples", "wan2_2", "run_ti2v.py")


def _run(model: str, out: str, tp: int) -> torch.Tensor:
    env = dict(os.environ, VLLM_NEURON_CPU_MODE="1", PJRT_DEVICE="CPU")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        RUNNER,
        "--model-path",
        model,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "9",
        "--tp",
        str(tp),
        "--sp",
        "on" if tp > 1 else "off",
        "--cp",
        "1",
        "--cfg-parallel",
        "1",
        "--vae-pp",
        "1",
        "--devices",
        ",".join(str(i) for i in range(tp)),
        "--eager",
        "--latents",
        "--output",
        out,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return torch.load(out + ".latents.pt")


def test_dmd2_tp2_matches_tp1(tmp_path):
    require_tokenizer()
    model = make_tiny_checkpoint(str(tmp_path / "dmd"), "dmd2-ti2v-5b")
    a = _run(model, str(tmp_path / "tp1"), 1)
    b = _run(model, str(tmp_path / "tp2"), 2)
    assert a.shape == b.shape == (1, 48, 3, 16, 16)
    err = float((a.float() - b.float()).norm() / a.float().norm())
    assert err < 0.02, f"TP2 vs TP1 rel-L2 {err:.4f}"
