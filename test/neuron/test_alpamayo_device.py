# SPDX-License-Identifier: Apache-2.0
"""End-to-end device accuracy for Alpamayo 1.5 on Trainium (tensor parallel through the vLLM-Omni runner).

Serves one request with ``examples/alpamayo/run.py`` and compares the trajectory and the
Chain-of-Causation tokens with an upstream fp32 greedy reference written by
``examples/alpamayo/parity_ref.py`` (same inputs, same flow-matching noise seed 0).

Needs a Neuron device, ``$ALPAMAYO_WEIGHTS`` (an Alpamayo-1.5-10B checkout) and
``$ALPAMAYO_PARITY_REF`` (the reference dump); multi-rank device visibility must already be set
(``NEURON_VISIBLE_DEVICES`` with as many cores as the stage config's ``devices:``; four for the
default ``alpamayo_stage_trn2.yaml``, or set ``$ALPAMAYO_STAGE_CONFIG``). Skips cleanly otherwise.
Gate: trajectory rel-L2 <= 1e-2 vs fp32 (bf16 on CPU measures 1.5e-3; Trn2 2.4e-3 at TP=2,
4.5e-3 at TP=4, 2.1e-3 at TP=8) and identical reasoning tokens.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

WEIGHTS = os.environ.get("ALPAMAYO_WEIGHTS", "")
REF = os.environ.get("ALPAMAYO_PARITY_REF", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
STAGE = os.environ.get(
    "ALPAMAYO_STAGE_CONFIG", os.path.join(ROOT, "examples", "alpamayo", "alpamayo_stage_trn2.yaml")
)


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_neuron_available() and os.path.isdir(WEIGHTS) and os.path.isfile(REF)),
    reason="needs a Neuron device, $ALPAMAYO_WEIGHTS and $ALPAMAYO_PARITY_REF",
)


def test_served_trajectory_matches_upstream_fp32(tmp_path):
    out = tmp_path / "alpamayo.npz"
    cmd = [
        sys.executable,
        os.path.join(ROOT, "examples", "alpamayo", "run.py"),
        "--model-path",
        WEIGHTS,
        "--stage-config",
        STAGE,
        "--reference",
        REF,
        "--output",
        str(out),
        "--repeat",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, timeout=7200)
    got = np.load(out)
    ref = torch.load(REF, weights_only=False)
    want = ref["pred_xyz"][0, 0, 0].numpy()
    xyz = got["pred_xyz"].reshape(want.shape)
    rel = float(np.linalg.norm(xyz - want) / np.linalg.norm(want))
    assert rel <= 1e-2, rel
    p = int(ref["prompt_len"])
    assert np.array_equal(got["generated"], ref["sequences"][0, p:].numpy())
