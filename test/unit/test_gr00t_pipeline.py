# SPDX-License-Identifier: Apache-2.0
"""CPU check of ``NeuronGr00tN1d7Pipeline`` end to end (observation -> decoded actions) against
upstream vLLM-Omni's ``Gr00tPolicy`` on the same observation and the same seeded noise.

Needs real weights and a local Qwen3-VL-2B-Instruct processor:
``GR00T_WEIGHTS=/path/to/gr00t-n17 GR00T_VLM_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct``.
The upstream policy also needs ``GR00T_HF_ROOT``: a directory holding ``nvidia/Cosmos-Reason2-2B``
and ``Qwen/Qwen3-VL-2B-Instruct`` (local copies or symlinks), since it resolves both from the Hub.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

WEIGHTS = os.environ.get("GR00T_WEIGHTS", "")
VLM = os.environ.get("GR00T_VLM_PROCESSOR", "")
HF_ROOT = os.environ.get("GR00T_HF_ROOT", "")
TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(WEIGHTS) and os.path.isdir(VLM)),
    reason="set GR00T_WEIGHTS and GR00T_VLM_PROCESSOR",
)


def make_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "video": {
            k: rng.integers(0, 255, (1, 2, 180, 320, 3), dtype=np.uint8)
            for k in ("exterior_image_1_left", "wrist_image_left")
        },
        "state": {
            "eef_9d": (rng.normal(size=(1, 1, 9)) * 0.1).astype(np.float32),
            "gripper_position": rng.uniform(size=(1, 1, 1)).astype(np.float32),
            "joint_position": (rng.normal(size=(1, 1, 7)) * 0.3).astype(np.float32),
        },
        "language": {
            "annotation.language.language_instruction": [
                ["pick up the red cube and put it in the bowl"]
            ]
        },
    }


@pytest.fixture(scope="module")
def pipeline():
    from vllm_omni_neuron.diffusion.models.gr00t import NeuronGr00tN1d7Pipeline

    od = SimpleNamespace(model=WEIGHTS, model_config={"embodiment_tag": TAG, "vlm_processor": VLM})
    return NeuronGr00tN1d7Pipeline(od_config=od)


def _request(obs, seed):
    sp = SimpleNamespace(extra_args={"robot_obs": obs}, seed=seed, generator=None)
    return SimpleNamespace(sampling_params=sp, request_id="test-0")


def test_pipeline_forward_contract_and_seed(pipeline):
    obs = make_obs()
    a = pipeline.forward(_request(obs, 3)).output["actions"]
    b = pipeline.forward(_request(obs, 3)).output["actions"]
    c = pipeline.forward(_request(obs, 4)).output["actions"]
    assert set(a) == {"eef_9d", "gripper_position", "joint_position"}
    assert a["joint_position"].shape == (1, 40, 7) and a["joint_position"].dtype == np.float32
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not all(np.array_equal(a[k], c[k]) for k in a)


@pytest.mark.skipif(not os.path.isdir(HF_ROOT), reason="set GR00T_HF_ROOT for the upstream policy")
def test_pipeline_matches_upstream_policy(pipeline, monkeypatch):
    from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy

    obs = make_obs()
    monkeypatch.setenv("GR00T_NOISE_SEED", "0")  # upstream 0.24: bf16 randn from a seeded generator
    mine = pipeline.forward(_request(obs, 0)).output["actions"]
    cwd = os.getcwd()
    os.chdir(HF_ROOT)
    try:
        ref, _ = Gr00tPolicy(embodiment_tag=TAG, model_path=WEIGHTS, device="cpu").get_action(obs)
    finally:
        os.chdir(cwd)
    for k in ref:
        r = np.linalg.norm(mine[k] - ref[k]) / max(np.linalg.norm(ref[k]), 1e-6)
        assert r < 0.05, (k, r)
