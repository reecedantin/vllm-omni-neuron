# SPDX-License-Identifier: Apache-2.0
"""Accuracy tiers 2 and 3 (Neuron): the whole policy on device, three-way against the upstream policy.

Baseline: upstream policy, DiT in fp32 on CPU. Expected: upstream policy in bf16 on CPU (the dtype
error alone). Actual: this port in bf16 with the DiT and the VAE encoder on the NeuronCore (the served default).

* tier 2: one Cosmos UniPC step (per-call error before it compounds);
* tier 3: the released 4-step DROID schedule end to end, with the action gate the port is accepted
  on: device-vs-fp32 action rel-L2 <= 2 x (CPU-bf16-vs-fp32) + 0.5%.

Uses the tiny structure model by default, or the released DROID policy when ``FLUX3_ACTION_DROID`` (and
``FLUX3_ACTION_BASE``) point at local copies. Needs ``FLUX_ACTION_SRC``; skips without a Neuron device.
"""

from __future__ import annotations

import os

import pytest
import torch

from ..unit.test_flux3_action_utils.fixtures import observation, rel, tiny, upstream  # noqa: F401
from ..unit.test_flux3_action_utils.reference import run_reference
from .test_flux3_action_dit_device import _neuron_available

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


@pytest.fixture(scope="module")
def setup(tiny):  # noqa: F811
    if os.environ.get("FLUX3_ACTION_DROID"):
        return os.environ["FLUX3_ACTION_DROID"], os.environ.get("FLUX3_ACTION_BASE")
    return tiny["droid"], tiny["base"]


@pytest.fixture(scope="module")
def device_policy(setup):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.flux3_action.policy import NeuronFlux3ActionPolicy

    policy_dir, base = setup
    pol = NeuronFlux3ActionPolicy(
        policy_dir,
        base_dir=base,
        device=torch.device("neuron", 0),
        vae_device=torch.device("neuron", 0),
    )
    pol.compile(get_compile_backend_name())
    return pol


def _three_way(setup, device_policy, observation, steps):  # noqa: F811
    from vllm_neuron.accuracy.testing import assert_close_three_way

    from vllm_omni_neuron.diffusion.models.flux3_action.observation import load_observation

    policy_dir, base = setup
    batch = load_observation(observation)
    fp32 = run_reference(policy_dir, base, batch, dtype="float32", num_steps=steps)
    bf16 = run_reference(policy_dir, base, batch, num_steps=steps)
    device_policy.config.num_inference_steps = steps
    out = device_policy.predict(batch)
    assert_close_three_way(
        fp32["targets"],
        bf16["targets"],
        out.targets,
        name=f"targets_{steps}step",
        plot_on_failure=False,
    )
    return fp32, bf16, out


def test_single_step_three_way(setup, device_policy, observation, upstream):  # noqa: F811
    _three_way(setup, device_policy, observation, steps=1)


def test_e2e_three_way_and_action_gate(setup, device_policy, observation, upstream):  # noqa: F811
    fp32, bf16, out = _three_way(setup, device_policy, observation, steps=4)
    floor = rel(bf16["actions"], fp32["actions"])
    assert rel(out.actions, fp32["actions"]) <= 2 * floor + 0.005, (
        rel(out.actions, fp32["actions"]),
        floor,
    )
