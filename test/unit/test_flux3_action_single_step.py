# SPDX-License-Identifier: Apache-2.0
"""Accuracy tier 2 (CPU): one denoising step of the whole policy against the upstream reference.

A single Cosmos UniPC step isolates the per-call error before it compounds across steps: text and
observation phases, both CFG branches, the joint DiT forward and the action/video heads, on the tiny
structure model. The device half is ``test/neuron/test_flux3_action_policy_device.py``.
"""

from __future__ import annotations

from vllm_omni_neuron.diffusion.models.flux3_action.observation import load_observation
from vllm_omni_neuron.diffusion.models.flux3_action.policy import NeuronFlux3ActionPolicy

from .test_flux3_action_utils.fixtures import cos, observation, rel, tiny, upstream  # noqa: F401
from .test_flux3_action_utils.reference import run_reference


def test_single_step_matches_upstream(tiny, observation, upstream):  # noqa: F811
    batch = load_observation(observation)
    ref = run_reference(tiny["droid"], tiny["base"], batch, num_steps=1)
    pol = NeuronFlux3ActionPolicy(tiny["droid"], base_dir=tiny["base"])
    pol.config.num_inference_steps = 1
    out = pol.predict(batch)
    assert rel(out.cond_latents, ref["cond_latent"]) < 2e-2  # bf16 VAE rounding
    assert rel(out.targets, ref["targets"]) < 1e-2
    assert cos(out.video_latents, ref["video_latents"]) > 0.9999
