# SPDX-License-Identifier: Apache-2.0
"""Accuracy tier 1 (Neuron): one DiT denoiser call, three-way.

fp32 CPU (baseline) / bf16 CPU (dtype error alone) / bf16 Neuron compiled (adds the device error):
``assert_close_three_way`` passes when the device error distribution matches the bf16 one. Runs every
DiT phase graph (text, observation, step, head, joint blocks, final) once. Uses the tiny structure
model by default, or the released DROID policy when ``FLUX3_ACTION_DROID`` points at a local copy.
Skips without a Neuron device.
"""

from __future__ import annotations

import os

import pytest
import torch

from ..unit.test_flux3_action_utils.fixtures import tiny  # noqa: F401


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


def _inputs(dims, n_txt=80, n_video=2 * 340, n_vcond=340, n_act=32):
    g = torch.Generator().manual_seed(0)

    def ids(n, axis):
        x = torch.zeros(1, n, 4, dtype=torch.long)
        x[0, :, axis] = torch.arange(n)
        return x

    return dict(
        ctx=torch.randn(1, n_txt, dims.context_in_dim, generator=g),
        ctx_ids=ids(n_txt, 3),
        video_ids=ids(n_video, 1),
        video_cond=torch.randn(1, n_vcond, 96, generator=g),
        video_cond_ids=ids(n_vcond, 2),
        action_ids=ids(n_act, 0),
        action_cond=torch.randn(1, 1, dims.cond_channels, generator=g),
        action_cond_ids=ids(1, 0),
        video=torch.randn(1, n_video, 96, generator=g),
        action=torch.randn(1, n_act, dims.action_dim, generator=g),
    )


def _run(policy_dir, dims, dtype, device, x, compile_backend=None):
    from vllm_omni_neuron.diffusion.models.flux3_action.dit import Flux3ActionDiT

    dit = Flux3ActionDiT(dims, dtype=dtype)
    dit.load(os.path.join(policy_dir, "model.safetensors"), device=device)
    if compile_backend is not None:
        dit.compile(compile_backend)
    req = dit.prepare(
        x["ctx"],
        x["ctx_ids"],
        video_ids=x["video_ids"],
        video_cond=x["video_cond"],
        video_cond_ids=x["video_cond_ids"],
        action_ids=x["action_ids"],
        action_cond=x["action_cond"],
        action_cond_ids=x["action_cond_ids"],
    )
    step = dit.prepare_step(req, 0.7, 0.7)
    v, a = dit.forward(req, step, x["video"].to(dtype), x["action"].to(dtype))
    return v.float(), a.float()


def test_dit_forward_three_way(tiny):  # noqa: F811
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.flux3_action.dit import DiTDims, load_policy_config

    policy_dir = os.environ.get("FLUX3_ACTION_DROID") or tiny["droid"]
    dims = DiTDims.from_policy_config(load_policy_config(policy_dir))
    x = _inputs(dims)
    fp32 = _run(policy_dir, dims, torch.float32, "cpu", x)
    bf16 = _run(policy_dir, dims, torch.bfloat16, "cpu", x)
    dev = _run(
        policy_dir, dims, torch.bfloat16, torch.device("neuron", 0), x, get_compile_backend_name()
    )
    for name, i in (("video_velocity", 0), ("action_velocity", 1)):
        assert_close_three_way(fp32[i], bf16[i], dev[i], name=name, plot_on_failure=False)
