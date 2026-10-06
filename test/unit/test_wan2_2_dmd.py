# SPDX-License-Identifier: Apache-2.0
"""DMD2 (FastWan2.2-TI2V-5B) sampler math vs FastVideo's DmdDenoisingStage."""

from __future__ import annotations

import os
import sys

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2_dmd import (  # noqa: E402
    DEFAULT_DMD_TIMESTEPS,
    _dmd_step,
    dmd_sigmas_for,
    dmd_training_sigma_table,
)

FASTVIDEO = os.environ.get("FASTVIDEO_SRC", "")


def test_sigma_table_endpoints():
    t, s = dmd_training_sigma_table()
    assert t.shape == s.shape == (1000,)
    assert float(s[0]) == pytest.approx(1.0)
    sig = dmd_sigmas_for(DEFAULT_DMD_TIMESTEPS)
    assert sig[0] == pytest.approx(1.0)
    assert sig[0] > sig[1] > sig[2] > 0
    # nearest table entry to t/1000
    for ts, sg in zip(DEFAULT_DMD_TIMESTEPS, sig, strict=True):
        assert abs(sg - ts / 1000) < 2e-3


def test_matches_fastvideo_scheduler_and_conversion():
    if not os.path.isdir(os.path.join(FASTVIDEO, "fastvideo")):
        pytest.skip("FastVideo source not present")
    sys.path.insert(0, FASTVIDEO)
    import types

    # FastVideo's package init pulls debugging-only deps; the two modules used here do not.
    if "remote_pdb" not in sys.modules:
        stub = types.ModuleType("remote_pdb")
        stub.RemotePdb = object
        sys.modules["remote_pdb"] = stub
    try:
        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from fastvideo.models.utils import pred_noise_to_pred_video
    except Exception as e:  # missing optional deps of the FastVideo package
        pytest.skip(f"cannot import FastVideo: {e}")
    sched = FlowMatchEulerDiscreteScheduler(shift=8.0)
    t, s = dmd_training_sigma_table()
    torch.testing.assert_close(t, sched.timesteps.float())
    torch.testing.assert_close(s, sched.sigmas.float())

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 6, 6, generator=g)
    flow = torch.randn(2, 4, 6, 6, generator=g)
    eps = torch.randn(2, 4, 6, 6, generator=g)
    sig = dmd_sigmas_for(DEFAULT_DMD_TIMESTEPS)
    t0 = torch.tensor([DEFAULT_DMD_TIMESTEPS[1]])
    want_x0 = pred_noise_to_pred_video(flow, x, t0, sched)
    want_next = sched.add_noise(want_x0, eps, torch.tensor([DEFAULT_DMD_TIMESTEPS[2]]))
    x0, nxt = _dmd_step(x, flow, eps, torch.tensor([sig[1], sig[2]]))
    torch.testing.assert_close(x0, want_x0.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nxt, want_next.float(), atol=1e-5, rtol=1e-5)
