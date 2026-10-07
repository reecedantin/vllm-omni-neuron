# SPDX-License-Identifier: Apache-2.0
"""CPU: the GR00T host fast path reproduces upstream's per-pose action decoding."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("seed", range(5))
def test_eef_rot6d_to_absolute_matches_upstream(seed):
    pytest.importorskip("vllm_omni.diffusion.models.gr00t.dataio.state_action.action_chunking")
    from vllm_omni.diffusion.models.gr00t.dataio.state_action.action_chunking import (
        EndEffectorActionChunk,
    )
    from vllm_omni.diffusion.models.gr00t.dataio.state_action.pose import EndEffectorPose
    from vllm_omni.diffusion.models.gr00t.dataio.types import ActionFormat

    from vllm_omni_neuron.diffusion.models.gr00t.host_fastpath import eef_rot6d_to_absolute

    rng = np.random.default_rng(seed)
    action = rng.normal(size=(40, 9)).astype(np.float32)
    state = rng.normal(size=9).astype(np.float32)
    fmt = ActionFormat.XYZ_ROT6D
    want = (
        EndEffectorActionChunk.from_array(action, fmt)
        .to_absolute_chunking(EndEffectorPose.from_action_format(state, fmt))
        .to(fmt)
    )
    got = eef_rot6d_to_absolute(action, state)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)  # float32 inputs: a few ulp at most


def test_torch_threads_restores():
    import torch

    from vllm_omni_neuron.diffusion.models.gr00t.host_fastpath import torch_threads

    before = torch.get_num_threads()
    with torch_threads(3):
        assert torch.get_num_threads() == 3
    assert torch.get_num_threads() == before
