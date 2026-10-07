# SPDX-License-Identifier: Apache-2.0
"""CPU checks that the LTX-2.5 host stages run with a fixed thread count by default (their results
depend on it, so a load-dependent count made outputs depend on the host load)."""

import os

import torch

from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import host_math_threads
from vllm_omni_neuron.diffusion.models.ltx2.pipeline_ltx25 import host_threads


def test_host_threads_fixed_unless_adaptive(monkeypatch):
    monkeypatch.delenv("LTX25_HOST_THREADS_ADAPTIVE", raising=False)
    monkeypatch.setattr(os, "getloadavg", lambda: (1e6, 0.0, 0.0))  # a saturated host
    prev = torch.get_num_threads()
    with host_threads(6) as k:
        assert k == 6 and torch.get_num_threads() == 6
    assert torch.get_num_threads() == prev
    monkeypatch.setenv("LTX25_HOST_THREADS_ADAPTIVE", "1")
    with host_threads(6) as k:
        assert k == 4  # floor under load


def test_host_math_threads_default_fixed(monkeypatch):
    monkeypatch.delenv("LTX2_HOST_MATH_THREADS", raising=False)
    prev = torch.get_num_threads()
    with host_math_threads():
        assert torch.get_num_threads() == 8
    assert torch.get_num_threads() == prev
