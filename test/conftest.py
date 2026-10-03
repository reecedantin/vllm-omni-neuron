# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Cosmos3-Edge tests."""

from __future__ import annotations

import os
import socket

import pytest

COSMOS3_EDGE_WEIGHTS = os.environ.get("COSMOS3_EDGE_WEIGHTS", "")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def vllm_single_rank():
    """A world-size-1 vLLM config + torch.distributed (gloo) + TP group, entered for the session."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    yield
    ctx.__exit__(None, None, None)


@pytest.fixture(scope="session")
def edge_weights() -> str:
    path = COSMOS3_EDGE_WEIGHTS
    if not path or not os.path.isdir(os.path.join(path, "transformer")):
        pytest.skip("set COSMOS3_EDGE_WEIGHTS to a local nvidia/Cosmos3-Edge checkout")
    return path
