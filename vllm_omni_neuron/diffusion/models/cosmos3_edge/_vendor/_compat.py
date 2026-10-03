# SPDX-License-Identifier: Apache-2.0
"""Shims for upstream vLLM-Omni symbols the vendored Cosmos3 code imports but the
pinned ``vllm-omni==0.24.0`` does not ship.

* World-model session state (RFC #4480) is an opt-in multi-turn feature; the Neuron
  port serves single requests, so it is reported as disabled.
* ``CacheDiTAdapterConfig`` only describes cache-dit acceleration, which the Neuron
  pipeline does not use; a plain record keeps the class attribute importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # newer vllm-omni
    from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig  # type: ignore
except ImportError:  # vllm-omni 0.24.0

    @dataclass
    class CacheDiTAdapterConfig:  # type: ignore[no-redef]
        block_forward_patterns: dict[str, Any] = field(default_factory=dict)
        has_separate_cfg: bool = False
        check_forward_pattern: bool = True


try:  # newer vllm-omni
    from vllm_omni.experimental.world_models.adapters.state_cosmos3_adapter import (  # type: ignore
        Cosmos3StateAdapter,
    )
    from vllm_omni.experimental.world_models.session_state import (  # type: ignore
        SessionStateManager,
        resolve_session_state_config,
    )
except ImportError:  # vllm-omni 0.24.0

    class SessionStateManager:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Session state manager is not available on vllm-omni 0.24.0")

    class Cosmos3StateAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Session state manager is not available on vllm-omni 0.24.0")

    def resolve_session_state_config(enable: bool = False, **_: Any) -> tuple[bool, int]:
        if enable:
            raise ValueError("enable_session_state_manager is not supported by the Neuron Cosmos3 port")
        return False, 0
