# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of NVIDIA GR00T N1.7 (and the GR00T-H-N1.7 post-train)."""

import logging as _logging

# Route this package's log lines through vLLM's logger (as vllm_omni does for its own tree), so
# INFO lines from the diffusion worker processes -- per-rank layout, pretranspose -- reach the
# job log with vLLM's worker prefix instead of being dropped by an unconfigured root logger.
_pkg_logger = _logging.getLogger(__name__)
_pkg_logger.parent = _logging.getLogger("vllm")
_pkg_logger.propagate = True

from .pipeline_gr00t import PIPELINE_REGISTRY, NeuronGr00tN1d7Pipeline  # noqa: E402

__all__ = ["PIPELINE_REGISTRY", "NeuronGr00tN1d7Pipeline"]
