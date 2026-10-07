# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of MiniMax-H3 and its 4-step distillation FastH3 (text -> video + audio)."""

from .pipeline_minimax_h3 import (
    PIPELINE_REGISTRY,
    NeuronMiniMaxH3Pipeline,
    get_minimax_h3_post_process_func,
)

__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronMiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]
