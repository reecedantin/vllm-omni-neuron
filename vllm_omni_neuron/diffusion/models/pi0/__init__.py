# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of the LeRobot pi0 family VLAs: pi0.5 / pi0.52 (``pi05``, ``pi052``)
and pi0 (PaliGemma + Gemma action expert, flow-matching action head).

One package, two pipelines; the combined ``PIPELINE_REGISTRY`` is what the plugin's
auto-discovery reads (it scans one level deep)."""

from . import _compat
from .model import NeuronPi05ActionModel
from .model_pi0 import NeuronPi0ActionModel
from .pipeline import PIPELINE_REGISTRY as _PI05_PIPELINE_REGISTRY
from .pipeline import NeuronPi05Pipeline, get_pi05_post_process_func
from .pipeline_pi0 import PI0_PIPELINE_REGISTRY as _PI0_PIPELINE_REGISTRY
from .pipeline_pi0 import NeuronPi0Pipeline, get_pi0_post_process_func
from .subtask import Pi052SubtaskGenerator, format_subtask_prompt

PIPELINE_REGISTRY = [*_PI05_PIPELINE_REGISTRY, *_PI0_PIPELINE_REGISTRY]

_compat.install()  # vllm-omni 0.24: resolve LeRobot ``type: pi0/pi05/pi052`` checkpoints

__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronPi05Pipeline",
    "NeuronPi0Pipeline",
    "NeuronPi05ActionModel",
    "NeuronPi0ActionModel",
    "Pi052SubtaskGenerator",
    "format_subtask_prompt",
    "get_pi05_post_process_func",
    "get_pi0_post_process_func",
]
