# SPDX-License-Identifier: Apache-2.0
"""NeuronCore-generation facts for the plugin.

The one place that decides whether NKI kernels can run, so nothing else hard-codes
"inf2" or "trn2":

* ``neuron_core_generation()`` -> 2 (Inf2 / Trn1, NeuronCore-v2), 3 (Trn2), 4 (Trn3).
  NKI kernels need NeuronCore-v3 or newer; on v2 every kernel call site must take the
  ``torch.compile``-lowered torch path.
* ``use_nki_kernels(tensor)`` -> whether an NKI kernel may run for this tensor: a v3+ core,
  a device tensor, and kernels not disabled (``vllm_neuron``'s own ``can_run_kernel``, which
  does not know about NeuronCore generations and would otherwise say yes on Inf2).
* ``supports_nki()`` -> the generation half alone (used by the plugin-wide kernel gate in
  ``wan2_2_transformer.can_run_kernel``, which every Wan / VAE / AdaLN kernel site goes through).

Overrides: ``COSMOS3_NEURON_CORE_GEN=<int>`` (forces a generation) and
``COSMOS3_EDGE_ATTN_IMPL=torch`` (forces the torch path everywhere).
"""

from __future__ import annotations

import logging
import os
from functools import cache

import torch

logger = logging.getLogger(__name__)

# Platform target (libtorch-neuronx-lite / NRT naming) -> NeuronCore generation.
_TARGET_TO_GEN = {
    "inf2": 2,
    "trn1": 2,
    "trn1n": 2,
    "trn2": 3,
    "trn2n": 3,
    "trn3": 4,
}


@cache
def _detect_generation() -> int:
    override = os.environ.get("COSMOS3_NEURON_CORE_GEN")
    if override:
        return int(override)
    try:
        from vllm_omni_neuron.lite_compat import get_platform_target

        target = str(get_platform_target()).lower()
    except Exception as exc:  # CPU-only environment without the runtime
        logger.info("vllm-omni-neuron: platform target unavailable (%s); assuming NeuronCore-v2", exc)
        return 2
    for key in sorted(_TARGET_TO_GEN, key=len, reverse=True):
        if target.startswith(key):
            return _TARGET_TO_GEN[key]
    logger.info("vllm-omni-neuron: unknown platform target %r; assuming NeuronCore-v2", target)
    return 2


# Called from code that torch.compile traces (kernel gates inside model forwards): evaluate it
# eagerly and bake the result into the graph instead of tracing the platform lookup.
@torch.compiler.assume_constant_result
def neuron_core_generation() -> int:
    return _detect_generation()


neuron_core_generation.cache_clear = _detect_generation.cache_clear  # type: ignore[attr-defined]


def supports_nki() -> bool:
    """NKI kernels need NeuronCore-v3+ (trn2/trn3); Inf2/Trn1 (v2) must take torch paths."""
    return neuron_core_generation() >= 3


def use_nki_kernels(tensor: torch.Tensor | None = None) -> bool:
    """Whether an NKI kernel may be used for ``tensor`` (or at all, when ``tensor`` is None)."""
    if os.environ.get("COSMOS3_EDGE_ATTN_IMPL", "auto").lower() == "torch":
        return False
    if neuron_core_generation() < 3:
        return False
    if tensor is not None and tensor.device.type == "cpu":
        return False
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ImportError:
        return False
    return bool(can_run_kernel(tensor if tensor is not None else "neuron"))
