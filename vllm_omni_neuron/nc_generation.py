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

Detection order:

1. ``VLLM_OMNI_NEURON_CORE_GEN=<int>`` (legacy alias ``COSMOS3_NEURON_CORE_GEN``) forces a generation.
2. The Lite platform target (honours ``NEURON_PLATFORM_TARGET_OVERRIDE``, else NRT auto-detect).
3. The Neuron driver's sysfs architecture record (``arch_type`` ``NDv2``/``NDv3``/...,
   ``device_name``), which needs no runtime and no visible cores.
4. Otherwise NeuronCore-v2 (the conservative choice: torch paths compile everywhere), with a
   warning, since on a Trn2 host this silently disables every NKI kernel.

``VLLM_OMNI_NEURON_ATTN_IMPL=torch`` (legacy alias ``COSMOS3_EDGE_ATTN_IMPL=torch``) makes
``use_nki_kernels`` return False everywhere.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from functools import cache

import torch

logger = logging.getLogger(__name__)

GEN_ENV = "VLLM_OMNI_NEURON_CORE_GEN"
_GEN_ENV_LEGACY = "COSMOS3_NEURON_CORE_GEN"
ATTN_IMPL_ENV = "VLLM_OMNI_NEURON_ATTN_IMPL"
_ATTN_IMPL_ENV_LEGACY = "COSMOS3_EDGE_ATTN_IMPL"

# Platform target (libtorch-neuronx-lite / NRT naming) -> NeuronCore generation.
_TARGET_TO_GEN = {
    "inf2": 2,
    "trn1": 2,
    "trn1n": 2,
    "trn2": 3,
    "trn2n": 3,
    "trn3": 4,
}

# Neuron driver sysfs: one ``info/architecture`` directory per device.
SYSFS_ARCH_GLOB = "/sys/devices/virtual/neuron_device/neuron*/info/architecture"

# ``device_name`` values -> generation (``arch_type`` ``NDv<N>`` is preferred when present).
_DEVICE_NAME_TO_GEN = {
    "inferentia2": 2,
    "trainium": 2,
    "trainium1": 2,
    "trainium2": 3,
    "trainium3": 4,
}


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def generation_from_target(target: str) -> int | None:
    """Map a platform target string (``trn2``, ``inf2``, ``trn1n``...) to a generation."""
    target = str(target).strip().lower()
    for key in sorted(_TARGET_TO_GEN, key=len, reverse=True):
        if target.startswith(key):
            return _TARGET_TO_GEN[key]
    return None


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def generation_from_sysfs(arch_glob: str | None = None) -> int | None:
    """Read the generation from the Neuron driver's sysfs record, or None when absent."""
    for arch_dir in sorted(glob.glob(arch_glob or SYSFS_ARCH_GLOB)):
        arch_type = _read(os.path.join(arch_dir, "arch_type"))
        m = re.fullmatch(r"NDv(\d+)", arch_type, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        name = _read(os.path.join(arch_dir, "device_name")).lower()
        if name in _DEVICE_NAME_TO_GEN:
            return _DEVICE_NAME_TO_GEN[name]
        gen = generation_from_target(_read(os.path.join(arch_dir, "instance_type")))
        if gen is not None:
            return gen
    return None


@cache
def _detect_generation() -> int:
    override = _env_first(GEN_ENV, _GEN_ENV_LEGACY)
    if override:
        return int(override)
    target = None
    try:
        from vllm_omni_neuron.lite_compat import get_platform_target

        target = str(get_platform_target())
    except Exception as exc:  # CPU-only environment, or the runtime cannot initialise
        logger.info("vllm-omni-neuron: platform target unavailable (%s); trying sysfs", exc)
    if target is not None:
        gen = generation_from_target(target)
        if gen is not None:
            return gen
        logger.info("vllm-omni-neuron: unknown platform target %r; trying sysfs", target)
    gen = generation_from_sysfs()
    if gen is not None:
        return gen
    logger.warning(
        "vllm-omni-neuron: NeuronCore generation undetectable; assuming NeuronCore-v2 (NKI kernels "
        "disabled). Set %s=3 on Trn2.",
        GEN_ENV,
    )
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


def nki_disabled_by_env() -> bool:
    impl = _env_first(ATTN_IMPL_ENV, _ATTN_IMPL_ENV_LEGACY) or "auto"
    return impl.lower() == "torch"


def use_nki_kernels(tensor: torch.Tensor | None = None) -> bool:
    """Whether an NKI kernel may be used for ``tensor`` (or at all, when ``tensor`` is None)."""
    if nki_disabled_by_env():
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
