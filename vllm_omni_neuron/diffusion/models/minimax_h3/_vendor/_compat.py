# SPDX-License-Identifier: Apache-2.0
"""Shims for the diffusers-main symbols the vendored MiniMax-H3 code imports but diffusers 0.38 lacks."""

from __future__ import annotations

import torch


class MiniMaxH3LoraLoaderMixin:
    """Stub: LoRA loading is not part of the Neuron port."""


# diffusers main, utils/torch_utils.py
_FP64_UNSUPPORTED_DEVICES = frozenset({"mps", "npu", "neuron"})
_INT64_UNSUPPORTED_DEVICES = frozenset({"mps", "npu", "neuron"})
_DTYPE_DOWNCAST = {torch.float64: torch.float32, torch.int64: torch.int32}
_DTYPE_UNSUPPORTED_DEVICES = {torch.float64: _FP64_UNSUPPORTED_DEVICES, torch.int64: _INT64_UNSUPPORTED_DEVICES}


def maybe_adjust_dtype_for_device(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    unsupported = _DTYPE_UNSUPPORTED_DEVICES.get(dtype)
    return _DTYPE_DOWNCAST[dtype] if unsupported and device.type in unsupported else dtype
