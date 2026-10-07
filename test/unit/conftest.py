# SPDX-License-Identifier: Apache-2.0
"""Unit tests run on CPU: set CPU mode before any plugin / vllm-omni import touches the runtime."""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("PJRT_DEVICE", "CPU")
