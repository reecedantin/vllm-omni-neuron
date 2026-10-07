# SPDX-License-Identifier: Apache-2.0
"""Re-export of :mod:`vllm_omni_neuron.nc_generation` for the Cosmos3-Edge modules."""

from vllm_omni_neuron.nc_generation import neuron_core_generation, supports_nki, use_nki_kernels

__all__ = ["neuron_core_generation", "supports_nki", "use_nki_kernels"]
