# SPDX-License-Identifier: Apache-2.0
"""Dense masked definition for small inputs; the (separately tested) gather form above
``DENSE_MAX`` tokens, where an N^2 mask would not fit."""

import math

from vllm_omni_neuron.diffusion.models.flux3_action.neighborhood import (
    neighborhood_attention,
    neighborhood_attention_reference,
)

DENSE_MAX = 4096


def _na(q, k, v, kernel_size, is_causal):
    n = math.prod(q.shape[1:-2])
    fn = neighborhood_attention_reference if n <= DENSE_MAX else neighborhood_attention
    return fn(q, k, v, list(kernel_size), is_causal)


def na2d(q, k, v, kernel_size, is_causal=None, attention_kwargs=None, **_):
    return _na(q, k, v, kernel_size, is_causal)


def na3d(q, k, v, is_causal=None, kernel_size=None, attention_kwargs=None, **_):
    return _na(q, k, v, kernel_size, is_causal)
