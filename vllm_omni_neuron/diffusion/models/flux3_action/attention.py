# SPDX-License-Identifier: Apache-2.0
"""Dense (unmasked, non-causal) attention for the FLUX 3 Action DiT, dispatched by NeuronCore generation.

The DiT attends over the whole joint sequence with no mask (upstream calls plain
``F.scaled_dot_product_attention``; padded caption tokens are attended to by design), so the call
site needs only one primitive:

* **nki** (NeuronCore-v3+, trn2): the vendor ``attention_cte`` flash kernel, the one Wan2.2 uses.
* **torch** (NeuronCore-v2, CPU, or ``FLUX3_ACTION_ATTN_IMPL=torch``): ``softmax(q k^T * scale) v``
  with an fp32 softmax, lowered by ``torch.compile``.

Layouts: q/k/v ``[B, H, S, D]``; returns ``[B, S, H * D]`` in ``q.dtype`` (the layout the output
projection consumes).
"""

from __future__ import annotations

import os

import torch

ATTN_IMPL = os.environ.get("FLUX3_ACTION_ATTN_IMPL", "auto").lower()


def _use_nki(t: torch.Tensor) -> bool:
    if ATTN_IMPL == "torch" or t.device.type == "cpu":
        return False
    from vllm_omni_neuron.nc_generation import use_nki_kernels

    if not use_nki_kernels(t):
        return False
    b, h, s, d = t.shape
    return b * h <= 512 and s <= 131072 and d <= 128


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)).float() * scale
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def _nki_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _wan_nki_attention

    b, h, s, d = q.shape
    q3 = (q * (d**-0.5)).reshape(b * h, s, d).contiguous()
    k3 = k.reshape(b * h, k.shape[2], d).contiguous()
    v3 = v.reshape(b * h, v.shape[2], d).contiguous()
    out = _wan_nki_attention(q3, k3, v3)  # [BH, D, S] (d-major)
    return out.reshape(b, h, d, s).permute(0, 1, 3, 2)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` x3 -> ``[B, S, H * D]``."""
    out = _nki_attention(q, k, v) if _use_nki(q) else torch_attention(q, k, v)
    b, h, s, d = out.shape
    return out.transpose(1, 2).reshape(b, s, h * d)
