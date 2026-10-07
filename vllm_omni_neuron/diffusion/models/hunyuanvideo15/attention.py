# SPDX-License-Identifier: Apache-2.0
"""Attention for the HunyuanVideo-1.5 port, dispatched by NeuronCore generation.

Layouts: q ``[B, H, Sq, D]``, k/v ``[B, H, Sk, D]``; returns ``[B, H, Sq, D]`` in ``q.dtype``.

Masking is always a *key* mask: the joint sequence is ``[video | encoder valid | encoder pad]``
and only the padded encoder keys are hidden, via ``key_bias``, an additive fp32 ``[B, 1, 1, Sk]``
bias (0 valid, ``MASK_VALUE`` pad). The encoder tokens are ordered valid-first, so the valid keys
are always a contiguous prefix (what ``attention_cte``'s ``bound_min``/``bound_max`` need).

Paths:

* **torch** (CPU, NeuronCore-v2, or any shape the kernel rejects): fp32 softmax, query rows
  processed in static blocks of ``HV15_ATTN_QBLOCK`` so the ``[B, H, Sq, Sk]`` score matrix never
  exists whole.
* **nki** (NeuronCore-v3+): the plugin's ``attention_cte`` flash kernel (Wan2.2's), unmasked only
  for now; a masked call falls back to torch. ``HV15_ATTN_IMPL=torch`` forces torch everywhere.
"""

from __future__ import annotations

import os

import torch

from vllm_omni_neuron.nc_generation import use_nki_kernels

MASK_VALUE = -30000.0  # finite: -inf poisons fully-masked rows and bf16 casts
QBLOCK = int(os.environ.get("HV15_ATTN_QBLOCK", "2048"))
ATTN_IMPL = os.environ.get("HV15_ATTN_IMPL", "auto")


def key_padding_bias(valid: torch.Tensor) -> torch.Tensor:
    """``[B, Sk]`` bool validity -> ``[B, 1, 1, Sk]`` fp32 additive bias."""
    return torch.where(valid.bool(), 0.0, MASK_VALUE).to(torch.float32)[:, None, None, :]


def torch_attention(q, k, v, scale, key_bias=None, qblock: int | None = None):
    qblock = QBLOCK if qblock is None else qblock
    kt = k.transpose(-1, -2)
    sq = q.shape[-2]
    outs = []
    step = sq if not qblock or sq <= qblock else qblock
    for s0 in range(0, sq, step):  # static loop: unrolled into the compiled graph
        scores = torch.matmul(q[:, :, s0 : s0 + step], kt).float() * scale
        if key_bias is not None:
            scores = scores + key_bias
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs.to(v.dtype), v))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
    return out.to(q.dtype)


def _nki_attention(q, k, v, scale):
    from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

    return _nki_attend(q, k, v, scale).transpose(2, 3)  # kernel output is d-major [B, H, D, Sq]


def hv15_attention(q, k, v, scale, key_bias=None):
    if ATTN_IMPL != "torch" and key_bias is None and use_nki_kernels(q):
        return _nki_attention(q, k, v, scale)
    return torch_attention(q, k, v, scale, key_bias=key_bias)
