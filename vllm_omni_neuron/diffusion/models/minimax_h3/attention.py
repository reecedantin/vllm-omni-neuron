# SPDX-License-Identifier: Apache-2.0
"""Attention for the MiniMax-H3 port, dispatched by NeuronCore generation.

MiniMax-H3 attention is full (non-causal) self-attention over one packed sequence with no mask: the request is a
single attention document. Two paths:

* **nki** (NeuronCore-v3+: Trn2 / Trn3, on device): the plugin's ``attention_cte`` flash kernel through Wan2.2's
  ``_nki_attend`` wrapper (the kernel the earlier trn2 FastH3 port found to be the single largest win).
* **torch** (CPU, NeuronCore-v2): fp32-softmax attention, blocked over queries on device so the score matrix never
  exists whole; ``scaled_dot_product_attention`` on the CPU.

Layout in and out: ``[B, S, H, D]`` (token-major, what the q/k/v projections produce).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm_omni_neuron.nc_generation import use_nki_kernels

QBLOCK = int(os.environ.get("MINIMAX_H3_ATTN_QBLOCK", "2048"))
ATTN_IMPL = os.environ.get("MINIMAX_H3_ATTN", "auto")  # auto | torch


def _torch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """[B, H, S, D] -> [B, H, S, D]; fp32 scores and softmax, P@V in the input dtype."""
    if (
        q.device.type == "cpu" and not torch.compiler.is_compiling()
    ):  # eager CPU (tests, the CPU oracle)
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    kt = k.transpose(-1, -2)
    sq = q.shape[-2]
    outs = []
    for s0 in range(0, sq, QBLOCK):  # static loop, unrolled into the compiled graph
        scores = torch.matmul(q[:, :, s0 : s0 + QBLOCK], kt).float() * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs.to(v.dtype), v))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)


def h3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` q/k/v (already normed and rotated) -> ``[B, S, H*D]`` attention output."""
    b, s, h, d = q.shape
    scale = d**-0.5
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]
    if ATTN_IMPL != "torch" and use_nki_kernels(q):
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

        out = _nki_attend(qh, kh, vh, scale)  # d-major [B, H, D, S]
        return out.permute(0, 3, 1, 2).reshape(b, s, h * d)
    out = _torch_attention(qh, kh, vh, scale)
    return out.transpose(1, 2).reshape(b, s, h * d)
