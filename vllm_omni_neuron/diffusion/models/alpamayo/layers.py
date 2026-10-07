# SPDX-License-Identifier: Apache-2.0
"""Shared tensor math for the Alpamayo backbone/expert: plain (non-TP, non-decode-cache) attention
for the vision tower, RMSNorm, rotate_half, and the GELU-tanh the Lite backend's compiled gelu hook
rejects (the ``approximate`` keyword) under dynamo."""

from __future__ import annotations

import math

import torch

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``F.gelu(x, approximate="tanh")`` written out: the Lite backend's gelu hook rejects the
    ``approximate`` keyword when Dynamo traces it."""
    return 0.5 * x * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def attention_full(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Plain full (non-causal, non-cached) attention for the vision tower: q/k/v
    ``[B, H, S, D]`` (``H`` already equal -- the vision tower has no GQA), fp32 softmax."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)
