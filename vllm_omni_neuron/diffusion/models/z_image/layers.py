# SPDX-License-Identifier: Apache-2.0
"""Small shared building blocks for the Z-Image Neuron port (norms, RoPE, attention, TP helpers).

Everything here is plain tensor math that ``torch.compile`` lowers on NeuronCore-v3 and that
runs unchanged on the CPU, which is what the CPU unit tests and the fp32 oracle use.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

MASK_VALUE = -30000.0  # finite: -inf poisons fully masked rows and bf16 casts
ATTN_QBLOCK = int(os.environ.get("Z_IMAGE_ATTN_QBLOCK", "0"))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """diffusers / HF RMSNorm: fp32 statistics, cast back to the input dtype, then scale."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * weight


def layer_norm_noaffine(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = (xf - mu).pow(2).mean(-1, keepdim=True)
    return ((xf - mu) * torch.rsqrt(var + eps)).to(x.dtype)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Complex RoPE on interleaved (even, odd) pairs, computed in fp32 like the reference.

    ``x`` [B, S, H, D]; ``cos``/``sin`` [B, S, D/2] fp32.
    """
    b, s, h, d = x.shape
    xf = x.float().reshape(b, s, h, d // 2, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    c, sn = cos[:, :, None, :], sin[:, :, None, :]
    out = torch.stack((x0 * c - x1 * sn, x0 * sn + x1 * c), dim=-1)
    return out.reshape(b, s, h, d).to(x.dtype)


def apply_rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half`` RoPE. ``x`` [B, H, S, D]; ``cos``/``sin`` [S, D] (already duplicated)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos + rot * sin).to(x.dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax attention, [B, H, S, D] layout, fp32 softmax, optional additive fp32 bias
    broadcastable to [B, H, Sq, Sk]. Queries are optionally processed in static blocks
    (``Z_IMAGE_ATTN_QBLOCK``) so the full score matrix is never materialised."""
    kt = k.transpose(-1, -2)
    sq = q.shape[-2]
    qb = ATTN_QBLOCK if 0 < ATTN_QBLOCK < sq else sq
    outs = []
    for s0 in range(0, sq, qb):
        sc = torch.matmul(q[:, :, s0 : s0 + qb], kt).float() * scale
        if bias is not None:
            bb = bias if bias.shape[-2] == 1 else bias[..., s0 : s0 + qb, :]
            sc = sc + bb
        p = torch.softmax(sc, dim=-1).to(v.dtype)
        outs.append(torch.matmul(p, v))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)


def tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialised."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    if size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=1)
    return size, rank, group


def all_reduce(x: torch.Tensor, tp_size: int, group) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(x, group=group)
    return x


def shard(t: torch.Tensor, dim: int, tp_size: int, tp_rank: int) -> torch.Tensor:
    if tp_size == 1:
        return t
    n = t.shape[dim] // tp_size
    return t.narrow(dim, tp_rank * n, n).contiguous()
