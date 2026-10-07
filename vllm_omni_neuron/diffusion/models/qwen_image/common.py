# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Qwen-Image Neuron port: TP state, sharded parameters, norms, attention.

Attention dispatch by NeuronCore generation (``vllm_omni_neuron.nc_generation``):

* **torch** (CPU, NeuronCore-v2, and any masked call): plain ops lowered by ``torch.compile``,
  fp32 softmax, query rows processed in static blocks so the ``[B, H, Sq, Sk]`` score matrix is
  never whole.
* **nki** (NeuronCore-v3+, unmasked calls): the plugin's ``attention_cte`` kernel, via the
  Wan2.2 wrapper.

Layouts: q ``[B, H, Sq, D]``, k/v ``[B, Hk, Sk, D]``; ``bias`` is an additive fp32 tensor
broadcastable to ``[B, 1, Sq, Sk]`` (0 = attend, ``MASK_VALUE`` = masked).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.nc_generation import use_nki_kernels

MASK_VALUE = -30000.0  # finite: -inf poisons fully masked rows and bf16 casts
ATTN_QBLOCK = int(os.environ.get("QWEN_IMAGE_ATTN_QBLOCK", "1024"))
# Query-row blocking is only used up to this many query rows. Above it (2048x2048 output: 16,384
# target tokens) neuronx-cc 2.27 miscompiles the blocked target graph: from the second DiT layer on,
# every query block after the first comes out wrong (device rel-L2 ~0.7 vs a CPU-bf16 band of 1%),
# whatever the block size. The same graph unblocked matches the CPU at the bf16 band. 4,096 rows
# (1024x1024) are verified correct blocked on device. Read at trace time.
ATTN_QBLOCK_MAX_ROWS = int(os.environ.get("QWEN_IMAGE_ATTN_QBLOCK_MAX_ROWS", "4096"))


def tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialized.

    Under TP>1 the group's partition is registered with the compiler's replica-group registry so
    the in-graph all-reduces can be legalized.
    """
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError):
        return 1, 0, None
    if size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=1)
    return size, rank, group


def param(shape, dtype, shard_dim: int | None = None, tp: int = 1) -> nn.Parameter:
    shape = list(shape)
    if shard_dim is not None:
        if shape[shard_dim] % tp:
            raise ValueError(f"dim {shard_dim} of {tuple(shape)} is not divisible by tp={tp}")
        shape[shard_dim] //= tp
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


def attach_shard_loaders(module: nn.Module, shard_dims: dict[str, int], tp: int) -> None:
    """Attach ``vllm_neuron`` sharding loaders to the named parameters (no-op at TP=1)."""
    if tp == 1:
        return
    from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

    params = dict(module.named_parameters())
    for name, dim in shard_dims.items():
        p = params[name]
        set_weight_loader(
            p, sharding_weight_loader(shard_dim=dim, shard_size=p.shape[dim], num_shards=tp)
        )


def load_sharded(
    module: nn.Module, ckpt_dir: str, mappings: dict, rank: int, tp: int, device
) -> None:
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    ckpt = SafetensorsCheckpoint(ckpt_dir)
    result = ckpt.load_sharded_pipelined(rank, tp, module, mappings, torch.device(device))
    module.load_state_dict(result.state_dict, strict=True, assign=True)


def all_reduce(x: torch.Tensor, tp: int, group) -> torch.Tensor:
    if tp > 1:
        dist.all_reduce(x, group=group)
    return x


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """diffusers / HF RMSNorm: fp32 statistics, cast back to the input dtype, then scale."""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y if weight is None else y * weight


def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Affine-free LayerNorm with fp32 statistics."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh-approximated GELU in plain ops, computed in fp32. (``F.gelu`` is rebound by the Neuron
    lite runtime to a wrapper that Dynamo cannot trace.)"""
    xf = x.float()
    return (0.5 * xf * (1.0 + torch.tanh(0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)))).to(
        x.dtype
    )


def rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Complex rotation of adjacent channel pairs (``apply_rotary_emb_qwen(use_real=False)``).

    ``x`` ``[B, S, H, D]``; ``cos``/``sin`` ``[S, D/2]`` fp32. Computed in fp32, cast back.
    """
    xf = x.float().unflatten(-1, (-1, 2))
    xr, xi = xf[..., 0], xf[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).flatten(-2).to(x.dtype)


def rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half`` RoPE. ``x`` ``[B, S, H, D]``; ``cos``/``sin`` ``[B, S, 1, D]`` (model dtype)."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return x[:, :, None].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)


def torch_attention(
    q, k, v, scale: float, bias: torch.Tensor | None = None, qblock: int | None = None
):
    sq = q.shape[-2]
    if qblock is None:
        qblock = ATTN_QBLOCK if sq <= ATTN_QBLOCK_MAX_ROWS else 0
    n_rep = q.shape[1] // k.shape[1]
    k, v = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
    kt = k.transpose(-1, -2)
    step = qblock if qblock and qblock < sq else sq
    outs = []
    for s0 in range(0, sq, step):  # static loop, unrolled into the graph
        qb = q[:, :, s0 : s0 + step]
        scores = torch.matmul(qb, kt).float() * scale
        if bias is not None:
            b = bias if bias.shape[-2] == 1 else bias[..., s0 : s0 + qb.shape[-2], :]
            scores = scores + b
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs.to(v.dtype), v))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)


def attention(
    q, k, v, scale: float, bias: torch.Tensor | None = None, qblock: int | None = None
) -> torch.Tensor:
    if (
        bias is None
        and os.environ.get("QWEN_IMAGE_ATTN_IMPL", "auto") != "torch"
        and use_nki_kernels(q)
    ):
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

        n_rep = q.shape[1] // k.shape[1]
        return _nki_attend(q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep), scale).transpose(2, 3)
    return torch_attention(q, k, v, scale, bias, qblock)


def host(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    x = x.detach().to("cpu")
    return (x.to(dtype) if dtype is not None else x).contiguous()
