# SPDX-License-Identifier: Apache-2.0
"""Attention and small layer helpers shared by the GR00T backbone and action head.

Everything here is plain tensor math that ``torch.compile`` lowers on any NeuronCore
generation. Softmax always runs in fp32. Masks are additive fp32 biases (0 = attend,
``MASK_VALUE`` = drop); a finite value keeps fully padded query rows NaN-free.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -30000.0


def _nki_enabled(q: torch.Tensor) -> bool:
    if q.device.type != "neuron" or os.environ.get("GR00T_NKI_ATTN", "0") != "1":
        return False
    from vllm_omni_neuron.nc_generation import use_nki_kernels

    return bool(use_nki_kernels(q))


def bias_from_keep(keep: torch.Tensor) -> torch.Tensor:
    """bool ``keep`` (any shape broadcastable to ``[B, H, Sq, Sk]``) -> fp32 additive bias."""
    return torch.where(keep, 0.0, MASK_VALUE).to(torch.float32)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
    allow_nki: bool = False,
) -> torch.Tensor:
    """q ``[B, H, Sq, D]``, k/v ``[B, Hk, Sk, D]`` (``H % Hk == 0``) -> ``[B, H, Sq, D]``.

    ``allow_nki``: unmasked call sites may use the plugin's NKI flash-attention kernel
    (NeuronCore-v3+ only; opt in with ``GR00T_NKI_ATTN=1``. Measured on trn2 for the ViT's
    4 x 256-token images: 34 ms vs 15 ms for the plain path, so it is off by default).
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if allow_nki and bias is None and q.shape[1] == k.shape[1] and _nki_enabled(q):
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

        return _nki_attend(q, k, v, scale).transpose(2, 3)
    h, hk = q.shape[1], k.shape[1]
    if h != hk:
        rep = h // hk
        b, _, s, d = k.shape
        k = k[:, :, None].expand(b, hk, rep, s, d).reshape(b, h, s, d)
        v = v[:, :, None].expand(b, hk, rep, s, d).reshape(b, h, s, d)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def row_parallel(lin: nn.Linear, x: torch.Tensor, group) -> torch.Tensor:
    """``lin(x)`` for a row-sharded ``lin``: partial matmul, all-reduce, then the (full) bias."""
    if group is None:
        return lin(x)
    import torch.distributed as dist

    w_t = getattr(lin, "weight_t", None)
    y = x @ w_t if w_t is not None else F.linear(x, lin.weight)
    dist.all_reduce(y, group=group)
    return y + lin.bias if lin.bias is not None else y


class PretransposedLinear(nn.Linear):
    """``nn.Linear`` whose matmul reads a contiguous ``[in, out]`` copy of the weight (``x @ W^T``
    without an in-graph transpose of ``W``). Same parameters, so state dicts are unchanged."""

    def forward(self, x):
        y = x @ self.weight_t
        return y + self.bias if self.bias is not None else y


def pretranspose_linears(module: nn.Module) -> int:
    """Switch every ``nn.Linear`` under ``module`` to :class:`PretransposedLinear` (call after any TP
    sharding). The ``[in, out]`` copy is a non-persistent buffer; the original weight is dropped from
    the device-resident set only in the sense that no graph reads it. Returns the count."""
    n = 0
    for m in module.modules():
        if type(m) is nn.Linear:
            m.register_buffer("weight_t", m.weight.detach().t().contiguous(), persistent=False)
            m.__class__ = PretransposedLinear
            n += 1
    return n


def pretranspose_status(module: nn.Module) -> tuple[int, int]:
    """(linears whose matmul reads a ``weight_t`` in step with their current, possibly sharded,
    weight; all linears) under ``module``. A layout check for logs and per-rank reports; call it on
    the host, before ``.to(device)``."""
    good = total = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            total += 1
            w_t = getattr(m, "weight_t", None)
            good += int(
                isinstance(m, PretransposedLinear)
                and w_t is not None
                and w_t.shape == m.weight.shape[::-1]
                and torch.equal(w_t[:, 0], m.weight[0])
            )
    return good, total


def shard_linear(lin: nn.Linear, dim: int, rank: int, size: int, parts: int = 1) -> None:
    """Keep this rank's slice of ``lin`` (``dim`` 0: output rows + bias; 1: input columns).

    ``parts``: the output is ``parts`` equal blocks laid side by side (e.g. a fused QKV projection);
    each block is sliced separately and the slices are concatenated in block order."""
    w = lin.weight.detach()
    b = lin.bias.detach() if (dim == 0 and lin.bias is not None) else None
    if dim == 0:
        blk = w.shape[0] // parts
        n = blk // size
        idx = torch.cat(
            [torch.arange(p * blk + rank * n, p * blk + (rank + 1) * n) for p in range(parts)]
        )
        lin.weight = nn.Parameter(w.index_select(0, idx).contiguous(), requires_grad=False)
        if b is not None:
            lin.bias = nn.Parameter(b.index_select(0, idx).contiguous(), requires_grad=False)
    else:
        n = w.shape[1] // size
        lin.weight = nn.Parameter(w.narrow(1, rank * n, n).contiguous(), requires_grad=False)
    if (
        getattr(lin, "weight_t", None) is not None
    ):  # keep a pretransposed copy in step with the shard
        lin.register_buffer("weight_t", lin.weight.detach().t().contiguous(), persistent=False)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF ``Qwen3VLTextRMSNorm``: fp32 statistics, cast back, then scale."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``F.gelu(x, approximate="tanh")`` written out: the Neuron Lite backend's gelu hook
    rejects the ``approximate`` keyword when Dynamo traces it."""
    return 0.5 * x * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """Exact ``F.gelu(x)`` (``nn.GELU()``), written out for the same reason."""
    return 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))
