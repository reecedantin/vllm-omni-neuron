# SPDX-License-Identifier: Apache-2.0
"""int8 weight-only linear layers (W8A16, per-output-channel symmetric absmax).

A storage trick, not a speed trick: the matmul runs in the activation dtype against the int8 values
cast to that dtype (exact, since ``|q| <= 127`` is representable in bf16), and the per-channel fp32
scale is applied to the fp32 output, so no dequantized weight copy is ever materialised. On
NeuronCore-v2 (Inf2/Trn1) this roughly halves DiT HBM with no matmul speedup; on Trn2 use it only
when a model is still memory-bound. Measured accuracy cost on a 33B video DiT: ~1.5 pts of final
video rel-L2 vs the bf16 port (inside the parity band).

Usage::

    n, saved = quantize_linears_(model.blocks, include=lambda name, mod: "adaln" not in name)

Keep AdaLN / modulation projections in high precision (they are the quantization-sensitive layers);
bake them to host tables instead (:mod:`vllm_omni_neuron.diffusion.layers.modulation_tables`).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 127


def quantize_weight_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) weight -> (int8 ``q``, fp32 per-row ``scale``) with ``weight ~= q * scale[:, None]``."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / QMAX
    q = torch.round(w / scale[:, None]).clamp_(-QMAX, QMAX).to(torch.int8)
    return q, scale.contiguous()


def int8_linear(
    x: torch.Tensor, weight_q: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.linear`` against an int8 weight: (x @ q^T) in x.dtype, * scale in fp32, + bias, -> x.dtype."""
    y = F.linear(x, weight_q.to(x.dtype)).float() * w_scale
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


class Int8WeightOnlyLinear(nn.Module):
    """Drop-in for ``nn.Linear``: int8 ``weight_q`` (out, in) + fp32 ``w_scale`` (out,) buffers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_dtype: torch.dtype = torch.bfloat16,
        device=None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            "weight_q", torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "w_scale", torch.ones(out_features, dtype=torch.float32, device=device)
        )
        self.bias = (
            nn.Parameter(
                torch.zeros(out_features, dtype=bias_dtype, device=device), requires_grad=False
            )
            if bias
            else None
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8WeightOnlyLinear:
        q, scale = quantize_weight_int8(linear.weight)
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            bias_dtype=linear.weight.dtype if linear.bias is None else linear.bias.dtype,
            device=linear.weight.device,
        )
        mod.weight_q.copy_(q)
        mod.w_scale.copy_(scale)
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.detach())
        return mod

    def load_weight(self, weight: torch.Tensor) -> None:
        """Quantize a full-precision (out, in) checkpoint weight into this layer's buffers."""
        q, scale = quantize_weight_int8(weight)
        self.weight_q.copy_(q.to(self.weight_q.device))
        self.w_scale.copy_(scale.to(self.w_scale.device))

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.weight_q.float() * self.w_scale[:, None]).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, self.weight_q, self.w_scale, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, w8a16"


def _bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def quantize_linears_(
    module: nn.Module,
    include: Callable[[str, nn.Linear], bool] | None = None,
) -> tuple[int, int]:
    """Replace every ``nn.Linear`` under ``module`` (that ``include(name, mod)`` accepts) in place.

    Returns ``(#layers replaced, bytes saved)``. ``module`` itself is never replaced, only children.
    """
    targets = [
        (name, mod)
        for name, mod in module.named_modules()
        if isinstance(mod, nn.Linear) and name and (include is None or include(name, mod))
    ]
    saved = 0
    for name, lin in targets:
        parent_name, _, child = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        q = Int8WeightOnlyLinear.from_linear(lin)
        saved += _bytes(lin.weight) - _bytes(q.weight_q) - _bytes(q.w_scale)
        setattr(parent, child, q)
    return len(targets), saved
