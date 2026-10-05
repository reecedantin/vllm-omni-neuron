# SPDX-License-Identifier: Apache-2.0
"""Alpamayo flow-matching "expert": a second decoder stack whose action tokens attend
non-causally over the VLM's KV cache (prompt + Chain-of-Causation) plus their own K/V.

Verified against the real checkpoint (not assumed): ``expert.layers.N.self_attn.{k,v}_proj`` are
``[kv_heads*head_dim, hidden_size] = [1024, 2048]`` -- the SAME ``kv_heads=8, head_dim=128`` as the
VLM's own text tower (``expert_cfg`` in config.json overrides only ``hidden_size``,
``num_attention_heads`` and ``intermediate_size``; the expert inherits ``num_hidden_layers``,
``num_key_value_heads`` and ``head_dim`` from ``self.vlm.config.text_config``). Upstream appends the
expert's K/V into the VLM's ``DynamicCache``, attends over the extended sequence, then crops the
cache back; here the expert reads the VLM's fixed-length cache and concatenates its own K/V
(:func:`.decode_backend.cross_step`) -- the same keys/values, with a static shape and no cache writes.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .decode_backend import cross_step
from .layers import rms_norm, rotate_half


def _tp_state(tp_group=None):
    if dist.is_initialized():
        group = tp_group if tp_group is not None else dist.group.WORLD
        return dist.get_world_size(group), dist.get_rank(group), group
    return 1, 0, None


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class _ExpertAttention(nn.Module):
    """Own Q/K/V/O at ``hidden_size`` (2048), K/V width forced to the VLM's ``kv_heads*head_dim``
    so the appended K/V slot into the shared cache is shape-compatible."""

    def __init__(self, ec: dict, tp_group=None):
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_group = _tp_state(tp_group)
        d, hd = ec["hidden_size"], ec["head_dim"]
        self.h_total, self.hk_total = ec["num_attention_heads"], ec["num_key_value_heads"]
        if self.h_total % self.tp_size or self.hk_total % self.tp_size:
            raise ValueError(
                f"TP size {self.tp_size} must divide both q_heads={self.h_total} "
                f"and kv_heads={self.hk_total}"
            )
        self.h, self.hk, self.hd = self.h_total // self.tp_size, self.hk_total // self.tp_size, hd
        bias = bool(ec.get("attention_bias", False))
        from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

        self.q_proj = ColumnParallelLinear(d, self.h_total * hd, bias=bias, tp_group=self.tp_group)
        self.k_proj = ColumnParallelLinear(d, self.hk_total * hd, bias=bias, tp_group=self.tp_group)
        self.v_proj = ColumnParallelLinear(d, self.hk_total * hd, bias=bias, tp_group=self.tp_group)
        self.o_proj = RowParallelLinear(self.h_total * hd, d, bias=bias, tp_group=self.tp_group)
        self.q_norm = _RMSNorm(hd, ec["rms_norm_eps"])
        self.k_norm = _RMSNorm(hd, ec["rms_norm_eps"])

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` ``[1, n_action_tokens, hidden]``; attends non-causally over the VLM layer's full
        fixed-length cache plus its own K/V (``bias`` masks the cache slots upstream's cropped
        cache would not hold). The cache is read, never written -- no crop needed between steps."""
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.hk, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.hk, self.hd).transpose(1, 2)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        o = cross_step(self.h // self.hk, self.hd**-0.5, q, k, v, k_cache, v_cache, bias)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, -1))


class _ExpertMLP(nn.Module):
    def __init__(self, ec: dict, tp_group=None):
        super().__init__()
        _, _, grp = _tp_state(tp_group)
        d, i = ec["hidden_size"], ec["intermediate_size"]
        from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

        self.gate_proj = ColumnParallelLinear(d, i, bias=False, tp_group=grp)
        self.up_proj = ColumnParallelLinear(d, i, bias=False, tp_group=grp)
        self.down_proj = RowParallelLinear(i, d, bias=False, tp_group=grp)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ExpertLayer(nn.Module):
    def __init__(self, ec: dict, tp_group=None):
        super().__init__()
        self.self_attn = _ExpertAttention(ec, tp_group)
        self.mlp = _ExpertMLP(ec, tp_group)
        self.input_layernorm = _RMSNorm(ec["hidden_size"], ec["rms_norm_eps"])
        self.post_attention_layernorm = _RMSNorm(ec["hidden_size"], ec["rms_norm_eps"])

    def forward(self, x, cos, sin, k_cache, v_cache, bias):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, k_cache, v_cache, bias)
        return x + self.mlp(self.post_attention_layernorm(x))


class Expert(nn.Module):
    """``forward(x, cos, sin, k_caches, v_caches, bias)`` -> the expert's final hidden state
    ``[1, n_tok, hidden]`` AFTER its final RMSNorm (upstream reads ``last_hidden_state`` of a
    ``Qwen3VLTextModel``, which is normed), one call per Euler step. ``k_caches``/``v_caches`` are
    the VLM's per-layer fixed-length caches (read-only here)."""

    def __init__(self, ec: dict, tp_group=None):
        super().__init__()
        self.ec = ec
        self.layers = nn.ModuleList(
            [_ExpertLayer(ec, tp_group) for _ in range(ec["num_hidden_layers"])]
        )
        self.norm = _RMSNorm(ec["hidden_size"], ec["rms_norm_eps"])

    def forward(self, x, cos, sin, k_caches, v_caches, bias: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, k_caches[i], v_caches[i], bias)
        return self.norm(x)


# ----------------------------------------------------------------------------------------
# Action in/out projections (Fourier time + per-coordinate encoders) -- same shape as
# alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2, re-implemented without hydra.
# ----------------------------------------------------------------------------------------


class _FourierEncoder(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0, bf16_freqs: bool = True):
        super().__init__()
        self.out_dim = dim
        # kept so a meta-device rebuild can regenerate `freqs` for real
        self.max_freq, self.bf16_freqs = max_freq, bf16_freqs
        self.register_buffer("freqs", fourier_freqs(dim, max_freq, bf16_freqs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 features regardless of ``x``'s dtype (the frequencies reach 100 x 2 pi)."""
        arg = x[..., None].float() * self.freqs.float() * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


def fourier_freqs(dim: int, max_freq: float, bf16: bool = True) -> torch.Tensor:
    """``[1, dim // 2]`` log-spaced frequencies. ``bf16=True`` ROUNDS them to bf16 like Alpamayo
    1.5 upstream: it builds ``FourierEncoderV2.freqs`` while the checkpoint's bf16 default dtype is
    active (verified: the buffer is bf16 -- ``[1, 1.671875, 2.78125, 4.65625, ...]`` -- even under
    ``from_pretrained(dtype=float32)``). Alpamayo 2 Super's fp32 model keeps them fp32."""
    f = torch.logspace(0, math.log10(max_freq), steps=dim // 2)
    return (f.to(torch.bfloat16).float() if bf16 else f.float())[None, :]


class _LayerRMSNorm(nn.Module):
    """Matches upstream ``action_in_proj.RMSNorm`` exactly (plain RMS, eps=1e-5)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class _MLPEncoder(nn.Module):
    """Matches upstream ``action_in_proj.MLPEncoder`` layer-for-layer: ``trunk`` is
    ``[Linear(in,hidden), SiLU, (RMSNorm, Linear, SiLU)*(n-1), RMSNorm, Linear(hidden,out)]`` -- the
    exact ``nn.Sequential`` index layout a real checkpoint's ``action_in_proj.encoder.trunk.{i}.*``
    keys were written at, so this must not reorder or merge any step."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        if num_enc_layers < 1:
            raise ValueError(f"num_enc_layers must be >= 1, got {num_enc_layers}")
        layers = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for i in range(num_enc_layers):
            if i < num_enc_layers - 1:
                layers += [
                    _LayerRMSNorm(hidden_size, eps=1e-5),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                ]
            else:
                layers += [_LayerRMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, outdim)]
        self.trunk = nn.Sequential(*layers)

    def forward(self, x):
        return self.trunk(x)


class ActionInProj(nn.Module):
    """``forward(x [B, T, action_dim], timesteps [B, 1]) -> [B, T, out_dim]``, matching upstream
    ``PerWaypointActionInProjV2`` tensor names (``sinus.{i}``, ``timestep_fourier_encoder``,
    ``encoder.trunk``, ``norm``) so a real checkpoint's ``action_in_proj.*`` loads unchanged."""

    def __init__(
        self,
        action_dim: int,
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
        bf16_freqs: bool = True,
    ):
        super().__init__()
        self.sinus = nn.ModuleList(
            [_FourierEncoder(num_fourier_feats, max_freq, bf16_freqs) for _ in range(action_dim)]
        )
        self.timestep_fourier_encoder = _FourierEncoder(num_fourier_feats, max_freq, bf16_freqs)
        num_input_feats = action_dim * num_fourier_feats + num_fourier_feats
        self.encoder = _MLPEncoder(num_input_feats, num_enc_layers, hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1]).repeat(1, t, 1)
        feats = torch.cat((action_feats, timestep_feats), dim=-1).to(self.norm.weight.dtype)
        return self.norm(self.encoder(feats.flatten(0, 1)).reshape(b, t, -1))
