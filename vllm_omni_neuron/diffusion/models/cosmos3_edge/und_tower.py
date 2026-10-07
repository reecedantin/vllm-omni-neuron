# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 UND (reasoner) tower for Neuron: Cosmos3-Edge, Cosmos3-Nano and Cosmos3-Super.

The Edge transformer is a Mixture-of-Transformers: every one of its 28 layers has an UND
(text) half and a GEN (diffusion) half. The UND half is a Nemotron dense decoder layer
(causal GQA attention without QK-norm, ReLU^2 MLP, RMSNorm, interleaved mRoPE
``[24, 20, 20]`` / theta 1e8). It runs once per request and CFG branch, and its only output
is, per layer, the GEN-facing key (``k_norm_und_for_gen`` RMSNorm, then RoPE) and the raw
value. The GEN tower cross-attends to those.

Cosmos3-Nano / Cosmos3-Super (no ``backbone_type`` in the config) use upstream's
``Cosmos3LanguageModel`` instead: a Qwen3-VL text decoder (per-head QK RMSNorm before RoPE,
SiLU-gated MLP, mRoPE theta 5e6), and the GEN-facing key is the same normed, rotated key the
UND attention uses. :class:`EdgeTextConfig` reads which backbone a checkpoint has.

This module re-implements upstream's ``Cosmos3EdgeLanguageModel`` (vendored in
``_vendor/transformer_cosmos3_edge.py``) in the plugin's style: raw ``nn.Parameter`` weights
loaded with ``vllm_neuron`` sharding loaders, explicit TP collectives, and attention routed
through :mod:`.attention`, so the same code runs on NeuronCore-v2 (torch path) and v3+
(NKI kernels where they apply). Under TP the heads are split across ranks and every rank
returns only its local K/V heads, which is exactly the slice its GEN tower shard needs.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ._vendor.transformer_cosmos3 import (
    Qwen3VLTextRotaryEmbedding,
    _apply_rotary_pos_emb,
    compute_mrope_position_ids_text,
)
from .attention import edge_attention


def _tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialized.

    Under TP>1 the group's full partition is also registered with the Neuron compiler's mesh
    registry so the in-graph all-reduces can be legalized (see ``register_replica_groups``).
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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM RMSNorm math: fp32 statistics, cast back, then scale."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * weight


COSMOS3_EDGE_BACKBONE_TYPE = "cosmos3_edge_nemotron_dense"


class EdgeTextConfig:
    """The UND-tower subset of ``transformer/config.json``.

    ``backbone`` is ``"nemotron"`` (Cosmos3-Edge: ReLU^2 MLP, no UND QK-norm, separate
    ``k_norm_und_for_gen``) or ``"qwen3"`` (Cosmos3-Nano / Super: SiLU-gated MLP, UND QK-norm).
    """

    def __init__(self, cfg: dict):
        backbone_type = cfg.get("backbone_type")
        if backbone_type is None:
            self.backbone = "qwen3"
        elif backbone_type == COSMOS3_EDGE_BACKBONE_TYPE:
            self.backbone = "nemotron"
        else:
            raise ValueError(f"unsupported Cosmos3 backbone_type={backbone_type!r}")
        self.gated_mlp = self.backbone == "qwen3"
        self.und_qk_norm = self.backbone == "qwen3" and bool(cfg.get("qk_norm_for_text", True))
        self.hidden_size = int(cfg.get("hidden_size", 2048))
        self.intermediate_size = int(cfg.get("intermediate_size", 9216))
        self.num_layers = int(cfg.get("num_hidden_layers", 28))
        self.num_heads = int(cfg.get("num_attention_heads", 16))
        self.num_kv_heads = int(cfg.get("num_key_value_heads", 8))
        self.head_dim = int(cfg.get("head_dim", 128))
        self.vocab_size = int(cfg.get("vocab_size", 131072))
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(cfg.get("rope_theta", 1e8))
        rs = cfg.get("rope_scaling") or {}
        self.mrope_section = list(rs.get("mrope_section", cfg.get("rope_axes_dim", [24, 20, 20])))
        self.use_und_k_norm_for_gen = self.backbone == "nemotron" and bool(
            cfg.get("use_und_k_norm_for_gen", True)
        )
        self.sound_gen = bool(cfg.get("sound_gen", False))

    def kv_heads_local(self, tp: int) -> int:
        """K/V heads per rank. When TP exceeds the K/V head count (Super at TP=16) every rank
        holds ONE full K/V head, shared by ``tp // num_kv_heads`` neighbouring ranks."""
        if tp <= self.num_kv_heads:
            if self.num_kv_heads % tp:
                raise ValueError(f"tp_size={tp} must divide num_kv_heads={self.num_kv_heads}")
            return self.num_kv_heads // tp
        if tp % self.num_kv_heads:
            raise ValueError(f"tp_size={tp} must be a multiple of num_kv_heads={self.num_kv_heads}")
        return 1

    @classmethod
    def from_model_dir(cls, model_path: str) -> EdgeTextConfig:
        with open(os.path.join(model_path, "transformer", "config.json")) as f:
            return cls(json.load(f))


def _sharded(shape, shard_dim, tp, dtype):
    shape = list(shape)
    shape[shard_dim] //= tp
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


def _param(shape, dtype):
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


def attach_tp_loaders(layers, tp_size: int, cfg: EdgeTextConfig) -> None:
    """Sharding loaders for one tower's layers (Q/K/V/up/gate column-, O/down row-parallel).

    K/V: plain row shards while ``tp <= num_kv_heads``; above that, rank ``r`` loads the whole
    head ``r * num_kv_heads // tp`` (KV-head replication), which is the head its local query
    heads belong to under GQA.
    """
    if tp_size == 1:
        return
    from vllm_neuron.utils.weight_loader import (
        SafetensorsWeightLoader,
        set_weight_loader,
        sharding_weight_loader,
    )

    d, nkv = cfg.head_dim, cfg.num_kv_heads
    replicate = tp_size > nkv

    def kv_rows(slices, rank):
        head = rank * nkv // tp_size
        return slices[0][head * d : (head + 1) * d, :]

    for layer in layers:
        for name, dim in (
            ("q_weight", 0),
            ("k_weight", 0),
            ("v_weight", 0),
            ("o_weight", 1),
            ("up_weight", 0),
            ("gate_weight", 0),
            ("down_weight", 1),
        ):
            prm = getattr(layer, name, None)
            if prm is None:
                continue
            if replicate and name in ("k_weight", "v_weight"):
                set_weight_loader(prm, SafetensorsWeightLoader(transform=kv_rows))
            else:
                set_weight_loader(
                    prm,
                    sharding_weight_loader(
                        shard_dim=dim, shard_size=prm.shape[dim], num_shards=tp_size
                    ),
                )


def mlp_forward(layer, hn: torch.Tensor) -> torch.Tensor:
    """MLP of one layer up to (excluding) the row-parallel all-reduce.

    SiLU-gated (Qwen3: Nano / Super) or ReLU^2 (Nemotron: Edge)."""
    if layer.cfg.gated_mlp:
        return F.linear(
            F.silu(F.linear(hn, layer.gate_weight)) * F.linear(hn, layer.up_weight),
            layer.down_weight,
        )
    mlp = F.relu(F.linear(hn, layer.up_weight))
    return F.linear(mlp * mlp, layer.down_weight)


class EdgeUndLayer(nn.Module):
    def __init__(self, cfg: EdgeTextConfig, tp_size: int, dtype: torch.dtype):
        super().__init__()
        h, d = cfg.hidden_size, cfg.head_dim
        self.cfg = cfg
        self.tp_size = tp_size
        self.n_heads = cfg.num_heads // tp_size
        self.n_kv = cfg.kv_heads_local(tp_size)
        self.q_weight = _sharded((cfg.num_heads * d, h), 0, tp_size, dtype)
        self.k_weight = _param((self.n_kv * d, h), dtype)
        self.v_weight = _param((self.n_kv * d, h), dtype)
        self.o_weight = _sharded((h, cfg.num_heads * d), 1, tp_size, dtype)
        if cfg.use_und_k_norm_for_gen:
            self.k_norm_gen_weight = nn.Parameter(torch.ones(d, dtype=dtype), requires_grad=False)
        if cfg.und_qk_norm:
            self.q_norm_weight = nn.Parameter(torch.ones(d, dtype=dtype), requires_grad=False)
            self.k_norm_weight = nn.Parameter(torch.ones(d, dtype=dtype), requires_grad=False)
        self.input_norm_weight = nn.Parameter(torch.ones(h, dtype=dtype), requires_grad=False)
        self.post_norm_weight = nn.Parameter(torch.ones(h, dtype=dtype), requires_grad=False)
        self.up_weight = _sharded((cfg.intermediate_size, h), 0, tp_size, dtype)
        if cfg.gated_mlp:
            self.gate_weight = _sharded((cfg.intermediate_size, h), 0, tp_size, dtype)
        self.down_weight = _sharded((h, cfg.intermediate_size), 1, tp_size, dtype)

    def _all_reduce(self, x, group):
        if self.tp_size > 1:
            dist.all_reduce(x, group=group)
        return x

    def forward(self, x, cos, sin, key_bias, group):
        cfg, d = self.cfg, self.cfg.head_dim
        b, s, _ = x.shape
        hn = rms_norm(x, self.input_norm_weight, cfg.rms_norm_eps)
        q = F.linear(hn, self.q_weight).view(b, s, self.n_heads, d)
        k = F.linear(hn, self.k_weight).view(b, s, self.n_kv, d)
        v = F.linear(hn, self.v_weight).view(b, s, self.n_kv, d)
        if cfg.und_qk_norm:
            q = F.rms_norm(q, (d,), self.q_norm_weight, eps=cfg.rms_norm_eps)
            k = F.rms_norm(k, (d,), self.k_norm_weight, eps=cfg.rms_norm_eps)
        q_r, k_r = _apply_rotary_pos_emb(q, k, cos, sin)
        if cfg.use_und_k_norm_for_gen:
            k_gen = F.rms_norm(k, (d,), self.k_norm_gen_weight, eps=cfg.rms_norm_eps)
            _, k_gen = _apply_rotary_pos_emb(q, k_gen, cos, sin)
        else:
            k_gen = k_r
        attn = edge_attention(
            q_r.transpose(1, 2),
            k_r.transpose(1, 2),
            v.transpose(1, 2),
            d**-0.5,
            causal=True,
            key_bias=key_bias,
            qk_fp32=True,  # <= a few hundred tokens, once per request: keep the reasoner exact
        ).transpose(1, 2)
        x = x + self._all_reduce(F.linear(attn.reshape(b, s, -1), self.o_weight), group)
        hn = rms_norm(x, self.post_norm_weight, cfg.rms_norm_eps)
        x = x + self._all_reduce(mlp_forward(self, hn), group)
        return x, k_gen, v


class NeuronCosmos3EdgeUND(nn.Module):
    """UND tower: ``forward(input_ids, cos, sin, key_bias) -> (k_0..k_{L-1}, v_0..v_{L-1})``.

    ``input_ids`` ``[B, S]`` (right-padded to a fixed bucket so one graph serves many prompts),
    ``cos``/``sin`` ``[B, S, 1, head_dim]`` from :meth:`rope_tables`, ``key_bias`` an optional
    ``[B, 1, 1, S]`` padding bias (causal attention already hides right padding from real
    tokens, so it only matters for the padded rows). K/V are ``[B, S, kv_heads/tp, head_dim]``.
    """

    def __init__(self, cfg: EdgeTextConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.tp_size, self.tp_rank, self.tp_group = _tp_state()
        cfg.kv_heads_local(self.tp_size)  # validates the TP degree
        self.embed_weight = nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=dtype), requires_grad=False
        )
        self.layers = nn.ModuleList(
            EdgeUndLayer(cfg, self.tp_size, dtype) for _ in range(cfg.num_layers)
        )
        # host-side rotary (pure math, never moved to the device)
        # (bypasses nn.Module registration so .to(device) leaves it on the host)
        object.__setattr__(
            self,
            "_rotary_host",
            Qwen3VLTextRotaryEmbedding(
                head_dim=cfg.head_dim, rope_theta=cfg.rope_theta, mrope_section=cfg.mrope_section
            ),
        )
        self._attach_weight_loaders()

    # -- weights --------------------------------------------------------------------------
    def _attach_weight_loaders(self) -> None:
        attach_tp_loaders(self.layers, self.tp_size, self.cfg)

    def checkpoint_mappings(self) -> dict[str, str]:
        m = {"embed_weight": "embed_tokens.weight"}
        for i in range(self.cfg.num_layers):
            p, c = f"layers.{i}", f"layers.{i}"
            m.update(
                {
                    f"{p}.q_weight": f"{c}.self_attn.to_q.weight",
                    f"{p}.k_weight": f"{c}.self_attn.to_k.weight",
                    f"{p}.v_weight": f"{c}.self_attn.to_v.weight",
                    f"{p}.o_weight": f"{c}.self_attn.to_out.weight",
                    f"{p}.input_norm_weight": f"{c}.input_layernorm.weight",
                    f"{p}.post_norm_weight": f"{c}.post_attention_layernorm.weight",
                    f"{p}.up_weight": f"{c}.mlp.up_proj.weight",
                    f"{p}.down_weight": f"{c}.mlp.down_proj.weight",
                }
            )
            if self.cfg.use_und_k_norm_for_gen:
                m[f"{p}.k_norm_gen_weight"] = f"{c}.self_attn.k_norm_und_for_gen.weight"
            if self.cfg.und_qk_norm:
                m[f"{p}.q_norm_weight"] = f"{c}.self_attn.norm_q.weight"
                m[f"{p}.k_norm_weight"] = f"{c}.self_attn.norm_k.weight"
            if self.cfg.gated_mlp:
                m[f"{p}.gate_weight"] = f"{c}.mlp.gate_proj.weight"
        return m

    def load_weights(self, model_path: str, device: torch.device | str = "cpu") -> None:
        """Load this rank's shard from ``<model_path>/transformer`` (needs torch.distributed)."""
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        ckpt = SafetensorsCheckpoint(os.path.join(model_path, "transformer"))
        result = ckpt.load_sharded_pipelined(
            self.tp_rank, self.tp_size, self, self.checkpoint_mappings(), torch.device(device)
        )
        self.load_state_dict(result.state_dict, strict=True, assign=True)

    # -- host helpers ---------------------------------------------------------------------
    def rope_tables(self, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``[B, S, 1, D]`` on the host, matching upstream ``_compute_rope_freqs``
        (real tokens get positions 0..n-1 on all three mRoPE axes, padding gets 0)."""
        b, s = text_mask.shape
        rows = []
        for i in range(b):
            n = int(text_mask[i].sum().item())
            pos, _ = compute_mrope_position_ids_text(n, temporal_offset=0)
            if n < s:
                pos = torch.cat([pos, torch.zeros(3, s - n, dtype=pos.dtype)], dim=1)
            rows.append(pos)
        pos_ids = torch.stack(rows, dim=1)  # [3, B, S]
        cos, sin = self._rotary_host(torch.empty(0, dtype=self.dtype), position_ids=pos_ids)
        return cos.unsqueeze(2).contiguous(), sin.unsqueeze(2).contiguous()

    # -- forward --------------------------------------------------------------------------
    def forward(self, input_ids, cos, sin, key_bias=None):
        x = F.embedding(input_ids, self.embed_weight)
        ks, vs = [], []
        for layer in self.layers:
            x, k, v = layer(x, cos, sin, key_bias, self.tp_group)
            ks.append(k)
            vs.append(v)
        return (*ks, *vs)
