# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text decoder for Neuron, as Qwen-Image 2.1's prompt encoder.

The pipeline needs one thing from the 8B Qwen3-VL: the hidden states after the LAST decoder
layer, before the final RMSNorm (upstream neutralizes that norm with a forward hook). So this
module is the decoder stack only: no LM head, no final norm. The token-embedding lookup runs on
the host (a 150k x 4096 table is 1.2 GB of HBM per rank for a gather the CPU does in
microseconds), and the device graph is ``forward(embeds, cos, sin, bias, deepstack)``.

Layers: RMSNorm -> GQA attention with per-head QK RMSNorm and rotate-half RoPE (Qwen3-VL's
interleaved mRoPE; for text tokens all three axes share one position, which reduces it to plain
RoPE) -> RMSNorm -> SwiGLU MLP. TP shards heads (Q/K/V/gate/up column, O/down row +
all-reduce). ``deepstack`` ``[n_ds, B, L, H]`` is added to the hidden states after the first
``n_ds`` layers (Qwen3-VL's DeepStack visual injection; zeros for a text-only prompt).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MASK_VALUE,
    all_reduce,
    attach_shard_loaders,
    attention,
    load_sharded,
    param,
    rms_norm,
    rope_half,
    tp_state,
)

CKPT_PREFIX = "model.language_model."


@dataclass
class Qwen3VLTextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: list = field(default_factory=lambda: [24, 20, 20])
    n_deepstack: int = 3

    @classmethod
    def from_model_dir(cls, model_path: str, subfolder: str = "text_encoder") -> Qwen3VLTextConfig:
        with open(os.path.join(model_path, subfolder, "config.json")) as f:
            full = json.load(f)
        t = full.get("text_config", full)
        rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
        vis = full.get("vision_config") or {}
        return cls(
            hidden_size=int(t["hidden_size"]),
            intermediate_size=int(t["intermediate_size"]),
            num_layers=int(t["num_hidden_layers"]),
            num_heads=int(t["num_attention_heads"]),
            num_kv_heads=int(t["num_key_value_heads"]),
            head_dim=int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]),
            vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 5000000.0))),
            mrope_section=list(rope.get("mrope_section", [24, 20, 20])),
            n_deepstack=len(vis.get("deepstack_visual_indexes", [])),
        )


def kv_shard_count(cfg: Qwen3VLTextConfig, tp: int) -> int:
    """How many ways the KV heads are split at ``tp``: ``tp`` itself up to the KV-head count, then
    the KV-head count (each KV head replicated on ``tp / num_kv_heads`` ranks, query heads still
    split ``tp`` ways)."""
    kv = min(tp, cfg.num_kv_heads)
    if cfg.num_heads % tp or cfg.num_kv_heads % kv or tp % kv:
        raise ValueError(
            f"tp={tp} does not fit {cfg.num_heads} query / {cfg.num_kv_heads} KV heads"
        )
    return kv


class _Layer(nn.Module):
    SHARD = {"q": 0, "k": 0, "v": 0, "o": 1, "gate": 0, "up": 0, "down": 1}

    def __init__(self, cfg: Qwen3VLTextConfig, tp: int, dtype):
        super().__init__()
        h, d = cfg.hidden_size, cfg.head_dim
        self.cfg, self.tp = cfg, tp
        kv_shards = kv_shard_count(
            cfg, tp
        )  # < tp: each KV head is replicated on tp // kv_shards ranks
        self.n_heads, self.n_kv = cfg.num_heads // tp, cfg.num_kv_heads // kv_shards
        self.ln1 = param((h,), dtype)
        self.ln2 = param((h,), dtype)
        self.q = param((cfg.num_heads * d, h), dtype, 0, tp)
        self.k = param((cfg.num_kv_heads * d, h), dtype, 0, kv_shards)
        self.v = param((cfg.num_kv_heads * d, h), dtype, 0, kv_shards)
        self.o = param((h, cfg.num_heads * d), dtype, 1, tp)
        self.q_norm = param((d,), dtype)
        self.k_norm = param((d,), dtype)
        self.gate = param((cfg.intermediate_size, h), dtype, 0, tp)
        self.up = param((cfg.intermediate_size, h), dtype, 0, tp)
        self.down = param((h, cfg.intermediate_size), dtype, 1, tp)

    def forward(self, x, cos, sin, bias, group):
        cfg, d = self.cfg, self.cfg.head_dim
        b, s, _ = x.shape
        hn = rms_norm(x, self.ln1, cfg.rms_norm_eps)
        q = rms_norm(
            F.linear(hn, self.q).view(b, s, self.n_heads, d), self.q_norm, cfg.rms_norm_eps
        )
        k = rms_norm(F.linear(hn, self.k).view(b, s, self.n_kv, d), self.k_norm, cfg.rms_norm_eps)
        v = F.linear(hn, self.v).view(b, s, self.n_kv, d)
        q, k = rope_half(q, cos, sin), rope_half(k, cos, sin)
        a = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), d**-0.5, bias)
        x = x + all_reduce(F.linear(a.transpose(1, 2).reshape(b, s, -1), self.o), self.tp, group)
        hn = rms_norm(x, self.ln2, cfg.rms_norm_eps)
        mlp = F.linear(F.silu(F.linear(hn, self.gate)) * F.linear(hn, self.up), self.down)
        return x + all_reduce(mlp, self.tp, group)


class NeuronQwen3VLTextEncoder(nn.Module):
    def __init__(self, cfg: Qwen3VLTextConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg, self.dtype = cfg, dtype
        self.tp, self.tp_rank, self.tp_group = tp_state()
        kv_shards = kv_shard_count(cfg, self.tp)
        self.layers = nn.ModuleList(_Layer(cfg, self.tp, dtype) for _ in range(cfg.num_layers))
        attach_shard_loaders(
            self,
            {
                f"layers.{i}.{n}": dim
                for i in range(cfg.num_layers)
                for n, dim in _Layer.SHARD.items()
            },
            self.tp,
        )
        if kv_shards < self.tp:
            # TP above the KV-head count: rank r holds query heads [r*Hq/tp, (r+1)*Hq/tp), which all
            # read KV head r // (tp / kv_shards); load that head's K/V shard on every such rank.
            from vllm_neuron.utils.weight_loader import (
                set_weight_loader,
                sharding_weight_loader,
                with_rank_override,
            )

            kv_rank = self.tp_rank // (self.tp // kv_shards)
            for i in range(cfg.num_layers):
                for n in ("k", "v"):
                    p = getattr(self.layers[i], n)
                    set_weight_loader(
                        p,
                        with_rank_override(
                            sharding_weight_loader(
                                shard_dim=0, shard_size=p.shape[0], num_shards=kv_shards
                            ),
                            kv_rank,
                        ),
                    )
        d = cfg.head_dim
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        object.__setattr__(self, "_inv_freq", inv)  # host-only (not moved by .to())
        object.__setattr__(self, "_embed", None)  # host-only token table, see load_weights

    def checkpoint_mappings(self) -> dict:
        m = {}
        for i in range(self.cfg.num_layers):
            p, c = f"layers.{i}", f"{CKPT_PREFIX}layers.{i}"
            m.update(
                {
                    f"{p}.ln1": f"{c}.input_layernorm.weight",
                    f"{p}.ln2": f"{c}.post_attention_layernorm.weight",
                    f"{p}.q": f"{c}.self_attn.q_proj.weight",
                    f"{p}.k": f"{c}.self_attn.k_proj.weight",
                    f"{p}.v": f"{c}.self_attn.v_proj.weight",
                    f"{p}.o": f"{c}.self_attn.o_proj.weight",
                    f"{p}.q_norm": f"{c}.self_attn.q_norm.weight",
                    f"{p}.k_norm": f"{c}.self_attn.k_norm.weight",
                    f"{p}.gate": f"{c}.mlp.gate_proj.weight",
                    f"{p}.up": f"{c}.mlp.up_proj.weight",
                    f"{p}.down": f"{c}.mlp.down_proj.weight",
                }
            )
        return m

    def load_weights(self, model_path: str, device="cpu", subfolder: str = "text_encoder") -> None:
        from safetensors import safe_open

        d = os.path.join(model_path, subfolder)
        load_sharded(self, d, self.checkpoint_mappings(), self.tp_rank, self.tp, device)
        key = f"{CKPT_PREFIX}embed_tokens.weight"
        index = os.path.join(d, "model.safetensors.index.json")
        fname = (
            json.load(open(index))["weight_map"][key]
            if os.path.exists(index)
            else "model.safetensors"
        )
        with safe_open(os.path.join(d, fname), "pt") as f:
            object.__setattr__(self, "_embed", f.get_tensor(key).to(self.dtype))

    # -- host helpers ---------------------------------------------------------------------
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.cpu(), self._embed)

    def rope_tables(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``position_ids`` ``[3, B, L]`` (t, h, w; equal for text) -> cos/sin ``[B, L, 1, D]``
        in the model dtype, matching HF ``Qwen3VLTextRotaryEmbedding`` (interleaved mRoPE)."""
        inv = self._inv_freq
        freqs = position_ids[..., None].float() * inv  # [3, B, L, D/2]
        f = freqs[0].clone()
        for dim, offset in ((1, 1), (2, 2)):  # interleave h / w into the t frequencies
            length = self.cfg.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            f[..., idx] = freqs[dim, ..., idx]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype)[:, :, None].contiguous(), emb.sin().to(self.dtype)[
            :, :, None
        ].contiguous()

    @staticmethod
    def causal_bias(valid: torch.Tensor) -> torch.Tensor:
        """``valid`` ``[B, L]`` bool -> ``[B, 1, L, L]`` fp32 causal + key-padding bias."""
        b, n = valid.shape
        allowed = torch.ones(n, n, dtype=torch.bool).tril()[None] & valid[:, None, :]
        allowed[:, :, 0] |= ~allowed.any(-1)
        return torch.where(allowed, 0.0, MASK_VALUE).float()[:, None].contiguous()

    # -- graph ----------------------------------------------------------------------------
    def forward(self, embeds, cos, sin, bias, deepstack=None):
        x = embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, bias, self.tp_group)
            if deepstack is not None and i < deepstack.shape[0]:
                x = x + deepstack[i]
        return x
