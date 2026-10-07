# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoder for Z-Image on Neuron.

Z-Image conditions on ``Qwen3Model(...).hidden_states[-2]``: the residual stream after the
first ``num_hidden_layers - 1`` decoder layers (the last layer and the final norm are never
used, so they are not loaded). Prompts are right-padded and attention is causal, so a valid
token never attends to padding: running a shorter static bucket (``Z_IMAGE_TEXT_BUCKETS``)
instead of the pipeline's 512-token padding gives identical valid-token outputs.

The token-embedding lookup runs on the host (it is a gather over a 0.8 GB table), so the
device graph takes ``inputs_embeds`` and holds only the decoder layers.

``fp32_residual`` keeps the residual stream and the RMSNorms (statistics and weight multiply) in
fp32 while every matmul and the attention stay bf16. The output is the residual stream after 35
layers, which never passes the model's final norm, so plain bf16 accumulates rounding in it layer
after layer: 1.71 % from fp32 in bf16, 0.35 % with the fp32 residual (CPU, Z-Image's prompt
template). That is what makes the encoder accurate enough to run on the NeuronCore for Z-Image
base at CFG 4.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MASK_VALUE, all_reduce, apply_rope_half, attention, rms_norm, shard, tp_state

TEXT_BUCKETS = tuple(
    int(b) for b in os.environ.get("Z_IMAGE_TEXT_BUCKETS", "128,256,512").split(",")
)


class Qwen3EncConfig:
    def __init__(self, cfg: dict):
        self.hidden_size = int(cfg["hidden_size"])
        self.intermediate_size = int(cfg["intermediate_size"])
        self.num_layers = int(cfg["num_hidden_layers"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim") or self.hidden_size // self.num_heads)
        self.vocab_size = int(cfg["vocab_size"])
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
        rp = cfg.get("rope_parameters") or {}
        self.rope_theta = float(cfg.get("rope_theta") or rp.get("rope_theta") or 1e6)
        self.n_used = self.num_layers - 1  # hidden_states[-2]

    @classmethod
    def from_model_dir(cls, model_path: str, subfolder: str = "text_encoder") -> Qwen3EncConfig:
        with open(os.path.join(model_path, subfolder, "config.json")) as f:
            return cls(json.load(f))


def _p(*shape, dtype):
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class _Layer(nn.Module):
    def __init__(self, c: Qwen3EncConfig, tp: int, dtype):
        super().__init__()
        h, d = c.hidden_size, c.head_dim
        self.q_proj = _p(c.num_heads * d // tp, h, dtype=dtype)
        self.k_proj = _p(c.num_kv_heads * d // tp, h, dtype=dtype)
        self.v_proj = _p(c.num_kv_heads * d // tp, h, dtype=dtype)
        self.o_proj = _p(h, c.num_heads * d // tp, dtype=dtype)
        self.q_norm = _p(d, dtype=dtype)
        self.k_norm = _p(d, dtype=dtype)
        self.gate_proj = _p(c.intermediate_size // tp, h, dtype=dtype)
        self.up_proj = _p(c.intermediate_size // tp, h, dtype=dtype)
        self.down_proj = _p(h, c.intermediate_size // tp, dtype=dtype)
        self.input_layernorm = _p(h, dtype=dtype)
        self.post_attention_layernorm = _p(h, dtype=dtype)


_KEYMAP = {
    "self_attn.q_proj.weight": ("q_proj", 0),
    "self_attn.k_proj.weight": ("k_proj", 0),
    "self_attn.v_proj.weight": ("v_proj", 0),
    "self_attn.o_proj.weight": ("o_proj", 1),
    "self_attn.q_norm.weight": ("q_norm", None),
    "self_attn.k_norm.weight": ("k_norm", None),
    "mlp.gate_proj.weight": ("gate_proj", 0),
    "mlp.up_proj.weight": ("up_proj", 0),
    "mlp.down_proj.weight": ("down_proj", 1),
    "input_layernorm.weight": ("input_layernorm", None),
    "post_attention_layernorm.weight": ("post_attention_layernorm", None),
}


class NeuronQwen3Encoder(nn.Module):
    """``forward(inputs_embeds [B, S, H], cos [S, D], sin [S, D]) -> hidden_states[-2]``."""

    def __init__(
        self,
        cfg: Qwen3EncConfig,
        dtype: torch.dtype = torch.bfloat16,
        tp: tuple | None = None,
        fp32_residual: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.fp32_residual = fp32_residual
        self.tp_size, self.tp_rank, self.tp_group = tp if tp is not None else tp_state()
        if cfg.num_kv_heads % self.tp_size:
            raise ValueError(
                f"TP={self.tp_size} must divide num_key_value_heads={cfg.num_kv_heads}"
            )
        self.layers = nn.ModuleList([_Layer(cfg, self.tp_size, dtype) for _ in range(cfg.n_used)])
        self.embed_tokens: torch.Tensor | None = None  # host-side table
        inv = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def load_weights(self, model_path: str, subfolder: str = "text_encoder") -> None:
        from safetensors import safe_open

        folder = os.path.join(model_path, subfolder)
        seen = set()
        for fn in sorted(f for f in os.listdir(folder) if f.endswith(".safetensors")):
            with safe_open(os.path.join(folder, fn), framework="pt") as f:
                for key in f.keys():
                    if key == "model.embed_tokens.weight":
                        self.embed_tokens = f.get_tensor(key).to(self.dtype)
                        continue
                    if not key.startswith("model.layers."):
                        continue
                    rest = key[len("model.layers.") :]
                    idx, sub = rest.split(".", 1)
                    idx = int(idx)
                    if idx >= self.cfg.n_used:
                        continue
                    name, dim = _KEYMAP[sub]
                    t = f.get_tensor(key)
                    if dim is not None:
                        t = shard(t, dim, self.tp_size, self.tp_rank)
                    p = getattr(self.layers[idx], name)
                    if tuple(t.shape) != tuple(p.shape):
                        raise ValueError(
                            f"{key}: checkpoint {tuple(t.shape)} vs model {tuple(p.shape)}"
                        )
                    p.data = t.to(self.dtype).contiguous()
                    seen.add((idx, name))
        if self.embed_tokens is None or len(seen) != self.cfg.n_used * len(_KEYMAP):
            raise KeyError(
                f"text encoder: loaded {len(seen)} of {self.cfg.n_used * len(_KEYMAP)} layer tensors"
            )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.cpu(), self.embed_tokens)

    def rope(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(torch.arange(s, dtype=torch.float32), self.inv_freq.cpu().float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def causal_bias(s: int) -> torch.Tensor:
        return torch.triu(torch.full((s, s), MASK_VALUE), diagonal=1)[None, None]

    def forward(self, x, cos, sin, bias):
        c = self.cfg
        b, s, _ = x.shape
        hq, hk, d = c.num_heads // self.tp_size, c.num_kv_heads // self.tp_size, c.head_dim
        rep = hq // hk
        mm = x.dtype  # matmul dtype
        if self.fp32_residual:
            x = x.float()

        def norm(t, w):
            if self.fp32_residual:  # fp32 statistics AND weight multiply, then bf16 for the matmuls
                return rms_norm(t, w.float(), c.rms_norm_eps).to(mm)
            return rms_norm(t, w, c.rms_norm_eps)

        for L in self.layers:
            h = norm(x, L.input_layernorm)
            q = rms_norm(
                F.linear(h, L.q_proj).view(b, s, hq, d), L.q_norm, c.rms_norm_eps
            ).transpose(1, 2)
            k = rms_norm(
                F.linear(h, L.k_proj).view(b, s, hk, d), L.k_norm, c.rms_norm_eps
            ).transpose(1, 2)
            v = F.linear(h, L.v_proj).view(b, s, hk, d).transpose(1, 2)
            q, k = apply_rope_half(q, cos, sin), apply_rope_half(k, cos, sin)
            if rep > 1:
                k = k[:, :, None].expand(b, hk, rep, s, d).reshape(b, hq, s, d)
                v = v[:, :, None].expand(b, hk, rep, s, d).reshape(b, hq, s, d)
            o = attention(q, k, v, 1.0 / math.sqrt(d), bias=bias)
            o = all_reduce(
                F.linear(o.transpose(1, 2).reshape(b, s, hq * d), L.o_proj),
                self.tp_size,
                self.tp_group,
            )
            x = x + o.to(x.dtype)
            h = norm(x, L.post_attention_layernorm)
            m = F.linear(F.silu(F.linear(h, L.gate_proj)) * F.linear(h, L.up_proj), L.down_proj)
            x = x + all_reduce(m, self.tp_size, self.tp_group).to(x.dtype)
        return x


def pick_text_bucket(n: int) -> int:
    for b in TEXT_BUCKETS:
        if n <= b:
            return b
    return TEXT_BUCKETS[-1]
