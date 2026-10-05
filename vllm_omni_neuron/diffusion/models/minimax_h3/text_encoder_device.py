# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's Qwen3-VL-32B text conditioner on the NeuronCores, tensor-parallel over the stage's TP group.

Only what ``hidden_states[layer]`` needs runs on the device: the first ``layer`` (50) decoder layers of the language
model, prefill only (no KV cache, no final norm, no LM head). For a text-only prompt Qwen3-VL's interleaved mRoPE is
plain 1-D RoPE (all three position streams are equal), so the tables are the standard ``theta = 5e6`` ones, cast to the
activation dtype as transformers does.

* q / k / v / gate / up: column-parallel; o / down: row-parallel + all-reduce over the TP group (the DiT's group).
* GQA: 64 query / 8 KV heads. With ``tp >= 8`` each rank holds ``64 / tp`` query heads and the one KV head they read
  (replicated ``tp / 8`` times); with ``tp < 8`` each rank holds ``8 / tp`` KV heads.
* Host: tokenisation, the token-embedding lookup (mmap'd rows of the checkpoint, no 1.5 GB table on the device),
  RoPE tables and the causal mask. Prompts are padded to a length bucket (64 / 128 / 256 / 512 tokens) so every prompt
  in a bucket reuses one compiled graph; the pad rows sit after the real tokens, so causality hides them.

Every CP replica of the stage runs the encoder on its own TP group (same inputs, same result), so no broadcast is
needed. Numerics follow ``transformers``' Qwen3-VL text model: fp32-statistics RMSNorm, per-head q/k norm before RoPE,
compute-dtype QK^T, fp32 softmax, compute-dtype PV.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BUCKETS = (64, 128, 256, 512)


def bucket(n: int) -> int:
    """Padded prompt length: the smallest bucket that fits, else the exact length (one graph per long prompt)."""
    return next((b for b in BUCKETS if n <= b), n)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _linear(out_f: int, in_f: int, dtype) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=False, dtype=dtype)
    lin.weight.requires_grad_(False)
    return lin


class _Layer(nn.Module):
    def __init__(self, cfg, tp: int, tp_rank: int, dtype, reduce):
        super().__init__()
        H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        hid, inter = cfg.hidden_size, cfg.intermediate_size
        if H % tp or inter % tp or (tp >= KV and tp % KV) or (tp < KV and KV % tp):
            raise ValueError(f"tp={tp} does not shard {H} q heads / {KV} kv heads / ffn {inter}")
        self.reduce, self.head_dim = reduce, D
        self.q_heads = H // tp
        self.kv_heads = 1 if tp >= KV else KV // tp
        self.kv_index = tp_rank // (tp // KV) if tp >= KV else tp_rank * self.kv_heads
        self.input_layernorm = _RMSNorm(hid, cfg.rms_norm_eps, dtype)
        self.post_attention_layernorm = _RMSNorm(hid, cfg.rms_norm_eps, dtype)
        self.q_proj = _linear(self.q_heads * D, hid, dtype)
        self.k_proj = _linear(self.kv_heads * D, hid, dtype)
        self.v_proj = _linear(self.kv_heads * D, hid, dtype)
        self.o_proj = _linear(hid, self.q_heads * D, dtype)
        self.q_norm = _RMSNorm(D, cfg.rms_norm_eps, dtype)
        self.k_norm = _RMSNorm(D, cfg.rms_norm_eps, dtype)
        self.gate_proj = _linear(inter // tp, hid, dtype)
        self.up_proj = _linear(inter // tp, hid, dtype)
        self.down_proj = _linear(hid, inter // tp, dtype)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(b, t, self.q_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(b, t, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if self.q_heads > self.kv_heads:
            k = k.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
        probs = torch.softmax(scores.float() + mask, dim=-1).to(q.dtype)
        attn = torch.matmul(probs, v).transpose(1, 2).reshape(b, t, self.q_heads * self.head_dim)
        x = x + self.reduce(self.o_proj(attn))
        h = self.post_attention_layernorm(x)
        return x + self.reduce(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))


class _Stack(nn.Module):
    def __init__(self, cfg, num_layers: int, tp: int, tp_rank: int, dtype, reduce):
        super().__init__()
        self.layers = nn.ModuleList(
            _Layer(cfg, tp, tp_rank, dtype, reduce) for _ in range(num_layers)
        )

    def forward(self, x, cos, sin, mask):
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x  # == transformers' hidden_states[num_layers] (pre final norm)


class DeviceTextEncoder:
    """``encode(prompt) -> (1, N, hidden)`` (compute dtype, host tensor) on every rank of the TP group."""

    def __init__(
        self,
        model_path: str,
        layer: int,
        tp_size: int,
        tp_rank: int,
        tp_group,
        dtype=torch.bfloat16,
    ):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer

        t0 = time.time()
        self.dir = os.path.join(model_path, "text_encoder")
        self.cfg = cfg = AutoConfig.from_pretrained(self.dir).text_config
        if layer >= cfg.num_hidden_layers:
            raise ValueError(
                f"hidden_states[{layer}] needs more than the {cfg.num_hidden_layers} layers available"
            )
        self.layer, self.dtype, self.tp_size, self.tp_group = layer, dtype, tp_size, tp_group
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self.model = _Stack(cfg, layer, tp_size, tp_rank, dtype, self._reduce).eval()
        with open(os.path.join(self.dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._files: dict[str, object] = {}
        self._safe_open = safe_open
        self._load_shard(tp_rank)
        self._embed_key = "model.language_model.embed_tokens.weight"
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") or getattr(
            cfg, "rope_theta"
        )
        d = cfg.head_dim
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        self.device = torch.device("cpu")
        self._fn = self.model
        self._static: dict[int, tuple] = {}
        logger.info(
            "MiniMax-H3 device text encoder: %d layers, tp %d, %.2f B params per rank, loaded in %.1fs",
            layer,
            tp_size,
            sum(p.numel() for p in self.model.parameters()) / 1e9,
            time.time() - t0,
        )

    # -- weights ---------------------------------------------------------------------------------------------------
    def _slice(self, key: str):
        fn = self.weight_map[key]
        if fn not in self._files:
            self._files[fn] = self._safe_open(os.path.join(self.dir, fn), framework="pt")
        return self._files[fn].get_slice(key)

    def _load_shard(self, r: int) -> None:
        d = self.cfg.head_dim

        def put(param, src):
            if tuple(param.shape) != tuple(src.shape):
                raise ValueError(f"shape {tuple(param.shape)} vs checkpoint {tuple(src.shape)}")
            param.data = src.to(param.dtype).contiguous()

        for i, ly in enumerate(self.model.layers):
            p = f"model.language_model.layers.{i}."
            qh, kvh, kvi, inter = ly.q_heads, ly.kv_heads, ly.kv_index, ly.gate_proj.out_features
            put(
                ly.q_proj.weight,
                self._slice(p + "self_attn.q_proj.weight")[r * qh * d : (r + 1) * qh * d],
            )
            put(
                ly.k_proj.weight,
                self._slice(p + "self_attn.k_proj.weight")[kvi * d : (kvi + kvh) * d],
            )
            put(
                ly.v_proj.weight,
                self._slice(p + "self_attn.v_proj.weight")[kvi * d : (kvi + kvh) * d],
            )
            put(
                ly.o_proj.weight,
                self._slice(p + "self_attn.o_proj.weight")[:, r * qh * d : (r + 1) * qh * d],
            )
            put(ly.q_norm.weight, self._slice(p + "self_attn.q_norm.weight")[:])
            put(ly.k_norm.weight, self._slice(p + "self_attn.k_norm.weight")[:])
            put(
                ly.gate_proj.weight,
                self._slice(p + "mlp.gate_proj.weight")[r * inter : (r + 1) * inter],
            )
            put(
                ly.up_proj.weight,
                self._slice(p + "mlp.up_proj.weight")[r * inter : (r + 1) * inter],
            )
            put(
                ly.down_proj.weight,
                self._slice(p + "mlp.down_proj.weight")[:, r * inter : (r + 1) * inter],
            )
            put(ly.input_layernorm.weight, self._slice(p + "input_layernorm.weight")[:])
            put(
                ly.post_attention_layernorm.weight,
                self._slice(p + "post_attention_layernorm.weight")[:],
            )

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            dist.all_reduce(x, group=self.tp_group)
        return x

    # -- placement -------------------------------------------------------------------------------------------------
    def to(self, device) -> DeviceTextEncoder:
        self.device = torch.device(device)
        self.model.to(self.device)
        self._static.clear()
        return self

    def compile(self, backend: str, compiler_args: list[str]) -> None:
        opts = {"model_name": "minimax_h3_text_encoder", "compiler_args": list(compiler_args)}
        self._fn = torch.compile(
            self.model, backend=backend, options=opts, fullgraph=True, dynamic=False
        )

    # -- host side -------------------------------------------------------------------------------------------------
    def token_ids(self, prompt: str) -> list[int]:
        """t2va presentation: the prompt verbatim, no chat template, no special tokens."""
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def _inputs(self, ids: list[int], t: int):
        sl = self._slice(self._embed_key)
        rows = torch.cat([sl[i : i + 1] for i in ids], 0).to(self.dtype)
        x = torch.cat([rows, rows.new_zeros((t - len(ids), rows.shape[1]))], 0)[None]
        if t not in self._static:
            pos = torch.arange(t, dtype=torch.float32)
            emb = torch.cat([pos[:, None] * self.inv_freq[None]] * 2, dim=-1)
            mask = torch.full((t, t), -1e9, dtype=torch.float32).triu(1)[None, None]
            self._static[t] = tuple(
                a.contiguous().to(self.device)
                for a in (emb.cos().to(self.dtype), emb.sin().to(self.dtype), mask)
            )
        return (x.contiguous().to(self.device), *self._static[t])

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        ids = self.token_ids(prompt)
        out = self._fn(*self._inputs(ids, bucket(len(ids))))
        return out.to("cpu")[:, : len(ids)].contiguous()
