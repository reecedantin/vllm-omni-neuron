# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo-1.5 MLLM prompt encoder (Qwen2.5-VL-7B text tower) on NeuronCores.

The pipeline only reads ``hidden_states[-3]`` of ``Qwen2_5_VLTextModel``: the output of decoder layer
``L - 3`` (0-based), before the final norm. So only the first ``L - 2`` layers (26 of 28) run, and the
final norm and the LM head are never needed.

* **TP by heads**, Megatron style: ``q/k/v_proj`` column-parallel by head (7 query heads and their one
  KV head per rank at TP=4: 28 query / 4 KV heads), ``o_proj`` row-parallel + all-reduce, the SwiGLU MLP
  column/row-parallel + all-reduce. Requires ``num_key_value_heads % tp == 0``.
* **Text groups**: the world is partitioned into contiguous groups of ``tp`` ranks (one Trn2 chip at
  TP=4), independent of the DiT's TP/CP/CFG groups. The prompts of a request (positive, negative) are
  dealt across the groups; every group runs the same number of encodes so the collectives stay SPMD,
  and each embedding is broadcast from its group's leader over the world's host group.
* **Host work**: tokenization and the embedding lookup (rows read straight from the safetensors file,
  so no rank holds the 1.1 GB table); the device runs ``L - 2`` decoder layers in fixed-shape graphs of
  ``HV15_TEXT_LAYERS_PER_GRAPH`` layers (identical graphs share one NEFF).
* Text-only input: Qwen2.5-VL's M-RoPE with equal temporal/height/width positions is the standard
  1D RoPE, positions ``0..S-1``. Attention is causal plus a key-padding mask (right padding), as
  ``transformers`` builds it.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

MASK_VALUE = -30000.0
LAYERS_PER_GRAPH = int(os.environ.get("HV15_TEXT_LAYERS_PER_GRAPH", "13"))
TEXT_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


def _p(*shape, dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def rms_norm(x, weight, eps):  # transformers Qwen2RMSNorm
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.arange(seq, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def attention_bias(mask: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` validity -> ``[B, 1, S, S]`` fp32 additive causal + key-padding bias."""
    s = mask.shape[1]
    causal = torch.ones(s, s, dtype=torch.bool).tril()
    ok = causal[None] & mask.bool()[:, None, :]
    return torch.where(ok, 0.0, MASK_VALUE).to(torch.float32)[:, None]


class QwenTextLayer(nn.Module):
    """One Qwen2 decoder layer, TP-sharded (parameter names follow the checkpoint)."""

    def __init__(self, cfg: dict, dtype, tp: int):
        super().__init__()
        h, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        d = h // nh
        self.nh, self.nkv, self.d, self.eps = nh // tp, nkv // tp, d, cfg["rms_norm_eps"]
        ff = cfg["intermediate_size"] // tp
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = _p(h, dtype=dtype)
        self.post_attention_layernorm = nn.Module()
        self.post_attention_layernorm.weight = _p(h, dtype=dtype)
        sa = self.self_attn = nn.Module()
        for name, rows in (
            ("q_proj", self.nh * d),
            ("k_proj", self.nkv * d),
            ("v_proj", self.nkv * d),
        ):
            m = nn.Module()
            m.weight, m.bias = _p(rows, h, dtype=dtype), _p(rows, dtype=dtype)
            setattr(sa, name, m)
        sa.o_proj = nn.Module()
        sa.o_proj.weight = _p(h, self.nh * d, dtype=dtype)
        mlp = self.mlp = nn.Module()
        for name, shape in (("gate_proj", (ff, h)), ("up_proj", (ff, h)), ("down_proj", (h, ff))):
            m = nn.Module()
            m.weight = _p(*shape, dtype=dtype)
            setattr(mlp, name, m)

    def forward(self, x, cos, sin, bias, group=None):
        b, s, _ = x.shape
        sa, d = self.self_attn, self.d
        hn = rms_norm(x, self.input_layernorm.weight, self.eps)
        q = (
            torch.nn.functional.linear(hn, sa.q_proj.weight, sa.q_proj.bias)
            .view(b, s, self.nh, d)
            .transpose(1, 2)
        )
        k = (
            torch.nn.functional.linear(hn, sa.k_proj.weight, sa.k_proj.bias)
            .view(b, s, self.nkv, d)
            .transpose(1, 2)
        )
        v = (
            torch.nn.functional.linear(hn, sa.v_proj.weight, sa.v_proj.bias)
            .view(b, s, self.nkv, d)
            .transpose(1, 2)
        )
        q = (q.float() * cos + rotate_half(q).float() * sin).to(x.dtype)
        k = (k.float() * cos + rotate_half(k).float() * sin).to(x.dtype)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)).float() * d**-0.5 + bias
        o = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
        o = torch.nn.functional.linear(
            o.transpose(1, 2).reshape(b, s, self.nh * d), sa.o_proj.weight
        )
        if group is not None:
            dist.all_reduce(o, group=group)
        x = x + o
        hn = rms_norm(x, self.post_attention_layernorm.weight, self.eps)
        m = self.mlp
        y = torch.nn.functional.silu(
            torch.nn.functional.linear(hn, m.gate_proj.weight)
        ) * torch.nn.functional.linear(hn, m.up_proj.weight)
        y = torch.nn.functional.linear(y, m.down_proj.weight)
        if group is not None:
            dist.all_reduce(y, group=group)
        return x + y


class _LayerGroup(nn.Module):
    def __init__(self, layers, group):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        object.__setattr__(self, "_group", group)

    def forward(self, x, cos, sin, bias):
        for layer in self.layers:
            x = layer(x, cos, sin, bias, self._group)
        return x


class NeuronQwenTextEncoder(nn.Module):
    """``hidden_states[-3]`` of the Qwen2.5-VL text tower, TP over a text group (``tp_rank`` of ``tp``)."""

    def __init__(
        self, model_dir: str, dtype=torch.bfloat16, tp: int = 1, tp_rank: int = 0, group=None
    ):
        super().__init__()
        self.dir = os.path.join(model_dir, "text_encoder")
        with open(os.path.join(self.dir, "config.json")) as f:
            self.cfg = json.load(f)
        c = self.cfg
        if (
            c["num_attention_heads"] % tp
            or c["num_key_value_heads"] % tp
            or c["intermediate_size"] % tp
        ):
            raise ValueError(f"text TP={tp} must divide heads/KV heads/FFN of {c}")
        self.dtype, self.tp, self.tp_rank, self.group = dtype, tp, tp_rank, group
        self.n_layers = c["num_hidden_layers"] - 2  # hidden_states[-3]
        self.layers = nn.ModuleList(QwenTextLayer(c, dtype, tp) for _ in range(self.n_layers))
        self._device = torch.device("cpu")
        self._fns: dict = {}
        self._backend, self._options = None, {}
        self._rope: dict = {}
        self._emb_key = None

    # -- weights ---------------------------------------------------------------------------------
    def _files(self):
        idx = os.path.join(self.dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                return json.load(f)["weight_map"]
        return None

    def load(self) -> None:
        from safetensors import safe_open

        t0 = time.time()
        wmap = self._files()
        single = os.path.join(self.dir, "model.safetensors")
        c, r, tp = self.cfg, self.tp_rank, self.tp
        d = c["hidden_size"] // c["num_attention_heads"]
        q_rows, kv_rows = c["num_attention_heads"] // tp * d, c["num_key_value_heads"] // tp * d
        ff = c["intermediate_size"] // tp
        rows = {
            "q_proj": q_rows,
            "k_proj": kv_rows,
            "v_proj": kv_rows,
            "gate_proj": ff,
            "up_proj": ff,
        }
        cols = {"o_proj": q_rows, "down_proj": ff}
        handles: dict = {}
        state = {}
        for name, p in self.layers.named_parameters():
            key = f"layers.{name}"
            fn = os.path.join(self.dir, wmap[key]) if wmap else single
            if fn not in handles:
                handles[fn] = safe_open(fn, framework="pt")
            sl = handles[fn].get_slice(key)
            mod = name.split(".")[-2]
            if mod in rows:
                n = rows[mod]
                t = sl[r * n : (r + 1) * n]
            elif mod in cols:
                n = cols[mod]
                t = sl[:, r * n : (r + 1) * n]
            else:
                t = sl[:]
            state[name] = t.to(self.dtype).contiguous()
        self.layers.load_state_dict(state, strict=True, assign=True)
        emb_fn = os.path.join(self.dir, wmap["embed_tokens.weight"]) if wmap else single
        self._emb_key = (emb_fn, "embed_tokens.weight")
        logger.info(
            "HunyuanVideo-1.5 text tower shard %d/%d loaded in %.1fs", r, tp, time.time() - t0
        )

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Host embedding lookup reading only the needed rows of the table."""
        from safetensors import safe_open

        fn, key = self._emb_key
        uniq, inv = torch.unique(ids.reshape(-1), return_inverse=True)
        with safe_open(fn, framework="pt") as f:
            sl = f.get_slice(key)
            rows = torch.stack([sl[int(i) : int(i) + 1][0] for i in uniq])
        return rows.to(self.dtype)[inv].reshape(*ids.shape, -1)

    # -- device ----------------------------------------------------------------------------------
    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.layers.to(self._device)
        return self

    def compile(self, backend: str, options: dict | None = None) -> None:
        self._backend, self._options = backend, dict(options or {})

    def _graph(self, s: int, start: int, end: int):
        key = (s, start)
        fn = self._fns.get(key)
        if fn is None:
            mod = _LayerGroup(self.layers[start:end], self.group)
            if self._backend is not None:
                opts = {
                    **self._options,
                    "model_name": f"hv15_text_layers{end - start}_s{s}",
                    "compiler_args": list(TEXT_COMPILER_ARGS),
                }
                fn = torch.compile(
                    mod, backend=self._backend, options=opts, fullgraph=True, dynamic=False
                )
            else:
                fn = mod
            self._fns[key] = fn
        return fn

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` ids + mask -> ``hidden_states[-3]`` ``[B, S, H]`` (host tensor, model dtype)."""
        s = input_ids.shape[1]
        if s not in self._rope:
            d = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
            cos, sin = rope_tables(s, d, float(self.cfg.get("rope_theta", 1e6)))
            self._rope[s] = (cos.to(self._device), sin.to(self._device))
        cos, sin = self._rope[s]
        x = self.embed(input_ids).to(self._device)
        bias = attention_bias(attention_mask).to(self._device)
        k = LAYERS_PER_GRAPH if LAYERS_PER_GRAPH > 0 else self.n_layers
        for st in range(0, self.n_layers, k):
            x = self._graph(s, st, min(st + k, self.n_layers))(x, cos, sin, bias)
        return x.to("cpu")


def text_groups(world: int, tp: int) -> list[list[int]]:
    return [list(range(g, g + tp)) for g in range(0, world, tp)]


def assignment(n_prompts: int, n_groups: int) -> list[list[int]]:
    """Per round, the slot each group encodes: prompt ``slot % n_prompts``, kept only when
    ``slot < n_prompts`` (otherwise a repeat whose result is dropped). Every group runs the same
    number of rounds, so all groups execute the same graphs."""
    rounds = -(-n_prompts // n_groups)
    return [[r * n_groups + g for g in range(n_groups)] for r in range(rounds)]


__all__ = ["NeuronQwenTextEncoder", "assignment", "attention_bias", "rope_tables", "text_groups"]
