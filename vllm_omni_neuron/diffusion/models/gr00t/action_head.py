# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 flow-matching action head as one fixed-shape graph.

Re-implements upstream ``Gr00tN1d7ActionHead`` (``vlln`` -> optional ``vl_self_attention`` ->
state encoder -> ``num_inference_timesteps`` Euler steps of the ``AlternateVLDiT``) with the
same parameter names, so ``action_head.*`` loads with ``strict=True``. Differences from the
CUDA reference, none of which change the math:

* the whole denoising loop is unrolled in one graph (the step count and the discretised
  timesteps ``t * buckets // steps`` are compile-time constants);
* the initial noise is an input, so a seeded host draw reproduces the reference exactly;
* the VL sequence is padded to a bucket; padded keys are masked out of every attention
  (upstream runs batch 1 unpadded, which is the same thing);
* embodiment-specific weights are gathered in-graph from a device ``embodiment_id``;
* optionally (:meth:`Gr00tActionHead.shard_tp`) the transformer blocks are sharded by attention
  head / FF column across a tensor-parallel group, with one all-reduce after each attention
  output and each FF down projection.

Real-time chunking (RTC inpainting from a previous chunk) is not wired yet.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gr00tConfig
from .layers import attention, gelu_tanh
from .layers import row_parallel as _row_parallel
from .layers import shard_linear as _shard_linear

# ----------------------------------------------------------------------------------------
# diffusers-equivalent building blocks (same parameter names as diffusers' modules)
# ----------------------------------------------------------------------------------------


class _Attention(nn.Module):
    """diffusers ``Attention`` (bias=True, out_bias=True, no QK norm), SDPA processor."""

    def __init__(self, query_dim: int, heads: int, dim_head: int, cross_attention_dim: int | None):
        super().__init__()
        inner = heads * dim_head
        kv_dim = cross_attention_dim or query_dim
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_k = nn.Linear(kv_dim, inner, bias=True)
        self.to_v = nn.Linear(kv_dim, inner, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim, bias=True)])
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        if self.heads % size:
            raise ValueError(f"{self.heads} attention heads do not split over TP={size}")
        for lin in (self.to_q, self.to_k, self.to_v):
            _shard_linear(lin, 0, rank, size)
        _shard_linear(self.to_out[0], 1, rank, size)
        self.heads //= size
        self.tp_group = group
        if getattr(self, "qkv_weight", None) is not None:
            self.fuse_qkv()  # re-fuse from the sharded projections

    def kv(self, context):
        b, sk, _ = context.shape
        k = self.to_k(context).view(b, sk, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(b, sk, self.heads, self.dim_head).transpose(1, 2)
        return k, v

    def fuse_qkv(self) -> None:
        """One [3*inner, d] projection for self-attention instead of three (``GR00T_HEAD_FUSE_QKV``)."""
        w = torch.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight]).detach()
        bias = torch.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias]).detach()
        self.register_buffer("qkv_weight", w.contiguous(), persistent=False)
        self.register_buffer("qkv_bias", bias.contiguous(), persistent=False)

    def forward(self, x, context=None, bias=None, kv=None):
        b, s, _ = x.shape
        if context is None and kv is None and getattr(self, "qkv_weight", None) is not None:
            q, k, v = (
                F.linear(x, self.qkv_weight, self.qkv_bias)
                .view(b, s, 3, self.heads, self.dim_head)
                .unbind(2)
            )
            o = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), bias)
            return _row_parallel(self.to_out[0], o.transpose(1, 2).reshape(b, s, -1), self.tp_group)
        q = self.to_q(x).view(b, s, self.heads, self.dim_head).transpose(1, 2)
        k, v = kv if kv is not None else self.kv(x if context is None else context)
        o = attention(q, k, v, bias)
        return _row_parallel(self.to_out[0], o.transpose(1, 2).reshape(b, s, -1), self.tp_group)


class _GELUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x):
        return gelu_tanh(self.proj(x))


class _FeedForward(nn.Module):
    """diffusers ``FeedForward(activation_fn="gelu-approximate")``: net = [GELU, Dropout, Linear]."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.ModuleList(
            [_GELUProj(dim, inner), nn.Identity(), nn.Linear(inner, dim, bias=True)]
        )
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        _shard_linear(self.net[0].proj, 0, rank, size)
        _shard_linear(self.net[2], 1, rank, size)
        self.tp_group = group

    def forward(self, x):
        return _row_parallel(self.net[2], self.net[0](x), self.tp_group)


class _AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.dim = dim

    def forward(self, x, temb=None, mod=None):
        """``mod`` = precomputed ``linear(silu(temb))`` ``[B, 2*dim]`` (AdaLN table row), else from ``temb``."""
        mod = self.linear(F.silu(temb)) if mod is None else mod
        scale, shift = mod.chunk(2, dim=1)
        return F.layer_norm(x, (self.dim,), eps=1e-5) * (1 + scale[:, None]) + shift[:, None]


class _Block(nn.Module):
    """gr00t ``BasicTransformerBlock``: [Ada]LayerNorm -> attn (self or cross) -> LayerNorm -> FF."""

    def __init__(self, dim: int, heads: int, dim_head: int, cross_dim: int | None, ada: bool):
        super().__init__()
        self.ada = ada
        self.norm1 = (
            _AdaLayerNorm(dim) if ada else nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        )
        self.attn1 = _Attention(dim, heads, dim_head, cross_dim)
        # DiT blocks: norm_elementwise_affine=False (no params); vl_self_attention: affine
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=not ada)
        self.ff = _FeedForward(dim)

    def forward(self, x, temb=None, context=None, bias=None, kv=None, mod=None):
        h = self.norm1(x, temb, mod) if self.ada else self.norm1(x)
        x = x + self.attn1(h, context, bias, kv)
        return x + self.ff(self.norm3(x))


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class _TimestepEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = _TimestepEmbedding(256, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)
        half = 128
        exponent = (
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / (half - 1)
        )
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return self.timestep_embedder(emb.to(self.timestep_embedder.linear_1.weight.dtype))


class _DiT(nn.Module):
    """gr00t ``AlternateVLDiT`` (interleaved self-attention, alternating text/image cross-attention)."""

    def __init__(self, cfg: dict, cross_dim: int):
        super().__init__()
        heads, hd = int(cfg["num_attention_heads"]), int(cfg["attention_head_dim"])
        inner = heads * hd
        self.inner = inner
        self.timestep_encoder = _TimestepEncoder(inner)
        self.transformer_blocks = nn.ModuleList(
            [
                _Block(inner, heads, hd, None if i % 2 == 1 else cross_dim, ada=True)
                for i in range(int(cfg["num_layers"]))
            ]
        )
        self.proj_out_1 = nn.Linear(inner, 2 * inner)
        self.proj_out_2 = nn.Linear(inner, int(cfg["output_dim"]))

    def cross_kv(self, vl):
        """K/V of every cross-attention block over the VL features. They do not depend on the
        denoising step, so the head computes them once and reuses them for all steps."""
        return {i: blk.attn1.kv(vl) for i, blk in enumerate(self.transformer_blocks) if i % 2 == 0}

    def temb(self, timestep: torch.Tensor) -> torch.Tensor:
        """The timestep embedding every block's AdaLN modulation is a linear function of."""
        return self.timestep_encoder(timestep)

    def forward(
        self,
        x,
        vl,
        timestep,
        text_bias,
        image_bias,
        attend_text_every_n_blocks: int,
        kvs=None,
        mods=None,
        out_mod=None,
    ):
        """``mods``/``out_mod``: baked AdaLN rows for this timestep (see ``Gr00tActionHead.bake_adaln``)."""
        temb = self.timestep_encoder(timestep) if mods is None else None
        for i, blk in enumerate(self.transformer_blocks):
            mod = None if mods is None else mods[i]
            if i % 2 == 1:
                x = blk(x, temb, mod=mod)
            else:
                bias = text_bias if i % (2 * attend_text_every_n_blocks) == 0 else image_bias
                x = blk(x, temb, vl, bias, None if kvs is None else kvs[i], mod=mod)
        out_mod = self.proj_out_1(F.silu(temb)) if out_mod is None else out_mod
        shift, scale = out_mod.chunk(2, dim=1)
        x = F.layer_norm(x, (self.inner,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)


class _VLSelfAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        heads, hd = int(cfg["num_attention_heads"]), int(cfg["attention_head_dim"])
        self.transformer_blocks = nn.ModuleList(
            [_Block(heads * hd, heads, hd, None, ada=False) for _ in range(int(cfg["num_layers"]))]
        )

    def forward(self, x, bias):
        for blk in self.transformer_blocks:
            x = blk(x, bias=bias)
        return x


# ----------------------------------------------------------------------------------------
# Embodiment-conditioned MLPs
# ----------------------------------------------------------------------------------------


class _CatLinear(nn.Module):
    def __init__(self, n: int, din: int, dout: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, din, dout))
        self.b = nn.Parameter(torch.empty(n, dout))

    def forward(self, x, cat):
        return torch.bmm(x, self.W.index_select(0, cat)) + self.b.index_select(0, cat)[:, None]


class _CatMLP(nn.Module):
    def __init__(self, n: int, din: int, dh: int, dout: int):
        super().__init__()
        self.layer1 = _CatLinear(n, din, dh)
        self.layer2 = _CatLinear(n, dh, dout)

    def forward(self, x, cat):
        return self.layer2(F.relu(self.layer1(x, cat)), cat)


class _ActionEncoder(nn.Module):
    def __init__(self, n: int, action_dim: int, dim: int):
        super().__init__()
        self.dim = dim
        self.W1 = _CatLinear(n, action_dim, dim)
        self.W2 = _CatLinear(n, 2 * dim, dim)
        self.W3 = _CatLinear(n, dim, dim)

    def forward(self, actions, timestep, cat):
        b, t, _ = actions.shape
        a = self.W1(actions, cat)
        half = self.dim // 2
        exponent = -torch.arange(half, dtype=torch.float32, device=actions.device) * (
            math.log(10000.0) / half
        )
        freqs = timestep[:, None, None].float().expand(b, t, 1) * torch.exp(exponent)
        tau = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).to(a.dtype)
        x = self.W2(torch.cat([a, tau], dim=-1), cat)
        x = x * torch.sigmoid(x)
        return self.W3(x, cat)


# ----------------------------------------------------------------------------------------
# The action head
# ----------------------------------------------------------------------------------------


class Gr00tActionHead(nn.Module):
    """forward(vl ``[B, S, D]``, valid/image_mask ``[B, S]`` bool, state ``[B, T_s, max_state]``,
    noise ``[B, H, max_action]``, embodiment_id ``[B]`` long) -> actions ``[B, H, max_action]``.
    """

    def __init__(self, cfg: Gr00tConfig):
        super().__init__()
        h = cfg.head
        self.h = h
        n = int(h["max_num_embodiments"])
        emb = int(h["input_embedding_dim"])
        hidden = int(h["hidden_size"])
        self.action_horizon = int(h["action_horizon"])
        self.steps = int(h["num_inference_timesteps"])
        self.buckets = int(h["num_timestep_buckets"])
        self.attend_every = int(h["attend_text_every_n_blocks"])
        self.model = _DiT(cfg.dit, int(h["backbone_embedding_dim"]))
        self.state_encoder = _CatMLP(
            n, int(h["max_state_dim"]) * int(h["state_history_length"]), hidden, emb
        )
        self.action_encoder = _ActionEncoder(n, int(h["max_action_dim"]), emb)
        self.action_decoder = _CatMLP(n, hidden, hidden, int(h["max_action_dim"]))
        self.vlln = (
            nn.LayerNorm(int(h["backbone_embedding_dim"])) if h["use_vlln"] else nn.Identity()
        )
        self.vl_self_attention = (
            _VLSelfAttention(h["vl_self_attention_cfg"])
            if cfg.vl_self_attention_layers > 0
            else None
        )
        if h["add_pos_embed"]:
            self.position_embedding = nn.Embedding(int(h["max_seq_len"]), emb)

    def shard_tp(self, rank: int, size: int, group) -> None:
        """Shard every transformer block (DiT and VL self-attention) over a TP group of ``size``.

        Attention is split by head (Q/K/V rows, output columns) and the FF by hidden column; each
        rank all-reduces the attention output and the FF output. Norms, AdaLN tables, the
        embodiment MLPs and the Euler update stay replicated, so every rank ends with the same
        actions. Call after the weights (and AdaLN tables) are loaded."""
        if size <= 1:
            return
        blocks = list(self.model.transformer_blocks)
        if self.vl_self_attention is not None:
            blocks += list(self.vl_self_attention.transformer_blocks)
        for blk in blocks:
            blk.attn1.shard_tp(rank, size, group)
            blk.ff.shard_tp(rank, size, group)

    def fuse_projections(self) -> None:
        """Fuse Q/K/V of every self-attention block (DiT odd blocks, VL self-attention)."""
        blocks = [b for i, b in enumerate(self.model.transformer_blocks) if i % 2 == 1]
        if self.vl_self_attention is not None:
            blocks += list(self.vl_self_attention.transformer_blocks)
        for blk in blocks:
            blk.attn1.fuse_qkv()

    def timesteps(self) -> list[int]:
        return [t * self.buckets // self.steps for t in range(self.steps)]

    @torch.no_grad()
    def bake_adaln(self, fingerprint: str = "", cache_dir: str | None = None) -> None:
        """Precompute every AdaLN modulation row for the fixed timestep schedule.

        The DiT conditions only on the discretised timestep, and inference always uses the same
        ``num_inference_timesteps`` values, so ``norm1.linear(silu(temb(t)))`` of every block and
        the output ``proj_out_1(silu(temb(t)))`` are constants: computing them once removes 33
        linears per step from the graph and their weight reads (~150M params/step for N1.7).

        Uses the shared :class:`ModulationTables` / :func:`bake_linear_modulation`, which reproduce
        the DEVICE rounding (silu in fp32, cast to the compute dtype, fp32-accumulated matmul, one
        rounding, bias add in the compute dtype) so the baked tables are bit-identical to the
        in-graph modulation. Regenerate if the step count changes.
        """
        from vllm_omni_neuron.diffusion.layers.modulation_tables import ModulationTables

        dtype = self.model.proj_out_1.weight.dtype
        block_linears = [b.norm1.linear for b in self.model.transformer_blocks]
        tembs = [
            self.model.temb(
                torch.full((1,), t, dtype=torch.long, device=self.model.proj_out_1.weight.device)
            )
            for t in self.timesteps()
        ]
        block = ModulationTables.from_linears(
            block_linears,
            compute_dtype=dtype,
            fingerprint=f"gr00t-block-{fingerprint}",
            cache_dir=cache_dir,
            out_shape=lambda y: y[0],
        )  # [1, 2i] -> [2i]
        out = ModulationTables.from_linears(
            [self.model.proj_out_1],
            compute_dtype=dtype,
            fingerprint=f"gr00t-out-{fingerprint}",
            cache_dir=cache_dir,
            out_shape=lambda y: y[0],
        )
        # Stack into buffers so forward() only slices by the (compile-time constant) step index and
        # the tables move with .to(device) -- no tensor hashing or host->device transfer inside the graph.
        self.register_buffer(
            "adaln_table", torch.stack([block.tables(t) for t in tembs]).to(dtype), persistent=False
        )  # [steps, L, 2i]
        self.register_buffer(
            "adaln_out_table",
            torch.stack([out.tables(t)[0] for t in tembs]).to(dtype),
            persistent=False,
        )  # [steps, 2i]

    def forward(self, vl, valid, image_mask, state, noise, embodiment_id):
        from .layers import bias_from_keep

        dtype = vl.dtype
        vl = self.vlln(vl)
        if self.vl_self_attention is not None:
            vl = self.vl_self_attention(vl, bias_from_keep(valid[:, None, None, :]))
        text_bias = bias_from_keep(((~image_mask) & valid)[:, None, None, :])
        image_bias = bias_from_keep((image_mask & valid)[:, None, None, :])
        b = vl.shape[0]
        state_feat = self.state_encoder(state.reshape(b, 1, -1).to(dtype), embodiment_id)
        actions = noise.to(dtype)
        pos = (
            self.position_embedding.weight[: self.action_horizon][None]
            if self.h["add_pos_embed"]
            else None
        )
        dt = 1.0 / self.steps
        kvs = self.model.cross_kv(vl) if os.environ.get("GR00T_HEAD_KV_REUSE", "1") == "1" else None
        table = getattr(self, "adaln_table", None)
        for step, t in enumerate(self.timesteps()):
            ts = torch.full((b,), t, dtype=torch.long, device=vl.device)
            a_feat = self.action_encoder(actions, ts, embodiment_id)
            if pos is not None:
                a_feat = a_feat + pos
            x = torch.cat([state_feat, a_feat], dim=1)
            if table is None:
                out = self.model(x, vl, ts, text_bias, image_bias, self.attend_every, kvs)
            else:
                mods = [table[step, i][None].expand(b, -1) for i in range(table.shape[1])]
                out_mod = self.adaln_out_table[step][None].expand(b, -1)
                out = self.model(
                    x, vl, ts, text_bias, image_bias, self.attend_every, kvs, mods, out_mod
                )
            pred = self.action_decoder(out, embodiment_id)[:, -self.action_horizon :]
            actions = actions + dt * pred
        return actions
