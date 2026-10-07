# SPDX-License-Identifier: Apache-2.0
"""Alpamayo 1.5 VLM backbone (Cosmos-Reason2-8B = Qwen3-VL-8B-Instruct, verified byte-identical,
see ``config.py``), tensor-parallel across NeuronCores via ``vllm_neuron.nn``
``ColumnParallelLinear``/``RowParallelLinear``.

* :class:`Gr2VisionTower` -- the ViT with DeepStack mergers (hidden 1152, depth 27, 3 DeepStack
  merge points).
* :class:`Gr2TextTower` -- the mRoPE decoder stack, in two modes:
  - ``do_prefill(...)``: the whole (bucket-padded) prompt in one call, returning fixed-length
    per-layer K/V caches;
  - ``do_decode(...)``: one new token through every layer against those caches (full-length read,
    additive mask, one-hot write), returning the updated caches.

The text tower's attention goes through :mod:`.decode_backend`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .decode_backend import DecodeConfig, decode_step, prefill
from .layers import attention_full, gelu_tanh, rms_norm, rotate_half


def _tp_state(tp_group=None):
    if dist.is_initialized():
        group = tp_group if tp_group is not None else dist.group.WORLD
        return dist.get_world_size(group), dist.get_rank(group), group
    return 1, 0, None


def _row_parallel_biased(in_f: int, out_f: int, tp_group, tp_size: int):
    """A biased ``RowParallelLinear``. vllm_neuron forbids a non-fp32 bias under TP>1 (XLA lowering
    bug, see rpl.py), but its forward adds the bias AFTER the all-reduce via a separate
    ``torch.add``, with ``F.linear(x, weight, None)`` doing the matmul bias-free. So under TP we
    build the layer in fp32 to clear the guard; load_weights then loads the WEIGHT as bf16
    (assign=True adopts the checkpoint dtype) while keeping the BIAS fp32. The fp32 bias promotes
    the reduced output to fp32, so :class:`_BiasedRowParallel` casts it back to the INPUT dtype
    inside the graph -- otherwise the fp32 residual stream reaches the next bf16 matmul and the
    lowering fails ("matmul: input datatypes mismatched"). On the tp_size==1
    CPU path the guard never fires and the layer stays the default dtype."""
    from vllm_neuron.nn import RowParallelLinear

    if tp_size > 1:

        class _BiasedRowParallel(RowParallelLinear):
            def forward(self, x):
                return super().forward(x).to(x.dtype)

        return _BiasedRowParallel(in_f, out_f, bias=True, tp_group=tp_group, dtype=torch.float32)
    return RowParallelLinear(in_f, out_f, bias=True, tp_group=tp_group)


# ----------------------------------------------------------------------------------------
# Vision tower
# ----------------------------------------------------------------------------------------


class _Conv3dPatchProj(nn.Module):
    """``nn.Conv3d`` with kernel==stride is a linear layer; named ``.proj`` to match the real
    checkpoint's ``patch_embed.proj.{weight,bias}`` key nesting."""

    def __init__(self, c: int, t: int, p: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, c, t, p, p))
        self.bias = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.weight.dtype), self.weight.flatten(1), self.bias)


class _PatchEmbed(nn.Module):
    def __init__(self, c: int, t: int, p: int, dim: int):
        super().__init__()
        self.proj = _Conv3dPatchProj(c, t, p, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int, tp_group=None):
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_group = _tp_state(tp_group)
        self.heads = heads // self.tp_size
        self.head_dim = dim // heads
        from vllm_neuron.nn import ColumnParallelLinear

        self.qkv = ColumnParallelLinear(dim, dim * 3, bias=True, tp_group=self.tp_group)
        self.proj = _row_parallel_biased(dim, dim, self.tp_group, self.tp_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n, length, _ = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        cs, sn = cos[:, :, None], sin[:, :, None]
        qf, kf = q.float(), k.float()
        q = (qf * cs + rotate_half(qf) * sn).to(x.dtype)
        k = (kf * cs + rotate_half(kf) * sn).to(x.dtype)
        o = attention_full(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(o.transpose(1, 2).reshape(n, length, self.heads * self.head_dim))


class _VisionMLP(nn.Module):
    def __init__(self, dim: int, inner: int, tp_group=None):
        super().__init__()
        tp_size, _, grp = _tp_state(tp_group)
        from vllm_neuron.nn import ColumnParallelLinear

        self.linear_fc1 = ColumnParallelLinear(dim, inner, bias=True, tp_group=grp)
        self.linear_fc2 = _row_parallel_biased(inner, dim, grp, tp_size)

    def forward(self, x):
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))


class _VisionBlock(nn.Module):
    def __init__(self, vc: dict, tp_group=None):
        super().__init__()
        d = vc["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.attn = _VisionAttention(d, vc["num_heads"], tp_group)
        self.mlp = _VisionMLP(d, vc["intermediate_size"], tp_group)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _Merger(nn.Module):
    def __init__(self, vc: dict, postshuffle: bool):
        super().__init__()
        self.hidden = vc["hidden_size"] * vc["spatial_merge_size"] ** 2
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(self.hidden if postshuffle else vc["hidden_size"], eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.linear_fc2 = nn.Linear(self.hidden, vc["out_hidden_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            self.norm(x.reshape(-1, self.hidden))
            if self.postshuffle
            else self.norm(x).reshape(-1, self.hidden)
        )
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Gr2VisionTower(nn.Module):
    """Qwen3-VL ViT with DeepStack. ``forward(pixels, pos_index, pos_weight, cos, sin, n_images)``
    -> ``(merged [N/m^2, out], *deepstack [N/m^2, out])``,
    TP shards every block's Linear layers; the 27-block stack is not yet N-block-graph-split
    (``BlockGraphRunner`` has no mid-stack capture hook for DeepStack's intermediate activations; a
    possible compile-time optimization, not needed for correctness)."""

    def __init__(self, vc: dict, tp_group=None):
        super().__init__()
        self.vc = vc
        self.patch_embed = _PatchEmbed(
            vc["in_channels"], vc["temporal_patch_size"], vc["patch_size"], vc["hidden_size"]
        )
        self.pos_embed = nn.Embedding(vc["num_position_embeddings"], vc["hidden_size"])
        self.blocks = nn.ModuleList([_VisionBlock(vc, tp_group) for _ in range(vc["depth"])])
        self.merger = _Merger(vc, postshuffle=False)
        self.deepstack_visual_indexes = list(vc["deepstack_visual_indexes"])
        self.deepstack_merger_list = nn.ModuleList(
            [_Merger(vc, postshuffle=True) for _ in self.deepstack_visual_indexes]
        )

    def forward(self, pixels, pos_index, pos_weight, cos, sin, n_images: int):
        x = self.patch_embed(pixels)
        pe = (self.pos_embed(pos_index) * pos_weight[:, :, None]).sum(1)
        x = x + pe.to(x.dtype)
        n_tok, c = x.shape
        length = n_tok // n_images
        x = x.reshape(n_images, length, c)
        cos_r, sin_r = cos.reshape(n_images, length, -1), sin.reshape(n_images, length, -1)
        deep = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos_r, sin_r)
            if i in self.deepstack_visual_indexes:
                deep.append(
                    self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](
                        x.reshape(n_tok, c)
                    )
                )
        return (self.merger(x.reshape(n_tok, c)), *deep)


# ----------------------------------------------------------------------------------------
# Text tower: mRoPE decoder, sharded, with prefill + fixed-length autoregressive decode
# ----------------------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class _TextAttention(nn.Module):
    def __init__(self, tc: dict, tp_group=None):
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_group = _tp_state(tp_group)
        d, hd = tc["hidden_size"], tc["head_dim"]
        self.h_total, self.hk_total, self.hd = (
            tc["num_attention_heads"],
            tc["num_key_value_heads"],
            hd,
        )
        if self.h_total % self.tp_size or self.hk_total % self.tp_size:
            raise ValueError(
                f"TP size {self.tp_size} must divide both q_heads={self.h_total} "
                f"and kv_heads={self.hk_total}"
            )
        self.h, self.hk = self.h_total // self.tp_size, self.hk_total // self.tp_size
        bias = bool(tc.get("attention_bias", False))
        from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

        self.q_proj = ColumnParallelLinear(d, self.h_total * hd, bias=bias, tp_group=self.tp_group)
        self.k_proj = ColumnParallelLinear(d, self.hk_total * hd, bias=bias, tp_group=self.tp_group)
        self.v_proj = ColumnParallelLinear(d, self.hk_total * hd, bias=bias, tp_group=self.tp_group)
        self.o_proj = RowParallelLinear(self.h_total * hd, d, bias=bias, tp_group=self.tp_group)
        self.q_norm = _RMSNorm(hd, tc["rms_norm_eps"])
        self.k_norm = _RMSNorm(hd, tc["rms_norm_eps"])

    def _qkv(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.hk, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.hk, self.hd).transpose(1, 2)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        return q, k, v

    def do_prefill(self, x, cos, sin, bias, cfg: DecodeConfig):
        q, k, v = self._qkv(x, cos, sin)
        o, kc, vc = prefill(cfg, q, k, v, bias)
        return self.o_proj(o.transpose(1, 2).reshape(*x.shape[:2], -1)), kc, vc

    def do_decode(self, x, cos, sin, k_cache, v_cache, write_mask, bias, cfg: DecodeConfig):
        q, k, v = self._qkv(x, cos, sin)
        o, kc, vc = decode_step(cfg, q, k, v, k_cache, v_cache, write_mask, bias)
        return self.o_proj(o.transpose(1, 2).reshape(*x.shape[:2], -1)), kc, vc


class _TextMLP(nn.Module):
    def __init__(self, tc: dict, tp_group=None):
        super().__init__()
        _, _, grp = _tp_state(tp_group)
        d, i = tc["hidden_size"], tc["intermediate_size"]
        from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

        self.gate_proj = ColumnParallelLinear(d, i, bias=False, tp_group=grp)
        self.up_proj = ColumnParallelLinear(d, i, bias=False, tp_group=grp)
        self.down_proj = RowParallelLinear(i, d, bias=False, tp_group=grp)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TextLayer(nn.Module):
    def __init__(self, tc: dict, tp_group=None):
        super().__init__()
        self.self_attn = _TextAttention(tc, tp_group)
        self.mlp = _TextMLP(tc, tp_group)
        self.input_layernorm = _RMSNorm(tc["hidden_size"], tc["rms_norm_eps"])
        self.post_attention_layernorm = _RMSNorm(tc["hidden_size"], tc["rms_norm_eps"])

    def do_prefill(self, x, cos, sin, bias, cfg):
        a, kc, vc = self.self_attn.do_prefill(self.input_layernorm(x), cos, sin, bias, cfg)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kc, vc

    def do_decode(self, x, cos, sin, k_cache, v_cache, write_mask, bias, cfg):
        a, kc, vc = self.self_attn.do_decode(
            self.input_layernorm(x), cos, sin, k_cache, v_cache, write_mask, bias, cfg
        )
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kc, vc


class Gr2TextTower(nn.Module):
    """Qwen3-VL text decoder at the checkpoint's own ``num_hidden_layers`` (the full
    backbone depth: this tower also writes the Chain-of-Causation text autoregressively)."""

    def __init__(self, tc: dict, tp_group=None):
        super().__init__()
        self.tc = tc
        self.tp_group = tp_group
        self.embed_tokens = nn.Embedding(tc["vocab_size"], tc["hidden_size"])
        self.layers = nn.ModuleList(
            [_TextLayer(tc, tp_group) for _ in range(tc["num_hidden_layers"])]
        )
        self.norm = _RMSNorm(tc["hidden_size"], tc["rms_norm_eps"])

    def decode_config(self, max_len: int, dtype=torch.bfloat16) -> DecodeConfig:
        tp_size, _, _ = _tp_state(self.tp_group)
        return DecodeConfig(
            q_heads=self.tc["num_attention_heads"] // tp_size,
            kv_heads=self.tc["num_key_value_heads"] // tp_size,
            head_dim=self.tc["head_dim"],
            max_len=max_len,
            dtype=dtype,
        )

    def do_prefill(
        self,
        input_ids,
        image_index,
        image_keep,
        image_embeds,
        cos,
        sin,
        bias,
        cfg: DecodeConfig,
        deepstack: tuple = (),
    ):
        """Whole (bucket-padded) prompt in one graph. Returns ``(hidden [1, S, d] pre-norm,
        k_caches, v_caches)`` -- one fixed ``[1, kv_heads, cfg.max_len, head_dim]`` pair per layer."""
        x = self.embed_tokens(input_ids)
        x = torch.where(image_keep, image_embeds[image_index].to(x.dtype), x)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, kc, vc = layer.do_prefill(x, cos, sin, bias, cfg)
            ks.append(kc)
            vs.append(vc)
            if i < len(deepstack):
                x = x + torch.where(image_keep, deepstack[i][image_index].to(x.dtype), zero)
        return x, tuple(ks), tuple(vs)

    def do_decode(
        self, token_embed, cos, sin, k_caches, v_caches, write_mask, bias, cfg: DecodeConfig
    ):
        """One token (``token_embed`` ``[1, 1, hidden]``, embedded on the host) through every layer
        against the fixed caches. Returns ``(hidden [1, 1, d] pre-norm, k_caches', v_caches')``."""
        x = token_embed
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, kc, vc = layer.do_decode(
                x, cos, sin, k_caches[i], v_caches[i], write_mask, bias, cfg
            )
            ks.append(kc)
            vs.append(vc)
        return x, tuple(ks), tuple(vs)
