# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 VLM backbone (Cosmos-Reason2-2B = Qwen3-VL-2B, first ``select_layer`` layers).

Two fixed-shape graphs, each a re-implementation of the matching ``transformers`` Qwen3-VL
module with identical parameter names (so the checkpoint's ``backbone.model.model.visual.*`` /
``backbone.model.model.language_model.*`` tensors load with ``strict=True``):

* :class:`Gr00tVision` -- the ViT with DeepStack mergers. All images of a request share one
  grid (the GR00T processor resizes every frame to the same size), so the packed
  ``cu_seqlens`` block-diagonal attention becomes a batched attention over ``[n_images, L]``.
* :class:`Gr00tText` -- the decoder stack. Image features are placed into the token sequence by
  a gather (``image_index``) plus a ``where`` on the image mask instead of ``masked_scatter``,
  and DeepStack features are added after the first ``len(deepstack)`` layers the same way. The
  output is the **pre-final-norm** hidden state of the last layer, which is what GR00T was
  trained on (upstream vLLM-Omni captures it with a hook on the final RMSNorm).

Everything data-dependent -- the 3-D mRoPE position ids, the vision RoPE angles, the
bilinear position-embedding taps, the causal/padding mask -- is computed on the host by
:class:`BackbonePrep` using ``transformers``' own helpers, then passed in as tensors.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    attention,
    bias_from_keep,
    gelu_erf,
    gelu_tanh,
    rms_norm,
    rotate_half,
    row_parallel,
    shard_linear,
)

# ----------------------------------------------------------------------------------------
# Vision tower
# ----------------------------------------------------------------------------------------


class _Conv3dPatch(nn.Module):
    """Holds the ``Conv3d`` patch-embed weight; kernel == stride, so it is a linear layer."""

    def __init__(self, c: int, t: int, p: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, c, t, p, p))
        self.bias = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.weight.dtype), self.weight.flatten(1), self.bias)


class _PatchEmbed(nn.Module):
    def __init__(self, vc: dict):
        super().__init__()
        self.proj = _Conv3dPatch(
            vc["in_channels"], vc["temporal_patch_size"], vc["patch_size"], vc["hidden_size"]
        )


class _VisionAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        shard_linear(self.qkv, 0, rank, size, parts=3)  # [q | k | v], each split by head
        shard_linear(self.proj, 1, rank, size)
        self.heads //= size
        self.tp_group = group

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x [n_img, L, C]; cos/sin [n_img, L, D] fp32
        n, length, c = x.shape
        q, k, v = self.qkv(x).reshape(n, length, 3, self.heads, -1).unbind(2)  # [n, L, H, D]
        cs, sn = cos[:, :, None], sin[:, :, None]
        qf, kf = q.float(), k.float()
        q = (qf * cs + rotate_half(qf) * sn).to(x.dtype)
        k = (kf * cs + rotate_half(kf) * sn).to(x.dtype)
        o = attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), allow_nki=True
        )  # [n, H, L, D]
        return row_parallel(self.proj, o.transpose(1, 2).reshape(n, length, -1), self.tp_group)


class _VisionMLP(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, inner, bias=True)
        self.linear_fc2 = nn.Linear(inner, dim, bias=True)
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        shard_linear(self.linear_fc1, 0, rank, size)
        shard_linear(self.linear_fc2, 1, rank, size)
        self.tp_group = group

    def forward(self, x):
        return row_parallel(self.linear_fc2, gelu_tanh(self.linear_fc1(x)), self.tp_group)


class _VisionBlock(nn.Module):
    def __init__(self, vc: dict):
        super().__init__()
        d = vc["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.attn = _VisionAttention(d, vc["num_heads"])
        self.mlp = _VisionMLP(d, vc["intermediate_size"])

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
        return self.linear_fc2(gelu_erf(self.linear_fc1(x)))


class Gr00tVision(nn.Module):
    """``Qwen3VLVisionModel`` for ``n_images`` equal-grid images.

    forward(pixels ``[N, C*T*P*P]``, pos_index ``[N, 4]`` long, pos_weight ``[N, 4]`` fp32,
    cos/sin ``[N, D]`` fp32) -> (merged ``[N/m^2, out]``, *deepstack ``[N/m^2, out]``).
    ``N = n_images * patches_per_image`` in the processor's (merge-block) patch order.
    """

    def __init__(self, vc: dict):
        super().__init__()
        self.vc = vc
        self.patch_embed = _PatchEmbed(vc)
        self.pos_embed = nn.Embedding(vc["num_position_embeddings"], vc["hidden_size"])
        self.blocks = nn.ModuleList([_VisionBlock(vc) for _ in range(vc["depth"])])
        self.merger = _Merger(vc, postshuffle=False)
        self.deepstack_visual_indexes = list(vc["deepstack_visual_indexes"])
        self.deepstack_merger_list = nn.ModuleList(
            [_Merger(vc, postshuffle=True) for _ in self.deepstack_visual_indexes]
        )

    def shard_tp(self, rank: int, size: int, group) -> None:
        """Shard every ViT block by head / MLP column (one all-reduce after attention and MLP each);
        the patch embed, position embedding and mergers stay replicated."""
        if self.vc["num_heads"] % size:
            raise ValueError(f"{self.vc['num_heads']} ViT heads do not split over TP={size}")
        for blk in self.blocks:
            blk.attn.shard_tp(rank, size, group)
            blk.mlp.shard_tp(rank, size, group)

    def forward(self, pixels, pos_index, pos_weight, cos, sin, n_images: int):
        x = self.patch_embed.proj(pixels)
        pe = (self.pos_embed(pos_index) * pos_weight[:, :, None]).sum(1)  # fp32, as HF
        x = x + pe.to(x.dtype)
        n_tok, c = x.shape
        length = n_tok // n_images
        x = x.reshape(n_images, length, c)
        cos = cos.reshape(n_images, length, -1)
        sin = sin.reshape(n_images, length, -1)
        deep = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin)
            if i in self.deepstack_visual_indexes:
                deep.append(
                    self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](
                        x.reshape(n_tok, c)
                    )
                )
        return (self.merger(x.reshape(n_tok, c)), *deep)


# ----------------------------------------------------------------------------------------
# Text tower
# ----------------------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class _TextAttention(nn.Module):
    def __init__(self, tc: dict):
        super().__init__()
        d, hd = tc["hidden_size"], tc["head_dim"]
        self.h, self.hk, self.hd = tc["num_attention_heads"], tc["num_key_value_heads"], hd
        bias = bool(tc.get("attention_bias", False))
        self.q_proj = nn.Linear(d, self.h * hd, bias=bias)
        self.k_proj = nn.Linear(d, self.hk * hd, bias=bias)
        self.v_proj = nn.Linear(d, self.hk * hd, bias=bias)
        self.o_proj = nn.Linear(self.h * hd, d, bias=bias)
        self.q_norm = _RMSNorm(hd, tc["rms_norm_eps"])
        self.k_norm = _RMSNorm(hd, tc["rms_norm_eps"])
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        if self.hk % size:
            raise ValueError(f"{self.hk} KV heads do not split over TP={size}")
        for lin in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
        ):  # GQA groups stay intact: H/size per H_kv/size
            shard_linear(lin, 0, rank, size)
        shard_linear(self.o_proj, 1, rank, size)
        self.h //= size
        self.hk //= size
        self.tp_group = group

    def forward(self, x, cos, sin, bias):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.hk, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.hk, self.hd).transpose(1, 2)
        cs, sn = cos[:, None], sin[:, None]
        q = q * cs + rotate_half(q) * sn
        k = k * cs + rotate_half(k) * sn
        o = attention(q, k, v, bias)
        return row_parallel(self.o_proj, o.transpose(1, 2).reshape(b, s, -1), self.tp_group)


class _TextMLP(nn.Module):
    def __init__(self, tc: dict):
        super().__init__()
        d, i = tc["hidden_size"], tc["intermediate_size"]
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)
        self.tp_group = None

    def shard_tp(self, rank: int, size: int, group) -> None:
        shard_linear(self.gate_proj, 0, rank, size)
        shard_linear(self.up_proj, 0, rank, size)
        shard_linear(self.down_proj, 1, rank, size)
        self.tp_group = group

    def forward(self, x):
        return row_parallel(
            self.down_proj, F.silu(self.gate_proj(x)) * self.up_proj(x), self.tp_group
        )


class _TextLayer(nn.Module):
    def __init__(self, tc: dict):
        super().__init__()
        self.self_attn = _TextAttention(tc)
        self.mlp = _TextMLP(tc)
        self.input_layernorm = _RMSNorm(tc["hidden_size"], tc["rms_norm_eps"])
        self.post_attention_layernorm = _RMSNorm(tc["hidden_size"], tc["rms_norm_eps"])

    def forward(self, x, cos, sin, bias):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, bias)
        return x + self.mlp(self.post_attention_layernorm(x))


class Gr00tText(nn.Module):
    """``Qwen3VLTextModel`` truncated to ``num_hidden_layers`` layers, returning pre-norm hidden.

    forward(input_ids ``[B, S]``, image_index ``[B, S]`` long, image_keep ``[B, S, 1]`` bool,
    image_embeds ``[M, D]``, cos/sin ``[B, S, hd]``, bias ``[B, 1, S, S]`` fp32, *deepstack)
    -> ``[B, S, D]``.
    """

    def __init__(self, tc: dict):
        super().__init__()
        self.tc = tc
        self.embed_tokens = nn.Embedding(tc["vocab_size"], tc["hidden_size"])
        self.layers = nn.ModuleList([_TextLayer(tc) for _ in range(tc["num_hidden_layers"])])
        self.norm = _RMSNorm(
            tc["hidden_size"], tc["rms_norm_eps"]
        )  # loaded, unused (pre-norm output)

    def shard_tp(self, rank: int, size: int, group) -> None:
        """Shard every decoder layer by head / MLP column; embeddings and norms stay replicated."""
        for layer in self.layers:
            layer.self_attn.shard_tp(rank, size, group)
            layer.mlp.shard_tp(rank, size, group)

    def forward(self, input_ids, image_index, image_keep, image_embeds, cos, sin, bias, *deepstack):
        x = self.embed_tokens(input_ids)
        x = torch.where(image_keep, image_embeds[image_index].to(x.dtype), x)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, bias)
            if i < len(deepstack):
                x = x + torch.where(image_keep, deepstack[i][image_index].to(x.dtype), zero)
        return x


# ----------------------------------------------------------------------------------------
# Host-side preparation
# ----------------------------------------------------------------------------------------


@dataclass
class BackboneInputs:
    """All tensors the two backbone graphs need, on the host, already bucketed."""

    pixels: torch.Tensor  # [N, C*T*P*P]
    pos_index: torch.Tensor  # [N, 4] long
    pos_weight: torch.Tensor  # [N, 4] fp32
    vis_cos: torch.Tensor  # [N, Dv] fp32
    vis_sin: torch.Tensor
    n_images: int
    input_ids: torch.Tensor  # [B, S] long (S = bucket)
    image_index: torch.Tensor  # [B, S] long
    image_keep: torch.Tensor  # [B, S, 1] bool
    txt_cos: torch.Tensor  # [B, S, hd] fp32
    txt_sin: torch.Tensor
    txt_bias: torch.Tensor  # [B, 1, S, S] fp32 (causal & key-valid)
    valid: torch.Tensor  # [B, S] bool (real tokens)
    image_mask: torch.Tensor  # [B, S] bool (image-token positions)
    real_len: int
    n_real_images: int = 0  # before image-count bucketing (``n_images`` is the bucket)
    vis_key: tuple = ()  # identifies the vision tables (pos taps, RoPE): equal key -> equal tensors
    txt_key: tuple = ()  # identifies every text/mask table: equal key -> equal tensors


# Image-count buckets for the vision graph: a request with n images runs the smallest bucket >= n
# (padded images are zero pixels whose tokens are never gathered), so a new camera/frame count
# reuses a compiled graph instead of compiling its own. Counts above the largest bucket run as is.
IMAGE_BUCKETS = tuple(
    int(b) for b in os.environ.get("GR00T_IMAGE_BUCKETS", "1,2,3,4,6,8").split(",") if b
)


def pick_image_bucket(n: int, buckets=IMAGE_BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    return n


_TEXT_CACHE_MAX = 32


class BackbonePrep:
    """Host math for the backbone, delegated to ``transformers``' own Qwen3-VL helpers."""

    def __init__(self, hf_config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )

        self.cfg = hf_config
        vc = hf_config.vision_config
        self.merge = int(vc.spatial_merge_size)
        self.side = int(vc.num_position_embeddings**0.5)
        self.image_token_id = int(hf_config.image_token_id)
        self.vis_rope = Qwen3VLVisionRotaryEmbedding(vc)
        self.txt_rope = Qwen3VLTextRotaryEmbedding(hf_config.text_config)
        shim = SimpleNamespace(config=hf_config)
        shim.get_vision_position_ids = lambda *a, **k: Qwen3VLModel.get_vision_position_ids(
            shim, *a, **k
        )
        self._rope_index = lambda *a, **k: Qwen3VLModel.get_rope_index(shim, *a, **k)
        self._vision_cache: dict = {}  # grid -> (taps, weights, cos, sin): a pure function of the grid
        # (token ids, mm types, grid, bucket) -> text tables. A robot repeats its instruction every
        # step, so mRoPE positions, the causal/padding bias and the image-token gather repeat too.
        self._text_cache: OrderedDict = OrderedDict()

    @torch.no_grad()
    def vision_tables(self, grid_thw: torch.Tensor):
        key = tuple(grid_thw.reshape(-1).tolist())
        cached = self._vision_cache.get(key)
        if cached is None:
            cached = self._vision_cache[key] = self._vision_tables(grid_thw)
        return cached

    def _vision_tables(self, grid_thw: torch.Tensor):
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        idx, w = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self.merge,
        )
        pos = get_vision_position_ids(grid_thw, self.merge)
        cos, sin = self.vis_rope(torch.zeros(1), pos)
        return idx.long(), w.float(), cos.float(), sin.float()

    @torch.no_grad()
    def __call__(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids=None,
        bucket: int | None = None,
        image_buckets=None,
    ) -> BackboneInputs:
        if input_ids.shape[0] != 1:
            raise NotImplementedError("GR00T on Neuron serves one observation per call (batch 1)")
        grid = image_grid_thw.long()
        if not bool((grid == grid[0]).all()):
            raise ValueError(f"all images of a request must share one grid, got {grid.tolist()}")
        n_real = int(grid[:, 0].sum())  # images have t == 1
        n_images = pick_image_bucket(
            n_real, IMAGE_BUCKETS if image_buckets is None else image_buckets
        )
        vgrid = grid[:1].expand(n_images, -1) if n_images != grid.shape[0] else grid
        pidx, pw, vcos, vsin = self.vision_tables(vgrid)
        pixels = pixel_values
        if n_images != n_real:  # zero images up to the bucket; their tokens are never gathered
            per = pixel_values.shape[0] // n_real
            pixels = torch.cat(
                [
                    pixel_values,
                    pixel_values.new_zeros((n_images - n_real) * per, pixel_values.shape[1]),
                ]
            )

        am = attention_mask.long()
        real = int(am.sum())
        s = bucket or real
        if real > s:
            raise ValueError(f"prompt has {real} tokens, bucket is {s}")
        grid_key = tuple(grid.reshape(-1).tolist())
        txt_key = (
            input_ids.numpy().tobytes(),
            am.numpy().tobytes(),
            None if mm_token_type_ids is None else mm_token_type_ids.numpy().tobytes(),
            grid_key,
            s,
        )
        txt = self._text_cache.get(txt_key)
        if txt is None:
            txt = self._text_tables(input_ids, am, grid, mm_token_type_ids, real, s)
            self._text_cache[txt_key] = txt
            while len(self._text_cache) > _TEXT_CACHE_MAX:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(txt_key)
        pid, index, img, tcos, tsin, bias, valid = txt
        return BackboneInputs(
            pixels=pixels,
            pos_index=pidx,
            pos_weight=pw,
            vis_cos=vcos,
            vis_sin=vsin,
            n_images=n_images,
            input_ids=pid,
            image_index=index,
            image_keep=img[..., None],
            txt_cos=tcos,
            txt_sin=tsin,
            txt_bias=bias,
            valid=valid,
            image_mask=img,
            real_len=real,
            n_real_images=n_real,
            vis_key=(grid_key[:3], n_images),
            txt_key=txt_key,
        )

    def _text_tables(self, input_ids, am, grid, mm_token_type_ids, real: int, s: int):
        keep_tok = am[0].bool()
        ids = input_ids[:, keep_tok]  # drop (left) padding: batch 1 has none in practice
        mm = mm_token_type_ids[:, keep_tok] if mm_token_type_ids is not None else None
        if mm is not None:
            pos3, _ = self._rope_index(
                ids, mm, image_grid_thw=grid, attention_mask=torch.ones_like(ids)
            )
        else:  # no mm_token_type_ids: HF falls back to 1-D positions
            pos3 = torch.arange(real).view(1, 1, -1).expand(3, 1, -1)
        pos = torch.zeros(3, 1, s, dtype=torch.long)
        pos[:, :, :real] = pos3
        tcos, tsin = self.txt_rope(torch.zeros(1, dtype=torch.float32), pos)

        pid = torch.zeros(1, s, dtype=torch.long)
        pid[:, :real] = ids
        valid = torch.zeros(1, s, dtype=torch.bool)
        valid[:, :real] = True
        img = (pid == self.image_token_id) & valid
        index = (img.long().cumsum(-1) - 1).clamp(min=0)
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool))
        bias = bias_from_keep(causal[None, None] & valid[:, None, None, :])
        return pid, index, img, tcos.float(), tsin.float(), bias, valid
