# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 DiT for Neuron, split into a PREFIX graph and a TARGET graph.

The upstream model (vendored ``_vendor/transformer_qwenimage21.py``) is a single-stream
transformer over one joint sequence ``[prefix | target]``: the prefix is the prompt's text
tokens with any condition-image latents dropped into the vision slots, the target is the image
being denoised. Two properties of 2.1 make the split exact:

* attention is block-causal, so no prefix token ever attends to a target token, and
* ``causal_condition``: prefix tokens are modulated from ``t = 0``, not the sampled timestep.

So the prefix's per-layer keys/values depend only on the prompt. Upstream exploits this with a
KV cache filled on the first step; here the prefix runs as its own graph once per prompt
(and CFG branch), and every denoising step (the first included) runs only the target tokens
against the cached prefix K/V. Both are fixed-shape graphs: the prefix is right-padded to a
token bucket, the target has one shape per output resolution.

Weights are raw tensors sharded over TP by heads (Q/K/V and the MLP's two input projections
column-parallel, the attention output and MLP output row-parallel + all-reduce). Everything
else (embedders, the shared modulation, the output head) is replicated.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    MASK_VALUE,
    all_reduce,
    attach_shard_loaders,
    attention,
    gelu_tanh,
    layer_norm,
    load_sharded,
    param,
    rms_norm,
    rope_pairs,
    tp_state,
)

IMG_TOKENS_PER_SLOT = 4  # one vision-language image slot = a 2x2 group of latent tokens


@dataclass
class QwenImage21DiTConfig:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    head_dim: int = 128
    num_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True

    @property
    def dim(self) -> int:
        return self.num_heads * self.head_dim

    @classmethod
    def from_dict(cls, d: dict) -> QwenImage21DiTConfig:
        if int(d.get("patch_size", 1)) != 1:
            raise ValueError("Qwen-Image 2.1 consumes latents unpatched (patch_size=1)")
        return cls(
            in_channels=int(d.get("in_channels", 64)),
            out_channels=int(d.get("out_channels") or d.get("in_channels", 64)),
            num_layers=int(d.get("num_layers", 32)),
            head_dim=int(d.get("attention_head_dim", 128)),
            num_heads=int(d.get("num_attention_heads", 32)),
            context_in_dim=int(d.get("context_in_dim", 4096)),
            mlp_ratio=int(d.get("mlp_ratio", 3)),
            axes_dims_rope=tuple(d.get("axes_dims_rope", (16, 56, 56))),
            eps=float(d.get("eps", 1e-6)),
            causal_condition=bool(d.get("causal_condition", True)),
        )

    @classmethod
    def from_model_dir(cls, model_path: str) -> QwenImage21DiTConfig:
        with open(os.path.join(model_path, "transformer", "config.json")) as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------------------------
# Host-side layout: RoPE tables and masks for one prompt
# --------------------------------------------------------------------------------------------
@dataclass
class PrefixLayout:
    """Everything the two graphs need about one prompt's joint sequence, built on the host.

    ``bucket`` = padded prefix length L. Tensors (fp32 unless noted):
      cos_p/sin_p ``[L, D/2]``, cos_t/sin_t ``[N, D/2]``, prefix_bias ``[B, 1, L, L]``,
      target_bias ``[B, 1, 1, L]`` (valid prefix keys), is_img ``[L, 1]`` (1 at condition-image
      tokens), and the index maps that place text embeddings / image latents into the prefix.
    """

    bucket: int
    prefix_len: int
    target_hw: tuple
    cos_p: torch.Tensor
    sin_p: torch.Tensor
    cos_t: torch.Tensor
    sin_t: torch.Tensor
    prefix_bias: torch.Tensor
    target_bias: torch.Tensor
    is_img: torch.Tensor
    text_pos: torch.Tensor  # [n_text] long: prefix positions of the text tokens, in order
    text_src: torch.Tensor  # [n_text] long: their indices in the text-embedding sequence
    img_pos: torch.Tensor  # [n_img] long: prefix positions of the condition-image tokens


def _rope_freqs(cfg: QwenImage21DiTConfig):
    # Same tables as upstream ``QwenImage21Rope`` (theta 10000, 8192 positive + 1024 negative).
    from ._vendor.transformer_qwenimage21 import QwenImage21Rope

    return QwenImage21Rope(theta=10000, axes_dim=list(cfg.axes_dims_rope))


def build_layout(
    cfg: QwenImage21DiTConfig,
    img_mask: torch.Tensor,
    text_valid: torch.Tensor,
    img_shapes: list,
    bucket: int,
) -> PrefixLayout:
    """``img_mask`` ``[S_vlm]`` bool over the prompt's VLM sequence INCLUDING the appended target
    slots (upstream ``image_pad_mask`` after ``append_target_slots``); ``text_valid`` ``[B, S_text]``
    bool over the text embeddings (all True when unpadded); ``img_shapes`` per-image
    ``(1, h, w)`` latent sizes, condition images first, target last."""
    repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)
    joint_img = torch.repeat_interleave(img_mask, repeats)  # [S_joint]
    rope = _rope_freqs(cfg)(
        img_shapes, joint_img, device=torch.device("cpu")
    )  # complex [S_joint, D/2]
    from ._vendor.transformer_qwenimage21 import QwenImage21Transformer2DModel

    image_ids, target_mask = QwenImage21Transformer2DModel.build_token_metadata(
        joint_img, img_shapes
    )
    prefix_len = int((~target_mask).sum())
    n_target = int(target_mask.sum())
    if not bool(target_mask[prefix_len:].all()):
        raise ValueError("the target image must be the last block of the joint sequence")
    if prefix_len > bucket:
        raise ValueError(f"prefix is {prefix_len} tokens; bucket is {bucket}")
    b = text_valid.shape[0]
    pad = bucket - prefix_len

    cos, sin = rope.real.float(), rope.imag.float()
    cos_p = torch.cat([cos[:prefix_len], cos.new_ones(pad, cos.shape[1])])
    sin_p = torch.cat([sin[:prefix_len], sin.new_zeros(pad, sin.shape[1])])

    # key validity over the prefix: text padding (per sample) and the bucket padding
    key_valid = torch.zeros(b, bucket, dtype=torch.bool)
    key_valid[:, :prefix_len] = True
    text_pos = (~joint_img[:prefix_len]).nonzero(as_tuple=True)[0]
    text_src = (~img_mask[: text_valid.shape[1]]).nonzero(as_tuple=True)[0]
    key_valid[:, text_pos] = text_valid.bool()[:, text_src]

    ids = F.pad(image_ids[:prefix_len], (0, pad), value=-1)
    q_idx = torch.arange(bucket)[:, None]
    k_idx = torch.arange(bucket)[None, :]
    same_block = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
    allowed = ((q_idx >= k_idx) | same_block)[None] & key_valid[:, None, :]
    # padded query rows: let them see key 0 so no row is fully masked (their output is unused)
    allowed[:, :, 0] |= ~allowed.any(-1)
    prefix_bias = torch.where(allowed, 0.0, MASK_VALUE).float()[:, None]
    target_bias = torch.where(key_valid, 0.0, MASK_VALUE).float()[:, None, None, :]
    is_img = joint_img[:prefix_len].float()
    is_img = torch.cat([is_img, is_img.new_zeros(pad)])[:, None]
    img_pos = joint_img[:prefix_len].nonzero(as_tuple=True)[0]
    h, w = img_shapes[-1][1], img_shapes[-1][2]
    assert h * w == n_target
    return PrefixLayout(
        bucket,
        prefix_len,
        (h, w),
        cos_p.contiguous(),
        sin_p.contiguous(),
        cos[prefix_len:].contiguous(),
        sin[prefix_len:].contiguous(),
        prefix_bias.contiguous(),
        target_bias.contiguous(),
        is_img.contiguous(),
        text_pos,
        text_src,
        img_pos,
    )


def assemble_prefix_inputs(
    layout: PrefixLayout,
    text_embeds: torch.Tensor,
    cond_latents: torch.Tensor | None,
    in_channels: int,
    dtype: torch.dtype,
):
    """Scatter text embeddings ``[B, S_text, C]`` and packed condition latents ``[B, n_img, Cin]``
    into prefix-shaped dense inputs ``txt [B, L, C]`` / ``img [B, L, Cin]`` (zeros elsewhere)."""
    b, _, c = text_embeds.shape
    txt = text_embeds.new_zeros(b, layout.bucket, c, dtype=dtype)
    txt[:, layout.text_pos] = text_embeds[:, layout.text_src].to(dtype)
    img = text_embeds.new_zeros(b, layout.bucket, in_channels, dtype=dtype)
    if layout.img_pos.numel():
        if cond_latents is None or cond_latents.shape[1] != layout.img_pos.numel():
            raise ValueError("condition latents do not match the image slots of the prompt")
        img[:, layout.img_pos] = cond_latents.to(dtype)
    return txt.contiguous(), img.contiguous()


# --------------------------------------------------------------------------------------------
# Device module
# --------------------------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, cfg: QwenImage21DiTConfig, tp: int, dtype):
        super().__init__()
        d, hd = cfg.dim, cfg.head_dim
        self.cfg, self.tp = cfg, tp
        self.heads = cfg.num_heads // tp
        self.q = param((d, d), dtype, 0, tp)
        self.k = param((d, d), dtype, 0, tp)
        self.v = param((d, d), dtype, 0, tp)
        self.o = param((d, d), dtype, 1, tp)
        self.norm_q = param((hd,), dtype)
        self.norm_k = param((hd,), dtype)
        self.mlp_proj = param((d * cfg.mlp_ratio, d), dtype, 0, tp)
        self.mlp_gate = param((d * cfg.mlp_ratio, d), dtype, 0, tp)
        self.mlp_out = param((d, d * cfg.mlp_ratio), dtype, 1, tp)

    SHARD = {"q": 0, "k": 0, "v": 0, "o": 1, "mlp_proj": 0, "mlp_gate": 0, "mlp_out": 1}

    def qkv(self, x, mod1, cos, sin):
        """Modulated QKV with QK-norm and RoPE. ``mod1`` = (scale, gate) broadcast over tokens."""
        b, s, _ = x.shape
        scale, _ = mod1
        h = layer_norm(x, self.cfg.eps) * (1 + scale)
        q = F.linear(h, self.q).view(b, s, self.heads, -1)
        k = F.linear(h, self.k).view(b, s, self.heads, -1)
        v = F.linear(h, self.v).view(b, s, self.heads, -1)
        q = rope_pairs(rms_norm(q, self.norm_q, self.cfg.eps), cos, sin)
        k = rope_pairs(rms_norm(k, self.norm_k, self.cfg.eps), cos, sin)
        return q, k, v

    def finish(self, x, attn, mod1, mod2, group):
        """Output projection, gated residuals and the SwiGLU MLP."""
        b, s = x.shape[:2]
        attn = all_reduce(F.linear(attn.reshape(b, s, -1), self.o), self.tp, group)
        x = x + mod1[1].tanh() * attn
        h = layer_norm(x, self.cfg.eps) * (1 + mod2[0])
        mlp = F.linear(
            F.silu(F.linear(h, self.mlp_gate)) * F.linear(h, self.mlp_proj), self.mlp_out
        )
        return x + mod2[1].tanh() * all_reduce(mlp, self.tp, group)


class NeuronQwenImage21Transformer(nn.Module):
    """``forward_prefix`` -> per-layer prefix (K, V); ``forward_target`` -> velocity for the target."""

    def __init__(self, cfg: QwenImage21DiTConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg, self.dtype = cfg, dtype
        self.tp, self.tp_rank, self.tp_group = tp_state()
        if cfg.num_heads % self.tp:
            raise ValueError(f"tp={self.tp} must divide num_heads={cfg.num_heads}")
        d, c = cfg.dim, cfg.context_in_dim
        self.txt_norm = param((c,), dtype)
        self.txt_in1 = param((d, c), dtype)
        self.txt_in2 = param((d, d), dtype)
        self.img_in = param((d, cfg.in_channels), dtype)
        self.t_lin1 = param((d, 256), dtype)
        self.t_lin2 = param((d, d), dtype)
        self.mod = param((4 * d, d), dtype)
        self.norm_out = param((d, d), dtype)
        self.proj_out = param((cfg.out_channels, d), dtype)
        self.blocks = nn.ModuleList(_Block(cfg, self.tp, dtype) for _ in range(cfg.num_layers))
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("t_freqs", freqs, persistent=False)
        attach_shard_loaders(
            self,
            {
                f"blocks.{i}.{n}": dim
                for i in range(cfg.num_layers)
                for n, dim in _Block.SHARD.items()
            },
            self.tp,
        )

    # -- weights --------------------------------------------------------------------------
    def checkpoint_mappings(self) -> dict:
        m = {
            "txt_norm": "txt_in.text_norm.weight",
            "txt_in1": "txt_in.in_layer.weight",
            "txt_in2": "txt_in.out_layer.weight",
            "img_in": "img_in.weight",
            "t_lin1": "time_text_embed.timestep_embedder.linear_1.weight",
            "t_lin2": "time_text_embed.timestep_embedder.linear_2.weight",
            "mod": "modulation.1.weight",
            "norm_out": "norm_out.linear.weight",
            "proj_out": "proj_out.weight",
        }
        for i in range(self.cfg.num_layers):
            p, c = f"blocks.{i}", f"transformer_blocks.{i}"
            m.update(
                {
                    f"{p}.q": f"{c}.attn.to_q.weight",
                    f"{p}.k": f"{c}.attn.to_k.weight",
                    f"{p}.v": f"{c}.attn.to_v.weight",
                    f"{p}.o": f"{c}.attn.to_out.0.weight",
                    f"{p}.norm_q": f"{c}.attn.norm_q.weight",
                    f"{p}.norm_k": f"{c}.attn.norm_k.weight",
                    f"{p}.mlp_proj": f"{c}.img_mlp.proj.weight",
                    f"{p}.mlp_gate": f"{c}.img_mlp.gate_layer.weight",
                    f"{p}.mlp_out": f"{c}.img_mlp.out.weight",
                }
            )
        return m

    def load_weights(self, model_path: str, device="cpu") -> None:
        load_sharded(
            self,
            os.path.join(model_path, "transformer"),
            self.checkpoint_mappings(),
            self.tp_rank,
            self.tp,
            device,
        )

    # -- pieces ---------------------------------------------------------------------------
    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        """Upstream ``QwenImage21TimestepProjEmbeddings``: the timestep is cast to the model dtype
        first (as upstream does), then embedded in fp32 and projected in the model dtype."""
        t = t.to(self.dtype).float() * 1000.0
        args = t[:, None] * self.t_freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.dtype)
        return F.linear(F.silu(F.linear(emb, self.t_lin1)), self.t_lin2)

    def _modulation(self, temb: torch.Tensor):
        """Shared modulation -> ((scale1, gate1), (scale2, gate2)), each ``[B, 1, D]``."""
        m = F.linear(F.silu(temb), self.mod)[:, None]
        s1, g1, s2, g2 = m.chunk(4, dim=-1)
        return (s1, g1), (s2, g2)

    # -- graphs ---------------------------------------------------------------------------
    def forward_prefix(self, txt, img, is_img, cos, sin, bias):
        """txt ``[B, L, C]``, img ``[B, L, Cin]``, is_img ``[L, 1]``, cos/sin ``[L, D/2]`` fp32,
        bias ``[B, 1, L, L]`` fp32 -> ``(k_0..k_{n-1}, v_0..v_{n-1})`` each ``[B, H/tp, L, D]``."""
        tf = txt.float()  # zero-centred RMSNorm (effective scale = weight + 1), all in fp32
        tf = tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        t = (tf * (self.txt_norm.float() + 1)).to(self.dtype)
        t = F.linear(gelu_tanh(F.linear(t, self.txt_in1)), self.txt_in2)
        x = torch.where(is_img.to(torch.bool)[None], F.linear(img, self.img_in), t)
        temb = self._temb(torch.zeros(x.shape[0], device=x.device))
        mod1, mod2 = self._modulation(temb)
        scale = self.cfg.head_dim**-0.5
        ks, vs = [], []
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv(x, mod1, cos, sin)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            ks.append(k)
            vs.append(v)
            if i == last:  # the prefix's last-layer output is never read
                break
            a = attention(q.transpose(1, 2), k, v, scale, bias).transpose(1, 2)
            x = blk.finish(x, a, mod1, mod2, self.tp_group)
        return (*ks, *vs)

    def forward_target(self, latents, timestep, cos, sin, key_bias, *kv):
        """latents ``[B, N, Cin]``, timestep ``[B]`` (in [0, 1]), cos/sin ``[N, D/2]`` fp32,
        key_bias ``[B, 1, 1, L]`` fp32 over the prefix keys, kv = prefix (k_0.., v_0..)
        -> velocity ``[B, N, Cout]``."""
        n = len(self.blocks)
        ks, vs = kv[:n], kv[n:]
        x = F.linear(latents, self.img_in)
        temb = self._temb(timestep)
        mod1, mod2 = self._modulation(temb)
        scale = self.cfg.head_dim**-0.5
        # keys ordered [target | prefix]: attention is order-free over keys, and this keeps the
        # always-valid keys first (the layout masked flash kernels want)
        bias = torch.cat([key_bias.new_zeros(*key_bias.shape[:-1], x.shape[1]), key_bias], dim=-1)
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv(x, mod1, cos, sin)
            k = torch.cat([k.transpose(1, 2), ks[i]], dim=2)
            v = torch.cat([v.transpose(1, 2), vs[i]], dim=2)
            a = attention(q.transpose(1, 2), k, v, scale, bias).transpose(1, 2)
            x = blk.finish(x, a, mod1, mod2, self.tp_group)
        s = F.linear(F.silu(temb), self.norm_out)[:, None]
        return F.linear(layer_norm(x, self.cfg.eps) * (1 + s), self.proj_out)
