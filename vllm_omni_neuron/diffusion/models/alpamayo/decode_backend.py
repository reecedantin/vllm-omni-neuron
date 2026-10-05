# SPDX-License-Identifier: Apache-2.0
"""Static-shape attention for the Alpamayo VLM reasoning tower and the flow-matching expert.

Stage 1 of Alpamayo is autoregressive (prompt prefill, then up to ``max_new_tokens`` one-token decode
steps); stage 2 runs the expert ``num_inference_steps`` times against the same KV cache. Every
graph here has a FIXED shape so the whole request compiles to one prefill graph per prompt bucket,
ONE decode graph and ONE expert graph:

* The KV cache is a fixed ``[1, kv_heads, max_len, head_dim]`` pair per layer
  (``max_len = largest prompt bucket + max_new_tokens``). Each decode step reads ALL of it; which
  slots are live is carried by an additive ``bias`` (``0`` = attend, :data:`MASK_VALUE` = masked),
  and the step's new K/V lands through a one-hot ``write_mask`` (``torch.where``) -- no
  data-dependent slicing, no growing shapes.
* The functions are pure: caches go in and the updated caches come out (no in-place mutation of
  graph inputs).
* Masked attention is spelled out as ``softmax((q @ k^T) * scale + bias) @ v`` in fp32: under
  ``neuron_native_lite``, ``F.scaled_dot_product_attention`` with ANY mask (``attn_mask=`` or
  ``is_causal=True``) lowers to an opaque fused-attention op that fails to legalize; only maskless SDPA decomposes. :func:`attention` uses SDPA only when ``bias is None``.
  The mask value is finite (``-inf`` poisons fully-masked rows and bf16 casts).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

MASK_VALUE = -30000.0


@dataclass
class DecodeConfig:
    """Static shape/precision contract for one request's KV cache."""

    q_heads: int
    kv_heads: int
    head_dim: int
    max_len: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def num_kv_groups(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    """GQA expansion ``[B, Hk, S, D] -> [B, Hk * n, S, D]`` (head ``h`` reads kv head ``h // n``)."""
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``q`` ``[B, H, Sq, D]``, ``k``/``v`` ``[B, H, Sk, D]`` (GQA already expanded), fp32 math,
    result in ``q.dtype``. ``bias``: additive mask broadcastable to ``[B, H, Sq, Sk]``."""
    if bias is None:
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale).to(
            q.dtype
        )
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale + bias.float()
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def prefill(cfg: DecodeConfig, q, k, v, bias):
    """Prompt attention over its own ``S`` tokens (``bias`` = causal + padding mask ``[1,1,S,S]``).
    Returns ``(out [1, q_heads, S, D], k_cache, v_cache)`` with the caches zero-padded from ``S`` to
    ``cfg.max_len`` (slots ``>= S`` are masked by every later step until written)."""
    out = attention(
        q, repeat_kv(k, cfg.num_kv_groups), repeat_kv(v, cfg.num_kv_groups), cfg.scale, bias
    )
    pad = cfg.max_len - k.shape[2]
    k_cache = F.pad(k.to(cfg.dtype), (0, 0, 0, pad))
    v_cache = F.pad(v.to(cfg.dtype), (0, 0, 0, pad))
    return out, k_cache, v_cache


def decode_step(cfg: DecodeConfig, q, k, v, k_cache, v_cache, write_mask, bias):
    """One new token: ``q``/``k``/``v`` ``[1, heads, 1, D]``. ``write_mask`` ``[1, 1, max_len, 1]``
    (bool, one-hot at the token's cache slot) places ``k``/``v``; ``bias`` ``[1, 1, 1, max_len]``
    keeps every slot ``<=`` that one. Returns ``(out, k_cache', v_cache')``."""
    k_cache = torch.where(write_mask, k.to(k_cache.dtype), k_cache)
    v_cache = torch.where(write_mask, v.to(v_cache.dtype), v_cache)
    out = attention(
        q,
        repeat_kv(k_cache, cfg.num_kv_groups),
        repeat_kv(v_cache, cfg.num_kv_groups),
        cfg.scale,
        bias,
    )
    return out, k_cache, v_cache


def cross_step(num_kv_groups: int, scale: float, q, k, v, k_cache, v_cache, bias):
    """The expert: ``n`` action tokens attend (non-causally) over the full fixed-length VLM cache
    PLUS their own ``n`` K/V, i.e. ``max_len + n`` keys. ``bias`` ``[1, 1, n, max_len + n]`` masks
    every cache slot that upstream's cropped cache would not hold. The cache is not modified."""
    kf = torch.cat([k_cache, k.to(k_cache.dtype)], dim=2)
    vf = torch.cat([v_cache, v.to(v_cache.dtype)], dim=2)
    return attention(q, repeat_kv(kf, num_kv_groups), repeat_kv(vf, num_kv_groups), scale, bias)
