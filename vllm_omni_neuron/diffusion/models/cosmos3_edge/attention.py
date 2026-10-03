# SPDX-License-Identifier: Apache-2.0
"""Attention for the Cosmos3-Edge port, dispatched by NeuronCore generation.

Two paths, chosen per call by :func:`nc_dispatch.use_nki_kernels`:

* **torch** (NeuronCore-v2: Inf2 / Trn1, and CPU): plain tensor ops lowered by
  ``torch.compile``. Softmax is always fp32; Q·K^T runs in bf16 or fp32 (``qk_fp32``);
  long query sequences are processed in static blocks of ``qblock`` queries so the
  ``[B, H, Sq, Sk]`` score matrix never exists whole (on trn1 this cut a 5k-token graph's
  scratchpad from 2.4 GB to 0.5 GB). GQA by repeating K/V heads.
* **nki** (NeuronCore-v3+: Trn2 / Trn3): the plugin's ``attention_cte`` flash kernel (the
  one Wan2.2 uses) for non-causal, unmasked attention; anything else falls back to torch.

Layouts: q ``[B, H, Sq, D]``, k/v ``[B, Hk, Sk, D]`` with ``H % Hk == 0``; returns
``[B, H, Sq, D]`` in ``q.dtype``. ``key_bias`` is an optional additive fp32 bias
broadcastable to ``[B, 1, 1, Sk]`` (0 for valid keys, a large negative value for padding).
"""

from __future__ import annotations

import os

import torch

from .nc_dispatch import use_nki_kernels

DEFAULT_QBLOCK = int(os.environ.get("COSMOS3_EDGE_ATTN_QBLOCK", "512"))
DEFAULT_QK_FP32 = os.environ.get("COSMOS3_EDGE_ATTN_QK_FP32", "0") == "1"
MASK_VALUE = -30000.0  # finite: -inf poisons fully-masked rows and bf16 casts


def key_padding_bias(valid: torch.Tensor) -> torch.Tensor:
    """``[B, Sk]`` bool validity -> ``[B, 1, 1, Sk]`` fp32 additive bias."""
    return torch.where(valid, 0.0, MASK_VALUE).to(torch.float32)[:, None, None, :]


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, hk, s, d = x.shape
    return x[:, :, None].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)


def expand_key_bias(key_bias: torch.Tensor | None, sk: int) -> torch.Tensor | None:
    """A PREFIX key bias ``[B, 1, 1, S_text]`` (text-bucket padding only) -> full ``[B, 1, 1, sk]``.

    The GEN tower passes only the text part: the video keys are never masked, and a long odd-length
    fp32 bias input (``S_text + S_gen``) trips an NC-v2 runtime IO-DMA transpose limit.
    """
    if key_bias is None or key_bias.shape[-1] == sk:
        return key_bias
    pad = sk - key_bias.shape[-1]
    return torch.cat([key_bias, key_bias.new_zeros(*key_bias.shape[:-1], pad)], dim=-1)


def _scores_to_out(scores, v, row0, total_sq, causal, key_bias):
    """Softmax(scores [+ bias] [+ causal]) @ v for query rows ``row0 .. row0 + block``."""
    key_bias = expand_key_bias(key_bias, scores.shape[-1])
    if key_bias is not None:
        scores = scores + key_bias
    if causal:
        block, sk = scores.shape[-2], scores.shape[-1]
        rows = torch.arange(block, device=scores.device)[:, None] + row0
        cols = torch.arange(sk, device=scores.device)[None, :]
        # bottom-right aligned causal mask (query i sees keys <= i + sk - total_sq)
        scores = scores.masked_fill(cols > rows + (sk - total_sq), MASK_VALUE)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(v.dtype), v)


def lean_attention(q, k, v, scale, *, key_bias=None, qblock=512, hblock=0):
    """Memory-lean non-causal attention for long sequences on the torch-lowered path.

    The plain path materialises fp32 scores and runs ~5 full passes over them (bias, max, exp,
    sum, divide, cast); at 12k x 12.6k keys x 16 heads that is tens of GB of HBM traffic per
    layer. Here: scale folded into q, scores and probabilities stay bf16, the row sum accumulates
    in fp32, and normalisation is applied after P@V (on [S, D] instead of [S, Sk]). Optional
    head blocking keeps one (head-block x query-block) score tile small enough to stay on-chip.
    """
    n_rep = q.shape[1] // k.shape[1]
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
    q = q * scale
    kt = k.transpose(-1, -2)
    key_bias = expand_key_bias(key_bias, k.shape[-2])
    bias = key_bias.to(q.dtype) if key_bias is not None else None
    hs = q.shape[1]
    hb = hblock or hs
    sq = q.shape[-2]
    qb = qblock or sq
    rows = []
    for s0 in range(0, sq, qb):
        heads = []
        for h0 in range(0, hs, hb):
            s = torch.matmul(q[:, h0 : h0 + hb, s0 : s0 + qb], kt[:, h0 : h0 + hb])
            if bias is not None:
                s = s + bias
            p = torch.exp(s - s.amax(dim=-1, keepdim=True))
            denom = torch.sum(p, dim=-1, keepdim=True, dtype=torch.float32)
            o = torch.matmul(p, v[:, h0 : h0 + hb])
            heads.append((o.float() / denom).to(q.dtype))
        rows.append(heads[0] if len(heads) == 1 else torch.cat(heads, dim=1))
    return rows[0] if len(rows) == 1 else torch.cat(rows, dim=-2)


ATTN_IMPL = os.environ.get("COSMOS3_EDGE_ATTN_KIND", "plain")
LEAN_HBLOCK = int(os.environ.get("COSMOS3_EDGE_ATTN_HBLOCK", "0"))
LEAN_MIN_SQ = int(os.environ.get("COSMOS3_EDGE_ATTN_LEAN_MIN_SQ", "2048"))


def torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    causal: bool = False,
    key_bias: torch.Tensor | None = None,
    qblock: int | None = None,
    qk_fp32: bool | None = None,
) -> torch.Tensor:
    qblock = DEFAULT_QBLOCK if qblock is None else qblock
    qk_fp32 = DEFAULT_QK_FP32 if qk_fp32 is None else qk_fp32
    if ATTN_IMPL == "lean" and not causal and not qk_fp32 and q.shape[-2] >= LEAN_MIN_SQ:
        return lean_attention(q, k, v, scale, key_bias=key_bias, qblock=qblock, hblock=LEAN_HBLOCK)
    n_rep = q.shape[1] // k.shape[1]
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
    kt = (k.float() if qk_fp32 else k).transpose(-1, -2)
    sq = q.shape[-2]
    if not qblock or sq <= qblock:
        scores = torch.matmul(q.float() if qk_fp32 else q, kt).float() * scale
        return _scores_to_out(scores, v, 0, sq, causal, key_bias).to(q.dtype)
    outs = []
    for s0 in range(0, sq, qblock):  # static loop: unrolled into the compiled graph
        qb = q[:, :, s0 : s0 + qblock]
        scores = torch.matmul(qb.float() if qk_fp32 else qb, kt).float() * scale
        outs.append(_scores_to_out(scores, v, s0, sq, causal, key_bias))
    return torch.cat(outs, dim=-2).to(q.dtype)


def _nki_attention(q, k, v, scale):
    from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

    n_rep = q.shape[1] // k.shape[1]
    out_dmajor = _nki_attend(q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep), scale)  # [B, H, D, Sq]
    return out_dmajor.transpose(2, 3)


def _nc2_kernel_ok(q, causal) -> bool:
    if causal or os.environ.get("COSMOS3_EDGE_ATTN_NC2", "1") != "1":
        return False
    if q.device.type != "neuron" or q.shape[0] != 1 or q.shape[-1] != 128 or q.shape[-2] < NC2_MIN_SQ:
        return False
    from .nc_dispatch import neuron_core_generation

    return neuron_core_generation() == 2


NC2_MIN_SQ = int(os.environ.get("COSMOS3_EDGE_ATTN_NC2_MIN_SQ", "256"))


def edge_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    causal: bool = False,
    key_bias: torch.Tensor | None = None,
    qblock: int | None = None,
    qk_fp32: bool | None = None,
) -> torch.Tensor:
    if not causal and key_bias is None and use_nki_kernels(q):
        return _nki_attention(q, k, v, scale)
    if _nc2_kernel_ok(q, causal):
        from .nki_attention_nc2 import nc2_attention

        return nc2_attention(q, k, v, scale, key_bias)
    return torch_attention(q, k, v, scale, causal=causal, key_bias=key_bias, qblock=qblock, qk_fp32=qk_fp32)
