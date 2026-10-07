# SPDX-License-Identifier: Apache-2.0
"""NeuronCore-v2 (Inf2 / Trn1) attention kernel for Cosmos3-Edge, written against NKI 0.6.

nkilib's ``attention_cte`` (the plugin's trn2 path) needs LNC=2 and DMA-transpose, neither of
which exists on NeuronCore-v2. This is a single-pass, full-row softmax kernel derived from the
``attn_fwd_v8a`` tutorial in aws-neuron/nki-samples (NKI 0.6 API), generalised to what the Edge
towers need:

* multi-head with GQA (query head ``h`` reads K/V head ``h // n_rep``),
* ``Sk != Sq`` (text keys + video keys),
* an additive per-KEY bias (text bucket padding) folded into the QK matmul as a 129th contraction
  row: ``[q; 1] . [k; bias] = q.k + bias`` -- one extra rank-1 accumulate per score tile instead
  of a [128, Sk] bias tile in SBUF.

Layouts (all bf16, prepared by :func:`edge_attention_v2`):
  q    [H,  128, Sq]   d-major, pre-scaled by 1/sqrt(d); Sq % 128 == 0
  k    [Hk, 128, Sk]   d-major; Sk % 512 == 0
  v    [Hk, Sk, 128]   key-major (loaded as [128-key, d] tiles -- no on-chip transpose)
  bias [1,  Sk]        additive key bias (0 or MASK_VALUE)
  out  [H,  Sq, 128]

Per (head, 128-query tile) the full score row (Sk fp32) stays in SBUF; at Sk = 12.8k that is
~50 KB/partition, well inside NC-v2's 192 KB.
"""

from __future__ import annotations

import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch

_PT_COPY = os.environ.get("COSMOS3_EDGE_NC2_PT_COPY", "auto")  # auto | vector | scalar | gpsimd (measured: auto fastest)
_PT_COPY_ENGINE = {"vector": nisa.engine.vector, "scalar": nisa.engine.unknown, "gpsimd": nisa.engine.gpsimd,
                   "auto": nisa.engine.unknown}[_PT_COPY]


@nki.jit
def edge_attn_fwd_nc2(q, k, v, bias, tile_flags):
    n_heads, d_head, seqlen_q = q.shape
    n_kv, _, seqlen_kv = k.shape
    n_rep = n_heads // n_kv

    PMAX = nl.tile_size.pmax  # 128
    FMAX = nl.tile_size.gemm_moving_fmax  # 512
    assert d_head == PMAX
    assert seqlen_q % PMAX == 0
    assert seqlen_kv % FMAX == 0
    num_tile_q = seqlen_q // PMAX
    num_kv_tiles = seqlen_kv // FMAX
    num_kv_128 = seqlen_kv // PMAX
    # ``tile_flags`` carries only static shape information (its data is never read):
    # shape[0] - 1 = number of leading 512-key tiles with a non-zero bias (text prefix),
    # shape[1] - 1 = 1 if the last tile has a non-zero bias (alignment padding), else 0.
    n_head_bias = min(tile_flags.shape[0] - 1, num_kv_tiles)
    tail_bias = tile_flags.shape[1] - 1
    tail_lo = num_kv_tiles - 1 if (tail_bias and num_kv_tiles - 1 >= n_head_bias) else num_kv_tiles
    segments = [(0, n_head_bias, True), (n_head_bias, tail_lo, False), (tail_lo, num_kv_tiles, True)]
    segments = [s for s in segments if s[1] > s[0]]

    out = nl.ndarray((n_heads, seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    ones_row = nl.ndarray((1, PMAX), dtype=q.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)
    bias_sbuf = nl.ndarray((1, seqlen_kv), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bias_sbuf, src=bias[0:1, 0:seqlen_kv])

    for kh in range(n_kv):
        k_sbuf = nl.ndarray((PMAX, seqlen_kv), dtype=k.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_sbuf, src=k[kh, 0:PMAX, 0:seqlen_kv])
        v_sbuf = nl.ndarray((PMAX, num_kv_128, PMAX), dtype=v.dtype, buffer=nl.sbuf)
        for j in range(num_kv_128):
            nisa.dma_copy(dst=v_sbuf[:, j, :], src=v[kh, j * PMAX:(j + 1) * PMAX, 0:PMAX])

        for r in range(n_rep):
            h = kh * n_rep + r
            for i_q in range(num_tile_q):
                q_sbuf = nl.ndarray((PMAX, PMAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_sbuf, src=q[h, 0:PMAX, i_q * PMAX:(i_q + 1) * PMAX])

                # --- scores = q.k + bias, row max ---
                qk_sbuf = nl.ndarray((PMAX, num_kv_tiles, FMAX), dtype=nl.float32, buffer=nl.sbuf)
                row_max_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
                # The rank-1 bias accumulate streams a full 512-column moving tile through the
                # TensorEngine (as costly as the q.k tile itself), so it only runs on the tiles
                # whose bias is non-zero: the text prefix and the alignment-padding tail.
                for lo, hi, with_bias in segments:
                    for j in range(lo, hi):
                        qk_psum = nl.zeros((PMAX, FMAX), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=qk_psum, stationary=q_sbuf,
                                       moving=k_sbuf[0:PMAX, j * FMAX:(j + 1) * FMAX])
                        if with_bias:
                            nisa.nc_matmul(dst=qk_psum, stationary=ones_row,
                                           moving=bias_sbuf[0:1, j * FMAX:(j + 1) * FMAX])
                        nisa.tensor_scalar_reduce(dst=qk_sbuf[:, j, :], data=qk_psum,
                                                  op0=nl.multiply, operand0=1.0,
                                                  reduce_op=nl.maximum,
                                                  reduce_res=row_max_kv[:, j:j + 1])
                neg_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=row_max_kv, axis=(1,), negate=True)

                # --- p = exp(s - max), row sums ---
                exp_row = nl.ndarray((PMAX, seqlen_kv), dtype=q.dtype, buffer=nl.sbuf)
                sum_kv = nl.ndarray((PMAX, num_kv_tiles), dtype=nl.float32, buffer=nl.sbuf)
                for j in range(num_kv_tiles):
                    nisa.activation(dst=exp_row[:, j * FMAX:(j + 1) * FMAX], op=nl.exp,
                                    data=qk_sbuf[:, j, :], bias=neg_max,
                                    reduce_op=nl.add, reduce_res=sum_kv[:, j:j + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)

                # --- p^T tiles for the PV contraction ---
                p_t = nl.ndarray((PMAX, num_kv_128, PMAX), dtype=q.dtype, buffer=nl.sbuf)
                for j in range(num_kv_128):
                    pt_psum = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)  # NC-v2: transpose writes fp32 PSUM
                    nisa.nc_transpose(dst=pt_psum, data=exp_row[:, j * PMAX:(j + 1) * PMAX])
                    # PSUM->SBUF evacuation engine (COSMOS3_EDGE_NC2_PT_COPY). Measured on inf2: moving
                    # it off the default engine to ScalarE/GpSimd did not help (ScalarE also does exp).
                    if _PT_COPY == "scalar":  # NC-v2: ScalarE copy is an activation(copy)
                        nisa.activation(dst=p_t[:, j, :], op=nl.copy, data=pt_psum)
                    else:
                        nisa.tensor_copy(dst=p_t[:, j, :], src=pt_psum, engine=_PT_COPY_ENGINE)

                # --- o = (p @ v) / sum ---
                o_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
                for j in range(num_kv_128):
                    nisa.nc_matmul(dst=o_psum, stationary=p_t[:, j, :], moving=v_sbuf[:, j, :])
                row_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=row_sum, op=nl.add, data=sum_kv, axis=(1,))
                inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)
                o_sbuf = nl.ndarray((PMAX, PMAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=o_sbuf, data=o_psum, op0=nl.multiply, operand0=inv_sum)
                nisa.dma_copy(dst=out[h, i_q * PMAX:(i_q + 1) * PMAX, 0:PMAX], src=o_sbuf)
    return out


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def prepare_nc2_inputs(q, k, v, scale, key_bias):
    """[B=1, H, S, D] torch tensors -> kernel layouts (padded). Returns (args, sq)."""
    from .attention import MASK_VALUE

    assert q.shape[0] == 1, "batch 1 only"
    _, h, sq, d = q.shape
    _, hk, sk, _ = k.shape
    sq_p, sk_p = _round_up(sq, 128), _round_up(sk, 512)
    qd = (q[0] * scale).to(torch.bfloat16)
    if sq_p != sq:
        qd = torch.nn.functional.pad(qd, (0, 0, 0, sq_p - sq))
    kk, vv = k[0].to(torch.bfloat16), v[0].to(torch.bfloat16)
    if sk_p != sk:
        kk = torch.nn.functional.pad(kk, (0, 0, 0, sk_p - sk))
        vv = torch.nn.functional.pad(vv, (0, 0, 0, sk_p - sk))
    # prefix (text) bias + zeros for the video keys + MASK for the alignment padding
    parts = []
    if key_bias is not None:
        parts.append(key_bias.reshape(1, key_bias.shape[-1]).to(torch.bfloat16))
    n_pre = parts[0].shape[-1] if parts else 0
    parts.append(torch.zeros(1, sk - n_pre, dtype=torch.bfloat16, device=q.device))
    if sk_p != sk:
        parts.append(torch.full((1, sk_p - sk), MASK_VALUE, dtype=torch.bfloat16, device=q.device))
    bias = torch.cat(parts, dim=-1)
    # static tile map for the kernel: leading tiles touched by the text prefix, plus the padded tail
    n_head_tiles = -(-n_pre // 512)
    has_tail = int(sk_p != sk)
    tile_flags = torch.zeros(n_head_tiles + 1, has_tail + 1, dtype=torch.bfloat16, device=q.device)
    args = (qd.transpose(1, 2).contiguous(), kk.transpose(1, 2).contiguous(), vv.contiguous(),
            bias.contiguous(), tile_flags)
    return args, sq


def _launch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
            tile_flags: torch.Tensor) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(edge_attn_fwd_nc2)[1](q, k, v, bias, tile_flags)  # NC-v2 has no LNC=2: one-core launch


try:
    from vllm_omni_neuron.lite_compat import nki_op

    _nc2_op = nki_op("cosmos3_edge::attn_nc2")(_launch)
except Exception:  # CPU-only environment without the Lite runtime
    _nc2_op = None


def nc2_attention(q, k, v, scale, key_bias=None):
    """[1, H, Sq, 128] q, [1, Hk, Sk, 128] k/v -> [1, H, Sq, 128] via the NC-v2 kernel."""
    args, sq = prepare_nc2_inputs(q, k, v, scale, key_bias)
    return _nc2_op(*args)[:, :sq][None].to(q.dtype)
