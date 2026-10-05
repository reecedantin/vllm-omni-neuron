# SPDX-License-Identifier: Apache-2.0
"""Index-list block-sparse attention kernel for Trainium2 (NeuronCore-v3, NKI 0.6).

One item = one head x ``QT`` = 128 query rows attending to ``KP`` listed key blocks of ``BLK`` rows
(64 or 128: SLA selects 64- or 128-row key blocks; SSTA's 384-token tiles are expanded by the caller
into three consecutive 128-row sub-blocks). The list is static in length and dynamic in content: the
caller pads short lists with the index of a trailing DUMMY block whose K rows are zero and whose
validity is zero, so every item does the same work and every pad contributes exactly nothing.

Per item the kernel walks the list in passes of ``PB`` blocks (``PB * BLK`` keys, ~2-3k, the same
softmax granularity as the dense kernel) with an online softmax across passes:

* K^T of the pass's blocks arrives by ONE indirect DMA transpose (one descriptor per block) into
  ``kT (D, BLK, PB)``: key column ``c = j * PB + t`` (``j`` = row within block, ``t`` = list slot).
* ``S = q^T K`` in fp32 (512-key chunks), running row max, ``P = exp(S - m)`` in bf16, ``P^T`` by
  512-key DMA transposes into 128-key blocks (block ``m`` = columns ``[128 m, 128 m + 128)``).
* V rows are gathered to match: for block ``m`` and partition ``p``, key ``c = 128 m + p`` is
  ``(t = p % PB, j = 128 m / PB + p // PB)``, so partition ``p`` reads rows
  ``blocks[t] * BLK + p // PB + m * (128 // PB)`` -- a strided run, one descriptor per partition per
  pass. ``V_ext = [V | valid | 0]`` (width ``D + 32``) carries a validity column so the softmax
  denominator is ``P @ valid``: pad rows of a real block (validity 0, K 0) and dummy blocks drop
  out exactly; zero-padded rows that MUST stay in the softmax (SSTA's zero pad) get validity 1.
* ``O = O * alpha + P^T V_ext`` with ``alpha = exp(m_old - m_new)``; the final ``1 / O[:, D]``
  normalises.

Constraints: ``D == 128``; ``PB % 16 == 0`` (the indirect DMA transpose gathers 16 blocks at a time)
and ``128 % PB == 0``, so ``PB`` is 16, 32, 64 or 128; ``KP % PB == 0``; ``(PB * BLK) % 512 == 0``;
``Lq % 128 == 0``. ``q`` is pre-scaled by the caller (``scale`` is not applied here). SPMD grid
``[2]`` shards the query tiles across the LNC2 pair when ``n_qt % 2 == 0``.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

QT = 128
KCH = 512
PMAX = 128
NEG_INIT = -30000.0  # running max before the first pass; exp(NEG_INIT - m) == 0 for any real m


def v_ext_width(d: int) -> int:
    return d + 32


@nki.jit
def bsa_index_list(qT, k3, v_ext, blocks_i32, rowstart_i32, pb: int):
    """qT (H, D, Lq) bf16 pre-scaled; k3 (H, NB1, BLK, D) bf16 block-major (last block = dummy
    zeros); v_ext (H, NB1 * BLK, DV) bf16 with validity at column D; blocks_i32 (H, n_qt, KP, 1)
    int32 block ids into k3's block axis; rowstart_i32 (H, n_qt, KP // pb, 128, 1) int32 =
    blocks[pass * pb + p % pb] * BLK + p // pb (per-pass, per-partition V row starts).
    Returns (H, Lq, D) bf16."""
    H, D, Lq = qT.shape
    _, NB1, BLK, _ = k3.shape
    DV = v_ext.shape[2]
    KP = blocks_i32.shape[2]
    n_pass = KP // pb
    NKEYS = pb * BLK  # keys per pass
    NCH = NKEYS // KCH
    NB = NKEYS // PMAX  # 128-key PV blocks per pass
    RSTRIDE = PMAX // pb  # V row stride between consecutive PV blocks for one partition
    n_qt = Lq // QT
    out = nl.ndarray((H, Lq, D), dtype=qT.dtype, buffer=nl.shared_hbm)
    grid_ndim = nl.program_ndim()
    n_prg, prg = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    qt_per_prg = n_qt // n_prg

    for h in range(H):
        for qtl in range(qt_per_prg):
            qt = prg * qt_per_prg + qtl
            q0 = qt * QT
            qT_sb = nl.ndarray((D, QT), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=qT_sb, src=qT[h, :, nl.ds(q0, QT)])
            o_acc = nl.ndarray((QT, DV), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(o_acc, 0.0)
            m_run = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(m_run, NEG_INIT)
            for ps in range(n_pass):
                idx_sb = nl.ndarray((pb, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=idx_sb, src=blocks_i32[h, qt, nl.ds(ps * pb, pb), :])
                ridx_sb = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=ridx_sb, src=rowstart_i32[h, qt, ps])
                # --- K^T of the pass's blocks: one indirect DMA transpose, one descriptor per block
                kT_sb = nl.ndarray((D, BLK, pb), dtype=k3.dtype, buffer=nl.sbuf)
                nisa.dma_transpose(
                    dst=kT_sb,
                    src=k3[h].vector_select(0, idx_sb.view(nl.uint32)),
                    dge_mode=nisa.dge_mode.swdge,
                )
                kT_flat = kT_sb.reshape((D, NKEYS))
                # --- V_ext rows in PV-block layout: partition p, block m -> key c = 128 m + p
                v_sb = nl.ndarray((PMAX, NB, DV), dtype=v_ext.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=v_sb,
                    src=v_ext[h].ap(
                        pattern=[[DV, PMAX], [RSTRIDE * DV, NB], [1, DV]],
                        vector_offset=ridx_sb.view(nl.uint32),
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                )
                # --- S = q^T K (fp32)
                s_sb = nl.ndarray((QT, NKEYS), dtype=nl.float32, buffer=nl.sbuf)
                for c in range(NCH):
                    s_ps = nl.ndarray((QT, KCH), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=s_ps, stationary=qT_sb, moving=kT_flat[:, c * KCH : (c + 1) * KCH]
                    )
                    nisa.tensor_copy(dst=s_sb[:, c * KCH : (c + 1) * KCH], src=s_ps)
                # --- online softmax bookkeeping
                m_loc = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=m_loc, op=nl.maximum, data=s_sb, axis=(1,), keepdims=True)
                m_new = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=m_new, data1=m_run, data2=m_loc, op=nl.maximum)
                diff = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=diff, data1=m_run, data2=m_new, op=nl.subtract)
                alpha = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=alpha, op=nl.exp, data=diff)
                nisa.tensor_copy(dst=m_run, src=m_new)
                negm = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=negm, data=m_new, op0=nl.multiply, operand0=-1.0)
                # --- P = exp(S - m) (bf16), P^T by DMA transposes into 128-key blocks
                p_sb = nl.ndarray((QT, NKEYS), dtype=qT.dtype, buffer=nl.sbuf)
                pT_sb = nl.ndarray((PMAX, NB, QT), dtype=qT.dtype, buffer=nl.sbuf)
                for c in range(NCH):
                    nisa.activation(
                        dst=p_sb[:, c * KCH : (c + 1) * KCH],
                        op=nl.exp,
                        data=s_sb[:, c * KCH : (c + 1) * KCH],
                        bias=negm,
                    )
                    nisa.dma_transpose(
                        dst=pT_sb.ap(
                            [[NB * QT, PMAX], [1, 1], [QT, KCH // PMAX], [1, QT]],
                            offset=c * (KCH // PMAX) * QT,
                        ),
                        src=p_sb.ap(
                            [[NKEYS, QT], [1, 1], [PMAX, KCH // PMAX], [1, PMAX]], offset=c * KCH
                        ),
                    )
                # --- O_pass = P^T V_ext; O = O * alpha + O_pass
                o_ps = nl.ndarray((QT, DV), dtype=nl.float32, buffer=nl.psum)
                for m in range(NB):
                    nisa.nc_matmul(dst=o_ps, stationary=pT_sb[:, m, :], moving=v_sb[:, m, :])
                nisa.tensor_scalar(dst=o_acc, data=o_acc, op0=nl.multiply, operand0=alpha)
                nisa.tensor_tensor(dst=o_acc, data1=o_acc, data2=o_ps, op=nl.add)
            inv = nl.ndarray((QT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv, data=o_acc[:, D : D + 1])
            o_sb = nl.ndarray((QT, D), dtype=qT.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_sb, data=o_acc[:, 0:D], op0=nl.multiply, operand0=inv)
            nisa.dma_copy(dst=out[h, nl.ds(q0, QT), :], src=o_sb)
    return out
