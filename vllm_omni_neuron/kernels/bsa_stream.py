# SPDX-License-Identifier: Apache-2.0
"""Streaming block-sparse attention kernel for Trainium2 (NeuronCore-v3, NKI 0.6).

Same contract as :mod:`vllm_omni_neuron.kernels.bsa_index_list` (one item = one head x one query
block attending to a fixed-width list of key blocks), built the way the dense ``attention_cte``
inner loop is built so that a listed key costs about what a dense key costs:

* **K/V reuse across the query block.** An item is a whole plan query block (128 or 384 rows =
  ``n_sub`` 128-row sub-tiles); every K/V pass is fetched once and applied to all sub-tiles.
* **One plain DMA per listed block, hardware descriptor generation.** K is pre-laid out as ``K^T``
  blocks ``(NBG, D, KB)`` and V as row blocks ``(NBG, KB, D)``; block ``t`` of a pass is ONE
  ``dma_copy`` with a dynamic (scalar) start address read from the SBUF list, on the Sync engine's
  HWDGE queue. No DMA transpose for K, no per-partition (software) gather for V.
* **Prefetch.** K/V passes are multi-buffered (``nbuf``) and issued ``nbuf - 1`` passes ahead
  (across item boundaries), so the loads overlap the previous passes' compute.
* **attention_cte's per-pass math.** ``S = q K^T`` in 512-key PSUM chunks; ``range_select`` copies
  each chunk to SBUF with the key mask and the running chunk max in one Vector instruction; one
  online-softmax rescale per pass; ``exp`` with the row sum fused (``activation_reduce``); ``P^T``
  by DMA transposes; ``P^T V`` accumulated in PSUM over the pass.
* **Masking by a bound, not by data.** Lists hold real blocks first (full ones, then the single
  partial block if any), then pads. ``bounds[i, :, p]`` = valid keys of item ``i`` counted from the
  start of pass ``p`` (``n_valid - p * NK``); keys at or past it are dropped exactly. Pad entries may
  point at any real block (they are masked), so no dummy block or validity column is needed.
* **Software pipelining** across (pass, sub-tile) steps: the QK^T + max of step ``s + 1`` is issued
  before the exp / PV of step ``s`` (two-deep skew, double-buffered score/probability tiles).

Layouts (``n_items`` even: the LNC2 pair takes half each; the host pads with a duplicate item):
``qT (D, n_items * QB)`` bf16, pre-scaled by the softmax scale, item ``i`` = columns
``[i * QB, (i + 1) * QB)``; ``kT_blk (NBG, D, KB)`` / ``v_blk (NBG, KB, D)`` bf16 with global block
ids (``head * NB + block``); ``lists (n_items, KP)`` int32 global block ids; ``bounds
(n_items, 128, n_pass)`` fp32 (the per-pass bound repeated on every partition). Returns
``(n_items * QB, D)`` bf16.

Constraints: ``D == 128``; ``KB`` in {32, 64} or a multiple of 128; ``NK = pb * KB`` a multiple of
128; ``KP == n_pass * pb``; ``QB % 128 == 0``.

The body is written for NKI's parser frontend: no inner functions, comprehensions or tuple loop
targets, so the three pipeline stages are inlined in one loop over (pass, sub-tile) steps.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
KCH = 512  # keys per QK^T matmul / range_select / exp / P-transpose chunk (one PSUM bank)
NBUF = 3  # default K/V pass buffers (prefetch depth NBUF - 1)
FP32_MIN = -3.4028235e38


@nki.jit
def bsa_stream(
    qT,
    kT_blk,
    v_blk,
    lists,
    bounds,
    q_block: int,
    pb: int,
    nbuf: int = NBUF,
    v_queue: int = 0,
    packed: int = 0,
    k_block: int = 0,
    kmask=None,
    qind=None,
    masked: int = 0,
):
    """``nbuf``: K/V pass buffers; ``v_queue``: 0 = all loads on the Sync HWDGE queue, 1 = split
    over the Sync and Scalar queues (V on Scalar; packed: alternate blocks).

    ``masked=1``: an additive per-key mask for G row groups of every 128-row sub-tile.
    ``kmask`` ``(n_items, G, KP * KB)`` bf16 (0 or a large negative per listed key, in list order),
    ``qind`` ``(G, 128)`` bf16 0/1 row-group indicator. One K = G matmul per 512-key chunk adds
    ``qind^T kmask`` into the QK^T PSUM before the masked copy (as attention_cte pre-loads its
    position bias). Uses: VSA's pairs of 64-row query tiles attending the union of their lists
    (G = 2, rows 0-63 / 64-127), super-blocks fetched whole with one half unselected (G = 1). Every
    row must keep at least one unmasked key.

    ``packed=1`` (``k_block % 128 == 0``): ``kT_blk`` is ONE packed tensor ``(NBG, 128, KB + s*D)``,
    row ``p`` = ``[K^T[d=p, 0:KB] | V[p] | V[128 + p] | ...]`` (``s = KB // 128``), so a listed block
    is a single DMA with ``4 * KB`` contiguous bytes per partition; ``v_blk`` is ignored and
    ``k_block`` gives ``KB``."""
    D = qT.shape[0]
    KB = kT_blk.shape[2]
    if packed == 1:
        KB = k_block
    n_items = lists.shape[0]
    KP = lists.shape[1]
    n_pass = bounds.shape[2]
    QB = q_block
    n_sub = QB // P
    NK = pb * KB
    NB = NK // P  # 128-key PV blocks per pass
    nch = (NK + KCH - 1) // KCH
    r_small = 1
    s_big = 1
    if KB < P:
        r_small = P // KB  # list slots per 128-key V tile
    else:
        s_big = KB // P  # 128-key V tiles per list slot
    nb = nbuf

    out = nl.ndarray((n_items * QB, D), dtype=qT.dtype, buffer=nl.shared_hbm)
    grid_ndim = nl.program_ndim()
    n_prg, prg = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    per = n_items // n_prg

    # ---------------------------------------------------------------- persistent buffers
    zero_b = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(zero_b, 0.0)
    km_sb = []
    if masked == 1:
        G = qind.shape[0]
        qind_sb = nl.ndarray((G, P), dtype=qind.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qind_sb, src=qind)
        for _i in range(nbuf):
            km_sb.append(nl.ndarray((G, pb * KB), dtype=kmask.dtype, buffer=nl.sbuf))
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    KBW = KB + s_big * D  # packed row width
    kT_sb = []
    v_sb = []
    kv_sb = []
    for _i in range(nb):
        if packed == 1:
            kv_sb.append(nl.ndarray((P, pb, KBW), dtype=kT_blk.dtype, buffer=nl.sbuf))
        else:
            kT_sb.append(nl.ndarray((D, NK), dtype=kT_blk.dtype, buffer=nl.sbuf))
            v_sb.append(nl.ndarray((P, NB, D), dtype=v_blk.dtype, buffer=nl.sbuf))
    lst_sb = []
    for _i in range(3):
        lst_sb.append(nl.ndarray((1, KP), dtype=nl.int32, buffer=nl.sbuf))
    q_sb = []
    bnd_sb = []
    s_sb = []
    p_sb = []
    pT_sb = []
    pmax = []
    psumc = []
    nm_step = []
    alpha = []
    lpass = []
    o_bf = []
    o_ps = []
    for _i in range(2):
        q_sb.append(nl.ndarray((D, QB), dtype=qT.dtype, buffer=nl.sbuf))
        bnd_sb.append(nl.ndarray((P, n_pass), dtype=nl.float32, buffer=nl.sbuf))
        s_sb.append(nl.ndarray((P, NK), dtype=nl.float32, buffer=nl.sbuf))
        p_sb.append(nl.ndarray((P, NK), dtype=qT.dtype, buffer=nl.sbuf))
        pT_sb.append(nl.ndarray((P, NB, P), dtype=qT.dtype, buffer=nl.sbuf))
        pmax.append(nl.ndarray((P, nch), dtype=nl.float32, buffer=nl.sbuf))
        psumc.append(nl.ndarray((P, nch), dtype=nl.float32, buffer=nl.sbuf))
        nm_step.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
        alpha.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
        lpass.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
        o_bf.append(nl.ndarray((P, D), dtype=qT.dtype, buffer=nl.sbuf))
        o_ps.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum))
    # per (item parity, sub-tile) online-softmax state, flat index (il % 2) * n_sub + j
    o_acc = []
    l_acc = []
    negm = []
    for _i in range(2 * n_sub):
        o_acc.append(nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf))
        l_acc.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
        negm.append(nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf))
    qk_ps = []
    for _i in range(4):
        qk_ps.append(nl.ndarray((P, KCH), dtype=nl.float32, buffer=nl.psum))

    n_gp = per * n_pass  # passes of this program
    n_steps = n_gp * n_sub
    qk_ctr = 0
    # iteration it: [prefetch K/V passes] [stage A(it)] [stages B, C (it - 1)]
    for it in range(n_steps + 1):
        # ======== K/V prefetch ========
        # iteration 0 loads passes 0 .. nb - 2; iteration it > 0 loads pass gp(it - 1) + nb - 1 when
        # step it - 1 opened pass gp(it - 1). That buffer was last read by C(it - 2), which precedes
        # this point in program order (loading it one iteration earlier would overwrite V that the
        # pending C(it - 1) of the previous pass still reads).
        lp_lo = 0
        lp_hi = 0
        if it == 0:
            lp_hi = min(nb - 1, n_gp)
        else:
            if (it - 1) % n_sub == 0:
                g_next = (it - 1) // n_sub + nb - 1
                if g_next < n_gp:
                    lp_lo = g_next
                    lp_hi = g_next + 1
        for lgp in range(lp_lo, lp_hi):
            l_il = lgp // n_pass
            l_ps = lgp % n_pass
            l_item = prg * per + l_il
            lb = lst_sb[l_il % 3]
            if l_ps == 0:
                nisa.dma_copy(dst=lb, src=lists[nl.ds(l_item, 1), :], dge_mode=nisa.dge_mode.hwdge)
            if masked == 1:
                nisa.dma_copy(
                    dst=km_sb[lgp % nb], src=kmask[l_item, :, l_ps * NK : (l_ps + 1) * NK]
                )
            for t in range(pb):
                slot = l_ps * pb + t
                idx = lb[0:1, slot : slot + 1]
                if packed == 1:
                    kv_dst = kv_sb[lgp % nb][:, t, :]
                    kv_src = kT_blk.ap(
                        pattern=[[KBW, P], [1, KBW]], offset=0, scalar_offset=idx, indirect_dim=0
                    )
                    if v_queue == 2 and t % 3 == 2:
                        # third queue: GpSimd software DGE (scalar-offset copy, not a gather)
                        nisa.dma_copy(dst=kv_dst, src=kv_src, dge_mode=nisa.dge_mode.swdge)
                    elif v_queue >= 1 and t % (v_queue + 1) == 1:
                        nisa.dma_copy(
                            dst=kv_dst,
                            src=kv_src,
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nisa.engine.scalar,
                        )
                    else:
                        nisa.dma_copy(
                            dst=kv_dst,
                            src=kv_src,
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nisa.engine.sync,
                        )
                else:
                    nisa.dma_copy(
                        dst=kT_sb[lgp % nb][:, t * KB : (t + 1) * KB],
                        src=kT_blk.ap(
                            pattern=[[KB, D], [1, KB]], offset=0, scalar_offset=idx, indirect_dim=0
                        ),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync,
                    )
                    if KB < P:
                        p0 = (t % r_small) * KB
                        v_dst = v_sb[lgp % nb][p0 : p0 + KB, t // r_small, :]
                        v_src = v_blk.ap(
                            pattern=[[D, KB], [1, D]], offset=0, scalar_offset=idx, indirect_dim=0
                        )
                    else:
                        v_dst = v_sb[lgp % nb][:, t * s_big : (t + 1) * s_big, :]
                        v_src = v_blk.ap(
                            pattern=[[D, P], [P * D, s_big], [1, D]],
                            offset=0,
                            scalar_offset=idx,
                            indirect_dim=0,
                        )
                    if v_queue == 2 and t % 2 == 1:
                        # third queue: GpSimd software DGE (scalar-offset copy, not a gather)
                        nisa.dma_copy(dst=v_dst, src=v_src, dge_mode=nisa.dge_mode.swdge)
                    elif v_queue >= 1:
                        nisa.dma_copy(
                            dst=v_dst,
                            src=v_src,
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nisa.engine.scalar,
                        )
                    else:
                        nisa.dma_copy(
                            dst=v_dst,
                            src=v_src,
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nisa.engine.sync,
                        )

        # ======== stage A(it): S = q K^T, masked copy + chunk max, softmax bookkeeping ========
        if it < n_steps:
            a_gp = it // n_sub
            a_j = it % n_sub
            a_il = a_gp // n_pass
            a_ps = a_gp % n_pass
            a_b = it % 2
            a_item = prg * per + a_il
            if a_ps == 0:
                if a_j == 0:
                    nisa.dma_copy(dst=q_sb[a_il % 2], src=qT[:, nl.ds(a_item * QB, QB)])
                    nisa.dma_copy(dst=bnd_sb[a_il % 2], src=bounds[a_item])
            qs = q_sb[a_il % 2][:, a_j * P : (a_j + 1) * P]
            for c in range(nch):
                off = c * KCH
                w = min(KCH, NK - off)
                ps_t = qk_ps[qk_ctr % 4]
                qk_ctr = qk_ctr + 1
                if packed == 1:
                    # one matmul per (chunk, list slot) segment: K^T columns are per-slot in SBUF
                    for t in range(off // KB, (off + w - 1) // KB + 1):
                        lo = max(off, t * KB)
                        hi = min(off + w, (t + 1) * KB)
                        nisa.nc_matmul(
                            dst=ps_t[:, lo - off : hi - off],
                            stationary=qs,
                            moving=kv_sb[a_gp % nb][:, t, lo - t * KB : hi - t * KB],
                            accumulate=False,
                        )
                else:
                    nisa.nc_matmul(
                        dst=ps_t[:, 0:w],
                        stationary=qs,
                        moving=kT_sb[a_gp % nb][:, off : off + w],
                        accumulate=False,
                    )
                if masked == 1:
                    nisa.nc_matmul(
                        dst=ps_t[:, 0:w],
                        stationary=qind_sb,
                        moving=km_sb[a_gp % nb][:, off : off + w],
                        accumulate=True,
                    )
                nisa.range_select(
                    dst=s_sb[a_b][:, off : off + w],
                    on_true_tile=ps_t[:, 0:w],
                    comp_op0=nl.greater_equal,
                    comp_op1=nl.less,
                    bound0=zero_b,
                    bound1=bnd_sb[a_il % 2][:, a_ps : a_ps + 1],
                    reduce_op=nl.maximum,
                    reduce_res=pmax[a_b][:, c : c + 1],
                    range_start=off,
                    on_false_value=FP32_MIN,
                )
            st = negm[(a_il % 2) * n_sub + a_j]
            if a_ps == 0:
                nisa.tensor_reduce(
                    dst=st, op=nl.maximum, data=pmax[a_b], axis=1, negate=True, keepdims=True
                )
                nisa.tensor_copy(dst=nm_step[a_b], src=st)
            else:
                nisa.tensor_reduce(
                    dst=nm_step[a_b],
                    op=nl.maximum,
                    data=pmax[a_b],
                    axis=1,
                    negate=True,
                    keepdims=True,
                )
                # -m_new = min(-m_old, -m_pass); alpha = exp(m_old - m_new) = exp(-m_new - (-m_old))
                nisa.tensor_tensor(dst=nm_step[a_b], data1=nm_step[a_b], data2=st, op=nl.minimum)
                nisa.activation(dst=alpha[a_b], op=nl.exp, data=st, bias=nm_step[a_b], scale=-1.0)
                nisa.tensor_copy(dst=st, src=nm_step[a_b])

        if it >= 1:
            si = it - 1
            c_gp = si // n_sub
            c_j = si % n_sub
            c_il = c_gp // n_pass
            c_ps = c_gp % n_pass
            b = si % 2
            # ======== stage B(si): P = exp(S - m) with fused row sums; P^T by DMA transposes ====
            nisa.memset(psumc[b], 0.0)  # activation_reduce ACCUMULATES into reduce_res
            for c in range(nch):
                off = c * KCH
                w = min(KCH, NK - off)
                nisa.activation_reduce(
                    dst=p_sb[b][:, off : off + w],
                    op=nl.exp,
                    data=s_sb[b][:, off : off + w],
                    reduce_op=nl.add,
                    reduce_res=psumc[b][:, c : c + 1],
                    bias=nm_step[b],
                )
                wb = w // P
                nisa.dma_transpose(
                    dst=pT_sb[b].ap([[NB * P, P], [1, 1], [P, wb], [1, P]], offset=(off // P) * P),
                    src=p_sb[b].ap([[NK, P], [1, 1], [P, wb], [1, P]], offset=off),
                )
            nisa.tensor_reduce(dst=lpass[b], op=nl.add, data=psumc[b], axis=1, keepdims=True)
            # ======== stage C(si): O_pass = P^T V; O = O * alpha + O_pass; finalise ========
            op = o_ps[b]
            for m in range(NB):
                # explicit flags: PSUM tiles are reused across steps, the inferred flag accumulates
                if packed == 1:
                    vt = m // s_big
                    vs = m % s_big
                    nisa.nc_matmul(
                        dst=op,
                        stationary=pT_sb[b][:, m, :],
                        moving=kv_sb[c_gp % nb][:, vt, KB + vs * D : KB + (vs + 1) * D],
                        accumulate=(m > 0),
                    )
                else:
                    nisa.nc_matmul(
                        dst=op,
                        stationary=pT_sb[b][:, m, :],
                        moving=v_sb[c_gp % nb][:, m, :],
                        accumulate=(m > 0),
                    )
            oa = o_acc[(c_il % 2) * n_sub + c_j]
            la = l_acc[(c_il % 2) * n_sub + c_j]
            if c_ps == 0:
                nisa.tensor_copy(dst=oa, src=op)
                nisa.tensor_copy(dst=la, src=lpass[b])
            else:
                nisa.scalar_tensor_tensor(
                    dst=oa, data=oa, op0=nl.multiply, operand0=alpha[b], op1=nl.add, operand1=op
                )
                nisa.scalar_tensor_tensor(
                    dst=la,
                    data=la,
                    op0=nl.multiply,
                    operand0=alpha[b],
                    op1=nl.add,
                    operand1=lpass[b],
                )
            if c_ps == n_pass - 1:
                nisa.reciprocal(dst=inv, data=la)
                ob = o_bf[b]
                nisa.tensor_scalar(dst=ob, data=oa, op0=nl.multiply, operand0=inv)
                row0 = (prg * per + c_il) * QB + c_j * P
                nisa.dma_copy(dst=out[nl.ds(row0, P), :], src=ob)
    return out
