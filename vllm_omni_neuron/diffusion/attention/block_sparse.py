# SPDX-License-Identifier: Apache-2.0
"""Shared block-sparse attention for DiTs on Trainium2: one plan, three executors.

A *plan* says, per (head, query block), which key blocks that query block attends to -- the
per-(head, query-block) index lists produced by a top-k selection (SLA: 128-row query blocks, top
15% of 64/128-row key blocks by pooled Q.K; SSTA: 384-token tiles, 3x3x3 window + top-k, text tiles
always kept). The plan is static in shape (every list has ``KP`` entries, short lists padded) so
every executor compiles to fixed shapes. Keys are given in the sequence's own packed order; blocks
are contiguous row ranges of it.

Three executors over the same plan, all returning ``(H, Lq, D)`` and all exact against
:func:`reference_attention` (fp32 masked softmax) up to bf16 rounding:

* **L1** :func:`attend_dense_bias` -- the dense kernel (nkilib ``attention_cte``) with an additive
  block bias (0 / NEG_BIAS) expanded from the plan. No speedup; this is the quality bridge a model
  uses for its SSIM gate before the sparse kernel is wired in.
* **L2** :func:`attend_gather_dense` -- gather each query block's listed K/V blocks into a
  contiguous buffer and run ``attention_cte`` on it, batched over query blocks, with
  ``bound_max`` masking the padded tail. Work scales with the kept fraction; the gather is a torch
  ``index_select`` over whole blocks.
* **L3** :func:`attend_index_list` -- :mod:`vllm_omni_neuron.kernels.bsa_index_list`: the key loop
  walks the index list and fetches each K/V block by indirect DMA inside the kernel (online softmax
  across passes, validity column for pads).

Off the device every executor runs the same math in torch (the plan logic is identical), so a
model can validate its selection on CPU with the executor it will use. Key validity: ``key_valid``
marks rows that count in the softmax when their block is selected -- SLA's trailing pad rows are
``False`` (dropped exactly); SSTA's zero-padded tile rows are ``True`` (they contribute
``exp(0 - m)``, as upstream). L2 supports invalid rows only as the trailing rows of ONE block per
head (the SLA tail); L1 and L3 support any pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import nki
import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.attention.attention_cte import attention_cte

NEG_BIAS = (
    -30720.0
)  # finite "masked" bias, exact in bf16 (-240 * 128); -inf / finfo.min risk NaN in fused softmax
LNC = 2


# ================================================================================================ plan
@dataclass
class BlockSparsePlan:
    """Per-(head, query block) key-block index lists with fixed width ``KP``.

    ``lists`` ``[H, nQ, KP]`` int32: key-block ids, real entries first, then ``-1`` pads;
    ``counts`` ``[H, nQ]`` int32: real entries per list; ``q_block`` / ``k_block``: rows per block;
    ``n_k_blocks``: blocks in the key sequence (``ceil(Lk / k_block)``); ``key_valid`` ``[Lk]`` bool.
    """

    lists: torch.Tensor
    counts: torch.Tensor
    q_block: int
    k_block: int
    n_k_blocks: int
    key_valid: torch.Tensor

    @property
    def heads(self) -> int:
        return int(self.lists.shape[0])

    @property
    def n_q_blocks(self) -> int:
        return int(self.lists.shape[1])

    @property
    def kp(self) -> int:
        return int(self.lists.shape[2])

    @property
    def kept_fraction(self) -> float:
        return float(self.counts.float().mean() / self.n_k_blocks)

    def block_mask(self) -> torch.Tensor:
        """``[H, nQ, nK]`` bool."""
        h, nq, kp = self.lists.shape
        mask = torch.zeros(h, nq, self.n_k_blocks, dtype=torch.bool)
        real = self.lists >= 0
        hi, qi, _ = torch.nonzero(real, as_tuple=True)
        mask[hi, qi, self.lists[real].long()] = True
        return mask

    def token_mask(self, lq: int, lk: int) -> torch.Tensor:
        """``[H, Lq, Lk]`` bool (small shapes only: the reference)."""
        bm = self.block_mask()
        tok = bm.repeat_interleave(self.q_block, dim=1)[:, :lq]
        tok = tok.repeat_interleave(self.k_block, dim=2)[:, :, :lk]
        return tok & self.key_valid[:lk].view(1, 1, lk)


def plan_from_block_mask(
    mask: torch.Tensor,
    q_block: int,
    k_block: int,
    key_valid: torch.Tensor | None = None,
    lk: int | None = None,
    pad_to: int = 1,
    kp: int | None = None,
) -> BlockSparsePlan:
    """``mask`` ``[H, nQ, nK]`` bool -> plan. Lists are ascending block ids, padded with -1 to
    ``kp`` (default: the max count rounded up to a multiple of ``pad_to``)."""
    assert mask.dim() == 3 and mask.dtype == torch.bool
    h, nq, nk = mask.shape
    counts = mask.sum(-1).to(torch.int32)
    kmax = int(counts.max())
    width = kp if kp is not None else -(-kmax // pad_to) * pad_to
    assert width >= kmax, (width, kmax)
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)[..., :width]
    lists = torch.where(
        torch.arange(width) < counts[..., None], order, torch.full_like(order, -1)
    ).to(torch.int32)
    lk = lk if lk is not None else nk * k_block
    if key_valid is None:
        key_valid = torch.ones(lk, dtype=torch.bool)
    assert key_valid.shape == (lk,)
    return BlockSparsePlan(lists, counts, q_block, k_block, nk, key_valid)


def plan_from_lists(
    lists: torch.Tensor,
    counts: torch.Tensor,
    q_block: int,
    k_block: int,
    n_k_blocks: int,
    key_valid: torch.Tensor | None = None,
    lk: int | None = None,
) -> BlockSparsePlan:
    """From a kernel-style ``[H, nQ, KP]`` list (``-1`` pads) and ``[H, nQ]`` counts."""
    lists = lists.to(torch.int32)
    lk = lk if lk is not None else n_k_blocks * k_block
    if key_valid is None:
        key_valid = torch.ones(lk, dtype=torch.bool)
    return BlockSparsePlan(lists, counts.to(torch.int32), q_block, k_block, n_k_blocks, key_valid)


def sla_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    q_block: int = 128,
    k_block: int = 64,
    keep_ratio: float = 0.15,
    smooth_k: bool = True,
    lq_real: int | None = None,
    lk_real: int | None = None,
) -> torch.Tensor:
    """SLA / SpargeAttn-style selection: ``mean_pool(q) . mean_pool(k - mean(k))`` per block pair,
    ``top-k`` per query block with ``k = max(1, min(nK, int(keep_ratio * nK)))``. ``q`` / ``k``
    ``[H, L, D]`` (padded to block multiples); means divide by the real rows of partial blocks.
    Returns ``[H, nQ, nK]`` bool."""
    h, lq, d = q.shape
    lk = k.shape[1]
    lq_real = lq if lq_real is None else lq_real
    lk_real = lk if lk_real is None else lk_real
    nq, nk = lq // q_block, lk // k_block
    assert lq % q_block == 0 and lk % k_block == 0
    kf = k.float()
    if smooth_k:
        kf = kf - kf[:, :lk_real].mean(dim=1, keepdim=True)
    q_valid = (torch.arange(lq) < lq_real).float().view(1, nq, q_block, 1)
    k_valid = (torch.arange(lk) < lk_real).float().view(1, nk, k_block, 1)
    qm = (q.float().view(h, nq, q_block, d) * q_valid).sum(2) / q_valid.sum(2).clamp_min(1)
    km = (kf.view(h, nk, k_block, d) * k_valid).sum(2) / k_valid.sum(2).clamp_min(1)
    score = qm @ km.transpose(1, 2)  # [H, nQ, nK], unscaled
    topk = max(1, min(nk, int(keep_ratio * nk)))
    idx = torch.topk(score, topk, dim=-1).indices
    mask = torch.zeros(h, nq, nk, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask


# ========================================================================================= reference
def reference_attention(q, k, v, plan: BlockSparsePlan, scale: float | None = None) -> torch.Tensor:
    """fp32 masked softmax over the plan's token mask. ``q`` ``[H, Lq, D]``, ``k``/``v`` ``[H, Lk, D]``.
    A query with no valid key returns zeros. Output in ``q.dtype``."""
    h, lq, d = q.shape
    lk = k.shape[1]
    scale = d**-0.5 if scale is None else scale
    tok = plan.token_mask(lq, lk)
    s = (q.float() @ k.float().transpose(1, 2)) * scale
    s = s.masked_fill(~tok, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    return (p @ v.float()).to(q.dtype)


def _kernel_ok(x: torch.Tensor) -> bool:
    if x.device.type == "cpu":
        return False
    try:
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

        return bool(can_run_kernel(x))
    except Exception:  # pragma: no cover - environment without the plugin's NKI gate
        return False


# ================================================================================ L1: dense + bias
def block_bias(plan: BlockSparsePlan, lq: int, lk: int, dtype=torch.bfloat16, device=None):
    """``[H, Lq, Lk]`` additive bias (0 kept / NEG_BIAS dropped) from the plan, built with 0/1
    expansion matmuls so it lowers as two matmuls when compiled for the device."""
    device = device or plan.lists.device
    bm = plan.block_mask().to(device)
    h, nq, nk = bm.shape
    eq = F.one_hot(torch.arange(lq, device=device) // plan.q_block, nq).to(dtype)  # [Lq, nQ]
    ek = F.one_hot(torch.arange(lk, device=device) // plan.k_block, nk).to(dtype)  # [Lk, nK]
    drop = (~bm).to(dtype)  # 1 where dropped
    bias = (eq @ drop) @ ek.t()  # [H, Lq, Lk] in {0, 1}
    invalid = (~plan.key_valid[:lk].to(device)).to(dtype).view(1, 1, lk)
    bias = torch.clamp(bias + invalid, max=1.0)
    return bias * NEG_BIAS


def _bias_attention_torch(q, k, v, bias, scale):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale + bias.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


def attend_dense_bias(
    q, k, v, plan: BlockSparsePlan, scale: float | None = None, q_chunk: int | None = None
):
    """L1. ``q`` ``[H, Lq, D]`` (``Lq % 128 == 0``), ``k``/``v`` ``[H, Lk, D]``. On the device runs
    ``attention_cte`` per head pair over query chunks of ``q_chunk`` rows (the bias is ``[chunk, Lk]``
    bf16, so pick ``q_chunk`` to bound HBM); elsewhere torch with the same bias.

    DEVICE STATUS (probe runs 1-2 at 7 x 4736 x 37888): the result is WRONG (rel 0.70 vs the masked
    reference) although the same call is exact on the NKI simulator at Lk <= 2048. Suspected: the
    kernel's dense ``position_bias`` under its flash-attention section path (Lk > 10240). Until that
    is resolved use :func:`attend_gather_dense` (exact on device) as the quality bridge; the probe's
    ``l1_zero_bias`` / ``l1_small_lk`` variants narrow the cause.
    """
    h, lq, d = q.shape
    lk = k.shape[1]
    scale = d**-0.5 if scale is None else scale
    q_chunk = q_chunk or lq
    assert lq % q_chunk == 0 and q_chunk % 128 == 0, (lq, q_chunk)
    bias = block_bias(plan, lq, lk, dtype=torch.bfloat16, device=q.device)
    if not _kernel_ok(q):
        return _bias_attention_torch(q, k, v, bias, scale)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    launch = wrap_nki(_dense_bias_kernel)[LNC]
    # Heads go in PAIRS: with an odd batch and seqlen_q >= 1024 attention_cte shards the LNC2 pair on
    # the SEQUENCE and the dense position_bias is read at shard-relative query offsets. An odd head
    # count repeats its last head. Every operand is cut by a COMPILED graph (an eager slice +
    # .contiguous() of a device tensor is refused by the Lite executor).
    outs = []
    for h0 in range(0, h, 2):
        hs = (h0, min(h0 + 1, h - 1))
        kk = _device_pair(k, hs)
        vv = _device_pair(v, hs)
        rows = []
        for c0 in range(0, lq, q_chunk):
            rows.append(
                launch(
                    q=_device_pair_chunk(q, hs, c0, q_chunk, scale),
                    k=kk,
                    v=vv,
                    bias=_device_pair_chunk(bias, hs, c0, q_chunk, None),
                )
            )
        pair = torch.cat(rows, dim=1) if len(rows) > 1 else rows[0]
        outs.append(pair if h0 + 1 < h else pair[:1])
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


_CUT_GRAPHS: dict = {}


def _compiled_cut(name, key, fn):
    """Per-(name, shape-key) compiled cut with its own code object (Dynamo's recompile limit is per
    code object; see the VAE gather fix) -- the device-safe way to slice a device tensor."""
    import types

    from vllm_neuron.envs import get_compile_backend_name

    g = _CUT_GRAPHS.get((name, key))
    if g is None:
        code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_{len(_CUT_GRAPHS)}")
        fresh = types.FunctionType(
            code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
        )
        g = torch.compile(
            fresh,
            backend=get_compile_backend_name(),
            fullgraph=True,
            dynamic=False,
            options={"model_name": f"bsa_{name}"},
        )
        _CUT_GRAPHS[(name, key)] = g
    return g


def _device_pair(x, hs):
    i0, i1 = hs

    def fn(t):
        return torch.stack((t[i0], t[i1]), dim=0)

    return _compiled_cut("pair", (tuple(x.shape), str(x.dtype), hs), fn)(x)


def _device_pair_chunk(x, hs, c0, n, scale):
    i0, i1 = hs

    def fn(t):
        pair = torch.stack((t[i0, c0 : c0 + n], t[i1, c0 : c0 + n]), dim=0)
        if scale is not None:
            pair = (pair.float() * scale).to(torch.bfloat16)
        return pair.clone(memory_format=torch.contiguous_format)

    return _compiled_cut("pair_chunk", (tuple(x.shape), str(x.dtype), hs, c0, n, scale), fn)(x)


# ============================================================================ L2: gather + dense
def _lists_gather_order(plan: BlockSparsePlan):
    """Lists re-ordered for the gather path: full blocks first, the (single) partial block last,
    pads (-1 -> dummy id n_k_blocks) after it. Returns (lists_g, bound_max [H, nQ] in keys)."""
    valid = plan.key_valid
    nk, bs = plan.n_k_blocks, plan.k_block
    lk = valid.shape[0]
    rows = torch.zeros(nk, dtype=torch.int64)
    for b in range(nk):
        seg = valid[b * bs : min((b + 1) * bs, lk)]
        n_valid = int(seg.sum())
        assert bool(seg[:n_valid].all()), (
            "L2: invalid keys must be the trailing rows of their block"
        )
        rows[b] = n_valid
    partial = (rows < bs).nonzero().flatten()
    assert partial.numel() <= 1, "L2 supports at most one partial key block"
    lists = plan.lists.long()
    real = lists >= 0
    is_partial = real & (lists == (partial.item() if partial.numel() else -2))
    # sort key: full real blocks 0, partial real 1, pads 2 (stable keeps ascending ids)
    key = torch.where(real, torch.where(is_partial, 1, 0), 2)
    order = torch.argsort(key.to(torch.int8), dim=-1, stable=True)
    lists_g = torch.gather(lists, -1, order)
    lists_g = torch.where(lists_g < 0, torch.full_like(lists_g, nk), lists_g)
    bound = torch.gather(torch.cat([rows, torch.zeros(1, dtype=torch.int64)]), 0, lists_g.flatten())
    bound_max = bound.view_as(lists_g).sum(-1).to(torch.int32)
    return lists_g.to(torch.int32), bound_max


def blockify_kv(x: torch.Tensor, k_block: int, with_dummy: bool = True) -> torch.Tensor:
    """``[H, Lk, D]`` -> ``[H, nK (+1), BLK, D]``: pad the tail block with zeros, append a zero dummy."""
    h, lk, d = x.shape
    nk = -(-lk // k_block)
    pad = nk * k_block - lk + (k_block if with_dummy else 0)
    xb = F.pad(x, (0, 0, 0, pad)) if pad else x
    return xb.reshape(h, nk + (1 if with_dummy else 0), k_block, d)


def gather_blocks(x_blocks: torch.Tensor, lists_g: torch.Tensor) -> torch.Tensor:
    """``x_blocks`` ``[H, nK+1, BLK, D]``, ``lists_g`` ``[H, nQ, KP]`` -> ``[H, nQ, KP*BLK, D]``."""
    h, nb1, blk, d = x_blocks.shape
    _, nq, kp = lists_g.shape
    flat = x_blocks.reshape(h, nb1, blk * d)
    idx = lists_g.long().reshape(h, nq * kp, 1).expand(h, nq * kp, blk * d)
    g = torch.gather(flat, 1, idx)
    return g.reshape(h, nq, kp * blk, d)


def attend_gather_dense(q, k, v, plan: BlockSparsePlan, scale: float | None = None):
    """L2. Gather listed K/V blocks per query block, dense ``attention_cte`` batched over
    ``(H * nQ)`` items with ``bound_max`` masking the dummy/pad tail."""
    h, lq, d = q.shape
    scale = d**-0.5 if scale is None else scale
    nq, bs = plan.n_q_blocks, plan.q_block
    assert lq == nq * bs, (lq, nq, bs)
    lists_g, bound_max = _lists_gather_order(plan)
    lists_g, bound_max = lists_g.to(q.device), bound_max.to(q.device)
    kb = blockify_kv(k, plan.k_block)
    vb = blockify_kv(v, plan.k_block)
    kg = gather_blocks(kb, lists_g).reshape(h * nq, plan.kp * plan.k_block, d)
    vg = gather_blocks(vb, lists_g).reshape(h * nq, plan.kp * plan.k_block, d)
    qs = (q.float() * scale).to(q.dtype).reshape(h * nq, bs, d)
    bmax = bound_max.reshape(h * nq, 1, 1).expand(h * nq, bs, 1).to(torch.int32).contiguous()
    if not _kernel_ok(q):
        pos = torch.arange(kg.shape[1], device=q.device).view(1, 1, -1)
        mask = pos < bmax
        s = (qs.float() @ kg.float().transpose(1, 2)).masked_fill(~mask, float("-inf"))
        out = (torch.softmax(s, -1) @ vg.float()).to(q.dtype)
        return out.reshape(h, lq, d)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    bmin = torch.zeros_like(bmax)
    out = wrap_nki(_dense_bounded_kernel)[LNC](
        q=qs.contiguous(), k=kg.contiguous(), v=vg.contiguous(), bound_min=bmin, bound_max=bmax
    )
    return out.reshape(h, lq, d)


# ============================================================================= L3: index-list kernel
KERNEL_BLOCK = 128  # the L3 kernel's DMA block (rows); larger plan blocks are split into sub-blocks
KERNEL_PB = 16  # blocks per softmax pass: the indirect DMA transpose needs a multiple of 16 blocks


def index_list_inputs(plan: BlockSparsePlan, k, v, pb: int = KERNEL_PB):
    """Kernel operands for :func:`attend_index_list`. The kernel always walks 128-row blocks: a plan
    with ``k_block = r * 128`` has each list entry ``t`` expanded to sub-blocks ``r t .. r t + r - 1``
    (a 64-row plan block is kept as its own kernel block of 64 rows: ``BLK`` then equals 64); a plan
    query block wider than 128 rows repeats its list for each 128-row kernel tile.
    Returns ``k3`` ``[H, nK' + 1, BLK, D]`` (zero dummy, invalid rows zeroed), ``v_ext``
    ``[H, (nK' + 1) * BLK, D + 32]`` (validity column at ``D``, invalid rows zeroed), ``blocks``
    ``[H, n_qt, KP', 1]`` int32 (pads -> dummy id, ``KP'`` a multiple of ``pb``), ``rowstart``
    ``[H, n_qt, KP' / pb, 128, 1]``."""
    from vllm_omni_neuron.kernels.bsa_index_list import v_ext_width

    h, lk, d = k.shape
    bs, nk = plan.k_block, plan.n_k_blocks
    assert pb % 16 == 0 and 128 % pb == 0, pb
    r = max(1, bs // KERNEL_BLOCK)
    blk = bs // r
    assert blk * r == bs and (pb * blk) % 512 == 0, (bs, blk, pb)
    nk_sub = nk * r
    sub = plan.lists.long().unsqueeze(-1) * r + torch.arange(r)  # [H, nQ, KP, r]
    sub = torch.where(plan.lists.long().unsqueeze(-1) < 0, torch.full_like(sub, -1), sub)
    lists = sub.reshape(h, plan.n_q_blocks, plan.kp * r)
    # the kernel's query tile is 128 rows: a wider plan query block repeats its list per tile
    rq = plan.q_block // 128
    assert rq * 128 == plan.q_block, plan.q_block
    lists = lists.repeat_interleave(rq, dim=1)  # [H, nQ * rq, KP * r]
    n_qt = lists.shape[1]
    kp = -(-lists.shape[-1] // pb) * pb
    lists = F.pad(lists, (0, kp - lists.shape[-1]), value=-1)
    lists = torch.where(lists < 0, torch.full_like(lists, nk_sub), lists)  # pads -> dummy block
    valid = plan.key_valid[:lk].to(k.device)
    kz = k * valid.view(1, lk, 1).to(k.dtype)
    k3 = blockify_kv(kz, blk)
    assert k3.shape[1] == nk_sub + 1, (k3.shape, nk_sub)
    dv = v_ext_width(d)
    v_ext = torch.zeros(h, (nk_sub + 1) * blk, dv, dtype=v.dtype, device=v.device)
    v_ext[:, :lk, :d] = v * valid.view(1, lk, 1).to(v.dtype)  # invalid rows: zero V AND validity
    v_ext[:, :lk, d] = valid.to(v.dtype)
    p = torch.arange(128)
    n_pass = kp // pb
    lp = lists.view(h, n_qt, n_pass, pb)
    rowstart = lp[..., p % pb] * blk + (p // pb)  # [H, n_qt, n_pass, 128]
    return (
        k3,
        v_ext,
        lists.to(torch.int32).unsqueeze(-1).to(k.device),
        rowstart.to(torch.int32).unsqueeze(-1).to(k.device),
    )


def attend_index_list(
    q, k, v, plan: BlockSparsePlan, scale: float | None = None, pb: int = KERNEL_PB
):
    """L3. ``pb`` = kernel blocks per softmax pass (a multiple of 16 dividing 128; 16 = 2048 keys
    per pass at 128-row blocks)."""
    h, lq, d = q.shape
    scale = d**-0.5 if scale is None else scale
    assert d == 128 and lq % 128 == 0, (d, lq)
    k3, v_ext, blocks, rowstart = index_list_inputs(plan, k, v, pb)
    if not _kernel_ok(q):
        return reference_attention(q, k, v, plan, scale)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.bsa_index_list import bsa_index_list

    qT = (q.float() * scale).to(torch.bfloat16).transpose(1, 2).contiguous()  # [H, D, Lq]
    n_qt = lq // 128
    if n_qt % 2:  # the LNC2 SPMD pair splits the query tiles evenly: pad one tile (repeat its list)
        qT = F.pad(qT, (0, 128))
        blocks = torch.cat([blocks, blocks[:, -1:]], dim=1).contiguous()
        rowstart = torch.cat([rowstart, rowstart[:, -1:]], dim=1).contiguous()
    out = wrap_nki(bsa_index_list)[LNC](
        qT=qT,
        k3=k3.contiguous(),
        v_ext=v_ext.contiguous(),
        blocks_i32=blocks,
        rowstart_i32=rowstart,
        pb=pb,
    )
    return out[:, :lq] if n_qt % 2 else out


# ======================================================================= L4: streaming kernel
STREAM_KEYS_MIN, STREAM_KEYS_MAX = 1024, 4096  # keys per softmax pass the L4 pass picker aims for


def stream_pass_blocks(n_entries: int, k_block: int) -> int:
    """Blocks per softmax pass for :mod:`bsa_stream`: the ``pb`` whose pass width ``pb * k_block``
    is a multiple of 128 within ``[STREAM_KEYS_MIN, STREAM_KEYS_MAX]`` (or the closest feasible)
    that pads the list least, ties to the wider pass (fewer online-softmax rescales)."""
    cands = []
    for pb in range(1, n_entries + 1):
        nk = pb * k_block
        if nk % 128 or nk > max(STREAM_KEYS_MAX, k_block):
            continue
        n_pass = -(-n_entries // pb)
        waste = n_pass * pb - n_entries
        short = max(0, min(STREAM_KEYS_MIN, n_entries * k_block) - nk)
        cands.append((short, waste, -nk, pb))
    assert cands, (n_entries, k_block)
    return min(cands)[3]


@dataclass
class StreamPlan:
    """Kernel-side form of a :class:`BlockSparsePlan` for L4 (:func:`attend_stream`).

    Built ONCE per plan on the host (:func:`stream_plan`); only ``lists`` / ``bounds`` change with
    the selection, every other field is static (part of the compiled graph's shape). Items are
    (head, query block) pairs in head-major order, padded to an even count for the LNC2 pair by
    repeating the last item. ``lists`` ``(n_items, KP)`` int32 hold GLOBAL block ids
    (``head * n_k_blocks + block``), full blocks first, the single partial block last, pads
    pointing at the item's first block; ``bounds`` ``(n_items, 128, n_pass)`` fp32 = valid keys
    counted from the start of each pass (keys at or beyond it are masked)."""

    lists: torch.Tensor
    bounds: torch.Tensor
    heads: int
    n_q_blocks: int
    q_block: int
    k_block: int
    n_k_blocks: int
    pb: int
    packed: bool
    n_real_items: int

    @property
    def n_items(self) -> int:
        return int(self.lists.shape[0])

    def to(self, device) -> "StreamPlan":
        return self.with_lists(self.lists.to(device), self.bounds.to(device))

    def with_lists(self, lists: torch.Tensor, bounds: torch.Tensor) -> "StreamPlan":
        """Same static geometry, new selection (from :func:`stream_lists`, e.g. per layer)."""
        assert lists.shape == self.lists.shape and bounds.shape == self.bounds.shape
        return StreamPlan(
            lists, bounds, self.heads, self.n_q_blocks,
            self.q_block, self.k_block, self.n_k_blocks, self.pb, self.packed, self.n_real_items,
        )  # fmt: skip


def stream_plan(
    plan: BlockSparsePlan, pb: int | None = None, packed: bool | None = None
) -> StreamPlan:
    """Host-side (CPU, integer) part of the L4 operands; see :class:`StreamPlan`. ``packed``
    defaults to ``k_block % 128 == 0`` (one DMA per listed block)."""
    h, nq, kb, nk = plan.heads, plan.n_q_blocks, plan.k_block, plan.n_k_blocks
    assert plan.q_block % 128 == 0, plan.q_block
    assert kb in (32, 64) or kb % 128 == 0, kb
    packed = (kb % 128 == 0) if packed is None else bool(packed)
    assert not packed or kb % 128 == 0, kb
    lists_g, bound_max = _lists_gather_order(plan)  # pads -> nk (dummy id)
    lists_g = lists_g.long()
    lists_g = torch.where(lists_g >= nk, lists_g[..., :1].expand_as(lists_g), lists_g)
    pb = pb or stream_pass_blocks(plan.kp, kb)
    kp = -(-plan.kp // pb) * pb
    if kp > plan.kp:
        lists_g = torch.cat([lists_g, lists_g[..., :1].expand(h, nq, kp - plan.kp)], dim=-1)
    n_pass, nkeys = kp // pb, pb * kb
    lists = (lists_g + (torch.arange(h) * nk).view(h, 1, 1)).reshape(h * nq, kp).to(torch.int32)
    bnd = bound_max.reshape(h * nq, 1).float() - torch.arange(n_pass).float().view(1, -1) * nkeys
    n_real = h * nq
    if n_real % 2:
        lists = torch.cat([lists, lists[-1:]], 0)
        bnd = torch.cat([bnd, bnd[-1:]], 0)
    bounds = bnd.view(-1, 1, n_pass).expand(-1, 128, n_pass).contiguous()
    return StreamPlan(lists.contiguous(), bounds, h, nq, plan.q_block, kb, nk, pb, packed, n_real)


def stream_q(sp: StreamPlan, q: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """``q`` ``[H, Lq, 128]`` -> ``qT`` ``(128, n_items * q_block)`` bf16, pre-scaled. Pure tensor
    ops (static shapes): runs inside the model's compiled graph."""
    h, lq, d = q.shape
    scale = d**-0.5 if scale is None else scale
    qi = (q.float() * scale).to(torch.bfloat16).reshape(h * sp.n_q_blocks, sp.q_block, d)
    if sp.n_items > sp.n_real_items:
        qi = torch.cat([qi, qi[-1:]], 0)
    return qi.reshape(-1, d).t().contiguous()


def stream_kv(sp: StreamPlan, k, v, key_valid: torch.Tensor | None = None):
    """``k``/``v`` ``[H, Lk, 128]`` -> ``(kT_blk, v_blk)`` in the kernel's block layouts (packed:
    ``kT_blk`` = ``[K^T | V]`` rows, ``v_blk`` a 1-element placeholder). ``key_valid`` (``[Lk]``
    bool, optional): rows to ZERO before blocking (SLA's tail pad); omit it when invalid rows are
    already zero or must stay in the softmax (SSTA). Pure tensor ops: runs in the compiled graph."""
    h, lk, d = k.shape
    assert d == 128, d
    if key_valid is not None:
        m = key_valid[:lk].to(device=k.device, dtype=k.dtype).view(1, lk, 1)
        k, v = k * m, v * m
    kbk = blockify_kv(k.to(torch.bfloat16), sp.k_block, with_dummy=False)
    vbk = blockify_kv(v.to(torch.bfloat16), sp.k_block, with_dummy=False)
    assert kbk.shape[1] == sp.n_k_blocks, (kbk.shape, sp.n_k_blocks)
    kT_blk = kbk.transpose(2, 3).reshape(h * sp.n_k_blocks, d, sp.k_block).contiguous()
    v_blk = vbk.reshape(h * sp.n_k_blocks, sp.k_block, d).contiguous()
    if sp.packed:
        return pack_kv_blocks(kT_blk, v_blk), torch.zeros(
            1, 1, 1, dtype=torch.bfloat16, device=k.device
        )
    return kT_blk, v_blk


def stream_lists(
    ids: torch.Tensor, counts: torch.Tensor, k_block: int, n_k_blocks: int, pb: int,
    n_tail_invalid: int = 0,
):  # fmt: skip
    """Device-side (compile-friendly, static shapes) builder of the L4 ``lists`` / ``bounds`` from a
    per-layer selection, for models that select on the device every layer.

    ``ids`` ``[H, nQ, K]`` int: selected key-block ids, ASCENDING, ``-1`` pads after the real
    entries (SLA's sorted top-k has no pads; SSTA's ``kept_tile_lists`` format); ``counts``
    ``[H, nQ]``. ``n_tail_invalid``: invalid (dropped) rows at the end of the LAST key block (SLA's
    sequence tail; 0 for SSTA, whose zero pad rows stay in the softmax). Ascending order puts that
    block last in any list that holds it, which is what the kernel's bound masking needs. ``K`` is
    padded up to a multiple of ``pb``. Returns ``(lists, bounds)`` as :class:`StreamPlan` holds
    them (even item count)."""
    h, nq, kk = ids.shape
    kp = -(-kk // pb) * pb
    ids = ids.long()
    if kp > kk:
        ids = torch.cat(
            [ids, torch.full((h, nq, kp - kk), -1, dtype=ids.dtype, device=ids.device)], -1
        )
    first = ids[..., :1].expand_as(ids)
    lists = torch.where(ids < 0, first, ids)
    lists = lists + (torch.arange(h, device=ids.device) * n_k_blocks).view(h, 1, 1)
    n_valid = counts.long() * k_block
    if n_tail_invalid:
        has_tail = (ids == n_k_blocks - 1).any(-1)
        n_valid = n_valid - has_tail.long() * n_tail_invalid
    n_pass, nkeys = kp // pb, pb * k_block
    off = torch.arange(n_pass, device=ids.device).view(1, n_pass) * nkeys
    bnd = n_valid.reshape(h * nq, 1).float() - off.float()
    lists = lists.reshape(h * nq, kp).to(torch.int32)
    if (h * nq) % 2:
        lists = torch.cat([lists, lists[-1:]], 0)
        bnd = torch.cat([bnd, bnd[-1:]], 0)
    return lists.contiguous(), bnd.view(-1, 1, n_pass).expand(-1, 128, n_pass).contiguous()


MASK_NEG = -30720.0  # exact in bf16; exp(MASK_NEG + s - m) underflows to 0 for any real row max m


def pair_union_plan(sel64: torch.Tensor, k_block: int, key_valid=None) -> BlockSparsePlan:
    """VSA-style selection on 64-row query tiles ``sel64`` ``[H, 2 * nP, nK]`` bool -> the plan of
    128-row items (pairs of adjacent tiles) attending the UNION of both tiles' lists; pair it with
    :func:`stream_half_masks` so each half keeps only its own blocks."""
    h, nq64, nk = sel64.shape
    assert nq64 % 2 == 0, nq64
    union = sel64[:, 0::2] | sel64[:, 1::2]
    return plan_from_block_mask(union, 128, k_block, key_valid=key_valid)


def stream_half_masks(sp: StreamPlan, sel64: torch.Tensor):
    """Per-half key masks for a :func:`pair_union_plan` StreamPlan: returns ``(kmask, qind)`` for the
    kernel's ``masked=1`` mode. ``kmask`` ``(n_items, 2, KP * KB)`` bf16 in the StreamPlan's list
    order (pads masked too; their keys are also beyond the bound), ``qind`` ``(2, 128)``."""
    h, nq64, nk = sel64.shape
    n_real, kp = sp.n_real_items, sp.lists.shape[1]
    ids = sp.lists[:n_real].long().view(h, nq64 // 2, kp) - (torch.arange(h) * nk).view(h, 1, 1)
    halves = sel64.view(h, nq64 // 2, 2, nk)
    member = torch.gather(halves, 3, ids.unsqueeze(2).expand(h, nq64 // 2, 2, kp))  # [H, nP, 2, KP]
    m = torch.where(member, 0.0, MASK_NEG).to(torch.bfloat16).reshape(n_real, 2, kp)
    if sp.n_items > n_real:
        m = torch.cat([m, m[-1:]], 0)
    kmask = m.repeat_interleave(sp.k_block, dim=2).contiguous()
    qind = torch.zeros(2, 128, dtype=torch.bfloat16)
    qind[0, :64] = 1
    qind[1, 64:] = 1
    return kmask, qind


def default_v_queue(sp: StreamPlan) -> int:
    """Measured (probes v2/v3, one logical core): 128-row packed blocks are DMA-throughput bound and
    gain from a third queue (Turbo-SLA 5.04 -> 4.09 ms); 384-row tiles are compute-limited at two."""
    return 2 if sp.k_block < 384 else 1


def superblock_plan(plan: BlockSparsePlan, factor: int = 2) -> BlockSparsePlan:
    """Coarsen a plan's key blocks by ``factor`` (e.g. 128 -> 256 rows): a super-block is listed when
    ANY of its sub-blocks is selected. Fewer, larger K/V DMAs; pair the result with
    :func:`superblock_masks` so the unselected sub-blocks are dropped exactly."""
    bm = plan.block_mask()
    h, nq, nk = bm.shape
    nks = -(-nk // factor)
    pad = nks * factor - nk
    if pad:
        bm = torch.cat([bm, torch.zeros(h, nq, pad, dtype=torch.bool)], -1)
    sup = bm.view(h, nq, nks, factor).any(-1)
    lk = nks * factor * plan.k_block
    kv = torch.zeros(lk, dtype=torch.bool)
    kv[: plan.key_valid.shape[0]] = plan.key_valid
    return plan_from_block_mask(sup, plan.q_block, plan.k_block * factor, key_valid=kv)


def superblock_masks(sp: StreamPlan, fine: BlockSparsePlan):
    """``(kmask, qind)`` for a :func:`superblock_plan` StreamPlan (kernel ``masked=1``, one row
    group): 0 on keys of selected sub-blocks, MASK_NEG on the rest, in list order."""
    bm = fine.block_mask()  # [H, nQ, nK] at the fine block size
    h, nq, nk = bm.shape
    factor = sp.k_block // fine.k_block
    nks = sp.n_k_blocks
    if nks * factor > nk:
        bm = torch.cat([bm, torch.zeros(h, nq, nks * factor - nk, dtype=torch.bool)], -1)
    n_real, kp = sp.n_real_items, sp.lists.shape[1]
    ids = sp.lists[:n_real].long().view(h, nq, kp) - (torch.arange(h) * nks).view(h, 1, 1)
    sub = ids.unsqueeze(-1) * factor + torch.arange(factor)  # [H, nQ, KP, factor]
    member = torch.gather(bm, 2, sub.reshape(h, nq, kp * factor)).view(h, nq, kp, factor)
    m = torch.where(member, 0.0, MASK_NEG).to(torch.bfloat16)
    m = m.repeat_interleave(fine.k_block, dim=-1).reshape(n_real, 1, kp * sp.k_block)
    if sp.n_items > n_real:
        m = torch.cat([m, m[-1:]], 0)
    return m.contiguous(), torch.ones(1, 128, dtype=torch.bfloat16)


def stream_kernel_kwargs(sp: StreamPlan, v_queue: int | None = None, nbuf: int = 3) -> dict:
    """Static (compile-time) kernel arguments (``v_queue`` default: :func:`default_v_queue`)."""
    v_queue = default_v_queue(sp) if v_queue is None else v_queue
    return dict(
        q_block=sp.q_block, pb=sp.pb, nbuf=nbuf, v_queue=v_queue, packed=int(sp.packed),
        k_block=sp.k_block,
    )  # fmt: skip


def stream_inputs(
    plan: BlockSparsePlan, q, k, v, scale: float | None = None, pb: int | None = None,
    packed: bool = False,
):  # fmt: skip
    """All kernel operands at once (tests / probes): ``dict(qT, kT_blk, v_blk, lists, bounds,
    q_block, pb[, packed, k_block])`` plus ``n_real_items``."""
    sp = stream_plan(plan, pb, packed)
    kT_blk, v_blk = stream_kv(sp, k, v, plan.key_valid)
    ops = dict(
        qT=stream_q(sp, q, scale), kT_blk=kT_blk, v_blk=v_blk, lists=sp.lists, bounds=sp.bounds,
        q_block=sp.q_block, pb=sp.pb, n_real_items=sp.n_real_items,
    )  # fmt: skip
    if packed:
        ops["packed"] = 1
        ops["k_block"] = sp.k_block
    return ops


def pack_kv_blocks(kT_blk: torch.Tensor, v_blk: torch.Tensor) -> torch.Tensor:
    """``(NBG, D, KB)`` K^T + ``(NBG, KB, D)`` V -> ``(NBG, 128, KB + (KB // 128) * D)``: row ``p`` =
    ``[K^T[d=p] | V[p] | V[128 + p] | ...]``, one contiguous DMA per block (``KB % 128 == 0``)."""
    nbg, d, kb = kT_blk.shape
    assert d == 128 and kb % 128 == 0, (d, kb)
    s = kb // 128
    vv = v_blk.reshape(nbg, s, 128, d).permute(0, 2, 1, 3).reshape(nbg, 128, s * d)
    return torch.cat([kT_blk, vv], dim=-1).contiguous()


def stream_emulate(
    qT, kT_blk, v_blk, lists, bounds, q_block: int, pb: int, packed: int = 0, k_block: int = 0
) -> torch.Tensor:
    """fp32 torch model of the kernel's own arithmetic (pass-wise online softmax over the listed
    blocks, bound masking) on its exact operands -- checks the operand builder without NKI."""
    d, ltot = qT.shape
    n_items, kp = lists.shape
    if packed:
        s_ = k_block // 128
        v_blk = kT_blk[:, :, k_block:].reshape(-1, 128, s_, d).permute(0, 2, 1, 3)
        v_blk = v_blk.reshape(-1, k_block, d)
        kT_blk = kT_blk[:, :, :k_block]
    kb = kT_blk.shape[2]
    n_pass, nkeys = bounds.shape[2], pb * kb
    out = torch.empty(ltot, d)
    for i in range(n_items):
        qi = qT[:, i * q_block : (i + 1) * q_block].float().t()  # (QB, D)
        m = torch.full((q_block, 1), float("-inf"))
        lsum = torch.zeros(q_block, 1)
        o = torch.zeros(q_block, d)
        for p in range(n_pass):
            ids = lists[i, p * pb : (p + 1) * pb].long()
            kt = kT_blk[ids].float().permute(1, 0, 2).reshape(d, nkeys)
            vv = v_blk[ids].float().reshape(nkeys, d)
            s = qi @ kt
            keep = torch.arange(nkeys).view(1, -1) < bounds[i, 0, p]
            s = s.masked_fill(~keep, float("-inf"))
            m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
            a = torch.exp(m - m_new).nan_to_num(0.0)
            pr = torch.exp(s - m_new)
            lsum = lsum * a + pr.sum(-1, keepdim=True)
            o = o * a + pr @ vv
            m = m_new
        out[i * q_block : (i + 1) * q_block] = o / lsum
    return out


def attend_stream(
    q, k, v, plan: BlockSparsePlan | StreamPlan, scale: float | None = None, pb: int | None = None,
    packed: bool | None = None, v_queue: int | None = None, key_valid: torch.Tensor | None = None,
):  # fmt: skip
    """L4: the streaming block-sparse kernel (:mod:`vllm_omni_neuron.kernels.bsa_stream`).

    ``q`` ``[H, Lq, 128]`` (this rank's rows), ``k``/``v`` ``[H, Lk, 128]`` (all keys). ``plan``: a
    :class:`BlockSparsePlan` (converted here, and its ``key_valid`` applied) or a prebuilt
    :class:`StreamPlan` (pass ``key_valid`` if pad rows must be zeroed). Same key-validity rule as
    L2: invalid keys only as the trailing rows of one block. Off the device: the fp32 reference."""
    h, lq, d = q.shape
    if isinstance(plan, BlockSparsePlan):
        if not _kernel_ok(q):
            return reference_attention(q, k, v, plan, scale)
        key_valid = plan.key_valid if key_valid is None else key_valid
        sp = stream_plan(plan, pb, packed)
    else:
        sp = plan
        assert _kernel_ok(q), (
            "a StreamPlan needs the device kernel; use the BlockSparsePlan off device"
        )
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.bsa_stream import bsa_stream

    if v_queue is None:
        v_queue = default_v_queue(sp)
    kT_blk, v_blk = stream_kv(sp, k, v, key_valid)
    out = wrap_nki(bsa_stream)[LNC](
        qT=stream_q(sp, q, scale), kT_blk=kT_blk, v_blk=v_blk,
        lists=sp.lists.to(q.device), bounds=sp.bounds.to(q.device),
        **stream_kernel_kwargs(sp, v_queue=v_queue),
    )  # fmt: skip
    return out[: sp.n_real_items * sp.q_block].reshape(h, lq, d)


# ================================================================================== NKI wrappers
# Plain module-level @nki.jit entry points (the NKI tracer reads the source: no closures, no
# imports inside the kernel body), mirroring the plugin's _vae_attention_kernel pattern.
@nki.jit
def _dense_bias_kernel(q, k, v, bias):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        position_bias=bias,
        bias_layout="dense",
    )


@nki.jit
def _dense_bounded_kernel(q, k, v, bound_min, bound_max):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        bound_min=bound_min,
        bound_max=bound_max,
    )


@nki.jit
def _dense_kernel(q, k, v):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def attend_dense(q, k, v, scale: float | None = None):
    """Dense ``attention_cte`` over ``[H, L, D]`` (the baseline the executors are measured against);
    torch SDPA off the device."""
    d = q.shape[-1]
    scale = d**-0.5 if scale is None else scale
    if not _kernel_ok(q):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale).to(
            q.dtype
        )
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    qs = (q.float() * scale).to(q.dtype).contiguous()
    return wrap_nki(_dense_kernel)[LNC](q=qs, k=k.contiguous(), v=v.contiguous())


__all__ = [
    "BlockSparsePlan",
    "NEG_BIAS",
    "attend_dense",
    "attend_dense_bias",
    "attend_gather_dense",
    "attend_index_list",
    "attend_stream",
    "block_bias",
    "blockify_kv",
    "gather_blocks",
    "index_list_inputs",
    "plan_from_block_mask",
    "plan_from_lists",
    "reference_attention",
    "sla_block_mask",
    "stream_emulate",
    "stream_kernel_kwargs",
    "stream_kv",
    "default_v_queue",
    "stream_lists",
    "stream_half_masks",
    "pair_union_plan",
    "superblock_plan",
    "superblock_masks",
    "stream_q",
    "stream_plan",
    "StreamPlan",
    "stream_inputs",
    "stream_pass_blocks",
]
