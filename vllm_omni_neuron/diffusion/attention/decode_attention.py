# SPDX-License-Identifier: Apache-2.0
"""Shared autoregressive decode-attention layer for text towers embedded in a diffusion pipeline
(Cosmos3's Qwen3-VL reasoner, Qwen-Image 2.1's prompt-enhancer LLM, pi0.5.2's text subtask) --
model-agnostic over hidden size, head count and KV-cache length, with a static-shape KV cache sized
for one request (no block-table paging: these towers run one sequence through a diffusion request,
not a multi-tenant LLM server).

Two phases, both with a torch fallback that runs anywhere (CPU included) and is the DEFAULT on
NC-v3+ too (see :func:`prefill_uses_nki_kernel`):

* **Prefill** (:func:`decode_prefill`): the whole prompt, causal, over all heads. The DEFAULT torch
  path uses an explicit additive causal mask (:func:`_causal_bias`) through SDPA's ``attn_mask=``,
  never ``is_causal=True`` -- round 9's device smoke found ``is_causal=True`` itself lowers as an
  opaque, illegal ``torch.operator`` under ``neuron_native_lite`` (``BackendCompilerFailed: failed
  to legalize operation torch.operator``), independent of the NKI kernel question below; this
  matches the plugin's own ``NeuronSDPABackend``/``NeuronSDPAImpl``
  (:mod:`vllm_omni_neuron.diffusion.attention.backends.sdpa`), which always passes an explicit
  ``mask_mode="broadcast_k"`` mask rather than ``is_causal``, and earlier XLA training work's
  finding that ``sdpa(is_causal=True)`` mis-lowers on XLA generally. ``_decode_step_torch``'s own
  multi-token causal mask uses the same finite-bias helper for the same reason. An NKI path through
  ``nkilib`` ``attention_cte`` also exists (:mod:`vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan`
  uses the same kernel for the VAE's attention, with ``causal_mask=False``) but is OPT-IN
  (``VLLM_OMNI_NEURON_DECODE_PREFILL_NKI=1``) and unverified: rounds 6-8 found and fixed contiguity
  and module-scope-kernel issues on it, but round 9 showed the ``BackendCompilerFailed`` the NKI
  path kept hitting was actually the ``is_causal=True`` call downstream of it in the smoke's
  torch-path comparison, not necessarily the kernel itself -- the kernel's own ``causal_mask=True``
  argument (still the only such call site in this plugin) remains unverified in isolation. Writes
  the K/V cache either way.
* **Decode** (:func:`decode_step`): one call (or a small block of ``S_tkg`` new tokens). The
  DEFAULT is the torch path (``_decode_step_torch``: project, RoPE, write the new K/V, attend
  against the whole position-masked cache, GQA-expand, project out -- ONE graph for every decode
  position, see :class:`StaticKVCache`), verified on Trn2 in smoke rounds 15-18 (decode steps at
  0.0037-0.0059 vs the CPU fp32 oracle, cache written and read back correctly). The OPT-IN kernel path
  (``VLLM_OMNI_NEURON_DECODE_STEP_NKI=1``, see :func:`decode_step_uses_nki_kernel` for why it is
  off) goes through
  ``vllm_neuron.functional.attention.attention_decode`` -- the production fused TKG kernel
  (``nkilib`` ``attention_block_tkg``: RMSNorm -> QKV projection -> RoPE -> GQA attention -> KV-cache
  update -> output projection, all in one NEFF). That API owns the QKV/output projections itself (it
  takes ``X`` + ``W_qkv``/``W_out``, not pre-split Q/K/V), so this layer's :class:`DecodeWeights` packs
  a tower's existing projection weights into the layout it expects once, up front.

Both phases take a :class:`StaticKVCache` sized at construction (``[1, kv_heads, max_len, head_dim]``,
batch pinned to 1): no block table, no dynamic allocation, so the compiled graph never changes shape
across the generation loop.

Kernel constraints (``attention_block_tkg``, decode phase; checked by :func:`can_use_decode_kernel`):
``batch * S_tkg * q_heads <= 128``, ``H`` (= q_heads * head_dim, the QKV projection's input width) a
multiple of 128, even head_dim, ``max_len`` a multiple of 128 (the fused mask-gen path requires
``s_prior % 128 == 0``, and this layer's flat cache makes ``s_prior == max_len``), NeuronCore-v3+.
Batch is always 1 in this layer, so the token-count limit is ``S_tkg <= 128 // q_heads`` -- decode
one token at a time for any tower with more than 128 query heads (none of the three target towers
do). A config that fails any of these simply never takes the kernel path: :func:`decode_step` is
always correct via the torch fallback, just without the kernel's throughput.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import nki
import nki.language as nl
import torch
from nkilib.core.attention.attention_cte import attention_cte

from vllm_omni_neuron.nc_generation import supports_nki

_PMAX = 128

# Finite additive-mask magnitude (matches neighborhood_attention.py's _tile_bias and the plugin's
# own NeuronSDPABackend convention of an explicit mask rather than SDPA's is_causal fast path):
# large enough that softmax zeroes a masked key's weight at any realistic score scale, but finite
# so it never produces NaN/inf through a fused softmax lowering the way float("-inf") or
# torch.finfo(dtype).min can on this backend.
_NEG_BIAS = -30000.0

PREFILL_NKI_ENV = "VLLM_OMNI_NEURON_DECODE_PREFILL_NKI"


def _causal_bias(s_q: int, s_k: int, offset: int, device, dtype) -> torch.Tensor:
    """Additive ``[s_q, s_k]`` causal mask: query row ``i`` (absolute position ``offset + i``) may
    attend key column ``j`` only when ``j <= offset + i``. Finite magnitude, see ``_NEG_BIAS``.

    Built with ``torch.where`` over an index comparison (iota-style), never ``torch.triu`` +
    ``float("-inf")``: the plugin's own SDPA backend (``diffusion/attention/backends/sdpa.py``)
    passes an explicit mask instead of ``is_causal=True`` for exactly this reason (round 9's smoke
    finding -- ``is_causal=True`` lowers as an opaque, illegal ``torch.operator`` under
    ``neuron_native_lite``; see this module's docstring).
    """
    q_pos = torch.arange(offset, offset + s_q, device=device).unsqueeze(-1)
    k_pos = torch.arange(s_k, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos
    return torch.where(
        allowed, torch.zeros((), dtype=dtype), torch.full((), _NEG_BIAS, dtype=dtype)
    ).to(device)


def _attention_matmul_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit scaled-dot-product attention: ``softmax((q @ k^T) * scale + bias) @ v``, fp32 softmax.

    q/k/v are ``[B, H, S, D]`` with ``H`` already equal across q and k (GQA expanded by the caller);
    ``bias`` is an additive mask broadcastable to ``[.., S_q, S_k]`` (e.g. :func:`_causal_bias`).

    Deliberately NOT ``F.scaled_dot_product_attention``: under ``neuron_native_lite``, SDPA with ANY
    mask -- ``is_causal=True`` (round 9) OR an explicit ``attn_mask=`` (round 10) -- lowers to an
    opaque fused-attention ``torch.operator`` that fails to legalize; only maskless SDPA decomposes.
    This spelled-out matmul/softmax/matmul is the pattern other Neuron model graphs in this plugin
    use and lowers to primitive ops the backend handles.
    """
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def prefill_uses_nki_kernel() -> bool:
    """Whether :func:`decode_prefill` takes the NKI ``attention_cte`` path instead of the default
    torch (SDPA) path. Opt-in (default off, even on NC-v3+): this is the only ``causal_mask=True``
    call site of ``attention_cte`` in this plugin, and it fails ``BackendCompilerFailed: failed to
    legalize operation torch.operator`` under ``neuron_native_lite`` -- the VAE's own use of the
    same kernel (``causal_mask=False``) compiles and runs correctly, so the causal path specifically
    is unverified. Set ``VLLM_OMNI_NEURON_DECODE_PREFILL_NKI=1`` once that is fixed or verified for
    your shape; the torch path is numerically exact (SDPA) and is what every integrator gets by
    default so a text tower is usable now rather than blocked on this kernel.
    """
    return os.environ.get(PREFILL_NKI_ENV, "0") == "1"


DECODE_STEP_NKI_ENV = "VLLM_OMNI_NEURON_DECODE_STEP_NKI"


def decode_step_uses_nki_kernel() -> bool:
    """Whether :func:`decode_step` may take ``vllm_neuron``'s fused ``attention_decode`` NKI kernel
    (when :func:`can_use_decode_kernel` also allows it) instead of the default torch path.

    OPT-IN (``VLLM_OMNI_NEURON_DECODE_STEP_NKI=1``), default off, after smoke round 15 on Trn2: with
    the KV cache verified written identically by both paths (fill, written-slot rel 0.0035 vs CPU,
    unfilled slots exactly 0), the torch path's three decode steps matched the CPU fp32 oracle to
    0.0037 while the kernel path's were at 1.03 -- an essentially uncorrelated output, not a
    precision gap -- and the kernel's own torch FALLBACK (``attention_decode`` on CPU with these
    exact arguments) matches this layer's torch path to 0.0. So the input convention is right and
    the divergence is inside the device kernel for THIS call shape: GQA (16 query heads over 2 KV
    heads) in bf16 with a per-head 3D block table, which ``vllm_neuron`` itself notes is reached in
    production only by its native-MX Qwen3-VL decoder (its bf16 GQA decoders stay on the torch
    fallback). Smoke round 17's ``decode_attention_kernel_diag`` (MHA 1.28 wrong, GQA with
    IDENTICAL KV heads 0.0067 right, real GQA 0.95 wrong) pinned it: the per-head block table must
    hold the kernel's GLOBAL pool indices ``block * kv_heads + head`` (see
    :meth:`StaticKVCache.active_blocks_table`), and this layer passed all zeros, so every query head
    read KV head 0. Fixed; round 18 confirmed it on device (MHA 0.0067, GQA 0.0070 vs the CPU-bf16
    band 0.0063). Still opt-in for now: every integrator is on the
    verified torch path, and flipping the default is a separate decision.
    """
    return os.environ.get(DECODE_STEP_NKI_ENV, "0") == "1"


@dataclass
class DecodeAttentionConfig:
    """Static shape/precision contract for one text tower's decode attention.

    ``q_heads`` / ``kv_heads`` must already reflect this rank's share under tensor parallelism
    (the caller shards weights; this layer is TP-oblivious). ``kv_heads == 1`` is MQA, dividing
    ``q_heads`` is GQA, ``q_heads == kv_heads`` is MHA.
    """

    q_heads: int
    kv_heads: int
    head_dim: int
    max_len: int
    rope_contiguous_layout: bool = True
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for the NKI kernel path, got {self.head_dim}")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"q_heads ({self.q_heads}) must be a multiple of kv_heads ({self.kv_heads})"
            )

    @property
    def hidden_size(self) -> int:
        """QKV projection input width (``H`` in the kernel constraints)."""
        return self.q_heads * self.head_dim

    @property
    def num_kv_groups(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5

    def max_decode_tokens(self) -> int:
        """Largest S_tkg the kernel path accepts at batch=1: floor(128 / q_heads), at least 1."""
        return max(1, _PMAX // self.q_heads)


@dataclass
class DecodeWeights:
    """A text tower's QKV/output projection weights, packed for :func:`decode_step`.

    ``W_qkv``: ``[hidden_in, (q_heads + 2*kv_heads) * head_dim]`` (Q then K then V along the out dim,
    the layout ``attention_decode`` splits by ``_can_use_attention_block_kernel`` / the torch
    fallback). ``W_out``: ``[q_heads * head_dim, hidden_out]`` or ``None`` to skip the output
    projection (caller applies its own). Build with :meth:`from_separate` from a tower's existing
    ``nn.Linear`` weights (no re-training, just a reshape/concat).
    """

    W_qkv: torch.Tensor
    W_out: torch.Tensor | None = None
    bias_qkv: torch.Tensor | None = None
    bias_out: torch.Tensor | None = None

    @classmethod
    def from_separate(
        cls,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        out_proj: torch.Tensor | None = None,
        *,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        out_bias: torch.Tensor | None = None,
    ) -> DecodeWeights:
        """Pack separate ``nn.Linear``-style weights (``[out_features, in_features]``, PyTorch's
        convention) into the ``[in_features, out_features]`` layout ``attention_decode`` expects."""
        w_qkv = torch.cat([q_proj, k_proj, v_proj], dim=0).t().contiguous()
        bias_qkv = torch.cat([q_bias, k_bias, v_bias], dim=0) if q_bias is not None else None
        w_out = out_proj.t().contiguous() if out_proj is not None else None
        return cls(W_qkv=w_qkv, W_out=w_out, bias_qkv=bias_qkv, bias_out=out_bias)

    def to(self, *args, **kwargs) -> DecodeWeights:
        return DecodeWeights(
            W_qkv=self.W_qkv.to(*args, **kwargs),
            W_out=self.W_out.to(*args, **kwargs) if self.W_out is not None else None,
            bias_qkv=self.bias_qkv.to(*args, **kwargs) if self.bias_qkv is not None else None,
            bias_out=self.bias_out.to(*args, **kwargs) if self.bias_out is not None else None,
        )


class StaticKVCache:
    """A fixed-shape, single-sequence KV cache: ``[1, kv_heads, max_len, head_dim]``.

    ``pos`` (an ``int32[1]`` DEVICE tensor) is the fill position: how many slots hold real (vs
    zero-pad) tokens. It is a tensor, not a Python int, on purpose: :func:`decode_step` is compiled,
    and a Python int read inside a compiled region is a constant Dynamo SPECIALIZES on -- with
    ``fill`` as an int, every decode position was a different graph (``K_cache[:, :, :pos0]`` has a
    different shape per step), so each step paid a Dynamo retrace plus a NEFF compile or cache
    lookup (~0.3 s warm, ~2.5 s on a cache miss in smoke rounds 15-18), and ``fullgraph=True``
    would hit the recompile limit on the ninth token. With the position a graph input, the decode
    step is ONE static graph for every position: the mask and the cache write are built from
    ``pos`` with ``arange``/compare/``where``, never from a data-dependent slice. ``fill`` (a host
    ``int``) is still available for introspection outside compiled code; it syncs the device.

    The cache tensors are device tensors this class owns; both write paths (prefill, decode) go
    through here so ``pos`` stays consistent regardless of which attention path wrote them.
    """

    def __init__(self, cfg: DecodeAttentionConfig, device: torch.device | str) -> None:
        self.cfg = cfg
        shape = (1, cfg.kv_heads, cfg.max_len, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=cfg.dtype, device=device)
        self.v = torch.zeros(shape, dtype=cfg.dtype, device=device)
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)

    @property
    def fill(self) -> int:
        """Host-side read of the fill position (syncs the device; NOT for use inside a compiled
        region -- read ``pos`` there)."""
        return int(self.pos.item())

    def write_prefill(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """``k``/``v``: ``[1, kv_heads, S, head_dim]``, written at position 0 (``S <= max_len``).

        Casts to the cache's own dtype and forces a contiguous layout first: an in-place ``copy_``
        from a non-contiguous source (a ``.transpose()``/``.permute()`` view, which every caller in
        this module passes) or across a dtype mismatch is tolerant on CPU but the Neuron/XLA
        lowering of the equivalent op is strict about both (``RuntimeError: Expected
        self.is_contiguous()`` / ``self.dtype() == dst.dtype()``), so this layer never relies on
        either reaching the device. The prompt length is a static shape, so this write is a plain
        ``slice_scatter`` at 0 (one prefill graph per prompt length, as for any prefill).
        """
        s = k.shape[2]
        if s > self.cfg.max_len:
            raise ValueError(f"prefill length {s} exceeds max_len {self.cfg.max_len}")
        self.k = torch.slice_scatter(self.k, k.to(self.k.dtype).contiguous(), dim=2, start=0, end=s)
        self.v = torch.slice_scatter(self.v, v.to(self.v.dtype).contiguous(), dim=2, start=0, end=s)
        self.pos = torch.full((1,), s, dtype=torch.int32, device=self.k.device)

    def write_decode(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """``k``/``v``: ``[1, kv_heads, S_tkg, head_dim]``, appended at the current ``pos`` (a device
        tensor), position-static: the target slots are a one-hot ``[S_tkg, max_len]`` selection
        built from ``pos`` with ``arange`` + compare, the new rows land through an exact fp32
        one-hot matmul, and the cache is REPLACED by a ``where`` (a plain graph output bound to the
        attribute, as before: round 14 found an in-place slice write to a graph input depends on the
        backend's aliasing pass). No ``slice_scatter``/``index_put`` with a data-dependent start.

        Overflow (``pos + S_tkg > max_len``) cannot raise inside a compiled region (the position is
        data); it is the caller's contract (size ``max_len`` for the request). An overflowing write
        is dropped (no slot matches), never wraps.
        """
        s = k.shape[2]
        dev = self.k.device
        slots = torch.arange(self.cfg.max_len, dtype=torch.int32, device=dev)  # [L]
        target = self.pos + torch.arange(s, dtype=torch.int32, device=dev)  # [S_tkg]
        onehot = slots[None, :] == target[:, None]  # [S_tkg, L] bool
        sel = onehot.to(torch.float32).transpose(0, 1)  # [L, S_tkg]
        written = onehot.any(dim=0)[None, None, :, None]  # [1, 1, L, 1]
        k_rows = torch.matmul(sel, k[0].to(torch.float32).contiguous())  # [kv_heads, L, head_dim]
        v_rows = torch.matmul(sel, v[0].to(torch.float32).contiguous())
        self.k = torch.where(written, k_rows[None].to(self.k.dtype), self.k)
        self.v = torch.where(written, v_rows[None].to(self.v.dtype), self.v)
        self.pos = self.pos + s

    def visible_bias(self, s_tkg: int, dtype: torch.dtype) -> torch.Tensor:
        """``[S_tkg, max_len]`` additive mask for ``S_tkg`` new tokens at ``pos .. pos+S_tkg-1``
        (call AFTER :meth:`write_decode`, which advanced ``pos`` past them -- it is computed back
        from the advanced position): new token ``i`` sees slots ``<= pos0 + i`` (the filled prefix
        plus itself and the earlier new tokens; causal among the block), ``-30000`` elsewhere (the
        finite-bias convention of :func:`_causal_bias`). Position-static: built from ``pos``."""
        dev = self.k.device
        slots = torch.arange(self.cfg.max_len, dtype=torch.int32, device=dev)
        last = self.pos - s_tkg + torch.arange(s_tkg, dtype=torch.int32, device=dev)  # [S_tkg]
        allowed = slots[None, :] <= last[:, None]
        return torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=dev),
            torch.full((), -30000.0, dtype=dtype, device=dev),
        )

    def as_4d_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``[num_blocks=1, kv_heads, block_len=max_len, head_dim]`` -- ``attention_decode``'s 4D cache."""
        return self.k, self.v

    def active_blocks_table(self, *, per_head: bool = True) -> torch.Tensor:
        """The single flat block, as the GQA-capable per-head table ``[1, kv_heads, 1]``
        (``per_head=True``, the form ``attention_decode``'s kernel path requires to route GQA to the
        NKI kernel) or the shared 2D table ``[1, 1]`` (``per_head=False``, the only form its torch
        fallback accepts for ``kv_heads > 1`` -- the fallback explicitly rejects a 3D table).

        Per-head entries are GLOBAL POOL indices, not block numbers: ``attention_block_tkg`` folds
        ``kv_heads`` into its batch dim and addresses the 4D cache ``[num_blocks, kv_heads, block_len,
        d]`` as a flat ``[num_blocks * kv_heads]`` pool, so the entry for KV head ``h`` of block ``b``
        is ``b * kv_heads + h`` (the kernel docstring's "block indices must use ``num_blocks *
        kv_heads`` addressing"; ``vllm_neuron``'s ``build_per_head_block_table`` encodes the same
        rule for the Qwen3-VL decoders). With one block that is simply ``h``. Smoke rounds 15-17 on
        Trn2 had this table all zeros: every query head then read KV head 0's slot, which is exactly
        why the kernel matched CPU only when all KV heads were identical (``gqa_identical_kv``
        0.0067) and was uncorrelated for MHA (1.28) and real GQA (0.95).
        """
        if per_head:
            return torch.arange(self.cfg.kv_heads, dtype=torch.int32, device=self.k.device).reshape(
                1, self.cfg.kv_heads, 1
            )
        return torch.zeros(1, 1, dtype=torch.int32, device=self.k.device)


def can_use_decode_kernel(cfg: DecodeAttentionConfig, s_tkg: int) -> bool:
    """Whether :func:`decode_step` can take the fused NKI kernel for this config and token count.

    ``max_len % 128 == 0`` is required because this layer always uses ``attention_decode``'s fused
    mask-gen path (``pos_ids``), whose on-chip mask generation (``gen_attention_decode_mask``)
    requires ``s_prior % 128 == 0`` -- and this layer's flat (non-block) cache makes ``s_prior ==
    max_len``. A config that fails this (or any other constraint here) simply never takes the
    kernel path -- :func:`decode_step` still works correctly via the torch fallback, just without
    the kernel's throughput.
    """
    return (
        supports_nki()
        and cfg.hidden_size % _PMAX == 0
        and s_tkg * cfg.q_heads <= _PMAX  # batch is always 1 in this layer
        and cfg.head_dim % 2 == 0
        and cfg.max_len % _PMAX == 0
    )


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, contiguous_layout: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q``/``k``: ``[1, heads, S, D]``; ``cos``/``sin``: ``[S, D // 2]``, already in ``q.dtype``.

    Does NOT cast ``cos``/``sin`` itself: an eager (uncompiled) cross-dtype ``.to()`` on a Neuron/XLA
    device tensor can fail outside a traced graph region (``RuntimeError: Expected self.dtype() ==
    dst.dtype()``) even though the identical cast succeeds on CPU -- the exact failure mode of the
    Trn2 smoke's decode_attention check, traced to this cast. Callers (:func:`decode_prefill`,
    :func:`decode_step`) cast on the HOST, before any device transfer, so this never has to.

    Forces ``q``/``k`` contiguous immediately: both arrive as a ``.transpose(1, 2)`` view of a QKV
    split (``[B, S, heads, D] -> [B, heads, S, D]``), and the slice-negate-``cat``/``stack`` chain
    below (splitting the last, now non-contiguous-strided dim) raised ``RuntimeError: Expected
    self.is_contiguous()`` on the real device even though the identical ops succeed on CPU for the
    same strided view -- the Trn2 smoke's actual failing line (round 5's per-tensor diagnostics
    pinned it to this function's input, not the kernel call this module had already made
    ``.contiguous()`` at).
    """
    if cos.dtype != q.dtype or sin.dtype != q.dtype:
        raise ValueError(
            f"cos/sin must already be in q's dtype ({q.dtype}), got {cos.dtype}/{sin.dtype} -- cast "
            "on the host before moving to device, not here (see this function's docstring)"
        )
    q, k = q.contiguous(), k.contiguous()
    cos_h, sin_h = cos, sin
    if contiguous_layout:
        cos_f = torch.cat([cos_h, cos_h], dim=-1)
        sin_f = torch.cat([sin_h, sin_h], dim=-1)

        def rotate(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    else:
        cos_f = torch.stack([cos_h, cos_h], dim=-1).flatten(-2)
        sin_f = torch.stack([sin_h, sin_h], dim=-1).flatten(-2)

        def rotate(x):
            x_even, x_odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    cos_f, sin_f = cos_f[None, None], sin_f[None, None]
    return q * cos_f + rotate(q) * sin_f, k * cos_f + rotate(k) * sin_f


@nki.jit
def _prefill_attention_cte_kernel(q, k, v):
    """Causal, multi-head ``attention_cte``, at module scope (not nested inside a function): a
    fresh local closure rebuilt on every call lowered as an opaque, explicitly-illegal
    ``torch.operator`` under ``torch.compile`` (``BackendCompilerFailed ... failed to legalize
    operation torch.operator``) instead of the recognized NKI HOP the VAE module's module-scope
    ``_vae_attention_kernel`` / ``wrap_nki`` pattern produces -- this mirrors that pattern exactly.
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=True,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def _prefill_attention_cte(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal multi-head ``attention_cte`` over ``(heads, S, head_dim)`` q/k/v (pre-scaled q, GQA
    already expanded to q_heads by the caller), run through ``wrap_nki``, the traceable NKI HOP
    (see :func:`_prefill_attention_cte_kernel`'s docstring for why the kernel must be module-scope).

    Not registered via ``lite_compat.nki_op``: that installs the raw ``@nki.jit`` kernel as its own
    fake impl, so Dynamo executes it during fake-tensor tracing and imports ``torch_neuronx``, which
    the Lite image lacks (see the VAE module's ``_vae_nki_attn``, the same pattern this mirrors).
    ``wrap_nki`` shape-infers in its meta impl instead.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(_prefill_attention_cte_kernel)
    return wrapped[2](q, k, v)


def decode_prefill(
    cfg: DecodeAttentionConfig,
    cache: StaticKVCache,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal prefill attention over the whole prompt. Default path: explicit additive mask + SDPA
    (:func:`_attention_matmul_softmax`, not ``F.scaled_dot_product_attention`` -- round 10 found SDPA
    with any mask, not just ``is_causal=True``, lowers as an opaque illegal op under
    ``neuron_native_lite``; see that helper's docstring). An opt-in NKI ``attention_cte`` path
    exists (:func:`prefill_uses_nki_kernel`).

    ``q``: ``[1, q_heads, S, head_dim]``; ``k``/``v``: ``[1, kv_heads, S, head_dim]`` -- already
    projected by the caller's own QKV weights (prefill has no fused-kernel QKV path; only decode does,
    through :class:`DecodeWeights`). RoPE is applied here (before GQA expansion) when ``cos``/``sin``
    (``[S, head_dim // 2]``) are given. Writes ``k``/``v`` (post-RoPE) into ``cache`` at position 0 and
    returns the attention output ``[1, q_heads, S, head_dim]``.

    Forces ``q``/``k``/``v`` contiguous immediately: every real caller passes a ``.transpose(1, 2)``
    view of a QKV split (``[B, S, heads, D] -> [B, heads, S, D]``), and that non-contiguous stride
    reaches real device ops here (RoPE's slice/cat chain, ``repeat_interleave``, the cache write,
    the kernel call) that raised ``RuntimeError: Expected self.is_contiguous()`` on Trn2 even though
    the identical ops succeed on CPU for the same view.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if cos is not None:
        q, k = _apply_rope(q, k, cos, sin, cfg.rope_contiguous_layout)
    cache.write_prefill(k, v)
    k_full = k.repeat_interleave(cfg.num_kv_groups, dim=1) if cfg.num_kv_groups > 1 else k
    v_full = v.repeat_interleave(cfg.num_kv_groups, dim=1) if cfg.num_kv_groups > 1 else v
    if q.device.type == "neuron" and supports_nki() and prefill_uses_nki_kernel():
        out = _prefill_attention_cte(
            (q[0] * cfg.scale).contiguous(), k_full[0].contiguous(), v_full[0].contiguous()
        )[None]
    else:
        s_q = q.shape[2]
        bias = _causal_bias(s_q, s_q, offset=0, device=q.device, dtype=q.dtype)
        out = _attention_matmul_softmax(q, k_full, v_full, cfg.scale, bias)
    return out


def decode_prefill_from_hidden(
    cfg: DecodeAttentionConfig,
    weights: DecodeWeights,
    cache: StaticKVCache,
    hidden: torch.Tensor,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill straight from the pre-QKV hidden state ``[1, S, hidden_size]``: projects through
    ``weights.W_qkv`` / ``bias_qkv``, splits, calls :func:`decode_prefill`, and (when ``W_out`` is
    set) applies the output projection -- returning ``[1 * S, hidden_out]``, the same contract as
    :func:`decode_step`, or ``[1, q_heads, S, head_dim]`` when ``W_out`` is ``None``.

    Prefer this over :func:`decode_prefill` when the call is the root of a ``torch.compile`` region on
    the device. Round 11's smoke showed why: the ``neuron_native_lite`` executor rejects a graph INPUT
    that is a non-contiguous view of a device tensor (``RuntimeError: Detected non-contiguous slicing
    for requested Device Tensor``) -- and the natural way to feed ``decode_prefill`` is exactly such a
    view, ``qkv[..., :q_end].view(1, S, heads, D).transpose(1, 2)``. The ``.contiguous()`` inside
    ``decode_prefill`` cannot help: it runs inside the graph, after the executor has already refused
    the input. With the projection and split inside the same compiled region, every graph input is a
    plain contiguous tensor (``hidden``, the weights, the RoPE tables, the cache).
    """
    B, S, _ = hidden.shape
    qkv = hidden @ weights.W_qkv
    if weights.bias_qkv is not None:
        qkv = qkv + weights.bias_qkv
    d = cfg.head_dim
    q_end, k_end = cfg.q_heads * d, cfg.q_heads * d + cfg.kv_heads * d
    q = qkv[..., :q_end].reshape(B, S, cfg.q_heads, d).transpose(1, 2).contiguous()
    k = qkv[..., q_end:k_end].reshape(B, S, cfg.kv_heads, d).transpose(1, 2).contiguous()
    v = qkv[..., k_end:].reshape(B, S, cfg.kv_heads, d).transpose(1, 2).contiguous()
    attn = decode_prefill(cfg, cache, q, k, v, cos=cos, sin=sin)
    if weights.W_out is None:
        return attn
    flat = attn.transpose(1, 2).reshape(B * S, cfg.q_heads * d)
    out = flat @ weights.W_out
    if weights.bias_out is not None:
        out = out + weights.bias_out
    return out


def decode_step(
    cfg: DecodeAttentionConfig,
    weights: DecodeWeights,
    cache: StaticKVCache,
    hidden: torch.Tensor,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """One decode call: project ``hidden`` through ``weights`` and attend ``S_tkg`` new tokens against
    the cache filled so far, through the fused ``attention_block_tkg`` kernel on NC-v3+
    (``vllm_neuron``'s ``attention_decode``), else a plain torch fallback with the same algorithm.

    ``hidden``: ``[1, S_tkg, hidden_size]``. Returns the tower's attention-block output: ``[1 * S_tkg,
    hidden_out]`` when ``weights.W_out`` is set (post output-projection), else ``[1, q_heads, head_dim,
    S_tkg]`` (``attention_decode``'s pre-projection layout). The cache is updated in place either way.
    """
    s_tkg = hidden.shape[1]
    # The position is a DEVICE tensor (cache.pos), never a Python int read here: an int would make
    # Dynamo specialize this compiled function on it -- one graph (and one NEFF) per decode position
    # (see StaticKVCache). Everything below that depends on the position is tensor arithmetic.
    pos_t = cache.pos  # int32 [1]
    pos_ids = (
        pos_t.to(torch.float32) + torch.arange(s_tkg, dtype=torch.float32, device=hidden.device)
    )[None]
    # [S_tkg, D//2] -> [D//2, 1, S_tkg]. Callers must already pass cos/sin in cfg.dtype (the
    # production caller, vllm_neuron.model.qwen3_vl, casts its own cos/sin before this call) --
    # this never casts on-device itself: an eager cross-dtype .to() on a Neuron/XLA tensor can fail
    # outside a traced graph region even though it succeeds on CPU (see _apply_rope's docstring for
    # the exact failure this guards against; it is the same cast, same quirk, same fix).
    if cos is not None and cos.dtype != cfg.dtype:
        raise ValueError(f"cos must already be in cfg.dtype ({cfg.dtype}), got {cos.dtype}")
    if sin is not None and sin.dtype != cfg.dtype:
        raise ValueError(f"sin must already be in cfg.dtype ({cfg.dtype}), got {sin.dtype}")
    rope_cos = cos.t()[:, None] if cos is not None else None
    rope_sin = sin.t()[:, None] if sin is not None else None

    if (
        decode_step_uses_nki_kernel()  # opt-in: see its docstring (round 15: kernel path rel 1.03)
        and can_use_decode_kernel(cfg, s_tkg)
        and hidden.device.type == "neuron"
    ):
        from vllm_neuron.functional.attention.attention_decode import attention_decode

        K_cache, V_cache = cache.as_4d_cache()
        update_idx = (pos_t + torch.arange(s_tkg, dtype=torch.int32, device=hidden.device)).to(
            torch.uint32
        )[None]
        out, k_new, v_new = attention_decode(
            X=hidden,
            W_qkv=weights.W_qkv,
            bias_qkv=weights.bias_qkv,
            cos=rope_cos,
            sin=rope_sin,
            rope_contiguous_layout=cfg.rope_contiguous_layout,
            active_blocks_table=cache.active_blocks_table(),
            K_cache=K_cache,
            V_cache=V_cache,
            softmax_scale=cfg.scale,
            pos_ids=pos_ids,
            update_cache=False,  # this layer owns cache.fill bookkeeping; write via StaticKVCache
            kv_cache_update_idx=update_idx,
            W_out=weights.W_out,
            bias_out=weights.bias_out,
        )
        # k_new comes back head-dim-major: the torch fallback returns [head_dim, B*kv_heads, S_tkg]
        # (its Stage 9 `k_for_return.permute(2, 0, 1)`), the device kernel [head_dim, B, kv_heads,
        # S_tkg] (smoke round 13: (128, 1, 2, 1) for kv_heads=2, S_tkg=1). Either way the trailing
        # order is (kv_heads, S_tkg) with head_dim in front, so one adjacent transpose of the
        # [head_dim, rest] flattening gives [rest, head_dim] -> [1, kv_heads, S_tkg, head_dim].
        # v_new is [B, kv_heads, S_tkg, head_dim] already (torch) / rank-flexible on device: reshape.
        k_new = _kv_new_to_cache_layout(
            k_new, cfg.kv_heads, s_tkg, cfg.head_dim, head_dim_first=True
        )
        v_new = _kv_new_to_cache_layout(
            v_new, cfg.kv_heads, s_tkg, cfg.head_dim, head_dim_first=False
        )
        cache.write_decode(k_new, v_new)
        return out

    out = _decode_step_torch(cfg, weights, cache, hidden, s_tkg, rope_cos, rope_sin)
    return out


def _kv_new_to_cache_layout(
    t: torch.Tensor, kv_heads: int, s_tkg: int, head_dim: int, *, head_dim_first: bool
) -> torch.Tensor:
    """Normalise ``attention_decode``'s returned new-token K/V (any rank, B=1) to the cache layout
    ``[1, kv_heads, S_tkg, head_dim]`` using reshapes and a single adjacent transpose only."""
    n = kv_heads * s_tkg * head_dim
    if t.numel() != n:
        raise ValueError(
            f"unexpected new K/V size {tuple(t.shape)} for kv_heads={kv_heads} s_tkg={s_tkg} head_dim={head_dim}"
        )
    if (
        head_dim_first
    ):  # [head_dim, (B, kv_heads, S_tkg) in any grouping] -> [kv_heads*S_tkg, head_dim]
        t = t.reshape(head_dim, kv_heads * s_tkg).transpose(0, 1)
    return t.reshape(1, kv_heads, s_tkg, head_dim).contiguous()


def _decode_step_torch(
    cfg: DecodeAttentionConfig,
    weights: DecodeWeights,
    cache: StaticKVCache,
    hidden: torch.Tensor,
    s_tkg: int,
    rope_cos: torch.Tensor | None,
    rope_sin: torch.Tensor | None,
) -> torch.Tensor:
    """Reference decode path: project, RoPE, write the new K/V into the cache, attend against the
    WHOLE ``max_len`` cache under a position mask, GQA-expand, project out. Runs on any device /
    NeuronCore generation; used as the fallback and as the CPU oracle.

    Position-static by construction (one compiled graph for every decode position): the cache is
    never sliced at the fill position -- the unfilled slots take part in the matmul (zeros) and are
    removed by the ``-30000`` mask from :meth:`StaticKVCache.visible_bias`, exactly like the
    causal mask among the new tokens. At ``max_len`` of a few thousand the extra MACs are
    negligible next to the projections."""
    B, S, _ = hidden.shape
    qkv = hidden @ weights.W_qkv
    if weights.bias_qkv is not None:
        qkv = qkv + weights.bias_qkv
    d = cfg.head_dim
    q_end, k_end = cfg.q_heads * d, cfg.q_heads * d + cfg.kv_heads * d
    q = qkv[..., :q_end].view(B, S, cfg.q_heads, d).transpose(1, 2).contiguous()
    k = qkv[..., q_end:k_end].view(B, S, cfg.kv_heads, d).transpose(1, 2).contiguous()
    v = qkv[..., k_end:].view(B, S, cfg.kv_heads, d).transpose(1, 2).contiguous()
    if rope_cos is not None:
        cos = rope_cos[:, 0].t()  # [D//2, 1, S] -> [S, D//2]
        sin = rope_sin[:, 0].t()
        q, k = _apply_rope(q, k, cos, sin, cfg.rope_contiguous_layout)

    cache.write_decode(k, v)
    k_full, v_full = cache.as_4d_cache()  # [1, kv_heads, max_len, D], new tokens included
    k_full, v_full = k_full.to(q.dtype), v_full.to(q.dtype)  # no-op when hidden is in cfg.dtype
    if cfg.num_kv_groups > 1:
        k_full = k_full.repeat_interleave(cfg.num_kv_groups, dim=1)
        v_full = v_full.repeat_interleave(cfg.num_kv_groups, dim=1)
    scores = torch.matmul(q, k_full.transpose(-1, -2)).float() * cfg.scale  # [1, q_heads, S, L]
    scores = scores + cache.visible_bias(s_tkg, scores.dtype)[None, None]
    attn = torch.matmul(torch.softmax(scores, dim=-1).to(v_full.dtype), v_full)

    if weights.W_out is not None:
        flat = attn.transpose(1, 2).reshape(B * S, cfg.q_heads * d)
        out = flat @ weights.W_out
        if weights.bias_out is not None:
            out = out + weights.bias_out
        return out
    return attn.permute(0, 1, 3, 2).contiguous()
