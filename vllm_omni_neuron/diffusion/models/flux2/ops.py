# SPDX-License-Identifier: Apache-2.0
"""Shared math for the FLUX.2 Neuron port: norms, RoPE, attention dispatch, TP helpers.

Everything here is a plain tensor function so the same code is traced into the compiled
NeuronCore graphs and run eagerly on CPU for the reference / unit tests.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ATTN_IMPL = os.environ.get("FLUX2_ATTN_IMPL", "auto").lower()  # auto | torch
MASK_VALUE = -30000.0  # finite: -inf poisons fully masked rows and bf16 casts


def tp_state() -> tuple[int, int, object]:
    """``(tp_size, tp_rank, tp_device_group)``; ``(1, 0, None)`` without an initialized vLLM TP group.

    Under TP>1 (or CP>1) the groups' full partitions are registered with the Neuron compiler's mesh
    registry so in-graph collectives can be legalized (``register_replica_groups``).
    """
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError):
        return 1, 0, None
    cp = cp_state()[0]
    if size > 1 or cp > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=cp)
    return size, rank, group


def cp_state() -> tuple[int, int, object]:
    """``(cp_size, cp_rank, cp_device_group)`` of the context-parallel (upstream: sequence-parallel)
    group; ``(1, 0, None)`` when it is not initialized or has one rank."""
    try:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import get_cp_group

        g = get_cp_group()
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    if g is None or g.world_size <= 1:
        return 1, 0, None
    return g.world_size, g.rank_in_group, g.device_group


def all_reduce(x: torch.Tensor, tp_size: int, group) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(x, group=group)
    return x


def all_gather_seq(x: torch.Tensor, size: int, group) -> torch.Tensor:
    """Concatenate ``x`` ``[B, S, ...]`` along the sequence axis over ``group`` (group-rank order)."""
    if size <= 1:
        return x
    seq_first = x.transpose(0, 1).contiguous()
    out = torch.empty(
        (seq_first.shape[0] * size, *seq_first.shape[1:]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(out, seq_first, group=group)
    return out.transpose(0, 1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """diffusers / HF RMSNorm: fp32 statistics, cast back to the input dtype, then scale."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * weight


def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Non-affine LayerNorm with fp32 statistics."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers ``apply_rotary_emb(use_real_unbind_dim=-1, sequence_dim=1)``.

    ``x`` ``[B, S, H, D]``; ``cos``/``sin`` ``[S, D]`` fp32 (pair-interleaved, as FLUX.2's
    ``Flux2PosEmbed`` returns them).
    """
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return (x.float() * c + x_rot.float() * s).to(x.dtype)


def rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF Llama/Mistral rotate-half RoPE. ``x`` ``[B, S, H, D]``, ``cos``/``sin`` ``[S, D]``."""
    d = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    c, s = cos[None, :, None, :].to(x.dtype), sin[None, :, None, :].to(x.dtype)
    return x * c + x_rot * s


def torch_attention(q, k, v, scale, bias=None):
    """Softmax attention, fp32 softmax. ``q`` ``[B, H, Sq, D]``, ``k``/``v`` ``[B, Hk, Sk, D]``."""
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(v.dtype), v)


def _use_nki(q: torch.Tensor) -> bool:
    if ATTN_IMPL == "torch" or q.device.type == "cpu":
        return False
    from vllm_omni_neuron.nc_generation import use_nki_kernels

    return use_nki_kernels(q)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Non-causal, unmasked attention ``[B, H, S, D]`` -> ``[B, H, S, D]``.

    NeuronCore-v3+ device tensors take the plugin's ``attention_cte`` flash kernel (the one
    Wan2.2 uses); CPU / NC-v2 / ``FLUX2_ATTN_IMPL=torch`` take the torch path.
    """
    if _use_nki(q):
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

        return _nki_attend(q, k, v, scale).transpose(2, 3)  # kernel returns d-major [B, H, D, S]
    return torch_attention(q, k, v, scale)


def param(shape, dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def segments_loader(dim: int, segments: list[tuple[int, int]], rank_div: int = 1):
    """TP loader: this rank's slice of each ``(offset, per_rank)`` segment along ``dim``, concatenated.

    Covers plain column/row parallel (one segment at offset 0) and fused projections whose
    output (or input) dimension concatenates several independently sharded blocks, e.g.
    FLUX.2's ``to_qkv_mlp_proj`` = ``[q | k | v | gate | up]``. ``rank_div > 1`` replicates each slice
    over ``rank_div`` consecutive ranks (KV heads when TP exceeds the KV-head count).
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    def transform(slices, rank):
        (sl,) = slices
        rank //= rank_div
        parts = []
        for off, n in segments:
            a, b = off + rank * n, off + (rank + 1) * n
            parts.append(sl[a:b] if dim == 0 else sl[:, a:b])
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)

    return SafetensorsWeightLoader(transform=transform)


def set_loader(
    p: nn.Parameter, dim: int, segments: list[tuple[int, int]], rank_div: int = 1
) -> None:
    from vllm_neuron.utils.weight_loader import set_weight_loader

    set_weight_loader(p, segments_loader(dim, segments, rank_div))


def cpu_mode() -> bool:
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


SYNC_EVERY = int(os.environ.get("FLUX2_SYNC_EVERY", "8"))
DEBUG_LAUNCH = os.environ.get("FLUX2_DEBUG_LAUNCH", "0") == "1"
TIMING = (
    os.environ.get("FLUX2_TIMING", "0") == "1"
)  # rank-0 per-call stage times (to device completion)


class LaunchThrottle:
    """Bound the number of compiled graphs queued on the NeuronCore.

    The Lite runtime executes compiled graphs asynchronously; a host loop that launches dozens
    of block graphs back to back (48 single-stream blocks per DiT call) overflows the runtime's
    execution queue ("Execution Queue Full"). Synchronizing every ``every`` launches keeps the
    queue bounded while still overlapping host dispatch with device work.
    """

    def __init__(self, device: torch.device, every: int = SYNC_EVERY):
        self.device, self.every, self.n = device, every, 0
        self._sync = None
        if device.type != "cpu" and every > 0:
            try:
                from libtorch_neuronx_lite._compiler import synchronize

                self._sync = synchronize
            except ImportError as exc:
                log_info(f"flux2: no Lite synchronize ({exc!r}); launches are not throttled")
                self._sync = None

    def tick(self, what: str = "") -> None:
        self.n += 1
        if DEBUG_LAUNCH:
            log_info(
                "launch %d %s (sync=%s)",
                self.n,
                what,
                self._sync is not None and self.n % self.every == 0,
            )
        if self._sync is not None and self.n % self.every == 0:
            self._sync(self.device)


def log_info(msg: str, *args) -> None:
    """Rank-0 progress line on stdout (the plugin's module loggers sit outside vLLM's logging config)."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[flux2] {msg % args if args else msg}", flush=True)
