# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 / FastH3 DiT for Neuron, tensor-parallel.

A re-implementation of diffusers' ``MiniMaxH3Transformer3DModel`` (vendored in ``_vendor/``) in the plugin's style:
raw ``nn.Parameter`` weights with ``vllm_neuron`` sharding loaders, explicit TP collectives and the attention
dispatch of :mod:`.attention`. Parameter names keep the checkpoint's module paths (``transformer_blocks.{i}.attn.
to_q.weight`` ...) so the checkpoint mapping is the identity except for the fused-SwiGLU split.

Sharding (``tp`` ranks):

* q / k / v: column-parallel by heads; per-head RMSNorm (``norm_q`` / ``norm_k``) replicated.
* attention out / FFN down: row-parallel + all-reduce. FFN ``net.0.proj`` holds ``[value; gate]`` halves, and each
  rank takes the matching slice of both.
* AdaLN modulation (``adaln_proj.linear``, 2688 -> 96768 per block, ~13B params in total): column-parallel over its
  output and all-gathered (a few rows per step), or (``adaln="host"``) not on the device at all: the host computes
  the per-step tables from the checkpoint and passes them in. Both are exact.
* everything else (patch / audio projections, timestep MLP, text embedder, output norm and heads) is replicated;
  the checkpoint's fp32 modules stay fp32.

The sequence is the contiguous ``[text | audio | video]`` layout of :mod:`.layout`, and the per-row AdaLN table
lookup becomes a per-segment broadcast: every row of a segment shares one ``(timestep, modality)`` row, so a
``segments`` tuple of ``(length, adaln_row, timestep_index)`` replaces the ``index_select`` over every row (which
would materialise a ``(L, 6 * hidden)`` tensor, 600 MB at 384x640).

Context parallelism (``cp`` ranks, vLLM-Omni's sequence-parallel group, set by ``ring_degree``): every rank keeps the
same TP shard of the weights and holds ``1 / cp`` of the packed sequence. The host plan (:class:`CPPlan`) orders the
rows, pads them to a multiple of ``cp`` with a zero row, and hands each rank its slice as graph INPUTS (row indices,
a per-row segment one-hot, the local RoPE rows), so every rank traces the same graph. Everything but attention is
per-row; attention all-gathers K/V over the CP group (and, for the dense path, drops the trailing pad keys before the
flash kernel). A VSA checkpoint runs in VSA slot order instead -- the residual stream itself is permuted into
``(tile, slot)`` order, so each rank owns whole tiles and the sparse mask is computed for its query tiles only. The
heads run per rank and only their 128-wide fp32 outputs are all-gathered. ``cp == 1`` is the TP-only graph unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from ._vendor.transformer_minimax_h3 import _apply_rotary_emb
from .attention import h3_attention
from .config import MiniMaxH3DiTConfig
from .vsa import TILE_ELEMS, VSAGeometry, vsa_attention, vsa_attention_cp

Segment = tuple[int, int, int]  # (num rows, AdaLN table row = t_idx * 3 + modality, timestep index)


def tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialized."""
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
    return size, rank, group


def cp_state() -> tuple[int, int, object]:
    """(cp_size, cp_rank, cp GroupCoordinator); (1, 0, None) without a context-parallel group (CP is vLLM-Omni's
    sequence-parallel group: ``ring_degree`` / ``sequence_parallel_size`` in the stage's parallel_config)."""
    try:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import get_cp_group

        g = get_cp_group()
        size, rank = g.world_size, g.rank_in_group
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    return size, rank, (g if size > 1 else None)


def register_groups(tp_size: int, cp_size: int) -> None:
    """Register the TP / CP partitions with the Neuron compiler's mesh registry so in-graph collectives legalize
    (``register_replica_groups``; on the trn2 64-core fabric these are the physical-mesh groups the worker built)."""
    if tp_size > 1 or cp_size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=tp_size, cp_size=cp_size)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """torch's ``F.rms_norm`` decomposition: fp32 statistics and weight product, cast back at the end."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        xf = xf * weight.float()
    return xf.to(x.dtype)


def _param(shape, dtype, fill: float | None = None) -> nn.Parameter:
    t = torch.empty(shape, dtype=dtype) if fill is None else torch.full(shape, fill, dtype=dtype)
    return nn.Parameter(t, requires_grad=False)


class _Linear(nn.Module):
    """A bare ``weight`` (+ ``bias``) holder: checkpoint layout ``[out, in]``."""

    def __init__(self, out_features: int, in_features: int, dtype: torch.dtype, bias: bool):
        super().__init__()
        self.weight = _param((out_features, in_features), dtype)
        self.bias = _param((out_features,), dtype) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


class _Norm(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.weight = _param((dim,), dtype, 1.0)


class _Attention(nn.Module):
    """``MiniMaxH3Attention`` with heads split over TP ranks."""

    def __init__(self, cfg: MiniMaxH3DiTConfig, tp: int, dtype: torch.dtype, vsa: bool = False):
        super().__init__()
        self.heads = cfg.num_attention_heads // tp
        self.head_dim = cfg.attention_head_dim
        local = self.heads * self.head_dim
        self.eps = cfg.qk_norm_eps
        self.to_q = _Linear(local, cfg.hidden_size, dtype, bias=False)
        self.to_k = _Linear(local, cfg.hidden_size, dtype, bias=False)
        self.to_v = _Linear(local, cfg.hidden_size, dtype, bias=False)
        self.norm_q = _Norm(self.head_dim, dtype)
        self.norm_k = _Norm(self.head_dim, dtype)
        self.to_out = nn.ModuleList([_Linear(cfg.hidden_size, local, dtype, bias=False)])
        # VSA students carry a learned compression gate (column-parallel by heads, like q/k/v)
        self.to_gate_compress = _Linear(local, cfg.hidden_size, dtype, bias=False) if vsa else None

    def forward(
        self,
        x,
        rotary_emb,
        reduce,
        vsa_geom: VSAGeometry | None = None,
        cp: _CPContext | None = None,
    ):
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.head_dim)
        k = self.to_k(x).view(b, s, self.heads, self.head_dim)
        v = self.to_v(x).view(b, s, self.heads, self.head_dim)
        q = rms_norm(q, self.norm_q.weight, self.eps)
        k = rms_norm(k, self.norm_k.weight, self.eps)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)
        gate = None
        if vsa_geom is not None and self.to_gate_compress is not None:
            gate = self.to_gate_compress(x).view(b, s, self.heads, self.head_dim)[0]
        if (
            cp is not None
        ):  # this rank's queries against every rank's keys (all-gathered over the CP group)
            k_all, v_all = cp.gather(k[0]), cp.gather(v[0])
            if vsa_geom is not None:
                out = vsa_attention_cp(q[0], k_all, v_all, gate, vsa_geom, *cp.vsa_local).reshape(
                    1, s, -1
                )
            else:  # the dense order pads at the end only: the first kv_len gathered rows are the real keys
                out = h3_attention(q, k_all[None, : cp.kv_len], v_all[None, : cp.kv_len])
            out = out.to(x.dtype)
        elif vsa_geom is not None:
            out = vsa_attention(q[0], k[0], v[0], gate, vsa_geom).reshape(1, s, -1).to(x.dtype)
        else:
            out = h3_attention(q, k, v).to(x.dtype)
        return reduce(self.to_out[0](out))


class _SwiGLU(nn.Module):
    """diffusers ``FeedForward(activation_fn="swiglu")``: ``net.0.proj`` = ``[value; gate]``, ``net.2`` down."""

    def __init__(self, cfg: MiniMaxH3DiTConfig, tp: int, dtype: torch.dtype):
        super().__init__()
        self.local_inner = cfg.ffn_dim // tp
        self.proj = _Linear(2 * self.local_inner, cfg.hidden_size, dtype, bias=False)
        self.down = _Linear(cfg.hidden_size, self.local_inner, dtype, bias=False)

    def forward(self, x, reduce):
        h, gate = self.proj(x).chunk(2, dim=-1)
        return reduce(self.down(h * F.silu(gate)))


class _RefinerBlock(nn.Module):
    def __init__(self, cfg: MiniMaxH3DiTConfig, tp: int, dtype: torch.dtype):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1 = _Norm(cfg.hidden_size, dtype)
        self.attn = _Attention(cfg, tp, dtype)
        self.norm2 = _Norm(cfg.hidden_size, dtype)
        self.ff = _SwiGLU(cfg, tp, dtype)

    def forward(self, x, reduce):
        x = x + self.attn(rms_norm(x, self.norm1.weight, self.eps), None, reduce)
        return x + self.ff(rms_norm(x, self.norm2.weight, self.eps), reduce)


class _Block(nn.Module):
    """``MiniMaxH3TransformerBlock``; AdaLN rows come in as ``mod`` = six ``(R, hidden)`` tables."""

    def __init__(
        self,
        cfg: MiniMaxH3DiTConfig,
        tp: int,
        dtype: torch.dtype,
        adaln_on_device: bool,
        vsa: bool = False,
    ):
        super().__init__()
        self.eps = cfg.norm_eps
        self.hidden = cfg.hidden_size
        self.norm1 = _Norm(cfg.hidden_size, dtype)
        self.attn = _Attention(cfg, tp, dtype, vsa=vsa)
        self.norm2 = _Norm(cfg.hidden_size, dtype)
        self.ff = _SwiGLU(cfg, tp, dtype)
        out = 6 * cfg.hidden_size * C.MODALITY_NUM
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = (
            _Linear(out // tp, cfg.time_embed_dim, dtype, bias=True) if adaln_on_device else None
        )

    def modulation(self, temb: torch.Tensor, gather) -> torch.Tensor:
        """``(T, time_embed_dim)`` fp32 temb -> ``(3T, 6 * hidden)`` table, row = t * 3 + modality. Uses the shared
        ``bake_linear_modulation`` rounding (act fp32 -> bf16 input -> fp32-accumulate matmul -> bf16, bias in bf16),
        so the on-device table equals the host-precomputed one (``adaln="host"``) bit for bit."""
        from vllm_omni_neuron.diffusion.layers.modulation_tables import bake_linear_modulation

        lin = self.adaln_proj.linear
        out = bake_linear_modulation(temb, lin.weight, lin.bias, compute_dtype=lin.weight.dtype)
        return gather(out).reshape(-1, 6 * self.hidden)  # (T, 18*hidden[/tp/cp]) -> gathered

    def forward(
        self, x, table, per_row, rotary_emb, reduce, vsa_geom=None, cp=None, sp_gather=None
    ):
        """``table``: ``(3T, 6 * hidden)``. ``per_row``: ``(3T, hidden)`` table -> ``(1, L, hidden)`` rows (a static
        per-segment broadcast, or under CP the local rows' segment one-hot times the three segment rows).
        ``sp_gather`` (sequence-parallel TP): ``x`` holds this TP rank's rows; the modulated activations are
        all-gathered before attention / FFN and ``reduce`` reduce-scatters their outputs back to these rows."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = table.chunk(6, dim=-1)
        gather = (lambda t: t) if sp_gather is None else sp_gather
        h = rms_norm(x, self.norm1.weight, self.eps)
        h = h * (1.0 + per_row(scale_msa)) + per_row(shift_msa)
        x = x + per_row(gate_msa) * self.attn(gather(h), rotary_emb, reduce, vsa_geom, cp)
        h = rms_norm(x, self.norm2.weight, self.eps)
        h = h * (1.0 + per_row(scale_mlp)) + per_row(shift_mlp)
        return x + per_row(gate_mlp) * self.ff(gather(h), reduce)


def segment_broadcast(segments, col: int = 1):
    """The TP-only per-row map: ``(R, hidden)`` -> ``(1, L, hidden)``, row ``seg[col]`` broadcast over each segment."""

    def per_row(t):
        return torch.cat(
            [t[seg[col] : seg[col] + 1].expand(seg[0], -1) for seg in segments], dim=0
        ).unsqueeze(0)

    return per_row


def onehot_rows(onehot: torch.Tensor, rows: tuple[int, int, int]):
    """The CP per-row map: ``onehot`` ``(L_local, 3)`` (text / audio / video; all-zero for a pad row) times the three
    segment rows of the table. Exactly one term per row, so the matmul reproduces the broadcast bit for bit."""

    def per_row(t):
        sel = torch.cat([t[r : r + 1] for r in rows], dim=0)  # (3, hidden)
        return torch.matmul(onehot.to(t.dtype), sel).unsqueeze(0)

    return per_row


class _CPContext:
    """What a CP attention layer needs inside the traced forward: the K/V gather, how many gathered rows are real
    keys (dense order), and this rank's VSA tile tensors (slot order)."""

    def __init__(self, gather, kv_len: int, vsa_local: tuple | None):
        self.gather, self.kv_len, self.vsa_local = gather, kv_len, vsa_local


@dataclass
class CPPlan:
    """Host-side context-parallel layout of the packed sequence over ``cp`` ranks.

    ``order`` lists, for every padded position, the packed row it holds (``L`` = the appended zero row); rank ``r``
    owns ``order[r * local_len:(r + 1) * local_len]``. Dense: rows in packed order, pads at the end. VSA: the slot
    order of the (tile-padded) VSA geometry, so each rank owns ``n_tiles / cp`` whole tiles. ``video_pos`` /
    ``audio_pos`` locate each video / audio row in the all-gathered head output.
    """

    cp: int
    local_len: int
    kv_len: int
    order: torch.Tensor  # (cp * local_len,) long
    seg: torch.Tensor  # (cp * local_len,) long: 0 text, 1 audio, 2 video, -1 pad
    video_pos: torch.Tensor
    audio_pos: torch.Tensor
    vsa_geom: VSAGeometry | None = None  # tile-padded, slot order (VSA checkpoints)

    @staticmethod
    def build(
        cp: int,
        n_text: int,
        n_audio: int,
        n_video: int,
        vsa_geom: VSAGeometry | None = None,
        row_multiple: int = 1,
    ) -> CPPlan:
        seq = n_text + n_audio + n_video
        seg_of_row = torch.cat(
            [
                torch.full((n,), i, dtype=torch.long)
                for i, n in enumerate((n_text, n_audio, n_video))
            ]
            + [torch.full((1,), -1, dtype=torch.long)]
        )  # + the zero row
        if vsa_geom is None:
            local = -(-seq // cp)
            local = (
                -(-local // row_multiple) * row_multiple
            )  # sequence-parallel TP splits the slice evenly
            order = torch.cat(
                [torch.arange(seq), torch.full((local * cp - seq,), seq, dtype=torch.long)]
            )
            pos = torch.arange(seq)
            kv_len, geom = seq, None
        else:
            geom = vsa_geom.pad_tiles(cp)
            order = geom.slot_to_row
            local = order.numel() // cp
            pos = geom.row_to_slot
            kv_len = order.numel()
        return CPPlan(
            cp,
            local,
            kv_len,
            order,
            seg_of_row[order],
            pos[n_text + n_audio :].clone(),
            pos[n_text : n_text + n_audio].clone(),
            geom,
        )

    def local(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank ``rank``'s (row indices, segment one-hot ``(local_len, 3)`` fp32)."""
        sl = slice(rank * self.local_len, (rank + 1) * self.local_len)
        seg = self.seg[sl]
        onehot = F.one_hot(seg.clamp(min=0), 3).float() * (seg >= 0).float()[:, None]
        return self.order[sl].clone(), onehot

    def vsa_local(self, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank ``rank``'s VSA tensors: slot validity ``(local_len,)``, tile sizes ``(local tiles,)`` (clamped to 1
        for pad tiles) and which of its tiles are prefix (dense-query) tiles."""
        g = self.vsa_geom
        ntl = self.local_len // TILE_ELEMS
        tiles = torch.arange(rank * ntl, (rank + 1) * ntl)
        valid = g.valid[rank * self.local_len : (rank + 1) * self.local_len].clone()
        # prefix flags as float (1.0 / 0.0): passed as a bool graph input they came out wrong on device
        return valid, g.tile_sizes[tiles].clamp(min=1.0), (tiles < g.n_prefix).float()


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`` (``max_period`` 10000)."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    emb = t.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1)


class NeuronMiniMaxH3Transformer(nn.Module):
    """``forward(text, audio_rows, video_rows, timestep, cos, sin[, ada_tables]) -> (video vel, audio vel)``.

    * ``text`` ``(1, N, text_dim)`` text-encoder hidden states; ``audio_rows`` ``(1, Na, 32)`` and ``video_rows``
      ``(1, Nv, 96)`` latent rows (fp32); ``timestep`` ``(2,)`` fp32 ``[t_video, t_audio]`` in ``[0, 1]``.
    * ``cos`` / ``sin`` ``(L, rotary_dim)`` fp32, from :meth:`.layout.H3Layout.rotary`.
    * ``ada_tables`` ``(num_layers, 3T, 6 * hidden)`` only with ``adaln="host"``.

    ``set_layout(n_text, n_audio, n_video)`` fixes the static segment structure before tracing.

    Under CP (``cp_size > 1``) ``cos`` / ``sin`` are this rank's rows and ``shard`` carries the rest of its slice;
    :meth:`shard_inputs` builds all three from the full RoPE tables.
    """

    def __init__(
        self,
        cfg: MiniMaxH3DiTConfig,
        dtype: torch.dtype = torch.bfloat16,
        adaln: str | None = None,
        vsa_sparsity: float | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        # VSA-H3 students (FastH3 8-Step-V2): sparsity from the checkpoint's inference contract; None = dense
        self.vsa_sparsity = vsa_sparsity
        self.vsa_geom: VSAGeometry | None = None
        self.dtype = dtype
        self.adaln = adaln or os.environ.get("MINIMAX_H3_ADALN", "device")
        if self.adaln not in ("device", "host"):
            raise ValueError(f"adaln must be 'device' or 'host', got {self.adaln!r}")
        self.tp_size, self.tp_rank, self.tp_group = tp_state()
        self.cp_size, self.cp_rank, self.cp_coord = cp_state()
        self.cp_group = self.cp_coord.device_group if self.cp_coord is not None else None
        register_groups(self.tp_size, self.cp_size)
        self.cp_plan: CPPlan | None = None
        # Sequence-parallel TP inside each CP slice (dense checkpoints; MINIMAX_H3_TP_SP=0 turns it off): the
        # residual stream, norms and AdaLN run on 1 / tp of the slice's rows; all-gather before the q/k/v and FFN
        # projections, reduce-scatter after the output and down projections (the same bytes as the all-reduce they
        # replace). TP=4 x CP=2 at 256x448: 0.51 -> 0.41 s per forward.
        self.sp = (
            os.environ.get("MINIMAX_H3_TP_SP", "1") == "1"
            and self.tp_size > 1
            and self.cp_size > 1
            and vsa_sparsity is None
        )
        tp = self.tp_size
        if cfg.num_attention_heads % tp or cfg.ffn_dim % tp:
            raise ValueError(
                f"tp={tp} must divide heads={cfg.num_attention_heads} and ffn_dim={cfg.ffn_dim}"
            )
        f32 = torch.float32
        h = cfg.hidden_size
        self.proj_in = _Linear(h, cfg.video_patch_dim, f32, bias=True)
        self.audio_proj_in = _Linear(h, cfg.audio_in_channels, f32, bias=True)
        self.context_embedder = _Linear(h, cfg.text_dim, dtype, bias=True)
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = _Linear(
            cfg.time_embed_hidden_dim, cfg.freq_dim, f32, bias=True
        )
        self.time_embedder.linear_2 = _Linear(
            cfg.time_embed_dim, cfg.time_embed_hidden_dim, f32, bias=True
        )
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList(
            _RefinerBlock(cfg, tp, dtype) for _ in range(cfg.num_refiner_layers)
        )
        self.token_refiner.final_norm = _Norm(h, dtype)
        on_dev = self.adaln == "device"
        vsa = vsa_sparsity is not None
        self.transformer_blocks = nn.ModuleList(
            _Block(cfg, tp, dtype, on_dev, vsa=vsa) for _ in range(cfg.num_layers)
        )
        self.norm_out = nn.Module()
        self.norm_out.norm = _Norm(h, dtype)
        self.norm_out.linear = _Linear(2 * h, cfg.time_embed_dim, dtype, bias=True)
        self.proj_out = _Linear(cfg.video_patch_dim, h, f32, bias=True)
        self.audio_proj_out = _Linear(cfg.audio_in_channels, h, f32, bias=True)
        self.segments: tuple[Segment, ...] | None = None
        self.num_text = self.num_audio = self.num_video = 0
        self._attach_weight_loaders()

    # -- layout --------------------------------------------------------------------------------------------------
    def set_layout(
        self,
        num_text: int,
        num_audio: int,
        num_video: int,
        video_grid: tuple[int, int, int] | None = None,
    ) -> None:
        """t2va: text rows take the video timestep (index 0), audio rows index 1, video rows index 0.
        ``video_grid`` = the video's ``(T, H/2, W/2)`` token grid, required for a VSA checkpoint."""
        self.num_text, self.num_audio, self.num_video = num_text, num_audio, num_video
        if self.vsa_sparsity is not None:
            if video_grid is None:
                raise ValueError("a VSA checkpoint needs the video token grid in set_layout()")
            dev = next(self.parameters()).device
            dense = tuple(n for n in (num_text, num_audio) if n > 0)
            self.vsa_geom = VSAGeometry.build(dense, video_grid, self.vsa_sparsity).to(dev)
        segs = [
            (num_text, 0 * C.MODALITY_NUM + C.TEXT_TAG, 0),
            (num_audio, 1 * C.MODALITY_NUM + C.AUDIO_TAG, 1),
            (num_video, 0 * C.MODALITY_NUM + C.VIDEO_TAG, 0),
        ]
        self.segments = tuple(s for s in segs if s[0] > 0)
        self.seg_table_rows = tuple(s[1] for s in segs)  # (text, audio, video) AdaLN table rows
        self.seg_time_rows = tuple(s[2] for s in segs)  # (text, audio, video) timestep index
        if self.cp_size > 1:
            dev = next(self.parameters()).device
            geom = self.vsa_geom.to("cpu") if self.vsa_geom is not None else None
            self.cp_plan = CPPlan.build(
                self.cp_size,
                num_text,
                num_audio,
                num_video,
                geom,
                row_multiple=self.tp_size if self.sp else 1,
            )
            if self.cp_plan.vsa_geom is not None:
                self.vsa_geom = self.cp_plan.vsa_geom.to(dev)
            self._video_pos = self.cp_plan.video_pos.to(dev)
            self._audio_pos = self.cp_plan.audio_pos.to(dev)

    def shard_inputs(self, cos: torch.Tensor, sin: torch.Tensor):
        """Full ``(L, rotary_dim)`` RoPE tables -> ``(cos, sin, shard)`` for this rank's forward (identity at
        ``cp == 1``). The shard tensors are forward INPUTS, so every CP rank traces the same graph."""
        if self.cp_size == 1:
            return cos, sin, None
        plan = self.cp_plan
        rows, onehot = plan.local(self.cp_rank)
        pad = lambda t: torch.cat([t, t.new_zeros((1, t.shape[1]))], dim=0).index_select(0, rows)  # noqa: E731
        shard = [rows, onehot.to(self.dtype)]
        if self.sp:  # this TP rank's share of the slice: the residual stream rows it owns between attention / FFN
            n = plan.local_len // self.tp_size
            sl = slice(self.tp_rank * n, (self.tp_rank + 1) * n)
            shard = [rows[sl].clone(), onehot[sl].to(self.dtype).clone()]
        if plan.vsa_geom is not None:
            shard.extend(plan.vsa_local(self.cp_rank))
        return pad(cos).contiguous(), pad(sin).contiguous(), tuple(shard)

    # -- collectives ---------------------------------------------------------------------------------------------
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            dist.all_reduce(x, group=self.tp_group)
        return x

    def _gather_last(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather the column-parallel AdaLN output along its last dim."""
        if self.tp_size == 1:
            return x
        t, local = x.shape
        x = x.contiguous()
        if x.device.type == "cpu":  # gloo (CPU tests) has no all_gather_into_tensor
            parts = [torch.empty_like(x) for _ in range(self.tp_size)]
            dist.all_gather(parts, x, group=self.tp_group)
            return torch.cat(parts, dim=-1)
        out = x.new_empty(
            (self.tp_size * t, local)
        )  # concatenated along dim 0, as the traced op expects
        dist.all_gather_into_tensor(out, x, group=self.tp_group)
        return out.view(self.tp_size, t, local).permute(1, 0, 2).reshape(t, self.tp_size * local)

    def _tp_reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """``(1, n, h)`` partial sums -> this TP rank's ``(1, n / tp, h)`` rows of the sum."""
        _, n, h = x.shape
        m = n // self.tp_size
        if (
            x.device.type == "cpu"
        ):  # gloo (CPU tests) has no reduce_scatter: reduce, then keep this rank's rows
            x = self._reduce(x.clone())
            return x[:, self.tp_rank * m : (self.tp_rank + 1) * m].contiguous()
        out = x.new_empty((m, h))
        dist.reduce_scatter_tensor(
            out, x.reshape(n, h).contiguous(), op=dist.ReduceOp.SUM, group=self.tp_group
        )
        return out.reshape(1, m, h)

    def _tp_gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        """``(1, m, h)`` per TP rank -> ``(1, tp * m, h)``, in TP rank order."""
        _, m, h = x.shape
        x = x.reshape(m, h).contiguous()
        if x.device.type == "cpu":
            parts = [torch.empty_like(x) for _ in range(self.tp_size)]
            dist.all_gather(parts, x, group=self.tp_group)
            return torch.cat(parts, dim=0).unsqueeze(0)
        out = x.new_empty((self.tp_size * m, h))
        dist.all_gather_into_tensor(out, x, group=self.tp_group)
        return out.unsqueeze(0)

    def _cp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather ``(n, ...)`` over the CP group along dim 0 (rank order) -> ``(cp * n, ...)``."""
        x = x.contiguous()
        if x.device.type == "cpu":  # gloo (CPU tests)
            parts = [torch.empty_like(x) for _ in range(self.cp_size)]
            dist.all_gather(parts, x, group=self.cp_group)
            return torch.cat(parts, dim=0)
        out = x.new_empty((self.cp_size * x.shape[0], *x.shape[1:]))
        dist.all_gather_into_tensor(out, x, group=self.cp_group)
        return out

    # -- weights -------------------------------------------------------------------------------------------------
    def _attach_weight_loaders(self) -> None:
        if self.tp_size == 1:
            return
        from vllm_neuron.utils.weight_loader import (
            SafetensorsWeightLoader,
            set_weight_loader,
            sharding_weight_loader,
        )

        tp = self.tp_size

        def shard(p, dim):
            set_weight_loader(
                p, sharding_weight_loader(shard_dim=dim, shard_size=p.shape[dim], num_shards=tp)
            )

        def swiglu(p):  # [value; gate] halves: this rank's slice of each
            n = p.shape[0] // 2

            def transform(slices, rank):
                full = slices[0]
                inner = full.get_shape()[0] // 2
                r = rank % tp
                return torch.cat(
                    [full[r * n : (r + 1) * n], full[inner + r * n : inner + (r + 1) * n]], dim=0
                )

            set_weight_loader(p, SafetensorsWeightLoader(transform=transform))

        for blk in [*self.token_refiner.refiner_blocks, *self.transformer_blocks]:
            for lin in (blk.attn.to_q, blk.attn.to_k, blk.attn.to_v):
                shard(lin.weight, 0)
            if getattr(blk.attn, "to_gate_compress", None) is not None:
                shard(blk.attn.to_gate_compress.weight, 0)
            shard(blk.attn.to_out[0].weight, 1)
            swiglu(blk.ff.proj.weight)
            shard(blk.ff.down.weight, 1)
            lin = getattr(getattr(blk, "adaln_proj", None), "linear", None)
            if lin is not None:
                shard(lin.weight, 0)
                shard(lin.bias, 0)

    def checkpoint_mappings(self) -> dict[str, str]:
        """Parameter name -> checkpoint key (diffusers naming)."""
        m = {}
        for name, _ in self.named_parameters():
            key = name.replace(".ff.proj.", ".ff.net.0.proj.").replace(".ff.down.", ".ff.net.2.")
            m[name] = key
        return m

    def load_weights(self, transformer_dir: str, device: torch.device | str = "cpu") -> None:
        """Load this rank's shard (needs torch.distributed initialized, as under vLLM)."""
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        ckpt = SafetensorsCheckpoint(transformer_dir)
        result = ckpt.load_sharded_pipelined(
            self.tp_rank, self.tp_size, self, self.checkpoint_mappings(), torch.device(device)
        )
        self.load_state_dict(result.state_dict, strict=True, assign=True)
        self._shard_adaln_over_cp()

    def _shard_adaln_over_cp(self) -> None:
        """Every CP rank keeps only 1 / cp of its TP shard of each block's AdaLN projection (``MINIMAX_H3_ADALN_CP=0``
        keeps the whole shard). The table is the same on all CP ranks, so each computes one slice and the slices are
        all-gathered over CP: 1 / cp of the weight reads per block, and of its HBM (2.8 GB per core at TP=8 x CP=8)."""
        self.adaln_cp = (
            os.environ.get("MINIMAX_H3_ADALN_CP", "1") == "1"
            and self.cp_size > 1
            and self.adaln == "device"
        )
        if not self.adaln_cp:
            return
        for blk in self.transformer_blocks:
            lin = blk.adaln_proj.linear
            m = lin.weight.shape[0] // self.cp_size
            sl = slice(self.cp_rank * m, (self.cp_rank + 1) * m)
            lin.weight = nn.Parameter(lin.weight.data[sl].clone(), requires_grad=False)
            lin.bias = nn.Parameter(lin.bias.data[sl].clone(), requires_grad=False)

    def _gather_adaln(self, x: torch.Tensor) -> torch.Tensor:
        """The AdaLN projection's column slice -> the full ``(T, 18 * hidden)`` row: over CP first (with
        ``MINIMAX_H3_ADALN_CP=1``), then over TP."""
        if getattr(self, "adaln_cp", False):
            t, local = x.shape
            x = x.contiguous()
            if x.device.type == "cpu":  # gloo (CPU tests)
                parts = [torch.empty_like(x) for _ in range(self.cp_size)]
                dist.all_gather(parts, x, group=self.cp_group)
                x = torch.cat(parts, dim=-1)
            else:
                out = x.new_empty((self.cp_size * t, local))
                dist.all_gather_into_tensor(out, x, group=self.cp_group)
                x = (
                    out.view(self.cp_size, t, local)
                    .permute(1, 0, 2)
                    .reshape(t, self.cp_size * local)
                )
        return self._gather_last(x)

    # -- host helpers --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def temb(self, timestep: torch.Tensor) -> torch.Tensor:
        return self._temb(timestep)

    def _temb(self, timestep: torch.Tensor) -> torch.Tensor:
        """``(T,)`` -> ``(T, time_embed_dim)`` fp32 (the reference's fp32 timestep MLP)."""
        e = _timestep_embedding(timestep, self.cfg.freq_dim)
        e = F.linear(e, self.time_embedder.linear_1.weight, self.time_embedder.linear_1.bias)
        return F.linear(
            F.silu(e), self.time_embedder.linear_2.weight, self.time_embedder.linear_2.bias
        )

    # -- forward -------------------------------------------------------------------------------------------------
    def forward(
        self, text, audio_rows, video_rows, timestep, cos, sin, ada_tables=None, shard=None
    ):
        if self.segments is None:
            raise RuntimeError("call set_layout() before the first forward")
        if self.cp_size > 1:
            return self._forward_cp(
                text, audio_rows, video_rows, timestep, cos, sin, ada_tables, shard
            )
        parts = self._embed(text, audio_rows, video_rows)
        x = torch.cat(
            [
                t
                for t, n in zip(parts, (self.num_text, self.num_audio, self.num_video), strict=True)
                if n > 0
            ],
            dim=1,
        )
        temb = self._temb(timestep)
        per_row = segment_broadcast(self.segments)
        for i, blk in enumerate(self.transformer_blocks):
            table = (
                blk.modulation(temb, self._gather_adaln) if ada_tables is None else ada_tables[i]
            )
            x = blk(x, table, per_row, (cos, sin), self._reduce, self.vsa_geom)
        return self.head(x, temb)

    def _embed(self, text, audio_rows, video_rows):
        """Text (through the replicated token refiner), audio and video rows -> their hidden states."""
        dt, reduce = self.dtype, self._reduce
        x_text = self.context_embedder(text.to(dt))
        for blk in self.token_refiner.refiner_blocks:
            x_text = blk(x_text, reduce)
        x_text = rms_norm(x_text, self.token_refiner.final_norm.weight, self.cfg.final_norm_eps)
        return (
            x_text,
            self.audio_proj_in(audio_rows.float()).to(dt),
            self.proj_in(video_rows.float()).to(dt),
        )

    def _embed_local(self, text, audio_rows, video_rows, rows, onehot):
        """This CP rank's hidden states: its audio / video rows are selected at their narrow input width (32 / 96)
        BEFORE the projections, so no rank embeds the whole sequence (at 768p that is a 37k x 5376 fp32 product,
        ~0.8 GB per rank, and a fusion group the compiler cannot fit). Text (a few dozen rows) runs the refiner in
        full. ``rows`` index the packed ``[text | audio | video]`` order (``L`` = pad); the one-hot keeps exactly one
        segment per row, so the result equals the full embedding's ``index_select`` bit for bit."""
        nt, na, nv = self.num_text, self.num_audio, self.num_video
        dt, reduce = self.dtype, self._reduce
        oh = onehot.to(dt)[None, :, :, None]  # (1, n, 3, 1)
        x = None
        if nt > 0:
            x_text = self.context_embedder(text.to(dt))
            for blk in self.token_refiner.refiner_blocks:
                x_text = blk(x_text, reduce)
            x_text = rms_norm(x_text, self.token_refiner.final_norm.weight, self.cfg.final_norm_eps)
            x_text = torch.cat([x_text, x_text.new_zeros((1, 1, x_text.shape[-1]))], dim=1)
            x = x_text.index_select(1, torch.where(rows < nt, rows, nt)) * oh[:, :, 0]
        for k, (n, off, src, proj) in enumerate(
            ((na, nt, audio_rows, self.audio_proj_in), (nv, nt + na, video_rows, self.proj_in)),
            start=1,
        ):
            if n == 0:
                continue
            idx = rows - off
            idx = torch.where((idx >= 0) & (idx < n), idx, n)
            src = torch.cat(
                [src.float(), src.new_zeros((1, 1, src.shape[-1]), dtype=torch.float32)], dim=1
            )
            part = proj(src.index_select(1, idx)).to(dt) * oh[:, :, k]
            x = part if x is None else x + part
        return x

    def _forward_cp(self, text, audio_rows, video_rows, timestep, cos, sin, ada_tables, shard):
        """CP forward: this rank's ``local_len`` rows (``shard`` = row indices, segment one-hot[, VSA tile
        tensors]) through every block; the heads' outputs are all-gathered and put back in packed row order."""
        rows, onehot, *vsa_local = shard
        x = self._embed_local(
            text, audio_rows, video_rows, rows, onehot
        )  # (1, local_len [/ tp], hidden)
        temb = self._temb(timestep)
        cp = _CPContext(
            self._cp_gather, self.cp_plan.kv_len, tuple(vsa_local) if vsa_local else None
        )
        per_row = onehot_rows(onehot, self.seg_table_rows)
        reduce = self._tp_reduce_scatter if self.sp else self._reduce
        sp_gather = self._tp_gather_rows if self.sp else None
        for i, blk in enumerate(self.transformer_blocks):
            table = (
                blk.modulation(temb, self._gather_adaln) if ada_tables is None else ada_tables[i]
            )
            x = blk(x, table, per_row, (cos, sin), reduce, self.vsa_geom, cp, sp_gather)
        lin = self.norm_out.linear
        shift, scale = F.linear(F.silu(temb).to(lin.weight.dtype), lin.weight, lin.bias).chunk(
            2, dim=-1
        )  # (T, h)
        rows_t = onehot_rows(onehot, self.seg_time_rows)
        h = rms_norm(x, self.norm_out.norm.weight, self.cfg.final_norm_eps)
        h = (h * rows_t(1.0 + scale) + rows_t(shift)).float()
        out = torch.cat([self.proj_out(h[0]), self.audio_proj_out(h[0])], dim=-1)  # (n, 128)
        if self.sp:
            out = self._tp_gather_rows(out[None])[0]
        out = self._cp_gather(out)  # (cp * local_len, 128)
        nv = self.cfg.video_patch_dim
        video = out.index_select(0, self._video_pos)[:, :nv].unsqueeze(0)
        audio = out.index_select(0, self._audio_pos)[:, nv:].unsqueeze(0)
        return video, audio

    def head(self, x: torch.Tensor, temb: torch.Tensor):
        """``norm_out`` (shift/scale per timestep row) + the fp32 heads over the audio and video segments."""
        lin = self.norm_out.linear
        shift, scale = F.linear(F.silu(temb).to(lin.weight.dtype), lin.weight, lin.bias).chunk(
            2, dim=-1
        )  # (T, h)
        h = rms_norm(x, self.norm_out.norm.weight, self.cfg.final_norm_eps)
        rows = torch.cat(
            [(1.0 + scale[ti : ti + 1]).expand(n, -1) for n, _, ti in self.segments], 0
        ).unsqueeze(0)
        offs = torch.cat(
            [shift[ti : ti + 1].expand(n, -1) for n, _, ti in self.segments], 0
        ).unsqueeze(0)
        h = (h * rows + offs).float()
        n0, n1 = self.num_text, self.num_text + self.num_audio
        audio = self.audio_proj_out(h[:, n0:n1])
        video = self.proj_out(h[:, n1:])
        return video, audio


def host_adaln_tables(
    transformer_dir: str,
    cfg: MiniMaxH3DiTConfig,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str | None = None,
):
    """``adaln="host"``: the blocks' AdaLN modulation computed on the host from the checkpoint, via the shared
    ``ModulationTables``. ``tables(temb)`` -> ``(num_layers, 3T, 6 * hidden)`` matching the device path's rounding
    (bake act in fp32 -> bf16 input -> fp32 matmul -> bf16), streaming each block's weights one at a time, cached
    per distinct ``temb`` in memory and (optionally) on disk — so a fixed-schedule sampler bakes each table once.
    """
    import glob
    import json as _json

    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.layers.modulation_tables import ModulationTables

    idx = glob.glob(os.path.join(transformer_dir, "*.safetensors.index.json"))
    if idx:
        with open(idx[0]) as f:
            weight_map = _json.load(f)["weight_map"]
    else:
        fn = os.path.basename(glob.glob(os.path.join(transformer_dir, "*.safetensors"))[0])
        with safe_open(os.path.join(transformer_dir, fn), "pt") as f:
            weight_map = {k: fn for k in f.keys()}
    handles: dict = {}

    def get_tensor(key: str) -> torch.Tensor:
        fn = weight_map[key]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(transformer_dir, fn), "pt")
        return handles[fn].get_tensor(key)

    fp = f"minimax_h3_adaln:{os.path.abspath(transformer_dir)}:{cfg.num_layers}:{dtype}"
    return ModulationTables.from_checkpoint(
        get_tensor,
        "transformer_blocks.{}.adaln_proj.linear",
        cfg.num_layers,
        compute_dtype=dtype,
        fingerprint=fp,
        cache_dir=cache_dir,
        out_shape=lambda y: y.reshape(-1, 6 * cfg.hidden_size),
    )
