# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo-1.5 diffusion transformer (DiT) for Neuron.

A re-implementation of diffusers' ``HunyuanVideo15Transformer3DModel`` (8.3B: 54 dual-stream
MMDiT blocks, hidden 2048 = 16 heads x 128) in the plugin's raw-parameter style:

* **Same parameter names as the checkpoint.** The module tree mirrors diffusers' (tiny
  ``_Linear`` / ``_Norm`` holders instead of ``nn.Linear`` / ``nn.LayerNorm``), so
  ``named_parameters()`` *is* the safetensors key set and the sharded loader needs no mapping.
* **TP shards heads.** Per block: ``to_q/k/v`` and ``add_q/k/v_proj`` column-parallel (by head),
  ``to_out.0`` / ``to_add_out`` row-parallel + all-reduce, both MLPs column/row-parallel. Every
  row-parallel bias is added once, after the all-reduce. The AdaLN projections, the token
  refiner and the embedders are small or per-token-cheap and stay replicated.
* **Fixed shapes.** Upstream compacts the encoder sequence to the valid tokens (a
  prompt-dependent length) and builds the order ``[image, byT5, MLLM]`` with boolean indexing.
  Here the host computes a gather index that puts every valid token first (same order as
  upstream) followed by padding, and the padded keys are masked with an additive bias. Attention
  is permutation-invariant over keys and encoder tokens carry no RoPE, so the video output is
  the same function as upstream's; the padded encoder *rows* differ but never reach the output.
* **T2V skips the image stream.** For T2V upstream feeds 729 all-zero SigLIP tokens whose mask is
  all-false; they are masked keys, so the graph simply omits them.

The forward is split into ``prologue`` (embeddings, token refiner, encoder assembly),
``run_blocks`` (any contiguous slice of the 54 blocks, so the DiT can be compiled as N-block
graphs) and ``epilogue`` (final AdaLN, projection, unpatchify).
"""

from __future__ import annotations

import json
import logging
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .attention import hv15_attention, key_padding_bias

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# config / TP plumbing
# ---------------------------------------------------------------------------------------------
class HV15Config:
    """The subset of ``transformer/config.json`` the DiT needs."""

    def __init__(self, cfg: dict):
        self.in_channels = int(cfg.get("in_channels", 65))
        self.out_channels = int(cfg.get("out_channels", 32))
        self.num_heads = int(cfg.get("num_attention_heads", 16))
        self.head_dim = int(cfg.get("attention_head_dim", 128))
        self.num_layers = int(cfg.get("num_layers", 54))
        self.num_refiner_layers = int(cfg.get("num_refiner_layers", 2))
        self.mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
        self.patch_size = int(cfg.get("patch_size", 1))
        self.patch_size_t = int(cfg.get("patch_size_t", 1))
        self.text_embed_dim = int(cfg.get("text_embed_dim", 3584))
        self.text_embed_2_dim = int(cfg.get("text_embed_2_dim", 1472))
        self.image_embed_dim = int(cfg.get("image_embed_dim", 1152))
        self.rope_theta = float(cfg.get("rope_theta", 256.0))
        self.rope_axes_dim = [int(x) for x in cfg.get("rope_axes_dim", [16, 56, 56])]
        self.use_meanflow = bool(cfg.get("use_meanflow", False))
        self.task_type = str(cfg.get("task_type", "t2v"))
        if cfg.get("qk_norm", "rms_norm") != "rms_norm":
            raise NotImplementedError(f"qk_norm={cfg.get('qk_norm')!r}")
        if self.use_meanflow:
            raise NotImplementedError(
                "HunyuanVideo-1.5 on Neuron: meanflow (distilled SR) checkpoints are not supported yet"
            )
        # sparse (SSTA) checkpoints, e.g. 720p_i2v_distilled_sparse. HV15_ATTN_MODE: auto (the checkpoint's
        # attention), dense (raster layout, dense masked attention), dense_tiles (SSTA's tile layout and text
        # rule with every key tile kept: the dense baseline of the sparse kernel)
        import dataclasses

        from .ssta import SSTAParams

        mode = os.environ.get("HV15_ATTN_MODE", "auto")
        self.ssta = SSTAParams.from_config(cfg)
        if mode == "dense":
            self.ssta = None
        elif mode == "dense_tiles" and self.ssta is not None:
            self.ssta = dataclasses.replace(self.ssta, dense=True)
        self.hidden = self.num_heads * self.head_dim
        self.mlp_hidden = int(self.hidden * self.mlp_ratio)
        if sum(self.rope_axes_dim) != self.head_dim:
            raise ValueError(
                f"rope_axes_dim {self.rope_axes_dim} must sum to head_dim {self.head_dim}"
            )

    @classmethod
    def from_model_dir(cls, model_path: str) -> HV15Config:
        with open(os.path.join(local_model_dir(model_path), "transformer", "config.json")) as f:
            return cls(json.load(f))


def local_model_dir(model: str) -> str:
    """A local directory for ``model``: itself if it is one, else the Hugging Face snapshot of that repo
    id (downloaded once, or found in the HF cache with ``HF_HUB_OFFLINE=1``)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialized.

    Under TP>1 the group's partition is also registered with the Neuron compiler's mesh registry
    so the in-graph all-reduces can be legalized (see ``register_replica_groups``).
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
    cp_size = cp_state()[0]
    if size > 1 or cp_size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=cp_size)
    return size, rank, group


def group_all_gather(coord, x: torch.Tensor, dim: int) -> torch.Tensor:
    """``coord.all_gather(x, dim)`` with the parts in the COORDINATOR's rank order (``coord.ranks``,
    which is ``rank_in_group`` order and decides each rank's CP slice / CFG branch).

    The trn2 physical-mesh layouts build groups from unsorted rank lists (e.g. ``[12, 8]``). On the
    NeuronCores the in-graph collective uses the registered replica groups, which keep that list
    order. A host (gloo) group from ``torch.distributed.new_group`` is sorted, so on CPU the parts
    come back as ``[8, 12]``; they are reordered here, which makes the CPU path (the unit tests)
    match the device."""
    out = coord.all_gather(x.contiguous(), dim=dim)
    if x.device.type != "cpu" or list(coord.ranks) == sorted(coord.ranks):
        return out
    parts = out.chunk(coord.world_size, dim=dim)
    c10d_order = sorted(coord.ranks)
    return torch.cat([parts[c10d_order.index(r)] for r in coord.ranks], dim=dim)


def cp_state() -> tuple[int, int, object]:
    """(cp_size, cp_rank, cp_group) of vLLM-Omni's sequence-parallel group (``ring_degree`` in the
    stage config); (1, 0, None) when it is absent. ``cp_group`` is the GroupCoordinator."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        g = get_sp_group()
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    if g is None or g.world_size <= 1:
        return 1, 0, None
    return g.world_size, g.rank_in_group, g


def _p(*shape, dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class _Linear(nn.Module):
    """``weight [out, in]`` (+ ``bias [out]``); ``shard`` = 0 column-parallel, 1 row-parallel."""

    def __init__(self, fan_in, fan_out, dtype, bias=True, shard=None, tp=1):
        super().__init__()
        if shard == 0:
            fan_out //= tp
        elif shard == 1:
            fan_in //= tp
        self.shard = shard
        self.weight = _p(fan_out, fan_in, dtype=dtype)
        self.bias = _p(fan_out, dtype=dtype) if bias else None

    def forward(self, x):
        # a row-parallel bias is added by the caller after the all-reduce
        return F.linear(x, self.weight, None if self.shard == 1 else self.bias)


class _Norm(nn.Module):
    def __init__(self, dim, dtype, bias=True):
        super().__init__()
        self.weight = _p(dim, dtype=dtype)
        self.bias = _p(dim, dtype=dtype) if bias else None


class _FFProj(nn.Module):  # diffusers GELU/LinearActivation keep the linear under ``.proj``
    def __init__(self, fan_in, fan_out, dtype, shard=None, tp=1):
        super().__init__()
        self.proj = _Linear(fan_in, fan_out, dtype, shard=shard, tp=tp)


class _FeedForward(nn.Module):
    """diffusers ``FeedForward`` naming: ``net.0.proj`` -> act -> ``net.2``."""

    def __init__(self, dim, inner, dtype, tp=1, sharded=False):
        super().__init__()
        self.net = nn.ModuleList(
            [
                _FFProj(dim, inner, dtype, shard=0 if sharded else None, tp=tp),
                nn.Identity(),
                _Linear(inner, dim, dtype, shard=1 if sharded else None, tp=tp),
            ]
        )


# ---------------------------------------------------------------------------------------------
# math helpers (diffusers semantics)
# ---------------------------------------------------------------------------------------------
def gelu(x, approximate: str = "none"):
    """The aten op directly: torch-neuronx / Lite wrap ``F.gelu`` in a Python function Dynamo
    cannot trace with ``fullgraph=True`` (the diffusion worker restores it; standalone runs do not)."""
    return torch.ops.aten.gelu.default(x, approximate=approximate)


def layer_norm(x, eps, norm: _Norm | None = None):
    w = None if norm is None else norm.weight.float()
    b = None if norm is None or norm.bias is None else norm.bias.float()
    return F.layer_norm(x.float(), (x.shape[-1],), w, b, eps).to(x.dtype)


def rms_norm(x, weight, eps=1e-6):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(x.dtype) * weight


def timestep_proj(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``, fp32."""
    half = dim // 2
    exponent = (
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * torch.exp(exponent)[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def apply_rope(x, cos, sin):
    """diffusers ``apply_rotary_emb(use_real=True, use_real_unbind_dim=-1)`` on ``[B, S, H, D]``;
    ``cos``/``sin`` ``[S, D]`` fp32 (each frequency repeated twice, interleaved)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos[None, :, None] + x_rot.float() * sin[None, :, None]).to(x.dtype)


def rope_tables(cfg: HV15Config, t: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side video RoPE ``(cos, sin)`` ``[t*h*w, head_dim]`` fp32 (post-patch grid sizes),
    identical to diffusers ``HunyuanVideo15RotaryPosEmbed`` (``get_1d_rotary_pos_embed``)."""
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    grids = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float32) for n in (t, h, w)], indexing="ij"
    )
    cos, sin = [], []
    for i in range(3):
        c, s = get_1d_rotary_pos_embed(
            cfg.rope_axes_dim[i], grids[i].reshape(-1), cfg.rope_theta, use_real=True
        )
        cos.append(c)
        sin.append(s)
    return torch.cat(cos, dim=1).float().contiguous(), torch.cat(sin, dim=1).float().contiguous()


# ---------------------------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------------------------
class _AdaNormZero(nn.Module):  # diffusers AdaLayerNormZero: linear(silu(temb)) -> 6 chunks
    def __init__(self, h, dtype):
        super().__init__()
        self.linear = _Linear(h, 6 * h, dtype)


class _BlockAttn(nn.Module):
    def __init__(self, cfg: HV15Config, dtype, tp):
        super().__init__()
        h, d = cfg.hidden, cfg.head_dim
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            setattr(self, name, _Linear(h, h, dtype, shard=0, tp=tp))
        self.to_out = nn.ModuleList([_Linear(h, h, dtype, shard=1, tp=tp)])
        self.to_add_out = _Linear(h, h, dtype, shard=1, tp=tp)
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            setattr(self, name, _Norm(d, dtype, bias=False))


class HV15TransformerBlock(nn.Module):
    """One dual-stream block (diffusers ``HunyuanVideo15TransformerBlock``), TP-sharded."""

    def __init__(self, cfg: HV15Config, dtype, tp: int):
        super().__init__()
        self.cfg, self.tp = cfg, tp
        self.n_heads = cfg.num_heads // tp
        h = cfg.hidden
        self.norm1 = _AdaNormZero(h, dtype)
        self.norm1_context = _AdaNormZero(h, dtype)
        self.attn = _BlockAttn(cfg, dtype, tp)
        self.ff = _FeedForward(h, cfg.mlp_hidden, dtype, tp=tp, sharded=True)
        self.ff_context = _FeedForward(h, cfg.mlp_hidden, dtype, tp=tp, sharded=True)

    def _reduce(self, x, group):
        if self.tp > 1:
            dist.all_reduce(x, group=group)
        return x

    def _ff(self, ff: _FeedForward, x, group):
        y = gelu(ff.net[0].proj(x), approximate="tanh")
        return self._reduce(ff.net[2](y), group) + ff.net[2].bias

    def forward(self, x, enc, temb_act, cos, sin, key_bias, group=None, cp=None, ssta=None):
        """``cp`` (GroupCoordinator) = context parallel: ``x`` holds this rank's slice of the
        (padded) video tokens, the encoder tokens are replicated; the video K/V are all-gathered so
        every query attends the whole sequence (``key_bias`` covers ``[all video | encoder]``).

        ``ssta`` = ``(SSTAStatic, slot_valid, win, tkeep, tq_idx)``: sparse checkpoints; ``x`` is then in
        the tile-major slot order (:mod:`.ssta`) and ``key_bias`` is unused."""
        b, sv, _ = x.shape
        se = enc.shape[1]
        d, nh, eps = self.cfg.head_dim, self.n_heads, 1e-6
        a = self.attn
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.norm1.linear(temb_act).chunk(6, dim=-1)
        c_sh_msa, c_sc_msa, c_g_msa, c_sh_mlp, c_sc_mlp, c_g_mlp = self.norm1_context.linear(
            temb_act
        ).chunk(6, dim=-1)

        nx = layer_norm(x, eps) * (1 + sc_msa[:, None]) + sh_msa[:, None]
        nc = layer_norm(enc, eps) * (1 + c_sc_msa[:, None]) + c_sh_msa[:, None]

        q = rms_norm(a.to_q(nx).view(b, sv, nh, d), a.norm_q.weight)
        k = rms_norm(a.to_k(nx).view(b, sv, nh, d), a.norm_k.weight)
        v = a.to_v(nx).view(b, sv, nh, d)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        eq = rms_norm(a.add_q_proj(nc).view(b, se, nh, d), a.norm_added_q.weight)
        ek = rms_norm(a.add_k_proj(nc).view(b, se, nh, d), a.norm_added_k.weight)
        ev = a.add_v_proj(nc).view(b, se, nh, d)
        if ssta is not None:
            from .ssta import ssta_block_attention

            ox, oc = ssta_block_attention(q, k, v, eq, ek, ev, *ssta, cp=cp)
            ox, oc = ox.reshape(b, sv, nh * d), oc.reshape(b, se, nh * d)
        else:
            if cp is not None:
                k = group_all_gather(cp, k, dim=1)
                v = group_all_gather(cp, v, dim=1)
            qq = torch.cat([q, eq], dim=1).transpose(1, 2)
            kk = torch.cat([k, ek], dim=1).transpose(1, 2)
            vv = torch.cat([v, ev], dim=1).transpose(1, 2)
            o = (
                hv15_attention(qq, kk, vv, d**-0.5, key_bias=key_bias)
                .transpose(1, 2)
                .reshape(b, sv + se, nh * d)
            )
            ox, oc = o[:, :sv], o[:, sv:]
        ax = self._reduce(a.to_out[0](ox), group) + a.to_out[0].bias
        ac = self._reduce(a.to_add_out(oc), group) + a.to_add_out.bias

        x = x + ax * g_msa[:, None]
        enc = enc + ac * c_g_msa[:, None]
        nx = layer_norm(x, eps) * (1 + sc_mlp[:, None]) + sh_mlp[:, None]
        nc = layer_norm(enc, eps) * (1 + c_sc_mlp[:, None]) + c_sh_mlp[:, None]
        x = x + g_mlp[:, None] * self._ff(self.ff, nx, group)
        enc = enc + c_g_mlp[:, None] * self._ff(self.ff_context, nc, group)
        return x, enc


class _RefinerAttn(nn.Module):
    def __init__(self, h, dtype):
        super().__init__()
        self.to_q, self.to_k, self.to_v = (_Linear(h, h, dtype) for _ in range(3))
        self.to_out = nn.ModuleList([_Linear(h, h, dtype)])


class _RefinerAdaNorm(nn.Module):
    def __init__(self, h, dtype):
        super().__init__()
        self.linear = _Linear(h, 2 * h, dtype)


class _RefinerBlock(nn.Module):
    def __init__(self, cfg: HV15Config, dtype):
        super().__init__()
        h = cfg.hidden
        self.norm1 = _Norm(h, dtype)
        self.attn = _RefinerAttn(h, dtype)
        self.norm2 = _Norm(h, dtype)
        self.ff = _FeedForward(h, int(h * 4.0), dtype)
        self.norm_out = _RefinerAdaNorm(h, dtype)


class _TimestepEmbedding(nn.Module):
    def __init__(self, fan_in, h, dtype):
        super().__init__()
        self.linear_1 = _Linear(fan_in, h, dtype)
        self.linear_2 = _Linear(h, h, dtype)

    def forward(self, x, act=F.silu):
        return self.linear_2(act(self.linear_1(x)))


class _TimeTextEmbed(nn.Module):
    def __init__(self, h, text_dim, dtype):
        super().__init__()
        self.timestep_embedder = _TimestepEmbedding(256, h, dtype)
        self.text_embedder = _TimestepEmbedding(
            text_dim, h, dtype
        )  # PixArtAlphaTextProjection(silu)


class _TokenRefinerStack(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            _RefinerBlock(cfg, dtype) for _ in range(cfg.num_refiner_layers)
        )


class _ContextEmbedder(nn.Module):  # diffusers HunyuanVideo15TokenRefiner
    def __init__(self, cfg: HV15Config, dtype):
        super().__init__()
        h = cfg.hidden
        self.time_text_embed = _TimeTextEmbed(h, cfg.text_embed_dim, dtype)
        self.proj_in = _Linear(cfg.text_embed_dim, h, dtype)
        self.token_refiner = _TokenRefinerStack(cfg, dtype)


class _ByT5Proj(nn.Module):
    def __init__(self, fan_in, hidden, out, dtype):
        super().__init__()
        self.norm = _Norm(fan_in, dtype)
        self.linear_1 = _Linear(fan_in, hidden, dtype)
        self.linear_2 = _Linear(hidden, hidden, dtype)
        self.linear_3 = _Linear(hidden, out, dtype)


class _ImageProj(nn.Module):
    def __init__(self, fan_in, h, dtype):
        super().__init__()
        self.norm_in = _Norm(fan_in, dtype)
        self.linear_1 = _Linear(fan_in, fan_in, dtype)
        self.linear_2 = _Linear(fan_in, h, dtype)
        self.norm_out = _Norm(h, dtype)


class _PatchEmbed(nn.Module):
    def __init__(self, cin, h, pt, p, dtype):
        super().__init__()
        self.proj = nn.Module()
        self.proj.weight = _p(h, cin, pt, p, p, dtype=dtype)  # Conv3d weight, applied as a linear
        self.proj.bias = _p(h, dtype=dtype)


class _TimeEmbed(nn.Module):
    def __init__(self, h, dtype):
        super().__init__()
        self.timestep_embedder = _TimestepEmbedding(256, h, dtype)


class _NormOut(
    nn.Module
):  # AdaLayerNormContinuous(elementwise_affine=False): linear -> (scale, shift)
    def __init__(self, h, dtype):
        super().__init__()
        self.linear = _Linear(h, 2 * h, dtype)


class NeuronHunyuanVideo15DiT(nn.Module):
    """HunyuanVideo-1.5 DiT. Full call: :meth:`forward`; split call: ``prologue`` ->
    ``run_blocks`` (any block range) -> ``epilogue``.

    Inputs (all fixed-shape, built by :class:`HV15HostInputs`):

    * ``x`` ``[B, in_channels, T, H, W]`` latents (+ cond latents + mask, concatenated upstream);
    * ``timestep`` ``[B]`` (any float dtype; upstream passes the model dtype);
    * ``text`` ``[B, Lt, text_embed_dim]`` MLLM embeddings, ``text_mask`` ``[B, Lt]`` (1 valid);
    * ``text2`` ``[B, L2, text_embed_2_dim]`` byT5 embeddings;
    * ``image`` ``[B, 729, image_embed_dim]`` SigLIP embeddings or ``None`` (T2V);
    * ``enc_index`` ``[B, Ne]`` int64: gather index into ``cat([image?, byT5, MLLM])``, valid first;
    * ``cos``/``sin`` ``[T*H*W, head_dim]`` fp32 video RoPE;
    * ``text_bias`` ``[B, 1, 1, Lt]`` / ``key_bias`` ``[B, 1, 1, S_video + Ne]`` fp32 key masks.
    """

    def __init__(
        self, cfg: HV15Config, dtype: torch.dtype = torch.bfloat16, tp: tuple | None = None
    ):
        super().__init__()
        self.cfg, self.dtype = cfg, dtype
        self.tp_size, self.tp_rank, self.tp_group = tp if tp is not None else tp_state()
        self.cp_size, self.cp_rank, self.cp_group = cp_state() if tp is None else (1, 0, None)
        if cfg.num_heads % self.tp_size or cfg.mlp_hidden % self.tp_size:
            raise ValueError(
                f"TP={self.tp_size} must divide heads={cfg.num_heads} and mlp={cfg.mlp_hidden}"
            )
        h, pt, p = cfg.hidden, cfg.patch_size_t, cfg.patch_size
        self.x_embedder = _PatchEmbed(cfg.in_channels, h, pt, p, dtype)
        self.image_embedder = _ImageProj(cfg.image_embed_dim, h, dtype)
        self.context_embedder = _ContextEmbedder(cfg, dtype)
        self.context_embedder_2 = _ByT5Proj(cfg.text_embed_2_dim, 2048, h, dtype)
        self.time_embed = _TimeEmbed(h, dtype)
        self.cond_type_embed = nn.Module()
        self.cond_type_embed.weight = _p(3, h, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            HV15TransformerBlock(cfg, dtype, self.tp_size) for _ in range(cfg.num_layers)
        )
        self.norm_out = _NormOut(h, dtype)
        self.proj_out = _Linear(h, pt * p * p * cfg.out_channels, dtype)
        self._attach_weight_loaders()

    # -- weights ------------------------------------------------------------------------------
    def _attach_weight_loaders(self) -> None:
        if self.tp_size == 1:
            return
        from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

        for mod in self.transformer_blocks.modules():
            if isinstance(mod, _Linear) and mod.shard is not None:
                w = mod.weight
                set_weight_loader(
                    w,
                    sharding_weight_loader(
                        shard_dim=mod.shard, shard_size=w.shape[mod.shard], num_shards=self.tp_size
                    ),
                )
                if mod.shard == 0 and mod.bias is not None:
                    set_weight_loader(
                        mod.bias,
                        sharding_weight_loader(
                            shard_dim=0, shard_size=mod.bias.shape[0], num_shards=self.tp_size
                        ),
                    )

    def load_weights(self, model_path: str, device: torch.device | str = "cpu") -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        ck_logger = logging.getLogger("vllm_neuron.utils.checkpoints")
        level = ck_logger.level
        ck_logger.setLevel(
            logging.ERROR
        )  # fp32 checkpoint -> model dtype: one warning per tensor otherwise
        try:
            result = SafetensorsCheckpoint(
                os.path.join(local_model_dir(model_path), "transformer")
            ).load_sharded_pipelined(self.tp_rank, self.tp_size, self, {}, torch.device(device))
        finally:
            ck_logger.setLevel(level)
        self.load_state_dict(result.state_dict, strict=True, assign=True)

    # -- forward pieces -----------------------------------------------------------------------
    def _refine(self, text, text_mask, text_bias, timestep):
        ce, cfg = self.context_embedder, self.cfg
        m = text_mask.float().unsqueeze(-1)
        pooled = ((text.float() * m).sum(dim=1) / m.sum(dim=1)).to(text.dtype)
        temb = ce.time_text_embed.timestep_embedder(timestep_proj(timestep).to(self.dtype))
        temb = temb + ce.time_text_embed.text_embedder(pooled)
        temb_act = F.silu(temb)
        h = ce.proj_in(text)
        b, s, _ = h.shape
        nh, d = cfg.num_heads, cfg.head_dim
        for blk in ce.token_refiner.refiner_blocks:
            n = layer_norm(h, 1e-6, blk.norm1)
            q = blk.attn.to_q(n).view(b, s, nh, d).transpose(1, 2)
            k = blk.attn.to_k(n).view(b, s, nh, d).transpose(1, 2)
            v = blk.attn.to_v(n).view(b, s, nh, d).transpose(1, 2)
            o = (
                hv15_attention(q, k, v, d**-0.5, key_bias=text_bias)
                .transpose(1, 2)
                .reshape(b, s, nh * d)
            )
            o = blk.attn.to_out[0](o)
            g_msa, g_mlp = blk.norm_out.linear(temb_act).chunk(2, dim=-1)
            h = h + o * g_msa[:, None]
            ff = blk.ff.net[2](F.silu(blk.ff.net[0].proj(layer_norm(h, 1e-6, blk.norm2))))
            h = h + ff * g_mlp[:, None]
        return h

    def _patchify(self, x):
        b, c, t, hh, ww = x.shape
        pt, p = self.cfg.patch_size_t, self.cfg.patch_size
        x = x.reshape(b, c, t // pt, pt, hh // p, p, ww // p, p).permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(b, (t // pt) * (hh // p) * (ww // p), c * pt * p * p)
        w = self.x_embedder.proj.weight.reshape(self.cfg.hidden, -1)
        return F.linear(x.to(self.dtype), w, self.x_embedder.proj.bias)

    def prologue(self, x, timestep, text, text_mask, text2, image, enc_index, text_bias):
        """-> (video tokens ``[B, Sv, H]``, encoder tokens ``[B, Ne, H]``, ``silu(temb)`` ``[B, H]``)."""
        dt = self.dtype
        temb = self.time_embed.timestep_embedder(timestep_proj(timestep).to(dt))
        hidden = self._patchify(x)
        cte = self.cond_type_embed.weight
        text = self._refine(text.to(dt), text_mask, text_bias, timestep) + cte[0]
        c2 = self.context_embedder_2
        t2 = layer_norm(text2.to(dt), 1e-5, c2.norm)
        t2 = c2.linear_3(gelu(c2.linear_2(gelu(c2.linear_1(t2)))))
        streams = [t2 + cte[1], text]
        if image is not None:
            ie = self.image_embedder
            im = layer_norm(image.to(dt), 1e-5, ie.norm_in)
            im = layer_norm(ie.linear_2(gelu(ie.linear_1(im))), 1e-5, ie.norm_out)
            streams.insert(0, im + cte[2])
        enc = torch.cat(streams, dim=1)
        enc = torch.gather(enc, 1, enc_index[:, :, None].expand(-1, -1, enc.shape[-1]))
        return hidden, enc, F.silu(temb)

    def run_blocks(
        self, hidden, enc, temb_act, cos, sin, key_bias, start: int = 0, end: int | None = None
    ):
        for blk in self.transformer_blocks[start:end]:
            hidden, enc = blk(
                hidden, enc, temb_act, cos, sin, key_bias, self.tp_group, self.cp_group
            )
        return hidden, enc

    def epilogue_tokens(self, hidden, temb_act):
        """Final AdaLN + projection, per token: ``[B, S, H]`` -> ``[B, S, out_ch * pt * p * p]``."""
        scale, shift = self.norm_out.linear(temb_act).chunk(2, dim=-1)
        hidden = layer_norm(hidden, 1e-6) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out(hidden)

    def unpatchify(self, out, grid: tuple[int, int, int]):
        cfg = self.cfg
        t, hh, ww = grid
        b, pt, p = out.shape[0], cfg.patch_size_t, cfg.patch_size
        out = out.reshape(b, t, hh, ww, cfg.out_channels, pt, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return out.reshape(b, cfg.out_channels, t * pt, hh * p, ww * p)

    def epilogue(self, hidden, temb_act, grid: tuple[int, int, int]):
        return self.unpatchify(self.epilogue_tokens(hidden, temb_act), grid)

    def forward(
        self, x, timestep, text, text_mask, text2, image, enc_index, cos, sin, text_bias, key_bias
    ):
        _, _, t, hh, ww = x.shape
        grid = (t // self.cfg.patch_size_t, hh // self.cfg.patch_size, ww // self.cfg.patch_size)
        hidden, enc, temb_act = self.prologue(
            x, timestep, text, text_mask, text2, image, enc_index, text_bias
        )
        hidden, _ = self.run_blocks(hidden, enc, temb_act, cos, sin, key_bias)
        return self.epilogue(hidden, temb_act, grid)


# ---------------------------------------------------------------------------------------------
# host-side input preparation
# ---------------------------------------------------------------------------------------------
def pick_bucket(n: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def encoder_layout(text_mask, text2_mask, image_mask=None):
    """Gather index ``[B, Ne]`` into ``cat([image?, byT5, MLLM])`` putting the valid tokens first in
    upstream's order (image, byT5, MLLM), then the rest; plus the ``[B, Ne]`` validity of the
    result (a valid prefix). Pure host math on the masks."""
    masks = [
        m.bool()
        for m in ((image_mask,) if image_mask is not None else ()) + (text2_mask, text_mask)
    ]
    full = torch.cat(masks, dim=1)
    idx, valid = [], []
    for row in full:
        pos = torch.arange(row.numel())
        order = torch.cat([pos[row], pos[~row]])
        idx.append(order)
        valid.append(torch.arange(row.numel()) < int(row.sum()))
    return torch.stack(idx).long(), torch.stack(valid)


def prepare_encoder_inputs(
    text_mask, text2_mask, image_mask=None, video_tokens: int = 0, video_pad: int = 0
):
    """Host tensors for the encoder side: ``(enc_index, text_bias, key_bias)``.

    ``key_bias`` covers the joint key sequence ``[video | video padding (masked) | encoder]``;
    ``video_pad`` > 0 only under context parallelism (video tokens padded to a multiple of CP)."""
    enc_index, enc_valid = encoder_layout(text_mask, text2_mask, image_mask)
    b = enc_valid.shape[0]
    joint = torch.cat(
        [
            torch.ones(b, video_tokens, dtype=torch.bool),
            torch.zeros(b, video_pad, dtype=torch.bool),
            enc_valid,
        ],
        dim=1,
    )
    return enc_index, key_padding_bias(text_mask.bool()), key_padding_bias(joint)


__all__ = [
    "HV15Config",
    "NeuronHunyuanVideo15DiT",
    "encoder_layout",
    "key_padding_bias",
    "local_model_dir",
    "pick_bucket",
    "cp_state",
    "prepare_encoder_inputs",
    "rope_tables",
    "tp_state",
]
