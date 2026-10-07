# SPDX-License-Identifier: Apache-2.0
"""FLUX Action video VAE (ViTNorm: Swin3D + neighborhood attention) for Neuron.

Module tree, parameter names and math follow upstream ``flux_action/models/video_vae.py`` so
``video_vae.safetensors`` loads strictly. Neuron-specific changes:

* NATTEN is replaced by gather-free formulations with the same window and scale definition: on
  the NeuronCore the plugin's shared halo-tiled
  :func:`~vllm_omni_neuron.diffusion.attention.neighborhood_attention.neighborhood_attention_tiled`
  (per-layer bias / selection matrices prepared on the host by
  :meth:`VideoVAE.prepare_device_attention`), on the host a dense or query-blocked banded form;
* the patch embedding's stride == kernel ``Conv3d`` runs as reshape + matmul (same weights);
* no CUDA memory probing / ``torch.compile`` wiring; the policy compiles the encoder
  (:meth:`VideoVAE.encode_frame_mu`) itself.

Only what inference needs is kept: single-frame encode (``encode_frame``, the conditioning latent
of the released serving path), full-clip encode (chunked like upstream), and decode.
"""

from __future__ import annotations

import copy
import json
import math
import os
import types
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .neighborhood import neighborhood_attention, neighborhood_attention_reference

_norm_layer = partial(nn.LayerNorm, eps=1e-5)
NA_IMPL = os.environ.get(
    "FLUX3_ACTION_NA_IMPL", "auto"
)  # auto | tiled | banded | gather | reference
# Query tiling of the device (tiled) neighborhood attention: "<N>" = N-wide tiles (an axis shorter
# than N is one tile; a partial last tile is padded), "pf<N>" = the largest divisor of each axis
# <= N (no padded tile; see neighborhood_attention.pad_free_tile). The default "pf24" gives 17x23
# query tiles at every encoder stage of the DROID canvas (136x184 ... 17x23): no padding, and the
# fastest of the tilings measured on Trn2 (stage-1 block 13 ms vs 663 ms with 16x16 tiles).
NA_TILE = os.environ.get("FLUX3_ACTION_NA_TILE", "pf24")
VAE_CONFIG_NAME = (
    "video_vae.json"  # optional sidecar with non-default dims (the tiny structure model)
)


@dataclass
class VideoVAEParams:
    z_dim: int = 96
    embed_dim: int = 256
    patch_size: list[int] = field(default_factory=lambda: [1, 4, 4])
    window_size: list[int] = field(default_factory=lambda: [5, 5, 5])
    enc_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    dec_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    num_heads: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    temporal: list[bool] = field(default_factory=lambda: [False, False, True, True])
    enc_causal: bool = True
    dec_causal: bool = False
    qk_norm: bool = True
    patch_norm: bool = False
    chunk_size_frames: int = 45

    @classmethod
    def for_weights(cls, weights_path: str) -> VideoVAEParams:
        """Defaults (the released VAE) unless a ``video_vae.json`` sidecar sits next to the weights."""
        side = os.path.join(os.path.dirname(os.path.abspath(weights_path)), VAE_CONFIG_NAME)
        if os.path.isfile(side):
            with open(side) as f:
                return replace(cls(), **json.load(f))
        return cls()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


class DistributedRunningStats(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.register_buffer("initialized", torch.tensor(False))

    def _shape(self, x: Tensor) -> tuple:
        return (1, -1) + (1,) * (x.dim() - 2)

    def normalize(self, x: Tensor) -> Tensor:
        s = self._shape(x)
        return (x - self.running_mean.view(s)) / self.running_var.sqrt().view(s)

    def denormalize(self, x: Tensor) -> Tensor:
        s = self._shape(x)
        return x * self.running_var.sqrt().view(s) + self.running_mean.view(s)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        b, d, h, w, c = x.shape
        x = x.reshape(b, d, h // 2, 2, w // 2, 2, c)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).flatten(4)
        return self.reduction(self.norm(x))


class TemporalMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        if d % 2 == 1:
            x = torch.concat([x[:, :1], x], dim=1)
        b, d, h, w, c = x.shape
        x = x.reshape(b, d // 2, 2, h, w, c)
        skip = x.mean(2)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(b, d // 2, h, w, 2 * c)
        return self.reduction(self.norm(x)) + skip


class PatchExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim, self.out_dim = dim, out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 4 * out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        x = self.expansion(self.norm(x))
        x = x.view(b, d, h, w, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, d, h * 2, w * 2, self.out_dim)


class TemporalExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim, self.out_dim = dim, out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 2 * out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        x = self.expansion(self.norm(x)) + torch.concat([x, x], -1)
        x = x.view(b, d, h, w, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(b, d * 2, h, w, self.out_dim)
        return x[:, 1:]


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, head_dim: int, base: float = 256.0):
        super().__init__()
        assert head_dim % 8 == 0, "head dimension must be divisible by 8"
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 4
        axis_inv_freq = 1.0 / (
            base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)
        )
        inv_freq = torch.stack(
            [axis_inv_freq, axis_inv_freq, axis_inv_freq, torch.zeros(self.chunk_dim // 2)]
        )
        self.register_buffer(
            "inv_freq", inv_freq
        )  # the checkpoint stores it (bf16): load it, do not recompute

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        _, t, h, w, _, _ = q.shape
        dev, dtype = q.device, q.dtype
        grids = torch.meshgrid(
            torch.arange(t, device=dev, dtype=torch.float32),
            torch.arange(h, device=dev, dtype=torch.float32),
            torch.arange(w, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
        freqs = torch.einsum("...a,af->...af", pos, self.inv_freq.float())
        freqs = freqs.reshape(1, t, h, w, 1, -1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


def _band_mask(length: int, kernel: int, causal: bool) -> Tensor:
    """``[length, length]`` bool neighborhood mask for one axis (NATTEN semantics)."""
    i = torch.arange(length)[:, None]
    j = torch.arange(length)[None, :]
    if causal:
        return (i - j >= 0) & (i - j < kernel)
    left, right = kernel // 2, kernel // 2 + (kernel % 2 - 1)
    center = i.clamp(left, length - 1 - right)
    return ((center - j >= 0) & (center - j <= left)) | ((j - center >= 0) & (j - center <= right))


_BAND_BIAS_CACHE: dict = {}


def _band_bias(axes: tuple, kernel: tuple, causal: tuple, device) -> Tensor:
    """Cached fp32 additive neighborhood bias ``[N, N]`` (0 inside, -3e4 outside), built on the HOST.

    Building it from a bool mask inside the traced graph tripped the Neuron compiler's mask
    propagation (NCC_IMPR902); as a precomputed constant it is just an fp32 add.
    """
    key = (axes, kernel, causal, str(device))
    hit = _BAND_BIAS_CACHE.get(key)
    if hit is not None:
        return hit
    n_ax = len(axes)
    n = 1
    for a in axes:
        n *= a
    mask = torch.ones(*([1] * (2 * n_ax)), dtype=torch.bool)
    for a in range(n_ax):
        view = [1] * (2 * n_ax)
        view[a] = axes[a]
        view[n_ax + a] = axes[a]
        mask = mask & _band_mask(axes[a], kernel[a], causal[a]).reshape(view)
    bias = torch.where(mask.reshape(n, n), 0.0, -30000.0).to(torch.float32).to(device)
    _BAND_BIAS_CACHE[key] = bias
    return bias


def neighborhood_attention_banded(
    q: Tensor, k: Tensor, v: Tensor, kernel: Sequence[int], causal: Sequence[bool] | None = None
) -> Tensor:
    """Dense attention over FULL keys with the neighborhood as a precomputed additive bias, computed in
    static query blocks so the ``[Lq_block, Lk]`` score tile is bounded (no gather, no full N^2
    materialised, no in-graph bool mask).

    The multi-axis neighborhood factorises: a (query, key) pair is admitted iff admitted on every axis.
    The bias is a host-side constant (see :func:`_band_bias`); query blocking bounds the score tile.
    """
    n_ax = q.ndim - 3
    kernel = tuple(kernel)
    causal = tuple(causal) if causal is not None else (False,) * n_ax
    axes = tuple(q.shape[1 : 1 + n_ax])
    b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
    n = 1
    for a in axes:
        n *= a
    bias = _band_bias(axes, kernel, causal, q.device)
    qf = q.reshape(b, n, heads, d).permute(0, 2, 1, 3).float()
    kf = k.reshape(b, n, heads, d).permute(0, 2, 1, 3).float()
    vf = v.reshape(b, n, heads, d).permute(0, 2, 1, 3).float()
    block = int(os.environ.get("FLUX3_ACTION_NA_QBLOCK", "512"))
    out = torch.empty_like(qf)
    scale = d**-0.5
    for s in range(0, n, block):
        e = min(s + block, n)
        scores = torch.matmul(qf[:, :, s:e], kf.transpose(-2, -1)) * scale + bias[s:e][None, None]
        out[:, :, s:e] = torch.matmul(torch.softmax(scores, dim=-1), vf)
    return out.permute(0, 2, 1, 3).reshape(q.shape).to(q.dtype)


def _na_tile(axes: Sequence[int], mode: str | None = None) -> tuple[int, ...]:
    """Query tile per axis for the device op (see ``NA_TILE``)."""
    mode = str(NA_TILE if mode is None else mode)
    if mode.startswith("pf"):
        from vllm_omni_neuron.diffusion.attention.neighborhood_attention import pad_free_tile

        return tuple(pad_free_tile(tuple(axes), (1,) * len(axes), max_tile=int(mode[2:])))
    return tuple(min(int(mode), int(n)) for n in axes)


def na_device_consts(
    axes: Sequence[int],
    kernel: Sequence[int],
    causal: Sequence[bool],
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> dict:
    """Host-built graph inputs of the tiled op for one layer's grid: the additive ``bias`` and the
    per-axis K/V window ``select`` matrices (built once, outside any compiled region)."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        neighborhood_bias,
        neighborhood_select_matrices,
    )

    tile = _na_tile(axes)
    bias = neighborhood_bias(tuple(axes), tuple(kernel), list(causal), tile).to(device)
    select = neighborhood_select_matrices(tuple(axes), tuple(kernel), tile, dtype, device)
    return {"bias": bias.contiguous(), "select": tuple(m.contiguous() for m in select)}


def _na_tiled(q: Tensor, k: Tensor, v: Tensor, kernel, causal, consts: dict | None) -> Tensor:
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        neighborhood_attention_tiled,
    )

    axes = tuple(q.shape[1:-2])
    consts = consts or {}
    return neighborhood_attention_tiled(
        q,
        k,
        v,
        list(kernel),
        list(causal) if causal is not None else None,
        list(_na_tile(axes)),
        bias=consts.get("bias"),
        select=consts.get("select"),
    )


def _na(q: Tensor, k: Tensor, v: Tensor, kernel, causal, consts: dict | None = None) -> Tensor:
    """Pick the neighborhood-attention implementation.

    ``tiled`` is the plugin's shared halo-tiled op (dense per-tile attention, selection-matmul
    windows, host-built additive bias): the device path. ``auto`` uses it whenever ``q`` is on a
    non-CPU device; ``consts`` are the layer's prepared ``bias`` / ``select`` graph inputs.

    ``gather`` is the memory-lean static-index path, but on the Neuron device its index tensor lowers
    to a Vector-DGE DMACopy the compiler rejects unless it starts at partition 0 (NCC_EBIR026).
    ``auto`` (default) therefore never gathers on device: a small grid uses the dense masked
    definition, a large one the query-blocked **banded** form (full keys, factorised neighborhood
    mask, bounded score tile -- still gather-free). ``reference`` forces dense (CPU oracle); ``gather``
    forces the gather path; ``banded`` forces the banded path.
    """
    import math as _m

    if NA_IMPL == "tiled" or (NA_IMPL == "auto" and q.device.type != "cpu"):
        return _na_tiled(q, k, v, kernel, causal, consts)
    if NA_IMPL == "reference":
        return neighborhood_attention_reference(q, k, v, list(kernel), causal)
    if NA_IMPL == "gather":
        return neighborhood_attention(q, k, v, list(kernel), causal)
    if NA_IMPL == "banded":
        return neighborhood_attention_banded(q, k, v, list(kernel), causal)
    tokens = _m.prod(q.shape[1:-2])
    if tokens <= int(os.environ.get("FLUX3_ACTION_NA_DENSE_MAX", "512")):
        return neighborhood_attention_reference(q, k, v, list(kernel), causal)
    return neighborhood_attention_banded(q, k, v, list(kernel), causal)


class Natten3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: list[int],
        num_heads: int,
        causal: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionEmbedding3D(self.head_dim)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        self.na_grid: tuple[int, int] | None = None  # single-frame grid the buffers were built for

    def set_na_consts(self, grid: tuple[int, int], consts: dict) -> None:
        """Register the tiled op's prepared inputs as (non-persistent) buffers, so a compiled
        encoder takes them as plain graph inputs on the module's device."""
        self.na_grid = tuple(grid)
        self.register_buffer("na_bias", consts["bias"], persistent=False)
        for i, m in enumerate(consts["select"]):
            self.register_buffer(f"na_select_{i}", m, persistent=False)

    def _na_consts(self, h: int, w: int) -> dict | None:
        if self.na_grid != (h, w):
            return None
        return {"bias": self.na_bias, "select": (self.na_select_0, self.na_select_1)}

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k)
        if t == 1:  # a single frame uses the 2-D kernel (upstream: na2d)
            out = _na(
                q[:, 0],
                k[:, 0],
                v[:, 0],
                self.window_size[1:],
                [False, False],
                self._na_consts(h, w),
            ).unsqueeze(1)
        else:
            out = _na(q, k, v, self.window_size, [self.causal, False, False])
        return self.proj(out.reshape(b, t, h, w, c))


class GLU_MLP(nn.Module):  # noqa: N801 (upstream name)
    def __init__(self, dim: int, align_to: int = 64):
        super().__init__()
        hidden = align_to * ((int(dim * 8 / 3) + align_to - 1) // align_to)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        causal: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = _norm_layer(dim)
        self.attn = Natten3D(dim, window_size, num_heads, causal=causal, qk_norm=qk_norm)
        self.norm2 = _norm_layer(dim)
        self.mlp = GLU_MLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed3d(nn.Module):
    def __init__(
        self, patch_size: list[int], in_channels: int = 3, embed_dim: int = 96, norm: bool = True
    ):
        super().__init__()
        self.ps = tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.ps, stride=self.ps)
        self.norm = _norm_layer(embed_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        pt, ph, pw = self.ps
        pad = [(p - s % p) % p for s, p in zip((t, h, w), self.ps, strict=True)]
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        _, _, t, h, w = x.shape
        # stride == kernel conv == matmul over non-overlapping patches
        x = (
            x.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .contiguous()
        )
        x = x.reshape(b, t // pt, h // ph, w // pw, c * pt * ph * pw)
        x = F.linear(x, self.proj.weight.reshape(self.proj.weight.shape[0], -1), self.proj.bias)
        return self.norm(x)


class DecoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch,
        patch_size,
        embed_dim,
        depths,
        temporal,
        num_heads,
        window_size,
        causal=False,
        qk_norm=False,
    ):
        super().__init__()
        self.ps = patch_size
        self.proj_in = nn.Linear(z_ch, embed_dim * 2 ** (len(depths) - 1))
        self.proj_out = nn.Linear(embed_dim, math.prod(patch_size) * 3)
        layers: list[nn.Module] = []
        for i_stage in reversed(range(len(depths))):
            dim = embed_dim * 2**i_stage
            layers.append(
                nn.Sequential(
                    *[
                        SwinTransformerBlock(
                            dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm
                        )
                        for _ in range(depths[i_stage])
                    ]
                )
            )
            if temporal[i_stage]:
                layers.append(TemporalExpansion(dim, dim))
                layers.append(
                    SwinTransformerBlock(
                        dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm
                    )
                )
            if i_stage > 0:
                layers.append(PatchExpansion(dim, dim // 2))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_in(x.permute(0, 2, 3, 4, 1).contiguous())
        x = self.proj_out(self.features(x))
        b, t, h, w, _ = x.shape
        x = x.view(b, t, h, w, self.ps[0], self.ps[1], self.ps[2], 3)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(b, t * self.ps[0], h * self.ps[1], w * self.ps[2], 3)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class EncoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch,
        patch_size,
        embed_dim,
        depths,
        temporal,
        num_heads,
        window_size,
        causal=False,
        qk_norm=False,
        patch_norm=True,
    ):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 2 ** (len(depths) - 1), z_ch)
        self.patch_embed = PatchEmbed3d(patch_size=patch_size, embed_dim=embed_dim, norm=patch_norm)
        layers: list[nn.Module] = []
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage
            layers.append(
                nn.Sequential(
                    *[
                        SwinTransformerBlock(
                            dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm
                        )
                        for _ in range(depths[i_stage])
                    ]
                )
            )
            downsampled = False
            if i_stage < len(depths) - 1:
                layers.append(PatchMerging(dim, 2 * dim))
                downsampled = True
            if temporal[i_stage]:
                head_stage = (
                    i_stage + 1 if downsampled else i_stage
                )  # upstream increments i_stage here
                if downsampled:
                    dim = 2 * dim
                layers.append(
                    SwinTransformerBlock(
                        dim, num_heads[head_stage], window_size, causal=causal, qk_norm=qk_norm
                    )
                )
                layers.append(TemporalMerging(dim, dim))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(self.features(self.patch_embed(x)))
        return x.permute(0, 4, 1, 2, 3).contiguous()


class ViTNorm(nn.Module):
    def __init__(self, p: VideoVAEParams):
        super().__init__()
        self.z_dim = p.z_dim
        self.encoder = EncoderSwin3D(
            2 * p.z_dim,
            p.patch_size,
            p.embed_dim,
            p.enc_depths,
            p.temporal,
            p.num_heads,
            p.window_size,
            causal=p.enc_causal,
            qk_norm=p.qk_norm,
            patch_norm=p.patch_norm,
        )
        self.decoder = DecoderSwin3D(
            p.z_dim,
            p.patch_size,
            p.embed_dim,
            p.dec_depths,
            p.temporal,
            p.num_heads,
            p.window_size,
            causal=p.dec_causal,
            qk_norm=p.qk_norm,
        )
        self.z_normalizer = DistributedRunningStats(p.z_dim)

    def encode(self, x: Tensor) -> Tensor:
        mu, _ = self.encoder(x).chunk(2, dim=-4)
        return self.z_normalizer.normalize(mu)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.z_normalizer.denormalize(z))


class VideoVAE(nn.Module):
    """``model`` (``ViTNorm``) under the checkpoint's ``model.`` prefix, plus the inference entry points."""

    TEMPORAL_DOWNSAMPLE = 4
    SPATIAL_DOWNSAMPLE = 32

    def __init__(self, params: VideoVAEParams | None = None):
        super().__init__()
        self.params = params or VideoVAEParams()
        self.model = ViTNorm(self.params)

    @classmethod
    def from_file(cls, path: str, dtype: torch.dtype = torch.bfloat16) -> VideoVAE:
        from safetensors.torch import load_file

        with torch.device("meta"):
            vae = cls(VideoVAEParams.for_weights(path))
        state = load_file(path, device="cpu")
        state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
        vae.load_state_dict(state, strict=True, assign=True)
        return vae.eval().requires_grad_(False)

    def prepare_device_attention(
        self,
        frame_hw: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[tuple[int, int]]:
        """Build every encoder attention layer's tiled-op inputs for single-frame encodes at
        ``frame_hw`` and place them on ``device`` (call after the VAE is on ``device``, before
        compiling :meth:`encode_frame`). Returns the per-layer grids."""
        enc = self.model.encoder
        ph, pw = enc.patch_embed.ps[1:]
        h, w = -(-int(frame_hw[0]) // ph), -(-int(frame_hw[1]) // pw)
        cache: dict = {}
        grids = []

        def visit(mod: nn.Module) -> None:
            nonlocal h, w
            if isinstance(mod, SwinTransformerBlock):
                attn = mod.attn
                key = (h, w, tuple(attn.window_size[1:]))
                if key not in cache:
                    cache[key] = na_device_consts(
                        (h, w), attn.window_size[1:], (False, False), dtype, device
                    )
                attn.set_na_consts((h, w), cache[key])
                grids.append((h, w))
            elif isinstance(mod, PatchMerging):
                h, w = -(-h // 2), -(-w // 2)
            elif isinstance(mod, nn.Sequential):
                for m in mod:
                    visit(m)

        visit(enc.features)
        return grids

    def compile_encoder(
        self, backend: str, options: dict | None = None, compiler_args: Sequence[str] = ()
    ) -> None:
        """Compile the encoder for the device as ONE GRAPH PER DISTINCT PIECE, replayed with each
        layer's own weights as graph inputs (the DiT's weights-as-arguments pattern): one graph per
        (layer type, input shape) -- the patch embedding, one transformer block per grid, the patch /
        temporal merges and the output projection, 10 graphs for the 23 blocks. A single graph for
        the whole encoder fails ``neuronx-cc`` (``NCC_INLA001`` BIR partition-access verification at
        the 34x46 stage) while every piece compiles and matches CPU on its own. Call after
        :meth:`prepare_device_attention`; :meth:`encode_frame_mu` then runs the compiled pieces."""
        from torch.func import functional_call

        enc = self.model.encoder
        base = dict(options or {})
        graphs: dict = {}

        def meta_template(mod: nn.Module) -> nn.Module:
            # A weightless (meta) copy of the piece: every call swaps in the real layer's tensors, and
            # no layer's tensors alias the template's (aliasing would make Dynamo guard on identity
            # and recompile the graph for the next layer).
            memo = {}
            for t in mod.parameters():
                memo[id(t)] = nn.Parameter(torch.empty_like(t, device="meta"), requires_grad=False)
            for t in mod.buffers():
                memo[id(t)] = torch.empty_like(t, device="meta")
            return copy.deepcopy(mod, memo)

        def piece_fn(template: nn.Module, name: str):
            def run(state: dict, x: Tensor) -> Tensor:
                return functional_call(template, state, (x,))

            # Dynamo caches compiled graphs per CODE OBJECT (recompile limit 8): give every piece
            # its own code object so the 10 piece graphs are 10 first compiles, not 10 recompiles
            # of one function.
            return types.FunctionType(
                run.__code__.replace(co_name=name), run.__globals__, name, None, run.__closure__
            )

        def wrap(mod: nn.Module):
            state = {**dict(mod.named_parameters()), **dict(mod.named_buffers())}

            def call(x: Tensor) -> Tensor:
                key = (type(mod).__name__, tuple(x.shape), tuple(sorted(state)))
                fn = graphs.get(key)
                if fn is None:
                    name = f"flux3_action_vae_{type(mod).__name__.lower()}_{len(graphs)}"
                    fn = graphs[key] = torch.compile(
                        piece_fn(meta_template(mod), name),
                        backend=backend,
                        fullgraph=True,
                        dynamic=False,
                        options={**base, "model_name": name, "compiler_args": list(compiler_args)},
                    )
                return fn(state, x)

            return call

        def leaves(m: nn.Module):
            for c in m:
                if isinstance(c, nn.Sequential):
                    yield from leaves(c)
                else:
                    yield c

        self._device_pieces = (
            [wrap(enc.patch_embed)] + [wrap(m) for m in leaves(enc.features)] + [wrap(enc.proj)]
        )
        self._device_graphs = graphs

    def encode_frame(self, frame: Tensor) -> Tensor:
        """``[B, 3, H, W]`` in [-1, 1] -> normalized latent ``[B, 96, 1, H/32, W/32]`` (upstream ``encode_frame``)."""
        return self.model.encode(frame[:, :, None])

    def encode_frame_mu(self, frame: Tensor) -> Tensor:
        """The encoder half of :meth:`encode_frame` (posterior mean, NOT normalized): the part that
        runs on the NeuronCore after :meth:`compile_encoder` (the result then comes back on the
        host); :meth:`normalize` finishes it."""
        pieces = getattr(self, "_device_pieces", None)
        if pieces is None:
            mu, _ = self.model.encoder(frame[:, :, None]).chunk(2, dim=-4)
            return mu
        x = frame[:, :, None]
        for run in pieces:
            x = run(x)
        # the (tiny) channels-first reorder on the host: an eager permute+copy of a device tensor is
        # not legal under the Lite runtime ("Expected self.is_contiguous()")
        mu, _ = x.to("cpu").permute(0, 4, 1, 2, 3).contiguous().chunk(2, dim=-4)
        return mu

    def normalize(self, mu: Tensor) -> Tensor:
        return self.model.z_normalizer.normalize(mu)

    def decode(self, z: Tensor) -> Tensor:
        """``[B, 96, T, h, w]`` -> pixels ``[B, 3, 4T - 3, 32h, 32w]`` in [-1, 1] (full, un-looped decode)."""
        return self.model.decode(z).clamp(-1, 1)

    def forward(self, frame: Tensor) -> Tensor:
        return self.encode_frame(frame)
