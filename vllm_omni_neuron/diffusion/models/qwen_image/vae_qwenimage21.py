# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE for Neuron: one compiled graph each for single-image decode and encode.

The 2.1 VAE (vendored ``_vendor/autoencoder_kl_qwenimage21.py``) is the image specialization of
Wan's causal 3D VAE: every "3D" conv is a 2D conv over a single frame, and the feature cache
only ever sees its first-chunk state (``None`` -> ``"Rep"``). A one-frame decode is therefore
a pure function of the latent; the wrapper below traces exactly upstream's first-chunk path
(fresh cache list, ``first_chunk=True``) as one fixed-shape graph per resolution, and the
pipeline-facing ``decode`` / ``encode`` keep upstream's host-tensor interface.

Large images decode as fixed-shape latent tiles (one compiled graph for every resolution). Under
tensor parallelism the tiles are dealt round-robin across the TP ranks with the shared
``run_tiles`` helper (``diffusion/layers/vae_tiling.py``) and gathered to rank 0 for the blend:
the VAE is not sharded, so every rank holds a full copy and would otherwise idle during decode.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

from .common import host

# Spatial tiling of the decode: "<tile>,<overlap>" in LATENT pixels (x16 for image pixels). One
# tile shape -> one compiled graph for every resolution; tiles overlap and are feather-blended.
# A latent no larger than one tile is decoded whole. "0" disables tiling. Read per decode.
DEFAULT_VAE_TILE = "16,4"

VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "-O1",
    "--internal-max-instruction-limit=15000000",
]


class _DecodeImage(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.n_conv = vae._cached_conv_counts["decoder"]

    def forward(self, z):
        x = self.post_quant_conv(z)
        out = self.decoder(x, feat_cache=[None] * self.n_conv, feat_idx=[0], first_chunk=True)
        return torch.clamp(out, min=-1.0, max=1.0)


class _EncodeImage(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.n_conv = vae._cached_conv_counts["encoder"]

    def forward(self, x):
        out = self.encoder(x[:, :, :1], feat_cache=[None] * self.n_conv, feat_idx=[0])
        return self.quant_conv(out)


class NeuronQwenImage21VAE(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path: str, subfolder: str = "vae", torch_dtype=torch.bfloat16):
        from ._vendor.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

        return cls(
            AutoencoderKLQwenImage21.from_pretrained(
                model_path, subfolder=subfolder, torch_dtype=torch_dtype
            ).eval()
        )

    def __init__(self, vae):
        super().__init__()
        if vae.config.patch_size is not None:
            raise NotImplementedError("Qwen-Image 2.1 VAE with patch_size is not supported")
        self.vae = vae
        self.config = vae.config
        self._dec = _DecodeImage(vae)
        self._enc = _EncodeImage(vae)
        self._dec_fn = self._dec
        self._enc_fn = self._enc
        self._device = torch.device("cpu")
        self.last_decode_s = 0.0

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.vae.to(self._device)
        return self

    def compile(
        self, backend: str, options: dict | None = None, compile_encoder: bool = False
    ) -> None:
        def opts(name):
            return {**(options or {}), "model_name": name, "compiler_args": list(VAE_COMPILER_ARGS)}

        self._dec_fn = torch.compile(
            self._dec,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options=opts("qwen_image21_vae_decode"),
        )
        if compile_encoder:
            self._enc_fn = torch.compile(
                self._enc,
                backend=backend,
                fullgraph=True,
                dynamic=False,
                options=opts("qwen_image21_vae_encode"),
            )

    def _decode_one(self, z: torch.Tensor) -> torch.Tensor:
        return self._dec_fn(host(z, self.dtype).to(self._device)).to("cpu")

    @staticmethod
    def _ramp(length: int, lo: int, hi: int) -> torch.Tensor:
        w = torch.ones(length)
        if lo:
            w[:lo] = torch.arange(1, lo + 1, dtype=torch.float32) / (lo + 1)
        if hi:
            w[length - hi :] = torch.minimum(
                w[length - hi :], torch.arange(hi, 0, -1, dtype=torch.float32) / (hi + 1)
            )
        return w

    def _decode_tiled(
        self, z: torch.Tensor, tile: int, overlap: int, group=None
    ) -> torch.Tensor | None:
        """Fixed-shape latent tiles, decoded on the ranks of ``group`` (round-robin, gathered to its
        rank 0; ``None`` = this process alone) and blended with linear ramps over each tile's
        overlap with its neighbours (a normalized weighted average). ``None`` on the other ranks."""
        from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, run_tiles

        _, _, _, h, w = z.shape
        r = 16
        th_in, tw_in = min(tile, h), min(tile, w)
        grid = TileGrid.for_axes(
            total=(h, w),
            tile=(th_in, tw_in),
            stride=(max(th_in - overlap, 1), max(tw_in - overlap, 1)),
        )
        rs, cs = grid.starts

        def one(n, idx):
            return self._decode_one(grid.slice_input(z, idx)).float()

        tiles = (
            run_tiles(grid, one, group=group)
            if group is not None
            else run_tiles(grid, one, world_size=1, rank=0)
        )
        if tiles is None:
            return None
        acc = wsum = None
        for (i, j), out in sorted(tiles.items()):
            y, x = rs[i], cs[j]
            if acc is None:
                acc = out.new_zeros(out.shape[0], out.shape[1], 1, h * r, w * r)
                wsum = out.new_zeros(1, 1, 1, h * r, w * r)
            th, tw = out.shape[-2:]
            top = (rs[i - 1] + th_in - y) * r if i > 0 else 0
            bot = (y + th_in - rs[i + 1]) * r if i + 1 < len(rs) else 0
            lef = (cs[j - 1] + tw_in - x) * r if j > 0 else 0
            rig = (x + tw_in - cs[j + 1]) * r if j + 1 < len(cs) else 0
            wt = self._ramp(th, top, bot)[:, None] * self._ramp(tw, lef, rig)[None, :]
            acc[..., y * r : y * r + th, x * r : x * r + tw] += out * wt
            wsum[..., y * r : y * r + th, x * r : x * r + tw] += wt
        return (acc / wsum).to(self.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, return_dict: bool = True, group=None):
        """``z`` ``[B, z_dim, 1, h, w]`` (denormalized) -> image ``[B, C, 1, 16h, 16w]`` on the host.

        ``group``: a host (gloo) process group whose ranks share the tiles of a tiled decode; every
        rank of it must call ``decode`` with the same ``z`` shape. Only its rank 0 gets the image
        (the others get ``None``); an untiled (single-tile) decode runs on that rank alone.
        """
        from diffusers.models.autoencoders.vae import DecoderOutput

        if z.shape[2] != 1:
            raise ValueError("the Qwen-Image 2.1 VAE decodes single images")
        t0 = time.time()
        spec = os.environ.get("QWEN_IMAGE_VAE_TILE", DEFAULT_VAE_TILE)
        tile, overlap = (int(v) for v in (spec.split(",") + ["0"])[:2])
        lead = group is None or torch.distributed.get_rank(group) == 0
        outs = []
        for zi in z.split(1):
            if tile and (zi.shape[-2] > tile or zi.shape[-1] > tile):
                outs.append(self._decode_tiled(zi, tile, overlap, group))
            else:
                outs.append(self._decode_one(zi) if lead else None)
        self.last_decode_s = time.time() - t0
        out = torch.cat(outs) if lead else None
        return DecoderOutput(sample=out) if return_dict else (out,)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        h = torch.cat(
            [
                self._enc_fn(host(xi, self.dtype).to(self._device)).to("cpu").float()
                for xi in x.split(1)
            ]
        )
        dist = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)


def tile_parallel_group():
    """The host (gloo) TP group the VAE decode deals its tiles across, or ``None`` (single
    process, TP=1, or ``QWEN_IMAGE_VAE_PARALLEL=0``). On by default: every TP rank holds the
    full, unsharded VAE anyway, so its tiles are free parallelism."""
    if os.environ.get("QWEN_IMAGE_VAE_PARALLEL", "1") != "1":
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
    except (AssertionError, ImportError):
        return None
    return tp.cpu_group if tp.world_size > 1 else None


def skip_redundant_decode() -> bool:
    """Only TP rank 0's image is returned. With tile parallelism (the default under TP) every rank
    decodes its share of the tiles. Without it the other ranks skip the decode (it would cost HBM
    and, cold, a duplicate compile behind the cache lock); ``QWEN_IMAGE_DECODE_ALL_RANKS=1``
    turns that off."""
    import torch.distributed as dist

    if (
        os.environ.get("QWEN_IMAGE_DECODE_ALL_RANKS", "0") == "1"
        or tile_parallel_group() is not None
    ):
        return False
    return dist.is_initialized() and dist.get_rank() != 0
