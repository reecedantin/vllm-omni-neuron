# SPDX-License-Identifier: Apache-2.0
"""diffusers ``AutoencoderKL`` decode on Neuron (Z-Image's Flux VAE; reused by the SD3 / SDXL ports).

Decode at 1024x1024 is one conv graph over a 128x128x16 latent -> 1024x1024x3; compiling it whole
takes several hundred GB of ``neuronx-cc`` host RSS. So the latent is TILED on the host:
one fixed tile-shaped decoder graph (``Z_IMAGE_VAE_TILE`` latent px, default 64 -> 512px output) is
compiled once and replayed per tile, and the overlapping tiles are blended on the host (diffusers'
``blend_v`` / ``blend_h``) OUTSIDE any graph. Tiling is on by default above the tile size; the
whole-latent graph is used below it (and the tiny CPU tests). The mid-block attention processor is
swapped for an explicit-softmax one that lowers on the NeuronCore.

With several ranks in the stage (TP / CFG-parallel) every rank holds the decoder and receives the
same latent, so the tiles are dealt round-robin over the stage's world group: each rank decodes its
share with the same tile graph and the decoded tiles are all-gathered on the host (gloo) before the
blend. The tile graph is deterministic, so the result is bit-identical to one rank decoding all of
them (``Z_IMAGE_VAE_DEAL=0`` turns dealing off).
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn

from .layers import attention

VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "-O1",
    "--internal-max-instruction-limit=15000000",
]
VAE_ATTN_QBLOCK = int(os.environ.get("Z_IMAGE_VAE_ATTN_QBLOCK", "4096"))
# tiling, in LATENT pixels: tile size and overlap fraction (0 disables tiling)
VAE_TILE = int(os.environ.get("Z_IMAGE_VAE_TILE", "64"))
VAE_TILE_OVERLAP = float(os.environ.get("Z_IMAGE_VAE_TILE_OVERLAP", "0.125"))
VAE_DEAL = os.environ.get("Z_IMAGE_VAE_DEAL", "1") != "0"


def _stage_world():
    """(rank, size, gloo group) of the stage's world group; (0, 1, None) when none is initialised."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        w = get_world_group()
        return w.rank_in_group, w.world_size, w.cpu_group
    except (AssertionError, ImportError, AttributeError):
        return 0, 1, None


class NeuronVaeAttnProcessor:
    """``AttnProcessor2_0`` for the VAE's single-head spatial self-attention, explicit softmax."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *a,
        **k,
    ):
        residual = hidden_states
        b, c, h, w = hidden_states.shape
        x = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)
        q, kk, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        hd = q.shape[-1] // attn.heads
        q = q.view(b, -1, attn.heads, hd).transpose(1, 2)
        kk = kk.view(b, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(b, -1, attn.heads, hd).transpose(1, 2)
        sq = q.shape[-2]
        qb = VAE_ATTN_QBLOCK if 0 < VAE_ATTN_QBLOCK < sq else sq
        outs = [
            attention(q[:, :, s0 : s0 + qb], kk, v, 1.0 / math.sqrt(hd)) for s0 in range(0, sq, qb)
        ]
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        o = o.transpose(1, 2).reshape(b, -1, attn.heads * hd).to(q.dtype)
        o = attn.to_out[1](attn.to_out[0](o))
        o = o.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            o = o + residual
        return o / attn.rescale_output_factor


class _Decode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)


class _Encode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return h


class NeuronAutoencoderKL(nn.Module):
    """Host-tensor facade: ``decode(z)`` / ``encode(x)`` with the upstream return conventions.

    Decode tiles on the host above ``Z_IMAGE_VAE_TILE`` latent px, replaying one fixed tile-shaped
    compiled decoder graph; below it, one whole-latent graph.
    """

    def __init__(self, vae, model_name: str = "z_image_vae"):
        super().__init__()
        vae.set_attn_processor(NeuronVaeAttnProcessor())
        self.vae = vae.eval()
        self.config = vae.config
        self.model_name = model_name
        self.scale = 2 ** (len(self.config.block_out_channels) - 1)
        self.tile_lat = VAE_TILE
        self.tile_overlap = VAE_TILE_OVERLAP
        self.deal = VAE_DEAL
        self._device = torch.device("cpu")
        self._dec = _Decode(self.vae)
        self._dec_fn = self._dec  # whole-latent graph
        self._tile_fn: nn.Module | None = None  # fixed tile-shaped graph
        self._enc_fn = None
        self._backend = None
        self._opts: dict = {}
        self.last_decode_s = 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: str = "vae",
        torch_dtype=torch.bfloat16,
        model_name: str = "z_image_vae",
    ):
        """``force_upcast`` in the VAE config (set for Z-Image's Flux VAE) keeps the decoder in fp32
        whatever ``torch_dtype`` is, as diffusers pipelines do: in bf16 on the NeuronCore the decoder's
        GroupNorm / conv stack drifts ~17x further from fp32 than bf16 on CPU. ``Z_IMAGE_VAE_DTYPE``
        (``float32`` / ``bfloat16``) overrides."""
        from diffusers import AutoencoderKL

        cfg = AutoencoderKL.load_config(model_path, subfolder=subfolder)
        override = os.environ.get("Z_IMAGE_VAE_DTYPE")
        if override:
            torch_dtype = getattr(torch, override)
        elif cfg.get("force_upcast"):
            torch_dtype = torch.float32
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder=subfolder, torch_dtype=torch_dtype
        )
        return cls(vae, model_name=model_name)

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if dtype is not None:
            self.vae.to(dtype)
        if device is not None:
            self._device = torch.device(device)
            self.vae.to(self._device)
        return self

    def compile(
        self, backend: str, options: dict | None = None, compile_encoder: bool = False, **kwargs
    ) -> None:
        self._backend = backend
        self._opts = dict(options or {})
        opts = {
            **self._opts,
            "model_name": self.model_name,
            "compiler_args": list(VAE_COMPILER_ARGS),
        }
        self._dec_fn = torch.compile(
            self._dec, backend=backend, options=opts, fullgraph=True, dynamic=False
        )
        topts = {
            **self._opts,
            "model_name": self.model_name + "_tile",
            "compiler_args": list(VAE_COMPILER_ARGS),
        }
        self._tile_fn = torch.compile(
            self._dec, backend=backend, options=topts, fullgraph=True, dynamic=False
        )
        if compile_encoder:
            eopts = {
                **self._opts,
                "model_name": self.model_name + "_enc",
                "compiler_args": list(VAE_COMPILER_ARGS),
            }
            self._enc_fn = torch.compile(
                _Encode(self.vae), backend=backend, options=eopts, fullgraph=True, dynamic=False
            )

    # -- host-side tiling ------------------------------------------------------------------------
    def _blend_v(self, a, b, ext):
        ext = min(a.shape[2], b.shape[2], ext)
        for y in range(ext):
            b[:, :, y, :] = a[:, :, -ext + y, :] * (1 - y / ext) + b[:, :, y, :] * (y / ext)
        return b

    def _blend_h(self, a, b, ext):
        ext = min(a.shape[3], b.shape[3], ext)
        for x in range(ext):
            b[:, :, :, x] = a[:, :, :, -ext + x] * (1 - x / ext) + b[:, :, :, x] * (x / ext)
        return b

    def _decode_tile(self, z_tile: torch.Tensor) -> torch.Tensor:
        fn = self._tile_fn if self._tile_fn is not None else self._dec
        with torch.no_grad():
            return fn(z_tile.contiguous().to(self._device)).to("cpu")

    def _decode_tiles(self, tiles: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode same-shaped tiles; dealt round-robin over the stage's ranks when there are several."""
        rank, size, group = _stage_world() if self.deal else (0, 1, None)
        if size <= 1 or group is None or len(tiles) < size:
            return [self._decode_tile(t) for t in tiles]
        import torch.distributed as dist

        mine = [self._decode_tile(t) for t in tiles[rank::size]]
        per = -(-len(tiles) // size)
        buf = torch.zeros((per, *mine[0].shape), dtype=mine[0].dtype)
        for k, t in enumerate(mine):
            buf[k] = t
        bufs = [torch.empty_like(buf) for _ in range(size)]
        dist.all_gather(bufs, buf, group=group)
        return [bufs[i % size][i // size] for i in range(len(tiles))]

    def _tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        tl = self.tile_lat
        step = max(1, int(tl * (1 - self.tile_overlap)))
        blend = int(tl * self.scale * self.tile_overlap)
        row_limit = tl * self.scale - blend
        grid, tiles = [], []
        for i in range(0, z.shape[2], step):
            for j in range(0, z.shape[3], step):
                tile = z[:, :, i : i + tl, j : j + tl]
                # pad a short edge tile up to the fixed graph shape, then crop the extra output
                ph, pw = tl - tile.shape[2], tl - tile.shape[3]
                if ph or pw:
                    tile = torch.nn.functional.pad(tile, (0, pw, 0, ph), mode="replicate")
                grid.append((i, (tl - ph) * self.scale, (tl - pw) * self.scale))
                tiles.append(tile)
        rows, prev_i = [], None
        for (i, vh, vw), dec in zip(grid, self._decode_tiles(tiles)):
            if i != prev_i:
                rows.append([])
                prev_i = i
            rows[-1].append(dec[:, :, :vh, :vw])
        out_rows = []
        for i, row in enumerate(rows):
            res = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend)
                res.append(tile[:, :, :row_limit, :row_limit])
            out_rows.append(torch.cat(res, dim=3))
        return torch.cat(out_rows, dim=2)

    def decode(self, z, return_dict: bool = True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        z = z.detach().to("cpu", self.dtype)
        t0 = time.time()
        if self.tile_lat and (z.shape[-1] > self.tile_lat or z.shape[-2] > self.tile_lat):
            out = self._tiled_decode(z)
        else:
            with torch.no_grad():
                out = self._dec_fn(z.contiguous().to(self._device)).to("cpu")
        self.last_decode_s = time.time() - t0
        return DecoderOutput(sample=out) if return_dict else (out,)

    def encode(self, x, return_dict: bool = True):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        fn = self._enc_fn if self._enc_fn is not None else _Encode(self.vae)
        with torch.no_grad():
            h = fn(x.detach().to("cpu", self.dtype).contiguous().to(self._device)).to("cpu").float()
        dist = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)
