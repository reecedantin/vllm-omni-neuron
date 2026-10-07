# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 VAE decode on NeuronCores: untiled up to a size threshold, tiled with whole-image
GroupNorm statistics above it.

``AutoencoderKLFlux2`` is a plain conv ResNet VAE (4 up blocks, one bottleneck self-attention).

* **Untiled** (latent side <= ``FLUX2_VAE_UNTILED_MAX``): the whole ``_decode`` as one graph per
  resolution. Exact vs the CPU decode up to bf16, 1.38 s warm at 1024 px, but a ~61 min / 388 GB
  cold compile per new resolution -- only worth it for a resolution that is compiled once and cached.
* **Tiled**: fixed-shape latent tiles with overlap (``vllm_omni_neuron.diffusion.layers.vae_tiling``), one NEFF
  per pass for any resolution, tiles dealt round-robin over all ranks (TP x CP), merged on rank 0 with
  diffusers' linear blend. Plain tiling normalises every GroupNorm with *per-tile* statistics, which
  shifts each tile's colour/brightness (24 dB vs untiled on real weights; visible tile rectangles).
  So decode runs in passes (``FLUX2_VAE_GN_PASSES``, default 2):

  1. *collect*: decode every tile, and at each GroupNorm record ``sum``, ``sum of squares`` and the
     element count over the part of the tile that the tile OWNS in the final image (its kept range,
     a disjoint partition of the image), while normalising with local statistics;
  2. combine across tiles and ranks (host, fp64) into whole-image mean / variance per GroupNorm
     layer and group, broadcast to every rank;
  3. *apply*: decode every tile again, normalising every GroupNorm with the whole-image statistics.

  With ``FLUX2_VAE_GN_PASSES=3`` the middle pass is *collect under global statistics* (stats are
  re-collected from activations that were themselves normalised globally), a fixed-point refinement.
  The bottleneck self-attention stays tile-local (its GroupNorm is global).

Knobs: ``FLUX2_VAE_TILE`` latent tile (default 64 = 512 px), ``FLUX2_VAE_OVERLAP`` latent overlap
(default 16 = 128 px), ``FLUX2_VAE_UNTILED_MAX`` largest latent side decoded untiled (default 0 =
always tiled), ``FLUX2_VAE_GN_PASSES`` (1 = plain tiling, 2 default, 3), ``FLUX2_VAE_DEVICE=0``
host-CPU decode, ``FLUX2_VAE_TILE_PARALLEL=0`` rank 0 decodes every tile, ``FLUX2_VAE_DUMP=path``
saves ``{latents, image}`` from rank 0 (parity checks); ``FLUX2_RANK_DUMP=dir``: see ``decode``.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni_neuron.diffusion.layers.vae_tiling import (
    TileGrid,
    kept_ranges,
    merge_tiles,
    run_tiles,
)

from .ops import log_info

logger = init_logger(__name__)

VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


class _GNCtx:
    """Per-call state shared by the wrapped GroupNorm layers (set inside the compiled pass)."""

    def __init__(self):
        self.mode = "local"  # local | collect | collect_glob | apply
        self.mask = None  # [1, 1, T, T] fp32 kept-range mask at latent-tile resolution
        self.glob = None  # [L, B, G, 2] (mean, var) fp32
        self.out: list = []
        self.i = 0


class _GN(nn.Module):
    """GroupNorm that can record masked statistics and/or normalise with external statistics."""

    def __init__(self, gn: nn.GroupNorm, ctx: _GNCtx):
        super().__init__()
        self.num_groups, self.eps, self.affine = gn.num_groups, gn.eps, gn.affine
        self.weight, self.bias = gn.weight, gn.bias
        object.__setattr__(self, "ctx", ctx)

    def _mask_like(self, x):
        m = self.ctx.mask
        if x.ndim == 4:
            return F.interpolate(m, size=x.shape[-2:], mode="nearest").reshape(1, 1, 1, -1)
        return m.reshape(1, 1, 1, -1)  # [B, C, L] attention input: L = latent tile pixels

    def forward(self, x):
        ctx = self.ctx
        if ctx.mode == "local":
            # exactly nn.GroupNorm's call, so the untiled graph matches the NEFF already built for it
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        b, c = x.shape[:2]
        g = self.num_groups
        xf = x.float().reshape(b, g, c // g, -1)
        if ctx.mode in ("collect", "collect_glob"):
            m = self._mask_like(x)
            s1 = (xf * m).sum((2, 3))
            s2 = (xf * xf * m).sum((2, 3))
            n = (m.sum() * (c // g)).expand_as(s1)
            ctx.out.append(torch.stack([s1, s2, n], dim=-1))
        if ctx.mode == "collect":
            mean = xf.mean((2, 3), keepdim=True)
            var = xf.var((2, 3), unbiased=False, keepdim=True)
        else:
            st = ctx.glob[ctx.i]
            ctx.i += 1
            mean, var = st[..., 0, None, None], st[..., 1, None, None]
        y = ((xf - mean) * torch.rsqrt(var + self.eps)).reshape(x.shape)
        if self.affine:
            shape = (1, c) + (1,) * (x.ndim - 2)
            y = y * self.weight.float().reshape(shape) + self.bias.float().reshape(shape)
        return y.to(x.dtype)


class NeuronFlux2Vae(nn.Module):
    """Device-decoding facade over ``AutoencoderKLFlux2``."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae.config
        self.bn = vae.bn
        self.ratio = 2 ** (len(vae.config.block_out_channels) - 1)
        self._device = torch.device("cpu")
        self._use_device = os.environ.get("FLUX2_VAE_DEVICE", "1") == "1"
        self.tile = int(os.environ.get("FLUX2_VAE_TILE", "64"))
        self.overlap = int(os.environ.get("FLUX2_VAE_OVERLAP", "16"))
        self.untiled_max = int(os.environ.get("FLUX2_VAE_UNTILED_MAX", "0"))
        self.gn_passes = int(os.environ.get("FLUX2_VAE_GN_PASSES", "2"))
        self.tile_parallel = os.environ.get("FLUX2_VAE_TILE_PARALLEL", "1") == "1"
        self.ctx = _GNCtx()
        self.n_gn = self._wrap_group_norms()
        self._fns = {
            k: getattr(self, f"_{k}")
            for k in ("untiled", "tile_local", "collect", "collect_glob", "apply")
        }
        self.stats = {"calls": 0, "seconds": 0.0}

    def _wrap_group_norms(self) -> int:
        n = 0
        for parent in list(self.vae.decoder.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.GroupNorm):
                    setattr(parent, name, _GN(child, self.ctx))
                    n += 1
        return n

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    def encode(self, x, return_dict=True):
        # conditioning-image path (I2I / editing), not on the text-to-image hot path: eager.
        return self.vae.encode(x.to(self.vae.dtype), return_dict=return_dict)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and self._use_device:
            self._device = torch.device(device)
            self.vae.to(self._device)
        return self

    # -- graph bodies (each compiled separately: one NEFF per body and tile shape) ------------
    def _decode_eager(self, z):
        if self.vae.post_quant_conv is not None:
            z = self.vae.post_quant_conv(z)
        return self.vae.decoder(z)

    def _untiled(self, z):
        self.ctx.mode = "local"
        return self._decode_eager(z)

    def _tile_local(self, z):
        self.ctx.mode = "local"
        return self._decode_eager(z)

    def _collect(self, z, mask):
        ctx = self.ctx
        ctx.mode, ctx.mask, ctx.out = "collect", mask, []
        self._decode_eager(z)
        out = torch.stack(ctx.out)
        ctx.mode, ctx.mask, ctx.out = "local", None, []
        return out

    def _collect_glob(self, z, mask, glob):
        ctx = self.ctx
        ctx.mode, ctx.mask, ctx.glob, ctx.out, ctx.i = "collect_glob", mask, glob, [], 0
        self._decode_eager(z)
        out = torch.stack(ctx.out)
        ctx.mode, ctx.mask, ctx.glob, ctx.out, ctx.i = "local", None, None, [], 0
        return out

    def _apply(self, z, glob):
        ctx = self.ctx
        ctx.mode, ctx.glob, ctx.i = "apply", glob, 0
        y = self._decode_eager(z)
        ctx.mode, ctx.glob, ctx.i = "local", None, 0
        return y

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        if not self._use_device:
            return
        for key in list(self._fns):
            # "untiled" keeps the earlier model_name so an already-built 1024 px NEFF is reused
            name = "flux2_vae" if key == "untiled" else f"flux2_vae_{key}_t{self.tile}"
            opts = {**(options or {}), "model_name": name, "compiler_args": list(VAE_COMPILER_ARGS)}
            self._fns[key] = torch.compile(
                getattr(self, f"_{key}"),
                backend=backend,
                options=opts,
                fullgraph=kwargs.get("fullgraph", True),
                dynamic=False,
            )

    # -- host orchestration --------------------------------------------------------------------
    def _dev(self, t):
        return t.to(self._device).contiguous()

    def _masks(self, grid: TileGrid):
        kept = [
            kept_ranges(grid.starts[a], grid.tile[a], grid.stride[a], grid.total[a])
            for a in range(2)
        ]
        out = {}
        for idx in grid.indices():
            m = torch.zeros(1, 1, *grid.tile, dtype=torch.float32)
            (b0, e0), (b1, e1) = kept[0][idx[0]], kept[1][idx[1]]
            s0, s1 = grid.starts[0][idx[0]], grid.starts[1][idx[1]]
            m[..., b0 - s0 : e0 - s0, b1 - s1 : e1 - s1] = 1.0
            out[idx] = m
        return out

    @staticmethod
    def _combine(per_tile: dict) -> torch.Tensor:
        """``{idx: [L, B, G, 3] (sum, sumsq, n)}`` -> ``[L, B, G, 2] (mean, var)`` fp32."""
        tot = sum(t.double() for t in per_tile.values())
        mean = tot[..., 0] / tot[..., 2]
        var = (tot[..., 1] / tot[..., 2] - mean * mean).clamp_min(0)
        return torch.stack([mean, var], dim=-1).float()

    def _bcast(self, obj, parallel: bool):
        if not parallel:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _decode_tiled(self, z, parallel: bool):
        b, _, h, w = z.shape
        r = self.ratio
        tile = min(self.tile, max(h, w))
        stride = max(1, tile - self.overlap)
        grid = TileGrid.for_axes(
            total=(h, w), tile=(tile, tile), stride=(stride, stride), out_scale=r
        )
        zp = grid.pad_input(z)
        kw = {} if parallel else {"world_size": 1, "rank": 0}
        tiles = {idx: grid.slice_input(zp, idx) for idx in grid.indices()}
        if self.gn_passes <= 1:
            res = run_tiles(
                grid, lambda n, idx: self._fns["tile_local"](self._dev(tiles[idx])).to("cpu"), **kw
            )
        else:
            masks = self._masks(grid)
            stats = run_tiles(
                grid,
                lambda n, idx: self._fns["collect"](
                    self._dev(tiles[idx]), self._dev(masks[idx])
                ).to("cpu"),
                **kw,
            )
            glob = self._bcast(self._combine(stats) if stats is not None else None, parallel)
            for _ in range(self.gn_passes - 2):
                gd = self._dev(glob)
                stats = run_tiles(
                    grid,
                    lambda n, idx: self._fns["collect_glob"](
                        self._dev(tiles[idx]), self._dev(masks[idx]), gd
                    ).to("cpu"),
                    **kw,
                )
                glob = self._bcast(self._combine(stats) if stats is not None else None, parallel)
            gd = self._dev(glob)
            res = run_tiles(
                grid, lambda n, idx: self._fns["apply"](self._dev(tiles[idx]), gd).to("cpu"), **kw
            )
        if res is None:
            return None, grid
        return merge_tiles(res, grid)[..., : h * r, : w * r].contiguous(), grid

    def decode(self, z, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        b, _, h, w = z.shape
        r = self.ratio
        empty = torch.zeros(b, self.config.out_channels, h * r, w * r, dtype=self.dtype)
        wrap = (lambda o: DecoderOutput(sample=o)) if return_dict else (lambda o: (o,))
        z = z.detach().to("cpu", self.dtype).contiguous()
        rank_dump = os.environ.get("FLUX2_RANK_DUMP")
        if rank_dump:  # all-rank agreement checks: every rank's denoised latents, as fed to the VAE
            os.makedirs(rank_dump, exist_ok=True)
            torch.save(
                {"latents": z.float(), "rank": _rank()},
                os.path.join(rank_dump, f"rank{_rank():02d}.pt"),
            )
        if not self._use_device:
            if _rank() != 0:
                return wrap(empty)
            self.ctx.mode = "local"
            with torch.no_grad():
                return wrap(self._decode_eager(z).float().to(self.dtype))
        t0 = time.time()
        parallel = self.tile_parallel and dist.is_initialized() and dist.get_world_size() > 1
        with torch.no_grad():
            if max(h, w) <= self.untiled_max:
                if _rank() != 0:
                    return wrap(empty)
                out, how = self._fns["untiled"](self._dev(z)).to("cpu"), "untiled"
            else:
                if not parallel and _rank() != 0:
                    return wrap(empty)
                out, grid = self._decode_tiled(z, parallel)
                if out is None:
                    return wrap(empty)
                how = (
                    f"{grid.num_tiles} tiles of {grid.tile[0] * r}px, overlap {self.overlap * r}px, "
                    f"GN passes {self.gn_passes}, {'tile-parallel' if parallel else 'rank 0'}"
                )
        self.stats["calls"] += 1
        self.stats["seconds"] += time.time() - t0
        log_info(
            "flux2 VAE decode %dx%d on device: %.2fs (%s)", h * r, w * r, time.time() - t0, how
        )
        dump = os.environ.get("FLUX2_VAE_DUMP")
        if dump:
            torch.save({"latents": z.float(), "image": out.float()}, dump)
        return wrap(out)
