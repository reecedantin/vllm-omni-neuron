# SPDX-License-Identifier: Apache-2.0
"""Device-tiled decode for the LTX-2 / LTX-2.5 video VAE (``AutoencoderKLLTX2Video``).

Why tiles, and why spatial only: the decoder consumes the whole latent in one call
(``vae.decoder(z, temb, causal=...)``; there is no per-frame feature cache like the Wan VAE), and
at a 512x768x121 request the full latent ``(1, 128, 16, 16, 24)`` does not compile:
``NCC_EVRF007``, 14.4M instructions against the 10M limit. One ``(16, 8)`` spatial tile compiles
(~19 min cold) and decodes warm in 0.46 s. So the decode runs as N calls of ONE compiled graph:

* every tile has exactly the compiled shape ``(TILE_H, TILE_W)`` latent; the last tile on each
  axis is pulled back to end at the boundary instead of being narrower, so no tile needs a graph
  of its own (a latent smaller than a tile is zero-padded and cropped);
* tiles overlap by ``OVERLAP`` latent units and are merged with diffusers' linear ramp
  (``AutoencoderKLLTX2Video.blend_h`` / ``blend_v``), applied over the ``OVERLAP * 32`` output
  pixels that start where the previous tile's kept region ends. For uniformly spaced tiles this is
  exactly diffusers' ``tiled_decode`` (keep ``stride`` pixels per tile, blend the first
  ``blend`` of them with the left / upper neighbour); the pinned last tile blends at the same
  place relative to what is already placed;
* the decoder runs on a device copy of ``vae.decoder``; the CPU VAE keeps its buffers
  (``latents_mean`` / ``latents_std``) for the pipeline's own de-normalisation;
* rows are merged along W first, then the row strips along H (diffusers blends V before H; the two
  orders agree everywhere except the overlap corners, and the default config has one row).

Tile-parallel across ranks (``group``, e.g. the TP group): the VAE is not TP-sharded
(onboarding-models.md §1c), but every tile is independent, so with a group of ``N`` ranks each
rank decodes tiles ``j % N == rank`` on its own core. Rank 0 broadcasts the latent, the other
ranks send their decoded tiles back over the group's CPU (gloo) group and rank 0 merges them, so
the output is bit-identical to the single-core decode. Rank 0 calls :meth:`decode`, the other
ranks :meth:`serve` at the same point of the request.
"""

from __future__ import annotations

import copy
import os

import torch

TILE_H = int(os.environ.get("LTX2_VAE_TILE_H", "16"))
TILE_W = int(os.environ.get("LTX2_VAE_TILE_W", "8"))
# 4 latent units (128 px) of overlap: on the real LTX-2.5 VAE (CPU fp32, random 16x24 latent)
# tiled vs untiled is 34.2 dB at overlap 2, 37.4 dB at 3, 39.4 dB at 4 (diffusers' own default
# 16-wide tiling: 39.9 dB). The decoder's receptive field is wider than 2 latents.
OVERLAP = int(os.environ.get("LTX2_VAE_TILE_OVERLAP", "4"))

VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]


def tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Start offsets of fixed-size tiles covering ``[0, total)``; the last one ends at ``total``."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile, stride))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def merge_1d(
    tiles: list[torch.Tensor],
    starts: list[int],
    tile: int,
    stride: int,
    total: int,
    scale: int,
    blend_px: int,
    dim: int,
) -> torch.Tensor:
    """Merge decoded tiles along ``dim`` (pixel space). ``starts`` / ``tile`` / ``stride`` /
    ``total`` are in latent units, ``scale`` = pixels per latent unit.

    Tile ``j`` contributes the output range ``[end_{j-1}, keep_end_j)``; its first ``blend``
    pixels are a linear ramp from tile ``j-1`` (0 -> 1, diffusers' ``blend_h`` weights
    ``x / blend``) where both tiles cover those positions.
    """
    pieces = []
    end_prev = 0  # absolute pixel where the already-placed output ends
    for j, (s, t) in enumerate(zip(starts, tiles, strict=True)):
        last = j == len(starts) - 1
        t0 = s * scale  # absolute pixel offset of this tile
        keep_end = (s + tile) * scale if last else (s + stride) * scale
        lo = end_prev - t0  # first column of this tile that is not placed yet
        piece = t.narrow(dim, lo, keep_end - end_prev).clone()
        if j > 0:
            prev, p0 = tiles[j - 1], starts[j - 1] * scale
            prev_end = (starts[j - 1] + tile) * scale
            n = max(0, min(blend_px, prev_end - end_prev, piece.shape[dim]))
            if n:  # linear ramp x / n over the n blended positions, in fp32, one op per seam
                w = _ramp(n, dim, piece.dim())
                a = prev.narrow(dim, end_prev - p0, n).float()
                b = piece.narrow(dim, 0, n).float()
                piece.narrow(dim, 0, n).copy_(a * (1 - w) + b * w)
        pieces.append(piece)
        end_prev = keep_end
    out = torch.cat(pieces, dim=dim)
    return out.narrow(dim, 0, total * scale)


class TiledLTX2VideoDecoder:
    """Fixed-shape tiled decode of an ``AutoencoderKLLTX2Video`` latent on one Neuron core."""

    def __init__(
        self,
        vae,
        device: torch.device | str = "cpu",
        tile_h: int = TILE_H,
        tile_w: int = TILE_W,
        overlap: int = OVERLAP,
        group=None,
    ):
        self.vae = vae
        self.device = torch.device(device)
        self.scale = int(getattr(vae, "spatial_compression_ratio", 32))
        self.tile_h, self.tile_w, self.overlap = tile_h, tile_w, overlap
        self.decoder = (
            vae.decoder if self.device.type == "cpu" else copy.deepcopy(vae.decoder).to(self.device)
        )
        self._fn = None
        # vLLM GroupCoordinator to spread the tiles over (rank 0 merges), or None: one core
        self.group = group if group is not None and group.world_size > 1 else None

    def compile(self, backend: str) -> None:
        self._fn = torch.compile(
            self._decode_tile,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options={
                "model_name": "ltx2_vae_decode_tile",
                "compiler_args": list(VAE_COMPILER_ARGS),
            },
        )

    def _decode_tile(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, None)

    def _run(self, z_tile: torch.Tensor) -> torch.Tensor:
        fn = self._fn or self._decode_tile
        return fn(z_tile.to(self.device).contiguous()).cpu()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z [B, C, F, H, W]`` latent -> ``[B, 3, F', H*32, W*32]`` pixels in [-1, 1] (on CPU).

        With a ``group``, this is rank 0's side: the other ranks must call :meth:`serve`."""
        if self.group is not None:
            z = self._broadcast_latent(z)
        return self._decode(z)

    @torch.no_grad()
    def serve(self) -> None:
        """Non-zero ranks of ``group``: receive the latent, decode this rank's tiles, send them."""
        self._decode(self._broadcast_latent(None))

    _DTYPES = (torch.float32, torch.bfloat16, torch.float16)

    def _broadcast_latent(self, z: torch.Tensor | None) -> torch.Tensor:
        import torch.distributed as dist

        g, src = self.group.cpu_group, self.group.ranks[0]
        meta = torch.zeros(6, dtype=torch.int64)
        if z is not None:
            meta[:5] = torch.tensor(z.shape)
            meta[5] = self._DTYPES.index(z.dtype)
        dist.broadcast(meta, src=src, group=g)
        shape, dtype = tuple(int(x) for x in meta[:5]), self._DTYPES[int(meta[5])]
        buf = z.contiguous() if z is not None else torch.empty(shape, dtype=dtype)
        dist.broadcast(buf.view(-1).view(torch.uint8), src=src, group=g)
        return buf

    def _decode(self, z: torch.Tensor) -> torch.Tensor | None:
        b, c, f, h, w = z.shape
        th, tw = self.tile_h, self.tile_w
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        if pad_h or pad_w:
            z = torch.nn.functional.pad(z, (0, pad_w, 0, pad_h))
        hh, ww = h + pad_h, w + pad_w
        sh, sw = max(th - self.overlap, 1), max(tw - self.overlap, 1)
        rows_s, cols_s = tile_starts(hh, th, sh), tile_starts(ww, tw, sw)
        pos = [(r, cc) for r in rows_s for cc in cols_s]
        world = self.group.world_size if self.group is not None else 1
        rank = self.group.rank_in_group if self.group is not None else 0
        mine = {
            j: self._run(z[:, :, :, r : r + th, cc : cc + tw])
            for j, (r, cc) in enumerate(pos)
            if j % world == rank
        }
        if rank != 0:
            import torch.distributed as dist

            dst, g = self.group.ranks[0], self.group.cpu_group
            for j in sorted(mine):
                dist.send(mine[j].contiguous().view(-1).view(torch.uint8), dst=dst, group=g)
            return None
        tiles = []
        for j in range(len(pos)):
            if j not in mine:  # rank 0 always owns tile 0, so the shape/dtype are known
                import torch.distributed as dist

                t = torch.empty_like(mine[0])
                dist.recv(
                    t.view(-1).view(torch.uint8),
                    src=self.group.ranks[j % world],
                    group=self.group.cpu_group,
                )
                mine[j] = t
            tiles.append(mine[j])
        blend_px = self.overlap * self.scale
        strips = []
        for i, _r in enumerate(rows_s):
            row = tiles[i * len(cols_s) : (i + 1) * len(cols_s)]
            strips.append(merge_1d(row, cols_s, tw, sw, ww, self.scale, blend_px, dim=4))
        out = merge_1d(strips, rows_s, th, sh, hh, self.scale, blend_px, dim=3)
        return out[:, :, :, : h * self.scale, : w * self.scale]

    def install(self, vae=None) -> None:
        """Route ``vae.decode`` (the call the diffusers pipeline makes) through the tiled decoder."""
        vae = vae or self.vae
        from diffusers.models.autoencoders.vae import DecoderOutput

        dump_dir = os.environ.get("LTX2_VAE_DUMP_DIR")  # parity: save the latent + tiled output

        def decode(z, temb=None, causal=None, return_dict=True):
            out = self.decode(z).to(vae.dtype)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(
                    {"latent": z.detach().cpu(), "tiled": out.detach().cpu()},
                    os.path.join(dump_dir, "vae_decode_dump.pt"),
                )
            return (out,) if not return_dict else DecoderOutput(sample=out)

        vae.decode = decode


def _ramp(n: int, dim: int, ndim: int) -> torch.Tensor:
    """``x / n`` for ``x`` in ``[0, n)`` along ``dim`` of an ``ndim``-d tensor (fp32)."""
    shape = [1] * ndim
    shape[dim] = n
    return (torch.arange(n, dtype=torch.float32) / n).view(shape)
