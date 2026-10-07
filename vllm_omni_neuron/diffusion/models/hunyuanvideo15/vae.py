# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo-1.5 VAE decode on Neuron.

diffusers' ``AutoencoderKLHunyuanVideo15`` decoder is a causal 3D VAE (16x spatial, 4x temporal):
``conv_in -> mid block (resnets + frame-causal self-attention over the whole clip) -> 5 up blocks
(causal-conv resnets, two of them with a 2x temporal upsampler) -> norm/act/conv_out``. Its norms are
per-position RMS norms over channels (no GroupNorm), so spatial tiles need no shared statistics.

Device decode (default), in three nested levels:

* **Spatial tiles** on the shared fixed-shape grid (``vllm_omni_neuron.diffusion.layers.vae_tiling``):
  every tile is exactly ``HV15_VAE_TILE`` latent units (default 11 = 176 px), one compiled shape per
  graph role, and the tiles are dealt round-robin across all ranks of the world (``run_tiles``) and
  merged on rank 0 with diffusers' linear blend.
* **Whole-clip head of the decoder** per tile: ``conv_in`` + the mid block run on all latent frames at
  latent resolution (the attention is causal over the whole clip, so it is not chunked; at 7 latent
  frames x 12 x 12 it is ~1k tokens).
* **Causal temporal chunks** for the up path: one latent frame at a time through the five up blocks and
  the output head, with every causal 3D conv's 2-frame left context carried between chunks as explicit
  graph inputs/outputs, one graph per up block (``HV15_VAE_UP_SPLIT``) in two variants: ``first`` (the
  clip's first frame: replicate padding and the special frame-0 temporal upsample, as diffusers) and
  ``rest`` (every later frame: cached context, uniform 2x upsample). This equals the whole-clip forward
  exactly (``test_hunyuanvideo15_vae.py``) and bounds every graph to one up block on one latent frame
  regardless of the clip length.

``HV15_VAE_HOST=1`` decodes on the host instead (diffusers' decoder in bf16 on the same tile grid,
``HV15_VAE_HOST_THREADS`` threads, rank 0 only).
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]
MASK_VALUE = -30000.0


# ---------------------------------------------------------------------------------------------
# traceable replacements for diffusers ops
# ---------------------------------------------------------------------------------------------
def causal_frame_mask(
    n_frame: int, n_hw: int, dtype, device, batch_size: int | None = None
) -> torch.Tensor:
    """Drop-in for ``HunyuanVideo15AttnBlock.prepare_causal_attention_mask`` (token i sees every
    token of frames <= its own), without the per-token Python loop (finite mask value)."""
    frame = torch.arange(n_frame * n_hw, device=device) // n_hw
    mask = torch.where(frame[None, :] <= frame[:, None], 0.0, MASK_VALUE).to(dtype)
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def replicate_pad_3d(x: torch.Tensor, pad: tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """``F.pad(x, pad, mode="replicate")`` on ``[B, C, T, H, W]`` built from slices + ``cat``.

    neuronx-cc 2.27 mis-lowers the replicate pad of a 256 px VAE tile (``NCC_EBIR033`` DMA
    access-pattern mismatch); edge-slice concatenation is exactly equivalent and compiles."""
    wl, wr, hl, hr, tl, tr = pad
    for dim, lo, hi in ((4, wl, wr), (3, hl, hr), (2, tl, tr)):
        parts = []
        if lo:
            parts.append(
                x.narrow(dim, 0, 1).expand(*[lo if d == dim else -1 for d in range(x.dim())])
            )
        parts.append(x)
        if hi:
            parts.append(
                x.narrow(dim, x.shape[dim] - 1, 1).expand(
                    *[hi if d == dim else -1 for d in range(x.dim())]
                )
            )
        if len(parts) > 1:
            x = torch.cat(parts, dim=dim)
    return x


def _causal_conv_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.pad_mode == "replicate":
        return self.conv(replicate_pad_3d(hidden_states, self.time_causal_padding))
    return self.conv(F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode))


def _install_patches() -> None:
    from diffusers.models.autoencoders import autoencoder_kl_hunyuanvideo15 as m

    m.HunyuanVideo15AttnBlock.prepare_causal_attention_mask = staticmethod(causal_frame_mask)
    m.HunyuanVideo15CausalConv3d.forward = _causal_conv_forward


# ---------------------------------------------------------------------------------------------
# causal temporal chunking of the up path (explicit conv caches)
# ---------------------------------------------------------------------------------------------
def up_path_convs(dec) -> list:
    """Every causal 3D conv of the up path + output head, in forward order (the cache slots)."""
    return block_convs(dec, 0, len(dec.up_blocks))


def _cached_conv(conv, x, cache):
    """Causal conv with explicit temporal context. ``cache=None``: the clip's first chunk (replicate
    the first frame, as diffusers' padding does). Returns ``(out, new_cache)``, ``new_cache`` = the
    last ``kt - 1`` frames of the padded input (this conv's context for the next chunk)."""
    wl, wr, hl, hr, tl, _ = conv.time_causal_padding
    ctx = x.narrow(2, 0, 1).expand(-1, -1, tl, -1, -1) if cache is None else cache
    xp = torch.cat([ctx, x], dim=2)
    new_cache = xp.narrow(2, xp.shape[2] - tl, tl)
    return conv.conv(replicate_pad_3d(xp, (wl, wr, hl, hr, 0, 0))), new_cache


def _resnet(res, x, c1, c2):
    h, n1 = _cached_conv(res.conv1, res.nonlinearity(res.norm1(x)), c1)
    h, n2 = _cached_conv(res.conv2, res.nonlinearity(res.norm2(h)), c2)
    sc = res.conv_shortcut(x) if res.conv_shortcut is not None else x
    return h + sc, n1, n2


def _upsample(ups, x, cache, first: bool):
    h, new = _cached_conv(ups.conv, x, cache)
    rearr = ups._dcae_upsample_rearrange
    if not ups.add_temporal_upsample:
        return rearr(h, r1=1, r2=2, r3=2) + rearr(
            x.repeat_interleave(ups.repeats, dim=1), r1=1, r2=2, r3=2
        ), new
    if not first:  # no frame of this chunk is the clip's frame 0: uniform 2x temporal upsample
        return rearr(h, r1=2, r2=2, r3=2) + rearr(
            x.repeat_interleave(ups.repeats, dim=1), r1=2, r2=2, r3=2
        ), new
    # the clip's first chunk: frame 0 gets diffusers' special treatment, frames 1.. the uniform one
    hf = rearr(h[:, :, :1], r1=1, r2=2, r3=2)
    hf = hf[:, : hf.shape[1] // 2]
    xf = rearr(x[:, :, :1], r1=1, r2=2, r3=2).repeat_interleave(ups.repeats // 2, dim=1)
    out = hf + xf
    if x.shape[2] > 1:
        hn = rearr(h[:, :, 1:], r1=2, r2=2, r3=2)
        xn = rearr(x[:, :, 1:], r1=2, r2=2, r3=2).repeat_interleave(ups.repeats, dim=1)
        out = torch.cat([out, hn + xn], dim=2)
    return out, new


class DecoderTrunk(nn.Module):
    """``conv_in`` + mid block on the whole clip (latent resolution). ``[B, z, t, h, w] -> [B, C, t, h, w]``."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z):
        d = self.dec
        return d.mid_block(d.conv_in(z) + z.repeat_interleave(repeats=d.repeat, dim=1))


class UpPathChunk(nn.Module):
    """Up blocks ``[start, end)`` (+ the output head when ``end`` is the last block) on one chunk of
    latent frames. ``first=True``: the clip's first chunk (``forward(x)``); ``first=False``: a later
    chunk (``forward(x, *caches)``). Returns ``(x, *new_caches)``, one cache per causal conv of the
    covered blocks (:func:`block_convs`). The default covers the whole up path; the device path uses
    one module per up block (``HV15_VAE_UP_SPLIT``) to bound each graph's instruction count."""

    def __init__(self, dec, first: bool, start: int = 0, end: int | None = None):
        super().__init__()
        self.dec, self.first = dec, first
        self.start, self.end = start, len(dec.up_blocks) if end is None else end
        self.with_head = self.end == len(dec.up_blocks)

    def n_caches(self) -> int:
        return len(block_convs(self.dec, self.start, self.end))

    def forward(self, x, *caches):
        d = self.dec
        caches = list(caches) if not self.first else [None] * self.n_caches()
        new: list[torch.Tensor] = []
        k = 0
        for blk in d.up_blocks[self.start : self.end]:
            for res in blk.resnets:
                x, n1, n2 = _resnet(res, x, caches[k], caches[k + 1])
                new += [n1, n2]
                k += 2
            for ups in blk.upsamplers or []:
                x, n = _upsample(ups, x, caches[k], self.first)
                new.append(n)
                k += 1
        if self.with_head:
            x, n = _cached_conv(d.conv_out, d.conv_act(d.norm_out(x)), caches[k])
            new.append(n)
        return (x, *new)


def block_convs(dec, start: int, end: int) -> list:
    """Causal convs of up blocks ``[start, end)`` (+ ``conv_out`` if ``end`` is the last block)."""
    convs = []
    for blk in dec.up_blocks[start:end]:
        for res in blk.resnets:
            convs += [res.conv1, res.conv2]
        for ups in blk.upsamplers or []:
            convs.append(ups.conv)
    if end == len(dec.up_blocks):
        convs.append(dec.conv_out)
    return convs


class UpPathPipeline:
    """The up path as a chain of :class:`UpPathChunk` segments (``bounds`` = block boundaries), each
    with its own ``first`` / ``rest`` callables and cache slice: one call per segment per chunk."""

    def __init__(self, dec, bounds: list[int], wrap=lambda m, name: m):
        self.segments = []
        for a, b in zip(bounds, bounds[1:]):
            self.segments.append(
                (
                    wrap(UpPathChunk(dec, True, a, b), f"up{a}-{b}_first"),
                    wrap(UpPathChunk(dec, False, a, b), f"up{a}-{b}_rest"),
                    len(block_convs(dec, a, b)),
                )
            )

    def first(self, x):
        caches = []
        for f, _, _ in self.segments:
            res = f(x)
            x = res[0]
            caches += list(res[1:])
        return (x, *caches)

    def rest(self, x, *caches):
        new, k = [], 0
        for _, r, n in self.segments:
            res = r(x, *caches[k : k + n])
            x = res[0]
            new += list(res[1:])
            k += n
        return (x, *new)


def decode_causal_chunks(trunk, first_fn, rest_fn, z: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    """Whole-clip trunk, then the up path ``chunk`` latent frames at a time with carried caches.
    ``trunk`` / ``first_fn`` / ``rest_fn`` are :class:`DecoderTrunk` / :class:`UpPathChunk` (compiled
    or eager). Equals ``decoder(z)``. Temporal slicing happens on the HOST (the trunk output is small,
    latent resolution): an eager ``narrow`` on a device tensor is not supported by the runtime."""
    dev = z.device
    h = trunk(z).to("cpu")
    t = h.shape[2]

    def piece(s, n):
        return h.narrow(2, s, n).contiguous().to(dev)

    outs = []
    res = first_fn(piece(0, min(chunk, t)))
    outs.append(res[0].to("cpu"))
    caches = res[1:]
    for s in range(chunk, t, chunk):
        n = min(chunk, t - s)
        if n < chunk:  # a short tail would be another graph shape: run it frame by frame
            for f in range(n):
                res = rest_fn(piece(s + f, 1), *caches)
                outs.append(res[0].to("cpu"))
                caches = res[1:]
            break
        res = rest_fn(piece(s, n), *caches)
        outs.append(res[0].to("cpu"))
        caches = res[1:]
    return torch.cat(outs, dim=2)


# ---------------------------------------------------------------------------------------------
def _maybe_dump_latents(z: torch.Tensor, scaling_factor: float) -> None:
    """``HV15_DUMP_LATENTS=<file>``: save the denoised latents (scaling undone) on global rank 0.
    ``HV15_RANK_DIGEST_DIR=<dir>``: every rank writes a digest of ITS latents there
    (:mod:`vllm_omni_neuron.testing.rank_agreement`), the all-rank agreement check of the gates.

    The served pipeline does not return latents, so the layout accuracy check reads them from here."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    ddir = os.environ.get("HV15_RANK_DIGEST_DIR")
    if ddir:
        from vllm_omni_neuron.testing.rank_agreement import write_rank_digest

        write_rank_digest(ddir, rank, {"latents": z.detach().float().cpu()})
    path = os.environ.get("HV15_DUMP_LATENTS")
    if not path or rank != 0:
        return
    torch.save(z.detach().float().cpu() * scaling_factor, path)


def sync_latents(z: torch.Tensor, group) -> torch.Tensor:
    """DEBUG (``HV15_SYNC_LATENTS=1``): check that every rank of ``group`` holds rank 0's latents
    before the device tile-parallel decode (every rank decodes tiles of its OWN copy and rank 0
    merges), log the ranks that disagree (checksums) and decode rank 0's copy. Off by default: the
    ranks agree by construction (CP / CFG gathers in coordinator order), and a disagreement means a
    bug upstream of the VAE that this would only hide."""
    import torch.distributed as dist

    if os.environ.get("HV15_SYNC_LATENTS", "0") != "1":
        return z
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return z
    zc = z.detach().float().cpu().contiguous()
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    sums = [None] * world
    dist.all_gather_object(sums, (float(zc.sum()), float(zc.abs().sum())), group=group)
    if rank == 0:
        bad = [r for r, s in enumerate(sums) if s != sums[0]]
        if bad:
            logger.warning(
                "hunyuanvideo15 VAE: ranks %s held different latents than rank 0 (checksums %s); "
                "decoding rank 0's",
                bad,
                sums,
            )
            if os.environ.get("HV15_PROFILE", "0") == "1":
                print(f"[hv15-prof] vae_latent_mismatch ranks={bad} sums={sums}", flush=True)
    dist.broadcast(zc, src=dist.get_global_rank(group, 0), group=group)
    return zc.to(z.dtype)


class NeuronHunyuanVideo15VAE(nn.Module):
    """Host-tensor facade over diffusers' VAE with a tiled, time-chunked compiled decoder."""

    @classmethod
    def from_pretrained(cls, model_path, subfolder="vae", torch_dtype=torch.bfloat16, **kwargs):
        from diffusers import AutoencoderKLHunyuanVideo15

        vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            model_path, subfolder=subfolder, torch_dtype=torch_dtype, **kwargs
        )
        return cls(vae.eval())

    def __init__(self, vae):
        super().__init__()
        _install_patches()
        # diffusers sets add_temporal_up/downsample from numpy comparisons (np.bool_), which Dynamo
        # treats as data-dependent branching; make them Python bools
        for mod in vae.modules():
            for attr in ("add_temporal_upsample", "add_temporal_downsample"):
                if hasattr(mod, attr):
                    setattr(mod, attr, bool(getattr(mod, attr)))
        self.vae = vae
        self.config = vae.config
        self.spatial_compression_ratio = vae.spatial_compression_ratio
        self.temporal_compression_ratio = vae.temporal_compression_ratio
        self.host = os.environ.get("HV15_VAE_HOST", "0") == "1"
        # latent units. Device default 11 (176 px): at 12 (192 px) the full-res 128-channel up block fails
        # with NCC_IBTN020 (access-pattern step out of int16 range; a padded 194x194 plane is 37636 > 32767)
        self.tile = int(os.environ.get("HV15_VAE_TILE", "16" if self.host else "11"))
        self.overlap = int(os.environ.get("HV15_VAE_OVERLAP", str(max(self.tile // 4, 1))))
        self.chunk = int(os.environ.get("HV15_VAE_TCHUNK", "1"))  # latent frames per up-path call
        self._device = torch.device("cpu")
        dec = vae.decoder
        # up-path segment boundaries (block indices); default one graph per up block -- the whole up
        # path for one latent frame of a 192 px tile is 13.8M instructions, past a practical compile
        n_up = len(dec.up_blocks)
        spec = os.environ.get("HV15_VAE_UP_SPLIT", ",".join(str(i) for i in range(n_up + 1)))
        self.up_bounds = sorted({0, n_up, *(int(b) for b in spec.split(",") if b.strip())})
        object.__setattr__(self, "_trunk", DecoderTrunk(dec))
        object.__setattr__(self, "_up", UpPathPipeline(dec, self.up_bounds))
        self.skip_decode = False  # set by the pipeline for host decode on ranks != 0
        self.tile_group = None  # host process group for device tile-parallel decode (None = world)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and torch.device(device).type != "cpu" and not self.host:
            self._device = torch.device(device)
            self.vae.decoder.to(self._device)
        return self  # host mode: the decoder stays on CPU

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """Image-to-video first-frame condition: diffusers' encoder on the host (it never moves to the
        device; one frame per request), threads raised like the host decode."""
        prev = torch.get_num_threads()
        torch.set_num_threads(int(os.environ.get("HV15_VAE_HOST_THREADS", "16")))
        try:
            with torch.no_grad():
                return self.vae.encode(
                    x.to("cpu", dtype=next(self.vae.encoder.parameters()).dtype),
                    return_dict=return_dict,
                )
        finally:
            torch.set_num_threads(prev)

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        """Per tile shape: the trunk (whole clip, latent res) and, per up-path segment, a ``first``
        and a ``rest`` chunk graph. ``fullgraph`` is used because each is a plain tensor function
        (the tiling and chunk loops run on the host)."""
        if self.host:
            return
        import torch._dynamo.config as dcfg

        # every UpPathChunk instance shares one forward code object, so Dynamo counts all of their
        # specializations (2 per segment) against one recompile budget (default 8): raise it
        need = 2 * len(self.up_bounds) + 8
        for key in ("recompile_limit", "cache_size_limit"):
            if hasattr(dcfg, key) and getattr(dcfg, key) < need:
                setattr(dcfg, key, need)

        def comp(mod, name):
            opts = {
                **(options or {}),
                "model_name": f"hunyuanvideo15_vae_{name}",
                "compiler_args": list(VAE_COMPILER_ARGS),
            }
            return torch.compile(mod, backend=backend, options=opts, fullgraph=True, dynamic=False)

        object.__setattr__(self, "_trunk", comp(DecoderTrunk(self.vae.decoder), "trunk"))
        object.__setattr__(self, "_up", UpPathPipeline(self.vae.decoder, self.up_bounds, wrap=comp))

    # -- decode -----------------------------------------------------------------------------------
    def _grid(self, h: int, w: int):
        from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid

        th, tw = min(self.tile, h), min(self.tile, w)
        sh, sw = max(th - self.overlap, 1), max(tw - self.overlap, 1)
        return TileGrid.for_axes(
            total=(h, w), tile=(th, tw), stride=(sh, sw), out_scale=self.spatial_compression_ratio
        )

    def _decode_tile(self, z_tile: torch.Tensor) -> torch.Tensor:
        x = z_tile.to(self.dtype).contiguous().to(self._device)
        with torch.no_grad():
            if self.host:
                return self.vae.decoder(x).float()
            return decode_causal_chunks(
                self._trunk, self._up.first, self._up.rest, x, self.chunk
            ).float()

    def decode_tiles(self, z: torch.Tensor) -> torch.Tensor | None:
        """``z`` ``[B, C, t, h, w]`` (already divided by the scaling factor) -> ``[B, 3, T, H, W]`` fp32
        on rank 0 of the tile group, ``None`` on the others (device tile-parallel)."""
        from vllm_omni_neuron.diffusion.layers.vae_tiling import merge_tiles, run_tiles

        grid = self._grid(z.shape[3], z.shape[4])
        zp = grid.pad_input(z)
        kw = {"world_size": 1, "rank": 0} if self.host else {"group": self._tile_group()}
        tiles = run_tiles(grid, lambda n, idx: self._decode_tile(grid.slice_input(zp, idx)), **kw)
        if tiles is None:
            return None
        out = merge_tiles(tiles, grid)
        return out[..., : grid.out_total()[0], : grid.out_total()[1]]

    def _tile_group(self):
        if self.tile_group is not None:
            return self.tile_group
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        return get_world_group().cpu_group

    def decode(self, z, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        t0 = time.time()
        _maybe_dump_latents(z, self.config.scaling_factor)
        b, _, t, h, w = z.shape
        r, tr = self.spatial_compression_ratio, self.temporal_compression_ratio
        blank = lambda: torch.zeros(b, 3, (t - 1) * tr + 1, h * r, w * r)  # noqa: E731
        if self.host:
            if self.skip_decode:
                out = blank()
                return DecoderOutput(sample=out) if return_dict else (out,)
            prev = torch.get_num_threads()
            # the Lite worker pins torch to one thread; 32 threads measured slower than 16 on a busy host
            torch.set_num_threads(int(os.environ.get("HV15_VAE_HOST_THREADS", "16")))
            try:
                out = self.decode_tiles(z)
            finally:
                torch.set_num_threads(prev)
        else:  # every rank decodes its share of the tiles; rank 0 merges
            out = self.decode_tiles(sync_latents(z, self._tile_group()))
            if out is None:
                out = blank()
        where = "host" if self.host else "device"
        logger.info(
            "hunyuanvideo15 VAE decode (%s) %s -> %s in %.1fs",
            where,
            tuple(z.shape),
            tuple(out.shape),
            time.time() - t0,
        )
        if os.environ.get("HV15_PROFILE", "0") == "1":  # the worker's logger does not reach stdout
            print(f"[hv15-prof] vae_decode_{where} {time.time() - t0:.4f}", flush=True)
        return DecoderOutput(sample=out) if return_dict else (out,)
