# SPDX-License-Identifier: Apache-2.0
"""Neuron-compiled MiniMax-H3 VAE decoders (video ViT + audio BigVGAN).

The diffusers VAEs (`_vendor/autoencoder_kl_minimax_h3*.py`) stay the source of truth for the math and the on-disk
weights; these wrappers compile only the two heavy, fixed-shape forwards onto a NeuronCore and keep the host-side
chunking / tiling / blending logic exactly as diffusers runs it (so the device output matches the CPU reference up
to Neuron rounding -- the VAE decode is judged against the CPU reference, not bit-reproduced).

* **Video** (`NeuronMiniMaxH3VideoVAE`): the decode is a host loop over temporal clips, each spatially tiled
  (MiniMax-H3 ships with tiling ON; **disabling it changes the output**, not just memory use -- an untiled
  384x640 decode measurably streaks the fur/water texture that a tiled decode does not, confirmed on CPU with no
  device involved: tiling off alone reproduces it in eager fp32. So the device path keeps diffusers' tiling on).
  Each clip is `post_quant_conv` (1x1 Conv3d) then the 36-layer ViT `decoder` per tile; every geometry's tiles are
  ≤256x256, so at most a couple of distinct token counts (the full tile and a shorter trailing one) ever reach the
  compiler -- `torch.compile` compiles one graph per shape it actually sees and reuses it.
* **Audio** (`NeuronMiniMaxH3AudioVAE`): BigVGAN is a single conv stack; its `decode` is compiled whole. NOT yet
  wired into the pipeline -- its weight-normalized convs do not lower to StableHLO, so the pipeline decodes audio
  on the host CPU (small, and it must run fp32). See the class docstring for the fold that would enable it.

Only rank 0 (the output rank) builds these; the VAEs are not tensor-parallel.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VIDEO_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]
AUDIO_COMPILER_ARGS = ["--model-type=unet-inference", "--auto-cast=none", "-O1"]


def _backend() -> str:
    from vllm_neuron.envs import get_compile_backend_name

    return get_compile_backend_name()


class CTEVideoAttnProcessor:
    """The decoder's attention (``MiniMaxH3VideoAttnProcessor``: fp32 q/k norm, partial RoPE) with the attention
    itself on nkilib's ``attention_cte`` flash kernel (Wan2.2's ``_nki_attend`` wrapper, the DiT's kernel) when NKI
    kernels can run, else the reference's SDPA. ``MINIMAX_H3_VAE_ATTN=sdpa`` keeps the reference processor."""

    def __call__(self, attn, hidden_states, rotary_emb=None):
        from vllm_omni_neuron.nc_generation import use_nki_kernels

        from ._vendor.autoencoder_kl_minimax_h3 import MiniMaxH3VideoAttnProcessor

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        if not use_nki_kernels(query):
            return MiniMaxH3VideoAttnProcessor()(attn, hidden_states, rotary_emb)
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import _nki_attend

        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query.float()).to(value.dtype)
        key = attn.norm_k(key.float()).to(value.dtype)
        if rotary_emb is not None:
            cos, sin = (t.to(query.dtype) for t in rotary_emb)
            rd = cos.shape[-1]

            def rope(x):
                xr, xp = x[..., :rd], x[..., rd:]
                a, b = xr.chunk(2, dim=-1)
                return torch.cat([xr * cos + torch.cat([-b, a], dim=-1) * sin, xp], dim=-1)

            query, key = rope(query), rope(key)
        b, s, h, d = query.shape
        out = _nki_attend(
            *(t.transpose(1, 2) for t in (query, key, value)), d**-0.5
        )  # d-major [B, H, D, S]
        return attn.to_out[0](out.permute(0, 3, 1, 2).reshape(b, s, h * d).to(value.dtype))


class NeuronMiniMaxH3VideoVAE(nn.Module):
    """Wraps ``AutoencoderKLMiniMaxH3`` so ``decode`` runs the ViT decoder on a NeuronCore, per spatial tile.

    The passed-in ``vae`` must already be loaded at the mixed precision the released checkpoint uses (load it with
    ``torch_dtype=dtype``): ``proj_in`` / ``proj_out`` / the block stack follow ``dtype``, while
    ``_keep_in_fp32_modules`` (``norm1``/``norm2``/``norm_out``/``scale1``/``scale2``) stays fp32. Casting the
    decoder's input to a dtype that does not match its own ``proj_in`` weight is a graph-compile error, not a
    silent one -- the wrapper does not re-cast the module, only the activation entering it.

    Tiling stays ON (diffusers' default geometry: 256x256 tiles, 64px overlap) -- see the module docstring for why
    turning it off is a correctness bug here, not just a memory trade-off.
    """

    def __init__(self, vae, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.vae = vae.eval()  # the VAE (incl. the tiny post_quant_conv) stays on the HOST
        self.device = device
        # Precision placement from the proven inf2/trn1/trn2 FastH3 port: the 36 transformer blocks run bf16 with
        # fp32 norms, but proj_in / norm_out / proj_out stay fp32 (diffusers does not pin these for this VAE, so a
        # plain torch_dtype=bf16 load downcasts them -- upcast back here). Measured to make little difference on
        # its own, but it is the released checkpoint's own precision contract, so it stays.
        dec = self.vae.decoder
        for m in (dec.proj_in, dec.norm_out, dec.proj_out):
            m.float()
        self.block_dtype = next(
            dec.transformer_blocks[0].ff.parameters()
        ).dtype  # bf16 (the block-stack dtype)
        self.dtype = torch.float32  # the dtype the compiled decoder's INPUT (proj_in) consumes
        self.config = vae.config
        assert self.vae.use_tiling  # diffusers' default; see class + module docstring
        self._compiled = (
            None  # compiled decoder.forward; torch.compile keys per input shape it actually sees
        )
        self.keep_clip_shm = (
            False  # set per request by the pipeline (engine path: hand the clip file over)
        )
        self._on_device = False
        if device.type != "cpu" and os.environ.get("MINIMAX_H3_VAE", "device") != "cpu":
            # move the 2.4 B-parameter ViT decoder now, not on the first request, so its host copy is released at
            # load time (every tile-parallel rank holds one: 64 ranks x 4.8 GB of host RAM otherwise)
            self.vae.decoder.to(device)
            self._on_device = True

    def _compiled_forward(self):
        """A callable that runs the ViT decoder's forward on the device, per tile (one NEFF per distinct tile
        shape -- the full tile and, for a canvas that doesn't divide evenly, one shorter trailing tile).

        Only the ViT ``decoder`` moves to the NeuronCore -- the host keeps ``post_quant_conv`` (a 1x1 Conv3d; eager
        convolution is not implemented on the Neuron backend) and the chunk / tile / blend loop.
        """
        if self._compiled is not None:
            return self._compiled
        if os.environ.get("MINIMAX_H3_VAE", "device") == "cpu" or self.device.type == "cpu":
            self._compiled = self.vae.decoder.forward
            return self._compiled
        self.vae.decoder.to(self.device)
        self._on_device = True
        if os.environ.get("MINIMAX_H3_VAE_ATTN", "cte") == "cte":
            for blk in self.vae.decoder.transformer_blocks:
                blk.attn.set_processor(CTEVideoAttnProcessor())
        opts = {
            "model_name": "minimax_h3_video_vae_decoder",
            "compiler_args": list(VIDEO_COMPILER_ARGS),
        }
        self._compiled = torch.compile(
            self.vae.decoder.forward,
            backend=_backend(),
            fullgraph=True,
            dynamic=False,
            options=opts,
        )
        return self._compiled

    def _run_decode(self, z: torch.Tensor, decoder_fn):
        """diffusers' host chunk/tile/blend loop (``_decode``) with every ViT call routed to ``decoder_fn``."""
        decoder = self.vae.decoder
        orig_forward = decoder.forward
        object.__setattr__(decoder, "forward", decoder_fn)
        try:
            return self.vae._decode(
                z.to(self.vae.post_quant_conv.weight.dtype)
            )  # post_quant_conv on the host
        finally:
            object.__setattr__(decoder, "forward", orig_forward)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, return_dict: bool = True, tp_group=None, to_unit=None):
        """Host-chunked decode (diffusers ``_decode``) with the per-tile ViT on the device.

        ``tp_group`` (a vLLM ``GroupCoordinator``) with more than one rank turns on **tile-parallel** decode: every
        rank holds the decoder, the (temporal chunk x spatial tile) work items are dealt round-robin over the
        ranks and decoded as one batched decoder call per rank, and rank 0 gathers the results and runs the blend.
        The loop's control flow does not depend on tensor values, so a first pass with placeholder outputs
        enumerates the work items exactly. Non-zero ranks return ``None``.

        ``to_unit`` (tile-parallel only): the per-channel map from decoder output to ``[0, 1]`` pixels. Each rank
        applies it to its own tiles and ships them as uint8; rank 0 blends the dequantised tiles, and the result is
        already in ``[0, 1]`` (``self.applied_unit`` is then True).
        """
        from diffusers.models.autoencoders.vae import DecoderOutput

        compiled = self._compiled_forward()
        dev, dtype = self.device, self.dtype
        self.tile_calls = (
            0  # (chunk x tile) decoder graphs THIS rank ran -- the parallelisable work unit
        )
        self.tile_shapes: dict = {}  # input token shape -> count (distinct NEFFs the compiler sees)
        self.applied_unit = False
        self.output_uint8 = False

        def on_device(
            x,
        ):  # x arrives on the HOST; cast there first (a Neuron copy will not cast), then move
            self.tile_calls += 1
            self.tile_shapes[tuple(x.shape)] = self.tile_shapes.get(tuple(x.shape), 0) + 1
            return compiled(x.to(dtype).to(dev)).to("cpu").float()

        world = getattr(tp_group, "world_size", 1) if tp_group is not None else 1
        # The chunk / tile / blend loop is host work inside the worker, which Lite pins to one torch thread; widen
        # it for the decode (the output rank's blend most, the other ranks' work-item enumeration a fair share).
        rank = getattr(tp_group, "rank_in_group", 0) if tp_group is not None else 0
        n = int(os.environ.get("MINIMAX_H3_VAE_THREADS", "32"))
        n = n if rank == 0 else max(1, min(n, (os.cpu_count() or 64) // (2 * world)))
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        self.timing = {"threads": n}
        try:
            if world == 1:
                dec = self._run_decode(z, on_device)
                return DecoderOutput(sample=dec) if return_dict else (dec,)
            out = self._decode_tile_parallel(z, on_device, tp_group, to_unit)
        finally:
            torch.set_num_threads(prev)
        if out is None:
            return None if return_dict else (None,)
        return DecoderOutput(sample=out) if return_dict else (out,)

    def _enumerate(self, z: torch.Tensor, decoder_fn) -> None:
        """Pass 1 of the tile-parallel decode: walk diffusers' chunk / tile loop only to collect the decoder inputs.
        The tile stitching and the temporal cross-fades are stubbed with shape-only results (every rank runs this
        pass; the real blends cost ~1 s of host time at 384x640 and would only be thrown away). The placeholders
        are meta tensors, so the final full-clip concatenation allocates nothing (0.75 s per rank at 768p otherwise)."""
        vae = self.vae

        def stitch(tiles, height_overlaps, width_overlaps):
            h = sum(row[0].shape[-2] for row in tiles) - sum(height_overlaps)
            w = sum(t.shape[-1] for t in tiles[0]) - sum(width_overlaps)
            return torch.empty((*tiles[0][0].shape[:-2], h, w), device="meta")

        vae._stitch_tiles, vae._blend = (
            stitch,
            lambda a, b, blend_extent, dim: b,
        )  # instance attrs shadow the class
        try:
            self._run_decode(z, decoder_fn)
        finally:
            del vae._stitch_tiles, vae._blend

    def _decode_tile_parallel(self, z: torch.Tensor, on_device, g, to_unit=None):
        import torch.distributed as dist

        dec = self.vae.decoder
        world, rank = g.world_size, g.rank_in_group
        inputs: list[torch.Tensor] = []

        def out_shape(x):
            b, _, t, h, w = x.shape
            return (
                b,
                dec.out_channels,
                t * dec.patch_size_t,
                h * dec.patch_size,
                w * dec.patch_size,
            )

        def record(x):  # pass 1: enumerate the work items; the output only needs the right shape
            inputs.append(x.clone())
            return torch.empty(out_shape(x), device="meta")  # shape only: no host memory

        t0 = time.perf_counter()
        self._enumerate(z, record)
        mine = list(range(rank, len(inputs), world))
        per = -(
            -len(inputs) // world
        )  # work items on the busiest rank: every rank pads its batch to this
        t1 = time.perf_counter()
        quant = to_unit is not None
        # transport dtype: uint8 pixels with `to_unit` (the clip is 8-bit in the end), else 16-bit decoder output
        gdt = (
            torch.uint8
            if quant
            else getattr(torch, os.environ.get("MINIMAX_H3_VAE_GATHER_DTYPE", "float16"))
        )

        def ship(o):
            if quant:
                return (to_unit(o).clamp_(0, 1) * 255.0).round_().to(torch.uint8)
            return o.to(gdt)

        if len({tuple(x.shape) for x in inputs}) == 1:
            # one shape: this rank's items as ONE batched decoder call (MINIMAX_H3_VAE_BATCH caps the batch; equal
            # chunks), zero-padded to the busiest rank's count so every rank runs the same graph
            cap = max(1, int(os.environ.get("MINIMAX_H3_VAE_BATCH", "8")))
            ncall = -(-per // cap)
            bsz = -(-per // ncall)
            x0 = inputs[0]
            stack = torch.cat(
                [inputs[i] for i in mine] + [x0.new_zeros(x0.shape)] * (ncall * bsz - len(mine)), 0
            )
            outs = torch.cat(
                [ship(on_device(stack[c * bsz : (c + 1) * bsz])) for c in range(ncall)], 0
            )[:per]
            t2 = time.perf_counter()
            if quant and self._shm_ok() and os.environ.get("MINIMAX_H3_VAE_COMPOSE", "1") != "0":
                plan = self._compose_plan(z, len(inputs))
                if (
                    plan is not None
                ):  # every rank blends its own output frames into one shared uint8 clip
                    out = self._compose_shared(outs.unsqueeze(1).contiguous(), plan, g)
                    self.applied_unit = self.output_uint8 = True
                    self.timing.update(
                        enumerate_s=t1 - t0,
                        tiles_s=t2 - t1,
                        compose_s=time.perf_counter() - t2,
                        batch=per,
                        transport="uint8-compose",
                    )
                    return out if rank == 0 else None
            by_rank = self._gather_stack(
                outs.unsqueeze(1).contiguous(), g
            )  # (per, 1, C, T, H, W) per rank
        else:  # mixed tile shapes (a canvas that does not tile evenly): per-item calls, object gather
            outs_l = [ship(on_device(inputs[i])) for i in mine]
            t2 = time.perf_counter()
            by_rank = [None] * world if rank == 0 else None
            dist.gather_object(outs_l, by_rank, dst=g.ranks[0], group=g.cpu_group)
        if rank != 0:
            self.timing.update(
                enumerate_s=t1 - t0, tiles_s=t2 - t1, gather_s=time.perf_counter() - t2
            )
            return None
        t3 = time.perf_counter()

        def take(i):
            t = by_rank[i % world][i // world].float()
            return t.div_(255.0) if quant else t

        results = iter([take(i) for i in range(len(inputs))])
        out = self._run_decode(
            z, lambda x: next(results)
        )  # pass 2 (rank 0): blend with the real outputs
        self.applied_unit = quant
        self.timing.update(
            enumerate_s=t1 - t0,
            tiles_s=t2 - t1,
            gather_s=t3 - t2,
            blend_s=time.perf_counter() - t3,
            batch=per,
            transport=str(gdt).split(".")[-1],
        )
        return out

    @staticmethod
    def _shm_ok() -> bool:
        return os.environ.get("MINIMAX_H3_VAE_SHM", "1") != "0" and os.path.isdir("/dev/shm")

    def _compose_plan(self, z: torch.Tensor, n_items: int):
        """The tile-parallel decode's blend as weights, per geometry (cached): diffusers' spatial stitch and temporal
        cross-fade are linear in the decoded tiles, so every output frame is ``sum_k w_k(y, x) * tile_k(frame t)``
        over at most two (temporal chunk, frame) sources. Derived by running diffusers' own ``_stitch_tiles`` /
        ``_decode`` once on one-hot inputs, so the weights are diffusers' by construction. Returns
        ``{"frames": [[(chunk, t, w), ...] per output frame], "tiles": [(y0, x0, weight map)], ...}`` or None when the
        loop structure does not match (the caller then gathers and blends on rank 0 as before)."""
        key = tuple(z.shape)
        cache = getattr(self, "_plans", None)
        if cache is None:
            cache = self._plans = {}
        if key in cache:
            return cache[key]
        vae, dec = self.vae, self.vae.decoder
        r = vae.spatial_compression_ratio
        height, width = z.shape[-2] * r, z.shape[-1] * r
        ys, hl, yo = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        xs, wl, xo = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        n_tiles = len(ys) * len(xs)
        clips: list[int] = []

        def clip_counter(zc):  # count chunks and their decoded frame count (meta: no work)
            clips.append(zc.shape[2] * dec.patch_size_t)
            return torch.empty((1, 1, clips[-1], height, width), device="meta")

        vae._decode_clip = clip_counter
        try:
            vae._decode(z.to(vae.post_quant_conv.weight.dtype).to("meta"))
        finally:
            del vae._decode_clip
        if (
            not clips
            or len(set(clips)) != 1
            or len(clips) * n_tiles != n_items
            or len(set(hl)) != 1
            or len(set(wl)) != 1
        ):
            cache[key] = None
            return None
        tc, n_clip = clips[0], len(clips)
        # temporal: one-hot channel per (chunk, frame), spatial 1x1
        it = iter(range(n_clip))

        def onehot_clip(zc):
            c = next(it)
            o = torch.zeros((1, n_clip * tc, tc, 1, 1), dtype=torch.float64)
            o[0, c * tc + torch.arange(tc), torch.arange(tc)] = 1.0
            return o

        vae._decode_clip = onehot_clip
        try:
            wt = vae._decode(z[:, :, :, :1, :1].to(vae.post_quant_conv.weight.dtype))[
                0, :, :, 0, 0
            ]  # (K, F)
        finally:
            del vae._decode_clip
        frames = []
        for f in range(wt.shape[1]):
            nz = torch.nonzero(wt[:, f].abs() > 0).flatten().tolist()
            frames.append([(k // tc, k % tc, float(wt[k, f])) for k in nz])
        # spatial: one-hot channel per tile through diffusers' stitch
        th, tw = hl[0], wl[0]
        rows = [[torch.zeros((1, n_tiles, 1, th, tw), dtype=torch.float64) for _ in xs] for _ in ys]
        for i in range(len(ys)):
            for j in range(len(xs)):
                rows[i][j][0, i * len(xs) + j] = 1.0
        ws = type(vae)._stitch_tiles(vae, rows, yo, xo)[0, :, 0]  # (n_tiles, H, W)
        if ws.shape[-2:] != (height, width) or not torch.allclose(
            ws.sum(0), torch.ones_like(ws[0]), atol=1e-6
        ):
            cache[key] = None
            return None
        tiles = [
            (
                ys[i],
                xs[j],
                ws[i * len(xs) + j, ys[i] : ys[i] + th, xs[j] : xs[j] + tw].float().contiguous(),
            )
            for i in range(len(ys))
            for j in range(len(xs))
        ]
        plan = {
            "frames": frames,
            "tiles": tiles,
            "n_tiles": n_tiles,
            "shape": (1, dec.out_channels, len(frames), height, width),
        }
        cache[key] = plan
        return plan

    def _compose_shared(self, local: torch.Tensor, plan, g):
        """Every rank writes its uint8 tiles to ``/dev/shm``; after a barrier each rank blends a contiguous range of
        output frames from the (memory-mapped) tiles of all ranks with the plan's weights and writes them as uint8
        into one shared clip file; after a second barrier rank 0 maps the clip. No gather to one process and no
        single-process blend. Rank 0 returns ``(1, C, F, H, W)`` uint8; the others None."""
        import uuid

        import numpy as np
        import torch.distributed as dist

        world, rank = g.world_size, g.rank_in_group
        tag = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(tag, src=g.ranks[0], group=g.cpu_group)
        base = f"/dev/shm/minimax_h3_vae_{tag[0]}"
        shape = plan["shape"]
        if rank == 0:
            with open(f"{base}_clip.bin", "wb") as f:
                f.truncate(int(np.prod(shape)))
        local.numpy().tofile(f"{base}_{rank}.bin")
        dist.all_reduce(
            torch.zeros(1), group=g.cpu_group
        )  # barrier: every tile file and the clip file exist
        tiles = [
            np.memmap(f"{base}_{r}.bin", dtype=np.uint8, mode="r", shape=tuple(local.shape))
            for r in range(world)
        ]
        clip = np.memmap(f"{base}_clip.bin", dtype=np.uint8, mode="r+", shape=shape)
        n_f = shape[2]
        f0, f1 = rank * n_f // world, (rank + 1) * n_f // world
        nt = plan["n_tiles"]
        for f in range(f0, f1):
            acc = torch.zeros(shape[1], shape[3], shape[4])
            for c, t, w in plan["frames"][f]:
                for k, (y0, x0, wk) in enumerate(plan["tiles"]):
                    i = c * nt + k  # work item -> (rank i % world, slot i // world)
                    src = torch.from_numpy(
                        np.asarray(tiles[i % world][i // world, 0, :, t])
                    )  # (C, h, w)
                    region = acc[:, y0 : y0 + wk.shape[0], x0 : x0 + wk.shape[1]]
                    region.add_(src.float() * (wk * (w / 255.0)))
            clip[0, :, f] = acc.clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).numpy()
        clip.flush()
        del tiles
        dist.all_reduce(torch.zeros(1), group=g.cpu_group)  # barrier: every frame is written
        if rank != 0:
            return None
        # copy-on-write map: no read pass here, and the pages stay valid after the file is unlinked
        out = torch.from_numpy(np.memmap(f"{base}_clip.bin", dtype=np.uint8, mode="c", shape=shape))
        for r in range(world):
            os.unlink(f"{base}_{r}.bin")
        if (
            self.keep_clip_shm
        ):  # the pipeline hands the clip file itself to the engine (see output_handle)
            out._h3_shm_name = os.path.basename(f"{base}_clip.bin")
            return out
        os.unlink(f"{base}_clip.bin")
        return out

    @staticmethod
    def _gather_stack(local: torch.Tensor, g):
        """Every rank's equal-shaped ``local`` stack -> rank 0 (a list by rank; None elsewhere). Through files in
        ``/dev/shm`` (one write per rank, a barrier, rank 0 reads them back) when it exists and
        ``MINIMAX_H3_VAE_SHM`` is not ``0``: the stage is on one host, and this avoids pushing ~1 GB through gloo's
        sockets at 768p. Otherwise a gloo gather."""
        import torch.distributed as dist

        world, rank = g.world_size, g.rank_in_group
        if os.environ.get("MINIMAX_H3_VAE_SHM", "1") != "0" and os.path.isdir("/dev/shm"):
            import uuid

            import numpy as np

            tag = [uuid.uuid4().hex if rank == 0 else None]
            dist.broadcast_object_list(tag, src=g.ranks[0], group=g.cpu_group)

            def path(r):
                return f"/dev/shm/minimax_h3_vae_{tag[0]}_{r}.bin"

            local.numpy().tofile(path(rank))
            dist.all_reduce(
                torch.zeros(1), group=g.cpu_group
            )  # barrier (dist.barrier probes the accelerator)
            if rank != 0:
                return None
            out = []
            for r in range(world):
                out.append(
                    torch.from_numpy(
                        np.fromfile(path(r), dtype=local.numpy().dtype).reshape(local.shape)
                    )
                )
                os.unlink(path(r))
            return out
        bufs = [torch.empty_like(local) for _ in range(world)] if rank == 0 else None
        dist.gather(local, bufs, dst=g.ranks[0], group=g.cpu_group)
        return bufs


def _rpad(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """``F.pad(x, (left, right), mode="replicate")`` as a concat of the expanded edge columns: same values, but no
    gather (the replicate pad lowers to one indirect DMA per output sample on the Neuron compiler)."""
    parts = []
    if left:
        parts.append(x[..., :1].expand(*x.shape[:-1], left))
    parts.append(x)
    if right:
        parts.append(x[..., -1:].expand(*x.shape[:-1], right))
    return torch.cat(parts, dim=-1) if len(parts) > 1 else x


def _upsample_rpad(self, x: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3AudioUpSample1d.forward`` with :func:`_rpad`."""
    c = x.shape[1]
    x = _rpad(x, self.pad, self.pad)
    x = self.ratio * F.conv_transpose1d(
        x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c
    )
    return x[..., self.pad_left : -self.pad_right]


def _lowpass_rpad(self, x: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3AudioLowPassFilter1d.forward`` with :func:`_rpad`."""
    c = x.shape[1]
    x = _rpad(x, self.pad_left, self.pad_right)
    return F.conv1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)


def _activation_folded(self, x: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3AudioActivation1d.forward`` (upsample x2 -> SnakeBeta -> downsample x2) with the long time axis
    folded into rows. In the decoder's last stages the tensor is ``(2, 8..32, 20k..46k)``: on a NeuronCore that is
    16-64 rows of a very long sequence, and the alias-free activation ran at 1/50 of the host's speed there (167 ms
    per activation at (2, 8, 46400) against 3.5 ms on 16 host threads). Every step is local in time, so the sequence
    is cut into ``nseg`` overlapping segments (``halo`` input samples of real context on each side), all segments of
    all channels go through the activation as rows of one tensor, and the halos are cropped. Interior samples are
    exact; the clip's two ends see the same replicate padding as the original up to the filter's DC response."""
    b, c, n = x.shape
    rows = int(os.environ.get("MINIMAX_H3_AUDIO_FOLD_ROWS", "256"))
    nseg = max(1, min(rows // max(1, b * c), n // 256))
    if nseg == 1:
        return type(self).forward(self, x)
    halo = 16
    seg = -(-n // nseg)
    right = halo + seg * nseg - n
    xp = (
        _rpad(x, halo, right)
        if getattr(self, "_h3_rpad", False)
        else F.pad(x, (halo, right), mode="replicate")
    )
    win = torch.stack(
        [xp[..., i * seg : i * seg + seg + 2 * halo] for i in range(nseg)], dim=2
    )  # (b, c, nseg, w)
    win = win.reshape(b, c * nseg, seg + 2 * halo)
    act = self.act
    alpha = torch.exp(act.alpha).repeat_interleave(nseg).view(1, -1, 1)
    beta = torch.exp(act.beta).repeat_interleave(nseg).view(1, -1, 1)
    h = self.upsample(win)
    h = h + (beta + 1e-9).reciprocal() * torch.sin(alpha * h).pow(2)
    h = self.downsample(h)  # (b, c * nseg, seg + 2 * halo)
    return h[..., halo : halo + seg].reshape(b, c, nseg * seg)[..., :n]


def fold_long_activations(module: nn.Module) -> int:
    """Use :func:`_activation_folded` for every alias-free SnakeBeta activation under ``module``.

    ``MINIMAX_H3_AUDIO_RPAD=1`` (off by default) also swaps every replicate pad for :func:`_rpad`. On its own an
    activation got faster (folded last stage 1.71 -> 0.97 ms on a core), but in the whole decoder window the
    expanded edge slabs are materialised and concatenated at every resampler of every stage: the NEFF grew 7.5x
    (15 -> 115 MB), compile 315 -> 2329 s and the window 100 -> 1599 ms. F.pad stays the default."""
    from ._vendor.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioActivation1d,
        MiniMaxH3AudioLowPassFilter1d,
        MiniMaxH3AudioSnakeBeta,
        MiniMaxH3AudioUpSample1d,
    )

    rpad = os.environ.get("MINIMAX_H3_AUDIO_RPAD", "0") == "1"
    n = 0
    for m in module.modules():
        if isinstance(m, MiniMaxH3AudioActivation1d) and isinstance(m.act, MiniMaxH3AudioSnakeBeta):
            m.forward = _activation_folded.__get__(m)
            m._h3_rpad = rpad
            n += 1
        elif rpad and isinstance(m, MiniMaxH3AudioUpSample1d):
            m.forward = _upsample_rpad.__get__(m)
        elif rpad and isinstance(m, MiniMaxH3AudioLowPassFilter1d):
            m.forward = _lowpass_rpad.__get__(m)
    return n


def audio_windows(total: int, chunk: int, halo: int) -> list[tuple[int, int, int]]:
    """Split ``total`` audio latent frames into ``ceil(total / chunk)`` pieces, each decoded from a window of the
    same length ``min(total, chunk + 2 * halo)`` (one compiled graph for all of them): ``(window start, keep start,
    keep end)`` in latent frames. Edge windows are shifted inward, so the clip's true start / end needs no halo."""
    w = min(total, chunk + 2 * halo)
    out = []
    for s0 in range(0, total, chunk):
        s1 = min(total, s0 + chunk)
        a0 = min(max(0, s0 - halo), total - w)
        out.append((a0, s0, s1))
    return out


class NeuronMiniMaxH3AudioVAE(nn.Module):
    """Wraps ``AutoencoderKLMiniMaxH3Audio`` so ``decode`` (BigVGAN) runs compiled on a NeuronCore, in fixed-length
    time windows that can be spread over ranks (:func:`audio_windows`).

    * The decoder is fully convolutional (the audio VAE's only attention is in the encoder's ``pre_block``), so a
      window with ``halo`` latent frames of context on each side reproduces the full decode inside its kept range:
      at halo 16 the stitched waveform is 110 dB SNR from the one-shot fp32 decode on the real checkpoint (halo 8:
      33 dB), measured on CPU. One window length means one compiled graph.
    * The ``weight_norm`` pre-hooks are folded into plain weights before compile (they recompute ``weight`` inside
      the traced graph, which does not lower); folding is exact.
    * The alias-free activations run with the time axis folded into rows (:func:`_activation_folded`): as written
      they ran at 1/50 of the host's speed in the last, 8-channel stage (one window 3.8 s on a core; folded 0.10 s,
      against 0.32-0.39 s on 16 host threads).
    * Always fp32 (diffusers' note: bf16 decodes ~20 dB quieter).
    """

    def __init__(self, vae, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.vae = vae.eval()
        self.device = device
        self.dtype = torch.float32
        self.config = vae.config
        self.hop = None
        self._compiled = None

    def _ensure_compiled(self):
        if self._compiled is not None:
            return
        dec = self.vae

        def fold(m):
            if hasattr(m, "weight_g"):
                torch.nn.utils.remove_weight_norm(m)

        dec.decoder.apply(fold)
        dec.dec_in_proj.apply(fold)
        if (
            os.environ.get("MINIMAX_H3_AUDIO_FOLD", "1") != "0"
        ):  # long, narrow late stages: time axis into rows
            fold_long_activations(dec.decoder)
        fn = lambda z: dec.decoder(dec.dec_in_proj(z))  # noqa: E731  (AutoencoderKLMiniMaxH3Audio.decode, fp32)
        if os.environ.get("MINIMAX_H3_VAE", "device") == "cpu" or self.device.type == "cpu":
            self._compiled = fn
            return
        dec.decoder.to(self.device)
        dec.dec_in_proj.to(self.device)
        opts = {
            "model_name": "minimax_h3_audio_vae_decoder",
            "compiler_args": list(AUDIO_COMPILER_ARGS),
        }
        self._compiled = torch.compile(
            fn, backend=_backend(), fullgraph=True, dynamic=False, options=opts
        )

    @torch.no_grad()
    def decode_window(
        self, z: torch.Tensor, window: tuple[int, int, int], length: int
    ) -> torch.Tensor:
        """``z`` ``(2, C, T)`` denormalised latents -> the waveform ``(2, 1, (keep end - keep start) * hop)`` of one
        window from :func:`audio_windows`."""
        self._ensure_compiled()
        a0, s0, s1 = window
        x = z[:, :, a0 : a0 + length].float().contiguous()
        out = self._compiled(x.to(self.device)).to("cpu").float()
        hop = out.shape[-1] // length
        return out[..., (s0 - a0) * hop : (s1 - a0) * hop]

    @torch.no_grad()
    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """One-process decode through the windows (the pipeline deals the windows over ranks instead)."""
        from diffusers.models.autoencoders.vae import DecoderOutput

        chunk = int(os.environ.get("MINIMAX_H3_AUDIO_CHUNK", "26"))
        halo = int(os.environ.get("MINIMAX_H3_AUDIO_HALO", "16"))
        wins = audio_windows(z.shape[-1], chunk, halo)
        length = min(z.shape[-1], chunk + 2 * halo)
        sample = torch.cat([self.decode_window(z, w, length) for w in wins], dim=-1)
        return DecoderOutput(sample=sample) if return_dict else (sample,)
