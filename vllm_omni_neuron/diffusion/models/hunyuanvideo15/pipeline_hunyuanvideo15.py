# SPDX-License-Identifier: Apache-2.0
"""Neuron HunyuanVideo-1.5 pipeline: upstream vLLM-Omni's ``HunyuanVideo15Pipeline`` with the
transformer and the VAE swapped for Neuron-compiled components.

Everything above the transformer call stays upstream's (``forward``): request parsing, prompt
templating, CFG branches, the flow-matching scheduler and the latent math. That host-side math
runs on the CPU (``self.device`` is CPU) and only two components touch the NeuronCores:

* ``self.transformer`` -> :class:`NeuronHunyuanVideo15Transformer`, a drop-in with upstream's
  call signature over :class:`.transformer.NeuronHunyuanVideo15DiT` (TP-sharded, fixed-shape
  compiled graphs; the encoder-side host prep is cached per CFG branch).
* ``self.vae`` -> :class:`.vae.NeuronHunyuanVideo15VAE` (compiled per-tile decoder).

The two text encoders (Qwen2.5-VL-7B text tower, byT5) are Hugging Face modules run once per
request on the host, on TP rank 0 only; the embeddings are broadcast to the other ranks.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
    format_text_input,
    get_hunyuan_video_15_post_process_func,
)

from .text_encoder import NeuronQwenTextEncoder, assignment, text_groups
from .transformer import (
    HV15Config,
    NeuronHunyuanVideo15DiT,
    group_all_gather,
    local_model_dir,
    pick_bucket,
    prepare_encoder_inputs,
    rope_tables,
)
from .vae import NeuronHunyuanVideo15VAE

logger = logging.getLogger(__name__)

PROFILE = os.environ.get("HV15_PROFILE", "0") == "1"
# MLLM prompt-length buckets (tokens after the template crop; upstream pads to 1000). One bucket
# = one set of compiled DiT graphs, so the default keeps a single, full-length bucket.
TEXT_BUCKETS = tuple(int(b) for b in os.environ.get("HV15_TEXT_BUCKETS", "1000").split(","))
# 0 = the whole DiT is one graph; k > 0 = prologue + ceil(54/k) k-block graphs + epilogue.
BLOCKS_PER_GRAPH = int(os.environ.get("HV15_BLOCKS_PER_GRAPH", "0"))
# Qwen2.5-VL text tower placement: "auto" = on the NeuronCores (TP=HV15_TEXT_TP over contiguous rank groups)
# when the world divides into such groups, else host; "host" = host CPU on global rank 0 (fallback).
TEXT_ENCODER = os.environ.get("HV15_TEXT_ENCODER", "auto")
TEXT_TP = int(os.environ.get("HV15_TEXT_TP", "4"))
TEXT_CACHE = int(os.environ.get("HV15_TEXT_CACHE", "16"))  # prompt-embedding LRU entries (0 = off)
TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

PIPELINE_REGISTRY = [
    {
        # model_index.json `_class_name` of hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*_t2v,
        # also the upstream vLLM-Omni registry key
        "model_arch": "HunyuanVideo15Pipeline",
        "class_name": "NeuronHunyuanVideo15Pipeline",
        "post_process_func_name": "get_hunyuan_video_15_post_process_func",
    },
]

# Upstream's MLLM system prompt, verbatim (the backslash continuations in upstream's source keep
# the 8-space indentation, and prompt_template_encode_start_idx=108 depends on its token count).
_IND = " " * 8
SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    f"{_IND}1. The main content and theme of the video. "
    f"{_IND}2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    f"{_IND}3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    f"{_IND}4. background environment, light, style and atmosphere. "
    f"{_IND}5. camera angles, movements, and transitions used in the video."
)


def _prof(name: str, t0: float, rank0_only: bool = False) -> None:
    if PROFILE and (not rank0_only or _global_rank() == 0):
        print(f"[hv15-prof] {name} {time.time() - t0:.4f}", flush=True)


def _host(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    x = x.detach().to("cpu")
    return (x.to(dtype) if dtype is not None else x).contiguous()


class _BlockGroup(nn.Module):
    """A contiguous slice of DiT blocks, compiled as one graph (identical slices share a NEFF)."""

    def __init__(self, dit: NeuronHunyuanVideo15DiT, start: int, end: int, ssta=None):
        super().__init__()
        self.blocks = nn.ModuleList(dit.transformer_blocks[start:end])
        object.__setattr__(self, "_group", dit.tp_group)
        object.__setattr__(self, "_cp", dit.cp_group)
        object.__setattr__(self, "_ssta", ssta)  # SSTAStatic (compile-time constants) or None

    def forward(self, hidden, enc, temb_act, cos, sin, key_bias, *ssta_inputs):
        ssta = (self._ssta, *ssta_inputs) if self._ssta is not None else None
        for blk in self.blocks:
            hidden, enc = blk(
                hidden, enc, temb_act, cos, sin, key_bias, self._group, self._cp, ssta
            )
        return hidden, enc


class _Prologue(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, *args):
        return self.dit.prologue(*args)


class _CPPrologue(nn.Module):
    """Prologue + this context-parallel rank's video tokens (``vidx``: a host-built gather index, so
    every CP rank runs the same graph)."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, vidx, *args):
        hidden, enc, temb = self.dit.prologue(*args)
        hidden = torch.gather(
            hidden, 1, vidx[None, :, None].expand(hidden.shape[0], -1, hidden.shape[-1])
        )
        return hidden, enc, temb


class _SSTAPrologue(nn.Module):
    """Prologue for the sparse (SSTA) checkpoints: this rank's video slots in the tile-major order
    (``vidx``) and the encoder padding zeroed (``enc_keep``), as upstream's ``zero_feat`` reorder."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, vidx, enc_keep, *args):
        hidden, enc, temb = self.dit.prologue(*args)
        hidden = torch.gather(
            hidden, 1, vidx[None, :, None].expand(hidden.shape[0], -1, hidden.shape[-1])
        )
        return hidden, enc * enc_keep, temb


class _EpilogueTokens(nn.Module):
    """Final AdaLN + projection per token; under context parallel the tokens of every CP rank are
    all-gathered IN the graph (``cp``, coordinator order), under CFG parallel both branches (``cfg``,
    along the batch: ``[positive, negative]``)."""

    def __init__(self, dit, cp=None, cfg=None):
        super().__init__()
        self.dit, self.cp, self.cfg = dit, cp, cfg

    def forward(self, hidden, temb_act):
        out = self.dit.epilogue_tokens(hidden, temb_act)
        if self.cp is not None:
            out = group_all_gather(self.cp, out, dim=1)
        if self.cfg is not None:
            out = group_all_gather(self.cfg, out, dim=0)
        return out


class _Epilogue(nn.Module):
    def __init__(self, dit, grid, cfg=None):
        super().__init__()
        self.dit, self.grid, self.cfg = dit, grid, cfg

    def forward(self, hidden, temb_act):
        out = self.dit.epilogue(hidden, temb_act, self.grid)
        if self.cfg is not None:
            out = group_all_gather(self.cfg, out, dim=0)
        return out


class NeuronHunyuanVideo15Transformer(nn.Module):
    """Drop-in for upstream ``HunyuanVideo15Transformer3DModel`` inside the pipeline."""

    def __init__(self, od_config):
        super().__init__()
        self.model_path = local_model_dir(od_config.model)
        self.cfg = HV15Config.from_model_dir(self.model_path)
        self.dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        self.dit = NeuronHunyuanVideo15DiT(self.cfg, dtype=self.dtype)
        self._device = torch.device("cpu")
        self._backend = None
        self._options: dict = {}
        self._fns: dict = {}
        self._enc_cache: dict = {}
        self._rope_cache: dict = {}
        self.stats = {"calls": 0, "dit_s": 0.0}
        # CFG-parallel: the pipeline sets the CFG GroupCoordinator around a branch forward; the call
        # then returns BOTH branches stacked on the batch ([positive, negative]), gathered in the graph
        self.cfg_gather = None

    @property
    def transformer_blocks(self):  # upstream reads transformer_blocks[0].norm1.linear.weight.dtype
        return self.dit.transformer_blocks

    def load(self) -> None:
        t0 = time.time()
        self.dit.load_weights(self.model_path, "cpu")
        logger.info(
            "HunyuanVideo-1.5 DiT weights loaded (TP rank %d/%d) in %.1fs",
            self.dit.tp_rank,
            self.dit.tp_size,
            time.time() - t0,
        )

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.dit.to(self._device)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        self._backend, self._options = backend, dict(options or {})
        self._fullgraph = kwargs.get("fullgraph", True)

    def _compiled(self, key, factory, name):
        fn = self._fns.get(key)
        if fn is None:
            mod = factory()
            if self._backend is not None:
                opts = {
                    **self._options,
                    "model_name": name,
                    "compiler_args": list(TRANSFORMER_COMPILER_ARGS),
                }
                fn = torch.compile(
                    mod,
                    backend=self._backend,
                    options=opts,
                    fullgraph=self._fullgraph,
                    dynamic=False,
                )
            else:
                fn = mod
            self._fns[key] = fn
        return fn

    # -- host prep ------------------------------------------------------------------------------
    def _encoder(self, text, tmask, text2, t2mask, image, sv, sv_pad=0):
        # Keyed on the caller's tensor addresses, so the entry MUST keep those tensors alive: a key
        # built from a freed tensor's data_ptr can be matched by a different tensor later allocated
        # at the same address (a CFG negative branch silently served the positive embeddings).
        srcs = (text, tmask, text2, t2mask, image)
        key = (
            text.data_ptr(),
            tuple(text.shape),
            tmask.data_ptr(),
            text2.data_ptr(),
            t2mask.data_ptr(),
            None if image is None else image.data_ptr(),
            sv,
            sv_pad,
        )
        hit = self._enc_cache.get(key)
        if hit is not None and all(a is b for a, b in zip(hit[0], srcs)):
            return hit[1]
        dev, dt = self._device, self.dtype
        tmask = _host(tmask).float()
        n_text = int(tmask.sum(dim=1).max().item())
        lt = pick_bucket(n_text, TEXT_BUCKETS)
        if lt < tmask.shape[1] and tmask[:, lt:].any():
            raise ValueError("MLLM prompt mask is not a valid prefix")
        text = _host(text, dt)[:, :lt]
        if text.shape[1] < lt:
            text = torch.cat(
                [text, text.new_zeros(text.shape[0], lt - text.shape[1], text.shape[2])], dim=1
            )
        tm = torch.zeros(tmask.shape[0], lt)
        tm[:, : min(lt, tmask.shape[1])] = tmask[:, :lt]
        t2mask = _host(t2mask).float()
        image_mask = None
        if image is not None:
            image_mask = torch.ones(image.shape[0], image.shape[1])
        enc_index, text_bias, key_bias = prepare_encoder_inputs(tm, t2mask, image_mask, sv, sv_pad)
        n_valid = (
            tm.sum(1) + t2mask.sum(1) + (0 if image_mask is None else image_mask.sum(1))
        ).long()
        out = (
            text.to(dev),
            tm.to(dt).contiguous().to(dev),
            _host(text2, dt).to(dev),
            None if image is None else _host(image, dt).to(dev),
            enc_index.to(dev),
            text_bias.to(dev),
            key_bias.to(dev),
            n_valid,
        )
        if len(self._enc_cache) > 8:
            self._enc_cache.clear()
        self._enc_cache[key] = (srcs, out)
        return out

    def _rope(self, grid):
        hit = self._rope_cache.get(grid)
        if hit is None:
            cos, sin = rope_tables(self.cfg, *grid)
            hit = self._rope_cache[grid] = (cos.to(self._device), sin.to(self._device))
        return hit

    def _cp_layout(self, grid):
        """Context parallel: (padded video length, this rank's gather index, its RoPE rows).

        The video tokens are padded to a multiple of CP and rank r owns ``[r*L, (r+1)*L)``. Padding
        slots gather token 0 (any valid token); they are masked as keys and their rows are dropped."""
        key = ("cp", grid)
        hit = self._rope_cache.get(key)
        if hit is None:
            cp, r = self.dit.cp_size, self.dit.cp_rank
            sv = grid[0] * grid[1] * grid[2]
            local = -(-sv // cp)
            pos = torch.arange(r * local, (r + 1) * local)
            idx = torch.where(pos < sv, pos, torch.zeros_like(pos))
            cos, sin = rope_tables(self.cfg, *grid)
            hit = self._rope_cache[key] = (
                local * cp,
                idx.to(self._device),
                cos[idx].contiguous().to(self._device),
                sin[idx].contiguous().to(self._device),
            )
        return hit

    def _cp_tokens(self, hidden, temb, tag):
        """Epilogue tokens of the WHOLE (padded) video: the CP ranks' slices are all-gathered on the
        device inside the epilogue graph (coordinator order), as are the CFG branches when set."""
        g, cg = self.dit.cp_group, self.cfg_gather
        epi = self._compiled(
            ("epi", tag), lambda: _EpilogueTokens(self.dit, g, cg), f"hv15_epilogue_{tag}"
        )
        return epi(hidden, temb).to("cpu")

    def _ssta_layout(self, grid, enc_len: int):
        """Sparse checkpoints: (SSTAStatic, this rank's device inputs, raster->slot index) per geometry."""
        from . import ssta as S

        key = ("ssta", grid, enc_len)
        hit = self._rope_cache.get(key)
        if hit is None:
            from vllm_omni_neuron.diffusion.attention.block_sparse import _kernel_ok

            dev = self._device
            probe = torch.zeros(1, device=dev)
            heads = self.cfg.num_heads // self.dit.tp_size
            st = S.build_static(
                self.cfg.ssta, grid, enc_len, self.dit.cp_size, heads, with_kernel=_kernel_ok(probe)
            )
            ri = S.rank_inputs(st, self.dit.cp_rank)
            cos, sin = rope_tables(self.cfg, *grid)
            vidx = ri["vidx"]
            dev_in = dict(
                vidx=vidx.to(dev),
                cos=cos[vidx].contiguous().to(dev),
                sin=sin[vidx].contiguous().to(dev),
                slot_valid=ri["slot_valid"].to(self.dtype).to(dev),
                win=ri["win"].to(dev),
                tq_idx=ri["tq_idx"].to(dev),
            )
            hit = self._rope_cache[key] = (st, dev_in, ri["inv"])
            if _global_rank() == 0:
                lay = st.lay
                logger.info(
                    "hv15 SSTA: grid %s -> %s tiles of %d (+%d text tiles), top-k %d, window %s, list width %d, "
                    "CP %d (%d video / %d text query tiles per rank), kernel %s",
                    grid,
                    lay.grid,
                    lay.tile_tokens,
                    lay.text_tiles,
                    st.topk,
                    st.window,
                    st.kmax,
                    st.cp,
                    st.nq,
                    st.ntq,
                    st.sp_video is not None,
                )
        return hit

    # -- forward (upstream signature) -------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep_r=None,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,
        image_embeds=None,
        image_embeds_mask=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        if timestep_r is not None:
            raise NotImplementedError(
                "HunyuanVideo-1.5 on Neuron: meanflow timestep_r is not supported"
            )
        _maybe_dump_inputs(
            self.stats["calls"],
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            image_embeds=image_embeds,
        )
        cfg, dev, dt = self.cfg, self._device, self.dtype
        b, _, t, h, w = hidden_states.shape
        grid = (t // cfg.patch_size_t, h // cfg.patch_size, w // cfg.patch_size)
        sv = grid[0] * grid[1] * grid[2]
        cp = self.dit.cp_size
        sparse = cfg.ssta is not None
        if (cp > 1 or sparse) and BLOCKS_PER_GRAPH <= 0:
            raise ValueError(
                "HunyuanVideo-1.5 context parallelism / SSTA needs HV15_BLOCKS_PER_GRAPH > 0"
            )
        sv_pad = sv
        if cp > 1 and not sparse:
            sv_pad, vidx, cos, sin = self._cp_layout(grid)
        image = None
        if image_embeds is not None:
            is_t2v = (
                bool((image_embeds == 0).all())
                if image_embeds_mask is None
                else not bool(image_embeds_mask.any())
            )
            image = None if is_t2v else image_embeds
        text, tm, text2, img, enc_index, text_bias, key_bias, n_valid = self._encoder(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image,
            sv,
            sv_pad - sv,
        )
        ssta_in = ()
        if sparse:
            from .ssta import text_tile_keep

            ne = enc_index.shape[1]
            st, sdev, inv = self._ssta_layout(grid, ne)
            vidx, cos, sin = sdev["vidx"], sdev["cos"], sdev["sin"]
            enc_keep = (torch.arange(ne)[None] < n_valid[:, None]).to(dt)[..., None].to(dev)
            tkeep = text_tile_keep(st, n_valid, cfg.ssta.use_text_mask).to(dev)
            ssta_in = (sdev["slot_valid"], sdev["win"], tkeep, sdev["tq_idx"])
        elif cp == 1:
            cos, sin = self._rope(grid)
        x = _host(hidden_states, dt).to(dev)
        ts = _host(timestep).float().reshape(-1)
        if ts.numel() == 1 and b > 1:
            ts = ts.expand(b)
        ts = ts.contiguous().to(dev)
        tag = (
            f"{grid[0]}x{grid[1]}x{grid[2]}_e{enc_index.shape[1]}"
            + (f"_cp{cp}" if cp > 1 else "")
            + (("_dtile" if cfg.ssta.dense else "_ssta") if sparse else "")
            + ("_cfgg" if self.cfg_gather is not None else "")
        )
        t0 = time.time()
        with torch.no_grad():
            if BLOCKS_PER_GRAPH <= 0:
                fn = self._compiled(("full", tag), lambda: self.dit, f"hv15_dit_{tag}")
                out = fn(x, ts, text, tm, text2, img, enc_index, cos, sin, text_bias, key_bias).to(
                    "cpu"
                )
                if (
                    self.cfg_gather is not None
                ):  # one-graph mode only: host gather (coordinator order)
                    from vllm_omni_neuron.diffusion.distributed.parallel_state import (
                        host_all_gather,
                    )

                    out = torch.cat(host_all_gather(self.cfg_gather, out.contiguous()), dim=0)
            else:
                if sparse:
                    pro = self._compiled(
                        ("pro", tag), lambda: _SSTAPrologue(self.dit), f"hv15_prologue_{tag}"
                    )
                    hidden, enc, temb = pro(
                        vidx, enc_keep, x, ts, text, tm, text2, img, enc_index, text_bias
                    )
                elif cp > 1:
                    pro = self._compiled(
                        ("pro", tag), lambda: _CPPrologue(self.dit), f"hv15_prologue_{tag}"
                    )
                    hidden, enc, temb = pro(vidx, x, ts, text, tm, text2, img, enc_index, text_bias)
                else:
                    pro = self._compiled(
                        ("pro", tag), lambda: _Prologue(self.dit), f"hv15_prologue_{tag}"
                    )
                    hidden, enc, temb = pro(x, ts, text, tm, text2, img, enc_index, text_bias)
                n, k = cfg.num_layers, BLOCKS_PER_GRAPH
                st_ = st if sparse else None
                for s in range(0, n, k):
                    e = min(s + k, n)
                    grp = self._compiled(
                        ("blk", tag, s),
                        lambda s=s, e=e: _BlockGroup(self.dit, s, e, st_),
                        f"hv15_blocks{e - s}_{tag}",
                    )
                    hidden, enc = grp(hidden, enc, temb, cos, sin, key_bias, *ssta_in)
                if sparse:
                    tokens = self._cp_tokens(hidden, temb, tag)  # [B(x2), all slots, C], tile-major
                    out = self.dit.unpatchify(tokens[:, inv], grid)
                elif cp > 1:
                    out = self.dit.unpatchify(self._cp_tokens(hidden, temb, tag)[:, :sv], grid)
                else:
                    cg = self.cfg_gather
                    epi = self._compiled(
                        ("epi", tag), lambda: _Epilogue(self.dit, grid, cg), f"hv15_epilogue_{tag}"
                    )
                    out = epi(hidden, temb)
            out = out.to("cpu")
        _maybe_dump_inputs(self.stats["calls"], out=out)
        self.stats["calls"] += 1
        self.stats["dit_s"] += time.time() - t0
        _prof("dit_call", t0)
        return (out,) if not return_dict else _Out(out)


def _dump_calls() -> set[int]:
    return {int(c) for c in os.environ.get("HV15_DUMP_DIT_CALLS", "0").split(",") if c.strip()}


def _maybe_dump_inputs(call: int, **tensors) -> None:
    """``HV15_DUMP_DIT_INPUTS=<file>``: save the DiT call's inputs on global rank 0 for the calls in
    ``HV15_DUMP_DIT_CALLS`` (default ``0``; a ``{call}`` in the path names one file per call), for
    teacher-forced step checks against a host reference. ``out`` (the device output) is saved too."""
    path = os.environ.get("HV15_DUMP_DIT_INPUTS")
    if not path or call not in _dump_calls() or _global_rank() != 0:
        return
    path = path.format(call=call)
    old = torch.load(path) if "out" in tensors and os.path.exists(path) else {}
    torch.save(
        {**old, **{k: None if v is None else v.detach().cpu() for k, v in tensors.items()}}, path
    )


class _Out:  # minimal Transformer2DModelOutput stand-in
    def __init__(self, sample):
        self.sample = sample

    def __getitem__(self, i):
        return (self.sample,)[i]


def _global_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def _host_threads(env: str, default: int):
    """Context manager: raise torch's intra-op thread count for a host-side phase. The Lite
    diffusion worker pins torch to ONE thread per worker (``_limit_lite_worker_threads``), which
    is right for device dispatch but starves host compute (text encoders, host VAE decode)."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        prev = torch.get_num_threads()
        torch.set_num_threads(int(os.environ.get(env, str(default))))
        try:
            yield
        finally:
            torch.set_num_threads(prev)

    return cm()


def _cfg_info():
    """(cfg_rank, cfg_size, cfg_group) of vLLM-Omni's CFG group; (0, 1, None) when absent."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            get_classifier_free_guidance_rank,
            get_classifier_free_guidance_world_size,
        )

        size = get_classifier_free_guidance_world_size()
        return get_classifier_free_guidance_rank(), size, get_cfg_group()
    except (AssertionError, ImportError):
        return 0, 1, None


def _cfg_gather_on_device(group) -> bool:
    """Whether the CFG branch exchange can run in-graph over ``group``.

    Device collectives need a routable group: one aligned 8-core block (a chip or an adjacent chip
    pair) or a validated physical-mesh layout. TP=8 x CFG=2 on 16 cores falls back to the arithmetic
    CFG pairs ``{r, r + 8}``, which have no device-to-device path (the NEFF fails to schedule), so
    those branches are exchanged on the host instead."""
    ranks = list(getattr(group, "ranks", None) or [])
    if not ranks or min(ranks) // 8 == max(ranks) // 8:
        return True
    try:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import _supports_physical_mesh

        from .transformer import cp_state

        return bool(_supports_physical_mesh(_tp_info()[1], cp_state()[0], len(ranks)))
    except Exception:  # noqa: BLE001
        return False


def _tp_info():
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        return (
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
            get_tp_group(),
        )
    except (AssertionError, ImportError):
        return 0, 1, None


class NeuronHunyuanVideo15Pipeline(HunyuanVideo15Pipeline):
    """Upstream pipeline; ``__init__`` builds host text encoders and the Neuron DiT / VAE."""

    # vLLM-Omni's engine warm-up would compile a throwaway 512x512 geometry (every graph is
    # shape-specialised on Neuron); skip it, the first real request compiles what it needs.
    dummy_run_num_frames = 0

    def __init__(self, *, od_config, prefix: str = ""):
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer, T5EncoderModel

        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = torch.device("cpu")  # host-side pipeline math; the DiT / VAE own their device
        dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        model = local_model_dir(od_config.model)
        local = os.path.exists(model)
        mcfg = dict(od_config.model_config or {})
        self._tp_rank, self._tp_size, self._tp_group = _tp_info()
        if PROFILE:  # rank -> core placement, to check a permuted stage `devices:` order
            print(
                f"[hv15-prof] placement rank={_global_rank()} core={os.environ.get('NEURON_RT_VISIBLE_CORES')} "
                f"tp_rank={self._tp_rank}/{self._tp_size} cfg_rank={_cfg_info()[0]}",
                flush=True,
            )

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local
        )
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local
        )
        self.text_encoder = self.text_encoder_2 = None
        self.text_tower = self._build_text_tower(model, dtype)
        self._dev_mllm: dict | None = None
        self._text_cache: OrderedDict = OrderedDict()
        if (
            _global_rank() == 0
        ):  # one host copy (~15 GB) for the whole world, not one per TP/CFG group
            t0 = time.time()
            if self.text_tower is None:
                self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
                    model, subfolder="text_encoder", local_files_only=local, torch_dtype=dtype
                ).eval()
            self.text_encoder_2 = T5EncoderModel.from_pretrained(
                model, subfolder="text_encoder_2", local_files_only=local, torch_dtype=dtype
            ).eval()
            logger.info("HunyuanVideo-1.5 host text encoders loaded in %.1fs", time.time() - t0)

        vae_dtype = getattr(torch, str(mcfg.get("vae_dtype", "bfloat16")))
        self.vae = NeuronHunyuanVideo15VAE.from_pretrained(
            model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local
        )
        # host decode (HV15_VAE_HOST=1): only global rank 0 decodes, the others return a placeholder.
        # Device decode: every rank decodes its share of the tiles and rank 0 merges (skip_decode unused).
        self.vae.skip_decode = _global_rank() != 0
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local
        )
        if od_config.flow_shift is not None:
            self.scheduler._shift = od_config.flow_shift
        self.transformer = NeuronHunyuanVideo15Transformer(od_config)
        self.use_meanflow = False
        self.weights_sources = []  # weights come from our own sharded loader

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio
        self.num_channels_latents = self.vae.config.latent_channels
        self.system_message = SYSTEM_MESSAGE
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729
        self.vision_states_dim = 1152
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(
                od_config, "enable_diffusion_pipeline_profiler", False
            )
        )

    def _build_text_tower(self, model: str, dtype):
        """This rank's shard of the device text tower, or ``None`` (host encoder on global rank 0)."""
        import torch.distributed as dist

        if TEXT_ENCODER == "host" or not dist.is_initialized():
            return None
        if TEXT_ENCODER == "auto" and os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1":
            return None
        world, rank, tp = dist.get_world_size(), dist.get_rank(), TEXT_TP
        if world % tp:
            logger.info(
                "HunyuanVideo-1.5: world %d does not split into text TP=%d groups; host text encoder",
                world,
                tp,
            )
            return None
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_world_group,
            init_model_parallel_group,
        )

        try:
            from vllm_neuron import envs

            backend = envs.get_dist_backend()
        except Exception:  # noqa: BLE001
            backend = dist.get_backend()
        groups = text_groups(world, tp)
        coord = init_model_parallel_group(
            group_ranks=groups,
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="data",
        )
        if tp > 1:
            from vllm_omni_neuron.lite_compat import register_process_group_replica_groups

            register_process_group_replica_groups(coord.device_group.group_name, groups)
        self._text_coord = coord
        return NeuronQwenTextEncoder(
            model,
            dtype=dtype,
            tp=tp,
            tp_rank=rank % tp,
            group=coord.device_group if tp > 1 else None,
        )

    # -- lifecycle ------------------------------------------------------------------------------
    def load_weights(self, weights=None):
        self.transformer.load()
        if self.text_tower is not None:
            self.text_tower.load()
        return None

    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if self.text_tower is not None:
            self.text_tower.to(*args, **kwargs)
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self.transformer.compile(backend, options, **kwargs)
        self.vae.compile(backend, options)
        if self.text_tower is not None:
            self.text_tower.compile(backend, options)
        return self

    # -- host overrides -------------------------------------------------------------------------
    def _mllm_tokens(self, prompt: str):
        t = self.tokenizer.apply_chat_template(
            format_text_input([prompt], self.system_message),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )
        return t.input_ids, t.attention_mask

    def _encode_mllm_on_device(self, prompts: list[str]) -> dict:
        """Every text group encodes its share of ``prompts`` (``text_encoder.assignment``); the results
        land on global rank 0 as ``{prompt: (hidden_states[-3], mask)}`` (empty on the other ranks)."""
        import torch.distributed as dist
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        rank, world, tp = dist.get_rank(), dist.get_world_size(), self.text_tower.tp
        n_groups, g = world // tp, rank // tp
        toks = [self._mllm_tokens(p) for p in prompts]
        mine: dict = {}
        for row in assignment(len(prompts), n_groups):
            slot = row[g]
            emb = self.text_tower(*toks[slot % len(prompts)])
            if slot < len(prompts) and rank % tp == 0:
                mine[slot] = emb
        cpu = get_world_group().cpu_group
        out = {}
        for i, p in enumerate(prompts):
            src, emb = (i % n_groups) * tp, None
            if src == 0:
                emb = mine.get(i)
            elif rank == src:
                dist.send(mine[i].contiguous(), 0, group=cpu)
            elif rank == 0:
                emb = torch.empty(
                    1,
                    toks[i][0].shape[1],
                    self.text_tower.cfg["hidden_size"],
                    dtype=self.text_tower.dtype,
                )
                dist.recv(emb, src, group=cpu)
            if rank == 0:
                out[p] = (emb, toks[i][1])
        return out

    def _get_mllm_prompt_embeds(self, prompt, device, dtype, num_hidden_layers_to_skip: int = 2):
        if self._dev_mllm is None:
            return super()._get_mllm_prompt_embeds(prompt, device, dtype, num_hidden_layers_to_skip)
        emb, mask = self._dev_mllm[prompt[0]]
        crop = self.prompt_template_encode_start_idx
        return emb[:, crop:].to(device=device, dtype=dtype), mask[:, crop:].to(device)

    def encode_prompt(
        self, prompt, device, dtype, negative_prompt=None, do_classifier_free_guidance=False
    ):
        """Encode once per request for the whole world and broadcast the embeddings from global rank 0.

        The Qwen2.5-VL tower runs on the NeuronCores (``HV15_TEXT_ENCODER=auto``: TP over contiguous rank
        groups, the prompts dealt across the groups) or on the host of rank 0 (``host``). byT5 (0.2B, only
        run when the prompt quotes glyph text) stays on the host of rank 0. Results are cached per prompt
        pair on every rank (``HV15_TEXT_CACHE`` entries; every rank sees the same requests)."""
        import torch.distributed as dist

        key = (
            prompt if isinstance(prompt, str) else tuple(prompt),
            negative_prompt,
            bool(do_classifier_free_guidance),
            dtype,
        )
        t0 = time.time()
        if TEXT_CACHE > 0 and key in self._text_cache:
            self._text_cache.move_to_end(key)
            _prof("encode_prompt", t0, rank0_only=True)
            return self._text_cache[key]
        if self.text_tower is not None:
            ps = [prompt] if isinstance(prompt, str) else list(prompt)
            if len(ps) != 1:
                raise ValueError("HunyuanVideo-1.5 encodes one prompt per request")
            prompts = list(
                dict.fromkeys(ps + ([negative_prompt or ""] if do_classifier_free_guidance else []))
            )
            td = time.time()
            self._dev_mllm = self._encode_mllm_on_device(prompts)
            _prof("text_tower_device", td, rank0_only=True)
        out = None
        try:
            if _global_rank() == 0:
                with torch.no_grad(), _host_threads("HV15_TEXT_THREADS", 32):
                    out = super().encode_prompt(
                        prompt, device, dtype, negative_prompt, do_classifier_free_guidance
                    )
        finally:
            self._dev_mllm = None
        if dist.is_initialized() and dist.get_world_size() > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            box = [out]
            dist.broadcast_object_list(box, src=0, group=get_world_group().cpu_group)
            out = box[0]
        _prof("encode_prompt", t0, rank0_only=True)
        dump = os.environ.get("HV15_DUMP_EMBEDS")
        if dump and _global_rank() == 0:
            torch.save([None if x is None else x.detach().cpu() for x in out], dump)
        if TEXT_CACHE > 0:
            self._text_cache[key] = out
            while len(self._text_cache) > TEXT_CACHE:
                self._text_cache.popitem(last=False)
        return out

    def prepare_latents(
        self, batch_size, height, width, num_frames, dtype, device, generator=None, latents=None
    ):
        """Draw the initial noise in fp32, then cast (a bf16 draw is a different, uncorrelated stream)."""
        lat = super().prepare_latents(
            batch_size, height, width, num_frames, torch.float32, device, generator, latents
        )
        return lat.to(dtype)

    def forward(self, req, *args, **kwargs):
        out = super().forward(req, *args, **kwargs)
        if _global_rank() == 0:
            logger.info("hv15 stats: %s", self.transformer.stats)
        return out

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg,
        true_cfg_scale,
        positive_kwargs,
        negative_kwargs,
        cfg_normalize=True,
        output_slice=None,
        kwargs=None,
    ):
        """CFG-parallel (``cfg_parallel_size=2``) with the branch exchange ON THE DEVICE.

        CFG rank 0 runs the positive branch, rank 1 the negative one. The transformer facade is told
        the CFG group for this call (``cfg_gather``) and all-gathers the two branch outputs along the
        batch inside its epilogue graph, in coordinator order (``[positive, negative]``), so every rank
        receives both, applies the same combine and steps the scheduler in lockstep (bit-equal to
        sequential CFG). With ``cfg_parallel_size == 1`` this defers to the sequential base."""
        cfg_rank, cfg_size, group = _cfg_info()
        if not do_true_cfg or cfg_size <= 1:
            return super().predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
                output_slice,
                kwargs,
            )
        if cfg_size != 2:
            raise ValueError(
                f"HunyuanVideo-1.5 CFG-parallel needs cfg_parallel_size 2, got {cfg_size}"
            )
        t0 = time.time()
        on_device = _cfg_gather_on_device(group)
        self.transformer.cfg_gather = group if on_device else None
        try:
            both = self.predict_noise(**(positive_kwargs if cfg_rank == 0 else negative_kwargs))
        finally:
            self.transformer.cfg_gather = None
        both = both.detach().to("cpu")
        if not on_device:
            from vllm_omni_neuron.diffusion.distributed.parallel_state import host_all_gather

            both = torch.cat(
                host_all_gather(group, both.contiguous()), dim=0
            )  # [positive, negative]
        pos, neg = both.chunk(2, dim=0)
        if output_slice is not None:
            pos, neg = pos[:, :output_slice], neg[:, :output_slice]
        res = self.combine_cfg_noise(
            pos,
            neg,
            true_cfg_scale,
            cfg_normalize,
            **({} if kwargs is None else {"kwargs": kwargs}),
        )
        _prof("cfg_step_total", t0)
        return res

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:  # convenience for tools/tests
        return self.vae.decode(latents.float() / self.vae.config.scaling_factor, return_dict=False)[
            0
        ]


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronHunyuanVideo15Pipeline",
    "NeuronHunyuanVideo15Transformer",
    "SYSTEM_MESSAGE",
    "get_hunyuan_video_15_post_process_func",
]
