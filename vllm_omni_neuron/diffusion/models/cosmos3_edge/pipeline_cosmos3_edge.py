# SPDX-License-Identifier: Apache-2.0
"""Neuron Cosmos3-Edge pipeline: upstream vLLM-Omni's Cosmos3 pipeline with the transformer and
the VAE swapped for Neuron-compiled components.

Everything above the transformer call stays upstream's (vendored ``pipeline_cosmos3.py``):
request parsing, prompt templating, tokenisation, CFG branches, the flow-matching scheduler,
I2V/action conditioning, latent normalisation and post-processing. That host-side math runs on
the CPU (``self.device`` is CPU), and only two components touch the NeuronCore:

* ``self.transformer`` -> :class:`NeuronCosmos3EdgeTransformer`, a drop-in for upstream's
  ``Cosmos3EdgeVFMTransformer`` with the same ``forward`` signature and per-CFG-branch cache
  attributes (``cached_kv`` / ``cached_freqs_gen``). It runs the UND tower once per branch
  and the GEN tower per denoising call, each a fixed-shape compiled graph.
* ``self.vae`` -> :class:`NeuronEdgeVae`, the plugin's compiled Wan VAE (the Edge checkpoint's
  Wan2.2-TI2V-5B VAE) behind a host-tensor interface.
"""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import os
import time
from typing import Any

import torch
import torch.nn as nn

from ._vendor import pipeline_cosmos3 as _up
from ._vendor.pipeline_cosmos3 import (
    COSMOS3_EDGE_VIDEO_DEFAULT_FLOW_SHIFT,
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
)
from .gen_tower import EdgeGenConfig, NeuronCosmos3EdgeGEN
from .und_tower import NeuronCosmos3EdgeUND

logger = logging.getLogger(__name__)

PROFILE = os.environ.get("COSMOS3_EDGE_PROFILE", "0") == "1"


def _prof(name: str, t0: float) -> None:
    if PROFILE:
        print(f"[edge-prof] {name} {time.time() - t0:.4f}", flush=True)


_randn_tensor_up = _up.randn_tensor


def _randn_fp32(shape, generator=None, device=None, dtype=None, layout=None):
    """Draw noise in fp32, then cast: torch's RNG stream depends on the dtype it samples in
    (a bf16 draw is uncorrelated with an fp32 draw from the same seed), so without this a bf16
    device run and the fp32 CPU oracle start from different noise and are not comparable."""
    out = _randn_tensor_up(
        shape, generator=generator, device=device, dtype=torch.float32, layout=layout
    )
    return out.to(dtype) if dtype is not None else out


_up.randn_tensor = _randn_fp32

TEXT_BUCKETS = tuple(
    int(b) for b in os.environ.get("COSMOS3_EDGE_TEXT_BUCKETS", "64,128,256,512,1024").split(",")
)

TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]

PIPELINE_REGISTRY = [
    {
        # model_index.json `_class_name` of nvidia/Cosmos3-Edge
        "model_arch": "Cosmos3OmniPipeline",
        "class_name": "NeuronCosmos3EdgePipeline",
        "pre_process_func_name": "get_cosmos3_pre_process_func",
        "post_process_func_name": "get_cosmos3_post_process_func",
    },
    {
        # upstream vLLM-Omni registry key for the same pipeline
        "model_arch": "Cosmos3OmniDiffusersPipeline",
        "class_name": "NeuronCosmos3EdgePipeline",
        "pre_process_func_name": "get_cosmos3_pre_process_func",
        "post_process_func_name": "get_cosmos3_post_process_func",
    },
]


def pick_text_bucket(n: int) -> int:
    for b in TEXT_BUCKETS:
        if n <= b:
            return b
    raise ValueError(
        f"prompt is {n} tokens; the largest text bucket is {TEXT_BUCKETS[-1]} (COSMOS3_EDGE_TEXT_BUCKETS)"
    )


def _host(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    x = x.detach().to("cpu")
    return (x.to(dtype) if dtype is not None else x).contiguous()


def _is_rank0() -> bool:
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


class NeuronCosmos3EdgeTransformer(nn.Module):
    """Drop-in for upstream ``Cosmos3EdgeVFMTransformer`` inside the Cosmos3 pipeline."""

    def __init__(
        self,
        od_config,
        temporal_compression_factor=None,
        sound_gen=False,
        sound_dim=None,
        sound_latent_fps=None,
    ):
        super().__init__()
        del (
            sound_gen,
            sound_dim,
            sound_latent_fps,
        )  # sound is not ported; the pipeline forces it off
        self.model_path = od_config.model
        self.cfg = EdgeGenConfig.from_model_dir(self.model_path)
        dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        self.dtype = dtype
        self.und = NeuronCosmos3EdgeUND(self.cfg, dtype=dtype)
        self.gen = NeuronCosmos3EdgeGEN(self.cfg, dtype=dtype)
        # attributes the upstream pipeline reads
        self.latent_channel_size = self.cfg.latent_channel
        self.action_gen = self.cfg.action_gen
        self.action_dim = self.cfg.action_dim
        self.sound_gen = False
        self.cached_kv: Any = (
            None  # (kv tuple on device, padded text mask) of the current CFG branch
        )
        self.cached_freqs_gen: Any = None  # (cos, sin, key_bias) on device, per geometry + branch
        self._device = torch.device("cpu")
        self._und_fn = self.und
        self._gen_fn = self.gen
        self._gen_action_fn = self.gen.forward_action
        self._init_context_parallel()
        self._gen_cp_fn = self.gen.forward_cp
        self._init_cfg_gather()
        self.stats = {"und_calls": 0, "gen_calls": 0, "und_s": 0.0, "gen_s": 0.0, "gen_skipped": 0}
        self._step_cache: dict = {}  # per-branch state of the training-free step cache

    # -- lifecycle ----------------------------------------------------------------------
    def _init_context_parallel(self) -> None:
        """CP from the stage config's ``ring_degree`` (vLLM-Omni's sequence-parallel group, rebuilt on
        Trn2's physical mesh for TP=8 x CP layouts by the plugin worker). Video GEN calls then split
        the patchified sequence across the CP group (:meth:`NeuronCosmos3EdgeGEN.forward_cp`); the
        UND prefill (<= 1k text tokens) and action-mode calls stay unsplit on every CP rank."""
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            cp = get_sp_group()
        except (AssertionError, ImportError, AttributeError):
            return
        if cp.world_size <= 1:
            return
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=self.gen.tp_size, cp_size=cp.world_size)
        self.gen.set_context_parallel(
            cp.world_size, cp.rank_in_group, cp.device_group, list(cp.ranks)
        )

    def _init_cfg_gather(self) -> None:
        """CFG-parallel branch exchange ON DEVICE: when the pipeline arms it (``_cfg_parts`` is a list,
        see :meth:`NeuronCosmos3EdgePipeline.predict_noise_maybe_with_cfg`), each GEN output is
        all-gathered over the CFG group's device group before it leaves the NeuronCore (a small
        compiled graph per output shape; the replica groups keep the list order = CFG group-rank
        order). The parts land in ``_cfg_parts``, one list per output element, in CFG rank order."""
        self._cfg_parts: list | None = None
        self._cfg_rank, self._cfg_gather_fn = 0, None
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_cfg_group

            cfg = get_cfg_group()
        except (AssertionError, ImportError, AttributeError):
            return
        if cfg.world_size <= 1:
            return
        from .gen_tower import group_all_gather_rows

        group, size, ranks = cfg.device_group, cfg.world_size, list(cfg.ranks)
        self._cfg_size, self._cfg_rank = size, cfg.rank_in_group

        def cfg_gather(x):
            return group_all_gather_rows(x, group, size, ranks)

        self._cfg_gather_raw = cfg_gather
        self._cfg_gather_fn = cfg_gather

    def _host_out(self, out: torch.Tensor, post=None) -> torch.Tensor:
        """A GEN output on the device -> this rank's host tensor (``post`` = host-side reshaping,
        e.g. unpatchify). With the CFG gather armed, the output is first all-gathered over the CFG
        device group and every branch's part is recorded in ``_cfg_parts`` (CFG rank order)."""
        post = post or (lambda x: x)
        if self._cfg_parts is None or self._cfg_gather_fn is None:
            return post(out.to("cpu"))
        g = self._cfg_gather_fn(out).to("cpu")
        parts = [post(p) for p in g.chunk(self._cfg_size, dim=0)]
        self._cfg_parts.append(parts)
        return parts[self._cfg_rank]

    def _gen_cp_call(self, hidden_states, ts, cg, sg, kb, nm, kv, t, h, w, s_video):
        """One video GEN call split over the CP group: this rank's contiguous token slice in (``cg`` /
        ``sg`` / ``nm`` are already this rank's slice, cut on the host), the full velocity out. The
        ``proj_out`` tokens are all-gathered over the CP device group inside the compiled graph
        (:meth:`NeuronCosmos3EdgeGEN.forward_cp`), in CP group-rank order -- no host collective, so the
        descending physical-mesh CP groups (e.g. ``[12, 8]``) reassemble correctly."""
        gen = self.gen
        sl = self._cp_slice(s_video)
        tok = gen._patchify(_host(hidden_states, self.dtype))[:, sl].contiguous()
        out = self._gen_cp_fn(tok.to(self._device), ts, cg, sg, kb, nm, *kv)
        return self._host_out(out, lambda x: gen._unpatchify(x, t, h, w))

    def _cp_slice(self, s: int) -> slice:
        size, rank = self.gen.cp_size, self.gen.cp_rank
        if s % size:
            raise ValueError(
                f"GEN sequence {s} is not divisible by the context-parallel degree {size}; "
                "pick a frame count / resolution whose token count is"
            )
        loc = s // size
        return slice(rank * loc, (rank + 1) * loc)

    def load(self) -> None:
        t0 = time.time()
        self.und.load_weights(self.model_path, "cpu")
        self.gen.load_weights(self.model_path, "cpu")
        logger.info("Cosmos3-Edge transformer weights loaded in %.1fs", time.time() - t0)

    def post_load_weights(self) -> None:  # upstream hook; the timestep embedder is fp32 already
        pass

    def validate_loaded_weights(self, loaded) -> None:  # strict load is done by our own loaders
        pass

    def reset_cache(self) -> None:
        self.cached_kv = None
        self.cached_freqs_gen = None
        self._step_cache = {}

    # -- training-free step cache (SeaCache/TeaCache-style) -----------------------------------
    # COSMOS3_EDGE_STEP_CACHE=<thresh> enables it (default off). On a step whose GEN input moved less
    # than <thresh> (relative L1 against the input of the last fully computed step) the GEN forward
    # is skipped and its output extrapolated from the last two computed outputs (first order in t,
    # NVIDIA cosmos-framework DiffusionCache style). Guard rails: the first STEP_CACHE_WARMUP calls of
    # a branch always compute, at most STEP_CACHE_MAX_SKIP consecutive skips, never skip below
    # the [STEP_CACHE_TMIN, STEP_CACHE_TMAX] timestep window (units of the call). Each CFG branch (``cached_kv`` identity) keeps its
    # own state; under CFG-parallel both ranks see the same latent, so they skip the same steps.
    @staticmethod
    def _sc_cfg():
        thr = float(os.environ.get("COSMOS3_EDGE_STEP_CACHE", "0") or 0)
        return (
            thr,
            int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_WARMUP", "2")),
            int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_MAX_SKIP", "2")),
            float(os.environ.get("COSMOS3_EDGE_STEP_CACHE_TMIN", "0")),
            int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_ORDER", "1")),
            float(os.environ.get("COSMOS3_EDGE_STEP_CACHE_TMAX", "1e9")),
        )

    def _sc_try_skip(self, x_host: torch.Tensor, t_val: float):
        """Return (skip_output_or_None, state). ``x_host`` is the GEN input on the host."""
        thr, warmup, max_skip, tmin, order, tmax = self._sc_cfg()
        if thr <= 0 or self.cached_kv is None:
            return None, None
        st = self._step_cache.setdefault(
            id(self.cached_kv[0]), {"n": 0, "skips": 0, "hist": [], "x_ref": None}
        )
        st["n"] += 1
        if (
            st["x_ref"] is None
            or st["n"] <= warmup
            or st["skips"] >= max_skip
            or not (tmin <= t_val <= tmax)
        ):
            return None, st
        ref = st["x_ref"]
        d = ((x_host.float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-8)).item()
        if os.environ.get("COSMOS3_EDGE_PROFILE") == "1":
            print(f"[edge-prof] sc_dist {d:.5f} t={t_val:.4f} n={st['n']}", flush=True)
        if d >= thr:
            return None, st
        (t1, o1) = st["hist"][-1]
        out = o1
        if order >= 1 and len(st["hist"]) >= 2:
            (t0_, o0) = st["hist"][-2]
            if abs(t1 - t0_) > 1e-12:
                out = o1 + (o1 - o0) * ((t_val - t1) / (t1 - t0_))
        st["skips"] += 1
        self.stats["gen_skipped"] += 1
        return out.to(o1.dtype), st

    @staticmethod
    def _sc_record(st, x_host: torch.Tensor, t_val: float, out: torch.Tensor) -> None:
        if st is None:
            return
        st["x_ref"] = x_host.float().clone()
        st["skips"] = 0
        st["hist"] = (st["hist"] + [(t_val, out)])[-2:]

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.und.to(self._device)
            self.gen.to(self._device)
            self.gen.t_freqs = self.gen.t_freqs.to(self._device)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(TRANSFORMER_COMPILER_ARGS)}

        self._und_fn = torch.compile(
            self.und, backend=backend, options=opts("cosmos3_edge_und"), **kw
        )
        self._gen_fn = torch.compile(
            self.gen, backend=backend, options=opts("cosmos3_edge_gen"), **kw
        )
        self._gen_action_fn = torch.compile(
            self.gen.forward_action, backend=backend, options=opts("cosmos3_edge_gen_action"), **kw
        )
        self._gen_cp_fn = torch.compile(
            self.gen.forward_cp, backend=backend, options=opts("cosmos3_edge_gen_cp"), **kw
        )
        if self._cfg_gather_fn is not None:
            self._cfg_gather_fn = torch.compile(
                self._cfg_gather_raw,
                backend=backend,
                options={**base, "model_name": "cosmos3_edge_cfg_gather"},
                **kw,
            )

    # -- forward (upstream signature) -------------------------------------------------
    def forward(
        self,
        hidden_states,
        timestep,
        text_ids,
        text_mask,
        video_shape,
        fps=None,
        action_latents=None,
        action_domain_ids=None,
        action_noisy_mask=None,
        action_start_frame_offset: int = 1,
        action_fps=None,
        sound_latents=None,
        noisy_frame_mask=None,
        control_latents=None,
        control_weights=None,
        transfer_share_vision_temporal_positions=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected Cosmos3 transformer kwargs: {sorted(kwargs)}")
        if sound_latents is not None:
            raise NotImplementedError("Cosmos3-Edge on Neuron: sound generation is not supported")
        if control_latents is not None:
            raise NotImplementedError(
                "Cosmos3-Edge on Neuron: transfer/control conditioning is not supported"
            )
        dev, dt = self._device, self.dtype
        t, h, w = (int(x) for x in video_shape)
        if h % self.cfg.patch or w % self.cfg.patch:
            raise ValueError(
                f"latent size {h}x{w} must be a multiple of the patch size {self.cfg.patch}"
            )
        t_host0 = time.time()
        b = text_ids.shape[0]
        real = int(text_mask.sum(dim=1).max().item())
        bucket = pick_text_bucket(real)
        ids = torch.zeros(b, bucket, dtype=torch.long)
        ids[:, : text_ids.shape[1]] = _host(text_ids).long()
        mask = torch.zeros(b, bucket, dtype=torch.long)
        mask[:, : text_mask.shape[1]] = _host(text_mask).long()

        s_action = int(action_latents.shape[1]) if action_latents is not None else 0
        s_video = t * (h // self.cfg.patch) * (w // self.cfg.patch)

        if self.cached_kv is None:
            t0 = time.time()
            cu, su = self.und.rope_tables(mask)
            with torch.no_grad():
                kv = self._und_fn(
                    ids.to(dev), cu.to(dt).contiguous().to(dev), su.to(dt).contiguous().to(dev)
                )
            self.cached_kv = (tuple(kv), mask)
            self.cached_freqs_gen = None
            self.stats["und_calls"] += 1
            self.stats["und_s"] += time.time() - t0
            _prof("und", t0)
        kv, mask = self.cached_kv
        geom = (
            t,
            h,
            w,
            s_action,
            None if fps is None else float(fps),
            action_start_frame_offset,
            None if action_fps is None else float(action_fps),
        )
        cp = self.gen.cp_size > 1 and s_action == 0
        if self.cached_freqs_gen is None or self.cached_freqs_gen[0] != geom:
            cg, sg = self.gen.rope_tables(
                mask,
                t,
                h,
                w,
                fps,
                t_action=s_action,
                action_start_frame_offset=action_start_frame_offset,
                action_fps=action_fps,
            )
            if cp:  # this CP rank's token slice, cut on the host
                sl = self._cp_slice(s_video)
                cg, sg = cg[:, sl], sg[:, sl]
            kb = self.gen.key_bias(mask, s_video + s_action)
            self.cached_freqs_gen = (
                geom,
                cg.to(dt).contiguous().to(dev),
                sg.to(dt).contiguous().to(dev),
                kb.contiguous().to(dev),
            )
        _, cg, sg, kb = self.cached_freqs_gen

        hw = (h // self.cfg.patch) * (w // self.cfg.patch)
        if noisy_frame_mask is not None:
            nm = (
                _host(noisy_frame_mask)
                .float()[:, 0, :, 0, 0]
                .unsqueeze(-1)
                .expand(-1, -1, hw)
                .reshape(b, -1, 1)
            )
        else:
            nm = torch.ones(b, s_video, 1)
        if cp:
            nm = nm[:, self._cp_slice(s_video)]
        x = _host(hidden_states, dt).to(dev)
        ts = _host(timestep).float().reshape(-1)
        if ts.numel() == 1 and b > 1:
            ts = ts.expand(b)
        ts = ts.contiguous().to(dev)
        nm = nm.to(dt).contiguous().to(dev)
        _prof("gen_host_in", t_host0)
        t0 = time.time()
        with torch.no_grad():
            if cp:
                out = self._gen_cp_call(hidden_states, ts, cg, sg, kb, nm, kv, t, h, w, s_video)
            elif s_action == 0:
                x_cpu = _host(hidden_states, dt)
                t_val = float(_host(timestep).float().reshape(-1)[0].item())
                skipped, sc_state = self._sc_try_skip(x_cpu, t_val)
                if skipped is not None:
                    _prof("gen_skip", t0)
                    return skipped
                out = self._host_out(self._gen_fn(x, ts, cg, sg, kb, nm, *kv))
                self._sc_record(sc_state, x_cpu, t_val, out)
            else:
                domain = (
                    0
                    if action_domain_ids is None
                    else int(_host(action_domain_ids).reshape(-1)[0].item())
                )
                w_in, b_in, w_out, b_out = self.gen.domain_weights(domain, dev)
                am = (
                    torch.ones(b, s_action, 1)
                    if action_noisy_mask is None
                    else _host(action_noisy_mask).float().reshape(b, s_action, 1)
                )
                video, action = self._gen_action_fn(
                    x,
                    ts,
                    cg,
                    sg,
                    kb,
                    nm,
                    _host(action_latents, dt).to(dev),
                    am.to(dt).contiguous().to(dev),
                    w_in,
                    b_in,
                    w_out,
                    b_out,
                    *kv,
                )
                out = (self._host_out(video), self._host_out(action))
        self.stats["gen_calls"] += 1
        self.stats["gen_s"] += time.time() - t0
        _prof("gen_call", t0)
        dump = os.environ.get("COSMOS3_EDGE_DUMP_GEN0")
        min_tok = int(
            os.environ.get("COSMOS3_EDGE_DUMP_GEN0_MIN_TOKENS", "1024")
        )  # skip the engine's dummy warm-up
        if (
            dump
            and s_action == 0
            and s_video >= min_tok
            and not getattr(self, "_gen0_dumped", False)
            and _is_rank0()
        ):
            self._gen0_dumped = True
            # first GEN call's exact inputs + output, for a CPU re-run of the same call (parity)
            torch.save(
                {
                    "hidden_states": _host(hidden_states, torch.float32),
                    "timestep": _host(timestep).float(),
                    "text_ids": _host(text_ids),
                    "text_mask": _host(text_mask),
                    "video_shape": (t, h, w),
                    "fps": None if fps is None else float(fps),
                    "noisy_frame_mask": None
                    if noisy_frame_mask is None
                    else _host(noisy_frame_mask).float(),
                    "out": out.float(),
                },
                dump,
            )
        return out


class NeuronEdgeVae(nn.Module):
    """Host-tensor facade over the plugin's compiled Wan VAE (``NeuronAutoencoderKLWan``)."""

    @classmethod
    def from_pretrained(cls, model_path, subfolder="vae", torch_dtype=torch.bfloat16, **kwargs):
        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            NeuronAutoencoderKLWan,
        )

        vae = NeuronAutoencoderKLWan.from_pretrained(
            model_path, subfolder=subfolder, torch_dtype=torch_dtype, **kwargs
        )
        return cls(vae.eval())

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae.config
        self._device = torch.device("cpu")
        self._compiled_encoder = False
        self._host_pp_group = None  # gloo group for host-gathered tile-parallel decode (long clips)
        self._host_encoder = None  # CPU copy of the encoder when encoding on the host

    @staticmethod
    def _encode_on_host() -> bool:
        """``COSMOS3_VAE_ENCODE=host|device`` (default ``device``). The conditioning-frame encode
        (I2V / action) runs on the NeuronCore; ``host`` runs it on an fp32 CPU copy of the encoder
        instead (bit-identical to the eager CPU encoder; the Trn2 fallback before tiled encode)."""
        return os.environ.get("COSMOS3_VAE_ENCODE", "auto") == "host"

    @staticmethod
    def encode_tile() -> tuple[int, int] | None:
        """``(tile_px, overlap_px)`` for the device encode, or ``None`` for one untiled graph.

        ``COSMOS3_VAE_ENCODE_TILE=<tile>,<overlap>`` (``0`` = untiled). Default ``192,96`` on
        NeuronCore-v3+ (Trn2), untiled on NeuronCore-v2 (validated on inf2). On Trn2 the Wan2.2-5B
        encoder graph compiles up to 192 px and fails from 208 px up (``NCC_IDDT901``), so frames
        above 192 px are encoded as fixed-shape 192 px tiles with the plugin's shared tiling (one
        compiled graph for every resolution). 96 px overlap is the blend knee measured on the real
        weights."""
        spec = os.environ.get("COSMOS3_VAE_ENCODE_TILE")
        if spec is None:
            from .nc_dispatch import neuron_core_generation

            return (192, 96) if neuron_core_generation() >= 3 else None
        if spec.strip() in ("", "0"):
            return None
        tile, overlap = (int(v) for v in spec.split(","))
        if not 0 <= overlap < tile:
            raise ValueError(f"COSMOS3_VAE_ENCODE_TILE: need 0 <= overlap < tile, got {spec!r}")
        return tile, overlap

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and torch.device(device).type != "cpu":
            if self._host_encoder is None and self._encode_on_host():
                import copy

                from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
                    NeuronWanEncoder3d,
                )

                self._host_encoder = NeuronWanEncoder3d(
                    copy.deepcopy(self.vae.encoder).float(),
                    copy.deepcopy(self.vae.quant_conv).float(),
                ).eval()
            self._device = torch.device(device)
            self.vae.to(self._device)
        return self

    def compile(
        self, backend: str, options: dict | None = None, compile_encoder: bool = False, **kwargs
    ) -> None:
        compile_encoder = compile_encoder and self._host_encoder is None
        opts = {
            **(options or {}),
            "model_name": "cosmos3_edge_vae",
            "compiler_args": list(VAE_COMPILER_ARGS),
        }
        self.vae.compile(
            backend=backend,
            options=opts,
            fullgraph=True,
            dynamic=False,
            compile_encoder=compile_encoder,
        )
        self._compiled_encoder = compile_encoder

    def _host_encode(self, x: torch.Tensor) -> torch.Tensor:
        """The device ``_encode`` loop, run eagerly in fp32 on the host CPU copy."""
        vae, p = self.vae, self.config.patch_size
        x = _host(x, torch.float32)
        b, c, num_frame, height, width = x.shape
        cache_ref = torch.empty(b, c * p * p, 1, height // p, width // p) if p is not None else x
        feat_map = vae._init_enc_feat_cache(cache_ref)
        feat_map = [f.to("cpu", torch.float32) if torch.is_tensor(f) else f for f in feat_map]
        chunks = []
        for i in range(1 + (num_frame - 1) // 4):
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            result = self._host_encoder(chunk, *feat_map, first_chunk=(i == 0), patch_size=p)
            chunks.append(result[0])
            feat_map = list(result[1:])
        return torch.cat(chunks, dim=2).to(self.dtype)

    def _ensure_decoder(self) -> None:
        """Uncompiled (eager) decoder graphs for CPU-mode runs (the CPU reference / oracle)."""
        if getattr(self.vae, "_compiled_decoder_first", None) is None:
            from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
                NeuronWanDecoder3d,
            )

            eager = NeuronWanDecoder3d(self.vae.post_quant_conv, self.vae.decoder)
            self.vae._compiled_decoder_first = eager
            self.vae._compiled_decoder_rest = eager

    def _skip_redundant_decode(self) -> bool:
        """Under CFG-parallel every rank reaches decode; without VAE patch-parallelism only rank 0's
        frames are returned, so a non-zero rank's full decode is wasted work that also competes for
        the chip's HBM (and, on a cold cache, re-compiles the same graphs behind a file lock)."""
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            return False
        ex = getattr(self.vae, "distributed_executor", None)
        return not (self.vae.use_tiling and ex is not None and self.vae.is_distributed_enabled())

    def decode(self, z, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        dump = os.environ.get("COSMOS3_EDGE_DUMP_LATENT")
        if dump and _is_rank0():  # every rank sees the same latent; one writer, no torn file
            torch.save(_host(z, torch.float32), dump)
        digest_dir = os.environ.get("COSMOS3_EDGE_RANK_DIGEST")
        if digest_dir:  # all-rank agreement gate: every rank digests the final latent it decodes
            import torch.distributed as dist

            from vllm_omni_neuron.testing import write_rank_digest

            n = self._digest_calls = getattr(self, "_digest_calls", 0) + 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            write_rank_digest(
                os.path.join(digest_dir, f"decode_{n:03d}"), rank, {"latents": _host(z)}
            )
        if (
            os.environ.get("COSMOS3_EDGE_DECODE_ALL_RANKS", "0") != "1"
            and self._skip_redundant_decode()
        ):
            b, _, t, h, w = z.shape
            r = self.vae.spatial_compression_ratio
            tr = int(getattr(self.vae.config, "scale_factor_temporal", 4) or 4)
            out = torch.zeros(b, 3, (t - 1) * tr + 1, h * r, w * r, dtype=self.dtype)
            return DecoderOutput(sample=out) if return_dict else (out,)
        self._ensure_decoder()
        t0 = time.time()
        if self._host_gather_decode(z):
            out = self._host_pp_decode(z)
        else:
            out = self.vae.decode(_host(z, self.dtype).to(self._device), return_dict=False)[0].to(
                "cpu"
            )
        _prof("vae_decode", t0)
        return DecoderOutput(sample=out) if return_dict else (out,)

    def _host_gather_decode(self, z) -> bool:
        """Optional fallback: patch-parallel decode with a HOST gather (point-to-point gloo) for clips
        longer than ``COSMOS3_EDGE_VAE_HOST_GATHER_T`` latent frames. Off by default (unset or 0):
        the shared device plane gather handles any plane count, including multi-chunk gathers."""
        if self._host_pp_group is None:
            return False
        return int(z.shape[2]) > _host_gather_t()

    def _host_pp_decode(self, z) -> torch.Tensor:
        """Same fixed-shape tiles as the device patch-parallel decode (the shared ``TileGrid`` /
        ``merge_tiles`` of ``NeuronAutoencoderKLWan.tiled_decode``), dealt round-robin over the gloo
        group; each rank decodes its tiles through the compiled decoder frame by frame. The tiles
        travel to rank 0 as raw tensors over point-to-point gloo (no pickling), and rank 0 posts its
        receives BEFORE decoding its own tile so the transfers overlap that decode. Bit-identical
        to the single-process tiled decode. Non-root ranks get zeros of the output shape (only rank
        0's frames are returned)."""
        import torch.distributed as dist

        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            unpatchify,
        )
        from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, merge_tiles

        vae, group = self.vae, self._host_pp_group
        b, _, t, height, width = z.shape
        ratio = vae.spatial_compression_ratio
        p = vae.config.patch_size
        out_ratio = ratio if p is None else ratio // p
        tile = (vae.tile_sample_min_height // ratio, vae.tile_sample_min_width // ratio)
        stride = (vae.tile_sample_stride_height // ratio, vae.tile_sample_stride_width // ratio)
        grid = TileGrid.for_axes(
            total=(height, width), tile=tile, stride=stride, out_scale=out_ratio
        )
        zp = grid.pad_input(_host(z, self.dtype))
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        idxs = grid.indices()
        tr = int(getattr(vae.config, "scale_factor_temporal", 4) or 4)
        if rank == 0:
            # every tile decodes to the same shape: [B, C_out, T_out, tile*out_ratio, tile*out_ratio]
            c_out = 3 if p is None else 3 * p * p
            shape = (b, c_out, (t - 1) * tr + 1, tile[0] * out_ratio, tile[1] * out_ratio)
            tiles, reqs = {}, []
            for n, idx in enumerate(idxs):
                if n % world:
                    tiles[idx] = torch.empty(shape, dtype=self.dtype)
                    reqs.append(
                        dist.irecv(
                            tiles[idx], src=dist.get_global_rank(group, n % world), group=group
                        )
                    )
            for n, idx in enumerate(idxs):
                if n % world == 0:
                    tiles[idx] = vae._tile_decode_one(grid.slice_input(zp, idx)).to(self.dtype)
            for r in reqs:
                r.wait()
            dec = merge_tiles(tiles, grid)
            if p is not None:
                dec = unpatchify(dec, patch_size=p)
            return torch.clamp(dec, min=-1.0, max=1.0).to("cpu")
        for n, idx in enumerate(idxs):
            if n % world == rank:
                dist.send(
                    vae._tile_decode_one(grid.slice_input(zp, idx)).to(self.dtype).contiguous(),
                    dst=dist.get_global_rank(group, 0),
                    group=group,
                )
        return torch.zeros(b, 3, (t - 1) * tr + 1, height * ratio, width * ratio, dtype=self.dtype)

    def _host_pp_decode_object_gather(self, z) -> torch.Tensor:
        """Reference implementation (the shared tiled_decode with a gloo tile_parallel_group,
        pickled gather_object); kept for the CPU equivalence test."""
        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            NeuronAutoencoderKLWan,
        )

        vae = self.vae
        saved, vae.tile_parallel_group = vae.tile_parallel_group, self._host_pp_group
        try:
            out = NeuronAutoencoderKLWan.tiled_decode(vae, _host(z, self.dtype), return_dict=False)[
                0
            ]
        finally:
            vae.tile_parallel_group = saved
        if out is None:
            b, _, t, h, w = z.shape
            r = vae.spatial_compression_ratio
            tr = int(getattr(vae.config, "scale_factor_temporal", 4) or 4)
            out = torch.zeros(b, 3, (t - 1) * tr + 1, h * r, w * r, dtype=self.dtype)
        return out.to("cpu")

    def encode(self, x, return_dict=True):
        """Conditioning-frame encode; the [B, 2*z, t, h, w] moments come back to the host, where
        the pipeline's latent math runs, before the Gaussian split.

        The host-side pipeline math (this included) runs on every TP rank's worker process on
        identical inputs, so only TP rank 0 encodes and the others receive the small moments
        tensor over the TP group's gloo ``cpu_group`` (the channel CFG-parallel combine uses).
        Rank 0 encodes on the NeuronCore (tiled above :meth:`encode_tile`'s size) or, with
        ``COSMOS3_VAE_ENCODE=host``, on the fp32 CPU copy."""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        t0 = time.time()
        # COSMOS3_EDGE_ENCODE_WATCHDOG_S=<s>: dump every thread's Python stack to stderr if this
        # encode has not returned after <s> seconds (diagnoses a rank stuck in a tile vs the gather).
        watchdog = float(os.environ.get("COSMOS3_EDGE_ENCODE_WATCHDOG_S", "0") or 0)
        if watchdog > 0:
            faulthandler.dump_traceback_later(watchdog, repeat=False)
        # Keep the shared VAE's own tiling OFF for the encode: its tiled_encode deals tiles across
        # the default (WORLD) group, which would deadlock a rank-0-only call. Device tiling is done
        # here, single-rank, by device_tiled_encode.
        tiling, self.vae.use_tiling = self.vae.use_tiling, False
        try:
            with torch.no_grad():
                if self._host_encoder is None and self._tile_parallel_encode(x):
                    h = self._rank0_broadcast(self._device_encode, x, all_ranks=True)
                else:
                    fn = (
                        self._host_encode if self._host_encoder is not None else self._device_encode
                    )
                    h = self._rank0_broadcast(fn, x)
        finally:
            self.vae.use_tiling = tiling
            if watchdog > 0:
                faulthandler.cancel_dump_traceback_later()
        _prof("vae_encode", t0)
        dump = os.environ.get("COSMOS3_EDGE_DUMP_ENCODE_LATENT")
        if dump and _is_rank0():
            torch.save(h.float(), dump)
        dist = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)

    def _device_encode(self, x: torch.Tensor, group=None) -> torch.Tensor | None:
        tile = self.encode_tile()
        height, width = x.shape[-2:]
        if tile is not None and max(height, width) > tile[0]:
            return device_tiled_encode(self.vae, _host(x, self.dtype), *tile, group=group)
        return self.vae._encode(_host(x, self.dtype).to(self._device)).to("cpu")

    def _tile_parallel_encode(self, x: torch.Tensor) -> bool:
        """Deal the encode tiles across the TP group (``COSMOS3_VAE_ENCODE_TP``, default on): every
        TP rank holds the compiled encoder, and the tiles are independent, so a 640 px frame's 36
        tiles run ~36/TP per rank instead of all on rank 0."""
        if os.environ.get("COSMOS3_VAE_ENCODE_TP", "1") == "0":
            return False
        tile = self.encode_tile()
        return tile is not None and max(x.shape[-2:]) > tile[0] and _tp_cpu_group()[0] > 1

    def _rank0_broadcast(self, fn, x: torch.Tensor, all_ranks: bool = False) -> torch.Tensor:
        """``fn(x)`` on TP rank 0 only (``all_ranks``: ``fn(x, group)`` on every TP rank, which
        returns the result on TP rank 0 and ``None`` elsewhere); the result is broadcast to the
        other TP ranks. Under CFG-parallel each CFG replica's TP group does this independently."""
        tp_size, group = _tp_cpu_group()
        if tp_size <= 1:
            return fn(x).to(self.dtype)
        import torch.distributed as dist

        rank, src = dist.get_rank(group), dist.get_global_rank(group, 0)
        h = fn(x, group) if all_ranks else (fn(x) if rank == 0 else None)
        if rank == 0:
            h = h.to(self.dtype).contiguous()
            shape = torch.tensor(list(h.shape), dtype=torch.long)
        else:
            shape = torch.zeros(5, dtype=torch.long)
        dist.broadcast(shape, src=src, group=group)
        if rank != 0:
            h = torch.empty(tuple(shape.tolist()), dtype=self.dtype)
        dist.broadcast(h, src=src, group=group)
        return h


def _host_gather_t() -> int:
    """``COSMOS3_EDGE_VAE_HOST_GATHER_T``: latent-frame threshold of the host-gather decode fallback
    (0 or unset = off)."""
    return int(os.environ.get("COSMOS3_EDGE_VAE_HOST_GATHER_T", "0") or 0)


def _tp_cpu_group() -> tuple[int, object]:
    """(TP world size, the TP group's gloo ``cpu_group``); (1, None) without a TP group."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
    except (AssertionError, ImportError):  # TP group not initialized: single process
        return 1, None
    return (size, get_tp_group().cpu_group) if size > 1 else (1, None)


def device_tiled_encode(
    vae, x: torch.Tensor, tile_px: int, overlap_px: int, group=None
) -> torch.Tensor | None:
    """Spatially tiled encode of host pixels ``x`` ``[B, C, T, H, W]`` through ``vae``'s encoder
    (compiled when ``vae.compile(..., compile_encoder=True)`` ran; eager on CPU otherwise), on THIS
    process only, or (``group``, a gloo process group) dealt round-robin across ``group``'s ranks.
    Same fixed-shape grid, per-tile encode and merge as the shared
    ``NeuronAutoencoderKLWan.tiled_encode``, whose tile-parallel path gathers to GLOBAL rank 0 of
    ``vae.tile_parallel_group``; here the gather goes to ``group``'s own first rank, so each CFG
    replica's TP group encodes independently. Returns the pre-split moments ``h`` on the host
    (``None`` on a non-root rank of ``group``)."""
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, merge_tiles

    height, width = x.shape[-2:]
    stride = tile_px - overlap_px
    grid = TileGrid.for_axes(
        total=(height, width),
        tile=(tile_px, tile_px),
        stride=(stride, stride),
        in_scale=vae.spatial_compression_ratio,
    )
    xp = grid.pad_input(x.detach().cpu())
    world, rank = (1, 0) if group is None else (dist.get_world_size(group), dist.get_rank(group))
    tiles = {
        idx: vae._tile_encode_one(grid.slice_input(xp, idx)).detach().cpu()
        for n, idx in enumerate(grid.indices())
        if n % world == rank
    }
    if world > 1:
        parts: list | None = [None] * world if rank == 0 else None
        dist.gather_object(tiles, parts, dst=dist.get_global_rank(group, 0), group=group)
        if rank != 0:
            return None
        tiles = {k: v for part in parts for k, v in part.items()}
    return merge_tiles(tiles, grid)


@contextlib.contextmanager
def _neuron_components():
    """Swap upstream's transformer / VAE classes (and its device) while its __init__ runs."""
    saved = (
        _up.resolve_cosmos3_transformer_cls,
        _up.DistributedAutoencoderKLWan,
        _up.get_local_device,
    )
    _up.resolve_cosmos3_transformer_cls = lambda _cfg: NeuronCosmos3EdgeTransformer
    _up.DistributedAutoencoderKLWan = NeuronEdgeVae
    _up.get_local_device = lambda: torch.device("cpu")  # host-side pipeline math stays on the CPU
    try:
        yield
    finally:
        (
            _up.resolve_cosmos3_transformer_cls,
            _up.DistributedAutoencoderKLWan,
            _up.get_local_device,
        ) = saved


class NeuronCosmos3EdgePipeline(Cosmos3OmniDiffusersPipeline):
    """Neuron Cosmos3 pipeline for every Cosmos3 checkpoint: Edge (Nemotron backbone) and
    Nano / Super / Super 4-step (Qwen3-VL backbone). The backbone is read from the
    transformer config; the towers and the host-side pipeline math are shared."""

    def __init__(self, *, od_config, prefix: str = ""):
        cfg = EdgeGenConfig.from_model_dir(od_config.model)
        # Sound generation is not ported: keep upstream from building the sound tokenizer and
        # make sound requests fail with upstream's own "transformer has no sound modules" error.
        args = getattr(od_config, "custom_pipeline_args", None)
        if isinstance(args, dict):
            args.setdefault("sound_gen", False)
        else:
            od_config.custom_pipeline_args = {"sound_gen": False}
        with _neuron_components():
            super().__init__(od_config=od_config, prefix=prefix)
        self.is_edge_model = cfg.backbone == "nemotron"
        if od_config.flow_shift is None:
            self._engine_init_flow_shift = (
                COSMOS3_EDGE_VIDEO_DEFAULT_FLOW_SHIFT
                if self.is_edge_model
                else _up.COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT
            )
            self._current_flow_shift = self._engine_init_flow_shift
        # weights come from our own sharded loaders, not the engine's weight iterator
        self.weights_sources = []
        self._compile_encoder = bool(
            (od_config.model_config or {}).get("compile_vae_encoder", True)
        )
        self._maybe_patch_parallel_vae(od_config)

    def _maybe_patch_parallel_vae(self, od_config) -> None:
        """``vae_patch_parallel_size > 1``: decode spatial tiles on every rank (the plugin's
        ``DistributedAutoencoderKLWan``), merged on rank 0. Under CFG-parallel both ranks reach the
        decode in lockstep, so the second core's otherwise redundant decode becomes half the work.

        Tile geometry ``COSMOS3_EDGE_VAE_TILE=h_min,w_min,h_stride,w_stride`` (pixels). The default
        480,480,480,416 cuts an 832x480 frame into exactly two equal 480x480 tiles with a 64 px blend
        (one compiled tile shape; a sliver tile would add graphs and work).
        """
        import torch.distributed as dist

        size = int(getattr(od_config.parallel_config, "vae_patch_parallel_size", 1) or 1)
        if size <= 1 or not dist.is_initialized():
            return
        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            DistributedAutoencoderKLWan,
        )

        world = dist.get_world_size()
        if size < world:
            # Every rank must build the same VAE: the conditioning frame is encoded per TP group, and
            # a rank left on a different VAE object encoded different latents for its CP slice.
            raise ValueError(
                f"vae_patch_parallel_size={size} must equal the stage's world size {world} "
                "(or be 1); smaller VAE groups are not supported"
            )
        size = min(size, world)
        group = dist.new_group(ranks=list(range(size)))  # collective: every rank calls it
        host_group = (
            dist.new_group(ranks=list(range(size)), backend="gloo")
            if _host_gather_t() > 0
            else None
        )
        vae = DistributedAutoencoderKLWan.from_pretrained(
            od_config.model, subfolder="vae", torch_dtype=self.vae.dtype
        ).eval()
        vae.init_distributed(group=group)
        vae.set_parallel_size(size)
        vae.use_tiling = True
        # host_group: opt-in host-gather fallback for long clips (COSMOS3_EDGE_VAE_HOST_GATHER_T > 0);
        # the env value is identical on every rank, so the collective new_group stays matched.
        hmin, wmin, hs, ws = (
            int(x) for x in os.environ.get("COSMOS3_EDGE_VAE_TILE", "480,480,480,416").split(",")
        )
        vae.tile_sample_min_height, vae.tile_sample_min_width = hmin, wmin
        vae.tile_sample_stride_height, vae.tile_sample_stride_width = hs, ws
        self.vae = NeuronEdgeVae(vae)
        self.vae._host_pp_group = host_group
        logger.info(
            "cosmos3_edge: VAE patch-parallel over %d ranks, tiles %s", size, (hmin, wmin, hs, ws)
        )

    def load_weights(self, weights=None):
        self.transformer.load()
        return None

    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self.transformer.compile(backend, options, **kwargs)
        self.vae.compile(backend, options, compile_encoder=self._compile_encoder)
        return self

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
        """CFG-parallel across replicas, branch outputs exchanged ON DEVICE.

        Each rank runs exactly one branch (CFG group rank 0 positive, 1 negative). The transformer
        facade all-gathers each GEN output over the CFG group's device group before copying it to the
        host (:meth:`NeuronCosmos3EdgeTransformer._host_out`; in CFG group-rank order, which on the
        Trn2 physical mesh is NOT the sorted order of the c10d groups for half the CFG groups). If a
        call produced no device output to gather (a step-cache skip, or a transformer without the CFG
        gather), the parts are gathered on the host with ``host_all_gather`` (also group-rank order).
        Every rank then applies the same combine, so the result matches sequential CFG. Tuple
        (action) outputs are gathered element by element.
        """
        if not do_true_cfg or not self._cfg_parallel_active():
            return super().predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
                output_slice,
                kwargs,
            )
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            get_classifier_free_guidance_rank,
        )

        from vllm_omni_neuron.diffusion.distributed.parallel_state import host_all_gather

        group = get_cfg_group()
        rank = get_classifier_free_guidance_rank()
        t_step = time.time()
        tf = self.transformer
        armed = getattr(tf, "_cfg_gather_fn", None) is not None
        if armed:
            tf._cfg_parts = []
        try:
            local = self.predict_noise(**(positive_kwargs if rank == 0 else negative_kwargs))
            dev_parts = tf._cfg_parts if armed else None
        finally:
            if armed:
                tf._cfg_parts = None
        t_g = time.time()
        local = local if isinstance(local, tuple) else (local,)
        if dev_parts is None or len(dev_parts) != len(local):
            dev_parts = [host_all_gather(group, p.detach().to("cpu")) for p in local]
        if output_slice is not None:
            dev_parts = [[q[:, :output_slice] for q in parts] for parts in dev_parts]
        pos = [parts[0] for parts in dev_parts]
        neg = [parts[1] for parts in dev_parts]
        _prof("cfg_gather", t_g)
        res = self.combine_cfg_noise(
            tuple(pos),
            tuple(neg),
            true_cfg_scale,
            cfg_normalize,
            **({} if kwargs is None else {"kwargs": kwargs}),
        )
        _prof("cfg_step_total", t_step)
        return res


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronCosmos3EdgePipeline",
    "get_cosmos3_post_process_func",
    "get_cosmos3_pre_process_func",
]
