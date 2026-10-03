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
    out = _randn_tensor_up(shape, generator=generator, device=device, dtype=torch.float32, layout=layout)
    return out.to(dtype) if dtype is not None else out


_up.randn_tensor = _randn_fp32

TEXT_BUCKETS = tuple(int(b) for b in os.environ.get("COSMOS3_EDGE_TEXT_BUCKETS", "64,128,256,512,1024").split(","))

TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
VAE_COMPILER_ARGS = ["--model-type=unet-inference", "--auto-cast=none", "--internal-max-instruction-limit=15000000", "-O1"]

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
    raise ValueError(f"prompt is {n} tokens; the largest text bucket is {TEXT_BUCKETS[-1]} (COSMOS3_EDGE_TEXT_BUCKETS)")


def _host(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    x = x.detach().to("cpu")
    return (x.to(dtype) if dtype is not None else x).contiguous()


class NeuronCosmos3EdgeTransformer(nn.Module):
    """Drop-in for upstream ``Cosmos3EdgeVFMTransformer`` inside the Cosmos3 pipeline."""

    def __init__(self, od_config, temporal_compression_factor=None, sound_gen=False, sound_dim=None,
                 sound_latent_fps=None):
        super().__init__()
        if sound_gen:
            raise NotImplementedError("Cosmos3-Edge on Neuron: sound generation is not supported")
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
        self.cached_kv: Any = None  # (kv tuple on device, padded text mask) of the current CFG branch
        self.cached_freqs_gen: Any = None  # (cos, sin, key_bias) on device, per geometry + branch
        self._device = torch.device("cpu")
        self._und_fn = self.und
        self._gen_fn = self.gen
        self._gen_action_fn = self.gen.forward_action
        self.stats = {"und_calls": 0, "gen_calls": 0, "und_s": 0.0, "gen_s": 0.0, "gen_skipped": 0}
        self._step_cache: dict = {}  # per-branch state of the training-free step cache

    # -- lifecycle ----------------------------------------------------------------------
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
        return (thr, int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_WARMUP", "2")),
                int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_MAX_SKIP", "2")),
                float(os.environ.get("COSMOS3_EDGE_STEP_CACHE_TMIN", "0")),
                int(os.environ.get("COSMOS3_EDGE_STEP_CACHE_ORDER", "1")),
                float(os.environ.get("COSMOS3_EDGE_STEP_CACHE_TMAX", "1e9")))

    def _sc_try_skip(self, x_host: torch.Tensor, t_val: float):
        """Return (skip_output_or_None, state). ``x_host`` is the GEN input on the host."""
        thr, warmup, max_skip, tmin, order, tmax = self._sc_cfg()
        if thr <= 0 or self.cached_kv is None:
            return None, None
        st = self._step_cache.setdefault(id(self.cached_kv[0]), {"n": 0, "skips": 0, "hist": [], "x_ref": None})
        st["n"] += 1
        if st["x_ref"] is None or st["n"] <= warmup or st["skips"] >= max_skip or not (tmin <= t_val <= tmax):
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

        self._und_fn = torch.compile(self.und, backend=backend, options=opts("cosmos3_edge_und"), **kw)
        self._gen_fn = torch.compile(self.gen, backend=backend, options=opts("cosmos3_edge_gen"), **kw)
        self._gen_action_fn = torch.compile(self.gen.forward_action, backend=backend,
                                            options=opts("cosmos3_edge_gen_action"), **kw)

    # -- forward (upstream signature) -------------------------------------------------
    def forward(self, hidden_states, timestep, text_ids, text_mask, video_shape, fps=None,
                action_latents=None, action_domain_ids=None, action_noisy_mask=None,
                action_start_frame_offset: int = 1, action_fps=None, sound_latents=None,
                noisy_frame_mask=None, control_latents=None, control_weights=None,
                transfer_share_vision_temporal_positions=True, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected Cosmos3 transformer kwargs: {sorted(kwargs)}")
        if sound_latents is not None:
            raise NotImplementedError("Cosmos3-Edge on Neuron: sound generation is not supported")
        if control_latents is not None:
            raise NotImplementedError("Cosmos3-Edge on Neuron: transfer/control conditioning is not supported")
        dev, dt = self._device, self.dtype
        t, h, w = (int(x) for x in video_shape)
        if h % self.cfg.patch or w % self.cfg.patch:
            raise ValueError(f"latent size {h}x{w} must be a multiple of the patch size {self.cfg.patch}")
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
                kv = self._und_fn(ids.to(dev), cu.to(dt).contiguous().to(dev), su.to(dt).contiguous().to(dev))
            self.cached_kv = (tuple(kv), mask)
            self.cached_freqs_gen = None
            self.stats["und_calls"] += 1
            self.stats["und_s"] += time.time() - t0
            _prof("und", t0)
        kv, mask = self.cached_kv
        geom = (t, h, w, s_action, None if fps is None else float(fps), action_start_frame_offset,
                None if action_fps is None else float(action_fps))
        if self.cached_freqs_gen is None or self.cached_freqs_gen[0] != geom:
            cg, sg = self.gen.rope_tables(mask, t, h, w, fps, t_action=s_action,
                                          action_start_frame_offset=action_start_frame_offset, action_fps=action_fps)
            kb = self.gen.key_bias(mask, s_video + s_action)
            self.cached_freqs_gen = (geom, cg.to(dt).contiguous().to(dev), sg.to(dt).contiguous().to(dev),
                                     kb.contiguous().to(dev))
        _, cg, sg, kb = self.cached_freqs_gen

        hw = (h // self.cfg.patch) * (w // self.cfg.patch)
        if noisy_frame_mask is not None:
            nm = _host(noisy_frame_mask).float()[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, hw).reshape(b, -1, 1)
        else:
            nm = torch.ones(b, s_video, 1)
        x = _host(hidden_states, dt).to(dev)
        ts = _host(timestep).float().reshape(-1)
        if ts.numel() == 1 and b > 1:
            ts = ts.expand(b)
        ts = ts.contiguous().to(dev)
        nm = nm.to(dt).contiguous().to(dev)
        _prof("gen_host_in", t_host0)
        t0 = time.time()
        with torch.no_grad():
            if s_action == 0:
                x_cpu = _host(hidden_states, dt)
                t_val = float(_host(timestep).float().reshape(-1)[0].item())
                skipped, sc_state = self._sc_try_skip(x_cpu, t_val)
                if skipped is not None:
                    _prof("gen_skip", t0)
                    return skipped
                out = self._gen_fn(x, ts, cg, sg, kb, nm, *kv).to("cpu")
                self._sc_record(sc_state, x_cpu, t_val, out)
            else:
                domain = 0 if action_domain_ids is None else int(_host(action_domain_ids).reshape(-1)[0].item())
                w_in, b_in, w_out, b_out = self.gen.domain_weights(domain, dev)
                am = (torch.ones(b, s_action, 1) if action_noisy_mask is None
                      else _host(action_noisy_mask).float().reshape(b, s_action, 1))
                video, action = self._gen_action_fn(
                    x, ts, cg, sg, kb, nm, _host(action_latents, dt).to(dev), am.to(dt).contiguous().to(dev),
                    w_in, b_in, w_out, b_out, *kv)
                out = (video.to("cpu"), action.to("cpu"))
        self.stats["gen_calls"] += 1
        self.stats["gen_s"] += time.time() - t0
        _prof("gen_call", t0)
        return out


class NeuronEdgeVae(nn.Module):
    """Host-tensor facade over the plugin's compiled Wan VAE (``NeuronAutoencoderKLWan``)."""

    @classmethod
    def from_pretrained(cls, model_path, subfolder="vae", torch_dtype=torch.bfloat16, **kwargs):
        from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            NeuronAutoencoderKLWan,
        )

        vae = NeuronAutoencoderKLWan.from_pretrained(model_path, subfolder=subfolder, torch_dtype=torch_dtype, **kwargs)
        return cls(vae.eval())

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae.config
        self._device = torch.device("cpu")
        self._compiled_encoder = False

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and torch.device(device).type != "cpu":
            self._device = torch.device(device)
            self.vae.to(self._device)
        return self

    def compile(self, backend: str, options: dict | None = None, compile_encoder: bool = False, **kwargs) -> None:
        opts = {**(options or {}), "model_name": "cosmos3_edge_vae", "compiler_args": list(VAE_COMPILER_ARGS)}
        self.vae.compile(backend=backend, options=opts, fullgraph=True, dynamic=False, compile_encoder=compile_encoder)
        self._compiled_encoder = compile_encoder

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

        if os.environ.get("COSMOS3_EDGE_DECODE_ALL_RANKS", "0") != "1" and self._skip_redundant_decode():
            b, _, t, h, w = z.shape
            r = self.vae.spatial_compression_ratio
            tr = int(getattr(self.vae.config, "scale_factor_temporal", 4) or 4)
            out = torch.zeros(b, 3, (t - 1) * tr + 1, h * r, w * r, dtype=self.dtype)
            return DecoderOutput(sample=out) if return_dict else (out,)
        self._ensure_decoder()
        t0 = time.time()
        out = self.vae.decode(_host(z, self.dtype).to(self._device), return_dict=False)[0].to("cpu")
        _prof("vae_decode", t0)
        return DecoderOutput(sample=out) if return_dict else (out,)

    def encode(self, x, return_dict=True):
        """Encoder graph on the NeuronCore (incl. patchify); the [B, 2*z, t, h, w] moments come
        back to the host, where the pipeline's latent math runs, before the Gaussian split."""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        t0 = time.time()
        tiling, self.vae.use_tiling = self.vae.use_tiling, False  # encode is one frame, ~0.3 s: never tile it
        try:
            with torch.no_grad():
                h = self.vae._encode(_host(x, self.dtype).to(self._device)).to("cpu")
        finally:
            self.vae.use_tiling = tiling
        _prof("vae_encode", t0)
        dist = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)


@contextlib.contextmanager
def _neuron_components():
    """Swap upstream's transformer / VAE classes (and its device) while its __init__ runs."""
    saved = (_up.resolve_cosmos3_transformer_cls, _up.DistributedAutoencoderKLWan, _up.get_local_device)
    _up.resolve_cosmos3_transformer_cls = lambda _cfg: NeuronCosmos3EdgeTransformer
    _up.DistributedAutoencoderKLWan = NeuronEdgeVae
    _up.get_local_device = lambda: torch.device("cpu")  # host-side pipeline math stays on the CPU
    try:
        yield
    finally:
        _up.resolve_cosmos3_transformer_cls, _up.DistributedAutoencoderKLWan, _up.get_local_device = saved


class NeuronCosmos3EdgePipeline(Cosmos3OmniDiffusersPipeline):
    def __init__(self, *, od_config, prefix: str = ""):
        with _neuron_components():
            super().__init__(od_config=od_config, prefix=prefix)
        self.is_edge_model = True
        if od_config.flow_shift is None:
            self._engine_init_flow_shift = COSMOS3_EDGE_VIDEO_DEFAULT_FLOW_SHIFT
            self._current_flow_shift = self._engine_init_flow_shift
        # weights come from our own sharded loaders, not the engine's weight iterator
        self.weights_sources = []
        self._compile_encoder = bool((od_config.model_config or {}).get("compile_vae_encoder", True))
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

        size = min(size, dist.get_world_size())
        group = dist.new_group(ranks=list(range(size)))
        vae = DistributedAutoencoderKLWan.from_pretrained(od_config.model, subfolder="vae",
                                                          torch_dtype=self.vae.dtype).eval()
        vae.init_distributed(group=group)
        vae.set_parallel_size(size)
        vae.use_tiling = True
        hmin, wmin, hs, ws = (int(x) for x in os.environ.get("COSMOS3_EDGE_VAE_TILE", "480,480,480,416").split(","))
        vae.tile_sample_min_height, vae.tile_sample_min_width = hmin, wmin
        vae.tile_sample_stride_height, vae.tile_sample_stride_width = hs, ws
        self.vae = NeuronEdgeVae(vae)
        logger.info("cosmos3_edge: VAE patch-parallel over %d ranks, tiles %s", size, (hmin, wmin, hs, ws))

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

    def predict_noise_maybe_with_cfg(self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs,
                                     cfg_normalize=True, output_slice=None, kwargs=None):
        """CFG-parallel across replicas with a HOST all-gather.

        Our transformer facade returns host tensors (the pipeline runs on CPU and only the towers
        are on the NeuronCore), so the branch outputs are exchanged over the CFG group's gloo
        ``cpu_group``. The latent is a few MB, negligible against a multi-second branch forward.
        Each rank runs exactly one branch (rank 0 positive, rank 1 negative); every rank then
        applies the same combine, so the result matches sequential CFG. Tuple (action/sound)
        outputs are gathered element by element.
        """
        if not do_true_cfg or not self._cfg_parallel_active():
            return super().predict_noise_maybe_with_cfg(do_true_cfg, true_cfg_scale, positive_kwargs,
                                                        negative_kwargs, cfg_normalize, output_slice, kwargs)
        import torch.distributed as dist
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            get_classifier_free_guidance_rank,
        )

        group = get_cfg_group()
        rank = get_classifier_free_guidance_rank()
        t_step = time.time()
        local = self.predict_noise(**(positive_kwargs if rank == 0 else negative_kwargs))
        t_g = time.time()
        local = local if isinstance(local, tuple) else (local,)
        if output_slice is not None:
            local = tuple(p[:, :output_slice] for p in local)
        pos, neg = [], []
        for p in local:
            p = p.detach().to("cpu").contiguous()
            bufs = [torch.empty_like(p) for _ in range(group.world_size)]
            dist.all_gather(bufs, p, group=group.cpu_group)
            pos.append(bufs[0])
            neg.append(bufs[1])
        _prof("cfg_gather", t_g)
        res = self.combine_cfg_noise(tuple(pos), tuple(neg), true_cfg_scale, cfg_normalize,
                                     **({} if kwargs is None else {"kwargs": kwargs}))
        _prof("cfg_step_total", t_step)
        return res


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronCosmos3EdgePipeline",
    "get_cosmos3_post_process_func",
    "get_cosmos3_pre_process_func",
]
