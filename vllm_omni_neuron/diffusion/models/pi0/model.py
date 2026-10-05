# SPDX-License-Identifier: Apache-2.0
"""Neuron pi0.5 / pi0.52 action model: the vendored upstream model as parameter owner, fixed-shape
compiled graphs (:mod:`.graphs`) for the hot path, and the flow-matching schedule.

Placement: everything the graphs read (SigLIP, projector, PaliGemma LM, action expert, action
projections, and the PaliGemma ``lm_head`` that pi0.52's subtask decode reads) moves to the
device; the action expert's unused ``lm_head`` and the timestep MLP stay on the host. The AdaRMS
conditioning of each Euler step is computed on the host in fp32 from the float64 sinusoid
exactly as upstream, then fed to the denoise graph, so the device never sees float64.

The Euler loop itself runs on the device (``denoise_mode``): by default one graph per Euler step
whose state stays on the device (:class:`.graphs.EulerLoopGraph` with one step), fed the AdaRMS
modulations of every norm precomputed once per schedule (``adarms_tables``), so the action state
never round-trips through the host between steps and no step recomputes ``dense(cond)``.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn

from ._vendor.pi05.config import Pi05Config
from ._vendor.pi05.modeling_pi05 import Pi05ForActionPrediction
from .graphs import (
    EulerLoopGraph,
    Pi05DenoiseGraph,
    Pi05EmbedImagesGraph,
    Pi05PrefixFromEmbeddingsGraph,
    Pi05PrefixGraph,
    Pi05SubtaskDecodeGraph,
    Pi05SubtaskPrefillGraph,
    Pi05TextPrefixGraph,
    Pi05TextPrefixGraphCached,
    shard_lm_tp,
)
from .subtask import Pi052SubtaskGenerator, _camera_stack

logger = logging.getLogger(__name__)

# Sub-modules kept in fp32 regardless of the compute dtype (LeRobot's bf16 layout keeps the
# norms fp32; they are tiny). The vision tower follows PI05_VISION_DTYPE.
_FP32_SELECTORS = ("layernorm", "layer_norm", "model.norm", ".norm.", "post_layernorm")

PROFILE = os.environ.get("PI05_PROFILE", "0") == "1"

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

# How the flow-matching loop runs: "device" (default; one graph per step, the state stays on the
# device), "unrolled" (every step in one graph) or "host" (one graph per step, Euler update on the
# host). Measured on Trn2 (pi0.52, 10 steps, AdaRMS tables): device 31.3 ms, host 37.4 ms,
# unrolled 80.6 ms -- the unrolled graph compiles to a slower schedule than ten launches.
DENOISE_MODES = ("unrolled", "device", "host")


def _no_force(step, logits):
    """``step_hook`` that forces nothing: selects the host-side greedy decode loop."""
    return None


class DenoiseLoop:
    """The flow-matching schedule over a one-step velocity graph, in one of :data:`DENOISE_MODES`.

    ``step`` is the eager one-step module; ``x_pos`` is where ``x_t`` sits in its arguments (the
    step's condition follows it). Compiled loop graphs are created per schedule length on first
    use, with the backend/options recorded by :meth:`set_compile` (eager when never compiled)."""

    def __init__(self, step: nn.Module, x_pos: int, name: str, mode: str = "device"):
        if mode not in DENOISE_MODES:
            raise ValueError(f"denoise_mode must be one of {DENOISE_MODES}, got {mode!r}")
        self.step, self.x_pos, self.name, self.mode = step, x_pos, name, mode
        self.step_fn = step
        self._compile = None  # (backend, options, kwargs)
        self._loops: dict[tuple[str, int], object] = {}
        self._dev_conds: dict[tuple, object] = {}

    def set_compile(self, backend, options, kwargs, step_fn) -> None:
        self._compile = (backend, dict(options or {}), dict(kwargs))
        self.step_fn = step_fn
        self._loops.clear()

    def _loop(self, unrolled: bool, num_steps: int):
        key = ("unrolled" if unrolled else "device", num_steps)
        fn = self._loops.get(key)
        if fn is None:
            g = EulerLoopGraph(
                self.step, num_steps if unrolled else 1, -1.0 / num_steps, self.x_pos
            )
            fn = g
            if self._compile is not None:
                backend, options, kw = self._compile
                tag = (
                    f"{self.name}_loop{num_steps}" if unrolled else f"{self.name}_euler{num_steps}"
                )
                options = {**options, "model_name": tag, "compiler_args": list(COMPILER_ARGS)}
                fn = torch.compile(g, backend=backend, options=options, **kw)
            self._loops[key] = fn
        return fn

    def _conds_on(self, conds: torch.Tensor, device, per_step: bool):
        """Device copies of the (cached, per schedule) step conditions: one ``[n, B, C]`` tensor,
        or ``n`` separate ``[1, B, C]`` base tensors (no slicing a device tensor eagerly)."""
        key = (conds.data_ptr(), tuple(conds.shape), str(device), per_step)
        hit = self._dev_conds.get(key)
        if hit is None:
            if per_step:
                hit = [conds[s : s + 1].contiguous().to(device) for s in range(conds.shape[0])]
            else:
                hit = conds.contiguous().to(device)
            if len(self._dev_conds) >= 16:
                self._dev_conds.clear()
            self._dev_conds[key] = hit
        return hit

    def run(self, x0: torch.Tensor, conds: torch.Tensor, ctx: tuple, device) -> torch.Tensor:
        """``x0`` ``[B, H, A]`` fp32 host noise, ``conds`` ``[n, B, C]`` fp32 host (cached by the
        caller per schedule), ``ctx`` the step's device inputs other than x / cond. Returns the
        final ``x`` fp32 on the host."""
        n = int(conds.shape[0])
        if self.mode == "host":
            dt = -1.0 / n
            x = x0
            for s in range(n):
                args = list(ctx)
                args.insert(self.x_pos, x.to(device))
                args.insert(self.x_pos + 1, conds[s].contiguous().to(device))
                x = x + dt * self.step_fn(*args).to("cpu").float()
            return x
        if self.mode == "unrolled":
            return self._loop(True, n)(
                x0.to(device), self._conds_on(conds, device, False), *ctx
            ).to("cpu")
        loop = self._loop(False, n)
        x = x0.to(device)
        for c in self._conds_on(conds, device, True):
            x = loop(x, c, *ctx)
        return x.to("cpu")


def _dtype_from_env(name: str, default: torch.dtype) -> torch.dtype:
    v = os.environ.get(name, "").strip().lower()
    return {
        "": default,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[v]


class NeuronPi05ActionModel(nn.Module):
    """``sample_actions`` drop-in for upstream ``Pi05ForActionPrediction`` on Neuron."""

    def __init__(
        self,
        config: Pi05Config,
        dtype: torch.dtype = torch.bfloat16,
        denoise_mode: str = "device",
        adarms_tables: bool = True,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.vision_dtype = _dtype_from_env("PI05_VISION_DTYPE", dtype)
        self.ref = Pi05ForActionPrediction(config)
        self.num_cameras = int(config.max_cameras)
        self.prefix = Pi05PrefixGraph(self.ref, self.num_cameras)
        self.denoise = Pi05DenoiseGraph(self.ref)
        # Precomputed AdaRMS modulation per (schedule step, norm): no dense(cond) per step.
        self.denoise.modulation_tables = bool(adarms_tables)
        self._prefix_fn = self.prefix
        self._denoise_fn = self.denoise
        self.denoise_loop = DenoiseLoop(self.denoise, 0, "pi05_denoise", denoise_mode)
        # pi0.52: the action prefix over the subtask decode's image embedding (no 2nd SigLIP).
        self.prefix_from_emb: Pi05PrefixFromEmbeddingsGraph | None = None
        self._prefix_from_emb_fn = None
        self._device = torch.device("cpu")
        self._time_cond_cache: dict[int, torch.Tensor] = {}
        self.stats = {"prefix_s": 0.0, "denoise_s": 0.0, "calls": 0}
        # pi0.52: a text-generation graph over the same prefix (shares its parameters), used to
        # produce the low-level subtask before each action chunk. Only built/compiled when asked
        # for (set_tokenizer), since pi0.5 proper never needs it.
        self.text_prefix: Pi05TextPrefixGraph | None = None
        self._text_prefix_fn = None
        self.embed_images_graph: Pi05EmbedImagesGraph | None = None
        self.text_prefix_cached: Pi05TextPrefixGraphCached | None = None
        self._embed_images_fn = None
        self._text_prefix_cached_fn = None
        self.subtask_gen: Pi052SubtaskGenerator | None = None
        # KV-cached subtask decode (prefill once + shared decode-attention steps).
        self.subtask_prefill: Pi05SubtaskPrefillGraph | None = None
        self.subtask_decode: Pi05SubtaskDecodeGraph | None = None
        self._subtask_prefill_fn = None
        self._subtask_decode_fn = None
        self._subtask_cache_cfg = None
        self.subtask_use_kv = True  # default path when the KV-cached decode is enabled
        self.subtask_sync_every = 1  # EOS check cadence of the device-resident greedy decode
        self.subtask_host_greedy = False  # True: read logits back and pick on the host (old path)

    # -- lifecycle -----------------------------------------------------------------------
    def load_checkpoint(self, model_dir: str) -> None:
        import safetensors.torch

        t0 = time.time()
        path = os.path.join(model_dir, "model.safetensors")
        state = safetensors.torch.load_file(path)
        self.ref.load_weights(state.items())
        del state
        self._apply_dtypes()
        logger.info("pi0.5 weights loaded from %s in %.1fs", path, time.time() - t0)

    def _apply_dtypes(self) -> None:
        pwe = self.ref.paligemma_with_expert
        for name, p in pwe.named_parameters():
            if any(s in name for s in _FP32_SELECTORS):
                dt = torch.float32
            elif ".vision_tower." in f".{name}" or "multi_modal_projector" in name:
                dt = self.vision_dtype
            else:
                dt = self.dtype
            p.data = p.data.to(dt)
        for name, p in self.ref.named_parameters():  # action / time projections: fp32 (tiny)
            if not name.startswith("paligemma_with_expert."):
                p.data = p.data.float()
        self.prefix.mm_dtype = self.dtype
        self.denoise.mm_dtype = self.dtype
        self.denoise.snapshot_modulation_weights()
        self._time_cond_cache.clear()

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is None:
            return self
        self._device = torch.device(device)
        pwe = self.ref.paligemma_with_expert
        pg = pwe.paligemma.model
        for m in (
            pg.vision_tower,
            pg.multi_modal_projector,
            pg.language_model,
            pwe.gemma_expert.model,
            self.ref.action_in_proj,
            self.ref.action_out_proj,
            pwe.paligemma.lm_head,
        ):
            m.to(self._device)
        self.prefix.inv_freq = self.prefix.inv_freq.to(self._device)
        self.denoise.inv_freq = self.denoise.inv_freq.to(self._device)
        if self.prefix.tp_slot is not None:
            self.prefix.tp_slot = self.prefix.tp_slot.to(self._device)
        if self.subtask_decode is not None:
            self.subtask_decode.move_weights(self._device)
        return self

    def shard_tp(self, rank: int, size: int, group) -> None:
        """Tensor-parallel PaliGemma LM (:func:`.graphs.shard_lm_tp`): the action prefix and the
        subtask prefill/decode split over ``size`` ranks; vision tower and action expert stay
        replicated. Call after ``load_checkpoint`` and before ``enable_subtask_generation``."""
        if self.subtask_decode is not None:
            raise RuntimeError("shard_tp() must run before enable_subtask_generation()")
        shard_lm_tp(
            self.prefix, rank, size, group, self.ref.paligemma_with_expert.paligemma.lm_head
        )

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(COMPILER_ARGS)}

        self._prefix_fn = torch.compile(
            self.prefix, backend=backend, options=opts("pi05_prefix"), **kw
        )
        self._denoise_fn = torch.compile(
            self.denoise, backend=backend, options=opts("pi05_denoise"), **kw
        )
        self.denoise_loop.set_compile(backend, base, kw, self._denoise_fn)
        if self.prefix_from_emb is not None:
            self._prefix_from_emb_fn = torch.compile(
                self.prefix_from_emb, backend=backend, options=opts("pi05_prefix_from_emb"), **kw
            )
        if self.text_prefix is not None:
            self._text_prefix_fn = torch.compile(
                self.text_prefix, backend=backend, options=opts("pi05_text_prefix"), **kw
            )
        if self.embed_images_graph is not None:
            self._embed_images_fn = torch.compile(
                self.embed_images_graph, backend=backend, options=opts("pi05_embed_images"), **kw
            )
            self._text_prefix_cached_fn = torch.compile(
                self.text_prefix_cached,
                backend=backend,
                options=opts("pi05_text_prefix_cached"),
                **kw,
            )
            if self.subtask_gen is not None:
                self.subtask_gen.embed_images_graph = self._embed_images_fn
                self.subtask_gen.text_graph_cached = self._text_prefix_cached_fn
        if self.subtask_decode is not None:
            self._subtask_prefill_fn = torch.compile(
                self.subtask_prefill, backend=backend, options=opts("pi05_subtask_prefill"), **kw
            )
            self._subtask_decode_fn = torch.compile(
                self.subtask_decode, backend=backend, options=opts("pi05_subtask_decode"), **kw
            )

    # -- pi0.52 subtask generation --------------------------------------------------------
    def enable_subtask_generation(
        self,
        tokenizer,
        buckets: tuple[int, ...] = (64, 96, 128, 192, 256),
        cache_image_prefix: bool = True,
        kv_cache: bool = True,
    ) -> None:
        """Build the text-generation graphs + :class:`Pi052SubtaskGenerator`. Call after
        ``load_checkpoint`` and before ``compile``/``to`` so the new graphs are covered by both.

        ``cache_image_prefix`` (default): also build the embed-images + cached text graphs so the
        subtask decode runs the vision tower once per request instead of once per generated token.

        ``kv_cache`` (default, needs ``cache_image_prefix``): decode with ONE prefill over the image
        + prompt prefix and one KV-cached step per generated token (the shared decode-attention
        layer's ``StaticKVCache`` / ``decode_step``), instead of re-running the LM over the whole
        prefix for every token. Matches upstream's ``use_kv_cache=True`` (generated tokens attend
        causally). ``kv_cache=False`` keeps the re-prefill path."""
        lm_head = self.ref.paligemma_with_expert.paligemma.lm_head
        self.text_prefix = Pi05TextPrefixGraph(self.prefix, lm_head)
        self._text_prefix_fn = self.text_prefix
        if cache_image_prefix:
            self.embed_images_graph = Pi05EmbedImagesGraph(self.prefix)
            self.text_prefix_cached = Pi05TextPrefixGraphCached(self.prefix, lm_head)
            self._embed_images_fn = self.embed_images_graph
            self._text_prefix_cached_fn = self.text_prefix_cached
        self.subtask_gen = Pi052SubtaskGenerator(
            self._text_prefix_fn,
            tokenizer,
            self.num_cameras,
            self.config.image_resolution[0],
            fast_skip_tokens=int(getattr(self.config, "fast_skip_tokens", 1152)),
            buckets=buckets,
            embed_images_graph=self._embed_images_fn,
            text_graph_cached=self._text_prefix_cached_fn,
        )
        if kv_cache and cache_image_prefix:
            from vllm_omni_neuron.diffusion.attention.decode_attention import (
                DecodeAttentionConfig,
            )

            p = self.prefix
            n_img = (self.config.image_resolution[0] // p.patch) ** 2
            need = self.num_cameras * n_img + self.subtask_gen.buckets[-1]
            self._subtask_cache_cfg = DecodeAttentionConfig(
                q_heads=p.n_heads,
                kv_heads=p.n_kv,
                head_dim=p.head_dim,
                max_len=-(-need // 128) * 128,
                dtype=self.dtype,
            )
            self.subtask_prefill = Pi05SubtaskPrefillGraph(p, lm_head, self.dtype)
            self.subtask_decode = Pi05SubtaskDecodeGraph(p, lm_head, self._subtask_cache_cfg)
            self._subtask_prefill_fn = self.subtask_prefill
            self._subtask_decode_fn = self.subtask_decode
        if cache_image_prefix:
            self.prefix_from_emb = Pi05PrefixFromEmbeddingsGraph(self.prefix)
            self._prefix_from_emb_fn = self.prefix_from_emb

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor | None:
        """SigLIP + projector once for an observation: ``[ncam, N, W]`` fp32 on the device, for
        :meth:`generate_subtask` and :meth:`sample_actions` (``img_emb=``). None when the image
        embedding graph was not built (pi0.5 without subtask generation)."""
        if self._embed_images_fn is None:
            return None
        t0 = time.time()
        out = self._embed_images_fn(_camera_stack(images, self.vision_dtype, self._device))
        if PROFILE:
            out.to("cpu")
        self.stats["embed_images_s"] = self.stats.get("embed_images_s", 0.0) + time.time() - t0
        return out

    @torch.no_grad()
    def generate_subtask(
        self,
        images,
        image_masks,
        task: str,
        max_new_tokens: int = 128,
        *,
        kv_cache: bool | None = None,
        **kwargs,
    ):
        """Greedy subtask decode. ``kv_cache`` None = KV-cached when it was enabled, else the
        re-prefill path; ``kwargs`` (``min_new_tokens``, ``return_ids``) go to the generator."""
        if self.subtask_gen is None:
            raise RuntimeError("call enable_subtask_generation() first")
        if kv_cache is None:
            kv_cache = self.subtask_decode is not None and self.subtask_use_kv
        use_kv = kv_cache
        if use_kv:
            if self.subtask_decode is None:
                raise RuntimeError("KV-cached subtask decode was not enabled")
            if self.subtask_host_greedy and "step_hook" not in kwargs:
                kwargs["step_hook"] = _no_force  # host argmax over host logits (comparison path)
            self.subtask_gen.embed_images_graph = self._embed_images_fn
            kwargs.setdefault("sync_every", self.subtask_sync_every)
            return self.subtask_gen.generate_kv(
                images,
                image_masks,
                task,
                self._device,
                self.vision_dtype,
                self._subtask_prefill_fn,
                self._subtask_decode_fn,
                self._subtask_cache_cfg,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        kwargs.pop("img_emb", None)  # the re-prefill fallback embeds its own images
        self.subtask_gen.text_graph = (
            self._text_prefix_fn
        )  # pick up compile() if it ran after enable
        self.subtask_gen.embed_images_graph = self._embed_images_fn
        self.subtask_gen.text_graph_cached = self._text_prefix_cached_fn
        return self.subtask_gen.generate(
            images,
            image_masks,
            task,
            self._device,
            self.vision_dtype,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # -- host-side helpers ---------------------------------------------------------------
    @torch.no_grad()
    def time_conds(self, num_steps: int, batch: int) -> torch.Tensor:
        """Per-step conditioning of the Euler schedule ``t = 1, 1 - 1/n, ...`` (fp32, host): the
        AdaRMS conditions ``[n, B, W]``, or with ``denoise.modulation_tables`` (default) every
        AdaRMS norm's modulation ``[n, B, 2L+1, 3W]``. Cached per (schedule, batch), so the
        device copies are cached too."""
        tables = self.denoise.modulation_tables
        key = (num_steps, batch, tables)
        if key not in self._time_cond_cache:
            dt = -1.0 / num_steps
            ts = torch.tensor([1.0 + s * dt for s in range(num_steps)], dtype=torch.float32)
            tc = self.ref.embed_timestep(ts).float()  # [n, W]
            tc = tc[:, None, :].expand(-1, batch, -1).contiguous()
            if tables:
                tc = self.denoise.modulation_table(tc).contiguous()
            self._time_cond_cache[key] = tc
        return self._time_cond_cache[key]

    # -- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        noise=None,
        num_steps=None,
        generator=None,
        img_emb=None,
    ) -> torch.Tensor:
        """Same contract as upstream ``Pi05ForActionPrediction.sample_actions``; returns fp32 on CPU.

        ``img_emb``: this observation's image embedding from :meth:`encode_images` (pi0.52 runs it
        once for the subtask decode); the prefix then skips the vision tower."""
        if num_steps is None:
            num_steps = self.ref.num_inference_steps
        if len(images) != self.num_cameras:
            raise ValueError(
                f"Expected exactly max_cameras={self.num_cameras} image views, got {len(images)}."
            )
        bsize = lang_tokens.shape[0]
        if noise is None:
            shape = (self.ref.action_horizon, self.ref.action_dim)
            if isinstance(generator, list):
                noise = torch.stack(
                    [torch.randn(shape, dtype=torch.float32, generator=g) for g in generator]
                )
            else:
                noise = torch.randn(bsize, *shape, dtype=torch.float32, generator=generator)
        noise = noise.detach().float().cpu()

        dev = self._device
        img_valid = torch.stack([m.detach().cpu() for m in image_masks], dim=1).float()
        tok = lang_tokens.detach().cpu().long()
        tok_valid = lang_masks.detach().cpu().float()

        t0 = time.time()
        if img_emb is not None and self._prefix_from_emb_fn is not None:
            k, v, valid = self._prefix_from_emb_fn(
                img_emb, img_valid.to(dev), tok.to(dev), tok_valid.to(dev)
            )
        else:
            pix = _camera_stack(images, self.vision_dtype, dev)
            k, v, valid = self._prefix_fn(pix, img_valid.to(dev), tok.to(dev), tok_valid.to(dev))
        if PROFILE:
            valid.to("cpu")  # an output of the prefix graph: waits for it to finish
        self.stats["prefix_s"] += time.time() - t0

        t0 = time.time()
        x_t = self.denoise_loop.run(noise, self.time_conds(num_steps, bsize), (k, v, valid), dev)
        self.stats["denoise_s"] += time.time() - t0
        self.stats["calls"] += 1
        return x_t
