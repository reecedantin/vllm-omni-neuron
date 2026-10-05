# SPDX-License-Identifier: Apache-2.0
"""Neuron pi0 (base) action model: the vendored upstream pi0 model as parameter owner, fixed-shape
compiled graphs (:mod:`.graphs_pi0`), and the flow-matching schedule (on the device by default:
:class:`.model.DenoiseLoop`).

Differs from :class:`.model.NeuronPi05ActionModel` in exactly the ways π0 differs from π0.5: a
continuous ``state`` input (no state discretized into the prompt), the float64 sinusoidal
timestep embedding concatenated onto the action embedding (not an AdaRMS condition), and a plain
(unconditioned) Gemma action expert. No subtask generation — π0 has no hierarchical language.
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.nn as nn

from .graphs_pi0 import Pi0DenoiseGraph, Pi0PrefixGraph
from .model import _FP32_SELECTORS, COMPILER_ARGS, DenoiseLoop, _dtype_from_env

logger = logging.getLogger(__name__)


def _sinusoidal_time_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0
) -> torch.Tensor:
    """Byte-for-byte the vendored ``create_sinusoidal_pos_embedding`` (float64 host math);
    ``time`` is ``[n]`` fp32/fp64, returns ``[n, dimension]`` fp64."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class NeuronPi0ActionModel(nn.Module):
    """``sample_actions`` drop-in for upstream ``Pi0ForActionPrediction`` on Neuron."""

    def __init__(
        self, config, dtype: torch.dtype = torch.bfloat16, ref_model=None, denoise_mode="device"
    ):
        super().__init__()
        from ._vendor.pi0.modeling_pi0 import Pi0ForActionPrediction

        self.config = config
        self.dtype = dtype
        self.vision_dtype = _dtype_from_env("PI0_VISION_DTYPE", dtype)
        self.ref = ref_model if ref_model is not None else Pi0ForActionPrediction(config)
        self.num_cameras = int(config.max_cameras)
        self.prefix = Pi0PrefixGraph(self.ref, self.num_cameras)
        self.denoise = Pi0DenoiseGraph(self.ref)
        self._prefix_fn = self.prefix
        self._denoise_fn = self.denoise
        # (state, x_t, time_sincos, k, v, valid): x_t sits at position 1. Default "device": one
        # graph per Euler step with the state kept on the device. The 10-step unrolled graph
        # compiles but returns NaN actions on Trn2 for pi0 (pi0.5's unrolled graph is exact), and
        # is no faster here (40.6 vs 38.2 ms per 10-step schedule).
        self.denoise_loop = DenoiseLoop(self.denoise, 1, "pi0_denoise", denoise_mode)
        self._device = torch.device("cpu")
        self._time_sincos_cache: dict[int, torch.Tensor] = {}
        self.stats = {"prefix_s": 0.0, "denoise_s": 0.0, "calls": 0}

    # -- lifecycle -----------------------------------------------------------------------
    def load_checkpoint(self, model_dir: str) -> None:
        import safetensors.torch

        t0 = time.time()
        path = os.path.join(model_dir, "model.safetensors")
        state = safetensors.torch.load_file(path)
        self.ref.load_weights(state.items())
        del state
        self._apply_dtypes()
        logger.info("pi0 weights loaded from %s in %.1fs", path, time.time() - t0)

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
        for name, p in self.ref.named_parameters():  # state/action/time projections: fp32 (tiny)
            if not name.startswith("paligemma_with_expert."):
                p.data = p.data.float()
        self.prefix.mm_dtype = self.dtype
        self.denoise.mm_dtype = self.dtype
        self._time_sincos_cache.clear()

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
            self.ref.state_proj,
            self.ref.action_in_proj,
            self.ref.action_out_proj,
            self.ref.action_time_mlp_in,
            self.ref.action_time_mlp_out,
        ):
            m.to(self._device)
        self.prefix.inv_freq = self.prefix.inv_freq.to(self._device)
        self.denoise.inv_freq = self.denoise.inv_freq.to(self._device)
        if self.prefix.tp_slot is not None:
            self.prefix.tp_slot = self.prefix.tp_slot.to(self._device)
        return self

    def shard_tp(self, rank: int, size: int, group) -> None:
        """Tensor-parallel PaliGemma LM for the action prefix (:func:`.graphs.shard_lm_tp`); the
        vision tower and the action expert stay replicated. Call after ``load_checkpoint``."""
        from .graphs import shard_lm_tp

        shard_lm_tp(self.prefix, rank, size, group)

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(COMPILER_ARGS)}

        self._prefix_fn = torch.compile(
            self.prefix, backend=backend, options=opts("pi0_prefix"), **kw
        )
        self._denoise_fn = torch.compile(
            self.denoise, backend=backend, options=opts("pi0_denoise"), **kw
        )
        self.denoise_loop.set_compile(backend, base, kw, self._denoise_fn)

    # -- host-side helpers ---------------------------------------------------------------
    @torch.no_grad()
    def time_sincos(self, num_steps: int, batch: int) -> torch.Tensor:
        """Float64-host sinusoidal time embeddings ``[n, B, dim]`` for the Euler schedule, cast to
        fp32 at the boundary (the device graph never sees float64 -- NC rejects it, see graphs.py
        lesson). Cached per (schedule, batch)."""
        key = (num_steps, batch)
        if key not in self._time_sincos_cache:
            dt = -1.0 / num_steps
            ts = torch.tensor([1.0 + s * dt for s in range(num_steps)], dtype=torch.float64)
            dim = self.ref.action_in_proj.out_features
            tc = _sinusoidal_time_embedding(ts, dim).float()  # [n, dim]
            self._time_sincos_cache[key] = tc[:, None, :].expand(-1, batch, -1).contiguous()
        return self._time_sincos_cache[key]

    # -- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=None,
        generator=None,
    ) -> torch.Tensor:
        """Same contract as upstream ``Pi0ForActionPrediction.sample_actions``; returns fp32 on CPU.
        ``generator`` (the request's seeded CPU generator) draws the noise when none is given."""
        if num_steps is None:
            num_steps = self.ref.num_inference_steps
        if len(images) != self.num_cameras:
            raise ValueError(f"Expected exactly {self.num_cameras} image views, got {len(images)}.")
        bsize = state.shape[0]
        if noise is None:
            noise = torch.randn(
                bsize,
                self.ref.action_horizon,
                self.ref.action_dim,
                dtype=torch.float32,
                generator=generator,
            )
        noise = noise.detach().float().cpu()

        dev = self._device
        pix = torch.stack([im.detach().cpu() for im in images], dim=1)
        pix = (
            pix.reshape(bsize * self.num_cameras, *pix.shape[2:]).to(self.vision_dtype).contiguous()
        )
        img_valid = torch.stack([m.detach().cpu() for m in image_masks], dim=1).float()
        tok = lang_tokens.detach().cpu().long()
        tok_valid = lang_masks.detach().cpu().float()
        state_cpu = state.detach().float().cpu()

        t0 = time.time()
        k, v, valid = self._prefix_fn(
            pix.to(dev), img_valid.to(dev), tok.to(dev), tok_valid.to(dev)
        )
        self.stats["prefix_s"] += time.time() - t0

        t0 = time.time()
        x_t = self.denoise_loop.run(
            noise, self.time_sincos(num_steps, bsize), (state_cpu.to(dev), k, v, valid), dev
        )
        self.stats["denoise_s"] += time.time() - t0
        self.stats["calls"] += 1
        return x_t
