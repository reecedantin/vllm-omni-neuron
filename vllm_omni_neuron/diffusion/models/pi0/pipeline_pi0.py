# SPDX-License-Identifier: Apache-2.0
"""Neuron pi0 (base) serving pipeline for the vLLM-Omni diffusion engine.

A drop-in for the vendored upstream ``Pi0Pipeline`` whose model is the Neuron
:class:`.model_pi0.NeuronPi0ActionModel` (fixed-shape compiled prefix + denoise graphs) instead
of the eager ``Pi0ForActionPrediction``. Preprocessing is the vendored ``build_model_inputs``
(images, 48-token PaliGemma prompt, continuous state); state normalization and action
unnormalization stay on the vendored model (identity for ``lerobot/pi0_base``).

Registered under ``Pi0Pipeline`` (the upstream key) and ``pi0`` (the LeRobot policy type).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

from ._vendor.pi0.config import Pi0Config
from ._vendor.pi0.pipeline_pi0 import get_pi0_post_process_func
from ._vendor.pi0.processor_pi0 import build_model_inputs
from .model_pi0 import NeuronPi0ActionModel
from .pipeline import _decode_noise, _resolve_dtype
from .request import DEFAULT_HOST_THREADS, RequestTimer, decode_robot_obs, torch_threads, tp_state

logger = logging.getLogger(__name__)


def _request_generator(sampling_params):
    """The request's noise generator (the model runner seeds a CPU one from ``seed``), if any."""
    gen = getattr(sampling_params, "generator", None)
    if isinstance(gen, list) and len(gen) == 1:
        gen = gen[0]
    return gen if isinstance(gen, torch.Generator) else None


DEFAULT_PI0_TOKENIZER = "google/paligemma-3b-pt-224"

PI0_PIPELINE_REGISTRY = [
    {
        "model_arch": "Pi0Pipeline",
        "class_name": "NeuronPi0Pipeline",
        "post_process_func_name": "get_pi0_post_process_func",
    },
    {
        "model_arch": "pi0",
        "class_name": "NeuronPi0Pipeline",
        "post_process_func_name": "get_pi0_post_process_func",
    },
]


class NeuronPi0Pipeline(nn.Module):
    """pi0 VLA on Neuron: raw robot obs -> continuous action chunk."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.model_dir = self._resolve_model_dir(od_config.model)
        cfg = Pi0Config.from_pretrained(self.model_dir) if self.model_dir else Pi0Config()
        if od_config.model_config:
            merged = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
            merged.update({k: v for k, v in od_config.model_config.items() if k in merged})
            cfg = Pi0Config.from_model_config(merged)
        self.config = cfg
        self._dtype = _resolve_dtype(od_config)
        # model_config, not custom_pipeline_args (reserved by the engine for a custom pipeline class).
        tok_src = (od_config.model_config or {}).get("tokenizer") or (
            self.model_dir
            if self.model_dir
            and os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json"))
            else DEFAULT_PI0_TOKENIZER
        )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_src), padding_side="right")
        mc = od_config.model_config or {}
        self.host_threads = int(mc.get("host_threads", DEFAULT_HOST_THREADS))
        self.model = NeuronPi0ActionModel(
            self.config, dtype=self._dtype, denoise_mode=str(mc.get("denoise_mode", "device"))
        )
        self.model.load_checkpoint(self.model_dir)
        self.tp_size, self.tp_rank, tp_group = tp_state()
        if self.tp_size > 1:  # the PaliGemma LM only; vision tower + action expert replicated
            self.model.shard_tp(self.tp_rank, self.tp_size, tp_group)

    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        if not model:
            return None
        if os.path.isdir(model):
            return model
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"]
        )

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs):
        self.model.compile(backend, options=options, **kwargs)
        return self

    def load_weights(self, weights=()):  # the model self-loads its checkpoint
        for _ in weights:
            pass
        return None

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra.get("robot_obs")
        num_steps = getattr(req.sampling_params, "num_inference_steps", None)
        if robot_obs is None:
            prompt = req.prompts[0] if req.prompts else ""
            prompt = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
            if prompt == "dummy run" or num_steps == 1:
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.max_action_dim), dtype=np.float32
                        )
                    }
                )
            return DiffusionOutput(
                error="NeuronPi0Pipeline.forward requires extra_args['robot_obs']."
            )

        timer = RequestTimer(extra)
        robot_obs = decode_robot_obs(robot_obs)
        timer.mark("decode")
        with torch_threads(self.host_threads):
            images, image_masks, lang_tokens, lang_masks, state = build_model_inputs(
                robot_obs, self.config, self.tokenizer, torch.device("cpu")
            )
            state = self.model.ref._normalize_state(state)
        timer.mark("preprocess")
        noise = _decode_noise(extra.get("noise"))
        st = self.model.stats
        p0, d0 = st["prefix_s"], st["denoise_s"]
        actions = self.model.sample_actions(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=None if num_steps is None else int(num_steps),
            generator=_request_generator(req.sampling_params),
        )
        timer.mark("actions")
        actions = self.model.ref._unnormalize_actions(actions)
        out = actions.squeeze(0).float().cpu().numpy()
        timer.mark("postprocess")
        timer.finish(
            actions=out,
            prefix_launch=round(1e3 * (st["prefix_s"] - p0), 3),
            denoise=round(1e3 * (st["denoise_s"] - d0), 3),
            tp_rank=self.tp_rank,
        )
        return DiffusionOutput(output={"actions": out})


__all__ = ["PI0_PIPELINE_REGISTRY", "NeuronPi0Pipeline", "get_pi0_post_process_func"]
