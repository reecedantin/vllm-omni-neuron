# SPDX-License-Identifier: Apache-2.0
"""Neuron pi0.5 / pi0.52 serving pipeline for the vLLM-Omni diffusion engine.

A drop-in for the vendored upstream ``Pi05Pipeline`` whose model is the Neuron
:class:`NeuronPi05ActionModel` (fixed-shape compiled prefix + denoise graphs) instead of the
eager ``Pi05ForActionPrediction``. The pipeline owns all preprocessing via the vendored
``Pi05Processor`` (normalization, relative actions, state discretization, tokenization), and for
pi0.52 runs the hierarchical subtask-generation step before building the low-level action prompt.

Registered under ``Pi05Pipeline`` and ``pi052`` so a deploy yaml's ``model_class_name`` resolves
here. One DIFFUSION stage, ``final_output_type='action'`` (see the vendored ``PI05_PIPELINE``
topology). Served multiprocess like every vLLM-Omni stage; the model runner calls ``compile`` once
(regional compilation) and ``forward(req)`` per request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import fields as dataclass_fields

import numpy as np
import torch
import torch.nn as nn
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

from ._vendor.pi05.config import Pi05Config
from ._vendor.pi05.pipeline_pi05 import get_pi05_post_process_func  # identity post-process, reused
from ._vendor.pi05.processor_pi05 import Pi05Processor, build_pi05_prompt
from .model import NeuronPi05ActionModel
from .request import DEFAULT_HOST_THREADS, RequestTimer, decode_robot_obs, torch_threads, tp_state

logger = logging.getLogger(__name__)


def _decode_noise(v):
    """``extra_args['noise']``: a tensor/array/nested list, or ``{"data", "shape", "dtype"}``."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu()
    if isinstance(v, dict) and "data" in v:
        v = np.frombuffer(v["data"], dtype=np.dtype(v.get("dtype", "float32"))).reshape(v["shape"])
    return torch.as_tensor(np.array(v, dtype=np.float32))


DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)

PIPELINE_REGISTRY = [
    {
        "model_arch": "Pi05Pipeline",
        "class_name": "NeuronPi05Pipeline",
        "post_process_func_name": "get_pi05_post_process_func",
    },
    {
        # LeRobot policy types, so a checkpoint's config.json `type` resolves here too.
        "model_arch": "pi05",
        "class_name": "NeuronPi05Pipeline",
        "post_process_func_name": "get_pi05_post_process_func",
    },
    {
        "model_arch": "pi052",
        "class_name": "NeuronPi05Pipeline",
        "post_process_func_name": "get_pi05_post_process_func",
    },
]


def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
    dt = od_config.dtype
    resolved = dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt).split(".")[-1], None)
    if resolved not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported pi0.5 dtype: {dt!r}. Supported: "
            f"{sorted(str(d).split('.')[-1] for d in SUPPORTED_DTYPES)}."
        )
    return resolved


class NeuronPi05Pipeline(nn.Module):
    """pi0.5 / pi0.52 VLA on Neuron: raw robot obs -> continuous action chunk."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)
        self._dtype = _resolve_dtype(od_config)
        self.tokenizer_source = self._resolve_tokenizer_source(od_config)
        self.tokenizer = self._load_tokenizer()
        mc = od_config.model_config or {}
        self.max_new_subtask_tokens = int(mc.get("max_new_subtask_tokens", 48))
        # pi0.52, benchmarking only: mask EOS for this many decode steps (fixed-length subtask).
        self.min_new_subtask_tokens = int(mc.get("min_new_subtask_tokens", 0))
        self.last_subtask_ids: list[int] = []
        self.host_threads = int(mc.get("host_threads", DEFAULT_HOST_THREADS))
        # pi0.52: run SigLIP once per request and reuse the embedding for the action prefix.
        self.reuse_image_embedding = str(mc.get("reuse_image_embedding", True)).lower() not in (
            "0",
            "false",
            "no",
        )

        self.model = NeuronPi05ActionModel(
            self.config,
            dtype=self._dtype,
            denoise_mode=str(mc.get("denoise_mode", "device")),
            adarms_tables=str(mc.get("adarms_tables", True)).lower() not in ("0", "false", "no"),
        )
        self.model.subtask_sync_every = int(mc.get("subtask_sync_every", 1))
        self.model.load_checkpoint(self.model_dir)
        self.tp_size, self.tp_rank, tp_group = tp_state()
        if self.tp_size > 1:  # the PaliGemma LM only; vision tower + action expert replicated
            self.model.shard_tp(self.tp_rank, self.tp_size, tp_group)
        if self.config.policy_type == "pi052":
            # Hierarchical language: generate a low-level subtask from the task before acting.
            # subtask_kv_cache (default on): prefill once + KV-cached decode steps.
            kv = mc.get("subtask_kv_cache", True)
            self.model.enable_subtask_generation(
                self.tokenizer, kv_cache=str(kv).lower() not in ("0", "false", "no")
            )
        self.processor = Pi05Processor(self.config, self.tokenizer, torch.device("cpu"))

    # -- construction helpers -----------------------------------------------------------
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

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        cfg = Pi05Config.from_pretrained(self.model_dir) if self.model_dir else None
        if cfg is None:
            return Pi05Config.from_model_config(od_config.model_config)
        if od_config.model_config:
            resolved = {
                f.name: getattr(cfg, f.name) for f in dataclass_fields(Pi05Config) if f.init
            }
            resolved.update({k: v for k, v in od_config.model_config.items() if k in resolved})
            cfg = Pi05Config.from_model_config(resolved)
        return cfg

    def _resolve_tokenizer_source(self, od_config: OmniDiffusionConfig) -> str:
        # model_config, not custom_pipeline_args: the engine reserves the latter for a custom
        # pipeline class (it requires a "pipeline_class" key there).
        tok = (od_config.model_config or {}).get("tokenizer")
        if tok:
            return str(tok)
        if self.model_dir and os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
            return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_source, padding_side="right")

    # -- framework hooks ----------------------------------------------------------------
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs):
        """Regional compilation hook (NeuronDiffusionModelRunner). Compiles the prefix, denoise,
        and (pi0.52) text-prefix graphs; returns self so the runner keeps driving this object."""
        self.model.compile(backend, options=options, **kwargs)
        return self

    def load_weights(self, weights=()):  # the model self-loads its checkpoint
        for _ in weights:
            pass
        return None

    # -- inference ----------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        robot_obs = extra.get("robot_obs")
        num_steps = getattr(req.sampling_params, "num_inference_steps", None)

        if robot_obs is None:
            # Warmup / dummy path: return zeros so engine capture doesn't crash (mirrors upstream).
            prompt = req.prompts[0] if req.prompts else ""
            prompt = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
            if prompt == "dummy run" or num_steps == 1:
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.action_dim), dtype=np.float32
                        )
                    }
                )
            return DiffusionOutput(
                error="NeuronPi05Pipeline.forward requires extra_args['robot_obs']."
            )

        timer = RequestTimer(extra)
        robot_obs = decode_robot_obs(robot_obs)
        timer.mark("decode")
        with torch_threads(self.host_threads):
            # One preprocessing pass: the subtask decode and the action prompt see the same images.
            images, image_masks, _, _ = self.processor.build_model_inputs(
                {**robot_obs, "prompt": robot_obs.get("prompt") or robot_obs.get("task") or ""}
            )
        timer.mark("preprocess")
        img_emb = None
        if self.config.policy_type == "pi052" and self.model.subtask_gen is not None:
            if self.reuse_image_embedding:
                img_emb = self.model.encode_images(images)  # SigLIP once: subtask + action prefix
            timer.mark("embed_images")
        robot_obs = self._maybe_generate_subtask(robot_obs, images, image_masks, img_emb)
        timer.mark("subtask")
        with torch_threads(self.host_threads):
            lang_tokens, lang_masks = self._build_prompt(robot_obs)
        timer.mark("prompt")
        if num_steps is not None:
            num_steps = int(num_steps)
        # A fixed noise tensor (extra_args['noise']) makes a run reproducible / comparable to a
        # reference; otherwise the generator (or the global RNG) draws it.
        noise = _decode_noise(extra.get("noise"))
        st = self.model.stats
        p0, d0 = st["prefix_s"], st["denoise_s"]
        actions = self.model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise,
            num_steps=num_steps,
            generator=getattr(req.sampling_params, "generator", None),
            img_emb=img_emb,
        )
        timer.mark("actions")
        out = self.processor.build_model_outputs(actions, robot_obs)
        timer.mark("postprocess")
        pi052 = self.config.policy_type == "pi052"
        timer.finish(
            actions=out,
            prefix_launch=round(1e3 * (st["prefix_s"] - p0), 3),
            denoise=round(1e3 * (st["denoise_s"] - d0), 3),
            subtask_text=robot_obs.get("prompt") if pi052 else None,
            subtask_steps=len(self.last_subtask_ids) if pi052 else None,
            subtask_ids_sha256=(
                hashlib.sha256(json.dumps(self.last_subtask_ids).encode()).hexdigest()[:16]
                if pi052
                else None
            ),
            tp_rank=self.tp_rank,
        )
        return DiffusionOutput(output={"actions": out})

    def _build_prompt(self, robot_obs: dict):
        """Language tokens for the action phase.

        pi0.5: the vendored processor's ``"Task: {task}, State: {bins};\\nAction: "`` padded to
        ``tokenizer_max_length``. pi0.52: LeRobot's ``_prepare_action_batch`` (non-joint) instead
        tokenizes ``"User: {subtask}, State: {bins};\\n"`` UNPADDED (``_build_text_batch`` with
        ``add_generation_prompt=False``)."""
        from ._vendor.pi05.processor_pi05 import (
            apply_norm,
            as_state_vector,
            build_pi05_prompt,
            discretize_state,
            tokenize_prompt,
        )

        raw_state = as_state_vector(robot_obs.get("state"), self.config.state_dim)
        norm_state = apply_norm(torch.from_numpy(raw_state), self.processor._state_norm).numpy()
        if self.config.policy_type != "pi052":
            prompt = build_pi05_prompt(
                task=robot_obs.get("prompt", "") or "",
                normalized_state=norm_state,
                state_num_bins=self.config.state_num_bins,
            )
            ids, attn = tokenize_prompt(self.tokenizer, prompt, self.config.tokenizer_max_length)
            return (
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([attn], dtype=torch.bool),
            )
        bins = discretize_state(norm_state, num_bins=self.config.state_num_bins)
        state_str = " ".join(str(int(x)) for x in bins.reshape(-1).tolist())
        subtask = (robot_obs.get("prompt") or robot_obs.get("task") or "").strip()
        # _format_messages([{user: content}]) == "User: {content}\n"; content = "{subtask}, State: {bins};".
        prompt = f"User: {subtask}, State: {state_str};\n"
        enc = self.tokenizer(prompt, add_special_tokens=True, return_tensors=None)
        return (
            torch.tensor([enc["input_ids"]], dtype=torch.long),
            torch.tensor([enc["attention_mask"]], dtype=torch.bool),
        )

    def _build_inputs(self, robot_obs: dict):
        """Images + language tokens for the action phase (the pre-split form, kept for callers
        that build a batch directly; :meth:`forward` preprocesses the images only once)."""
        images, image_masks, _, _ = self.processor.build_model_inputs(robot_obs)
        lang_tokens, lang_masks = self._build_prompt(robot_obs)
        return images, image_masks, lang_tokens, lang_masks

    def _maybe_generate_subtask(
        self, robot_obs: dict, images=None, image_masks=None, img_emb=None
    ) -> dict:
        """pi0.52: replace the high-level task with a generated low-level subtask, matching
        LeRobot ``_generate_low_level_subtask`` (non-joint path). pi0.5 passes through unchanged.

        The action prompt is built from ``robot_obs['prompt']`` (``_assemble_model_inputs``), so the
        generated subtask is written there; ``task`` is also set for callers that read it.
        ``images``/``image_masks``/``img_emb``: this observation's preprocessed cameras (and image
        embedding), when the caller already has them."""
        if self.config.policy_type != "pi052" or self.model.subtask_gen is None:
            return robot_obs
        task = robot_obs.get("prompt") or robot_obs.get("task") or ""
        if not task:
            return robot_obs
        if images is None:
            images, image_masks, _, _ = self.processor.build_model_inputs(
                {**robot_obs, "prompt": task}
            )
        raw, self.last_subtask_ids = self.model.generate_subtask(
            images,
            image_masks,
            task,
            max_new_tokens=self.max_new_subtask_tokens,
            min_new_tokens=self.min_new_subtask_tokens,
            return_ids=True,
            img_emb=img_emb,
        )
        # LeRobot: a non-empty generation is used verbatim (whitespace-collapsed); an empty one
        # falls back to the task string itself (_fallback_subtask_from_task, non-navigation case).
        subtask = " ".join(raw.strip().split()) if raw and raw.strip() else task
        logger.info("pi0.52 subtask: %r -> %r (generated=%r)", task, subtask, raw)
        out = dict(robot_obs)
        out["prompt"] = subtask
        out["task"] = subtask
        return out


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronPi05Pipeline",
    "get_pi05_post_process_func",
    "build_pi05_prompt",
]
