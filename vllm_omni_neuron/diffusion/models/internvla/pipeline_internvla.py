# SPDX-License-Identifier: Apache-2.0
"""Neuron InternVLA-A1.5 pipeline: a policy served through vLLM-Omni's offline/online runner.

Mirrors the shape of upstream's own policy pipeline for GR00T
(``vllm_omni.diffusion.models.gr00t.pipeline_gr00t.Gr00tN1d7Pipeline``, read from the installed
package, not imported): a plain ``nn.Module`` (no diffusers ``DiffusionPipeline`` base needed for
an action-only policy), ``forward(req)`` reads the robot observation from
``req.sampling_params.extra_args["robot_obs"]``, and returns the action chunk through
``DiffusionOutput(output={"actions": ...})`` -- the engine's "empty output" guard only requires
``output`` to be non-empty, so this plain dict return (not the richer Cosmos3-Edge
``custom_output["action"]`` + ``action_post_process_func`` route, which exists for pipelines that
*also* produce a video) is the right shape for a pure action policy.

A1.5 has no upstream Omni pipeline to subclass (unlike GR00T): this class owns the whole request
contract itself, wrapping :class:`~.policy.InternVLAA15Runner` for the three device graphs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import preprocess as pp
from .policy import InternVLAA15, InternVLAA15Runner

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # the checkpoint has no model_index.json _class_name of its own (it is a LeRobot policy
        # folder, not a diffusers pipeline repo); callers set engine_args.model_class_name to this
        "model_arch": "InternVLAA15Pipeline",
        "class_name": "NeuronInternVLAA15Pipeline",
    },
]

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


def _dummy_actions(policy: InternVLAA15) -> dict[str, np.ndarray]:
    """Zero action chunk of the right shape, for the engine's warm-up/dummy request."""
    p = policy.cfg.policy
    return {"action": np.zeros((1, p.n_action_steps, p.action_dim), dtype=np.float32)}


def _robot_obs_to_batch(robot_obs: Mapping[str, Any], cfg) -> dict[str, torch.Tensor]:
    """``robot_obs`` (as GR00T's pipeline normalises it: ``images``/``video``, ``state``,
    ``language``/``prompt``) -> the model's batch dict (upstream's ``predict_action_chunk`` input
    shape). Real Qwen3-VL tokenisation/image-processing is pushed to :mod:`.preprocess` so this
    function stays about request shape, not VLM preprocessing details.
    """
    images = robot_obs.get("images")
    if images is None:
        images = robot_obs.get("video")
    if images is None:
        raise ValueError("robot_obs must include 'images' or 'video'")
    images = [images] if not isinstance(images, (list, tuple)) else list(images)
    imgs = torch.stack([torch.as_tensor(np.asarray(im)) for im in images]).float()
    if imgs.max() > 1.5:  # uint8 pixels -> [0,1], as the Qwen2-VL image processor expects
        imgs = imgs / 255.0
    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):  # HWC -> CHW
        imgs = imgs.permute(0, 3, 1, 2)
    text = robot_obs.get("language") or robot_obs.get("prompt") or ""
    patches, (gh, gw) = pp.images_to_patches(imgs, cfg.vlm.vision)
    n = imgs.shape[0]
    tok = _text_tokens(str(text), cfg)
    merge = cfg.vlm.vision.spatial_merge_size
    n_tok = (gh * gw) // (merge * merge)
    ids = tok[:1] + [cfg.vlm.vision_start_token_id]
    for _ in range(n):
        ids += [cfg.vlm.image_token_id] * n_tok
    ids += [cfg.vlm.vision_end_token_id] + tok[1:]
    input_ids = torch.tensor([ids], dtype=torch.long)
    state = robot_obs.get("state")
    state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32).reshape(1, -1) if state is not None \
        else torch.zeros(1, cfg.policy.max_state_dim)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": patches,
        "image_grid_thw": torch.tensor([[1, gh, gw]] * n, dtype=torch.long),
        "state": state_t,
    }


def _text_tokens(text: str, cfg) -> list[int]:
    """Minimal placeholder tokenisation (BOS-less byte fallback) until a real Qwen3.5 tokenizer
    directory is wired in; keeps request plumbing testable without the Hub."""
    tok_path = os.environ.get("INTERNVLA_TOKENIZER")
    if tok_path:
        from transformers import Qwen3_5Tokenizer

        tk = Qwen3_5Tokenizer.from_pretrained(tok_path)
        return tk(text).input_ids
    ids = [min(1000 + b, cfg.vlm.text.vocab_size - 1) for b in text.encode("utf-8")] or [1000]
    return ids[: cfg.policy.tokenizer_max_length]


class NeuronInternVLAA15Pipeline(nn.Module):
    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config or {}
        vlm_config = model_config.get("vlm_config") or os.environ.get("INTERNVLA_VLM_CONFIG")
        self.model_path = od_config.model
        self.device = "cpu"
        logger.info("Loading InternVLA-A1.5 from %s", self.model_path)
        policy = InternVLAA15.from_pretrained(self.model_path, dtype=torch.bfloat16, vlm_config=vlm_config)
        # NOT a registered submodule (object.__setattr__, bypassing nn.Module.__setattr__): the
        # engine's generic weight loader walks every registered submodule's named_parameters()
        # and tries to match them against the checkpoint's safetensors keys directly, which fails
        # ("not initialized from checkpoint") because the weights are already loaded under our own
        # tensor-name mapping. GR00T's pipeline does the same for the identical reason.
        object.__setattr__(self, "_policy", policy)
        self._runner: InternVLAA15Runner | None = None  # built in to(), once the final device is known

    # -- engine hooks ------------------------------------------------------------------------
    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()  # loaded directly by InternVLAA15.from_pretrained, like GR00T's Gr00tPolicy

    def load_weights(self, weights) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(f"{type(self).__name__}.load_weights received {len(consumed)} tensors; "
                               "weights_sources=() should prevent this.")
        return set()

    def to(self, *args, **kwargs):
        device, _dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = torch.device(device)
            self._policy.to(self.device)
        self._runner = InternVLAA15Runner(self._policy, self.device)
        return self

    def compile(self, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def wrap(mod, name):
            return torch.compile(mod, backend=backend, options={**base, "model_name": name,
                                                               "compiler_args": list(COMPILER_ARGS)}, **kw)

        self._runner = InternVLAA15Runner(self._policy, self.device, wrap)
        return self

    def reset(self) -> dict[str, Any]:
        return {}

    # -- inference ---------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, req, **kwargs):
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID

        del kwargs
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                return DiffusionOutput(output={"actions": _dummy_actions(self._policy)})
            return DiffusionOutput(error=f"{type(self).__name__}.forward expects "
                                   "sampling_params.extra_args['robot_obs'].")
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(error=f"robot_obs must be a dict, got {type(robot_obs).__name__}.")

        batch = _robot_obs_to_batch(robot_obs, self._policy.cfg)
        seed = getattr(req.sampling_params, "seed", None)
        noise = pp.initial_noise(self._policy.cfg, batch=1, seed=int(seed) if seed is not None else 0)
        runner = self._runner or InternVLAA15Runner(self._policy, self.device)
        actions = runner.sample_actions(batch, noise)
        return DiffusionOutput(output={"actions": {"action": actions.cpu().float().numpy()}})
