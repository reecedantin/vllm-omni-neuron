# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline wrapper for FLUX 3 Action, so the policy can be SERVED through the plugin's
stage harness (the same multi-process path Cosmos3-Edge uses; TP today runs TP1, TP4 later via the
stage ``devices:`` list).

This is deliberately thin: all the real work lives in :class:`NeuronFlux3ActionPolicy`
(:mod:`.policy`). The pipeline only adapts the Omni request/response contract (the minimal shape the
``helloworld`` reference pipeline uses) to the policy:

* a request carries the DROID observation as ``multi_modal_data['image']`` (the 540x640 wrist/exterior
  composite a RoboLab/Cosmos client sends, or a pre-tiled canvas), the robot ``state`` and ``task``
  through ``sampling_params.extra_args``, and the seed through ``sampling_params.seed``;
* ``forward`` runs the policy and returns ``DiffusionOutput(output={'actions': ..., 'video': ...})`` --
  the action-envelope shape the action post-process func unpacks.

The DiT and the frozen VAE encoder run on the NeuronCore (the encoder on rank 0 only under TP);
``model_config.vae_device: host`` (or ``FLUX3_ACTION_VAE_DEVICE=host``) keeps the encoder on the
host instead. Video decode is optional and host-side.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        "model_arch": "Flux3ActionPipeline",
        "class_name": "NeuronFlux3ActionPipeline",
        "post_process_func_name": "get_flux3_action_post_process_func",
        "action_post_process_func_name": "get_flux3_action_post_process_func",
    },
]


def _extra(request: Any) -> dict:
    sp = getattr(request, "sampling_params", None)
    ex = getattr(sp, "extra_args", None)
    return ex if isinstance(ex, dict) else {}


def _observation_from_request(request: Any) -> tuple[dict, str, int]:
    """Pull the DROID observation batch, task caption and seed out of an Omni request.

    Accepts either a pre-tiled DROID composite (``multi_modal_data['image']``, HWC uint8 540x640 or
    a path/PIL) plus ``state`` in ``extra_args``, or the three named camera arrays directly in
    ``extra_args`` (``images.wrist`` / ``images.left`` / ``images.right``).
    """
    import numpy as np

    ex = _extra(request)
    sp = getattr(request, "sampling_params", None)
    seed = int(getattr(sp, "seed", 0) or 0)
    task = ex.get("task") or getattr(request, "prompt", "") or ""
    if isinstance(task, dict):
        task = task.get("prompt", "")
    state = ex.get("state")
    if state is None:
        # The engine's startup _dummy_run sends a bare request (no state/images) only to warm the
        # compiled graphs. Synthesize a zero DROID observation so compilation proceeds; a real request
        # always carries state + cameras.
        zero_cam = torch.zeros(1, 3, 360, 640, dtype=torch.uint8)
        batch = {k: zero_cam.clone() for k in ("images.wrist", "images.left", "images.right")}
        batch.update(
            state=torch.zeros(1, 8),
            task=[task if isinstance(task, str) else ""],
            _composite=False,
            _dummy=True,
        )
        return batch, "", seed
    state_t = torch.as_tensor(np.asarray(state, dtype=np.float32))

    def _as_hwc_uint8(v):
        if isinstance(v, dict) and "data" in v:  # raw bytes: {"data", "shape", "dtype": "uint8"}
            return np.frombuffer(v["data"], dtype=np.uint8).reshape(v["shape"]).copy()
        if isinstance(v, (bytes, bytearray, memoryview)):
            raise ValueError("raw camera bytes need their shape: send {'data', 'shape'}")
        if isinstance(v, str):
            from PIL import Image

            v = np.asarray(Image.open(v).convert("RGB"), dtype=np.uint8)
        elif hasattr(v, "convert"):  # PIL
            v = np.asarray(v.convert("RGB"), dtype=np.uint8)
        return np.asarray(v, dtype=np.uint8)

    cams = {}
    for key in ("images.wrist", "images.left", "images.right"):
        if key in ex:
            arr = _as_hwc_uint8(ex[key])  # (360, 640, 3)
            cams[key] = torch.from_numpy(arr).permute(2, 0, 1)[None].contiguous()
    if cams:
        batch = {**cams, "state": state_t[None], "task": [task]}
        batch["_composite"] = False
        return batch, task, seed
    # composite path: a single 540x640 image, split back into the three DROID tiles
    mm = getattr(request, "multi_modal_data", None) or {}
    img = mm.get("image") if isinstance(mm, dict) else None
    if img is None:
        raise ValueError("flux3_action request needs images.* in extra_args or a composite image")
    composite = torch.from_numpy(_as_hwc_uint8(img))  # (540, 640, 3)
    return (
        {"_composite_image": composite, "state": state_t, "task": task, "_composite": True},
        task,
        seed,
    )


class NeuronFlux3ActionPipeline(nn.Module):
    """Thin Omni pipeline over :class:`NeuronFlux3ActionPolicy` (DROID default profile)."""

    weights_sources: list = []  # weights come from the policy's own loaders
    vae = None  # the policy owns its (host) VAE; this satisfies Omni's vae_* attribute checks

    def __init__(self, *, od_config: Any, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self._policy = None
        self._weights_requested = False
        self._compiled = False
        self._device = torch.device("cpu")
        base = (getattr(od_config, "model_config", None) or {}).get(
            "flux3_action_base"
        ) or os.environ.get("FLUX3_ACTION_BASE")
        self._base_dir = base
        # Where the VAE encoder runs: "device" (default: the NeuronCore, rank 0 under TP) or "host".
        self._vae_on = str(
            (getattr(od_config, "model_config", None) or {}).get("vae_device")
            or os.environ.get("FLUX3_ACTION_VAE_DEVICE", "device")
        ).lower()
        if self._vae_on not in ("device", "host"):
            raise ValueError(f"vae_device must be 'device' or 'host', got {self._vae_on!r}")
        # Where the caption encoder's decoder layers run: "device" (default; over the TP group,
        # cached per caption) or "host" (transformers on the CPU, the fallback).
        self._text_on = str(
            (getattr(od_config, "model_config", None) or {}).get("text_encoder_device")
            or os.environ.get("FLUX3_ACTION_TEXT_DEVICE", "device")
        ).lower()
        if self._text_on not in ("device", "host"):
            raise ValueError(
                f"text_encoder_device must be 'device' or 'host', got {self._text_on!r}"
            )
        self._decode = (
            bool((getattr(od_config, "model_config", None) or {}).get("decode_frames", False))
            or os.environ.get("FLUX3_ACTION_DECODE") == "1"
        )

    # -- lifecycle (Omni calls load_weights -> to -> compile) ----------------------------
    # The policy is BUILT LAZILY (in _ensure_policy) rather than in load_weights, because the engine
    # calls load_weights BEFORE to(device): building earlier would load the 14 GB DiT onto the CPU and
    # then to()/compile would run graphs whose weights never left the host ("two different devices").
    def load_weights(self, weights: object = None) -> set[str]:
        self._weights_requested = True
        return set()

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return self

    def _ensure_policy(self):
        if self._policy is not None:
            return self._policy
        from .policy import NeuronFlux3ActionPolicy

        pc = getattr(self.od_config, "parallel_config", None)
        tp = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        cfg_size = int(getattr(pc, "cfg_parallel_size", 1) or 1)
        rank, group = 0, None
        par = {}
        if tp > 1:
            from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import _tp_state

            tp, rank, group = _tp_state()
        if tp > 1 or cfg_size > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            world = get_world_group()
            par.update(world_rank=world.rank_in_group, world_group=world.cpu_group)
        if cfg_size > 1:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_cfg_group,
                get_classifier_free_guidance_rank,
            )

            par.update(
                cfg_size=cfg_size,
                cfg_rank=get_classifier_free_guidance_rank(),
                cfg_group=get_cfg_group(),  # the coordinator: host_all_gather needs its rank order
            )
        self._policy = NeuronFlux3ActionPolicy(
            self.od_config.model,
            base_dir=self._base_dir,
            device=self._device,
            tp_size=tp,
            tp_rank=rank,
            tp_group=group,
            vae_device=self._device if self._vae_on == "device" else torch.device("cpu"),
            text_device=self._device if self._text_on == "device" else torch.device("cpu"),
            **par,
        )
        return self._policy

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        self._ensure_policy()
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self._policy.compile(backend, options)
        self._compiled = True
        return self

    # -- inference ------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, request: Any):
        import time

        from vllm_omni.diffusion.data import DiffusionOutput

        t_enter = time.time()
        self._ensure_policy()
        batch, _task, seed = _observation_from_request(request)
        t_parsed = time.time()
        if batch.get("_composite"):
            actions = self._policy.predict_from_composite(
                batch["_composite_image"], batch["state"], batch["task"], seed
            )
            out = {"actions": actions}
        else:
            batch_is_dummy = bool(batch.get("_dummy"))
            batch = {k: v for k, v in batch.items() if not k.startswith("_")}
            # optional per-request step count (e.g. a 1-step accuracy probe); the DiT graphs are
            # step-independent, only the host step tables change
            steps = _extra(request).get("num_inference_steps")
            default_steps = self._policy.config.num_inference_steps
            if steps is not None:
                self._policy.config.num_inference_steps = int(steps)
            try:
                result = self._policy.predict(batch, seed=seed)
            finally:
                self._policy.config.num_inference_steps = default_steps
            out = {
                "actions": result.actions,
                "video_latents": result.video_latents,
                "cond_latents": result.cond_latents,
                "timing": dict(result.timing),
            }
            if self._decode and not batch_is_dummy:
                out["video"] = self._policy.decode_frames(result)
        if "timing" in out:
            # Whole-request accounting inside the worker (wall clock, same host as the client):
            # the client's send time rides in extra_args, so transport in = enter - sent.
            sent = _extra(request).get("client_send_time")
            out["timing"].update(
                worker_enter_time=t_enter,
                parse_s=t_parsed - t_enter,
                forward_s=time.time() - t_enter,
                worker_exit_time=time.time(),
            )
            if sent is not None:
                out["timing"]["transport_in_s"] = t_enter - float(sent)
        return DiffusionOutput(output=out)


def get_flux3_action_post_process_func(od_config: Any):
    """Return a post-process callable: ``DiffusionOutput`` -> request-output payload.

    Keeps the action envelope as a dict so the client reads ``actions`` (and ``video`` when decoded).
    """

    def post_process(output: Any, *args: Any, **kwargs: Any) -> Any:
        payload = getattr(output, "output", output)
        if isinstance(payload, dict):
            clean = {}
            for k, v in payload.items():
                clean[k] = v.detach().cpu() if hasattr(v, "detach") else v
            return [{"payload": clean}]
        return [payload]

    return post_process
