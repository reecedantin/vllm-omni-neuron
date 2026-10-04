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

import contextlib
import functools
import logging
import os
import time
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


def _camera(im) -> np.ndarray:
    """One camera frame as an array. Accepts an array/list, a PIL image, or raw bytes as
    ``{"data": bytes, "shape": [H, W, C]}`` (uint8 unless ``"dtype"`` says otherwise). Raw bytes
    are the cheap transport: a nested-list frame costs seconds to serialize between the client
    and the worker, a bytes frame microseconds."""
    if isinstance(im, Mapping) and "data" in im:
        return (
            np.frombuffer(im["data"], dtype=np.dtype(im.get("dtype", "uint8")))
            .reshape(im["shape"])
            .copy()
        )
    if isinstance(im, (bytes, bytearray, memoryview)):
        raise ValueError(
            "raw camera bytes need their shape: send {'data': ..., 'shape': [H, W, C]}"
        )
    if hasattr(im, "convert"):  # PIL
        return np.asarray(im.convert("RGB"))
    return np.asarray(im)


def _robot_obs_to_batch(robot_obs: Mapping[str, Any], cfg, tokenizer) -> dict[str, torch.Tensor]:
    """``robot_obs`` (as GR00T's pipeline normalises it: ``images``/``video``, ``state``,
    ``language``/``prompt``, optional ``control_mode`` (default ``joint``) and
    ``language_memory``) -> the model's batch dict, built exactly as upstream's inference
    transform builds it (:func:`.preprocess.observation_request`): Qwen3.5 chat template, one
    ``<|vision_start|><|image_pad|>...<|vision_end|>`` per camera, the state as prompt text.
    ``state`` must already be normalised with the checkpoint's statistics (upstream normalises
    before the chat transform)."""
    images = robot_obs.get("images")
    if images is None:
        images = robot_obs.get("video")
    if images is None:
        raise ValueError("robot_obs must include 'images' or 'video'")
    images = [images] if not isinstance(images, (list, tuple)) else list(images)
    imgs = torch.stack([torch.as_tensor(_camera(im)) for im in images]).float()
    if imgs.max() > 1.5:  # uint8 pixels -> [0,1], as the Qwen2-VL image processor expects
        imgs = imgs / 255.0
    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):  # HWC -> CHW
        imgs = imgs.permute(0, 3, 1, 2)
    text = robot_obs.get("language") or robot_obs.get("prompt") or ""
    if tokenizer is None:
        raise ValueError(
            "InternVLA-A1.5 needs the Qwen3.5 tokenizer: set model_config.tokenizer in the stage "
            "config or $INTERNVLA_TOKENIZER to a directory holding tokenizer.json"
        )
    state = robot_obs.get("state")
    return pp.observation_request(
        cfg,
        imgs,
        str(text),
        tokenizer,
        state,
        control_mode=str(robot_obs.get("control_mode", "joint")),
        language_memory=str(robot_obs.get("language_memory", "") or ""),
    )


@functools.lru_cache(maxsize=4)
def load_tokenizer(path: str):
    """The Qwen3.5 tokenizer (vocabulary + chat template), loaded once per worker."""
    from transformers import Qwen3_5Tokenizer

    return Qwen3_5Tokenizer.from_pretrained(path)


def tokenizer_path(model_config: Mapping[str, Any] | None, model_path: str | None) -> str | None:
    """``model_config.tokenizer`` > ``$INTERNVLA_TOKENIZER`` > the model directory when it holds a
    ``tokenizer.json`` (the checkpoint itself ships none)."""
    path = (model_config or {}).get("tokenizer") or os.environ.get("INTERNVLA_TOKENIZER")
    if not path and model_path and os.path.isfile(os.path.join(model_path, "tokenizer.json")):
        path = model_path
    return path or None


def host_threads() -> int:
    """Torch threads for the host stages inside the worker (``INTERNVLA_HOST_THREADS``, default 8):
    the diffusion worker pins torch to one thread, which would also serialise the image resize."""
    return int(os.environ.get("INTERNVLA_HOST_THREADS", "8"))


@contextlib.contextmanager
def torch_threads(n: int):
    if n <= 0:
        yield
        return
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


class NeuronInternVLAA15Pipeline(nn.Module):
    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config or {}
        vlm_config = model_config.get("vlm_config") or os.environ.get("INTERNVLA_VLM_CONFIG")
        self.model_path = od_config.model
        self.device = "cpu"
        logger.info("Loading InternVLA-A1.5 from %s", self.model_path)
        policy = InternVLAA15.from_pretrained(
            self.model_path, dtype=torch.bfloat16, vlm_config=vlm_config
        )
        # NOT a registered submodule (object.__setattr__, bypassing nn.Module.__setattr__): the
        # engine's generic weight loader walks every registered submodule's named_parameters()
        # and tries to match them against the checkpoint's safetensors keys directly, which fails
        # ("not initialized from checkpoint") because the weights are already loaded under our own
        # tensor-name mapping. GR00T's pipeline does the same for the identical reason.
        object.__setattr__(self, "_policy", policy)
        self._runner: InternVLAA15Runner | None = (
            None  # built in to(), once the final device is known
        )
        self._tokenizer_path = tokenizer_path(model_config, self.model_path)
        if self._tokenizer_path is None:
            logger.warning(
                "InternVLA-A1.5: no tokenizer configured (model_config.tokenizer / "
                "$INTERNVLA_TOKENIZER); requests will be refused"
            )

    def _tokenizer(self):
        return load_tokenizer(self._tokenizer_path) if self._tokenizer_path else None

    # -- engine hooks ------------------------------------------------------------------------
    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()  # loaded directly by InternVLAA15.from_pretrained, like GR00T's Gr00tPolicy

    def load_weights(self, weights) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"{type(self).__name__}.load_weights received {len(consumed)} tensors; "
                "weights_sources=() should prevent this."
            )
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
            return torch.compile(
                mod,
                backend=backend,
                options={**base, "model_name": name, "compiler_args": list(COMPILER_ARGS)},
                **kw,
            )

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
            return DiffusionOutput(
                error=f"{type(self).__name__}.forward expects "
                "sampling_params.extra_args['robot_obs']."
            )
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(
                error=f"robot_obs must be a dict, got {type(robot_obs).__name__}."
            )

        t0 = time.perf_counter()
        with torch_threads(host_threads()):
            try:
                batch = _robot_obs_to_batch(robot_obs, self._policy.cfg, self._tokenizer())
            except (ValueError, NotImplementedError) as exc:
                return DiffusionOutput(error=f"{type(self).__name__}: {exc}")
            seed = getattr(req.sampling_params, "seed", None)
            noise = pp.initial_noise(
                self._policy.cfg, batch=1, seed=int(seed) if seed is not None else 0
            )
        t1 = time.perf_counter()
        runner = self._runner or InternVLAA15Runner(self._policy, self.device)
        actions = runner.sample_actions(batch, noise)
        timings = {
            "request_prep_ms": round(1000 * (t1 - t0), 2),
            **{k.replace("_s", "_ms"): round(1000 * v, 2) for k, v in runner.timings.items()},
        }
        logger.debug("InternVLA-A1.5 request timings: %s", timings)
        return DiffusionOutput(output={"actions": {"action": actions.cpu().float().numpy()}})
