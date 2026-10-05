# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for NVIDIA Alpamayo 1.5 and Alpamayo 2 Super on Neuron.

Alpamayo has no upstream vLLM-Omni pipeline to subclass: this pipeline owns
the whole serving contract directly. ``sampling_params.extra_args["robot_obs"]`` carries the
VLM-tokenized observation (``input_ids``/``attention_mask``/``pixel_values``/``image_grid_thw``,
as produced by the checkpoint's own Qwen3-VL processor) plus the ego-motion history
(``ego_history_xyz``/``ego_history_rot``) that the UnicycleAccelCurvatureActionSpace needs to
integrate the flow-matching output into world-frame waypoints. The request's seed (materialised
by the runner into ``sampling_params.generator``) draws the Euler ODE's initial noise, so
repeating a request with the same seed reproduces its trajectory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from .model import NeuronAlpamayo1_5

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # vLLM-Omni's generic fallback (single-element config.json "architectures" list, no
        # special-cased model family) uses architectures[0] VERBATIM as model_class_name --
        # it does NOT append "Pipeline". The real checkpoint ships architectures=["Alpamayo1_5"].
        "model_arch": "Alpamayo1_5",
        "class_name": "NeuronAlpamayo1_5Pipeline",
    },
    {
        # nvidia/Alpamayo2-Super ships architectures=["Alpamayo2Super"]; same pipeline, the model
        # reads the variant from config.json
        "model_arch": "Alpamayo2Super",
        "class_name": "NeuronAlpamayo1_5Pipeline",
    },
]


class NeuronAlpamayo1_5Pipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.model_path = od_config.model
        self.device = "cpu"  # host-side generator/seed device; the model graphs live on Neuron
        model = NeuronAlpamayo1_5.from_pretrained(self.model_path, dtype=torch.bfloat16)
        # not a registered submodule: load_weights() above already loaded the real checkpoint,
        # so the engine's own weight loader (which would see every parameter as uninitialised,
        # since weights_sources=()) must not walk into it.
        object.__setattr__(self, "_model", model)
        logger.info(
            "Alpamayo 1.5 on Neuron: %s, n_waypoints=%d", self.model_path, model.n_waypoints
        )

    # -- engine hooks --------------------------------------------------------------------
    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self._model.compile(
            backend, options, **{k: v for k, v in kwargs.items() if k == "fullgraph"}
        )
        return self

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"NeuronAlpamayo1_5Pipeline.load_weights received {len(consumed)} weight tensors; "
                "weights_sources=() should prevent this. Weights are loaded by NeuronAlpamayo1_5.from_pretrained."
            )
        return set()

    def _dummy_output(self) -> DiffusionOutput:
        n_wp = self._model.n_waypoints
        zeros_xyz = torch.zeros((1, n_wp, 3), dtype=torch.float32).numpy()
        zeros_rot = torch.eye(3, dtype=torch.float32).expand(1, n_wp, 3, 3).numpy()
        return DiffusionOutput(
            output=_envelope(
                {
                    "actions": torch.zeros((1, n_wp, 2)).numpy(),
                    "pred_xyz": zeros_xyz,
                    "pred_rot": zeros_rot,
                    "cot": [""],
                }
            )
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        del kwargs
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                return self._dummy_output()
            return DiffusionOutput(
                error="NeuronAlpamayo1_5Pipeline.forward expects sampling_params.extra_args['robot_obs']."
            )
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(
                error=f"robot_obs must be a dict, got {type(robot_obs).__name__}."
            )

        try:
            inputs = _tokenized_inputs(robot_obs)
        except KeyError as exc:
            return DiffusionOutput(error=f"robot_obs missing required key: {exc}")

        generator = _request_generator(req.sampling_params, self.device)
        ego_xyz = _as_tensor(robot_obs.get("ego_history_xyz"))
        ego_rot = _as_tensor(robot_obs.get("ego_history_rot"))
        out = self._model.get_action(
            inputs, generator=generator, ego_history_xyz=ego_xyz, ego_history_rot=ego_rot
        )

        result: dict[str, Any] = {
            "actions": out["actions"].numpy(),
            "generated": out["generated"].numpy(),
            "offset": out["offset"],
            "timing_ms": out["timing_ms"],
        }
        if "pred_xyz" in out:
            result["pred_xyz"] = out["pred_xyz"].numpy()
            result["pred_rot"] = out["pred_rot"].numpy()
        decoded = self._model.tokenizer.decode(out["generated"][: out["n_cot_tokens"]].tolist())
        result["cot"] = [decoded]
        digest_dir = os.environ.get("ALPAMAYO_RANK_DIGEST_DIR")
        if digest_dir:
            self._n_requests = getattr(self, "_n_requests", 0) + 1
            _write_rank_digest(digest_dir, self._n_requests, result)
        return DiffusionOutput(output=_envelope(result))


def _write_rank_digest(out_dir: str, request_index: int, result: Mapping[str, Any]) -> None:
    """Gate hook (``ALPAMAYO_RANK_DIGEST_DIR``): every TP rank writes the SHA-256 of its own
    host copy of the request's outputs, so a gate can check that all ranks computed the same
    trajectory and tokens, not only the rank-0 result the engine returns."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    digest = {
        key: hashlib.sha256(np.ascontiguousarray(result[key]).tobytes()).hexdigest()
        for key in ("actions", "pred_xyz", "pred_rot", "generated")
        if key in result
    }
    digest["cot"] = result["cot"][0]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"req{request_index:03d}_rank{rank:02d}.json")
    with open(path, "w") as f:
        json.dump(digest, f)


def _envelope(result: dict[str, Any]) -> dict[str, Any]:
    """vLLM-Omni surfaces a diffusion output as an action result ONLY through
    ``output["actions"]`` (``output_formatter.normalize_diffusion_postprocess_output`` ->
    ``OmniRequestOutput.multimodal_output["actions"]``); any other top-level dict is treated as an
    image payload. So the whole Alpamayo result --
    trajectory, raw action, CoC text -- rides inside ``actions``."""
    return {"actions": result}


def _as_tensor(x: Any) -> torch.Tensor | None:
    """Request payloads cross the engine boundary serialized: arrays arrive as numpy (or nested
    lists), never as torch tensors, so normalise them before the model touches ``.long()`` etc."""
    if x is None or isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _tokenized_inputs(robot_obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Pull the VLM-tokenized observation out of ``robot_obs``, either nested under
    ``tokenized_data`` (the shape the checkpoint's own reference script produces) or inlined
    at the top level, as torch tensors."""
    src = robot_obs.get("tokenized_data", robot_obs)
    return {
        key: _as_tensor(src[key])
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }


def _request_generator(
    sampling_params: OmniDiffusionSamplingParams, device: str
) -> torch.Generator | None:
    """The request's noise generator (the runner materialises ``seed`` into a CPU generator on
    Neuron), else one seeded from ``sampling_params.seed``, else ``None`` (global RNG). Alpamayo
    draws exactly one noise tensor per request, so a list of several generators is ignored with a warning."""
    generator = sampling_params.generator
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, torch.Generator):
        return generator
    if generator is not None:
        received = (
            f"a list of {len(generator)} generators"
            if isinstance(generator, list)
            else type(generator).__name__
        )
        logger.warning(
            "Alpamayo expects a single torch.Generator per request but got %s; ignoring it. The "
            "initial noise falls back to sampling_params.seed=%s, or to the global RNG when that "
            "is None.",
            received,
            sampling_params.seed,
        )
    if sampling_params.seed is not None:
        return torch.Generator(device=device).manual_seed(int(sampling_params.seed))
    return None


# Variant-neutral name: the same pipeline serves Alpamayo 1.5 and Alpamayo 2 Super.
NeuronAlpamayoPipeline = NeuronAlpamayo1_5Pipeline
