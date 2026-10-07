# SPDX-License-Identifier: Apache-2.0
"""Neuron GR00T N1.7 pipeline: upstream vLLM-Omni's ``Gr00tN1d7Pipeline`` with the policy's
model swapped for :class:`~.model.NeuronGr00tModel`.

Everything around the model stays upstream's: the OpenPI observation normalisation, the
GR00T processor (image transforms, state normalisation, chat template, Qwen3-VL tokenisation),
the collator, action decoding / unnormalisation, ``policy_server_config`` validation and the
seed -> generator plumbing. The host runs that; only the three model graphs (vision, text,
action head) run on the NeuronCore.

Offline operation: upstream resolves the backbone's processor from the Hub
(``Qwen/Qwen3-VL-2B-Instruct``). Point ``model_config.vlm_processor`` (or
``$GR00T_VLM_PROCESSOR``) at a local Qwen3-VL-2B-Instruct checkout; the backbone
architecture itself is resolved by :mod:`.config` without the Hub.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from vllm_omni.diffusion.models.gr00t import policy as _up_policy
from vllm_omni.diffusion.models.gr00t.modeling import processing_gr00t_n1d7 as _up_processing
from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline

from .host_fastpath import host_threads, torch_threads
from .host_fastpath import install as install_host_fastpath
from .model import NeuronGr00tModel

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # vLLM-Omni maps config.json model_type "Gr00tN1d7" to this pipeline class name
        "model_arch": "Gr00tN1d7Pipeline",
        "class_name": "NeuronGr00tN1d7Pipeline",
    },
]


class _PolicyModelAdapter:
    """What ``Gr00tPolicy`` expects of ``self.model``: ``get_action(inputs=..., options=...)``,
    ``eval()``, ``to()``, ``config``. Noise: ``options["generator"]`` (vLLM-Omni main), else
    ``$GR00T_NOISE_SEED`` (vLLM-Omni 0.24), else the global CPU RNG -- each drawn on the host in
    bf16, as upstream draws it on a CPU device."""

    def __init__(self, model: NeuronGr00tModel):
        self.model = model
        self.config = model.cfg
        self.generator: torch.Generator | None = None  # set per request by the pipeline
        self.noise: torch.Tensor | None = (
            None  # explicit initial noise for parity checks (per request)
        )
        self.last_pred: torch.Tensor | None = None  # normalized action_pred of the last call
        self.t_enter = self.t_exit = 0.0  # model call boundaries (per-stage timing)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None, **_):
        gen = (options or {}).get("generator") or self.generator
        if gen is None and os.environ.get("GR00T_NOISE_SEED") is not None:
            gen = torch.Generator().manual_seed(int(os.environ["GR00T_NOISE_SEED"]))
        if gen is not None and gen.device.type != "cpu":
            gen = torch.Generator().manual_seed(int(gen.initial_seed()))
        self.t_enter = time.perf_counter()
        out = self.model.get_action(inputs, noise=self.noise, generator=gen)
        self.t_exit = time.perf_counter()
        self.last_pred = out["action_pred"]
        return out


class _AutoModelShim:
    model: NeuronGr00tModel | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _PolicyModelAdapter(cls.model)


@contextlib.contextmanager
def _neuron_policy(model: NeuronGr00tModel, vlm_processor: str | None):
    """Build an upstream ``Gr00tPolicy`` whose model is ours and whose VLM processor is local."""
    _AutoModelShim.model = model
    saved = (_up_policy.AutoModel, _up_processing.QWEN3_VL_2B_PROCESSOR)
    _up_policy.AutoModel = _AutoModelShim
    if vlm_processor:
        _up_processing.QWEN3_VL_2B_PROCESSOR = vlm_processor
    try:
        yield
    finally:
        _up_policy.AutoModel, _up_processing.QWEN3_VL_2B_PROCESSOR = saved
        _AutoModelShim.model = None


class NeuronGr00tN1d7Pipeline(Gr00tN1d7Pipeline):
    def __init__(self, *, od_config, prefix: str = ""):
        model_config = od_config.model_config or {}
        model = NeuronGr00tModel.from_pretrained(od_config.model, dtype=torch.bfloat16)
        tp_size, tp_rank, tp_group = _tp_state()
        tp_backbone = bool(
            model_config.get("tp_backbone", os.environ.get("GR00T_TP_BACKBONE", "0") == "1")
        )
        if tp_size > 1:  # action head always; vision tower + text decoder too with tp_backbone
            model.head.shard_tp(tp_rank, tp_size, tp_group)
            if tp_backbone:
                model.vision.shard_tp(tp_rank, tp_size, tp_group)
                model.text.shard_tp(tp_rank, tp_size, tp_group)
        self._rank_info = _rank_layout(model, tp_rank, tp_size, tp_backbone)
        logger.info("GR00T rank %d/%d layout: %s", tp_rank, tp_size, self._rank_info["layout"])
        self._rank_report = _RankReport.from_env(self._rank_info)
        vlm = model_config.get("vlm_processor") or os.environ.get("GR00T_VLM_PROCESSOR")
        if vlm and not Path(vlm).is_dir():
            raise FileNotFoundError(f"vlm_processor {vlm!r} is not a directory")
        with _neuron_policy(model, vlm):
            super().__init__(od_config=od_config, prefix=prefix)
        install_host_fastpath(self.policy)
        self._host_threads = int(model_config.get("host_threads", host_threads()))
        # not a registered submodule: the weights are loaded here, not by the engine's loader
        # (which would otherwise report every parameter as uninitialised) -- as upstream's policy
        object.__setattr__(self, "_model", model)
        self.device = "cpu"  # host-side generator/seed device; the model graphs live on Neuron
        self._adapter: _PolicyModelAdapter = self.policy.model
        logger.info(
            "GR00T on Neuron: %s, embodiment=%s, horizon=%d, TP=%d (backbone %s), host threads=%d",
            od_config.model,
            self.embodiment_tag,
            self._model.head.action_horizon,
            tp_size,
            "sharded" if tp_size > 1 and tp_backbone else "replicated",
            self._host_threads,
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

    @torch.inference_mode()
    def forward(self, req, **kwargs):
        # upstream 0.24's policy does not forward options to the model; hand it the generator here
        sp = req.sampling_params
        extra = getattr(sp, "extra_args", None) or {}
        self._adapter.generator = _request_generator(sp)
        self._adapter.noise = _as_tensor(extra.get("initial_noise"))
        t0 = time.perf_counter()
        try:
            # the worker runs torch on one thread; the processor's image ops scale to ~8
            with torch_threads(self._host_threads):
                out = super().forward(req, **kwargs)
            t_end = time.perf_counter()
            if (
                extra.get("return_action_pred")
                and self._adapter.last_pred is not None
                and out.output
            ):
                # parity checks: the normalized model output, before action decoding
                out.output["actions"]["action_pred"] = self._adapter.last_pred.float().numpy()
            if extra.get("return_timing") and out.output:
                out.output["actions"]["timing_ms"] = self._timing_ms(t0, t_end)
            if self._rank_report is not None and self._adapter.last_pred is not None:
                self._rank_report.request(self._adapter.last_pred)
            return out
        finally:
            self._adapter.generator = self._adapter.noise = self._adapter.last_pred = None
            st = self._model.stats
            if "device_s" in st:
                st["request_s"] = time.perf_counter() - t0
                logger.debug(
                    "GR00T request %.1f ms (model prep %.1f, device %.1f)",
                    1e3 * st["request_s"],
                    1e3 * st["prep_s"],
                    1e3 * st["device_s"],
                )

    def _timing_ms(self, t0: float, t_end: float):
        """Per-request stages in ms, in NVIDIA's deployment-benchmark split: [data processing (processor +
        collator), backbone (host prep + vision + text), action head, action decoding, request total].
        Backbone/head split needs ``GR00T_STAGE_TIMING=1`` (one extra sync); otherwise backbone holds the
        whole model call and head is 0."""
        import numpy as np

        st, a = self._model.stats, self._adapter
        if "backbone_s" in st:
            bb, head = st["backbone_s"], st["head_s"]
        else:
            bb, head = a.t_exit - a.t_enter, 0.0
        return 1e3 * np.array(
            [a.t_enter - t0, bb, head, t_end - a.t_exit, t_end - t0], dtype=np.float32
        )


def _as_tensor(v) -> torch.Tensor | None:
    """``extra_args["initial_noise"]``: ``{"data": bytes, "shape": [...]}`` fp32, or a nested list."""
    if v is None:
        return None
    import numpy as np

    if isinstance(v, dict) and "data" in v:
        return torch.from_numpy(
            np.frombuffer(v["data"], dtype=np.float32).reshape(v["shape"]).copy()
        )
    return torch.as_tensor(np.asarray(v, dtype=np.float32))


def _tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) outside an initialised vLLM TP group.

    Under TP>1 the group's partition is registered with the Neuron compiler's mesh registry so the
    in-graph all-reduces legalize (``register_replica_groups``)."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError):
        return 1, 0, None
    if size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=1)
    return size, rank, group


def _rank_layout(model: NeuronGr00tModel, rank: int, size: int, tp_backbone: bool) -> dict:
    """This rank's weight layout, read off the model after any TP sharding (on the host)."""
    from .layers import pretranspose_status

    parts = {}
    for name in ("vision", "text", "head"):
        good, total = pretranspose_status(getattr(model, name))
        parts[name] = {"pretransposed": good, "linears": total}
    layout = ", ".join(
        f"{n} {p['pretransposed']}/{p['linears']} linears pretransposed" for n, p in parts.items()
    )
    sharded = "head + backbone" if size > 1 and tp_backbone else ("head" if size > 1 else "none")
    return {
        "rank": rank,
        "tp_size": size,
        "sharded": sharded,
        "parts": parts,
        "layout": f"TP shards: {sharded}; {layout}",
    }


class _RankReport:
    """Per-rank evidence for TP checks (``$GR00T_RANK_REPORT`` = a directory): every rank writes
    ``layout_<r>.json`` (its pretransposed/total linears per part after sharding), then for each
    request ``req<i>/rank_digest_<r>.json`` -- the shared :func:`write_rank_digest` digest of the
    normalized action chunk it computed. The gate compares them with
    :func:`compare_rank_digest_files`."""

    def __init__(self, root: Path, info: dict):
        self.root, self.rank, self.n = root, info["rank"], 0
        with open(root / f"layout_{self.rank}.json", "w") as f:
            json.dump({k: v for k, v in info.items() if k != "layout"}, f)

    @classmethod
    def from_env(cls, info: dict) -> _RankReport | None:
        d = os.environ.get("GR00T_RANK_REPORT")
        if not d:
            return None
        Path(d).mkdir(parents=True, exist_ok=True)
        return cls(Path(d), info)

    def request(self, pred: torch.Tensor) -> None:
        from vllm_omni_neuron.testing import write_rank_digest

        write_rank_digest(str(self.root / f"req{self.n:04d}"), self.rank, {"action_pred": pred})
        self.n += 1


def _request_generator(sampling_params) -> torch.Generator | None:
    """The request's noise generator (the runner materialises ``seed`` into a CPU generator on
    Neuron), else one seeded from ``sampling_params.seed``, else ``None`` (global RNG)."""
    gen = getattr(sampling_params, "generator", None)
    if isinstance(gen, list) and len(gen) == 1:
        gen = gen[0]
    if isinstance(gen, torch.Generator):
        return gen
    seed = getattr(sampling_params, "seed", None)
    return torch.Generator().manual_seed(int(seed)) if seed is not None else None
