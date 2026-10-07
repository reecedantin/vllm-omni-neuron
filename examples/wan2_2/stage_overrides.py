# SPDX-License-Identifier: Apache-2.0
"""Stage-config overrides shared by the Wan2.2 example runners.

The runners take a stage YAML (devices, parallel layout, engine args) and let the command line
override the parts that change per run: which NeuronCores to use, the TP / CP / CFG / VAE-patch
parallel degrees and the expert-switch boundary. The overridden config is written next to the
output so a run can be reproduced from its artifacts.
"""

from __future__ import annotations

import argparse
import copy
import os

import yaml


def add_stage_override_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("stage overrides")
    g.add_argument(
        "--devices",
        default=None,
        help="NeuronCores for the stage as a comma list of indices into NEURON_VISIBLE_DEVICES "
        "(e.g. '0,1,2,3'). Defaults to every visible core when NEURON_VISIBLE_DEVICES or "
        "NEURON_RT_VISIBLE_CORES restricts this process, else to the stage YAML.",
    )
    g.add_argument("--tp", type=int, default=None, help="tensor_parallel_size")
    g.add_argument("--cp", type=int, default=None, help="context parallel (ring_degree)")
    g.add_argument("--cfg-parallel", type=int, default=None, help="cfg_parallel_size")
    g.add_argument("--vae-pp", type=int, default=None, help="vae_patch_parallel_size")
    g.add_argument(
        "--sp",
        choices=["on", "off"],
        default=None,
        help="TP sequence parallel (model_config.tp_sequence_parallel)",
    )
    g.add_argument("--no-vae-tiling", action="store_true", help="decode the VAE untiled")
    g.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default=None,
        help="engine dtype (float32 = CPU reference trajectory for parity)",
    )
    g.add_argument(
        "--dump-noise-pred",
        default=None,
        help="write the first DiT call's inputs + noise prediction to this .pt (rank 0), for "
        "teacher-forced parity (model_config.dump_noise_pred)",
    )
    g.add_argument(
        "--eager",
        action="store_true",
        help="enforce_eager: skip torch.compile (CPU reference runs with VLLM_NEURON_CPU_MODE=1)",
    )


def _pinned_cores() -> str | None:
    """Default stage ``devices`` for a process already restricted to a set of cores.

    vLLM's Neuron worker refuses to start while ``NEURON_RT_VISIBLE_CORES`` is set; it takes the
    visible cores from ``NEURON_VISIBLE_DEVICES`` and the stage ``devices`` as a comma list of
    logical indices into it. So: with ``NEURON_VISIBLE_DEVICES`` set, use all of its cores
    (``0,1,...``); with a ``NEURON_RT_VISIBLE_CORES`` pin, move the pin into
    ``NEURON_VISIBLE_DEVICES`` the same way.
    """
    if os.environ.get("VLLM_NEURON_CPU_MODE", "0") not in ("", "0"):
        # CPU mode never opens a core; keep any pin in place as a guard.
        return None
    pin = os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    os.environ.pop("NEURON_RT_NUM_CORES", None)
    if pin and not os.environ.get("NEURON_VISIBLE_DEVICES"):
        cores: list[str] = []
        for part in pin.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                cores += [str(c) for c in range(int(lo), int(hi) + 1)]
            elif part:
                cores.append(part)
        os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(cores)
    visible = os.environ.get("NEURON_VISIBLE_DEVICES")
    if not visible:
        return None
    return ",".join(str(i) for i in range(len(visible.split(","))))


def prepare_compile_env() -> None:
    """Make the process environment usable by the Lite (``neuron_native``) compiler.

    * The Lite compiler forwards ``NEURON_CC_FLAGS`` to ``neuronx-cc compile``, which rejects the
      torch-neuronx ``--cache_dir=`` flag; strip it and point Lite's own NEFF cache
      (``TORCH_NEURONX_NEFF_CACHE_DIR``, default ``/tmp/neff_cache``) at that directory instead.
    * CPU mode (``VLLM_NEURON_CPU_MODE=1``) must never compile or open a core, so Lite is off.
    """
    if os.environ.get("VLLM_NEURON_CPU_MODE", "0") not in ("", "0"):
        os.environ["VLLM_NEURON_LIBTORCH_NEURONX_LITE"] = "0"
        return
    flags = os.environ.get("NEURON_CC_FLAGS", "").split()
    cache = [f.split("=", 1)[1] for f in flags if f.startswith("--cache_dir=")]
    kept = [f for f in flags if not f.startswith("--cache_dir=")]
    if cache:
        os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", os.path.join(cache[-1], "lite"))
        if kept:
            os.environ["NEURON_CC_FLAGS"] = " ".join(kept)
        else:
            os.environ.pop("NEURON_CC_FLAGS", None)
    tmp = os.environ.get("TMPDIR")
    if tmp:
        os.environ.setdefault("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", os.path.join(tmp, "neff-local"))


def apply_stage_overrides(base_yaml: str, args: argparse.Namespace, out_path: str) -> str:
    """Write ``base_yaml`` with the command-line overrides applied to ``out_path``."""
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    stage = cfg["stage_args"][0]
    runtime = stage.setdefault("runtime", {})
    engine = stage.setdefault("engine_args", {})
    parallel = engine.setdefault("parallel_config", {})
    model_config = engine.setdefault("model_config", {}) or {}
    engine["model_config"] = model_config

    prepare_compile_env()
    pinned = _pinned_cores()
    devices = args.devices or pinned
    if devices:
        runtime["devices"] = str(devices)
    for flag, key in (
        ("tp", "tensor_parallel_size"),
        ("cp", "ring_degree"),
        ("cfg_parallel", "cfg_parallel_size"),
        ("vae_pp", "vae_patch_parallel_size"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            parallel[key] = value
    if getattr(args, "sp", None) is not None:
        model_config["tp_sequence_parallel"] = args.sp == "on"
    if getattr(args, "no_vae_tiling", False):
        engine["vae_use_tiling"] = False
    if getattr(args, "dtype", None):
        engine["dtype"] = args.dtype
    if getattr(args, "dump_noise_pred", None):
        model_config["dump_noise_pred"] = os.path.abspath(args.dump_noise_pred)
    if getattr(args, "eager", False):
        engine["enforce_eager"] = True

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def world_size(stage_yaml: str) -> int:
    with open(stage_yaml) as f:
        cfg = yaml.safe_load(f)
    p = cfg["stage_args"][0]["engine_args"].get("parallel_config", {})
    return (
        int(p.get("tensor_parallel_size", 1))
        * int(p.get("ring_degree", 1))
        * int(p.get("ulysses_degree", 1))
        * int(p.get("cfg_parallel_size", 1))
    )
