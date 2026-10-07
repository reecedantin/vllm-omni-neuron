# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-dev text-to-image on Neuron via the Omni entrypoint.

Usage:
    python examples/flux2/run.py --model-path <FLUX.2-dev dir>                 # 1024x1024, 50 steps, 32 cores
    python examples/flux2/run.py --model-path <dir> --stage-config examples/flux2/flux2_stage_tp8.yaml  # 8 cores
    python examples/flux2/run.py --model-path <dir> --height 512 --width 512 --steps 28
    python examples/flux2/run.py --model-path <dir> --profile                  # + one warm timed run
    python examples/flux2/run.py --model-path <dir> --context-parallel-size 2   # TP8 x CP2, 16 cores

Prints a one-line JSON summary (timings, output path) as the last line.
"""

import argparse
import json
import os
import tempfile
import time
from dataclasses import replace

import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

parser = argparse.ArgumentParser(description="FLUX.2-dev text-to-image on Neuron")
parser.add_argument("--model-path", default="black-forest-labs/FLUX.2-dev")
parser.add_argument(
    "--stage-config", default=None, help="stage YAML (default: flux2_stage.yaml next to this file)"
)
parser.add_argument(
    "--devices",
    default=None,
    help="NeuronCores for the stage, e.g. '0-7' (default: the stage YAML's devices)",
)
parser.add_argument("--tensor-parallel-size", type=int, default=None)
parser.add_argument(
    "--context-parallel-size",
    type=int,
    default=None,
    help="DiT context parallelism (stage `ring_degree`): the image and text tokens are split "
    "over this many TP groups; the stage then spans TP x CP NeuronCores",
)
parser.add_argument(
    "--repeat", type=int, default=1, help="with --profile: number of timed warm runs (median)"
)
parser.add_argument(
    "--prompt",
    default="A cozy reading nook by a rain-streaked window, warm lamp light, "
    "a cat asleep on a stack of books, photorealistic",
)
parser.add_argument(
    "--negative-prompt", default=None, help="enables true CFG (FLUX.2-dev normally runs without)"
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--guidance-scale", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="flux2_output.png")
parser.add_argument(
    "--profile", action="store_true", help="after the first (compile) run, time one warm run"
)
parser.add_argument(
    "--parity-latents",
    default=None,
    help="M1: request output_type=latent and dump the denoised latents to this .pt "
    "for offline comparison against a diffusers CPU fp32 run (parity_ref.py)",
)
parser.add_argument("--platform-target", choices=["trn2", "trn3"], default=None)
args = parser.parse_args()

STAGE_PATH = args.stage_config or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flux2_stage.yaml"
)


def _stage_world() -> int:
    """TP x CP ranks of the stage (CLI overrides first, then the stage YAML)."""
    with open(STAGE_PATH) as f:
        pc = yaml.safe_load(f)["stage_args"][0]["engine_args"]["parallel_config"]
    return (args.tensor_parallel_size or int(pc.get("tensor_parallel_size", 1))) * (
        args.context_parallel_size or int(pc.get("ring_degree", 1))
    )


# Neuron env for the trn2 native backend (same profile Wan2.2 uses); set before Omni spawns workers.
# Host threads are split across the TP worker processes (an inherited OMP_NUM_THREADS would
# oversubscribe the host once per rank).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.pop(_var, None)
# The DiT launches ~20 block graphs per call; the runtime's default per-core hardware queue
# depth rejects that with "Execution Queue Full". 63 is the runtime's maximum.
os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "63")
env_profiles.apply(
    replace(
        env_profiles.WAN22_T2V,
        NEURON_PLATFORM_TARGET_OVERRIDE=args.platform_target,
        TORCH_NEURONX_DEBUG_DIR=os.environ.get("TORCH_NEURONX_DEBUG_DIR", "./compile_dir"),
    ),
    env_profiles.thread_limits(_stage_world()),
)


def _stage_config() -> str:
    path = STAGE_PATH
    # vLLM-Omni places multi-process workers with NEURON_VISIBLE_DEVICES + the stage `devices:`
    # field (a comma list of logical indices into it) and refuses NEURON_RT_VISIBLE_CORES.
    # A core pin inherited from the environment is converted into NEURON_VISIBLE_DEVICES.
    pinned = os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    if pinned and "NEURON_VISIBLE_DEVICES" not in os.environ:
        lo_hi = [int(x) for x in pinned.split("-")] if "-" in pinned else None
        cores = (
            list(range(lo_hi[0], lo_hi[1] + 1)) if lo_hi else [int(x) for x in pinned.split(",")]
        )
        os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(map(str, cores))
    os.environ.pop("NEURON_RT_NUM_CORES", None)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    stage = cfg["stage_args"][0]
    pc = stage["engine_args"]["parallel_config"]
    tp = args.tensor_parallel_size or int(pc.get("tensor_parallel_size", 1))
    cp = args.context_parallel_size or int(pc.get("ring_degree", 1))
    devices = args.devices
    if devices is None and "NEURON_VISIBLE_DEVICES" in os.environ:
        # the stage world is TP x CP ranks: the first TP x CP visible cores
        visible = os.environ["NEURON_VISIBLE_DEVICES"].split(",")
        if len(visible) < tp * cp:
            raise SystemExit(
                f"stage needs TP x CP = {tp * cp} NeuronCores, {len(visible)} visible "
                f"(pick a smaller stage config, e.g. flux2_stage_tp8.yaml)"
            )
        devices = ",".join(str(i) for i in range(tp * cp))
    if devices is not None and "-" in devices:  # ranges are not accepted: expand to a comma list
        lo, hi = (int(x) for x in devices.split("-"))
        devices = ",".join(str(i) for i in range(lo, hi + 1))
    if (
        devices is None
        and args.tensor_parallel_size is None
        and args.context_parallel_size is None
        and not args.parity_latents
    ):
        return path
    if devices is not None:
        stage["runtime"]["devices"] = devices
    if args.tensor_parallel_size is not None:
        stage["engine_args"]["parallel_config"]["tensor_parallel_size"] = args.tensor_parallel_size
    if args.context_parallel_size is not None:
        stage["engine_args"]["parallel_config"]["ring_degree"] = args.context_parallel_size
    if args.parity_latents:
        # The post-process func keys on engine_args.output_type: 'latent' makes it the identity,
        # so the engine returns the pipeline's raw denoised latents instead of PIL images.
        stage["engine_args"]["output_type"] = "latent"
    fd, out = tempfile.mkstemp(prefix="flux2_stage_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def main() -> None:
    stage_cfg = _stage_config()
    timeout = int(
        os.environ.get("FLUX2_HANDSHAKE_TIMEOUT_S", "7200")
    )  # cold compiles exceed the 600 s default
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    t0 = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )
    init_s = time.perf_counter() - t0

    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if args.parity_latents:
        params.output_type = "latent"
    prompt = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt

    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    first_s = time.perf_counter() - t0
    print(f"[run] first request (compiles on a cold cache): {first_s:.2f}s")
    warm_s = None
    if args.profile:
        times = []
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params)
            times.append(time.perf_counter() - t0)
            print(f"[profile] warm request: {times[-1]:.2f}s")
        warm_s = sorted(times)[len(times) // 2]

    out = result[0].request_output.images[0]
    if args.parity_latents:
        import torch

        lat = (
            out.detach().to("cpu", torch.float32)
            if hasattr(out, "detach")
            else torch.as_tensor(out)
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.parity_latents)) or ".", exist_ok=True)
        torch.save(
            {
                "latents": lat,
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            },
            args.parity_latents,
        )
        print(f"[parity] device latents {tuple(lat.shape)} -> {args.parity_latents}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "parity_latents": os.path.abspath(args.parity_latents),
                    "shape": list(lat.shape),
                    "init_s": round(init_s, 2),
                    "first_s": round(first_s, 2),
                    "warm_s": None if warm_s is None else round(warm_s, 2),
                }
            )
        )
        return

    image = out
    if not hasattr(image, "save"):
        raise SystemExit(f"unexpected output type {type(image)}")
    image.save(args.output)
    print(f"[run] saved {args.output}")
    print(
        json.dumps(
            {
                "ok": True,
                "output": os.path.abspath(args.output),
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "init_s": round(init_s, 2),
                "first_s": round(first_s, 2),
                "warm_s": None if warm_s is None else round(warm_s, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
