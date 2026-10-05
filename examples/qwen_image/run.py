"""Qwen-Image 2.1 text-to-image on Neuron via the Omni entrypoint (offline).

Usage:
    python examples/qwen_image/run.py --output qwen21.png
    python examples/qwen_image/run.py --height 1328 --width 1328 --steps 40 --output qwen21.png
    python examples/qwen_image/run.py --stage-config examples/qwen_image/qwen_image21_stage_tp1.yaml --profile
"""

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parser = argparse.ArgumentParser(description="Qwen-Image 2.1 on Neuron")
parser.add_argument(
    "--model-path", default=os.environ.get("QWEN_IMAGE21_WEIGHTS", "Qwen/Qwen-Image-2.1")
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--prompt",
    default="A cozy bookshop window on a rainy evening, warm light, a hand-lettered sign "
    'that reads "Open Late"',
)
parser.add_argument("--negative-prompt", default=None)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument(
    "--true-cfg-scale", type=float, default=1.0, help="2.1 is sampled without guidance by default"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="qwen_image21.png")
parser.add_argument("--profile", action="store_true", help="warm-up request, then a timed one")
parser.add_argument(
    "--devices",
    default=None,
    help='NeuronCore ids for the stage, e.g. "36-39" (overrides the stage config\'s devices)',
)
parser.add_argument("--summary", default=None, help="write a JSON timing summary here")
args = parser.parse_args()


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qwen_image21_stage.yaml"
    )
    if args.devices:  # write a copy of the stage config with this core list
        import tempfile

        import yaml

        with open(stage_cfg) as f:
            cfg = yaml.safe_load(f)
        for stage in cfg["stage_args"]:
            stage["runtime"]["devices"] = args.devices
        fd, stage_cfg = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f)
    timeout = int(
        os.environ.get("QWEN_IMAGE_HANDSHAKE_TIMEOUT_S", "7200")
    )  # cold compiles exceed the default
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
        seed=args.seed,
        true_cfg_scale=args.true_cfg_scale,
    )
    prompt: dict = {"prompt": args.prompt, "modalities": ["image"]}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    first_s = time.perf_counter() - t0
    print(f"[run] first request (includes compile on a cold cache): {first_s:.2f}s")
    warm_s = None
    if args.profile:
        t0 = time.perf_counter()
        result = omni.generate(prompt, params)
        warm_s = time.perf_counter() - t0
        print(f"[profile] warm request: {warm_s:.2f}s")
    img = result[0].request_output.images[0]
    img.save(args.output)
    print(f"[run] saved {args.output} {img.size} {img.mode}")
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(
                {
                    "init_s": init_s,
                    "first_s": first_s,
                    "warm_s": warm_s,
                    "hw": [args.height, args.width],
                    "steps": args.steps,
                    "output": args.output,
                },
                f,
            )


if __name__ == "__main__":
    main()
