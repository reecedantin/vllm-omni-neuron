"""Z-Image / Z-Image-Turbo on Neuron via the Omni entrypoint (offline).

Usage:
    python examples/z_image/run.py --model-path $WEIGHTS/z-image-turbo --guidance-scale 0 --steps 9 \
        --output turbo_t2i.png
    python examples/z_image/run.py --model-path $WEIGHTS/z-image --guidance-scale 4 --steps 50 \
        --cfg-normalize 1.0 --output base_t2i.png --profile
"""

from __future__ import annotations

import argparse
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports

parser = argparse.ArgumentParser(description="Z-Image on Neuron")
parser.add_argument(
    "--model-path", default=os.environ.get("Z_IMAGE_WEIGHTS", "Tongyi-MAI/Z-Image-Turbo")
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--prompt",
    default="A red sports car parked on a wet city street at golden hour, photorealistic",
)
parser.add_argument("--negative-prompt", default=None)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=9)
parser.add_argument("--guidance-scale", type=float, default=0.0)
parser.add_argument("--cfg-normalize", type=float, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", default="z_image_out.png")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--warm-runs",
    type=int,
    default=1,
    help="with --profile: warm requests to time (median printed)",
)
args = parser.parse_args()


def main() -> None:
    # multiprocessing spawn re-imports this module in each worker; everything that launches a
    # process (Omni's engine, which forks diffusion workers) must be guarded behind __main__.
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Trn2 runs LNC=2. At LNC=1 the 1024 px block graph's scratchpad overflows at NEFF load.
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    if args.profile:
        os.environ["Z_IMAGE_PROFILE"] = "1"

    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "z_image_stage.yaml"
    )
    timeout = int(
        os.environ.get("Z_IMAGE_HANDSHAKE_TIMEOUT_S", "3600")
    )  # cold compiles exceed the 600s default
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )

    kw = dict(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if args.cfg_normalize is not None:
        kw["extra_args"] = {"cfg_truncation": 1.0}
        kw["cfg_normalize"] = args.cfg_normalize
    params = OmniDiffusionSamplingParams(**kw)

    prompt: dict = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt

    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    print(
        f"[run] first request (includes compile/load on a cold cache): {time.perf_counter() - t0:.2f}s"
    )
    if args.profile:
        warm = []
        for _ in range(max(1, args.warm_runs)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params)
            warm.append(time.perf_counter() - t0)
            print(f"[profile] warm request: {warm[-1]:.2f}s")
        print(f"[profile] warm median of {len(warm)}: {sorted(warm)[len(warm) // 2]:.2f}s")

    outs = result[0].request_output.images
    for i, out in enumerate(outs):
        path = args.output if args.output.endswith(".png") else f"{args.output}_{i}.png"
        if hasattr(out, "save"):  # PIL image
            out.save(path)
        else:
            import numpy as np
            from PIL import Image

            arr = out.detach().cpu().float().numpy() if hasattr(out, "detach") else np.asarray(out)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 0.5 + 0.5 if arr.min() < 0 else arr, 0, 1)
                arr = (arr * 255).round().astype("uint8")
            Image.fromarray(arr).save(path)
        print(f"[run] saved {path}")


if __name__ == "__main__":
    main()
