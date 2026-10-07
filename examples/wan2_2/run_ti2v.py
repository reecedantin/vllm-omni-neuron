# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 TI2V-5B and its DMD2 few-step distillation (FastWan2.2-TI2V-5B) on Neuron.

Usage:
    # TI2V-5B text-to-video (50 steps, CFG 5)
    python examples/wan2_2/run_ti2v.py --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers
    # TI2V-5B image-to-video: the image conditions the first latent frame
    python examples/wan2_2/run_ti2v.py --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers --image in.jpg
    # FastWan2.2-TI2V-5B: 3 DMD2 steps, no CFG
    python examples/wan2_2/run_ti2v.py --model-path FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
    # pick cores / layout without editing the YAML
    python examples/wan2_2/run_ti2v.py --model-path ... --devices 0-3 --tp 4 --dev

The stage config is chosen from the checkpoint's ``model_index.json`` and the inputs
(``WanDMDPipeline`` -> ``wan22_dmd2_stage.yaml``; ``--image`` -> ``wan22_ti2v_i2v_stage.yaml``;
otherwise ``wan22_ti2v_stage.yaml``) unless ``--stage-config`` is given.
"""

import argparse
import json
import os
import sys
import time

import torch

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_overrides import (  # noqa: E402
    add_stage_override_args,
    apply_stage_overrides,
    world_size,
)

torch.nn.functional.gelu = torch.ops.aten.gelu.default

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Wan2.2 TI2V-5B / FastWan DMD2 on Neuron")
parser.add_argument("--model-path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--stage-config", type=str, default=None)
parser.add_argument("--dev", action="store_true", help="small: 256x448, 17 frames")
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="denoise steps (ignored by DMD2)")
parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG (ignored by DMD2)")
parser.add_argument("--image", type=str, default=None, help="first-frame image (TI2V I2V)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--latents-in",
    type=str,
    default=None,
    help="path to a .pt initial-noise tensor; injected so CPU and device denoise from identical "
    "noise (the correct parity method — same-seed noise differs across devices).",
)
parser.add_argument(
    "--save-init-latents",
    type=str,
    default=None,
    help="draw the initial noise for this shape/seed on CPU, save it to this path, and exit.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="A fluffy orange cat walking gracefully across a sunny garden path, high quality",
)
parser.add_argument("--negative-prompt", type=str, default=None)
parser.add_argument("--output", type=str, default="wan22_ti2v.mp4")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--save-frames", action="store_true", help="also save frames as .npy")
parser.add_argument(
    "--latents",
    action="store_true",
    help="return the final latents (no VAE decode) and save them as <output>.latents.pt",
)
parser.add_argument("--repeat", type=int, default=1, help="generate N times (warm timing)")
add_stage_override_args(parser)
args = parser.parse_args()


def _pipeline_class(model_path: str) -> str | None:
    index = os.path.join(model_path, "model_index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f).get("_class_name")
    return "WanDMDPipeline" if "FastWan" in model_path else None


def main():
    if args.save_init_latents:
        import torch as _t

        # TI2V-5B: 48 latent channels, 16x spatial, 4x temporal.
        h = args.height or 256
        w = args.width or 448
        nf = args.num_frames or 17
        shape = (1, 48, (nf - 1) // 4 + 1, h // 16, w // 16)
        g = _t.Generator(device="cpu").manual_seed(args.seed)
        _t.save(_t.randn(shape, generator=g, dtype=_t.float32), args.save_init_latents)
        print("SAVED_INIT " + args.save_init_latents + " " + str(list(shape)))
        return
    is_dmd = _pipeline_class(args.model_path) == "WanDMDPipeline"
    if is_dmd:
        default_yaml = "wan22_dmd2_stage.yaml"
    elif args.image:
        default_yaml = "wan22_ti2v_i2v_stage.yaml"
    else:
        default_yaml = "wan22_ti2v_stage.yaml"
    base_yaml = args.stage_config or os.path.join(HERE, default_yaml)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    stage_yaml = apply_stage_overrides(
        base_yaml, args, os.path.join(out_dir, os.path.basename(args.output) + ".stage.yaml")
    )
    env_profiles.apply(env_profiles.WAN22_T2V, env_profiles.thread_limits(world_size(stage_yaml)))
    # the Wan env profile points this at ./compile_dir (CWD-relative); keep artifacts with the run
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = os.path.join(out_dir, "compile_dir")

    cold_timeout = int(os.environ.get("WAN_HANDSHAKE_TIMEOUT_S", "3600"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as _sdp

        if getattr(_sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0) < cold_timeout:
            _sdp._HANDSHAKE_POLL_TIMEOUT_S = cold_timeout
    except Exception as e:  # pragma: no cover - older vllm-omni
        print(f"[init] could not raise handshake timeout ({e!r})")

    with open(stage_yaml) as f:
        import yaml

        model_config = dict(
            yaml.safe_load(f)["stage_args"][0]["engine_args"].get("model_config") or {}
        )

    t0 = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_yaml,
        stage_init_timeout=cold_timeout,
        init_timeout=cold_timeout,
        model_config=model_config,
    )
    init_s = time.perf_counter() - t0

    if args.dev:
        height, width, num_frames = 256, 448, 17
    else:
        height, width, num_frames = 704, 1280, 121
    height = args.height or height
    width = args.width or width
    num_frames = args.num_frames or num_frames
    steps = args.steps or (3 if is_dmd else 50)

    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=1.0 if is_dmd else args.guidance_scale,
        seed=args.seed,
    )
    if args.latents_in:
        import torch as _t

        params.latents = _t.load(args.latents_in)
    if args.latents:
        params.output_type = "latent"
    prompt: dict = {"prompt": args.prompt}
    if args.negative_prompt and not is_dmd:
        prompt["negative_prompt"] = args.negative_prompt
    if args.image:
        import PIL.Image

        prompt["multi_modal_data"] = {"image": PIL.Image.open(args.image).convert("RGB")}

    mode = "DMD2" if is_dmd else ("I2V" if args.image else "T2V")
    print(f"Generating TI2V-5B {mode}: {num_frames}x{height}x{width}, {steps} steps")
    latencies = []
    result = None
    for _ in range(max(1, args.repeat)):
        t = time.perf_counter()
        result = omni.generate(prompt, params)
        latencies.append(time.perf_counter() - t)

    import numpy as np
    from diffusers.utils import export_to_video

    if args.latents:
        lat = result[0].request_output.images[0]
        lat = lat if isinstance(lat, torch.Tensor) else torch.as_tensor(np.asarray(lat))
        lat = lat.detach().cpu().float()
        torch.save(lat, args.output + ".latents.pt")
        summary = {
            "mode": mode,
            "model": args.model_path,
            "latents_shape": list(lat.shape),
            "steps": steps,
            "init_s": round(init_s, 1),
            "latency_s": [round(x, 2) for x in latencies],
            "finite": bool(torch.isfinite(lat).all()),
            "mean": round(float(lat.mean()), 5),
            "std": round(float(lat.std()), 5),
            "output": os.path.abspath(args.output + ".latents.pt"),
        }
        print("SUMMARY " + json.dumps(summary))
        return

    video = result[0].request_output.images[0]
    video = video.detach().cpu().numpy() if hasattr(video, "detach") else np.asarray(video)
    if video.ndim == 5:
        video = video[0]
    if video.shape[0] in (3, 4) and video.shape[-1] not in (3, 4):
        video = np.transpose(video, (1, 2, 3, 0))
    if np.issubdtype(video.dtype, np.floating):
        if float(video.min()) < 0.0:
            video = np.clip(video, -1.0, 1.0) * 0.5 + 0.5
        video = np.clip(video, 0.0, 1.0)
    video = video.astype(np.float32)
    export_to_video(list(video), args.output, fps=args.fps)
    if args.save_frames:
        np.save(args.output + ".npy", video)
    summary = {
        "mode": mode,
        "model": args.model_path,
        "frames": int(video.shape[0]),
        "height": int(video.shape[1]),
        "width": int(video.shape[2]),
        "steps": steps,
        "init_s": round(init_s, 1),
        "latency_s": [round(x, 2) for x in latencies],
        "finite": bool(np.isfinite(video).all()),
        "mean": round(float(video.mean()), 4),
        "std": round(float(video.std()), 4),
        "output": os.path.abspath(args.output),
    }
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
