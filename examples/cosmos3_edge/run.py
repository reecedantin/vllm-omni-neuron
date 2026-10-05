"""NVIDIA Cosmos3-Edge on Neuron via the Omni entrypoint (offline).

Usage:
    python examples/cosmos3_edge/run.py --mode t2i --output edge_t2i.png
    python examples/cosmos3_edge/run.py --mode t2v --height 256 --width 256 --num-frames 49 --output edge.mp4
    python examples/cosmos3_edge/run.py --mode i2v --image frame0.png --num-frames 49 --output edge_i2v.mp4
    python examples/cosmos3_edge/run.py --mode t2i --profile          # warm-up + timed run
"""

from __future__ import annotations

import argparse
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

DEFAULTS = {  # upstream Cosmos3-Edge defaults (pipeline_cosmos3.py)
    "t2i": dict(height=640, width=640, num_frames=1, steps=50),
    "t2v": dict(height=480, width=832, num_frames=49, steps=35),
    "i2v": dict(height=480, width=832, num_frames=49, steps=35),
    # action modes (robot world-model): 17 frames = 1 conditioning + 16-step action chunk
    "policy": dict(height=256, width=256, num_frames=17, steps=30),
    "forward_dynamics": dict(height=256, width=256, num_frames=17, steps=30),
    "inverse_dynamics": dict(height=256, width=256, num_frames=17, steps=30),
}
ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")

parser = argparse.ArgumentParser(description="Cosmos3-Edge on Neuron")
parser.add_argument("--mode", choices=sorted(DEFAULTS), default="t2i")
parser.add_argument(
    "--model-path", default=os.environ.get("COSMOS3_EDGE_WEIGHTS", "nvidia/Cosmos3-Edge")
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--prompt",
    default="A red sports car parked on a wet city street at golden hour, photorealistic",
)
parser.add_argument("--negative-prompt", default=None)
parser.add_argument(
    "--image", default=None, help="conditioning image (i2v, policy, forward_dynamics)"
)
parser.add_argument("--video", default=None, help="conditioning video (inverse_dynamics)")
parser.add_argument("--domain", default="droid_lerobot", help="action embodiment domain")
parser.add_argument("--raw-action-dim", type=int, default=7)
parser.add_argument("--action-chunk", type=int, default=16)
parser.add_argument("--action-fps", type=float, default=12.0)
parser.add_argument(
    "--resolution", default="256", help="action-mode resolution class (256 / 480 / 720)"
)
parser.add_argument(
    "--actions",
    default=None,
    help="forward_dynamics: JSON file with [chunk x raw_action_dim] actions",
)
parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--num-frames", type=int)
parser.add_argument("--steps", type=int)
parser.add_argument(
    "--guidance-scale", type=float, default=None, help="default: the pipeline's per-mode default"
)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", default="cosmos3_edge_out")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--warm-repeats",
    type=int,
    default=1,
    help="with --profile: number of warm repeats of the request (reports each and the median)",
)
parser.add_argument(
    "--action-only",
    action="store_true",
    help="action modes: return actions only, skip the VAE video decode (extra_args action_only)",
)
args = parser.parse_args()

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")  # NeuronCore-v2 (inf2/trn1); trn2 uses 2


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cosmos3_edge_stage.yaml"
    )
    timeout = int(
        os.environ.get("COSMOS3_HANDSHAKE_TIMEOUT_S", "7200")
    )  # cold compiles exceed the 600 s default
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )

    d = DEFAULTS[args.mode]
    kw = dict(
        height=args.height or d["height"],
        width=args.width or d["width"],
        num_frames=args.num_frames or d["num_frames"],
        num_inference_steps=args.steps or d["steps"],
        seed=args.seed,
        fps=args.fps,
    )
    if args.guidance_scale is not None:
        kw["guidance_scale"] = args.guidance_scale
    if args.mode in ACTION_MODES:
        extra = {
            "action_mode": args.mode,
            "domain_name": args.domain,
            "raw_action_dim": args.raw_action_dim,
            "action_chunk_size": args.action_chunk,
            "resolution": args.resolution,
            "action_fps": args.action_fps,
        }
        if args.action_only:
            extra["action_only"] = True
        if args.mode == "forward_dynamics":
            import json

            if not args.actions:
                raise SystemExit("--actions is required for forward_dynamics")
            with open(args.actions) as f:
                extra["action"] = json.load(f)
        kw["extra_args"] = extra
    params = OmniDiffusionSamplingParams(**kw)

    prompt: dict = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    if args.mode == "t2i":
        prompt["modalities"] = ["image"]
    if args.mode in ("i2v", "policy", "forward_dynamics"):
        from PIL import Image

        if not args.image:
            raise SystemExit(f"--image is required for --mode {args.mode}")
        prompt["multi_modal_data"] = {"image": Image.open(args.image).convert("RGB")}
    if args.mode == "inverse_dynamics":
        import av
        from PIL import Image

        if not args.video:
            raise SystemExit("--video is required for --mode inverse_dynamics")
        frames = [
            Image.fromarray(f.to_ndarray(format="rgb24"))
            for f in av.open(args.video).decode(video=0)
        ]
        prompt["multi_modal_data"] = {"video": frames[: kw["num_frames"]]}

    print(f"[run] {args.mode}: {kw}")
    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    print(
        f"[run] first request (includes compile/load on a cold cache): {time.perf_counter() - t0:.2f}s"
    )
    if args.profile:
        warm = []
        for _ in range(max(1, args.warm_repeats)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params)
            warm.append(time.perf_counter() - t0)
            print(f"[profile] warm request: {warm[-1]:.2f}s")
        if len(warm) > 1:
            print(f"[profile] warm median of {len(warm)}: {sorted(warm)[len(warm) // 2]:.2f}s")
    if args.mode in ACTION_MODES:
        save_actions(result)
    save(result)


def save_actions(result) -> None:
    """Find the action tensor in the request output (post-processed action envelope) and dump it."""
    import json

    import numpy as np

    ro = result[0].request_output
    print(
        "[run] request_output fields:",
        {
            n: type(getattr(ro, n, None)).__name__
            for n in dir(ro)
            if not n.startswith("_") and not callable(getattr(ro, n, None))
        },
    )
    found = {}
    imgs = getattr(ro, "images", None) or []
    for i, item in enumerate(imgs):
        if isinstance(item, dict):
            found[f"images[{i}]"] = item.get("payload", item)
    for name in ("custom_output", "multimodal_output"):
        val = getattr(ro, name, None)
        if isinstance(val, dict):
            print(f"[run] {name} keys: {list(val)}")
    for name in dir(ro):
        if name.startswith("_"):
            continue
        val = getattr(ro, name, None)
        if isinstance(val, dict) and any("action" in str(k) for k in val):
            found[name] = val
    out = {}
    for src, d in found.items():
        for k, v in d.items():
            if "action" in str(k):
                arr = (
                    v.detach().cpu().float().numpy()
                    if hasattr(v, "detach")
                    else np.asarray(v, dtype=object)
                )
                out[f"{src}.{k}"] = arr.tolist() if arr.dtype != object else str(v)
    path = (args.output.rsplit(".", 1)[0]) + "_actions.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[run] actions -> {path}: {list(out)}")


def save(result) -> None:
    import numpy as np

    outs = result[0].request_output.images
    for i, out in enumerate(outs):
        if isinstance(out, dict):  # action envelope: {"video": ..., "actions": ...}
            print(f"[run] output {i} is a dict with keys {list(out)}")
            out = out.get("payload", out)
            out = out.get("video", out.get("image"))
            if out is None:
                continue
        arr = out.detach().cpu().float().numpy() if hasattr(out, "detach") else np.asarray(out)
        if hasattr(out, "save"):  # PIL image
            path = args.output if args.output.endswith(".png") else f"{args.output}_{i}.png"
            out.save(path)
            print(f"[run] saved {path}")
            continue
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim < 3:
            print(f"[run] output {i}: non-image value of shape {arr.shape}, skipped")
            continue
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 3, 0) if arr.ndim == 4 else (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 0.5 + 0.5 if arr.min() < 0 else arr, 0, 1)
        if arr.ndim == 3 or (arr.ndim == 4 and arr.shape[0] == 1):
            from PIL import Image

            img = arr[0] if arr.ndim == 4 else arr
            img = (img * 255).round().astype("uint8") if img.dtype != np.uint8 else img
            path = args.output if args.output.endswith(".png") else f"{args.output}_{i}.png"
            Image.fromarray(img).save(path)
        else:
            from diffusers.utils import export_to_video

            path = args.output if args.output.endswith(".mp4") else f"{args.output}_{i}.mp4"
            export_to_video(
                list(arr.astype(np.float32) if arr.dtype != np.uint8 else arr / 255.0),
                path,
                fps=args.fps,
            )
        print(f"[run] saved {path}")


if __name__ == "__main__":
    main()
