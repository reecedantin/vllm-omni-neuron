"""HunyuanVideo-1.5 text-to-video on Neuron via the Omni entrypoint (offline).

Usage:
    HV15_BLOCKS_PER_GRAPH=6 python examples/hunyuanvideo15/run.py --num-frames 25 --output hv15.mp4
    python examples/hunyuanvideo15/run.py --model-path <local 480p_t2v dir> --height 480 --width 848 \
        --num-frames 25 --steps 50 --profile          # warm-up + timed second request

``--stage-config`` selects the layout (default ``hunyuanvideo15_stage.yaml``, TP=8 x CFG-parallel 2 on 16 cores).
"""

from __future__ import annotations

import argparse
import os
import time

parser = argparse.ArgumentParser(description="HunyuanVideo-1.5 T2V on Neuron")
parser.add_argument(
    "--model-path",
    default=os.environ.get(
        "HV15_WEIGHTS", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    ),
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--prompt", default="A golden retriever runs across a sunlit meadow, slow motion, cinematic."
)
parser.add_argument("--negative-prompt", default=None)
parser.add_argument(
    "--image", default=None, help="image-to-video: first-frame image (I2V checkpoints)"
)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=848)
parser.add_argument("--num-frames", type=int, default=121)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument(
    "--guidance-scale", type=float, default=None, help="default: the pipeline's (6.0)"
)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="hunyuanvideo15.mp4")
parser.add_argument("--profile", action="store_true", help="run a second, warm request and time it")
parser.add_argument(
    "--warm-runs",
    type=int,
    default=1,
    help="with --profile: warm requests to time (median printed)",
)
args = parser.parse_args()

# vLLM's Neuron worker selects cores through NEURON_VISIBLE_DEVICES (+ the stage `devices:` field) and
# refuses NEURON_RT_VISIBLE_CORES. Translate a pinned core range into the equivalent visible-devices
# baseline so the stage's logical devices 0..N-1 map onto exactly the same cores.
_pinned = os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
if _pinned and "NEURON_VISIBLE_DEVICES" not in os.environ:
    lo, _, hi = _pinned.partition("-")
    os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(
        str(c) for c in range(int(lo), int(hi or lo) + 1)
    )

import vllm_omni_neuron.bootstrap  # noqa: E402,F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hunyuanvideo15_stage.yaml"
    )
    timeout = int(
        os.environ.get("HV15_HANDSHAKE_TIMEOUT_S", "14400")
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
    print(f"[run] engine up in {time.perf_counter() - t0:.1f}s")

    kw = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        seed=args.seed,
        fps=args.fps,
    )
    if args.guidance_scale is not None:
        kw["guidance_scale"] = args.guidance_scale
    params = OmniDiffusionSamplingParams(**kw)
    prompt: dict = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    if args.image is not None:
        from PIL import Image

        prompt["multi_modal_data"] = {"image": Image.open(args.image).convert("RGB")}

    print(f"[run] {'i2v' if args.image else 't2v'}: {kw}")
    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    print(
        f"[run] first request (includes compile on a cold cache): {time.perf_counter() - t0:.2f}s"
    )
    if args.profile:
        warm = []
        for _ in range(max(args.warm_runs, 1)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params)
            warm.append(time.perf_counter() - t0)
            print(f"[profile] warm request: {warm[-1]:.2f}s")
        if len(warm) > 1:
            print(f"[profile] warm median of {len(warm)}: {sorted(warm)[len(warm) // 2]:.2f}s")
    save(result)


def save(result) -> None:
    import numpy as np

    frames = result[0].request_output.images
    arr = []
    for f in frames:
        a = np.asarray(f)
        if hasattr(f, "detach"):
            a = f.detach().cpu().float().numpy()
        arr.append(a)
    video = np.stack(arr) if len(arr) > 1 else np.asarray(arr[0])
    if video.ndim == 5:
        video = video[0]
    if video.dtype != np.uint8:
        print(
            f"[run] frames float stats: mean={float(np.nanmean(video)):.4f} std={float(np.nanstd(video)):.4f} "
            f"nan={int(np.isnan(video).sum())}"
        )
        video = (np.clip(np.nan_to_num(video), 0, 1) * 255).round().astype(np.uint8)
    print(f"[run] {video.shape[0]} frames {video.shape[1:]}")
    import av

    with av.open(args.output, mode="w") as c:
        s = c.add_stream("libx264", rate=args.fps)
        s.height, s.width, s.pix_fmt = video.shape[1], video.shape[2], "yuv420p"
        for fr in video:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    np.save(os.path.splitext(args.output)[0] + "_frames.npy", video[:: max(1, len(video) // 8)])
    print(f"[run] saved {args.output}")


if __name__ == "__main__":
    main()
