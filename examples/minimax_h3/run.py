"""FastH3 / MiniMax-H3 text-to-video+audio on Neuron via the Omni entrypoint (offline).

Usage:
    python examples/minimax_h3/run.py --model-path <FastH3 checkout> --output fasth3.mp4
    python examples/minimax_h3/run.py --height 384 --width 640 --num-frames 124 --profile
    python examples/minimax_h3/run.py --tp 2 --stage-config my_stage.yaml ...
    python examples/minimax_h3/run.py --stage-config examples/minimax_h3/minimax_h3_stage_tp8.yaml ...  # 8 cores

Writes an H.264 + AAC mp4 (PyAV) and a ``<output>.json`` with the request timings.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports

parser = argparse.ArgumentParser(description="MiniMax-H3 / FastH3 on Neuron")
parser.add_argument(
    "--model-path",
    default=os.environ.get(
        "MINIMAX_H3_WEIGHTS", "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
    ),
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--tp", type=int, default=None, help="override tensor_parallel_size (and the stage devices)"
)
parser.add_argument(
    "--cp", type=int, default=None, help="override the context-parallel degree (ring_degree)"
)
parser.add_argument(
    "--text-encoder",
    choices=("device", "host"),
    default=None,
    help="where the Qwen3-VL text encoder runs (default: the stage config's model_config, else device)",
)
parser.add_argument(
    "--extra-prompts",
    default=None,
    help="text file, one prompt per line: after the main request, run each one twice (new, then "
    "cached prompt) at the same geometry and record the text-encoder time",
)
parser.add_argument("--adaln", choices=("device", "host"), default=None)
parser.add_argument(
    "--prompt",
    default="A golden retriever runs through the surf at sunset, waves crashing "
    "around its paws, the camera tracking alongside.",
)
parser.add_argument("--height", type=int, default=384)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--num-frames", type=int, default=124)
parser.add_argument(
    "--steps", type=int, default=None, help="sigma grid points (FastH3: 5 = 4 forwards)"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="minimax_h3.mp4")
parser.add_argument(
    "--profile", action="store_true", help="run warm requests after the first and time them"
)
parser.add_argument(
    "--repeat", type=int, default=1, help="warm requests with --profile (the median is reported)"
)
parser.add_argument(
    "--prompt-embeds", default=None, help=".pt with 'prompt_embeds' (skips the text encoder)"
)
parser.add_argument(
    "--eager", action="store_true", help="no torch.compile (CPU-mode plumbing checks)"
)
args = parser.parse_args()

# vLLM's multiprocess Neuron workers refuse NEURON_RT_VISIBLE_CORES and read NEURON_VISIBLE_DEVICES instead; keep
# the same cores, expressed the way vLLM wants them.
_cores = os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
if _cores and "NEURON_VISIBLE_DEVICES" not in os.environ:
    ids = []
    for tok in _cores.split(","):
        a, _, b = tok.partition("-")
        ids.extend(range(int(a), int(b or a) + 1))
    os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(map(str, ids))

from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def _stage_config() -> str:
    path = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "minimax_h3_stage.yaml"
    )
    if (
        args.tp is None
        and args.cp is None
        and args.adaln is None
        and args.text_encoder is None
        and not args.eager
    ):
        return path
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    stage = cfg["stage_args"][0]
    pc = stage["engine_args"]["parallel_config"]
    if args.tp is not None:
        pc["tensor_parallel_size"] = args.tp
    if args.cp is not None:
        pc["ring_degree"] = args.cp
    if args.tp is not None or args.cp is not None:
        world = pc["tensor_parallel_size"] * pc.get("ring_degree", 1)
        stage["runtime"]["devices"] = ",".join(str(i) for i in range(world))  # no ranges here
    if args.eager:
        stage["engine_args"]["enforce_eager"] = True
    if args.adaln is not None:
        stage["engine_args"].setdefault("model_config", {})["adaln"] = args.adaln
    if args.text_encoder is not None:
        stage["engine_args"].setdefault("model_config", {})["text_encoder"] = args.text_encoder
    fd, out = tempfile.mkstemp(suffix=".yaml", prefix="minimax_h3_stage_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def save_mp4(video, audio, path: str, fps: int = 24, sample_rate: int = 32000) -> None:
    """video (1, 3, F, H, W): uint8 pixels, or float in [0, 1]; audio (1, 2, S) float."""
    import av
    import numpy as np
    import torch

    v = video[0].permute(1, 2, 3, 0)
    frames = (
        v.numpy()
        if v.dtype == torch.uint8
        else (v.float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    )
    with av.open(path, "w") as c:
        vs = c.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        aud = (
            audio[0].float().clamp(-1, 1).numpy().astype(np.float32) if audio is not None else None
        )
        astream = (
            c.add_stream("aac", rate=sample_rate, layout="stereo") if aud is not None else None
        )
        for f in frames:
            for p in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
        if astream is not None:
            af = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(aud), format="fltp", layout="stereo"
            )
            af.sample_rate = sample_rate
            for p in astream.encode(af):
                c.mux(p)
            for p in astream.encode():
                c.mux(p)


def _find_output(result):
    r = result[0]
    imgs = getattr(r, "images", None) or []
    mm = getattr(r, "multimodal_output", None) or {}
    if (
        imgs and hasattr(imgs[0], "shape") and "audio" in mm
    ):  # the engine files "video" under images
        stats = (getattr(r, "custom_output", None) or {}).get("minimax_h3_stats")
        return {"video": imgs[0], **mm, "stats": stats}
    for ro in (r, getattr(r, "request_output", None)):
        if ro is None:
            continue
        for name in ("images", "videos", "multimodal_output", "custom_output", "outputs"):
            val = getattr(ro, name, None)
            items = val if isinstance(val, list) else [val]
            if isinstance(val, dict) and "video" in val:
                return val
            for item in items:
                if isinstance(item, dict) and "video" in item:
                    return item
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("payload"), dict)
                    and "video" in item["payload"]
                ):
                    return item["payload"]
    desc = {
        n: (
            type(getattr(r, n)).__name__,
            list(getattr(r, n))[:5]
            if isinstance(getattr(r, n), dict)
            else [type(x).__name__ for x in getattr(r, n)][:3]
            if isinstance(getattr(r, n), list)
            else None,
        )
        for n in ("images", "multimodal_output", "custom_output", "outputs")
    }
    raise RuntimeError(f"no video in request output: {desc}")


def main() -> None:
    timeout = int(
        os.environ.get("MINIMAX_H3_HANDSHAKE_TIMEOUT_S", "7200")
    )  # cold compiles exceed the default
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    t0 = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=_stage_config(),
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )
    t_init = time.perf_counter() - t0
    kw = dict(
        height=args.height, width=args.width, num_frames=args.num_frames, seed=args.seed, fps=24
    )
    if args.steps:
        kw["num_inference_steps"] = args.steps
    if args.prompt_embeds:
        kw["extra_args"] = {"prompt_embeds_file": os.path.abspath(args.prompt_embeds)}
    params = OmniDiffusionSamplingParams(**kw)
    prompt = {"prompt": args.prompt}
    t0 = time.perf_counter()
    result = omni.generate(prompt, params)
    t_first = time.perf_counter() - t0
    # Includes any graph compile the compile cache does not already hold (all of them on an empty cache), plus the
    # NEFF loads and the text encoder's first use.
    print(f"[run] init {t_init:.1f}s, first request {t_first:.1f}s")
    cold_stats = _find_output(result).get("stats")
    t_warm = None
    warm_stats = None
    warm_runs = []
    if args.profile:
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params)
            warm_runs.append(
                {"request_s": time.perf_counter() - t0, "stats": _find_output(result).get("stats")}
            )
            print(f"[profile] warm request: {warm_runs[-1]['request_s']:.2f}s")
        mid = sorted(range(len(warm_runs)), key=lambda i: warm_runs[i]["request_s"])[
            len(warm_runs) // 2
        ]
        t_warm, warm_stats = (
            warm_runs[mid]["request_s"],
            warm_runs[mid]["stats"],
        )  # the median request
        if len(warm_runs) > 1:
            print(f"[profile] warm median of {len(warm_runs)}: {t_warm:.2f}s")
    extra = []
    if args.extra_prompts:
        with open(args.extra_prompts) as f:
            for p in (line.strip() for line in f):
                if not p:
                    continue
                for kind in ("new", "cached"):
                    t0 = time.perf_counter()
                    st = _find_output(omni.generate({"prompt": p}, params)).get("stats") or {}
                    extra.append(
                        {
                            "prompt": p,
                            "kind": kind,
                            "request_s": time.perf_counter() - t0,
                            "text_s": st.get("text_s"),
                            "num_text_tokens": st.get("num_text_tokens"),
                        }
                    )
                    print(
                        f"[text] {kind} {st.get('num_text_tokens')} tokens: text {st.get('text_s'):.3f}s"
                    )
    out = _find_output(result)
    if (
        os.environ.get("MINIMAX_H3_SAVE_AUDIO") and out.get("audio") is not None
    ):  # raw waveform (audio gates)
        import torch

        torch.save(out["audio"].float().cpu(), os.environ["MINIMAX_H3_SAVE_AUDIO"])
    if os.environ.get(
        "MINIMAX_H3_SAVE_VIDEO"
    ):  # raw uint8 frames (per-frame pixel gates, no H.264 in between)
        import torch

        v = out["video"]
        v = v if v.dtype == torch.uint8 else (v.float().clamp(0, 1) * 255).round().to(torch.uint8)
        torch.save(v.cpu().contiguous(), os.environ["MINIMAX_H3_SAVE_VIDEO"])
    save_mp4(
        out["video"],
        out.get("audio"),
        args.output,
        out.get("fps", 24),
        out.get("audio_sample_rate", 32000),
    )
    summary = {
        "output": args.output,
        "init_s": t_init,
        "first_request_s": t_first,
        "warm_request_s": t_warm,
        "video_shape": list(out["video"].shape),
        "audio_shape": list(out["audio"].shape) if out.get("audio") is not None else None,
        "cold_stats": cold_stats,
        "warm_stats": warm_stats,
        "warm_runs": warm_runs,
        "extra_prompts": extra,
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
