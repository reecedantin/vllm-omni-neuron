"""InternVLA-A1.5 on Neuron via the Omni entrypoint (offline, served policy).

Sends one observation (camera views + a language instruction + robot state) and saves the
decoded action chunk. Mirrors ``examples/gr00t/run.py``'s shape for the GR00T policy.

Usage:
    python examples/internvla/serve.py --model-path /path/to/InternVLA-A1.5-base \
        --vlm-config /path/to/qwen3.5-config --tokenizer /path/to/qwen3.5-tokenizer --output actions.npz
    python examples/internvla/serve.py ... --image-dir frames/ --repeat 5   # real frames, warm timing
    python examples/internvla/serve.py ... --obs-npz episode.npz --n-images 3 --vary --repeat 30

Without ``--image-dir`` / ``--obs-npz`` the camera frames are seeded random noise (a plumbing
check, not a task). ``--obs-npz`` replays recorded steps; with ``--vary`` every timed request is
a new observation (new pixels and a new state in the prompt), as on a robot.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parser = argparse.ArgumentParser(description="InternVLA-A1.5 on Neuron")
parser.add_argument(
    "--model-path",
    default=os.environ.get("INTERNVLA_WEIGHTS", "InternRobotics/InternVLA-A1.5-base"),
)
parser.add_argument("--vlm-config", default=os.environ.get("INTERNVLA_VLM_CONFIG"))
parser.add_argument(
    "--tokenizer",
    default=os.environ.get("INTERNVLA_TOKENIZER"),
    help="Qwen3.5 tokenizer directory (tokenizer.json + tokenizer_config.json)",
)
parser.add_argument("--stage-config", default=None)
parser.add_argument("--prompt", default="pick up the red cube and put it in the bowl")
parser.add_argument(
    "--image-dir", default=None, help="image_<i>.png, i=0..n_images-1 (224x224 RGB)"
)
parser.add_argument(
    "--obs-npz",
    default=None,
    help="recorded observations: images [T,V,H,W,3] uint8, state [T,S] (normalised), task",
)
parser.add_argument(
    "--obs-index", type=int, default=0, help="--obs-npz step of the first (saved) request"
)
parser.add_argument(
    "--vary",
    action="store_true",
    help="with --obs-npz, each timed request sends the next recorded step (new pixels and state)",
)
parser.add_argument("--n-images", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repeat", type=int, default=1, help="extra timed requests after the first")
parser.add_argument(
    "--camera-format",
    default="raw",
    choices=["raw", "array"],
    help="raw: frames as {'data': bytes, 'shape'} (cheap to serialize); array: numpy arrays",
)
parser.add_argument("--output", default="internvla_actions.npz")
args = parser.parse_args()


def _frames(images):
    if args.camera_format == "raw":
        return [
            {"data": np.ascontiguousarray(im, dtype=np.uint8).tobytes(), "shape": list(im.shape)}
            for im in images
        ]
    return list(images)


@functools.lru_cache(maxsize=1)
def _recorded(path: str) -> dict:
    return dict(np.load(path))  # in memory: an NpzFile re-reads an array on every access


def observation(step: int | None = None) -> dict:
    if args.obs_npz:
        rec = _recorded(args.obs_npz)
        t = (args.obs_index if step is None else step) % rec["images"].shape[0]
        return {
            "images": _frames(rec["images"][t, : args.n_images]),
            "state": rec["state"][t],
            "prompt": str(rec["task"]),
        }
    rng = np.random.default_rng(args.seed)
    if args.image_dir:
        from PIL import Image

        images = [
            np.asarray(Image.open(os.path.join(args.image_dir, f"image_{i}.png")).convert("RGB"))
            for i in range(args.n_images)
        ]
    else:
        images = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(args.n_images)]
    return {
        "images": _frames(images),
        "state": (rng.normal(size=(32,)) * 0.1).astype(np.float32),
        "prompt": args.prompt,
    }


def actions_of(result) -> np.ndarray:
    out = result[0]
    for holder in (out, getattr(out, "request_output", None)):
        mm = getattr(holder, "multimodal_output", None)
        if isinstance(mm, dict) and "actions" in mm:
            return np.asarray(mm["actions"]["action"], dtype=np.float32)
    raise RuntimeError(f"no actions in the result: {out!r}"[:2000])


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "internvla_stage.yaml"
    )
    timeout = int(
        os.environ.get("INTERNVLA_INIT_TIMEOUT_S", "3600")
    )  # cold compiles exceed the default
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    if args.vlm_config:
        os.environ.setdefault("INTERNVLA_VLM_CONFIG", args.vlm_config)
    if args.tokenizer:
        os.environ.setdefault("INTERNVLA_TOKENIZER", args.tokenizer)
    t0 = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )
    print(f"[run] engine up in {time.perf_counter() - t0:.1f}s")
    obs = observation()

    def request(o):
        params = OmniDiffusionSamplingParams(seed=args.seed, extra_args={"robot_obs": o})
        return omni.generate({"prompt": o["prompt"]}, params)

    t0 = time.perf_counter()
    actions = actions_of(request(obs))
    first = time.perf_counter() - t0
    lat = []
    for k in range(args.repeat):
        o = observation(args.obs_index + 1 + k) if args.vary else obs
        t0 = time.perf_counter()
        actions_of(request(o))
        lat.append(time.perf_counter() - t0)
    again = actions_of(request(obs)) if args.repeat else None
    deterministic = bool(np.array_equal(actions, again)) if args.repeat else None
    np.savez(args.output, action=actions)
    summary = {
        "n_images": args.n_images,
        "varying_observations": bool(args.vary and args.obs_npz),
        "first_request_s": round(first, 3),
        "warm_request_ms": round(1000 * float(np.mean(lat)), 2) if lat else None,
        "warm_request_ms_median": round(1000 * float(np.median(lat)), 2) if lat else None,
        "warm_request_ms_min": round(1000 * float(np.min(lat)), 2) if lat else None,
        "deterministic_same_seed": deterministic,
        "action_shape": list(actions.shape),
        "output": os.path.abspath(args.output),
    }
    print("[run]", json.dumps(summary))
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump({**summary, "first_steps": actions[0, :3].tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
