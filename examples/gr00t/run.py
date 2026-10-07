"""NVIDIA GR00T N1.7 on Neuron via the Omni entrypoint (offline).

Sends one DROID-layout observation (two cameras x two frames, end-effector/gripper/joint
state, a language instruction) and saves the decoded action chunk.

Usage:
    GR00T_VLM_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct \
    python examples/gr00t/run.py --model-path /path/to/GR00T-N1.7-3B --output actions.npz
    python examples/gr00t/run.py ... --image-dir frames/ --repeat 20   # real frames, warm timing

Without ``--image-dir`` the camera frames are seeded random noise (a plumbing check, not a task).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

CAMERAS = ("exterior_image_1_left", "wrist_image_left")

parser = argparse.ArgumentParser(description="GR00T N1.7 on Neuron")
parser.add_argument("--model-path", default=os.environ.get("GR00T_WEIGHTS", "nvidia/GR00T-N1.7-3B"))
parser.add_argument("--stage-config", default=None)
parser.add_argument("--prompt", default="pick up the red cube and put it in the bowl")
parser.add_argument("--image-dir", default=None, help="<camera>_<t>.png for t in 0,1 (180x320 RGB)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repeat", type=int, default=1, help="extra timed requests after the first")
parser.add_argument("--output", default="gr00t_actions.npz")
parser.add_argument(
    "--reference",
    default=None,
    help="send the observation saved by reference.py (e.g. a LIBERO frame) instead of DROID noise",
)
parser.add_argument(
    "--timing",
    action="store_true",
    help="report per-stage medians (data processing / backbone / action head / decode); set "
    "GR00T_STAGE_TIMING=1 in the worker to split backbone from action head",
)
args = parser.parse_args()


def observation() -> dict:
    rng = np.random.default_rng(args.seed)
    video = {}
    for cam in CAMERAS:
        if args.image_dir:
            from PIL import Image

            frames = [
                np.asarray(
                    Image.open(os.path.join(args.image_dir, f"{cam}_{t}.png")).convert("RGB")
                )
                for t in (0, 1)
            ]
            video[cam] = np.stack(frames)[None]
        else:
            video[cam] = rng.integers(0, 255, (1, 2, 180, 320, 3), dtype=np.uint8)
    return {
        "video": video,
        "state": {
            "eef_9d": (rng.normal(size=(1, 1, 9)) * 0.1).astype(np.float32),
            "gripper_position": rng.uniform(size=(1, 1, 1)).astype(np.float32),
            "joint_position": (rng.normal(size=(1, 1, 7)) * 0.3).astype(np.float32),
        },
        "language": {"annotation.language.language_instruction": [[args.prompt]]},
    }


def actions_of(result) -> dict:
    out = result[0]
    for holder in (out, getattr(out, "request_output", None)):
        mm = getattr(holder, "multimodal_output", None)
        if isinstance(mm, dict) and "actions" in mm:
            return {k: np.asarray(v, dtype=np.float32) for k, v in mm["actions"].items()}
    raise RuntimeError(f"no actions in the result: {out!r}"[:2000])


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gr00t_stage.yaml"
    )
    timeout = int(
        os.environ.get("GR00T_INIT_TIMEOUT_S", "3600")
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
    print(f"[run] engine up in {time.perf_counter() - t0:.1f}s")
    obs = observation()
    if args.reference:
        import torch

        obs = torch.load(args.reference, weights_only=False)["obs"]
    extra = {"robot_obs": obs, "return_timing": args.timing}

    def request():
        params = OmniDiffusionSamplingParams(seed=args.seed, extra_args=extra)
        return omni.generate({"prompt": args.prompt}, params)

    t0 = time.perf_counter()
    actions = actions_of(request())
    first = time.perf_counter() - t0
    lat, stages = [], []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        again = actions_of(request())
        lat.append(time.perf_counter() - t0)
        if "timing_ms" in again:
            stages.append(again.pop("timing_ms"))
    actions.pop("timing_ms", None)
    deterministic = (
        all(np.array_equal(actions[k], again[k]) for k in actions) if args.repeat else None
    )
    np.savez(args.output, **actions)
    summary = {
        "first_request_s": round(first, 3),
        "warm_request_ms": round(1000 * float(np.median(lat)), 2) if lat else None,  # median
        "warm_request_ms_mean": round(1000 * float(np.mean(lat)), 2) if lat else None,
        "warm_request_ms_min": round(1000 * float(np.min(lat)), 2) if lat else None,
        "deterministic_same_seed": deterministic,
        "actions": {k: list(v.shape) for k, v in actions.items()},
        "output": os.path.abspath(args.output),
    }
    if (
        stages
    ):  # medians over the warm requests, ms; e2e = the sum, as NVIDIA's benchmark reports it
        med = np.median(np.stack(stages).reshape(len(stages), -1), axis=0)
        names = ("data_processing", "backbone", "action_head", "decode", "in_worker_total")
        summary["stages_ms"] = {n: round(float(v), 2) for n, v in zip(names, med)}
        summary["stages_ms"]["e2e_sum"] = round(float(med[0] + med[1] + med[2]), 2)
    print("[run]", json.dumps(summary))
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(
            {**summary, "first_steps": {k: v[0, :2].tolist() for k, v in actions.items()}},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
