"""LeRobot pi0 / pi0.5 / pi0.52 VLA policies on Neuron via the Omni entrypoint (offline).

One request = one robot observation (camera images + state + instruction) -> one action chunk
[chunk_size, action_dim], saved as JSON.

Usage:
    python examples/pi0/run.py --model lerobot/pi052_base
    python examples/pi0/run.py --model <local pi052 dir> --tokenizer <local paligemma-3b-pt-224 dir>
    python examples/pi0/run.py --model lerobot/pi0_base --stage-config examples/pi0/pi0_stage.yaml
    python examples/pi0/run.py --model lerobot/pi052_base --image base_0_rgb=frame.png --state state.json
    python examples/pi0/run.py --model lerobot/pi052_base --profile        # warm-up + timed request
    # real LIBERO-style input: two cameras, the third left out (masked), a fixed 12-token subtask
    python examples/pi0/run.py --model lerobot/pi052_base --profile \
        --image base_0_rgb=agentview.png --image left_wrist_0_rgb=wrist.png \
        --empty-camera right_wrist_0_rgb --state state32.json --task "put the white mug on the plate" \
        --model-config min_new_subtask_tokens=12 --model-config max_new_subtask_tokens=12

Cameras without an --image get a seeded random frame (a smoke test, not a meaningful policy
input); --state takes a JSON list (default: zeros).
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

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="pi0 / pi0.5 / pi0.52 on Neuron")
parser.add_argument("--model", default=os.environ.get("PI0_MODEL", "lerobot/pi052_base"))
parser.add_argument(
    "--stage-config", default=None, help="default: pi052_stage.yaml (pi0_stage.yaml for pi0)"
)
parser.add_argument("--task", default="pick up the red cube and place it in the bowl")
parser.add_argument(
    "--image",
    action="append",
    default=[],
    help="CAMERA=PATH, camera name as in the checkpoint (e.g. base_0_rgb); repeatable",
)
parser.add_argument(
    "--empty-camera",
    action="append",
    default=[],
    help="camera left out of the request (the processor fills it with -1 and masks it, as "
    "LeRobot does for a camera the robot lacks); repeatable",
)
parser.add_argument("--state", default=None, help="JSON file with the robot state vector")
parser.add_argument("--steps", type=int, default=10, help="flow-matching steps")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--tokenizer",
    default=None,
    help="PaliGemma tokenizer (repo id or local dir); overrides the stage config's",
)
parser.add_argument("--output", default="pi0_actions.json")
parser.add_argument(
    "--model-config",
    action="append",
    default=[],
    help="KEY=VALUE override of the stage's model_config (e.g. min_new_subtask_tokens=12); "
    "VALUE is parsed as YAML; repeatable",
)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--repeat", type=int, default=10, help="--profile: timed warm requests")
parser.add_argument(
    "--camera-format",
    choices=("bytes", "array"),
    default="bytes",
    help="cameras as raw bytes {data, shape, dtype} (default) or as ndarrays",
)
parser.add_argument(
    "--noise-seed",
    type=int,
    default=None,
    help="send a fixed initial noise drawn from this seed (reproducible actions across runs)",
)
parser.add_argument(
    "--save-request", default=None, help="save the observation (+ noise) to this .npz"
)
args = parser.parse_args()


def encode_camera(frame) -> dict:
    """HWC uint8 frame -> the raw-bytes request form (cheapest to serialize between processes)."""
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    return {"data": frame.tobytes(), "shape": list(frame.shape), "dtype": "uint8"}


def _policy_type(model: str) -> str:
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            return json.load(f).get("type", "pi05")
    return "pi0" if os.path.basename(model.rstrip("/")).startswith("pi0_") else "pi05"


def _camera_names(model: str) -> list[str]:
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            feats = json.load(f).get("input_features", {})
        names = [k.split(".")[-1] for k in feats if k.startswith("observation.images.")]
        if names:
            return names
    return ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


def _observation() -> dict:
    from PIL import Image

    # uint8 HWC: the one image format the pi0 and pi0.5 processors read identically (their float
    # conventions differ: pi0.5 expects [0, 1], pi0 assumes [-1, 1]).
    rng = np.random.default_rng(args.seed)
    given = dict(item.split("=", 1) for item in args.image)
    obs: dict = {"prompt": args.task}
    for cam in _camera_names(args.model):
        key = f"observation.images.{cam}"
        if cam in args.empty_camera:
            continue
        if cam in given:
            obs[key] = np.asarray(
                Image.open(given[cam]).convert("RGB").resize((224, 224)), dtype=np.uint8
            )
        else:
            obs[key] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        if args.camera_format == "bytes":
            obs[key] = encode_camera(obs[key])
    if args.state:
        with open(args.state) as f:
            obs["state"] = np.asarray(json.load(f), dtype=np.float32)
    else:  # pi0.5 has no default state; send zeros of the checkpoint's state width
        obs["state"] = np.zeros(_state_dim(args.model), dtype=np.float32)
    return obs


def _state_dim(model: str) -> int:
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            shape = json.load(f).get("input_features", {}).get("observation.state", {}).get("shape")
        if shape:
            return int(shape[0])
    return 32


def _actions(result) -> np.ndarray:
    item = result[0]
    for holder in (item, getattr(item, "request_output", None)):
        mm = getattr(holder, "multimodal_output", None) if holder is not None else None
        if isinstance(mm, dict) and "actions" in mm:
            return np.asarray(mm["actions"], dtype=np.float32)
    raise RuntimeError("no 'actions' in the request output")


def main() -> None:
    kind = _policy_type(args.model)
    stage_cfg = args.stage_config or os.path.join(
        HERE, "pi0_stage.yaml" if kind == "pi0" else "pi052_stage.yaml"
    )
    if args.tokenizer or args.model_config:
        import tempfile

        import yaml

        overrides = {}
        for item in args.model_config:
            key, value = item.split("=", 1)
            overrides[key] = yaml.safe_load(value)
        if args.tokenizer:
            overrides["tokenizer"] = args.tokenizer
        with open(stage_cfg) as f:
            cfg = yaml.safe_load(f)
        for stage in cfg["stage_args"]:
            stage["engine_args"].setdefault("model_config", {}).update(overrides)
        fd, stage_cfg = tempfile.mkstemp(suffix="_stage.yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f)
    timeout = int(
        os.environ.get("PI0_HANDSHAKE_TIMEOUT_S", "3600")
    )  # cold compiles exceed the 600 s default
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )

    obs = _observation()
    noise = None
    if args.noise_seed is not None:
        with open(os.path.join(args.model, "config.json")) as f:
            c = json.load(f)
        shape = (1, int(c.get("chunk_size", 50)), int(c.get("max_action_dim", 32)))
        noise = np.random.default_rng(args.noise_seed).standard_normal(shape).astype(np.float32)
    if args.save_request:
        arrays = {
            k: (
                np.frombuffer(v["data"], np.uint8).reshape(v["shape"]) if isinstance(v, dict) else v
            )
            for k, v in obs.items()
            if k != "prompt"
        }
        np.savez(
            args.save_request,
            prompt=args.task,
            **arrays,
            **({} if noise is None else {"noise": noise}),
        )

    def params():
        # client_send_ts: the pipeline reports the client -> worker transport time from it
        extra = {"robot_obs": obs, "client_send_ts": time.time()}
        if noise is not None:
            extra["noise"] = {
                "data": noise.tobytes(),
                "shape": list(noise.shape),
                "dtype": "float32",
            }
        return OmniDiffusionSamplingParams(
            num_inference_steps=args.steps, seed=args.seed, extra_args=extra
        )

    prompt = {"prompt": args.task}
    t0 = time.perf_counter()
    result = omni.generate(prompt, params())
    print(
        f"[run] first request (includes compile/load on a cold cache): {time.perf_counter() - t0:.2f}s"
    )
    if args.profile:
        times = []
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            result = omni.generate(prompt, params())
            times.append(time.perf_counter() - t0)
        times.sort()
        print(
            f"[profile] warm request ({len(times)}, cameras as {args.camera_format}): median "
            f"{times[len(times) // 2]:.4f}s, min {times[0]:.4f}s"
        )
    actions = _actions(result)
    with open(args.output, "w") as f:
        json.dump({"policy": kind, "task": args.task, "actions": actions.tolist()}, f)
    print(
        f"[run] {kind}: actions {list(actions.shape)} finite={bool(np.isfinite(actions).all())} -> {args.output}"
    )


if __name__ == "__main__":
    main()
