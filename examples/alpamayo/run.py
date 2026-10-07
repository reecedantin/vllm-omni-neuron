"""NVIDIA Alpamayo 1.5 / Alpamayo 2 Super on Neuron via the Omni entrypoint (offline).

Sends one real tokenized observation (saved by ``reference_1_5.py`` / ``parity_ref_super.py`` from
the upstream CPU reference run: VLM input_ids/attention_mask/pixel_values/image_grid_thw plus the
ego-motion history) through the plugin pipeline and saves the decoded trajectory. The checkpoint's
``config.json`` selects the variant.

Usage:
    python examples/alpamayo/run.py --model-path /path/to/alpamayo-1.5-10b \
        --reference /path/to/ref_bf16_cpu.pt --stage-config alpamayo_stage_trn2.yaml \
        --output alpamayo_actions.npz
    python examples/alpamayo/run.py --model-path /path/to/alpamayo2-super \
        --reference /path/to/parity_super_fp32.pt \
        --stage-config alpamayo2_super_stage_trn2.yaml --output alpamayo2_super.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np
import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parser = argparse.ArgumentParser(description="Alpamayo 1.5 / 2 Super on Neuron")
parser.add_argument(
    "--model-path",
    default=os.environ.get(
        "ALPAMAYO_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", ""), "alpamayo-1.5-10b")
    ),
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--reference",
    required=True,
    help="a .pt with model_inputs (reference_1_5.py / parity_ref_super.py)",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repeat", type=int, default=1, help="extra timed requests after the first")
parser.add_argument("--output", default="alpamayo_actions.npz")
args = parser.parse_args()


def observation() -> dict:
    ref = torch.load(args.reference, weights_only=False)
    mi = ref["model_inputs"]
    tok = mi["tokenized_data"]
    return {
        "input_ids": tok["input_ids"].numpy(),
        "attention_mask": tok["attention_mask"].numpy(),
        "pixel_values": tok["pixel_values"].numpy(),
        "image_grid_thw": tok["image_grid_thw"].numpy(),
        "ego_history_xyz": mi["ego_history_xyz"].numpy(),
        "ego_history_rot": mi["ego_history_rot"].numpy(),
    }


def result_output(result) -> dict:
    out = result[0]
    for holder in (out, getattr(out, "request_output", None)):
        mm = getattr(holder, "multimodal_output", None)
        if isinstance(mm, dict) and isinstance(mm.get("actions"), dict):
            return {
                k: (np.asarray(v) if not isinstance(v, (list, str, int, dict)) else v)
                for k, v in mm["actions"].items()
            }
    raise RuntimeError(f"no actions/pred_xyz in the result: {out!r}"[:2000])


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "alpamayo_stage_trn2.yaml"
    )
    timeout = int(
        os.environ.get("ALPAMAYO_INIT_TIMEOUT_S", "3600")
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

    def request():
        params = OmniDiffusionSamplingParams(seed=args.seed, extra_args={"robot_obs": obs})
        return omni.generate({"prompt": ""}, params)

    t0 = time.perf_counter()
    out = result_output(request())
    first = time.perf_counter() - t0
    lat, timings = [], []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        again = result_output(request())
        lat.append(time.perf_counter() - t0)
        timings.append(again.get("timing_ms") or {})
    deterministic = (
        np.array_equal(out["pred_xyz"], again["pred_xyz"])
        if args.repeat and "pred_xyz" in out
        else None
    )
    np.savez(args.output, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    summary = {
        "first_request_s": round(first, 3),
        "warm_request_ms": round(1000 * float(np.mean(lat)), 2) if lat else None,
        "warm_request_ms_min": round(1000 * float(np.min(lat)), 2) if lat else None,
        "warm_request_ms_median": round(1000 * float(np.median(lat)), 2) if lat else None,
        "deterministic_same_seed": deterministic,
        # in-model breakdown (rank 0), median over the warm requests
        "warm_breakdown_ms": {
            k: round(float(np.median([t[k] for t in timings])), 2)
            for k in (timings[0] if timings else {})
            if timings[0][k] is not None
        },
        "cot": out.get("cot"),
        "output_shapes": {k: list(v.shape) for k, v in out.items() if isinstance(v, np.ndarray)},
        "output": os.path.abspath(args.output),
    }
    print("[run]", json.dumps(summary))
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
