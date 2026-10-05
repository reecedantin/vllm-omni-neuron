# SPDX-License-Identifier: Apache-2.0
"""Served-path action gate: send the observation and initial noise saved by ``reference.py``
through the Omni runner (processor, model graphs, decoding) and compare the normalized action
chunk with upstream fp32. Works for any stage config (TP=1 or a TP=2 action head).

    python examples/gr00t/served_parity.py --model-path M --stage-config S --reference ref.pt --out res.json

Exits nonzero if MSE > --max-mse or cosine < --min-cos.
"""

from __future__ import annotations

import argparse
import json
import os

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np
import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True)
ap.add_argument("--stage-config", required=True)
ap.add_argument("--reference", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-mse", type=float, default=1e-3)
ap.add_argument("--min-cos", type=float, default=0.999)
ap.add_argument("--repeat", type=int, default=0, help="also time this many warm requests (per-stage medians)")
args = ap.parse_args()


def main() -> None:
    ref = torch.load(args.reference, weights_only=False)
    timeout = int(os.environ.get("GR00T_INIT_TIMEOUT_S", "3600"))
    omni = Omni(model=args.model_path, stage_configs_path=args.stage_config, stage_init_timeout=timeout,
                init_timeout=timeout)
    noise = ref["noise"].float().numpy()
    extra = {"robot_obs": ref["obs"], "return_action_pred": True, "return_timing": args.repeat > 0,
             "initial_noise": {"data": noise.tobytes(), "shape": list(noise.shape)}}
    res = omni.generate({"prompt": ""}, OmniDiffusionSamplingParams(seed=0, extra_args=extra))
    out = res[0]
    acts = None
    for holder in (out, getattr(out, "request_output", None)):
        mm = getattr(holder, "multimodal_output", None)
        if isinstance(mm, dict) and "actions" in mm:
            acts = mm["actions"]
    if acts is None or "action_pred" not in acts:
        raise SystemExit(f"no action_pred in the result: {out!r}"[:2000])
    a = torch.as_tensor(np.asarray(acts["action_pred"], dtype=np.float32))
    r = ref["action_pred"].float().reshape(a.shape)
    summary = {"mse": float(((a - r) ** 2).mean()), "rel": float((a - r).norm() / r.norm()),
               "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0))}
    dec = ref.get("decoded") or {}
    summary["decoded_rel"] = {k: float(np.linalg.norm(np.asarray(acts[k]) - np.asarray(v)) / np.linalg.norm(v))
                              for k, v in dec.items() if k in acts}
    summary["ok"] = summary["mse"] <= args.max_mse and summary["cos"] >= args.min_cos
    if args.repeat:  # warm timing on the same request: per-stage medians (NVIDIA's split) + wall
        import time

        wall, stages = [], []
        for _ in range(args.repeat):
            t = time.perf_counter()
            r = omni.generate({"prompt": ""}, OmniDiffusionSamplingParams(seed=0, extra_args=extra))
            wall.append(1e3 * (time.perf_counter() - t))
            mm = getattr(r[0], "multimodal_output", None) or getattr(r[0].request_output, "multimodal_output")
            stages.append(np.asarray(mm["actions"]["timing_ms"], dtype=np.float32).reshape(-1))
        med = np.median(np.stack(stages), axis=0)
        names = ("data_processing", "backbone", "action_head", "decode", "in_worker_total")
        summary["stages_ms"] = {n: round(float(v), 2) for n, v in zip(names, med)}
        summary["stages_ms"]["e2e_sum"] = round(float(med[0] + med[1] + med[2]), 2)
        summary["served_wall_ms"] = round(float(np.median(wall)), 2)
        summary["repeat"] = args.repeat
    print("[parity]", json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
