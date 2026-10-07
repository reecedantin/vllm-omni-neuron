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
ap.add_argument(
    "--repeat", type=int, default=0, help="also time this many warm requests (per-stage medians)"
)
ap.add_argument(
    "--rank-report",
    default=None,
    help="directory for per-rank layout + action digests ($GR00T_RANK_REPORT); the gate then also "
    "requires every TP rank to report, to agree on every request, and (unless "
    "GR00T_PRETRANSPOSE=none) to have every linear pretransposed",
)
args = ap.parse_args()
if args.rank_report:
    os.environ["GR00T_RANK_REPORT"] = args.rank_report  # read by every worker rank at pipeline init


def rank_check(d: str, n_requests: int) -> dict:
    """All-rank agreement (shared ``compare_rank_digest_files``, bit-exact) on the last ``n_requests``
    requests, plus every rank's pretranspose layout, from the reports the pipeline wrote."""
    import glob

    from vllm_omni_neuron.testing import compare_rank_digest_files

    layouts = {}
    for f in glob.glob(os.path.join(d, "layout_*.json")):
        lay = json.load(open(f))
        layouts[lay["rank"]] = lay
    size = next(iter(layouts.values()))["tp_size"] if layouts else 0
    want_pt = os.environ.get("GR00T_PRETRANSPOSE", "all") not in ("", "none", "0")
    pt_ok = bool(layouts) and all(
        p["pretransposed"] == p["linears"] if want_pt else p["pretransposed"] == 0
        for lay in layouts.values()
        for p in lay["parts"].values()
    )
    # the engine's warm-up dummy request may add a leading directory: check the last n_requests
    reqs = sorted(glob.glob(os.path.join(d, "req*")))[-n_requests:]
    reports = [compare_rank_digest_files(r, world_size=size) for r in reqs]
    bad = [
        os.path.basename(r) + ": " + rep.summary() for r, rep in zip(reqs, reports) if not rep.ok
    ]
    agree = len(reports) == n_requests and not bad
    return {
        "tp_size": size,
        "ranks_reporting": sorted(layouts),
        "requests": len(reports),
        "all_ranks_agree": agree,
        "first_failures": bad[:3],
        "action_sha256_rank0": reports[-1].to_json()["sha256_rank0"].get("action_pred")
        if reports
        else None,
        "pretranspose_expected": want_pt,
        "pretranspose_ok": pt_ok,
        "layout": {r: v["parts"] for r, v in sorted(layouts.items())},
        "ok": agree and pt_ok and len(layouts) == size,
    }


def main() -> None:
    import time

    ref = torch.load(args.reference, weights_only=False)
    timeout = int(os.environ.get("GR00T_INIT_TIMEOUT_S", "3600"))
    t_init = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=args.stage_config,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )
    t_first = time.perf_counter()
    noise = ref["noise"].float().numpy()
    extra = {
        "robot_obs": ref["obs"],
        "return_action_pred": True,
        "return_timing": args.repeat > 0,
        "initial_noise": {"data": noise.tobytes(), "shape": list(noise.shape)},
    }
    res = omni.generate({"prompt": ""}, OmniDiffusionSamplingParams(seed=0, extra_args=extra))
    t_done = time.perf_counter()
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
    summary = {
        "mse": float(((a - r) ** 2).mean()),
        "rel": float((a - r).norm() / r.norm()),
        "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0)),
    }
    dec = ref.get("decoded") or {}
    summary["decoded_rel"] = {
        k: float(np.linalg.norm(np.asarray(acts[k]) - np.asarray(v)) / np.linalg.norm(v))
        for k, v in dec.items()
        if k in acts
    }
    summary["ok"] = summary["mse"] <= args.max_mse and summary["cos"] >= args.min_cos
    # engine init (weights + the warm-up dummy run) and the first real request (graph compiles or NEFF loads)
    summary["engine_init_s"] = round(t_first - t_init, 1)
    summary["first_request_s"] = round(t_done - t_first, 1)
    if args.repeat:  # warm timing on the same request: per-stage medians (NVIDIA's split) + wall
        wall, stages = [], []
        for _ in range(args.repeat):
            t = time.perf_counter()
            r = omni.generate({"prompt": ""}, OmniDiffusionSamplingParams(seed=0, extra_args=extra))
            wall.append(1e3 * (time.perf_counter() - t))
            mm = getattr(r[0], "multimodal_output", None) or getattr(
                r[0].request_output, "multimodal_output"
            )
            stages.append(np.asarray(mm["actions"]["timing_ms"], dtype=np.float32).reshape(-1))
        med = np.median(np.stack(stages), axis=0)
        names = ("data_processing", "backbone", "action_head", "decode", "in_worker_total")
        summary["stages_ms"] = {n: round(float(v), 2) for n, v in zip(names, med)}
        summary["stages_ms"]["e2e_sum"] = round(float(med[0] + med[1] + med[2]), 2)
        summary["served_wall_ms"] = round(float(np.median(wall)), 2)
        summary["served_wall_ms_all"] = [round(w, 2) for w in wall]
        summary["repeat"] = args.repeat
    if args.rank_report:
        summary["ranks"] = rank_check(args.rank_report, 1 + args.repeat)
        summary["ok"] = summary["ok"] and summary["ranks"]["ok"]
    print("[parity]", json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
