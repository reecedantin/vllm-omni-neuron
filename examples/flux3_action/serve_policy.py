# SPDX-License-Identifier: Apache-2.0
"""Serve FLUX 3 Action through the vLLM-Omni stage harness and run one DROID policy request.

    python examples/flux3_action/serve_policy.py --policy <flux-3-action-droid> \
        --base <flux-3-action-base> --obs observation.npz --out-dir out/ [--warm 2] [--reference ref.pt]
    # TP4 on the four cores of one chip:
    #   NEURON_VISIBLE_DEVICES=<4 cores> (NEURON_RT_VISIBLE_CORES unset) ... \
    #   --stage-config examples/flux3_action/flux3_action_stage_tp4.yaml

The pipeline (:class:`NeuronFlux3ActionPipeline`) runs the policy in the vLLM Omni stage worker; this
script builds the request from a saved observation, calls ``omni.generate``, and saves the returned
actions (and decoded frames when ``--decode``) plus a ``served_report.json`` with per-stage timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--policy", required=True)
ap.add_argument(
    "--base",
    default=os.environ.get("FLUX3_ACTION_BASE"),
    help="local flux-3-action-base copy (default: $FLUX3_ACTION_BASE, else the Hub)",
)
ap.add_argument("--obs", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--stage-config", default=None)
ap.add_argument("--decode", action="store_true")
ap.add_argument("--reference", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument(
    "--warm", type=int, default=1, help="warm requests after the first (per-stage latency)"
)
ap.add_argument(
    "--list-cameras",
    action="store_true",
    help="send cameras as nested lists (the pre-bytes request form; for comparison)",
)
ap.add_argument(
    "--accuracy-steps",
    type=int,
    default=None,
    help="also send one request with this many sampler steps (e.g. 1 = single-step accuracy probe)",
)
ap.add_argument(
    "--accuracy-reference",
    default=None,
    help="reference .pt for the --accuracy-steps request (compared on actions)",
)
ap.add_argument(
    "--new-captions",
    type=int,
    default=0,
    help="after the warm requests, send N requests with captions the worker has not seen",
)
ap.add_argument(
    "--rank-check",
    action="store_true",
    help="every request also gathers each rank's actions + video-latent digest and reports whether "
    "all ranks agree (FLUX3_ACTION_RANK_CHECK=1 in the workers)",
)
args = ap.parse_args()


def payload_task(obs: str) -> str:
    from vllm_omni_neuron.diffusion.models.flux3_action.observation import load_observation

    return load_observation(obs)["task"][0]


def find_payload(result):
    """The action envelope (``{"actions", "timing", ...}``) in an ``omni.generate`` result."""
    ro = result[0].request_output
    payload = None
    for name in ("images", "custom_output", "multimodal_output", "output"):
        val = getattr(ro, name, None)
        if isinstance(val, list) and val and isinstance(val[0], dict) and "payload" in val[0]:
            payload = val[0]["payload"]
            break
        if isinstance(val, dict) and "actions" in val:
            payload = val
            break
    if payload is None:
        raise SystemExit(
            f"could not find the action payload in request_output; fields: "
            f"{[n for n in dir(ro) if not n.startswith('_')]}"
        )
    return payload


def main() -> None:
    if args.base:
        os.environ["FLUX3_ACTION_BASE"] = args.base
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    import vllm_omni_neuron.bootstrap  # noqa: F401  must precede vllm imports
    from vllm_omni_neuron.diffusion.models.flux3_action.observation import observation_extra_args

    os.makedirs(args.out_dir, exist_ok=True)
    stage = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "flux3_action_stage.yaml"
    )
    if args.decode:
        os.environ["FLUX3_ACTION_DECODE"] = "1"
    if args.rank_check:
        os.environ["FLUX3_ACTION_RANK_CHECK"] = "1"
    timeout = int(os.environ.get("FLUX3_HANDSHAKE_TIMEOUT_S", "7200"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")

    policy = args.policy
    if not os.path.isfile(os.path.join(policy, "model_index.json")):
        # released packages ship no model_index.json; build a symlinked serving dir that adds one
        from make_serving_dir import make_serving_dir

        policy = make_serving_dir(policy, os.path.join(args.out_dir, "serving"))
    omni = Omni(
        model=policy, stage_configs_path=stage, stage_init_timeout=timeout, init_timeout=timeout
    )

    rank_checks: list[dict] = []

    def request(steps=None, caption=None):
        # built per request, like a real client: the observation -> request conversion is timed
        t0 = time.time()
        extra = observation_extra_args(args.obs)
        if caption is not None:
            extra["task"] = caption
        if args.list_cameras:
            for key in [k for k in extra if k.startswith("images.")]:
                c = extra[key]
                extra[key] = np.frombuffer(c["data"], dtype=np.uint8).reshape(c["shape"]).tolist()
        build_s = time.time() - t0
        if steps is not None:
            extra["num_inference_steps"] = steps
        extra["client_send_time"] = time.time()
        params = OmniDiffusionSamplingParams(seed=args.seed, extra_args=extra)
        res = omni.generate({"prompt": extra["task"]}, params)
        if args.rank_check:
            t = find_payload(res).get("timing") or {}
            rank_checks.append(
                {k: t.get(k) for k in ("rank_check_ranks", "ranks_agree", "rank_action_max_rel")}
            )
        return res, t0, build_s, time.time()

    def text_timing(res) -> dict:
        t = find_payload(res).get("timing") or {}
        return {k: t.get(k) for k in ("text_s", "text_new_captions", "text_device")}

    result, t0, _, t1 = request()
    first_s = t1 - t0
    first_text = text_timing(result)
    # warm requests: the first includes cold compile/load; time the next ``--warm`` for a real latency
    warm_s, breakdowns = [], []
    for _ in range(args.warm):
        result, t0, build_s, t1 = request()
        warm_s.append(round(t1 - t0, 3))
        breakdowns.append((result, t0, build_s, t1))
    payload = find_payload(result)
    actions = payload["actions"]
    actions = actions.cpu().float() if hasattr(actions, "detach") else torch.as_tensor(actions)
    timing = payload.get("timing") or {}
    rep = {
        "served": True,
        "camera_encoding": "list" if args.list_cameras else "bytes",
        "first_s": round(first_s, 2),
        "actions_shape": list(actions.shape),
        "actions_mean": float(actions.mean()),
        "warm_s": warm_s,
        "warm_median_s": float(np.median(warm_s)) if warm_s else None,
        "stage_timing": timing,
    }
    if breakdowns and "worker_exit_time" in timing:
        # end-to-end split of the LAST warm request (wall clock on one host)
        _, t0, build_s, t1 = breakdowns[-1]
        enter, exit_ = timing["worker_enter_time"], timing["worker_exit_time"]
        rep["request_breakdown_s"] = {
            "client_build": round(build_s, 4),
            "client_to_worker": round(timing.get("transport_in_s", enter - t0 - build_s), 4),
            "worker_parse": round(timing["parse_s"], 4),
            "vae_encode": round(timing.get("vae_encode_s", 0.0), 4),
            "denoise": round(timing.get("denoise_s", 0.0), 4),
            "worker_forward_total": round(timing["forward_s"], 4),
            "worker_to_client": round(t1 - exit_, 4),
            "total": round(t1 - t0, 4),
        }
    np.save(os.path.join(args.out_dir, "actions.npy"), actions.numpy())
    if "video" in payload and payload["video"] is not None:
        vid = payload["video"]
        vid = vid.cpu().float() if hasattr(vid, "detach") else torch.as_tensor(vid)
        torch.save(vid, os.path.join(args.out_dir, "frames.pt"))
        rep["frames_shape"] = list(vid.shape)
    rep["first_request_text"] = first_text
    if args.new_captions:
        # requests whose caption the worker has not seen: the caption-encoder cost on the hot path
        runs = []
        for i in range(args.new_captions):
            cap = f"{payload_task(args.obs)} carefully, attempt {i + 1}"
            res, t0, _, t1 = request(caption=cap)
            runs.append({"total_s": round(t1 - t0, 4), **text_timing(res)})
        rep["new_caption_requests"] = runs
        rep["new_caption_median_s"] = float(np.median([r["total_s"] for r in runs]))
        rep["new_caption_text_median_s"] = float(np.median([r["text_s"] for r in runs]))
    if args.accuracy_steps is not None:
        res, *_ = request(args.accuracy_steps)
        acc = find_payload(res)["actions"]
        acc = acc.cpu().float() if hasattr(acc, "detach") else torch.as_tensor(acc).float()
        rep["accuracy_probe"] = {"num_inference_steps": args.accuracy_steps}
        np.save(os.path.join(args.out_dir, f"actions_{args.accuracy_steps}step.npy"), acc.numpy())
        if args.accuracy_reference:
            aref = torch.load(args.accuracy_reference)["actions"].float()
            rep["accuracy_probe"]["action_rel"] = float((acc - aref).norm() / aref.norm())
    if args.reference:
        ref = torch.load(args.reference)["actions"].float()
        rep["parity"] = {
            "action_rel": float((actions - ref).norm() / ref.norm()),
            "action_mse": float(((actions - ref) ** 2).mean()),
        }
    if args.rank_check:
        rep["rank_check"] = {
            "requests": len(rank_checks),
            "all_requests_all_ranks_agree": bool(rank_checks)
            and all(r["ranks_agree"] is True for r in rank_checks),
            "ranks": rank_checks[-1]["rank_check_ranks"] if rank_checks else None,
            "max_rel_vs_rank0": max((r["rank_action_max_rel"] or 0.0) for r in rank_checks)
            if rank_checks
            else None,
        }
    with open(os.path.join(args.out_dir, "served_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
