#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Multi-draw served accuracy sweep: many (camera frames, initial noise) draws through ONE served
engine, then a CPU replay of every draw, so the device error is a distribution, not one sample.

    # 1. on the device (one engine, one request per draw), every rank writes PI0_STATS_FILE records
    PI0_STATS_FILE=out/stats.jsonl python examples/pi0/served_sweep.py serve --model <ckpt> \
        --stage-config examples/pi0/pi052_stage_tp2.yaml --out out --noise-seeds 0-7 --frame-seeds 1-4
    # 2. on the CPU: fp32 reference + bf16 bar per draw, all-rank digest check per request
    python examples/pi0/served_sweep.py parity --model <ckpt> --out out [--stats out/stats.jsonl]

A draw is (frame seed, noise seed): ``--noise-seeds`` vary the noise with the frames of
``--frame-seed`` (default 0), ``--frame-seeds`` vary the random 224 px frames with noise seed
``--noise-seed`` (default 0). The bar per draw is ``k x (CPU bf16 rel-L2) + 0.5%`` (k = ``--bar-k``,
2), as in ``served_parity.py``. ``parity`` prints one JSON line with every draw and the summary:
mean / max of device error, of the bar and of their ratio, and the number of misses.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np


def _seeds(spec: str | None) -> list[int]:
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _draws(args) -> list[tuple[int, int]]:
    draws = [(args.frame_seed, n) for n in _seeds(args.noise_seeds)]
    draws += [(f, args.noise_seed) for f in _seeds(args.frame_seeds)]
    return draws or [(args.frame_seed, args.noise_seed)]


def _ckpt(model: str) -> dict:
    with open(os.path.join(model, "config.json")) as f:
        return json.load(f)


def _observation(model: str, task: str, frame_seed: int) -> dict:
    """Random uint8 frames for every camera of the checkpoint (as ``run.py`` without ``--image``)
    and a zero state of the checkpoint's width."""
    feats = _ckpt(model).get("input_features", {})
    rng = np.random.default_rng(frame_seed)
    obs: dict = {"prompt": task}
    for key in feats:
        if key.startswith("observation.images."):
            obs[key] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    width = int(feats.get("observation.state", {}).get("shape", [32])[0])
    obs["state"] = np.zeros(width, dtype=np.float32)
    return obs


def _noise(model: str, seed: int) -> np.ndarray:
    c = _ckpt(model)
    shape = (1, int(c.get("chunk_size", 50)), int(c.get("max_action_dim", 32)))
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def serve(args) -> int:
    import tempfile

    import yaml
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_neuron.diffusion.models.pi0.request import encode_camera

    with open(args.stage_config) as f:
        cfg = yaml.safe_load(f)
    for stage in cfg["stage_args"]:
        stage["engine_args"].setdefault("model_config", {})["tokenizer"] = args.tokenizer
    fd, stage_cfg = tempfile.mkstemp(suffix="_stage.yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), 3600)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    omni = Omni(
        model=args.model, stage_configs_path=stage_cfg, stage_init_timeout=3600, init_timeout=3600
    )
    os.makedirs(args.out, exist_ok=True)
    draws = _draws(args)
    for i, (fs, ns) in enumerate(draws):
        obs = _observation(args.model, args.task, fs)
        noise = _noise(args.model, ns)
        np.savez(os.path.join(args.out, f"request_{i:03d}.npz"), noise=noise, **obs)
        wire = {
            k: (encode_camera(v) if k.startswith("observation.images.") else v)
            for k, v in obs.items()
        }
        extra = {
            "robot_obs": wire,
            "noise": {"data": noise.tobytes(), "shape": list(noise.shape), "dtype": "float32"},
        }
        t0 = time.perf_counter()
        result = omni.generate(
            {"prompt": args.task},
            OmniDiffusionSamplingParams(num_inference_steps=args.steps, seed=0, extra_args=extra),
        )
        item = result[0]
        actions = None
        for holder in (item, getattr(item, "request_output", None)):
            mm = getattr(holder, "multimodal_output", None) if holder is not None else None
            if isinstance(mm, dict) and "actions" in mm:
                actions = np.asarray(mm["actions"], dtype=np.float32)
        if actions is None:
            raise RuntimeError("no 'actions' in the request output")
        with open(os.path.join(args.out, f"actions_{i:03d}.json"), "w") as f:
            json.dump({"frame_seed": fs, "noise_seed": ns, "actions": actions.tolist()}, f)
        print(
            f"[sweep] draw {i} frames={fs} noise={ns}: {time.perf_counter() - t0:.3f}s "
            f"finite={bool(np.isfinite(actions).all())}"
        )
    with open(os.path.join(args.out, "draws.json"), "w") as f:
        json.dump(draws, f)
    return 0


def _metrics(a: np.ndarray, ref: np.ndarray) -> dict:
    a, ref = a.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    return {
        "rel_l2": float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-12)),
        "cos": float(a @ ref / max(np.linalg.norm(a) * np.linalg.norm(ref), 1e-12)),
    }


def _rank_digests(stats_path: str) -> dict[int, list[tuple]]:
    by_rank: dict[int, list[tuple]] = {}
    with open(stats_path) as f:
        for line in f:
            rec = json.loads(line)
            by_rank.setdefault(int(rec.get("tp_rank", 0)), []).append(
                (rec.get("actions_sha256"), rec.get("subtask_ids_sha256"), rec.get("subtask_text"))
            )
    return by_rank


def parity(args) -> int:
    import types

    import torch
    from served_parity import _pipe  # same CPU pipeline construction

    with open(os.path.join(args.out, "draws.json")) as f:
        draws = [tuple(d) for d in json.load(f)]
    reqs = []
    for i in range(len(draws)):
        z = np.load(os.path.join(args.out, f"request_{i:03d}.npz"), allow_pickle=False)
        obs = {k: z[k] for k in z.files if k not in ("prompt", "noise")}
        obs["prompt"] = str(z["prompt"])
        reqs.append((obs, torch.from_numpy(z["noise"])))
    refs: dict[str, list[np.ndarray]] = {}
    texts: dict[str, list] = {}
    for dtype in ("float32", "bfloat16"):
        pipe = _pipe(args.model, args.tokenizer, dtype)
        refs[dtype], texts[dtype] = [], []
        for obs, noise in reqs:
            sp = types.SimpleNamespace(
                extra_args={"robot_obs": dict(obs), "noise": noise.clone()},
                num_inference_steps=args.steps,
                generator=None,
            )
            with torch.inference_mode():
                out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
            refs[dtype].append(np.asarray(out.output["actions"], dtype=np.float32))
            texts[dtype].append(getattr(pipe, "last_subtask_ids", None))
        del pipe
    ranks = _rank_digests(args.stats) if args.stats else {}
    rows, ok = [], True
    for i, (fs, ns) in enumerate(draws):
        with open(os.path.join(args.out, f"actions_{i:03d}.json")) as f:
            dev = np.asarray(json.load(f)["actions"], dtype=np.float32)
        bf16 = _metrics(refs["bfloat16"][i], refs["float32"][i])["rel_l2"]
        bar = args.bar_k * bf16 + 0.005
        m = _metrics(dev, refs["float32"][i])
        row = {
            "draw": i,
            "frame_seed": fs,
            "noise_seed": ns,
            "device_rel": m["rel_l2"],
            "device_cos": m["cos"],
            "bf16_rel": bf16,
            "bar": bar,
            "ratio_to_bar": m["rel_l2"] / bar,
            "device_over_bf16": m["rel_l2"] / max(bf16, 1e-12),
            "pass": m["rel_l2"] <= bar,
            "cpu_subtask_ids_equal": texts["float32"][i] == texts["bfloat16"][i],
        }
        if ranks:
            per = [seq[i] if i < len(seq) else None for seq in ranks.values()]
            row["ranks_agree"] = all(p == per[0] for p in per) and per[0] is not None
            row["served_subtask_text"] = per[0][2] if per[0] else None
            ok &= row["ranks_agree"]
        rows.append(row)
    dev = np.array([r["device_rel"] for r in rows])
    bars = np.array([r["bar"] for r in rows])
    bf = np.array([r["bf16_rel"] for r in rows])
    summary = {
        "draws": len(rows),
        "ranks": sorted(ranks) if ranks else [0],
        "all_ranks_agree": all(r.get("ranks_agree", True) for r in rows),
        "misses": int(sum(not r["pass"] for r in rows)),
        "device_rel_mean": float(dev.mean()),
        "device_rel_max": float(dev.max()),
        "bf16_rel_mean": float(bf.mean()),
        "bar_mean": float(bars.mean()),
        "ratio_to_bar_mean": float((dev / bars).mean()),
        "ratio_to_bar_max": float((dev / bars).max()),
        "device_over_bf16_mean": float((dev / bf).mean()),
        "device_over_bf16_median": float(np.median(dev / bf)),
        "mean_within_bar": bool(dev.mean() <= bars.mean()),
        "bar": f"{args.bar_k:g} x CPU bf16 + 0.5%",
    }
    report = {"summary": summary, "draws": rows}
    with open(os.path.join(args.out, args.report), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if ok and summary["mean_within_bar"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("serve", "parity"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--stage-config", default=None, help="serve: the stage yaml (TP layout)")
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frame-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-seeds", default=None, help="e.g. 0-7 (frames of --frame-seed)")
    ap.add_argument("--frame-seeds", default=None, help="e.g. 1-4 (noise of --noise-seed)")
    ap.add_argument("--bar-k", type=float, default=2.0)
    ap.add_argument("--stats", default=None, help="parity: the serve run's PI0_STATS_FILE")
    ap.add_argument("--report", default="sweep_parity.json")
    args = ap.parse_args()
    if args.mode == "serve":
        if not args.stage_config:
            ap.error("serve needs --stage-config")
        return serve(args)
    return parity(args)


if __name__ == "__main__":
    raise SystemExit(main())
