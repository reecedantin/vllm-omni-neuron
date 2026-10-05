#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Served-vs-CPU parity: replay a request saved by ``run.py --save-request`` through the same
pipeline class on the CPU (fp32, and bf16 for the dtype-only bar) and compare the action chunks
the served runs returned (``run.py --output`` JSON files).

The served noise comes from the request seed (the model runner seeds a CPU generator with
``sampling_params.seed``); this replays it the same way, or uses the request's saved noise.

    python examples/pi0/served_parity.py --model <ckpt> --request req.npz --served a.json [b.json ...] \
        [--stats stats_a.jsonl ...] [--model-config min_new_subtask_tokens=12 ...]

Prints one JSON line: rel-L2 / cosine of every served chunk vs CPU fp32, the CPU bf16 error, the
bar ``k x bf16 + 0.5%`` (k = ``--bar-k``, 2) and the subtask text each CPU run generated (pi0.52).
With ``--stats`` (the served run's ``PI0_STATS_FILE``) it also checks that every tensor-parallel
rank returned the same action chunk and subtask token ids for every request.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import numpy as np
import torch


def _pipe(model: str, tokenizer: str, dtype: str, model_config: dict | None = None):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0Pipeline, NeuronPi05Pipeline

    with open(os.path.join(model, "config.json")) as f:
        kind = json.load(f).get("type", "pi05")
    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype = model, dtype
    od.model_config = {**(model_config or {}), "tokenizer": tokenizer}
    return (NeuronPi0Pipeline if kind == "pi0" else NeuronPi05Pipeline)(od_config=od)


def _rank_agreement(stats_path: str) -> dict:
    """Compare the per-request digests every tensor-parallel rank wrote to ``PI0_STATS_FILE``."""
    by_rank: dict[int, list[dict]] = {}
    with open(stats_path) as f:
        for line in f:
            rec = json.loads(line)
            by_rank.setdefault(int(rec.get("tp_rank", 0)), []).append(rec)
    keys = ("actions_sha256", "subtask_ids_sha256")
    seqs = {r: [tuple(rec.get(k) for k in keys) for rec in recs] for r, recs in by_rank.items()}
    first = next(iter(seqs.values()))
    agree = all(s == first for s in seqs.values()) and all(d[0] for d in first)
    return {
        "ranks": sorted(seqs),
        "requests_per_rank": {r: len(s) for r, s in seqs.items()},
        "distinct_action_digests": len({d[0] for s in seqs.values() for d in s}),
        "all_ranks_agree": bool(agree),
    }


def _metrics(a: np.ndarray, ref: np.ndarray) -> dict:
    a, ref = a.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    return {
        "rel_l2": float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-12)),
        "cos": float(a @ ref / max(np.linalg.norm(a) * np.linalg.norm(ref), 1e-12)),
        "max_abs": float(np.abs(a - ref).max()),
    }


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--served", nargs="+", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bar-k", type=float, default=2.0, help="bar = k x CPU bf16 rel-L2 + 0.5%%")
    ap.add_argument(
        "--model-config",
        action="append",
        default=[],
        help="KEY=VALUE model_config the served run used (e.g. min_new_subtask_tokens=12)",
    )
    ap.add_argument(
        "--stats",
        nargs="*",
        default=[],
        help="PI0_STATS_FILE of each served run: every TP rank's action/subtask digests must agree",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import yaml

    mc = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.model_config)}

    z = np.load(args.request, allow_pickle=False)
    obs = {k: z[k] for k in z.files if k not in ("prompt", "noise")}
    obs["prompt"] = str(z["prompt"])
    noise = torch.from_numpy(z["noise"]) if "noise" in z.files else None

    refs, texts = {}, {}
    for dtype in ("float32", "bfloat16"):
        pipe = _pipe(args.model, args.tokenizer, dtype, mc)
        extra = {"robot_obs": dict(obs)}
        if noise is not None:
            extra["noise"] = noise.clone()
        sp = types.SimpleNamespace(
            extra_args=extra,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        )
        if getattr(pipe, "model", None) is not None and hasattr(pipe.model, "subtask_gen"):
            sub = pipe._maybe_generate_subtask(dict(obs))
            texts[dtype] = {"text": sub.get("prompt"), "steps": len(pipe.last_subtask_ids)}
        out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
        refs[dtype] = np.asarray(out.output["actions"], dtype=np.float32)
        del pipe
    bf16 = _metrics(refs["bfloat16"], refs["float32"])
    report = {
        "cpu_bf16_vs_fp32": bf16,
        "bar_rel_l2": args.bar_k * bf16["rel_l2"] + 0.005,
        "bar": f"{args.bar_k:g} x CPU bf16 + 0.5%",
        "model_config": mc,
        "cpu_subtask": texts,
        "served": {},
        "ranks": {},
    }
    ok = True
    for path in args.stats:
        agree = _rank_agreement(path)
        with open(path) as f:
            served_texts = {json.loads(line).get("subtask_text") for line in f}
        agree["subtask_text"] = sorted(t for t in served_texts if t is not None)
        if texts.get("float32", {}).get("text") is not None:
            agree["subtask_matches_cpu_fp32"] = agree["subtask_text"] == [texts["float32"]["text"]]
        ok &= agree["all_ranks_agree"]
        report["ranks"][os.path.basename(path)] = agree
    for path in args.served:
        with open(path) as f:
            a = np.asarray(json.load(f)["actions"], dtype=np.float32)
        m = _metrics(a, refs["float32"])
        m["pass"] = m["rel_l2"] <= report["bar_rel_l2"]
        ok &= m["pass"]
        report["served"][os.path.basename(path)] = m
    report["ok"] = bool(ok)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
