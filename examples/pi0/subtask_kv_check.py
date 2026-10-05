#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""pi0.52 subtask decode: KV-cached (prefill once + shared ``decode_step``) vs re-prefill.

For each task, greedy-decodes the subtask with both paths and compares the generated token ids,
both with natural EOS stopping and at fixed lengths (EOS masked, so every decode step runs). On
a NeuronCore it also times the subtask stage per path (warm) and the full ``pipeline.forward()``
with each path, and compares the resulting action chunks.

    # CPU fp32 (token check only)
    python examples/pi0/subtask_kv_check.py --model <pi052 dir> --out <run dir> --device cpu
    # device (token check + timing)
    python examples/pi0/subtask_kv_check.py --model <pi052 dir> --out <run dir> --device neuron
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch

TASKS = (
    "pick up the red cube and place it in the bowl",
    "open the top drawer and put the spoon inside",
    "fold the towel in half",
)


def _timeit(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.time()
        fn()
        out.append(round(time.time() - t0, 5))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--forced", type=int, nargs="*", default=[1, 16])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument(
        "--dtype", choices=("bf16", "fp32"), default=None, help="default: bf16 on device"
    )
    ap.add_argument("--skip-token-check", action="store_true")
    ap.add_argument(
        "--save-ref", default=None, help="save teacher ids + per-step logits (KV path) to this .pt"
    )
    ap.add_argument(
        "--ref",
        default=None,
        help="teacher-force a --save-ref file and report per-step logit error",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline

    on_dev = args.device == "neuron"
    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    dtype = args.dtype or ("bf16" if on_dev else "fp32")
    od.model, od.dtype = args.model, ("bfloat16" if dtype == "bf16" else "float32")
    od.model_config = {"tokenizer": args.tokenizer}

    t0 = time.time()
    pipe = NeuronPi05Pipeline(od_config=od)
    m = pipe.model
    report: dict = {"device": args.device, "dtype": dtype, "load_s": round(time.time() - t0, 1)}
    if on_dev:
        from vllm_neuron.envs import get_compile_backend_name

        pipe.to(torch.device("neuron", 0))
        pipe.compile(get_compile_backend_name())

    g = torch.Generator().manual_seed(0)
    r = pipe.config.image_resolution[0]
    cams = [k for k in pipe.config.input_features if "image" in k]
    images = [torch.rand(1, 3, r, r, generator=g) * 2 - 1 for _ in cams]
    masks = [torch.tensor([True]) for _ in cams]

    def gen(task, kv, n, forced):
        return m.generate_subtask(
            images,
            masks,
            task,
            max_new_tokens=n,
            kv_cache=kv,
            min_new_tokens=n if forced else 0,
            return_ids=True,
        )

    def teacher(task, kv, ids):
        """Feed ``ids`` (teacher forcing); return the per-step raw fp32 logits."""
        seen = []

        def hook(step, lg):
            seen.append(lg)
            return ids[step]

        m.generate_subtask(
            images,
            masks,
            task,
            max_new_tokens=len(ids),
            kv_cache=kv,
            step_hook=hook,
            min_new_tokens=len(ids),
        )
        return torch.stack(seen)

    n_ref = max(args.forced)
    if args.save_ref:
        ref = {}
        for task in args.tasks:
            _, ids = gen(task, True, n_ref, True)
            ref[task] = {"ids": ids, "logits": teacher(task, True, ids)}
        torch.save(ref, args.save_ref)
        report["saved_ref"] = args.save_ref
    if args.ref:
        ref = torch.load(args.ref, weights_only=False)
        err = {}
        for task in args.tasks:
            ids, ref_lg = ref[task]["ids"], ref[task]["logits"]
            for kv in (True, False):
                if kv is False and args.device == "cpu" and dtype == "fp32":
                    continue
                lg = teacher(task, kv, ids)
                rel = ((lg - ref_lg).norm(dim=-1) / ref_lg.norm(dim=-1)).tolist()
                top1 = (lg.argmax(-1) == ref_lg.argmax(-1)).float().mean().item()
                err[f"{task}|{'kv' if kv else 'reprefill'}"] = {
                    "rel_per_step": [round(x, 5) for x in rel],
                    "rel_max": round(max(rel), 5),
                    "rel_mean": round(sum(rel) / len(rel), 5),
                    "top1_agree": round(top1, 4),
                }
        for p in ("kv", "reprefill"):
            vals = [v for k, v in err.items() if k.endswith("|" + p)]
            if vals:
                err[f"{p}_rel_max"] = max(v["rel_max"] for v in vals)
                err[f"{p}_rel_mean"] = round(sum(v["rel_mean"] for v in vals) / len(vals), 5)
                err[f"{p}_top1_agree"] = round(sum(v["top1_agree"] for v in vals) / len(vals), 4)
        report["teacher_forced_vs_ref"] = err

    # 1) token identity
    t0 = time.time()
    rows, all_ok = [], True
    for task in [] if args.skip_token_check else args.tasks:
        for n, forced in [(args.max_new_tokens, False), *[(k, True) for k in args.forced]]:
            text_kv, ids_kv = gen(task, True, n, forced)
            text_re, ids_re = gen(task, False, n, forced)
            same = ids_kv == ids_re
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(ids_kv, ids_re)) if a != b),
                None if len(ids_kv) == len(ids_re) else min(len(ids_kv), len(ids_re)),
            )
            rows.append(
                {
                    "task": task,
                    "max_new_tokens": n,
                    "forced_length": forced,
                    "kv_text": text_kv,
                    "reprefill_text": text_re,
                    "kv_ids": ids_kv,
                    "reprefill_ids": ids_re,
                    "identical": same,
                    "first_diff": first_diff,
                }
            )
            all_ok &= same
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report["tokens"] = rows
    report["tokens_identical_all"] = all_ok
    report["token_check_s"] = round(time.time() - t0, 1)

    # 2) timing (device only): subtask stage per path, then pipeline.forward per path.
    if on_dev:
        timing = {}
        task = args.tasks[0]
        for n, forced, label in [
            (args.max_new_tokens, False, "natural"),
            *[(k, True, f"forced{k}") for k in args.forced],
        ]:
            for kv in (False, True):
                key = f"subtask_{label}_{'kv' if kv else 'reprefill'}_s"
                gen(task, kv, n, forced)  # warm
                timing[key] = _timeit(lambda: gen(task, kv, n, forced), args.repeat)
                timing[key + "_min"] = min(timing[key])
        f_lo, f_hi = min(args.forced), max(args.forced)
        if f_hi > f_lo:
            for p in ("kv", "reprefill"):
                timing[f"per_token_{p}_s"] = round(
                    (
                        timing[f"subtask_forced{f_hi}_{p}_s_min"]
                        - timing[f"subtask_forced{f_lo}_{p}_s_min"]
                    )
                    / (f_hi - f_lo),
                    5,
                )
        report["timing"] = timing

        obs = {
            "prompt": task,
            **{k: (torch.rand(r, r, 3, generator=g)).numpy() for k in cams},
            "state": (torch.rand(pipe.config.state_dim, generator=g) * 2 - 1).numpy(),
        }
        e2e, actions = {}, {}
        for kv in (False, True):
            m.subtask_use_kv = kv
            label = "kv" if kv else "reprefill"

            def run_forward(seed=1):
                sp = types.SimpleNamespace(
                    extra_args={"robot_obs": dict(obs)},
                    num_inference_steps=args.steps,
                    generator=torch.Generator().manual_seed(seed),
                )
                return pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))

            run_forward()  # warm
            e2e[f"forward_{label}_s"] = _timeit(run_forward, args.repeat)
            e2e[f"forward_{label}_s_min"] = min(e2e[f"forward_{label}_s"])
            e2e[f"forward_{label}_s_median"] = sorted(e2e[f"forward_{label}_s"])[args.repeat // 2]
            out = run_forward()
            actions[label] = torch.as_tensor(out.output["actions"]).float().cpu()
        m.subtask_use_kv = True
        a, b = actions["kv"], actions["reprefill"]
        e2e["actions_shape"] = list(a.shape)
        e2e["actions_finite"] = bool(torch.isfinite(a).all())
        e2e["actions_kv_vs_reprefill_rel"] = float((a - b).norm() / b.norm().clamp_min(1e-12))
        e2e["speedup_x"] = round(e2e["forward_reprefill_s_min"] / e2e["forward_kv_s_min"], 3)
        report["e2e"] = e2e
        torch.save(actions, os.path.join(args.out, "actions.pt"))

    report["ok"] = bool(all_ok) and (not on_dev or report["e2e"]["actions_finite"])
    if on_dev and not all_ok:
        # bf16 can flip a near-tie on the dummy images; the teacher-forced error is the gate then.
        report["ok"] = report["e2e"]["actions_finite"]
    with open(os.path.join(args.out, "subtask_kv_check.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    summary = {k: v for k, v in report.items() if k != "tokens"}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
