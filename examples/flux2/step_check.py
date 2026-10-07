# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced DiT step checks: replay device steps on CPU from the device's own inputs.

A Neuron run with ``FLUX2_STEP_DUMP=<dir>`` saves the full inputs and the output of selected DiT
calls (default steps 0, 24 and 49). This script replays each saved call through the diffusers
``Flux2Transformer2DModel`` on CPU in fp32 and in bf16 and scores, per step,

* ``rel_dev``  = rel-L2(device output, CPU fp32 output)
* ``rel_bf16`` = rel-L2(CPU bf16 output, CPU fp32 output) (the bf16 floor)
* pass if ``rel_dev <= k * rel_bf16 + 0.005`` (k = 2).

One model is resident at a time (fp32 ~128 GB host RAM, then bf16), shared by every dump dir::

    python examples/flux2/step_check.py --model-path <FLUX.2-dev> --out steps.json <dump_dir> [...]
"""

import argparse
import glob
import json
import os
import time

import torch

K, ABS = 2.0, 0.005
INPUTS = ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids", "guidance")


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def replay(model_path, dtype, dumps):
    from diffusers import Flux2Transformer2DModel

    t0 = time.time()
    model = Flux2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    ).eval()
    print(f"[step] {dtype} transformer loaded in {time.time() - t0:.0f}s", flush=True)
    outs = {}
    for n, path in enumerate(dumps, 1):
        print(f"[step] {dtype} {n}/{len(dumps)} start {path}", flush=True)
        d = torch.load(path)
        kw = {k: d[k] for k in INPUTS if k in d}
        for k in ("hidden_states", "encoder_hidden_states", "timestep", "guidance"):
            if k in kw:
                kw[k] = kw[k].to(dtype)
        t0 = time.time()
        with torch.no_grad():
            outs[path] = model(**kw, return_dict=False)[0].float()
        print(f"[step] {dtype} {path}: {time.time() - t0:.0f}s", flush=True)
    del model
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dirs", nargs="+")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dumps = [p for d in a.dump_dirs for p in sorted(glob.glob(os.path.join(d, "step_*.pt")))]
    if not dumps:
        raise SystemExit(f"no step_*.pt in {a.dump_dirs}")
    ref32 = replay(a.model_path, torch.float32, dumps)
    ref16 = replay(a.model_path, torch.bfloat16, dumps)
    rows = []
    for p in dumps:
        dev = torch.load(p)["output"].float()
        r = {
            "dump": p,
            "rel_dev": rel(dev, ref32[p]),
            "rel_bf16": rel(ref16[p], ref32[p]),
            "cos_dev": float(
                torch.nn.functional.cosine_similarity(dev.flatten(), ref32[p].flatten(), dim=0)
            ),
            "nonfinite_dev": int((~torch.isfinite(dev)).sum()),
        }
        r["bar"] = K * r["rel_bf16"] + ABS
        r["k"] = r["rel_dev"] / r["rel_bf16"] if r["rel_bf16"] else float("inf")
        r["pass"] = r["nonfinite_dev"] == 0 and r["rel_dev"] <= r["bar"]
        rows.append(r)
        print(f"[step] {json.dumps(r)}", flush=True)
    summary = {"ok": all(r["pass"] for r in rows), "k": K, "abs": ABS, "steps": rows}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"ok": summary["ok"], "n": len(rows), "out": a.out}))
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
