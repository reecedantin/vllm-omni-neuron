# SPDX-License-Identifier: Apache-2.0
"""InternVLA-A1.5 on Neuron: load, compile the three graphs, sample an action chunk, check parity.

Standalone runner (no vLLM engine): it drives :class:`InternVLAA15Runner` directly on one
NeuronCore, which is the whole model (~2.7B params, bf16) on trn2. One synthetic request (random
pixels/tokens with the real token layout) and fixed fp32 noise make the run reproducible.

    python examples/internvla/run.py --model /path/InternVLA-A1.5-base [--device neuron] \
        [--compare-cpu] [--golden golden.pt] [--repeat 5] [--out-dir DIR]

``--compare-cpu`` also runs the port eagerly on the CPU in fp32 (equal to upstream to ~1e-6,
see ``test/unit/test_internvla_a15_cpu.py``) and bf16, and reports the device error next to the
CPU-bf16 error. ``--golden`` compares with upstream's own fp32 output saved by
``examples/internvla/make_golden.py``. The last line printed is a one-line JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni_neuron.diffusion.models.internvla import InternVLAA15, InternVLAA15Runner
from vllm_omni_neuron.diffusion.models.internvla import preprocess as pp

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--vlm-config", default=None, help="Qwen3.5 config.json (default: see config.py)")
    ap.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--grid", type=int, nargs=2, default=(16, 16), help="patch grid per image (16x16 = 256x256 px)")
    ap.add_argument("--text-before", type=int, default=14)
    ap.add_argument("--text-after", type=int, default=90)
    ap.add_argument("--bucket", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3, help="warm runs after the first")
    ap.add_argument("--compare-cpu", action="store_true")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--out-dir", default=os.environ.get("FLEET_RUNS", "."))
    ap.add_argument("--tag", default="run")
    return ap.parse_args()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def neuron_compile_fn():
    from vllm_neuron.envs import get_compile_backend_name

    backend = get_compile_backend_name()

    def wrap(mod, name):
        return torch.compile(mod, backend=backend, fullgraph=True, dynamic=False,
                             options={"model_name": name, "compiler_args": list(COMPILER_ARGS)})

    return wrap


def main() -> int:
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    summary: dict = {"tag": a.tag, "model": os.path.basename(os.path.normpath(a.model)), "device": a.device}
    dev = torch.device("neuron", 0) if a.device == "neuron" else torch.device("cpu")

    t0 = time.time()
    model = InternVLAA15.from_pretrained(a.model, dtype=torch.bfloat16, device=dev, vlm_config=a.vlm_config)
    summary["load_s"] = round(time.time() - t0, 1)
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    summary["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    summary["weights_gb"] = round(nbytes / 1e9, 2)
    cfg = model.cfg
    batch = pp.synthetic_request(cfg, n_images=a.n_images, grid=tuple(a.grid), text_before=a.text_before,
                                 text_after=a.text_after, seed=a.seed)
    noise = pp.initial_noise(cfg, seed=a.seed)
    summary["prefix_tokens"] = int(batch["input_ids"].shape[1])

    runner = InternVLAA15Runner(model, dev, neuron_compile_fn() if a.device == "neuron" else None)
    t0 = time.time()
    out = runner.sample_actions(batch, noise, bucket=a.bucket)
    summary["first_call_s"] = round(time.time() - t0, 1)
    summary["bucket"] = runner.prepare(batch, a.bucket)["length"]
    warm, outs = [], []
    for _ in range(a.repeat):
        t0 = time.time()
        outs.append(runner.sample_actions(batch, noise, bucket=a.bucket))
        warm.append(time.time() - t0)
    if warm:
        summary["warm_s"] = round(min(warm), 4)
        summary["warm_breakdown"] = {k: round(v, 4) for k, v in runner.timings.items()}
        summary["deterministic"] = all(torch.equal(o, outs[0]) for o in outs) and torch.equal(outs[0], out)
    summary["finite"] = bool(torch.isfinite(out).all())
    torch.save({"actions": out, "noise": noise, "batch": batch}, os.path.join(a.out_dir, f"{a.tag}_actions.pt"))

    ok = summary["finite"]
    if a.compare_cpu:
        del runner, model
        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            t0 = time.time()
            m = InternVLAA15.from_pretrained(a.model, dtype=dt, vlm_config=a.vlm_config)
            ref = InternVLAA15Runner(m).sample_actions(batch, noise, bucket=a.bucket)
            summary[f"cpu_{name}_s"] = round(time.time() - t0, 1)
            torch.save(ref, os.path.join(a.out_dir, f"{a.tag}_cpu_{name}.pt"))
            if name == "fp32":
                ref32 = ref
            else:
                ref16 = ref
            del m
        summary["rel_dev_vs_cpu32"] = round(rel(out, ref32), 5)
        summary["cos_dev_vs_cpu32"] = round(cos(out, ref32), 6)
        summary["rel_cpu16_vs_cpu32"] = round(rel(ref16, ref32), 5)
        summary["rel_dev_vs_cpu16"] = round(rel(out, ref16), 5)
        if a.device == "neuron":
            ok = ok and summary["rel_dev_vs_cpu32"] <= max(2.0 * summary["rel_cpu16_vs_cpu32"], 0.02)
    if a.golden:
        g = torch.load(a.golden)
        summary["rel_dev_vs_upstream32"] = round(rel(out, g["actions_fp32"]), 5)
        summary["cos_dev_vs_upstream32"] = round(cos(out, g["actions_fp32"]), 6)
        if "actions_bf16" in g:
            summary["rel_upstream16_vs_upstream32"] = round(rel(g["actions_bf16"], g["actions_fp32"]), 5)
    summary["ok"] = bool(ok)
    with open(os.path.join(a.out_dir, f"{a.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
