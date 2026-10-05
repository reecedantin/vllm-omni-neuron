# SPDX-License-Identifier: Apache-2.0
"""pi0.5 / pi0.52 device check: load, compile, run the action path on one NeuronCore and
compare it against the CPU fp32 upstream model (and the same graphs in bf16 on the CPU, which
isolates the pure dtype error from the Neuron-specific error).

    python examples/pi0/device_check.py --model /path/to/pi052-base --out runs/m0 [--steps 10]

Prints one final JSON line: load/compile/warm timings, action rel-L2 / cosine / MSE vs CPU fp32,
the bf16-CPU reference error and the device HBM in use (when neuron-monitor is available).
Exits nonzero when the device result is not deterministic or misses the bar: rel-L2 vs CPU fp32
<= ``--bar-k`` x (CPU bf16 rel-L2) + 0.5% (the same bar as ``served_parity.py``; ``--max-rel``
replaces it with a fixed value, and is the only check under ``--skip-cpu-ref``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch


def _metrics(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a, ref = a.float().flatten(), ref.float().flatten()
    return {
        "rel_l2": ((a - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        "cos": torch.nn.functional.cosine_similarity(a, ref, dim=0).item(),
        "mse": torch.mean((a - ref) ** 2).item(),
        "max_abs": (a - ref).abs().max().item(),
    }


def _hbm_snapshot() -> dict | None:
    """Device memory of this process from neuron-monitor (best effort): the runtime whose pid is
    ours, else the per-core usage of the cores in NEURON_RT_VISIBLE_CORES."""
    cfg = {
        "period": "1s",
        "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "memory_used"}]}],
    }
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"nmon_{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    lo, _, hi = cores.partition("-")
    mine = {str(c) for c in range(int(lo), int(hi or lo) + 1)} if lo.isdigit() else set()
    p = None
    try:
        p = subprocess.Popen(
            ["neuron-monitor", "-c", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        t0 = time.time()
        seen = set()
        while time.time() - t0 < 20:
            line = p.stdout.readline()
            if not line.strip().startswith("{"):
                continue
            for rt in json.loads(line).get("neuron_runtime_data", []):
                seen.add(rt.get("pid"))
                mu = (
                    rt.get("report", {}).get("memory_used", {}).get("neuron_runtime_used_bytes", {})
                )
                per_core = mu.get("usage_breakdown", {}).get("neuroncore_memory_usage", {})
                ours = {c: v for c, v in per_core.items() if c in mine and sum(v.values()) > 0}
                if str(rt.get("pid")) == str(os.getpid()) or ours:
                    gb = {c: round(sum(v.values()) / 2**30, 3) for c, v in ours.items()}
                    return {
                        "pid": rt.get("pid"),
                        "device_gb": round((mu.get("neuron_device") or 0) / 2**30, 3),
                        "per_core_gb": gb,
                        "per_core": ours,
                    }
        return {"error": "no runtime on our cores", "pids": sorted(map(str, seen))}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        if p is not None:
            p.kill()


def _observation(cfg, tokenizer, seed: int, task: str):
    """A fixed, recorded-style observation: seeded images, a real prompt with a seeded state."""
    import numpy as np

    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.processor_pi05 import (
        build_pi05_prompt,
        tokenize_prompt,
    )

    g = torch.Generator().manual_seed(seed)
    r = cfg.image_resolution[0]
    images = [torch.rand(1, 3, r, r, generator=g) * 2 - 1 for _ in range(cfg.max_cameras)]
    masks = [torch.tensor([True]) for _ in range(cfg.max_cameras)]
    state = (torch.rand(cfg.state_dim, generator=g) * 2 - 1).numpy().astype(np.float32)
    prompt = build_pi05_prompt(task=task, normalized_state=state, state_num_bins=cfg.state_num_bins)
    ids, att = tokenize_prompt(tokenizer, prompt, cfg.tokenizer_max_length)
    tokens = torch.tensor([ids], dtype=torch.long)
    tmask = torch.tensor([att], dtype=torch.bool)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return images, masks, tokens, tmask, noise, prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--bar-k", type=float, default=2.0, help="bar = k x CPU bf16 rel-L2 + 0.5%%")
    ap.add_argument("--max-rel", type=float, default=None, help="fixed rel-L2 bar instead")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-cpu-ref", action="store_true")
    ap.add_argument(
        "--device", default="neuron", choices=["neuron", "cpu"], help="cpu: dry run of the script"
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    report: dict = {"model": os.path.basename(os.path.normpath(args.model))}
    cfg = Pi05Config.from_pretrained(args.model)
    report["policy_type"] = cfg.policy_type
    tok = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right")
    images, masks, tokens, tmask, noise, prompt = _observation(cfg, tok, args.seed, args.task)
    report["prompt_tokens"] = int(tmask.sum())
    torch.save(
        {
            "images": images,
            "masks": masks,
            "tokens": tokens,
            "tmask": tmask,
            "noise": noise,
            "prompt": prompt,
        },
        os.path.join(args.out, "observation.pt"),
    )

    # CPU references: upstream fp32 (the oracle) and our graphs in bf16 on the CPU.
    refs = {}
    if not args.skip_cpu_ref:
        t0 = time.time()
        m32 = NeuronPi05ActionModel(cfg, dtype=torch.float32)
        m32.load_checkpoint(args.model)
        with torch.no_grad():
            refs["cpu_fp32_upstream"] = m32.ref.sample_actions(
                images, masks, tokens, tmask, noise=noise.clone(), num_steps=args.steps
            )
            refs["cpu_fp32_graphs"] = m32.sample_actions(
                images, masks, tokens, tmask, noise=noise.clone(), num_steps=args.steps
            )
        del m32
        m16 = NeuronPi05ActionModel(cfg, dtype=torch.bfloat16)
        m16.load_checkpoint(args.model)
        with torch.no_grad():
            refs["cpu_bf16_graphs"] = m16.sample_actions(
                images, masks, tokens, tmask, noise=noise.clone(), num_steps=args.steps
            )
        del m16
        report["cpu_ref_s"] = round(time.time() - t0, 1)
        torch.save(refs, os.path.join(args.out, "cpu_refs.pt"))

    # Device.
    dev = torch.device("neuron", 0) if args.device == "neuron" else torch.device("cpu")
    t0 = time.time()
    m = NeuronPi05ActionModel(cfg, dtype=torch.bfloat16)
    m.load_checkpoint(args.model)
    report["host_load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    m.to(dev)
    report["to_device_s"] = round(time.time() - t0, 1)
    if args.device == "neuron":
        m.compile(get_compile_backend_name())
    outs, times = [], []
    for i in range(args.repeat):
        t0 = time.time()
        with torch.no_grad():
            outs.append(
                m.sample_actions(
                    images, masks, tokens, tmask, noise=noise.clone(), num_steps=args.steps
                )
            )
        times.append(time.time() - t0)
        if i == 0:
            report["first_call_s"] = round(times[0], 1)
            report["first_prefix_s"] = round(m.stats["prefix_s"], 1)
    warm = times[1:] or times
    report["warm_s"] = round(min(warm), 4)
    import vllm_omni_neuron.diffusion.models.pi0.model as pi_model

    pi_model.PROFILE = True  # sync after the prefix graph so the split below is real
    with torch.no_grad():
        m.stats.update(prefix_s=0.0, denoise_s=0.0)
        m.sample_actions(images, masks, tokens, tmask, noise=noise.clone(), num_steps=args.steps)
    pi_model.PROFILE = False
    report["warm_prefix_s"] = round(m.stats["prefix_s"], 4)
    report["warm_denoise_s"] = round(m.stats["denoise_s"], 4)
    report["deterministic"] = all(torch.equal(outs[0], o) for o in outs[1:])
    report["hbm"] = _hbm_snapshot() if args.device == "neuron" else None
    torch.save(outs[0], os.path.join(args.out, "device_actions.pt"))

    ok = report["deterministic"] and bool(torch.isfinite(outs[0]).all())
    if refs:
        ref = refs["cpu_fp32_upstream"]
        report["graphs_fp32_vs_upstream"] = _metrics(refs["cpu_fp32_graphs"], ref)
        report["cpu_bf16_vs_fp32"] = _metrics(refs["cpu_bf16_graphs"], ref)
        report["device_vs_fp32"] = _metrics(outs[0], ref)
        report["device_vs_cpu_bf16"] = _metrics(outs[0], refs["cpu_bf16_graphs"])
        bar = args.max_rel
        if bar is None:
            bar = args.bar_k * report["cpu_bf16_vs_fp32"]["rel_l2"] + 0.005
        report["bar_rel_l2"] = bar
        ok = ok and report["device_vs_fp32"]["rel_l2"] <= bar
    report["ok"] = bool(ok)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
