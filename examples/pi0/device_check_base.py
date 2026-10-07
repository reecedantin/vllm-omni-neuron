#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""pi0 (base) device check: load, compile, run the action path on one NeuronCore and compare
against the CPU fp32 upstream model (and the same graphs in bf16 on CPU, isolating pure dtype
error from the Neuron-specific error). Mirrors ``device_check.py`` (pi0.5); the only functional
difference is the extra ``state`` input and no hierarchical subtask path.

    python examples/pi0/device_check_base.py --model /path/to/pi0 --out /path/to/run [--steps 10]

Exits nonzero when the device result is not deterministic or misses the bar: rel-L2 vs CPU fp32
<= ``--bar-k`` x (CPU bf16 rel-L2) + 0.5% (``--max-rel`` replaces it with a fixed value).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch


def _metrics(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a, ref = a.float().flatten(), ref.float().flatten()
    return {
        "rel_l2": ((a - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        "cos": torch.nn.functional.cosine_similarity(a, ref, dim=0).item(),
        "mse": torch.mean((a - ref) ** 2).item(),
        "max_abs": (a - ref).abs().max().item(),
    }


def _observation(cfg, tokenizer, seed: int, task: str):
    g = torch.Generator().manual_seed(seed)
    r = cfg.image_resolution[0]
    images = [torch.rand(1, 3, r, r, generator=g) * 2 - 1 for _ in range(cfg.max_cameras)]
    masks = [torch.tensor([True]) for _ in range(cfg.max_cameras)]
    state = torch.rand(1, cfg.max_state_dim, generator=g) * 2 - 1
    enc = tokenizer(
        task,
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    tokens = torch.tensor([enc["input_ids"]], dtype=torch.long)
    tmask = torch.tensor([enc["attention_mask"]], dtype=torch.bool)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return images, masks, tokens, tmask, state, noise


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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi0.config import Pi0Config

    report: dict = {"model": os.path.basename(os.path.normpath(args.model))}
    cfg = Pi0Config.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right")
    images, masks, tokens, tmask, state, noise = _observation(cfg, tok, args.seed, args.task)
    report["prompt_tokens"] = int(tmask.sum())
    torch.save(
        {
            "images": images,
            "masks": masks,
            "tokens": tokens,
            "tmask": tmask,
            "state": state,
            "noise": noise,
        },
        os.path.join(args.out, "observation.pt"),
    )

    refs = {}
    if not args.skip_cpu_ref:
        t0 = time.time()
        m32 = NeuronPi0ActionModel(cfg, dtype=torch.float32)
        m32.load_checkpoint(args.model)
        with torch.no_grad():
            refs["cpu_fp32_upstream"] = m32.ref.sample_actions(
                images, masks, tokens, tmask, state, noise=noise.clone(), num_steps=args.steps
            )
            refs["cpu_fp32_graphs"] = m32.sample_actions(
                images, masks, tokens, tmask, state, noise=noise.clone(), num_steps=args.steps
            )
        del m32
        m16 = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
        m16.load_checkpoint(args.model)
        with torch.no_grad():
            refs["cpu_bf16_graphs"] = m16.sample_actions(
                images, masks, tokens, tmask, state, noise=noise.clone(), num_steps=args.steps
            )
        del m16
        report["cpu_ref_s"] = round(time.time() - t0, 1)
        torch.save(refs, os.path.join(args.out, "cpu_refs.pt"))

    dev = torch.device("neuron", 0)
    t0 = time.time()
    m = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
    m.load_checkpoint(args.model)
    report["host_load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    m.to(dev)
    report["to_device_s"] = round(time.time() - t0, 1)
    m.compile(get_compile_backend_name())
    outs, times = [], []
    for i in range(args.repeat):
        t0 = time.time()
        with torch.no_grad():
            outs.append(
                m.sample_actions(
                    images, masks, tokens, tmask, state, noise=noise.clone(), num_steps=args.steps
                )
            )
        times.append(time.time() - t0)
        if i == 0:
            report["first_call_s"] = round(times[0], 1)
            report["first_prefix_s"] = round(m.stats["prefix_s"], 1)
    warm = times[1:] or times
    report["warm_s"] = round(min(warm), 4)
    with torch.no_grad():
        m.stats.update(prefix_s=0.0, denoise_s=0.0)
        m.sample_actions(
            images, masks, tokens, tmask, state, noise=noise.clone(), num_steps=args.steps
        )
    report["warm_prefix_s"] = round(m.stats["prefix_s"], 4)
    report["warm_denoise_s"] = round(m.stats["denoise_s"], 4)
    report["deterministic"] = all(torch.equal(outs[0], o) for o in outs[1:])
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
    raise SystemExit(main())
