# SPDX-License-Identifier: Apache-2.0
"""Per-graph warm latency of GR00T on one NeuronCore, plus action parity against a saved upstream
reference. One configuration per process: the ``GR00T_*`` env knobs are read at trace time.

Reports the median of ``--iters`` warm calls for each graph (inputs already on device), the
whole device chain, and ``get_action`` -- the model API the pipeline calls, i.e. the device chain
plus host table prep and host->device uploads.

    GR00T_FUSED_BACKBONE=1 python examples/gr00t/perf_profile.py --model M --reference ref.pt --tag fused
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--reference", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

from vllm_omni_neuron.diffusion.models.gr00t.model import (  # noqa: E402
    NeuronGr00tModel,
    pick_bucket,
)

ref = torch.load(args.reference, weights_only=False)
inputs, noise = ref["inputs"], ref["noise"]
dev = torch.device("neuron", 0)
m = NeuronGr00tModel.from_pretrained(args.model, dtype=torch.bfloat16, device=dev)
m.compile(backend=get_compile_backend_name())
dt = torch.bfloat16
bucket = pick_bucket(int(inputs["attention_mask"].sum()))
t0 = time.perf_counter()
bi = m.prep(
    inputs["input_ids"],
    inputs["attention_mask"],
    inputs["pixel_values"],
    inputs["image_grid_thw"],
    inputs.get("mm_token_type_ids"),
    bucket=bucket,
)
prep_cold_ms = 1000 * (time.perf_counter() - t0)


def d(x, dtype=None):
    return (x.to(dtype) if dtype is not None else x).contiguous().to(dev)


vis_in = (
    d(bi.pixels, dt),
    d(bi.pos_index),
    d(bi.pos_weight, torch.float32),
    d(bi.vis_cos, torch.float32),
    d(bi.vis_sin, torch.float32),
    bi.n_images,
)
txt_in = (
    d(bi.input_ids),
    d(bi.image_index),
    d(bi.image_keep),
    d(bi.txt_cos, dt),
    d(bi.txt_sin, dt),
    d(bi.txt_bias),
)
head_in = (
    d(bi.valid),
    d(bi.image_mask),
    d(inputs["state"], dt),
    d(noise, dt),
    d(inputs["embodiment_id"].long().reshape(-1)),
)
H = m._fn("head")
if m.fused_backbone:
    B = m._fn("backbone")

    def backbone():
        return B(*vis_in, *txt_in)
else:
    V, T = m._fn("vision"), m._fn("text")

    def backbone():
        vis = V(*vis_in)
        return T(*txt_in[:3], vis[0], *txt_in[3:], *vis[1:])


def run():
    hid = backbone()
    return hid, H(hid, *head_in)


t0 = time.perf_counter()
with torch.no_grad():
    hid, act = run()
    act.cpu()
first_s = time.perf_counter() - t0


def timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        (r[-1] if isinstance(r, tuple) else r).cpu()
        out.append(1000 * (time.perf_counter() - t))
    out.sort()
    return round(out[len(out) // 2], 2)


with torch.no_grad():
    res = {}
    if m.fused_backbone:
        res["backbone_ms"] = timed(backbone, args.iters)
    else:
        vis = V(*vis_in)
        res["vision_ms"] = timed(lambda: V(*vis_in), args.iters)
        res["text_ms"] = timed(lambda: T(*txt_in[:3], vis[0], *txt_in[3:], *vis[1:]), args.iters)
    res["head_ms"] = timed(lambda: H(hid, *head_in), args.iters)
    res["total_ms"] = timed(lambda: run()[1], args.iters)
    m.get_action(inputs, noise=noise)  # warms the host/device table caches
    res["get_action_ms"] = timed(
        lambda: m.get_action(inputs, noise=noise)["action_pred"], args.iters
    )
    res["get_action_prep_ms"] = round(1000 * m.stats["prep_s"], 2)
    act = m.get_action(inputs, noise=noise)["action_pred"]
a, r = act.float(), ref["action_pred"].float()
summary = {
    "tag": args.tag,
    "fused_backbone": m.fused_backbone,
    "n_images": bi.n_images,
    "adaln_tables": os.environ.get("GR00T_ADALN_TABLES", "1"),
    "first_s": round(first_s, 1),
    "prep_cold_ms": round(prep_cold_ms, 2),
    **res,
    "vs_upstream_rel": round(((a - r).norm() / r.norm()).item(), 5),
    "vs_upstream_cos": round(
        torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0).item(), 6
    ),
    "vs_upstream_mse": float(f"{((a - r) ** 2).mean().item():.3e}"),
}
print(json.dumps(summary))
if args.out:
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"profile_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
