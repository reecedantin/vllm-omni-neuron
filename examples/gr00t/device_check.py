# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 on one NeuronCore: compile the three graphs, run, and check against CPU.

Three-way check (per the plugin's onboarding guide): FP32 CPU (this port; equal to upstream
to ~1e-6, see ``test/unit/test_gr00t_tiny.py``) vs BF16 CPU (dtype error alone) vs BF16
Neuron (adds the device error). Optionally also against a saved upstream reference
(``--reference``, written by ``examples/gr00t/reference.py``).

    python examples/gr00t/device_check.py --model /path/to/gr00t-n17 --out $FLEET_RUNS/m0 [--reference ref.pt]

Prints one JSON line; exits nonzero if the device result is outside ``--max-rel``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--reference",
    default=None,
    help="torch.save'd dict from reference.py (inputs, noise, action_pred)",
)
ap.add_argument("--warm", type=int, default=10, help="warm timed device calls")
ap.add_argument(
    "--max-rel", type=float, default=0.05, help="fail threshold: device-vs-fp32 action rel-L2"
)
ap.add_argument(
    "--skip-cpu", action="store_true", help="skip the CPU bf16/fp32 runs (reference only)"
)
ap.add_argument("--bucket", type=int, default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "unit")
)

from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


summary: dict = {"model": args.model}
ref = None
if args.reference:
    ref = torch.load(args.reference, weights_only=False)
    inputs, noise = ref["inputs"], ref["noise"]
else:
    from test_gr00t_tiny import synthetic_inputs  # noqa: E402

    inputs = synthetic_inputs()
    noise = None

if noise is None:
    from vllm_omni_neuron.diffusion.models.gr00t.config import Gr00tConfig

    h = Gr00tConfig.from_model_dir(args.model).head
    noise = torch.randn(
        (1, int(h["action_horizon"]), int(h["max_action_dim"])),
        generator=torch.Generator().manual_seed(0),
    )

results = {}
if not args.skip_cpu:
    for name, dt in (("cpu_fp32", torch.float32), ("cpu_bf16", torch.bfloat16)):
        t0 = time.time()
        m = NeuronGr00tModel.from_pretrained(args.model, dtype=dt)
        t1 = time.time()
        results[name] = m.get_action(inputs, noise=noise, bucket=args.bucket)["action_pred"]
        summary[f"{name}_s"] = round(time.time() - t1, 3)
        summary[f"{name}_load_s"] = round(t1 - t0, 2)
        del m

from vllm_neuron.envs import get_compile_backend_name  # noqa: E402

dev = torch.device("neuron", 0)
t0 = time.time()
m = NeuronGr00tModel.from_pretrained(args.model, dtype=torch.bfloat16, device=dev)
summary["device_load_s"] = round(time.time() - t0, 2)
summary["params_b"] = round(sum(p.numel() for p in m.parameters()) / 1e9, 3)
summary["weights_gb"] = round(sum(p.numel() * p.element_size() for p in m.parameters()) / 2**30, 3)
m.compile(backend=get_compile_backend_name())
t0 = time.time()
out = m.get_action(inputs, noise=noise, bucket=args.bucket)["action_pred"]
summary["first_call_s"] = round(time.time() - t0, 2)  # includes compile (or NEFF cache load)
lat = []
for _ in range(args.warm):
    t0 = time.time()
    out2 = m.get_action(inputs, noise=noise, bucket=args.bucket)["action_pred"]
    lat.append(time.time() - t0)
summary["warm_ms_mean"] = round(1000 * sum(lat) / max(len(lat), 1), 2)
summary["warm_ms_min"] = round(1000 * min(lat), 2) if lat else None
summary["prep_ms"] = round(1000 * m.stats["prep_s"], 2)
summary["bucket"] = m.stats["bucket"]
summary["real_len"] = m.stats["real_len"]
summary["deterministic"] = bool(torch.equal(out, out2))
summary["finite"] = bool(torch.isfinite(out).all())
summary["host_peak_rss_gb"] = round(rss_gb(), 2)
results["neuron_bf16"] = out
if ref is not None:
    results["upstream"] = ref["action_pred"]

base = results.get("cpu_fp32", results.get("upstream"))
for k, v in results.items():
    if v is base:
        continue
    summary[f"{k}_rel"] = round(rel(v, base), 5)
    summary[f"{k}_cos"] = round(cos(v, base), 6)
    summary[f"{k}_mse"] = float(f"{mse(v, base):.3e}")
if "upstream" in results and base is not results["upstream"]:
    for k in ("cpu_fp32", "cpu_bf16", "neuron_bf16"):
        if k in results:
            summary[f"{k}_vs_upstream_rel"] = round(rel(results[k], results["upstream"]), 6)
            summary[f"{k}_vs_upstream_mse"] = float(f"{mse(results[k], results['upstream']):.3e}")
torch.save({k: v.cpu() for k, v in results.items()}, os.path.join(args.out, "actions.pt"))
ok = summary["finite"] and summary.get("neuron_bf16_rel", 0.0) <= args.max_rel
summary["ok"] = ok
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary))
sys.exit(0 if ok else 1)
