# SPDX-License-Identifier: Apache-2.0
"""Warm-latency benchmark for InternVLA-A1.5: mean over N served action requests, broken down into
the VLM prefix (once/request), the denoise loop (num_inference_steps Euler steps) and the host prep.

    python examples/internvla/bench_warm.py --model /path/to/InternVLA-A1.5-base \
        --vlm-config /path/to/qwen3.5-config [--n-images 3 --grid 16 16 --repeat 10]
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_neuron.envs import get_compile_backend_name
from vllm_omni_neuron.diffusion.models.internvla import InternVLAA15, InternVLAA15Runner
from vllm_omni_neuron.diffusion.models.internvla import preprocess as pp

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint directory or Hub id")
    ap.add_argument("--vlm-config", default=None, help="Qwen3.5-2B config.json (see config.py for resolution order)")
    ap.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--grid", type=int, nargs=2, default=(16, 16))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=10, help="warm requests to average after the first")
    a = ap.parse_args()

    dev = torch.device("neuron", 0) if a.device == "neuron" else torch.device("cpu")
    be = get_compile_backend_name()

    def wrap(mod, name):
        if a.device == "cpu":
            return mod
        return torch.compile(mod, backend=be, fullgraph=True, dynamic=False,
                             options={"model_name": name, "compiler_args": list(COMPILER_ARGS)})

    m = InternVLAA15.from_pretrained(a.model, dtype=torch.bfloat16, device=dev, vlm_config=a.vlm_config)
    r = InternVLAA15Runner(m, dev, wrap)
    batch = pp.synthetic_request(m.cfg, n_images=a.n_images, grid=tuple(a.grid), seed=a.seed)
    noise = pp.initial_noise(m.cfg, seed=a.seed)

    _ = r.sample_actions(batch, noise)  # warm every graph

    prefix, denoise, total = [], [], []
    for _ in range(a.repeat):
        _ = r.sample_actions(batch, noise)
        prefix.append(r.timings["prefix_s"])
        denoise.append(r.timings["denoise_s"])
        total.append(r.timings["total_s"])

    print(f"prefix:  min={min(prefix)*1000:.2f}ms mean={st.mean(prefix)*1000:.2f}ms max={max(prefix)*1000:.2f}ms")
    print(f"denoise: min={min(denoise)*1000:.2f}ms mean={st.mean(denoise)*1000:.2f}ms")
    print(f"total:   min={min(total)*1000:.2f}ms mean={st.mean(total)*1000:.2f}ms")


if __name__ == "__main__":
    main()
