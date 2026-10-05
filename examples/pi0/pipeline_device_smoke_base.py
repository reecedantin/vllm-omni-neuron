#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""pi0 (base) M2 + M3 on one NeuronCore: serve one request through NeuronPi0Pipeline, compare its
action chunk with a CPU fp32 pipeline on the same observation + noise, and time each graph warm
(synced: every timed call ends with a host copy of its output).

    python examples/pi0/pipeline_device_smoke_base.py --model <pi0 dir> --out <run dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import numpy as np
import torch


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def _timeit(fn, n):
    out = []
    for _ in range(n):
        t0 = time.time()
        fn()
        out.append(time.time() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--max-rel", type=float, default=0.10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_neuron.envs import get_compile_backend_name
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0Pipeline
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi0.processor_pi0 import build_model_inputs

    def od(dtype):
        o = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
        for f in dataclasses.fields(OmniDiffusionConfig):
            setattr(o, f.name, None)
        o.model, o.dtype, o.model_config = args.model, dtype, {}
        o.model_config = {"tokenizer": args.tokenizer}
        return o

    g = torch.Generator().manual_seed(0)
    report: dict = {}
    cpu = NeuronPi0Pipeline(od_config=od("float32"))
    r = cpu.config.image_resolution[0]
    cams = [k for k in cpu.config.input_features if "image" in k]
    obs = {
        "prompt": args.task,
        **{k: torch.rand(r, r, 3, generator=g).numpy() for k in cams},
        "state": (torch.rand(cpu.config.max_state_dim, generator=g) * 2 - 1).numpy(),
    }
    noise = torch.randn(1, cpu.config.chunk_size, cpu.config.max_action_dim, generator=g)

    def req():
        sp = types.SimpleNamespace(
            extra_args={"robot_obs": dict(obs), "noise": noise.clone()},
            num_inference_steps=args.steps,
        )
        return types.SimpleNamespace(sampling_params=sp, prompts=[])

    ref = torch.from_numpy(cpu.forward(req()).output["actions"]).float()
    del cpu

    pipe = NeuronPi0Pipeline(od_config=od("bfloat16"))
    dev = torch.device("neuron", 0)
    pipe.to(dev)
    pipe.compile(get_compile_backend_name())
    t0 = time.time()
    out = pipe.forward(req())
    report["first_forward_s"] = round(time.time() - t0, 1)
    a = torch.from_numpy(np.asarray(out.output["actions"])).float()
    report["forward_warm_s"] = _timeit(lambda: pipe.forward(req()), args.repeat)
    a2 = torch.from_numpy(np.asarray(pipe.forward(req()).output["actions"])).float()

    # Per-graph, synced.
    m = pipe.model
    im, mk, lt, lm, st = build_model_inputs(obs, pipe.config, pipe.tokenizer, torch.device("cpu"))
    pix = torch.stack(im, dim=1).reshape(len(im), *im[0].shape[1:]).to(m.vision_dtype).to(dev)
    iv = torch.stack([x.float() for x in mk], dim=1).to(dev)
    lt, lmv, std = lt.to(dev), lm.float().to(dev), st.float().to(dev)
    with torch.no_grad():
        k, v, valid = m._prefix_fn(pix, iv, lt, lmv)
    tsc = m.time_sincos(args.steps, 1)[0].to(dev)
    x = noise.to(dev)

    def prefix():
        with torch.no_grad():
            m._prefix_fn(pix, iv, lt, lmv)[2].to("cpu")

    def denoise():
        with torch.no_grad():
            m._denoise_fn(std, x, tsc, k, v, valid).to("cpu")

    prefix()
    denoise()
    report["prefix_warm_s"] = _timeit(prefix, args.repeat)
    report["denoise_step_warm_s"] = _timeit(denoise, args.repeat)
    for key in ("forward_warm_s", "prefix_warm_s", "denoise_step_warm_s"):
        report[key + "_median"] = _median(report[key])

    rel = ((a - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), ref.flatten(), dim=0).item()
    report.update(
        device_vs_cpu_fp32={"rel_l2": rel, "cos": cos, "mse": torch.mean((a - ref) ** 2).item()},
        deterministic=bool(torch.equal(a, a2)),
        finite=bool(torch.isfinite(a).all()),
        actions_shape=list(a.shape),
    )
    report["ok"] = report["deterministic"] and report["finite"] and rel <= args.max_rel
    with open(os.path.join(args.out, "pipeline_device_smoke.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
