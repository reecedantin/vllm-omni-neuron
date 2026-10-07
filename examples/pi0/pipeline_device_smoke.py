#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M2 device serving smoke: build NeuronPi05Pipeline, move it to a NeuronCore, compile the graphs,
and run one pi0.52 request (subtask generation + action chunk) end to end on device.

    python examples/pi0/pipeline_device_smoke.py --model <pi052 checkpoint dir> --out runs/m2dev

Prints a JSON line with the generated subtask, action shape, finiteness, and timings. Exits
nonzero if the action chunk is not finite or has the wrong shape.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-rel", type=float, default=0.10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_neuron.envs import get_compile_backend_name
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline

    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype, od.model_config = args.model, "bfloat16", {}
    od.model_config = {"tokenizer": args.tokenizer}

    report: dict = {}
    t0 = time.time()
    pipe = NeuronPi05Pipeline(od_config=od)
    report["build_s"] = round(time.time() - t0, 1)

    g = torch.Generator().manual_seed(0)
    r = pipe.config.image_resolution[0]
    cams = [k for k in pipe.config.input_features if "image" in k]
    obs = {
        "prompt": args.task,
        **{k: (torch.rand(r, r, 3, generator=g)).numpy() for k in cams},
        "state": (torch.rand(pipe.config.state_dim, generator=g) * 2 - 1).numpy(),
    }
    noise = torch.randn(
        1,
        pipe.config.chunk_size,
        pipe.config.max_action_dim,
        generator=torch.Generator().manual_seed(2),
        dtype=torch.float32,
    )

    # CPU fp32 reference pipeline on the identical observation + noise, before moving the shared
    # device pipeline's model to the NeuronCore (so this stays on host).
    cpu_pipe = NeuronPi05Pipeline(od_config=od)
    cpu_pipe.model.dtype = torch.float32
    cpu_pipe.model._apply_dtypes()
    sp_cpu = types.SimpleNamespace(
        extra_args={"robot_obs": dict(obs), "noise": noise.clone()},
        num_inference_steps=args.steps,
        generator=None,
    )
    t0 = time.time()
    ref_out = cpu_pipe.forward(types.SimpleNamespace(sampling_params=sp_cpu, prompts=[]))
    report["cpu_ref_s"] = round(time.time() - t0, 1)
    ref_actions = torch.from_numpy(np.asarray(ref_out.output["actions"])).float()

    pipe.to(torch.device("neuron", 0))
    pipe.compile(get_compile_backend_name())

    sp = types.SimpleNamespace(
        extra_args={"robot_obs": dict(obs), "noise": noise.clone()},
        num_inference_steps=args.steps,
        generator=None,
    )
    for i in range(2):
        t0 = time.time()
        out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
        dt = time.time() - t0
        report["first_s" if i == 0 else "warm_s"] = round(dt, 3)
    a = np.asarray(out.output["actions"])
    report.update(
        error=out.error,
        actions_shape=list(a.shape),
        finite=bool(np.isfinite(a).all()),
        subtask_stats=pipe.model.stats,
    )
    dev_actions = torch.from_numpy(a).float()
    rel = ((dev_actions - ref_actions).norm() / ref_actions.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(
        dev_actions.flatten(), ref_actions.flatten(), dim=0
    ).item()
    report["device_vs_cpu"] = {
        "rel_l2": rel,
        "cos": cos,
        "mse": torch.mean((dev_actions - ref_actions) ** 2).item(),
    }
    ok = (
        out.error is None
        and a.shape == (pipe.config.chunk_size, pipe.config.action_dim)
        and report["finite"]
        and rel <= args.max_rel
    )
    report["ok"] = ok
    with open(os.path.join(args.out, "pipeline_device_smoke.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
