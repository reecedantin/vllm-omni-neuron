#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Second half of M2 e2e parity: feed the observation saved by ``lerobot_e2e_parity.py`` to our
``NeuronPi05Pipeline`` and compare the action chunk against LeRobot's full predict_action_chunk.

    python examples/pi0/e2e_parity_ours.py --model <pi052 checkpoint dir> --run <run dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-mse", type=float, default=1e-3)
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args()

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline

    obs = torch.load(os.path.join(args.run, "e2e_obs.pt"), weights_only=False)
    ref = torch.load(os.path.join(args.run, "lerobot_e2e.pt"), weights_only=False)
    ref_actions = ref["actions"].float()

    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype, od.model_config = args.model, "float32", {}
    od.model_config = {"tokenizer": args.tokenizer}
    pipe = NeuronPi05Pipeline(od_config=od)

    # Our pipeline's processor expects HWC images in [0,1]; the saved tensors are CHW [0,1].
    robot_obs = {"prompt": obs["task"], "state": obs["state"][0].numpy()}
    for k in obs["cam_keys"]:
        chw = obs["images01"][k][0]  # [3,H,W] in [0,1]
        robot_obs[k] = chw.permute(1, 2, 0).numpy()  # HWC

    sp = types.SimpleNamespace(
        extra_args={"robot_obs": robot_obs, "noise": ref.get("noise")},
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(0),
    )
    out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
    assert out.error is None, out.error
    ours = torch.from_numpy(np.asarray(out.output["actions"])).float()[
        : ref_actions.shape[0], : ref_actions.shape[1]
    ]

    a, b = ref_actions.flatten(), ours.flatten()
    mse = torch.mean((a - b) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((b - a).norm() / a.norm().clamp_min(1e-12)).item()
    ok = mse <= args.max_mse and cos >= args.min_cos
    report = {
        "lerobot_subtask": ref.get("subtask"),
        "actions_shape": list(ours.shape),
        "mse": mse,
        "cos": cos,
        "rel_l2": rel,
        "max_abs": (a - b).abs().max().item(),
        "gate": {"max_mse": args.max_mse, "min_cos": args.min_cos},
        "ok": ok,
    }
    with open(os.path.join(args.run, "e2e_parity.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
