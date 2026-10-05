#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M1 LeRobot parity reference for pi0.5 / pi0.52.

Runs the LeRobot ``PI05Pytorch`` action kernel (the ``pi052`` subclass, which is the real
model class for ``lerobot/pi052_base``) in fp32 on CPU over the SAME raw inputs saved by
``examples/pi0/device_check.py`` (``observation.pt``), and compares it against the Neuron
device actions (``device_actions.pt``). This is the lower-level ``model.sample_actions`` entry,
so it bypasses the LeRobot batch/processor pipeline (image normalization, hierarchical subtask
generation) that our Neuron path does not yet implement — an apples-to-apples check of the flow
head itself. Run in the A11 LeRobot reference venv:

    PYTHONPATH=<lerobot checkout>/src \
      <lerobot reference venv>/bin/python examples/pi0/lerobot_parity.py \
        --model <pi052 checkpoint dir> --run <run dir>

Gate: action MSE <= 1e-3 AND cos >= 0.999 vs LeRobot fp32.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def _load_lerobot_model(model_dir: str):
    """Build the LeRobot PI05Pytorch kernel for a pi052 checkpoint and load its weights (fp32)."""
    from lerobot.policies.pi052.configuration_pi052 import PI052Config
    from lerobot.policies.pi052.modeling_pi052 import PI05Pytorch

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    raw.pop("type", None)
    allowed = {f.name for f in __import__("dataclasses").fields(PI052Config)}
    cfg = PI052Config(**{k: v for k, v in raw.items() if k in allowed})
    cfg.device = "cpu"
    model = PI05Pytorch(cfg)

    import safetensors.torch

    sd = safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))
    sd = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in sd.items()}
    own = dict(model.named_parameters())

    def fix(k: str) -> str:
        # This transformers version flattens SigLIP (no ``vision_model.`` segment); the LeRobot
        # checkpoint carries it. Drop it when the flattened name is the one the model owns.
        vt = "paligemma_with_expert.paligemma.model.vision_tower."
        if k.startswith(vt + "vision_model.") and (vt + k[len(vt + "vision_model.") :]) in own:
            return vt + k[len(vt + "vision_model.") :]
        return k

    sd = {fix(k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in own}  # drop tied lm_head etc. the kernel does not own
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m and not m.endswith(".inv_freq")]
    if missing:
        raise RuntimeError(f"LeRobot model missing {len(missing)} params: {missing[:8]}")
    model.eval().float()
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--run", required=True, help="device_check.py --out dir (observation.pt, device_actions.pt)"
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-mse", type=float, default=1e-3)
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args()

    obs = torch.load(os.path.join(args.run, "observation.pt"), weights_only=False)
    images = [im.float() for im in obs["images"]]
    masks = [m for m in obs["masks"]]
    tokens, tmask, noise = obs["tokens"].long(), obs["tmask"], obs["noise"].float()

    model, cfg = _load_lerobot_model(args.model)
    img_masks = [m.to(torch.bool).reshape(-1) for m in masks]
    with torch.no_grad():
        ref = model.sample_actions(
            images,
            img_masks,
            tokens,
            tmask.to(torch.bool),
            noise=noise.clone(),
            num_steps=args.steps or cfg.num_inference_steps,
        )
    ref = ref[:, :, : cfg.max_action_dim].float()

    report = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "ref": "lerobot_pi052_fp32_cpu",
    }
    cmp = {}
    dev_path = os.path.join(args.run, "device_actions.pt")
    cpu_refs_path = os.path.join(args.run, "cpu_refs.pt")
    if os.path.exists(dev_path):
        cmp["neuron_bf16"] = torch.load(dev_path, weights_only=False).float()
    if os.path.exists(cpu_refs_path):
        r = torch.load(cpu_refs_path, weights_only=False)
        cmp["our_cpu_fp32"] = r["cpu_fp32_graphs"].float()
    ok = bool(cmp)
    a, b = ref.flatten(), None
    for name, act in cmp.items():
        b = act.flatten()
        mse = torch.mean((b - a) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((b - a).norm() / a.norm().clamp_min(1e-12)).item()
        report[f"{name}_vs_lerobot"] = {
            "mse": mse,
            "cos": cos,
            "rel_l2": rel,
            "max_abs": (b - a).abs().max().item(),
        }
        if name == "neuron_bf16":
            ok = ok and (mse <= args.max_mse and cos >= args.min_cos)
    report["gate"] = {"max_mse": args.max_mse, "min_cos": args.min_cos}
    report["ok"] = ok
    with open(os.path.join(args.run, "lerobot_parity.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
