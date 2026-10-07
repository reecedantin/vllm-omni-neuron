#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M1 LeRobot parity reference for pi0 (base).

Runs LeRobot's ``PI0Pytorch.sample_actions`` (fp32, CPU) over the SAME raw inputs saved by
``examples/pi0/device_check_base.py`` (``observation.pt``) and compares against the Neuron device
actions (``device_actions.pt``) and our CPU fp32 graphs (``cpu_refs.pt``). Run in a CPU-only
LeRobot reference venv with the LeRobot source on PYTHONPATH:

    PYTHONPATH=<lerobot>/src <lerobot-venv>/bin/python examples/pi0/lerobot_parity_base.py \
        --model <pi0 checkpoint dir> --run <device_check_base.py --out dir>

Gate: action MSE <= 1e-3 AND cos >= 0.999 vs LeRobot fp32.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch


def _lerobot_model(model_dir: str):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    raw.pop("type", None)
    allowed = {f.name for f in dataclasses.fields(PI0Config)}
    cfg = PI0Config(**{k: v for k, v in raw.items() if k in allowed})
    cfg.device = "cpu"
    model = PI0Pytorch(cfg)

    import safetensors.torch

    sd = safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))
    sd = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in sd.items()}
    own = dict(model.named_parameters())
    vt = "paligemma_with_expert.paligemma.model.vision_tower."

    def fix(k: str) -> str:
        # transformers >= 5.4 flattens SigLIP (no ``vision_model.``); the checkpoint carries it.
        if k.startswith(vt + "vision_model.") and (vt + k[len(vt + "vision_model.") :]) in own:
            return vt + k[len(vt + "vision_model.") :]
        if k == "paligemma_with_expert.paligemma.lm_head.weight" and k not in own:
            return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        return k

    sd = {fix(k): v for k, v in sd.items()}
    # The checkpoint stores PaliGemma's tied text embedding only as lm_head.weight.
    emb = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if emb not in sd and "paligemma_with_expert.paligemma.lm_head.weight" in sd:
        sd[emb] = sd["paligemma_with_expert.paligemma.lm_head.weight"]
    sd = {k: v for k, v in sd.items() if k in own}
    missing, _ = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m and not m.endswith(".inv_freq")]
    if missing:
        raise RuntimeError(f"LeRobot pi0 model missing {len(missing)} params: {missing[:8]}")
    model.eval().float()
    return model, cfg


def _metrics(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a, ref = a.float().flatten(), ref.float().flatten()
    return {
        "mse": torch.mean((a - ref) ** 2).item(),
        "cos": torch.nn.functional.cosine_similarity(a, ref, dim=0).item(),
        "rel_l2": ((a - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        "max_abs": (a - ref).abs().max().item(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-mse", type=float, default=1e-3)
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args()

    obs = torch.load(os.path.join(args.run, "observation.pt"), weights_only=False)
    model, cfg = _lerobot_model(args.model)
    with torch.no_grad():
        ref = model.sample_actions(
            [im.float() for im in obs["images"]],
            [m.bool().reshape(-1) for m in obs["masks"]],
            obs["tokens"].long(),
            obs["tmask"].bool(),
            obs["state"].float(),
            noise=obs["noise"].float().clone(),
            num_steps=args.steps or cfg.num_inference_steps,
        )
    ref = ref[:, :, : cfg.max_action_dim].float()

    report = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "ref": "lerobot_pi0_fp32_cpu",
    }
    dev = torch.load(os.path.join(args.run, "device_actions.pt"), weights_only=False)
    report["neuron_bf16_vs_lerobot"] = _metrics(dev, ref)
    refs_path = os.path.join(args.run, "cpu_refs.pt")
    if os.path.exists(refs_path):
        report["our_cpu_fp32_vs_lerobot"] = _metrics(
            torch.load(refs_path, weights_only=False)["cpu_fp32_graphs"], ref
        )
    n = report["neuron_bf16_vs_lerobot"]
    report["gate"] = {"max_mse": args.max_mse, "min_cos": args.min_cos}
    report["ok"] = n["mse"] <= args.max_mse and n["cos"] >= args.min_cos
    with open(os.path.join(args.run, "lerobot_parity.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
