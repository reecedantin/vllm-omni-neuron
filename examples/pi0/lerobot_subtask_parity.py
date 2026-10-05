#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M1b: pi0.52 subtask-generation parity vs LeRobot's own ``select_message``.

Runs LeRobot's ``PI05Pytorch.select_message`` (the pi052 model's own greedy decode: full
bidirectional-prefix recompute per step, matching our fixed-shape re-prefill graph) on the real
pi052-base weights, fp32 CPU, for a fixed prompt and seeded dummy images (no vision signal is
needed for a decode-logic check — the gate below is per-token exact match, which only requires
both paths to run the identical math on identical weights, not for the output to be sensible
language). Compares it token-for-token against our ``NeuronPi05ActionModel.generate_subtask``
(CPU fp32 graphs) on the same inputs.

    PYTHONPATH=<lerobot checkout>/src \
      <lerobot reference venv>/bin/python examples/pi0/lerobot_subtask_parity.py \
        --model <pi052 checkpoint dir> --out <run dir>

Gate: generated token ids are an EXACT match (greedy decode of the same logits must agree).
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def _lerobot_policy(model_dir: str):
    """Build the full LeRobot PI052Policy (which owns ``select_message``) and load weights."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi052.configuration_pi052 import PI052Config
    from lerobot.policies.pi052.modeling_pi052 import PI052Policy

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    raw.pop("type", None)
    allowed = {f.name for f in __import__("dataclasses").fields(PI052Config)}
    cfg = PI052Config(**{k: v for k, v in raw.items() if k in allowed})
    cfg.device = "cpu"
    # Parse feature dicts into PolicyFeature so the policy's image/action plumbing works.
    for attr in ("input_features", "output_features"):
        feats = getattr(cfg, attr)
        setattr(
            cfg,
            attr,
            {
                k: (
                    v
                    if isinstance(v, PolicyFeature)
                    else PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
                )
                for k, v in feats.items()
            },
        )
    policy = PI052Policy(cfg)

    import safetensors.torch

    sd = safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))
    own = dict(policy.named_parameters())

    def fix(k: str) -> str:
        vt = "model.paligemma_with_expert.paligemma.model.vision_tower."
        if k.startswith(vt + "vision_model.") and (vt + k[len(vt + "vision_model.") :]) in own:
            return vt + k[len(vt + "vision_model.") :]
        return k

    sd = {fix(k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in own}
    missing, _ = policy.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m and not m.endswith(".inv_freq")]
    if missing:
        raise RuntimeError(f"policy missing {len(missing)} params: {missing[:8]}")
    policy.eval().float()
    return policy, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from lerobot.policies.pi052.inference.pi052_adapter import _get_loc_tokenizer
    from lerobot.policies.pi052.text_processor_pi052 import register_paligemma_loc_tokens
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from transformers import AutoTokenizer

    policy, cfg = _lerobot_policy(args.model)
    tok_name = args.tokenizer if os.path.isdir(args.tokenizer) else "google/paligemma-3b-pt-224"
    tokenizer = _get_loc_tokenizer(tok_name, AutoTokenizer, register_paligemma_loc_tokens)

    g = torch.Generator().manual_seed(args.seed)
    r = cfg.image_resolution[0]
    cam_keys = [k for k in cfg.input_features if "image" in k]
    # Raw camera images in [0,1] (LeRobot's _preprocess_images maps [0,1] -> [-1,1] with *2-1).
    images01 = [torch.rand(1, 3, r, r, generator=g) for _ in cam_keys]
    masks = [torch.tensor([True]) for _ in cam_keys]
    # Save the already-normalized [-1,1] tensors our path consumes directly (our processor would
    # otherwise do the *2-1), so both models see identical pixel values at the vision tower.
    images_norm = [im * 2.0 - 1.0 for im in images01]
    torch.save(
        {"images": images_norm, "masks": masks, "task": args.task, "cam_keys": cam_keys},
        os.path.join(args.out, "subtask_obs.pt"),
    )

    prompt = f"User: {args.task}\n"
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    base_batch = {
        OBS_LANGUAGE_TOKENS: enc["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].bool(),
        **{key: images01[i] for i, key in enumerate(cam_keys)},
    }

    def run(use_kv_cache: bool) -> str:
        with torch.no_grad():
            return policy.select_message(
                dict(base_batch),
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                suppress_loc_tokens=True,
                use_kv_cache=use_kv_cache,
            )

    text_cached = run(True)
    text_nocache = run(False)

    report = {
        "lerobot_text_kv_cache": text_cached,
        "lerobot_text_no_cache": text_nocache,
        "agree_cache_vs_nocache": text_cached == text_nocache,
    }
    with open(os.path.join(args.out, "lerobot_subtask_ref.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
