#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M2 end-to-end parity: LeRobot pi052 full predict_action_chunk vs our serving pipeline.

Runs LeRobot's ``PI052Policy.predict_action_chunk`` (fp32 CPU) on a fixed observation — it
generates the low-level subtask with the LM head, builds the ``"User: {subtask}, State: {bins};"``
action prompt, and samples the action chunk — and saves the observation + resulting actions. The
companion ``e2e_parity_ours.py`` (shared venv) feeds the identical raw observation to
``NeuronPi05Pipeline.forward`` and compares.

    PYTHONPATH=<lerobot checkout>/src \
      <lerobot reference venv>/bin/python examples/pi0/lerobot_e2e_parity.py \
        --model <pi052 checkpoint dir> --out <run dir>

Gate: action MSE <= 1e-3 AND cos >= 0.999 vs LeRobot fp32.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_lsp",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot_subtask_parity.py"),
    )
    L = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(L)
    from lerobot.utils.constants import OBS_STATE

    policy, cfg = L._lerobot_policy(args.model)
    tok_dir = args.tokenizer if os.path.isdir(args.tokenizer) else "google/paligemma-3b-pt-224"
    cfg.tokenizer_name = (
        tok_dir  # _prepare_action_batch / select_message build the tokenizer from this
    )
    policy.config.tokenizer_name = tok_dir
    policy.reset()

    g = torch.Generator().manual_seed(args.seed)
    r = cfg.image_resolution[0]
    cam_keys = [k for k in cfg.input_features if "image" in k]
    # Raw [0,1] images (LeRobot normalizes to [-1,1]); CHW as the env delivers.
    images01 = {k: torch.rand(1, 3, r, r, generator=g) for k in cam_keys}
    state = torch.rand(1, cfg.max_state_dim, generator=g) * 2 - 1
    torch.save(
        {"images01": images01, "state": state, "task": args.task, "cam_keys": cam_keys},
        os.path.join(args.out, "e2e_obs.pt"),
    )

    batch = {**images01, OBS_STATE: state, "task": [args.task]}
    # Fix the flow-matching noise so the comparison is deterministic across the two runtimes
    # (predict_action_chunk draws noise from the global RNG otherwise).
    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=torch.Generator().manual_seed(args.seed + 100),
        dtype=torch.float32,
    )
    with torch.no_grad():
        prepared = policy._prepare_action_batch(dict(batch))
        images_p, img_masks_p = policy._preprocess_images(prepared)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        acts = policy.model.sample_actions(
            images_p,
            img_masks_p,
            prepared[OBS_LANGUAGE_TOKENS],
            prepared[OBS_LANGUAGE_ATTENTION_MASK],
            noise=noise.clone(),
            num_steps=args.steps,
        )
    actions = acts[0, :, : cfg.output_features["action"].shape[0]].float()
    subtask = (policy.last_subtasks or [None])[0]
    torch.save(
        {"actions": actions, "subtask": subtask, "noise": noise},
        os.path.join(args.out, "lerobot_e2e.pt"),
    )
    print(
        json.dumps(
            {
                "subtask": subtask,
                "actions_shape": list(actions.shape),
                "actions_mean": float(actions.mean()),
                "actions_std": float(actions.std()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
