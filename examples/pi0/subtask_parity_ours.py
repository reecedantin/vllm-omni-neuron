#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Second half of the M1b check: load the LeRobot reference text + saved observation produced by
``lerobot_subtask_parity.py`` (run in the separate LeRobot venv) and compare against our
``NeuronPi05ActionModel.generate_subtask`` on the same inputs. Kept as a separate process/script
because the two reference venvs cannot share one Python process (conflicting torch/transformers).

    python examples/pi0/subtask_parity_ours.py --model <pi052 checkpoint dir> --run <run dir>
"""

from __future__ import annotations

import argparse
import json
import os

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi05.config import Pi05Config

    obs = torch.load(os.path.join(args.run, "subtask_obs.pt"), weights_only=False)
    images = [im.float() for im in obs["images"]]
    masks = [m.bool() for m in obs["masks"]]
    task = obs["task"]

    with open(os.path.join(args.run, "lerobot_subtask_ref.json")) as f:
        ref = json.load(f)

    cfg = Pi05Config.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right")
    m = NeuronPi05ActionModel(cfg, dtype=torch.float32)
    m.load_checkpoint(args.model)
    m.enable_subtask_generation(tok)
    ours_text = m.generate_subtask(images, masks, task, max_new_tokens=args.max_new_tokens)

    report = dict(ref)
    report["ours_text"] = ours_text
    report["ok"] = ours_text == ref["lerobot_text_no_cache"]
    with open(os.path.join(args.run, "subtask_parity.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
