# Quickstart: Offline action generation with pi0 / pi0.5 on Neuron

<!-- meta: description: Generate a robot action chunk offline with the LeRobot pi0, pi0.5 or pi0.52
Vision-Language-Action policies on AWS Trainium2 using the vLLM Omni Neuron plugin. Covers a quick
smoke test and a run with your own camera frames and robot state. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, pi0, pi0.5, pi0.52, LeRobot, VLA, robot policy,
action chunk, offline, Trainium2, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to run a LeRobot π0-family policy offline on Trainium2 with the vLLM
Omni Neuron plugin. When you finish, you have a JSON action chunk (50 steps x 32 action dims)
produced by the [π0.5 / π0.52](../models/pi05.md) or [π0](../models/pi0.md) model from one robot
observation.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` (or any Trainium2 instance; the model uses one NeuronCore).
- The environment prepared with the [setup guide](setup-guide.md).
- Network access to download the policy (`lerobot/pi052_base`, `lerobot/pi05_base` or
  `lerobot/pi0_base`, about 14-17 GB each) and the tokenizer from Hugging Face. The policies are
  derived from PaliGemma and distributed under the Gemma license; the tokenizer repo
  `google/paligemma-3b-pt-224` is gated, so accept its terms and log in (`hf auth login`) first.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/pi0/pi052_stage.yaml` (π0.5 / π0.52) or `examples/pi0/pi0_stage.yaml` (π0), picked
> from the checkpoint's policy type, or from `--stage-config`.

## Step 1: Run a quick smoke test

```bash
python examples/pi0/run.py --model lerobot/pi052_base
```

Expected: one action chunk written to `pi0_actions.json` with
`[run] pi052: actions [50, 32] finite=True`. The cameras get seeded random frames, so the actions
are not meaningful: this checks the install, the download and the compile. The first request
compiles the graphs (about 1-2 minutes); later runs reuse the compile cache.

For π0:

```bash
python examples/pi0/run.py --model lerobot/pi0_base
```

## Step 2: Run with your own observation

```bash
python examples/pi0/run.py --model lerobot/pi052_base \
  --task "pick up the red cube and place it in the bowl" \
  --image base_0_rgb=base.png --image left_wrist_0_rgb=left.png --image right_wrist_0_rgb=right.png \
  --state state.json --profile
```

`state.json` holds the robot state as a JSON list, already normalized the way the checkpoint was
trained (the base checkpoints ship no normalization statistics, so the state passes through).

### Change the inputs and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `lerobot/pi052_base` | Hugging Face repo id or local checkpoint directory |
| `--task` | a pick-and-place instruction | Natural-language instruction (π0.52 turns it into a subtask first) |
| `--image CAMERA=PATH` | seeded random frames | One camera frame; camera names come from the checkpoint |
| `--state` | zeros | JSON file with the robot state vector |
| `--steps` | `10` | Flow-matching integration steps |
| `--tokenizer` | the stage config's | PaliGemma tokenizer repo id or local directory |
| `--profile` | off | Time five warm requests after the first |
| `--output` | `pi0_actions.json` | Output file |

## Common issues

| Symptom | Fix |
|---|---|
| `401` / gated repo error loading the tokenizer | Accept the `google/paligemma-3b-pt-224` terms and log in, or pass `--tokenizer <local dir>` |
| The first request times out | Raise `PI0_HANDSHAKE_TIMEOUT_S` (default 3600 s) for a cold compile on a busy host |
| Actions are identical for different images | You are running the smoke test with seeded frames; pass `--image` |

## Clean up

Remove `pi0_actions.json`. The compile cache lives where `TORCH_NEURONX_NEFF_CACHE_DIR` points
(see the [setup guide](setup-guide.md)); delete it to force a recompile.

## Next steps

- [π0.5 / π0.52 model card](../models/pi05.md)
- [π0 model card](../models/pi0.md)
- [Tutorial: Deploy pi0 / pi0.5 / pi0.52 with vLLM Omni Neuron](../tutorials/tutorial-pi0.md)
