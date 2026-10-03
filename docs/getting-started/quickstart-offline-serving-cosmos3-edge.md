# Quickstart: Offline generation with Cosmos3-Edge on Inferentia2

<!-- meta: description: Generate an image, a video, or a robot action chunk offline with NVIDIA Cosmos3-Edge on AWS
Inferentia2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full two-core run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Cosmos3-Edge, text-to-image, image-to-video, robot policy,
offline, Inferentia2, inf2, NeuronCore-v2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate media offline with the vLLM Omni Neuron plugin on an inf2.8xlarge. When you
finish, you will have an image, an image-to-video clip and a robot action chunk from the
[Cosmos3-Edge](../models/cosmos3-edge.md) world model.

## Prerequisites

- One SSH-accessible `inf2.8xlarge` (or larger) instance.
- The environment prepared with the [setup guide](setup-guide.md).
- Network access to download `nvidia/Cosmos3-Edge` from Hugging Face. It is about 9 GB and not gated. The license is
  OpenMDW 1.1.

Inferentia2 has NeuronCore-v2, which does not use logical-core grouping. Set:

```bash
export NEURON_LOGICAL_NC_CONFIG=1
```

`examples/cosmos3_edge/run.py` sets it by default.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/cosmos3_edge/cosmos3_edge_stage.yaml` (one core) or `--stage-config`.

## Step 1: Run a quick smoke test

Run a 640x640 text-to-image generation on one NeuronCore:

```bash
python examples/cosmos3_edge/run.py --mode t2i --output edge_t2i.png
```

Expected: `edge_t2i.png`. The first run compiles the UND and GEN towers and the VAE decoder, which takes several
minutes. A repeated run takes about 4 s.

## Step 2: Run a full offline generation on both cores

Use the two-core fast stage, which applies CFG parallelism and a VAE decode split across both cores:

```bash
FAST=examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml

# Image-to-video, 832x480, 121 frames, 35 steps (~122 s warm)
python examples/cosmos3_edge/run.py --mode i2v --image frame0.png \
  --height 480 --width 832 --num-frames 121 --stage-config $FAST --output edge_i2v.mp4

# Robot policy: image + instruction -> 16 x 7 action chunk, actions only (no video decode)
python examples/cosmos3_edge/run.py --mode policy --image robot_view.png \
  --prompt "pick up the bowl and place it on the plate" \
  --domain droid_lerobot --raw-action-dim 7 --action-chunk 16 --action-only \
  --stage-config $FAST --output edge_policy
```

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `t2i` | `t2i`, `t2v`, `i2v`, `policy`, `forward_dynamics`, `inverse_dynamics` |
| `--height`, `--width`, `--num-frames` | per mode | Output shape (t2v/i2v: 480x832x49; action modes: 256x256x17) |
| `--steps` | per mode | Denoising steps (t2i 50, t2v/i2v 35, action modes 30) |
| `--guidance-scale` | per mode | CFG scale |
| `--resolution` | `256` | Resolution class for the action modes (`256`, `480`, `720`) |
| `--action-only` | off | Action modes: return actions only and skip the VAE decode |
| `--profile` | off | Run a warm-up, then a timed run |

## Common issues

| Symptom | Fix |
|---|---|
| A long pause on the first request | Normal: each new shape compiles once, for roughly 10-35 minutes. Repeated shapes use the compile cache. |
| `NEURON_LOGICAL_NC_CONFIG` error on inf2 | Set `NEURON_LOGICAL_NC_CONFIG=1`. NeuronCore-v2 has no LNC=2. |
| Guardrail import errors | The example stages set `guardrails: false`. Enabling guardrails needs the `cosmos-guardrail` package. |

## Clean up

Remove the generated outputs. Keep the Neuron compile cache if you plan to rerun the same shapes.

## Next steps

- [Cosmos3-Edge model card](../models/cosmos3-edge.md)
- [Tutorial: Deploy Cosmos3-Edge with vLLM Omni Neuron](../tutorials/tutorial-cosmos3-edge.md)
