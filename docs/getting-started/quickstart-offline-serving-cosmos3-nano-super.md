# Quickstart: Offline generation with Cosmos3-Nano / Super on Neuron

<!-- meta: description: Generate an image or video offline with NVIDIA Cosmos3-Nano or Cosmos3-Super
on AWS Trainium2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Cosmos3-Nano, Cosmos3-Super, text-to-image,
image-to-video, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate an image or video offline on Trainium2 with the vLLM Omni
Neuron plugin. When you finish, you have a PNG or MP4 produced offline by the
[Cosmos3-Nano / Super](../models/cosmos3-nano-super.md) models.

## Prerequisites

- One SSH-accessible `trn2` instance.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`nvidia/Cosmos3-Nano` or `nvidia/Cosmos3-Super`, and the
  4-step distilled variants) from Hugging Face. The checkpoints are not gated.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from a stage
> config file: `examples/cosmos3_edge/cosmos3_nano_stage_trn2_tp4.yaml` for Nano (one chip, TP=4), or
> `examples/cosmos3_edge/cosmos3_super_stage_trn2_tp16.yaml` for Super (one torus row, TP=16).

## Step 1: Run a quick smoke test

```bash
python examples/cosmos3_edge/run.py --mode t2i --model-path /path/to/Cosmos3-Nano \
  --stage-config examples/cosmos3_edge/cosmos3_nano_stage_trn2_tp4.yaml \
  --height 256 --width 256 --steps 10 --output nano_smoke.png
```

Expected: a `nano_smoke.png` in the current directory. The first run compiles every graph on a cold
cache and takes several minutes; a repeat run with `--profile` reports a warm latency under a second.

## Step 2: Run a full offline generation

```bash
# Cosmos3-Nano, text-to-image at the checkpoint's own default (50 steps, guidance 7.0)
python examples/cosmos3_edge/run.py --mode t2i --model-path /path/to/Cosmos3-Nano \
  --stage-config examples/cosmos3_edge/cosmos3_nano_stage_trn2_tp4.yaml \
  --height 640 --width 640 --steps 50 --output nano_t2i.png

# Cosmos3-Super, image-to-video (TP=16)
python examples/cosmos3_edge/run.py --mode i2v --image conditioning.png --model-path /path/to/Cosmos3-Super \
  --stage-config examples/cosmos3_edge/cosmos3_super_stage_trn2_tp16.yaml \
  --height 480 --width 832 --num-frames 49 --steps 50 --output super_i2v.mp4

# Cosmos3-Super 4-step distilled I2V -- always 4 steps, a different (SDE) scheduler
python examples/cosmos3_edge/run.py --mode i2v --image conditioning.png \
  --model-path /path/to/Cosmos3-Super-i2v-4step \
  --stage-config examples/cosmos3_edge/cosmos3_super4step_i2v_stage_trn2_tp16.yaml \
  --height 640 --width 640 --num-frames 25 --steps 4 --output super4step_i2v.mp4
```

**Do not lower a base checkpoint's step count, and do not raise a 4-step checkpoint's.** See the
model card's Known limitations: running Nano / Super base below their 50-step default visibly
undersamples the scheduler (moire, banding) and looks like a device defect but is not one.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `t2i` | `t2i`, `t2v`, `i2v`, `policy`, `forward_dynamics`, `inverse_dynamics` |
| `--steps` | per-mode (`run.py` `DEFAULTS`) | denoising steps; see the step-count note above |
| `--guidance-scale` | the pipeline's per-mode default (7.0 for T2I) | classifier-free guidance scale |
| `--image` | none | conditioning image (`i2v`, `policy`, `forward_dynamics`) |
| `--seed` | 1 | sampling seed |
| `--profile` | off | also run a warm (second) request and report its latency |

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | TP>1 runs multiple worker processes; unset `NEURON_RT_VISIBLE_CORES` and set `NEURON_VISIBLE_DEVICES` to your core range instead. |
| `ValueError: Logical devices must be non-negative integers` | With `NEURON_VISIBLE_DEVICES` set, a stage's `devices:` must be a comma list (`"0,1,2,3"`), not a range (`"0-3"`). |
| Moire / fine striping over the whole frame | Base (non-distilled) checkpoint run at too few steps; use its 50-step default (see Known limitations). |
| `NCC_IDDT901` during I2V / action | The untiled VAE encoder graph does not compile above 192 px on Trn2. Leave `COSMOS3_VAE_ENCODE_TILE` unset (192 px tiles is the Trn2 default) or set `COSMOS3_VAE_ENCODE=host`. |

## Clean up

Remove the Neuron compile cache (`$NEURON_COMPILE_CACHE_URL` / `TORCH_NEURONX_NEFF_CACHE_DIR`) to force
a fresh compile, and delete any output files under your chosen `--output` path.

## Next steps

- [Cosmos3-Nano / Super model card](../models/cosmos3-nano-super.md)
