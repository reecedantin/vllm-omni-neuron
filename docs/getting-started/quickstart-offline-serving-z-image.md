# Quickstart: Offline text-to-image with Z-Image on Neuron

<!-- meta: description: Generate an image offline with Z-Image or Z-Image-Turbo on AWS Trainium2 using the vLLM Omni
Neuron plugin. Covers a quick smoke test and a full 1024 x 1024 run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Z-Image, Z-Image-Turbo, text-to-image, offline, trn2,
quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate an image on Trainium2 with the vLLM Omni Neuron plugin. When you finish, you
have a PNG produced offline by the [Z-Image](../models/z-image.md) model.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`Tongyi-MAI/Z-Image-Turbo` or `Tongyi-MAI/Z-Image`) from Hugging Face. The
  checkpoints are not gated.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/z_image/z_image_stage.yaml` (four logical cores: TP 2 x CFG 2, for Z-Image). The Turbo commands below
> pass `examples/z_image/z_image_stage_tp4.yaml` (TP 4 on the same four cores), since Turbo has no CFG branch.

## Step 1: Run a quick smoke test

```bash
python examples/z_image/run.py --model-path Tongyi-MAI/Z-Image-Turbo --height 512 --width 512 \
    --steps 9 --guidance-scale 0 --stage-config examples/z_image/z_image_stage_tp4.yaml --output turbo_512.png
```

Expected: `turbo_512.png` is written. The first run compiles the graphs, which takes a few minutes; later runs reuse
the compile cache.

## Step 2: Run a full offline generation

Z-Image-Turbo (guidance-distilled, no CFG):

```bash
python examples/z_image/run.py --model-path Tongyi-MAI/Z-Image-Turbo --height 1024 --width 1024 \
    --steps 9 --guidance-scale 0 --stage-config examples/z_image/z_image_stage_tp4.yaml --profile --output turbo_1024.png
```

Z-Image base (CFG, model-card defaults):

```bash
python examples/z_image/run.py --model-path Tongyi-MAI/Z-Image --height 1024 --width 1024 \
    --steps 50 --guidance-scale 4 --profile --output base_1024.png
```

`--profile` runs the request twice and prints the warm latency.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height`, `--width` | 1024 | Output size; multiples of 16 |
| `--steps` | 9 | Scheduler steps (Turbo: 9; base: 28 to 50) |
| `--guidance-scale` | 0.0 | CFG scale; 0 disables CFG (Turbo), 3 to 5 for base |
| `--cfg-normalize` | off | Z-Image CFG renormalization threshold |
| `--negative-prompt` | none | Negative prompt (base only) |
| `--seed` | 1 | Random seed |

## Common issues

| Symptom | Fix |
|---|---|
| `Could not load the model status=4 message=Allocation Failure` at 1024 px | Keep `tensor_parallel_size` at 2 or more (one of the shipped stage configs) |
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Set `NEURON_VISIBLE_DEVICES` to the cores to use and unset `NEURON_RT_VISIBLE_CORES` |
| First request times out | Raise `Z_IMAGE_HANDSHAKE_TIMEOUT_S` (default 3600) |

## Clean up

Remove the generated PNGs. The compile cache lives under `NEURON_COMPILE_CACHE_URL`; delete it to force a fresh
compile.

## Next steps

- [Z-Image model card](../models/z-image.md)
