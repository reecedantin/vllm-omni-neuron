# Quickstart: Offline image generation with FLUX.2-dev on Neuron

<!-- meta: description: Generate an image offline with FLUX.2-dev on AWS Trainium2 using the vLLM Omni Neuron
plugin. Covers a quick smoke test and a full 1024x1024 run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, FLUX.2-dev, text-to-image, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-06 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate an image on Trainium2 with the vLLM Omni Neuron plugin. When you finish,
you have a 1024x1024 PNG produced offline by the [FLUX.2-dev](../models/flux2-dev.md) model.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance (32 of its 64 logical NeuronCores are used; 8 with the TP=8 alternative).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`black-forest-labs/FLUX.2-dev`, ~106 GB) from Hugging Face. The
  repository is gated: accept the FLUX Non-Commercial License on the model page and log in with
  `huggingface-cli login` first.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/flux2/flux2_stage.yaml` (TP=8 x CP=4, 32 cores); pass
> `--stage-config examples/flux2/flux2_stage_tp8.yaml` for the 8-core layout. Pick the cores with `NEURON_VISIBLE_DEVICES` (a comma list); the
> stage's `devices:` field indexes into it.

## Step 1: Run a quick smoke test

```bash
python examples/flux2/run.py --model-path black-forest-labs/FLUX.2-dev \
    --height 512 --width 512 --steps 4 --output flux2_smoke.png
```

Expected: `flux2_smoke.png` and a last line of JSON with `"ok": true`. The first run compiles the text-encoder, DiT
and VAE graphs. Most of that time is the two VAE tile graphs, which compile once for every resolution.

## Step 2: Run a full offline generation

```bash
python examples/flux2/run.py --model-path black-forest-labs/FLUX.2-dev \
    --prompt "A cozy reading nook by a rain-streaked window, warm lamp light, a cat asleep on a stack of books" \
    --height 1024 --width 1024 --steps 50 --profile --output flux2_1024.png
```

`--profile` times one warm request after the first. On trn2.48xlarge with the default stage config (TP=8 x CP=4,
32 NeuronCores), 1024x1024 at 50 steps takes 10.3 s warm (26.1 s on 8 cores with `flux2_stage_tp8.yaml`).

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height`, `--width` | 1024 | image size in pixels (multiples of 16) |
| `--steps` | 50 | denoising steps (FLUX.2-dev is a 50-step model; few steps leave visible texture) |
| `--guidance-scale` | 4.0 | distilled guidance value (one pass per step) |
| `--seed` | 42 | noise seed |
| `--negative-prompt` | none | enables true CFG (a second pass per step) |
| `--tensor-parallel-size` | 8 | TP degree (must divide 48 heads) |
| `--devices` | stage config | logical core indices, comma list |

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | Use `NEURON_VISIBLE_DEVICES=<comma list>` instead; `run.py` converts an inherited `NEURON_RT_VISIBLE_CORES` itself |
| `Execution Queue Full` | Keep `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63` (`run.py` sets it): the DiT launches ~16 block graphs per step |
| The first request times out | Raise `FLUX2_HANDSHAKE_TIMEOUT_S` (default 7200 s); cold VAE compiles are long |

## Clean up

Remove the outputs and, to force recompilation, the NEFF cache directory configured in your environment
(`TORCH_NEURONX_NEFF_CACHE_DIR`).

## Next steps

- [FLUX.2-dev model card](../models/flux2-dev.md)
- [Tutorial: Deploy FLUX.2-dev](../tutorials/tutorial-flux2-dev.md)
