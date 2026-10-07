# Quickstart: Offline text-to-video with LTX-2.5 on Neuron

<!-- meta: description: Generate a video with synchronized audio offline with Lightricks LTX-2.5 (distilled) on
AWS Trainium2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full 512x768, 121-frame run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, LTX-2.5, text-to-video, audio, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate a video with synchronized 48 kHz audio from a text prompt on Trainium2
with the vLLM Omni Neuron plugin. When you finish, you have an MP4 produced offline by the
[LTX-2.5](../models/ltx2-5.md) model.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance. The default stage uses four logical NeuronCores (one chip at LNC=2).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download `Lightricks/LTX-2.5-Diffusers` from Hugging Face. The repository is gated: accept the
  LTX-2.x Community License on the model page and log in with `hf auth login` first.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/ltx2/ltx2_stage.yaml` (TP=4).

The stage runs four worker processes, so device visibility is set with `NEURON_VISIBLE_DEVICES` and the stage
`devices:` list, not `NEURON_RT_VISIBLE_CORES` (vLLM's multi-process executor refuses the latter):

```bash
unset NEURON_RT_VISIBLE_CORES
export NEURON_VISIBLE_DEVICES=0,1,2,3      # the four logical cores the stage may use
```

## Step 1: Run a quick smoke test

```bash
python examples/ltx2/run.py --height 64 --width 96 --num-frames 9 --steps 2 --output ltx25_smoke.mp4
```

Expected: `ltx25_smoke.mp4` with a video and an audio stream. The first run compiles the transformer graphs
(a few minutes); later runs hit the compile cache.

## Step 2: Run a full offline generation

```bash
python examples/ltx2/run.py --height 512 --width 768 --num-frames 121 --steps 8 \
    --prompt "A red fox walking through a snowy forest at dawn, the camera tracking alongside." \
    --output ltx25_t2v.mp4 --profile
```

The first request at a new resolution also compiles the VAE decoder tile graph (~17 min cold at 512x768x121).
`--profile` times a second, warm request.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height` / `--width` | 512 / 768 | Output resolution, multiples of 32 |
| `--num-frames` | 121 | Frames; `8k+1` keeps the causal VAE's frame grid exact |
| `--steps` | 8 | Denoising steps (distilled checkpoint) |
| `--fps` | 24 | Frame rate of the output and of the audio duration |
| `--seed` | 42 | Sampling seed |
| `--model-path` | `Lightricks/LTX-2.5-Diffusers` | HF repo id or local directory (also `LTX25_WEIGHTS`) |
| `--warm-repeats` | 1 | Number of timed warm requests with `--profile` |

Environment switches: `LTX2_VAE_DEVICE=0` decodes the video on the host instead of the device;
`LTX25_PROMPT_CACHE=0` disables the prompt cache; `LTX25_TEXT_ENCODER_DEVICE=0` runs the Gemma text encoder on
the host instead of the NeuronCores; `LTX25_HOST_THREADS` sets the host threads for the host stages (default 24).

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Unset it and set `NEURON_VISIBLE_DEVICES` (see Prerequisites) |
| `Model class ... not found in diffusion model registry` | The stage `model_class_name` must be `LTX2Pipeline` |
| First request takes ~20 min | The VAE tile graph is compiling; it is cached for later runs at the same resolution and frame count |
| Out of device memory with `tensor_parallel_size: 1` | The 19B transformer needs TP=4 (8.7 GB per core) |

## Clean up

Remove the outputs and, if you no longer need them, the compile cache directory (`TORCH_NEURONX_NEFF_CACHE_DIR`).

## Next steps

- [LTX-2.5 model card](../models/ltx2-5.md)
- [Tutorial: Deploy LTX-2.5](../tutorials/tutorial-ltx2-5.md)
