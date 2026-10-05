# Quickstart: Offline text-to-video with HunyuanVideo-1.5 on Neuron

<!-- meta: description: Generate a 480p video offline with HunyuanVideo-1.5 on AWS Trainium2 using the vLLM
Omni Neuron plugin. Covers a quick smoke test and a full run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, HunyuanVideo-1.5, text-to-video, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate a video from a text prompt on one Trainium2 chip with the vLLM Omni
Neuron plugin. When you finish, you have an 848x480 MP4 produced offline by the
[HunyuanVideo-1.5](../models/hunyuanvideo15.md) model.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance (the default stage config uses 16 logical NeuronCores, four chips;
  `hunyuanvideo15_stage_tp2_cfg2.yaml` runs on one chip).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, about
  52 GB) from Hugging Face. The weights are not gated.
- About 60 GB of free host memory for weight loading.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/hunyuanvideo15/hunyuanvideo15_stage.yaml` (TP=8 x CFG-parallel 2 on cores 0-15).

## Step 1: Run a quick smoke test

```bash
export HV15_BLOCKS_PER_GRAPH=6
python examples/hunyuanvideo15/run.py --height 256 --width 256 --num-frames 5 --steps 2 --output hv15_smoke.mp4
```

Expected: `[run] saved hv15_smoke.mp4`. The first run compiles the DiT graphs for this shape, which takes several
minutes; repeated runs reuse the NEFF cache.

## Step 2: Run a full offline generation

```bash
export HV15_BLOCKS_PER_GRAPH=6
python examples/hunyuanvideo15/run.py \
  --prompt "A red fox trots through fresh snow in a pine forest at sunrise, cinematic, shallow depth of field." \
  --height 480 --width 848 --num-frames 25 --steps 50 --output hv15_480p.mp4 --profile
```

`--profile` runs the request twice and prints the warm latency (about 60 s on 16 cores, 171 s on one chip). The first request
at a new geometry compiles the DiT and VAE graphs (tens of minutes); later runs reuse the NEFF cache.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height` / `--width` | 480 / 848 | output size in pixels (multiples of 16) |
| `--num-frames` | 121 | frame count, `4k + 1` |
| `--steps` | 50 | denoising steps |
| `--guidance-scale` | 6.0 | classifier-free guidance scale |
| `--seed` | 42 | noise seed |
| `--stage-config` | `hunyuanvideo15_stage.yaml` | `_tp4_cfg2.yaml` (8 cores), `_tp2_cfg2.yaml` / `_tp4.yaml` (one chip) |

Environment variables: `HV15_VAE_HOST=1` (host VAE decode instead of device), `HV15_VAE_HOST_THREADS` (default 16),
`HV15_VAE_TILE` (device VAE tile in latent units, default 11 = 176 px),
`HV15_TEXT_ENCODER=host` (Qwen2.5-VL tower on the host CPU instead of the NeuronCores), `HV15_TEXT_CACHE`
(prompt-embedding cache entries, default 16, 0 = off), `HV15_TEXT_THREADS` (host text encoders, default 32),
`HV15_BLOCKS_PER_GRAPH` (DiT blocks per compiled graph),
`HV15_PROFILE=1` (per-call timings).

### Image-to-video at 720p (distilled sparse checkpoint)

Build the checkpoint once with `examples/hunyuanvideo15/convert_distilled_sparse.py` (see its docstring), then run
on 16 cores with TP=8 x context parallel 2 and the sparse (SSTA) attention:

```bash
HV15_BLOCKS_PER_GRAPH=6 python examples/hunyuanvideo15/run.py \
  --model-path <dir>/720p_i2v_distilled_sparse \
  --stage-config examples/hunyuanvideo15/hunyuanvideo15_i2v_stage_tp8_cp2.yaml \
  --image first_frame.png --prompt "A cat sleeps on a stack of books, then stretches and looks out at the rain." \
  --height 720 --width 1280 --num-frames 121 --steps 50 --output hv15_i2v_720p.mp4
```

The image is resized with a centre crop to the output size. The checkpoint is CFG-distilled, so no negative prompt
is used. `HV15_ATTN_MODE=dense_tiles` runs dense attention over the same layout (33 frames at this layout).

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | unset it and set `NEURON_VISIBLE_DEVICES` to a comma list of cores; the stage `devices:` field indexes into it |
| The first request seems to hang | it is compiling; check for `neuronx-cc` processes and the compile cache directory |
| `NCC_IBTN020` compiling the VAE | keep `HV15_VAE_TILE` at 11 or below (176 px) |

## Clean up

Delete the output MP4 and, to force a fresh compile, the Neuron compile cache directory.

## Next steps

- [HunyuanVideo-1.5 model card](../models/hunyuanvideo15.md)
- [Tutorial: Deploy HunyuanVideo-1.5](../tutorials/tutorial-hunyuanvideo15.md)
