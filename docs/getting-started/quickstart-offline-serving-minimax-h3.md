# Quickstart: Offline video-and-audio generation with FastH3 on Trainium2

<!-- meta: description: Generate a 5-second video with a stereo soundtrack offline with FastH3 (MiniMax-H3
distilled) on AWS Trainium2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, FastH3, MiniMax-H3, text-to-video, text-to-audio, offline,
trn2, quickstart -->
<!-- meta: date_updated: 2026-10-05 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate a video with a synchronized stereo soundtrack from a text prompt on
Trainium2 with the vLLM Omni Neuron plugin. When you finish, you have an H.264 + AAC MP4 produced offline by the
[FastH3](../models/minimax-h3.md) model.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance. The default configuration uses all 64 logical NeuronCores (16
  Trainium2 chips); `examples/minimax_h3/minimax_h3_stage_tp8cp4.yaml` runs on 32 and
  `examples/minimax_h3/minimax_h3_stage_tp8.yaml` on 8 (two chips).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- About 200 GB of free disk for the checkpoint (DiT, Qwen3-VL-32B text encoder, VAEs), and about 300 GB of host RAM.
- Network access to download `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree` from Hugging Face. The
  weights are under the MiniMax H3 Community License.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/minimax_h3/minimax_h3_stage.yaml` (TP=8 × CP=8).

## Step 1: Run a quick smoke test

A 256x256, 124-frame clip with the 4-step checkpoint:

```bash
python examples/minimax_h3/run.py \
  --model-path FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \
  --height 256 --width 256 --num-frames 124 --output fasth3_256.mp4
```

Expected: `fasth3_256.mp4` (256x256, 124 frames at 24 fps, stereo 32 kHz audio) and a `fasth3_256.json` timing
summary. The first run loads the text encoder and compiles the DiT and video-VAE graphs for this shape, which takes a
few minutes. Later runs hit the compile cache.

## Step 2: Run a full offline generation

1344x768, 124 frames, timing three warm requests after the first:

```bash
python examples/minimax_h3/run.py \
  --model-path FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \
  --height 768 --width 1344 --num-frames 124 --profile --repeat 3 \
  --prompt "A golden retriever runs through the surf at sunset, waves crashing around its paws." \
  --output fasth3_768.mp4
```

The first request compiles the text-encoder, DiT, video-VAE and audio-VAE graphs for this shape (about 12 minutes on
an empty compile cache, a little over 2 minutes in a new process once they are cached). A warm request takes about 6.4 s.

For the 8-step student, pass `--model-path FastVideo/FastVideo-FastH3-8-Step-V2 --steps 9 --cp 2 --height 384
--width 640` (9 grid points = 8 forwards, TP=8 × CP=2 on 16 cores; see the model card's known limitations for the
layouts its sparse-attention graph compiles at). Its sparse attention and scheduler shifts are read from the
checkpoint.

### Base MiniMax-H3 (50 steps)

Point `--model-path` at the MiniMax-H3 checkpoint root (`MODEL_ROOT`, the Diffusers layout with `transformer/`,
`vae/`, `audio_vae/` and `text_encoder/`) and use the base stage config. This is the upstream T2VA request (prompt,
seed, 50 steps, shifts 12 / 3) at 124 frames:

```bash
python examples/minimax_h3/run.py \
  --model-path "${MODEL_ROOT}" \
  --stage-config examples/minimax_h3/minimax_h3_base_stage.yaml \
  --prompt "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts." \
  --height 768 --width 1344 --num-frames 124 --seed 1101 \
  --output minimax_h3_768.mp4
```

On a trn2.48xlarge this writes a 5.17 s H.264 clip with a 32 kHz stereo soundtrack. The first request in a new
process with the graphs already in the compile cache took 320 s (stage init 69 s before it); on an empty cache the
DiT graph compiles too, and a later warm request takes about 69 s (the model card's Performance section).

For the base checkpoint `--steps` counts denoiser evaluations (default 50), and `--flow-shift` / `--audio-flow-shift`
override the video / audio sigma shifts (default 12 / 3).

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height`, `--width` | 384, 640 | Canvas; multiples of 32 |
| `--num-frames` | 124 | Rounded up to `17n + 5` (the video VAE's chunking) |
| `--steps` | the checkpoint's | FastH3: sigma-grid points (5 for 4-step, 9 for 8-Step-V2); base MiniMax-H3: denoiser evaluations (50) |
| `--seed` | 0 | Noise seed (drawn on the host in fp32) |
| `--prompt-embeds` | none | `.pt` with `prompt_embeds`; skips the text encoder |
| `--tp` | 8 | Tensor-parallel degree; must divide 56 heads and the FFN width |
| `--cp` | 8 | Context-parallel degree (`ring_degree`); the stage uses `tp × cp` cores |
| `--text-encoder` | device | `host` runs the Qwen3-VL text encoder on the host CPU instead |
| `--adaln` | device | `host` computes the AdaLN modulation on the host instead |
| `--profile` | off | Run warm requests after the first and report per-stage timings |
| `--repeat` | 1 | Warm requests with `--profile` (the median is reported) |

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | Set `NEURON_VISIBLE_DEVICES` to a comma list of cores instead (`run.py` converts an inherited `NEURON_RT_VISIBLE_CORES` itself). |
| Stage initialization times out on the first run | Cold compiles exceed vLLM Omni's default handshake; `run.py` raises it to 2 h (`MINIMAX_H3_HANDSHAKE_TIMEOUT_S`). |
| Host runs out of memory on the first request with `--text-encoder host` | The host text encoder needs about 50 GB on top of the DiT shards; use the device encoder (the default) or pass `--prompt-embeds`. |

## Clean up

Remove the outputs (`*.mp4`, `*.json`). The compile cache lives in `TORCH_NEURONX_NEFF_CACHE_DIR` (or the default
Neuron cache directory); keep it to skip recompiling.

## Next steps

- [MiniMax-H3 / FastH3 model card](../models/minimax-h3.md)
- [Tutorial: Deploy FastH3 with vLLM Omni Neuron](../tutorials/tutorial-minimax-h3.md)
