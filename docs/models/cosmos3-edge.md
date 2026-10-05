# Cosmos3-Edge Model Card

<!-- meta: description: Model card for NVIDIA Cosmos3-Edge on AWS Inferentia2 and Trainium2 with the vLLM Omni Neuron
plugin. Covers supported modalities (text-to-image, text-to-video, image-to-video, robot policy, forward and inverse
dynamics), the recommended two-core inf2 and one-chip trn2 configurations, accuracy against NVIDIA golden tests,
performance, and known issues. -->
<!-- meta: keywords: Cosmos3-Edge, NVIDIA Cosmos, model card, text-to-image, text-to-video, image-to-video, world
model, robot policy, forward dynamics, inverse dynamics, diffusion, vLLM, vLLM Omni, Neuron, Inferentia2, inf2,
NeuronCore-v2, Trainium2, trn2, BF16, CFG parallelism, VAE patch parallelism, NKI -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-07 -->

## Introduction

[Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) is NVIDIA's compact (~4B parameter) Cosmos3 world model. It
has a Mixture-of-Transformers design:

- a causal understanding (UND) tower that encodes the text and conditioning;
- a bidirectional generation (GEN) tower that cross-attends to the UND tower's cached keys and values at every
  denoising step;
- a Wan2.2-5B VAE.

One checkpoint serves six modalities: text-to-image, text-to-video, image-to-video, and three robot world-model modes.
The robot modes are policy (an image plus an instruction gives an action chunk and a predicted video), forward dynamics
(an image plus actions gives a video), and inverse dynamics (a video gives actions).

Cosmos3-Edge is supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
using the Neuron SDK on AWS Inferentia2 (`inf2`, NeuronCore-v2) and on AWS Trainium2 (`trn2`, one chip; see
[Trn2](#trn2)).

**License:** [OpenMDW 1.1](https://openmdw.ai/license/1-1/) (NVIDIA). The checkpoint is not gated.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Cosmos3-Edge | [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) | Inf2, Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image (up to 640x640) | ✅ |
| | Text-to-video (832x480, up to 121 frames) | ✅ |
| | Image-to-video (832x480, up to 121 frames) | ✅ |
| | Robot policy (image + instruction → actions + video) | ✅ |
| | Forward dynamics (image + actions → video) | ✅ |
| | Inverse dynamics (video → actions) | ✅ |
| | Action-only output (skip the VAE decode) | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP=2) | ✅ |
| | CFG parallelism (one guidance branch per core) | ✅ |
| | VAE patch parallelism (2 tiles) | ✅ |
| **Performance** | NC-v2 NKI attention kernel | ✅ |
| | Training-free step cache (opt-in) | ✅ |
| **Serving** | Offline `Omni.generate` | ✅ |
| | Online `vllm serve --omni` (T2I via `/v1/images/generations`) | ✅ |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Cosmos3-Edge on inf2. The [Trn2](#trn2) section lists what is validated on
  Trainium2.

### Recommended configuration

On an inf2.8xlarge (one Inferentia2 chip, two NeuronCore-v2, 16 GB each), use
[`examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml`](../../examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml).
It sets `cfg_parallel_size: 2`, `tensor_parallel_size: 1` and `vae_patch_parallel_size: 2` on `devices: 0,1`:

- **CFG parallelism** runs the conditional and unconditional passes of each step on separate cores at the same time.
  With TP=2 instead, every attention call pays a cross-core reduction.
- **VAE patch parallelism** splits each 832x480 frame into two 480x480 tiles with a 64-pixel blended overlap, one tile
  per core. The output is seamless: 37.4 dB PSNR against the untiled decode.
- **The NC-v2 attention kernel** (`nki_attention_nc2.py`) is selected automatically on NeuronCore-v2 by
  `vllm_omni_neuron/nc_generation.py`. NeuronCore-v3+ (trn2/trn3) keeps the bundled `nkilib` kernels.

[`cosmos3_edge_stage.yaml`](../../examples/cosmos3_edge/cosmos3_edge_stage.yaml) is the single-core baseline.

### Step cache (optional)

A training-free step cache skips every other GEN forward inside a timestep window and reuses the last result,
extrapolated linearly. It is off by default. To enable it:

```bash
export COSMOS3_EDGE_STEP_CACHE=1e9 COSMOS3_EDGE_STEP_CACHE_MAX_SKIP=1 \
       COSMOS3_EDGE_STEP_CACHE_TMIN=100 COSMOS3_EDGE_STEP_CACHE_TMAX=900
```

The window matters. A plain change-threshold rule skipped the early, layout-defining steps and degraded the output to
12-13 dB.

## Accuracy Evaluation

**Benchmark: NVIDIA golden robot tests** (`bridge_orig_lerobot`, from NVIDIA's Cosmos3 reference):

| Test | Metric | Inf2 | Threshold |
|---|---|---|---|
| Inverse dynamics | action MSE vs recorded trajectory | **0.011** | ≤ 0.05 |
| Forward dynamics | video PSNR vs recorded video | **24.5 dB** | ≥ 14 dB |
| Policy | video PSNR (unpadded 640x480 region) | **22.0 dB** | ≥ 14 dB |
| Policy | action MSE vs one recorded trajectory | 0.344 | n/a (a sampled plan, not a regression target) |

**Numerical parity against CPU fp32:**

- UND tower: cosine ≥ 0.999, worst relative error 3.5%.
- One GEN tower call: relative error 1.7%. A CPU bf16 run gives 1.9%.
- Text-to-video, 4 steps, same seed: 29.9 dB PSNR against CPU fp32, same scene and layout. Differences at 35 steps
  come from bf16 trajectory drift.

**Generator benchmarks:**

- **UniGenBench (T2I, all 1170 prompts at 640x640): 70.1%.** Judged by an LLM on Amazon Bedrock rather than NVIDIA's
  Gemini judge. Scores move by about 12 points between judges, so compare only runs scored by the same judge.
- **RBench I2V subset (36 clips, 832x480, 121 frames):** a 5-task mean of 2.53 on a 1-5 scale. The motion metrics
  were not run, so the result cannot be compared directly with NVIDIA's 55.1.

**Reproduce** the inverse-dynamics golden test (inputs from NVIDIA's Cosmos3 reference assets):

```bash
python examples/cosmos3_edge/run.py --mode inverse_dynamics --video <bridge_orig_lerobot clip>.mp4 \
  --domain bridge_orig_lerobot --raw-action-dim 10 --action-chunk 16 --action-fps 5 --fps 5 \
  --height 544 --width 736 --num-frames 17 --resolution 480 --seed 0 --action-only \
  --stage-config examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml --output inv
```

## Performance

On inf2.8xlarge, bf16, fast stage, warm (after the first-request compile):

| Mode | Shape | Steps | Warm latency |
|---|---|---|---|
| Image-to-video | 832x480, 121 frames | 35 | 122.0 s; **82.9 s** with the step cache |
| Text-to-video | 832x480, 121 frames | 35 | 126.4 s; 82.0 s with the step cache |
| Text-to-image | 640x640 | 50 | 3.02 s |
| Policy | 736x544, 17 frames + 16 actions | 30 | 11.27 s; **7.06 s** action-only |
| Forward dynamics | 736x544, 17 frames | 30 | 11.23 s |
| Inverse dynamics | 17-frame video → 16 actions | 30 | 12.99 s; **7.98 s** action-only |

Image-to-video optimization history at 480p/121 frames:

| Step | Warm time |
|---|---|
| TP=2 baseline | 206.7 s |
| CFG parallelism | 176.9 s |
| NC-v2 attention kernel | 144.7 s |
| VAE patch parallelism | 122.0 s |
| Step cache | 82.9 s |

The first request of a new shape compiles its graphs, which takes roughly 10-35 minutes depending on the shape. Later
runs hit the compile cache.

## Trn2

Cosmos3-Edge fits on one Trainium2 chip (4 logical NeuronCores at LNC=2, about 24 GB HBM each). It uses the same
pipeline as inf2; on NeuronCore-v3 the bundled `nkilib` attention kernels are selected instead of the NC-v2 kernel.

### Recommended configuration

| Layout | Cores | Stage config | Use |
|---|---|---|---|
| **TP=2 x CFG-parallel 2**, VAE patch-parallel 4 (recommended) | 4 (one chip) | [`cosmos3_edge_stage_trn2_tp2cfg2.yaml`](../../examples/cosmos3_edge/cosmos3_edge_stage_trn2_tp2cfg2.yaml) | every mode |
| TP=1 | 1 | [`cosmos3_edge_stage.yaml`](../../examples/cosmos3_edge/cosmos3_edge_stage.yaml) | smallest footprint; accuracy validated, latency not measured |

With the recommended layout, export `COSMOS3_EDGE_VAE_TILE=256,256,240,208`. The decode then runs as full-size 256 px
tiles, two to three per rank, so every resolution reuses one compiled tile graph.

```bash
export NEURON_LOGICAL_NC_CONFIG=2 COSMOS3_EDGE_VAE_TILE=256,256,240,208
python examples/cosmos3_edge/run.py --mode t2v --height 480 --width 832 --num-frames 121 \
  --stage-config examples/cosmos3_edge/cosmos3_edge_stage_trn2_tp2cfg2.yaml --output edge_t2v.mp4
```

The robot modes run at guidance 1.0 by default, so they take no CFG pass and the second TP=2 group repeats the first.

### Accuracy (Trn2)

**NVIDIA golden robot tests** (`bridge_orig_lerobot`, the same inputs and command as the inf2 card, 736x544, 17 frames,
30 steps):

| Test | Threshold | TP=2 x CFG 2 | TP=1 | Inf2 |
|---|---|---|---|---|
| Inverse dynamics, action MSE | ≤ 0.05 | **0.0112** | 0.0113 | 0.011 |
| Forward dynamics, video PSNR | ≥ 14 dB | **24.3 dB** | 24.4 dB | 24.5 dB |
| Policy, video PSNR | ≥ 14 dB | **22.0 dB** | 22.0 dB | 22.0 dB |
| Policy, action MSE vs one recorded trajectory | n/a (sampled plan) | 0.344 | 0.344 | 0.344 |

**All ranks agree.** At TP=2 x CFG 2 all 4 ranks hold bit-identical final latents on every request of every row on
this page. The action-only runs, which skip the decode, return actions bit-identical to the full runs. CFG parallelism
is exact: TP=1 x CFG 2 output is bit-identical to TP=1, and TP=2 x CFG 2 output is bit-identical to TP=2.

**Numerical parity against CPU fp32** (one core; fp32 CPU / bf16 CPU / bf16 Neuron):

- UND tower (`test_cosmos3_edge_und_device.py`): cosine ≥ 0.9995, worst relative error 3.3%. A CPU bf16 run gives up
  to 4.2%.
- One GEN tower call at the T2I 640x640 geometry (`test_cosmos3_edge_gen_device.py`): relative error 1.69%, cosine
  0.9999. A CPU bf16 run gives 1.94%. The output is deterministic across calls.

**Video, 4 steps, seed 1, against CPU fp32** (mean per-frame PSNR / SSIM; the CPU bf16 column is the dtype error
alone):

| Clip | TP=2 x CFG 2 | TP=1 | CPU bf16 |
|---|---|---|---|
| T2V 256x256, 49 frames (the inf2 check; inf2 gave 29.9 dB) | **30.0 dB** / 0.907 | 29.4 dB / 0.900 | 29.0 dB / 0.899 |
| I2V 256x256, 49 frames, prompt describing the image | **26.8 dB** / 0.888 | 29.4 dB / 0.919 | 28.2 dB / 0.904 |
| I2V 256x256, 49 frames, prompt unrelated to the image | 19.8 dB / 0.632 | 22.2 dB / 0.737 | 21.7 dB / 0.717 |
| T2V 832x480, 121 frames (the Performance shape) | **29.1 dB** / 0.890 | not run | 28.1 dB / 0.874 |
| I2V 832x480, 121 frames, bridge frame (the Performance shape) | **32.9 dB** / 0.917 | not run | 33.1 dB / 0.920 |

The 256x256 I2V gap at TP=2 comes from the TP=2 bf16 reduction order: TP=2 without CFG parallelism gives a
bit-identical video, and both keep the scene, layout and motion of the CPU clip. At the full 832x480x121 shape TP=2 x
CFG 2 is within 0.2 dB of CPU bf16. The unrelated-prompt clip diverges quickly under any bf16 rounding: CPU bf16 lands
at 21.7 dB as well.

### Performance (Trn2)

On one trn2 chip, TP=2 x CFG 2, bf16, the inf2 card's shapes and steps. Each row is the median of 3 timed requests
after one uncounted warm-up, with the host 1-minute load below 20 at every timed request:

| Mode | Shape | Steps | Warm latency | Inf2 (2 cores) |
|---|---|---|---|---|
| Image-to-video | 832x480, 121 frames | 35 | **47.6 s** (GEN 40.6 s, VAE decode 4.1 s) | 122.0 s |
| Text-to-video | 832x480, 121 frames | 35 | **46.7 s** (GEN 40.6 s, VAE decode 4.1 s) | 126.4 s |
| Text-to-image | 640x640 | 50 | **1.27 s** | 3.02 s |
| Policy | 736x544, 17 frames + 16 actions | 30 | **4.50 s**; 3.36 s action-only | 11.27 s; 7.06 s |
| Forward dynamics | 736x544, 17 frames | 30 | **4.52 s** | 11.23 s |
| Inverse dynamics | 17-frame video → 16 actions | 30 | **6.10 s**; 4.89 s action-only | 12.99 s; 7.98 s |

Accuracy behind these rows: the 121-frame rows reuse the shapes and graphs of the 4-step 832x480x121 check above (a
35-step CPU fp32 clip drifts away from any bf16 run, as on inf2); the text-to-image row is covered by the GEN-call
parity at 640x640 and by the full 50-step image against CPU fp32 (26.8 dB, SSIM 0.954; CPU bf16 25.3 dB, SSIM 0.948,
same seed and default prompt); the robot rows are the golden tests above. All 4 ranks agree on every timed request.
The first request of each shape took 15-209 s in these runs (209 s for text-to-video, 23 s for text-to-image); a fully
cold compile cache was not measured on trn2. The TP=1 layout was not timed.

The step cache, online serving and the UniGenBench / RBench evaluations were run on inf2 only.

## Known limitations

- **Batch size one.** Concurrent requests run serially.
- **Text-to-video prompt adherence is weaker than image-to-video.** This is the model's behavior: the device matches
  CPU fp32 at 4 steps.
- **Guardrails are off in the example stage configs** (`guardrails: false`). NVIDIA's Cosmos guardrails need the
  separate `cosmos-guardrail` package.
- **Trn2 is validated on one chip only**, with the two layouts and the shapes in [Trn2](#trn2). Multi-chip layouts
  were not run for this model.

## Tutorials

- [Quickstart: Offline generation with Cosmos3-Edge on Inferentia2](../getting-started/quickstart-offline-serving-cosmos3-edge.md)
- [Tutorial: Deploy Cosmos3-Edge with vLLM Omni Neuron](../tutorials/tutorial-cosmos3-edge.md)
