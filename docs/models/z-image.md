# Z-Image Model Card

<!-- meta: description: Model card for Z-Image and Z-Image-Turbo on AWS Trainium2 with the vLLM Omni Neuron
plugin: supported features, recommended configuration, accuracy on Neuron, performance, and known issues. -->
<!-- meta: keywords: Z-Image, Z-Image-Turbo, model card, text-to-image, diffusion, single-stream DiT, vLLM,
vLLM Omni, Neuron, trn2, BF16, tensor parallelism, CFG parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) is Alibaba Tongyi-MAI's 6B-parameter text-to-image foundation
model. It is a single-stream diffusion transformer: image patches and caption tokens share one sequence through 30
transformer blocks, after two noise-refiner and two context-refiner blocks. Captions come from a Qwen3-4B text encoder
(its second-to-last hidden layer), and pixels from a Flux-style 16-channel `AutoencoderKL`.
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is the guidance-distilled variant with the same
architecture: 8 denoising passes and no classifier-free guidance.

Both are supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using
the Neuron SDK on AWS Trainium2 (`trn2`, NeuronCore-v3).

**License:** Apache-2.0. The checkpoints are not gated.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Z-Image | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | Trn2 | BF16 |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | 512 x 512 to 1024 x 1024 | ✅ |
| | Image-to-image | - |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ (TP 2, 4, 8; heads zero-padded 30 -> 32 for TP 4 and 8) |
| | Context Parallelism (CP) | - |
| | CFG Parallelism | ✅ (Z-Image base; Turbo is guidance-distilled) |
| | VAE Patch Parallelism | - |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Z-Image.
- Limited: accepted, with the caveat noted.
- `-`: not supported.

### Recommended configuration

| Checkpoint | Stage config | Cores | Warm latency, 1024 x 1024 |
|---|---|---|---|
| Z-Image (50 steps, CFG 4) | `examples/z_image/z_image_stage.yaml` (TP 2 x CFG 2) | 4 (one chip) | 39.9 s |
| Z-Image-Turbo (9 steps) | `examples/z_image/z_image_stage_tp4.yaml` (TP 4) | 4 (one chip) | 5.6 s |
| Z-Image, lowest latency | `examples/z_image/z_image_stage_tp4_cfg2.yaml` (TP 4 x CFG 2) | 8 (adjacent chip pair) | 24.6 s |
| Z-Image-Turbo, lowest latency | `examples/z_image/z_image_stage_tp8.yaml` (TP 8) | 8 (adjacent chip pair) | 3.9 s |

The one-chip layouts are recommended: they give the most images per core-second (Z-Image 160 core-seconds per
image at TP 2 x CFG 2 against 196 at TP 4 x CFG 2; Z-Image-Turbo 23 against 31), so a trn2.48xlarge serves the most
requests as sixteen one-chip instances. The 8-core layouts cut the latency of a single request by 1.6x (Z-Image) and
1.5x (Z-Image-Turbo) for 2x the cores. Every layout in the Performance table passes the full-size gate
(Accuracy Evaluation).
`z_image_stage_tp2.yaml` (TP 2, sequential CFG, 2 cores) is the smallest layout: 76.6 s / 9.0 s.

At 1024 x 1024 a single logical core does not fit the served DiT working set, so the DiT is tensor-parallel. Its 30
heads are zero-padded to 32 for TP 4 and 8 (the padded heads have zero Q/K/V and output weights and contribute
exactly 0). With `cfg_parallel_size: 2` each guidance branch runs on its own TP group at the same time and the two
noise predictions are exchanged over the CFG group on the host (a 1 MB fp32 latent per step). The pipeline splits
the work into separately compiled graphs:

- the text encoder (tensor-parallel over the same TP group);
- the DiT prologue (embedders and refiners), one shared 5-block graph replayed for the 30 main blocks, and the final
  layer;
- one VAE decoder tile graph (512 x 512 output tiles, 1/8 overlap, blended on the host), in fp32 (the VAE config's
  `force_upcast`). With several ranks the 9 tiles at 1024 px are dealt round-robin over the stage's ranks and
  all-gathered on the host; the result is bit-identical to one rank decoding all of them (`Z_IMAGE_VAE_DEAL=0`
  turns this off).

The Qwen3 text encoder runs on the NeuronCores with bf16 matmuls around an fp32 residual stream and fp32 RMSNorms.
Z-Image conditions on the residual stream after 35 layers, which never passes the model's final norm, so plain bf16
accumulates rounding in it: against CPU fp32 the prompt embedding is 1.71 % off in CPU bf16, 1.28 % in bf16 on the
NeuronCore and 0.51 % with the fp32 residual. Outputs are cached per token sequence (`Z_IMAGE_TEXT_CACHE`, default
64), so the empty negative prompt and repeated prompts are encoded once. `text_encoder_on_host: true` keeps the
previous behavior (CPU bf16, eager). Classifier-free guidance (Z-Image base) runs the two branches
at batch 1, sequentially or, with `cfg_parallel_size: 2`, one per TP group. A request without a guidance scale (or
with 0, as for Turbo) runs without CFG: vLLM Omni fills an omitted scale with 1.0, which Z-Image's
`pos + g * (pos - neg)` would treat as a real CFG pass, so the pipeline maps it back to 0.

## Accuracy Evaluation

**Full-size gate at every layout in the performance table** (jobs `1005-145522-A10-T-retime-gate`,
`1005-175850-A10-T-gate-turbo-tp8` and, for Z-Image TP 2 / TP 4 / TP 8 and Z-Image-Turbo TP 2,
`1006-144133-A10-T-gate-rest`; CPU references `1005-150008-A10-T-cpu-refs-1024`). One
served request per layout at 1024 x 1024, full steps (Z-Image 50 steps CFG 4, Z-Image-Turbo 9 steps), the car prompt,
seed 1. Every rank's initial noise, final latent and decoded image were saved: on every layout all ranks are
bitwise equal (one latent hash per layout). The CPU fp32 and bf16 references are the diffusers `ZImagePipeline`
started from the same noise. Bar: Neuron at most 2x the CPU bf16 error + 0.5 %, on the final latent (rel-L2) and on
the decoded image (1 - SSIM).

| Checkpoint | Layout | Ranks agree | Latent rel-L2: Neuron / CPU bf16 / bar | Image SSIM: Neuron / CPU bf16 (1 - SSIM bar) | PSNR: Neuron / CPU bf16 | Result |
|---|---|---|---|---|---|---|
| Z-Image | TP 2 | 2 / 2 | 15.1 % / 12.5 % / 25.5 % | 0.926 / 0.958 (8.9 %) | 26.7 / 29.0 dB | pass |
| Z-Image | TP 4 | 4 / 4 | 15.0 % / 12.5 % / 25.5 % | 0.924 / 0.958 (8.9 %) | 26.9 / 29.0 dB | pass |
| Z-Image | TP 2 x CFG 2 | 4 / 4 | 15.1 % / 12.5 % / 25.5 % | 0.926 / 0.959 (8.6 %) | 26.7 / 29.0 dB | pass |
| Z-Image | TP 8 | 8 / 8 | 14.9 % / 12.5 % / 25.5 % | 0.926 / 0.958 (8.9 %) | 26.9 / 29.0 dB | pass |
| Z-Image | TP 4 x CFG 2 | 8 / 8 | 15.0 % / 12.5 % / 25.5 % | 0.925 / 0.959 (8.6 %) | 26.9 / 29.0 dB | pass |
| Z-Image-Turbo | TP 2 | 2 / 2 | 10.0 % / 10.5 % / 21.5 % | 0.962 / 0.972 (6.2 %) | 31.2 / 31.4 dB | pass |
| Z-Image-Turbo | TP 4 | 4 / 4 | 8.7 % / 10.5 % / 21.5 % | 0.970 / 0.973 (5.9 %) | 32.8 / 31.4 dB | pass |
| Z-Image-Turbo | TP 8 | 8 / 8 | 8.0 % / 10.5 % / 21.5 % | 0.970 / 0.973 (5.9 %) | 33.2 / 31.4 dB | pass |

The four rows from job `1006-144133` are scored against the same cached CPU fp32 / bf16 latents (same noise) and
their CPU-decoded images as stored 8-bit PNGs; re-scoring the TP 2 x CFG 2 and Turbo TP 4 runs that way reproduces
their latent numbers exactly and moves SSIM by at most 0.001. Z-Image TP 2 produces the same final latent bit for
bit as TP 2 x CFG 2.

The Neuron image is the served output (decoded on the NeuronCores, 3 x 3 tiles); the references are decoded untiled
by the CPU fp32 VAE. Of the Neuron image's distance to fp32, the tiling alone accounts for SSIM 0.976 / PSNR 36.5 dB:
the CPU fp32 VAE run through the same tiling on the same Neuron latent reproduces the Neuron image exactly (PSNR > 100
dB). Layout equivalence (same request, two layouts): Z-Image TP 2 x CFG 2 vs TP 4 x CFG 2 latent 9.1 %, SSIM 0.977;
Z-Image-Turbo TP 4 vs TP 8 8.9 %, SSIM 0.981, both below the CPU bf16 error. By eye all eight Neuron images match the
fp32 composition (car pose, street, reflections) with no seams or artifacts.

**Single sample, M1:** final-latent parity against a CPU fp32 run of the diffusers `ZImagePipeline` from the same
seed, same bar.

| Checkpoint | Shape | Neuron vs fp32 (rel-L2 / cos) | CPU bf16 vs fp32 (rel-L2 / cos) | Result |
|---|---|---|---|---|
| Z-Image-Turbo | 1024 x 1024, 9 steps, no CFG | 4.65 % / 0.9989 | 6.82 % / 0.9977 | pass |
| Z-Image | 512 x 512, 20 steps, CFG 4 | 7.49 % / 0.9972 | 4.93 % / 0.9988 | pass (bar 10.35 %), text encoder on the host |

With the text encoder on the NeuronCores (fp32 residual, the shipped default) the same Z-Image sample lands 13.2 %
from fp32, although its prompt embedding is 0.51 % from fp32 against the host encoder's 1.71 %. The extra distance
comes from the trajectory's sensitivity to the conditioning, not from the device (job `1005-175850-A10-T-cpu-fox-gate`,
all runs CPU fp32 DiT and scheduler, same noise):

| fp32 DiT conditioned on | Embedding vs fp32 | Final latent vs fp32 |
|---|---|---|
| the device encoder's embeddings | 0.51 % | 11.9 % |
| the host bf16 encoder's embeddings | 1.71 % | 4.3 % |
| fp32 embeddings + random noise of 0.51 % (two draws) | 0.51 % | 12.1 %, 48.6 % |

An exact fp32 DiT given the device encoder's embeddings already lands 11.9 % away, and a random perturbation of the
same size lands 12 % or 49 % away (the second draw changes the composition). The final distance on this sample
depends on the direction of a sub-percent conditioning error, not its size: the host encoder's larger error happens to
point the trajectory closer. Two CPU bf16 runs of this sample landed 4.9 % and 46.8 % from fp32. By eye all four
images (fp32, the two fp32-DiT continuations, Neuron) show the same fox, pose and background. The base gates (a) to
(c) below pass with the device encoder.

`test/neuron/test_z_image_accuracy.py` covers the three tiers (`assert_close_three_way`: fp32 CPU / bf16 CPU / bf16
Neuron; σ-ratio is the Neuron RMS error over the CPU bf16 RMS error):

| Tier | Test | σ-ratio | Result |
|---|---|---|---|
| Component | DiT, one call on a CFG pair, 512 px | 1.02 | pass |
| Component | Qwen3 text encoder (on the NeuronCore) | 1.48 | pass |
| Component | VAE decode, 1024 px, 3 x 3 tiles (fp32 on Neuron) | 0.0001 | pass |
| Single step | Z-Image, CFG 4, 512 px | 0.95 | pass |
| Single step | Z-Image-Turbo, 512 px | 0.95 | pass |
| End to end | Two Neuron runs of the same request (repeatability) | rel-L2 0 | pass |

**End to end against CPU fp32** (512 x 512, final latent vs the diffusers `ZImagePipeline` in fp32; CPU bf16 is the
floor). Z-Image-Turbo is gated on one sample: Neuron at most 2x the CPU bf16 error + 0.5 %.

| Sample | Neuron vs fp32 | CPU bf16 vs fp32 | Bar | Result |
|---|---|---|---|---|
| Turbo, car / 7 | 16.2 % | 13.3 % | 27.0 % | pass |

Z-Image base at CFG 4 over 20 steps is chaotic, so its final latent is not a per-sample gate. On some prompt / seed
pairs a bf16 trajectory settles on a different composition (camera angle, car pose) than fp32, and it can happen on
either path: Neuron on car / 7 and car / 2, CPU bf16 on fox / 42. A per-step dump of those three samples traced the
cause:

- The inputs are identical on all three paths (token ids, the empty negative prompt and its padding, timesteps,
  initial noise, the fp32 scheduler step).
- Teacher-forced (fp32's exact latent and timestep at every step, each path's own text embeddings), the combined CFG
  noise prediction of the Neuron DiT is 3.3 to 10.0 % from fp32 and the CPU bf16 DiT 3.8 to 11.1 %; Neuron is at
  or below CPU bf16 on 45 of the 60 steps and at most 1.27x it.
- On its own trajectory the Neuron output stays 4 to 10 % from the fp32 DiT evaluated on the same latent: the device
  error does not grow along the run.
- The composition is decided by step 4, where the Neuron latent is about 0.2 % from fp32. On car / 7 and car / 2, an
  fp32 run continued from the Neuron step-4 latent lands on the Neuron composition (9 to 14 % from the Neuron
  final, 63 % from the fp32 final); fp32 plus a random perturbation of the same norm stays on the fp32 composition.

The device adds no error beyond bf16; the flip is a property of the trajectory. Z-Image base is therefore gated on:

| Gate | Measured | Bar | Result |
|---|---|---|---|
| (a) Per-step teacher-forced combined error, car / 7, car / 2, fox / 42 (60 steps) | Neuron at most 10.0 %, CPU bf16 at most 11.1 % (host encoder); worst margin -3.7 pp with the device encoder | 2x CPU bf16 + 0.5 % at every step | pass |
| (b) Composition flips over 15 prompt / seed pairs (final rel-L2 > 40 %) | Neuron 2 (car / 7, car / 2), CPU bf16 1 (fox / 42); same with the device encoder | CPU bf16 + 2 | pass |
| (c) Median over the 15 pairs of the final rel-L2 ratio Neuron / CPU bf16 | 0.98 (host encoder), 1.00 (device encoder) | 1.5 | pass |
| (d) Two Neuron runs of the same request | rel-L2 0 | 1e-3 | pass |

The 15 pairs are the car prompt with seeds 1 to 8 and the fox prompt with seeds 1 to 5, 7 and 42. Outside the three
flips every sample on either path is at most 29 % from fp32, and the flipped ones 47 to 64 %; by eye the flipped
Neuron images keep the subject and scene. The tier-3 CPU references take about 2.5 hours to compute cold on a
trn2.48xlarge host; set `Z_IMAGE_REF_DIR` to cache them (about 40 MB) so later runs only pay for the Neuron runs
(about 6 minutes for the five end-to-end tests).

**Reproduce:**

```bash
python examples/z_image/device_check.py --model <Z-Image-Turbo dir> --height 1024 --width 1024 \
    --steps 9 --guidance 0 --out ./z_image_check
python examples/z_image/bf16_cpu_floor.py --model <Z-Image-Turbo dir> --height 1024 --width 1024 \
    --steps 9 --guidance 0 --ref-latent ./z_image_check/oracle_latent.pt --out ./z_image_floor
```

## Performance

Served with `examples/z_image/run.py --profile --warm-runs 3` on a trn2.48xlarge, 1024 x 1024. Warm latency is the
median of 3 requests after a warmup request; the stages are one request on the slowest rank, each timed to device
completion (outputs copied back to the host). The single-step check is the tier-2 gate at that layout (final latent
of a 1-step run from the same noise vs CPU fp32; bar 2x the CPU bf16 error + 0.5 %).

Text encoder on the NeuronCores, VAE tiles dealt over the ranks. Warm and first-request latency come from a
quiet-host pass (job `1006-134106-A10-T-quiet-timing`, 2026-10-06, every layout in turn on the same 8 cores, no other
CPU-heavy work on the host; the 1-minute load average was 6 to 19 before every timed request). The stage breakdown,
HBM and single-step columns come from job `1005-145522-A10-T-retime-gate` (2026-10-05, same code, busier host).
Bold rows are the recommended layouts.

| Checkpoint | Layout | Cores | Warm (median of 3) | First request | DiT (calls x per call) | CFG exchange | VAE decode | HBM / core | Single step vs fp32 (CPU bf16) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Image, 50 steps, CFG 4 | TP 2 | 2 | 76.6 s | 76.8 s | 78.9 s (100 x 0.79 s) | - | 2.4 s | 11.5 GB | 5.34 % (5.81 %) pass |
| | TP 4 | 4 | 48.1 s | 48.1 s | 46.7 s (100 x 0.47 s) | - | 1.4 s | 6.8 GB | 5.28 % pass |
| | **TP 2 x CFG 2** | 4 | **39.9 s** | 40.0 s | 38.7 s (50 x 0.77 s) | 0.15 s | 1.4 s | 11.5 GB | 5.34 % pass |
| | TP 8 | 8 | 32.4 s | 32.9 s | 36.9 s (100 x 0.37 s) | - | 1.1 s | 4.5 GB | 5.34 % pass |
| | TP 4 x CFG 2 | 8 | 24.6 s | 24.6 s | 25.0 s (50 x 0.50 s) | 0.41 s | 1.1 s | 6.8 GB | 5.28 % pass |
| Z-Image-Turbo, 9 steps | TP 2 | 2 | 9.0 s | 9.1 s | 7.0 s (9 x 0.77 s) | - | 2.3 s | 11.5 GB | 3.35 % (3.68 %) pass |
| | **TP 4** | 4 | **5.6 s** | 5.6 s | 4.5 s (9 x 0.50 s) | - | 1.5 s | 6.8 GB | 3.33 % pass |
| | TP 8 | 8 | 3.9 s | 4.0 s | 2.9 s (9 x 0.33 s) | - | 1.0 s | 4.5 GB | 3.55 % pass |

First request is the first one after start-up with the graphs already compiled and cached; a cold start compiles
the layout's text-encoder, DiT and VAE graphs in 3 to 15 minutes. A new prompt costs about 0.02 s of text encoding
on the NeuronCores (0.3 s on an idle host in CPU bf16); repeated prompts and the empty negative prompt hit the
encoder cache. CFG-parallel halves the DiT calls per rank; its single-step latent is identical to sequential CFG at
the same TP. TP scales the DiT 1.6x per call from 2 to 4 cores and 1.3x from 4 to 8, where the per-block
all-reduces start to show.

**Change from the previous configuration, same layout.** Before: text encoder on the host in bf16 and every rank
decoding all 9 VAE tiles (jobs `1004-060853`, `1004-071839`, `1004-080218`). After: job
`1005-145522-A10-T-retime-gate`, busier host; the quiet-host warm numbers above are 1 to 14 % lower.

| Checkpoint | Layout | Cores | Before | After | Text encoder before (host) | VAE decode before -> after |
|---|---|---|---|---|---|---|
| Z-Image | TP 2 | 2 | 82.4 s | 81.5 s | 2.6 s | 4.1 -> 2.4 s |
| | TP 4 | 4 | 53.6 s | 48.4 s | 2.3 s | 4.1 -> 1.4 s |
| | TP 2 x CFG 2 | 4 | 45.6 s | 40.5 s | 3.2 s | 4.2 -> 1.4 s |
| | TP 8 | 8 | 38.1 s | 37.6 s | 1.8 s | 4.1 -> 1.1 s |
| | TP 4 x CFG 2 | 8 | 29.8 s | 26.6 s | 1.7 s | 4.2 -> 1.1 s |
| Z-Image-Turbo | TP 2 | 2 | 12.2 s | 9.3 s | 1.5 s | 4.1 -> 2.3 s |
| | TP 4 | 4 | 9.3 s | 5.8 s | 0.9 s | 4.1 -> 1.5 s |
| | TP 8 | 8 | 8.1 s | 4.0 s | 0.9 s | 4.1 -> 1.0 s |

The host text encoder cost 0.9 to 3.2 s per request (two prompts per CFG request, contending with the worker
processes); on the NeuronCores it adds one encoder shard per TP rank (TP 2: 8.2 -> 11.5 GB per core). The two
Z-Image TP 8 rows gain least: their DiT ran at 0.37 s per call in that job against 0.32 s before. On the quiet host
Z-Image TP 8 takes 32.4 s (about 0.31 s per DiT call), so the slower calls in that job were host contention.

## Known limitations

- 1024 x 1024 needs TP >= 2; a single logical core cannot load the 1024 px DiT graph in the served path. TP 3 is
  not supported (the FFN width 10240 is not divisible by 3).
- `cfg_parallel_size: 2` with Z-Image-Turbo is accepted but buys nothing (no guidance branch); use
  `z_image_stage_tp4.yaml` or `z_image_stage_tp8.yaml`.
- Image-to-image and the Z-Image Omni (SigLIP-conditioned) variant are not supported.
- Z-Image base end to end: like CPU bf16, a Neuron run can settle on a different composition than CPU fp32 for a
  given prompt and seed (2 of 15 samples, vs 1 of 15 for CPU bf16); see Accuracy Evaluation.

## Tutorials

- [Tutorial: Deploy Z-Image with vLLM Omni Neuron](../tutorials/tutorial-z-image.md)
- [Quickstart: Offline generation with Z-Image](../getting-started/quickstart-offline-serving-z-image.md)
