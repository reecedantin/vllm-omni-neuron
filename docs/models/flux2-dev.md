# FLUX.2-dev Model Card

<!-- meta: description: Model card for Black Forest Labs FLUX.2 [dev] on AWS Trainium2 with the vLLM Omni Neuron
plugin: supported features, the recommended TP=8 x CP=4 configuration, accuracy against CPU references, performance, and
known limitations. -->
<!-- meta: keywords: FLUX.2, FLUX.2-dev, Black Forest Labs, model card, text-to-image, rectified flow, diffusion
transformer, Mistral, vLLM, vLLM Omni, Neuron, trn2, Trainium2, BF16, tensor parallelism, context parallelism, VAE tiling -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev) is Black Forest Labs' 32-billion-parameter
rectified-flow transformer for text-to-image generation. It is guidance-distilled, so one forward pass per step
takes a guidance value and needs no unconditional pass. It has three components:

- a 32B DiT with 8 double-stream blocks (separate image/text weights, joint attention) and 48 single-stream
  blocks (one fused QKV+MLP projection);
- a 24B Mistral-Small-3 text encoder, whose hidden states after layers 10, 20 and 30 are stacked into the
  conditioning;
- a 32-channel KL VAE.

FLUX.2-dev is supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
using the Neuron SDK on AWS Trainium2 (`trn2`, NeuronCore-v3).

**License:** [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md)
(non-commercial, non-production use). The checkpoint is gated on Hugging Face: accept the license on the model page
before downloading.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| FLUX.2-dev | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image | ✅ |
| | 1024x1024 and 2048x2048 (multiples of 16 px; 64 px with CP) | ✅ |
| | Image editing / multi-reference conditioning | - |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP), DiT and text encoder | ✅ |
| | Context Parallelism (CP), DiT (text + image tokens) | ✅ |
| | CFG Parallelism | n/a (guidance-distilled) |
| | VAE tile parallelism (tiles dealt across all TP x CP ranks) | ✅ |
| **Compilation** | torch.compile, per-block graphs reused across blocks | ✅ |
| **Serving** | Offline (`Omni` entrypoint) | ✅ |
| | Online (`vllm serve --omni`) | Limited (not validated, see below) |

**Status legend:**

- ✅ Supported: integrated and tested for FLUX.2-dev.
- Limited: accepted, with the caveat noted.
- `-`: not supported.

### Recommended configuration

Trn2, TP=8 x CP=4 on 32 logical NeuronCores (eight chips at LNC=2), matching `examples/flux2/flux2_stage.yaml`.
`examples/flux2/flux2_stage_tp8_cp2.yaml` (16 cores) and `examples/flux2/flux2_stage_tp8.yaml` (8 cores) are the
smaller alternatives; choose by core budget (see Performance).

- **Text encoder:** only the Mistral layers the pipeline reads (0-29 of 40) are built. One compiled 10-layer
  graph serves all three segments, and each segment's output is one of the stacked hidden states. It runs TP=8
  inside each CP group.
- **DiT:** sharded over heads and the FFN hidden dimension (TP). The 8 double blocks run as one-block graphs and
  the 48 single blocks as six-block graphs, so all blocks of a kind share one NEFF.
- **Context parallelism** (`ring_degree`): each of the four TP groups processes a quarter of the 512 text and
  the image tokens, and the groups all-gather K/V once per block. CP splits activations, not weights: every TP
  group holds a full TP=8 shard. CP must divide 512 and the image-token count (resolutions that are multiples of
  64 px).
- **Memory:** 8.8 GiB DiT + 3.9 GiB text-encoder weights per core; 14.4 GiB HBM per core peak at 1024 px,
  15.6 GiB at 2048 px.
- **VAE:** decodes on the NeuronCores as 512 px tiles with 128 px overlap, dealt round-robin across all
  TP x CP ranks (32 at the recommended layout; a 1024 px image has 9 tiles, so at most 9 ranks get one) and
  merged on rank 0. Two passes make the tiled decode match the untiled one: the first gathers each GroupNorm
  layer's statistics over the whole image, and the second normalises every tile with them.
- **Runtime settings:** `examples/flux2/run.py` sets `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63` (the
  runtime's maximum per-core execution-queue depth). A DiT call launches about 20 block graphs back to back,
  and the default depth rejects that with `Execution Queue Full`. The DiT also synchronises with the device
  every `FLUX2_SYNC_EVERY` launches (default 8) to keep the queue bounded. Set both when you drive the
  pipeline without `run.py`.

## Accuracy Evaluation

Accuracy is measured in three tiers, following
[Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md). Tiers 1 and 2 use
`vllm_neuron.accuracy.testing.assert_close_three_way`: fp32 CPU baseline, bf16 CPU expected, bf16 Neuron actual.
The DiT and text encoder also match diffusers / transformers on CPU to a relative error below 1e-5
(`test/unit/test_flux2_components.py`).

| Tier | Scope | Metric | Trn2 (TP=8) | Reference / threshold |
|---|---|---|---|---|
| 1 | VAE, tiled + whole-image GroupNorm, vs untiled fp32 CPU, 1024 px | PSNR | 40.9 dB (seams 41.0, interiors 40.8) | >= 35 dB |
| 1 | VAE, plain per-tile GroupNorm (for comparison) | PSNR | 24.0 dB | – |
| 2 | Pipeline, 1 step, 256 px (text encoder + DiT + scheduler), shared fp32 noise | rel-L2 vs fp32 CPU | 0.0223 (CPU bf16: 0.0227), cosine 0.9998 | <= 2x CPU bf16 + 0.5% |
| 3a | End to end, 512 px, 8 steps, shared fp32 noise, vs the same pipeline on CPU (untiled host VAE) | latent rel-L2 / image 1-SSIM vs fp32 | 0.202 / 0.045 (CPU bf16 floor: 0.217 / 0.060) | <= 2x floor + 0.5% (0.439 / 0.124) |
| 3b | End to end, 1024 px, 50 steps | device VAE PSNR vs CPU fp32 untiled; SSIM vs an accepted device image (repeatability) | pass | >= 35 dB; SSIM >= 0.9 |
| 3c | End to end, 1024 px, 50 steps, TP=8 x CP=4 (32 ranks), shared fp32 noise | all-rank agreement of the denoised latents; device VAE PSNR vs CPU fp32 untiled decode of the same latents | 32 / 32 ranks bit-identical; 41.6 dB | identical; >= 35 dB |
| 3d | Same, TP=8 x CP=8 (64 ranks) | all-rank agreement; device VAE PSNR; layout equivalence vs TP=8 x CP=4 (image PSNR / SSIM, latent rel-L2) | 64 / 64 ranks bit-identical; 41.6 dB; 38.6 dB / 0.996 (worst 128 px block 0.90), latent 0.059 | identical; >= 35 dB; no structural difference |
| 3e | Teacher-forced DiT steps 0 / 24 / 49 of the 1024 px 50-step run (3c, 3d): each step's device inputs replayed on CPU (diffusers) | rel-L2 vs CPU fp32 (CPU bf16 floor) | TP=8 x CP=4: 0.0106 (0.0130) / 0.0124 (0.0183) / 0.0177 (0.0194); TP=8 x CP=8: 0.0130 (0.0130) / 0.0121 (0.0176) / 0.0178 (0.0193); cosine >= 0.9998 | <= 2x floor + 0.5% per step (0.031-0.044) |
| 3f | Tiers 3c-3e at TP=8 (8 ranks) and TP=8 x CP=2 (16 ranks): 1024 px, 50 steps, shared fp32 noise | all-rank agreement; device VAE PSNR; layout equivalence vs TP=8 x CP=4 (image PSNR / SSIM, latent rel-L2); teacher-forced steps 0 / 24 / 49 rel-L2 vs CPU fp32 (CPU bf16 floor) | TP=8: 8 / 8 bit-identical; 41.6 dB; 41.2 dB / 0.996, latent 0.052; 0.0103 (0.0130) / 0.0122 (0.0178) / 0.0171 (0.0192). TP=8 x CP=2: 16 / 16 bit-identical; 41.6 dB; 42.4 dB / 0.997, latent 0.051; 0.0134 (0.0130) / 0.0131 (0.0183) / 0.0181 (0.0191) | identical; >= 35 dB; no structural difference; <= 2x floor + 0.5% per step |

Tiers 1, 3a and 3b were run at TP=8 (`flux2_stage_tp8.yaml`; set `FLUX2_STAGE_CONFIG` to run them on another
layout); tiers 3c-3f at every TP=8-based layout in the Performance table (8, 16, 32 and 64 cores). Tier 3e stands in for a CPU run of the whole 50-step
pipeline (8-11 h): it checks each sampled step from the device's own trajectory, so per-step error cannot hide
behind trajectory drift. Every step is at 0.68-1.03x the CPU bf16 error. Tier 2 was also run at every layout in
the Performance table, and all pass (TP=8 x CP=4: 0.0234,
cosine 0.9997).

**Why single-step for tier 2:** FLUX.2-dev's few-step trajectories are chaotic. The 4-step 256 px rel-L2 is
0.28 against a 0.21 band, but the step-1 error is identical to CPU bf16's own, and the gap is the same with the
NKI attention kernel and with the torch attention path. It is compounding of an in-band per-step bf16 error,
not a Neuron defect.

**Reproduce:**

```bash
pytest test/unit/test_flux2_*.py -q                        # CPU, tiny random model, no weights
FLUX2_WEIGHTS=<FLUX.2-dev dir> NEURON_RT_VISIBLE_CORES=0 \
  pytest test/neuron/test_flux2_components_device.py -q    # tier 1, one core
FLUX2_WEIGHTS=<FLUX.2-dev dir> FLUX2_GOLDEN=<golden.png> \
  pytest test/neuron/test_flux2_pipeline_device.py -q      # tiers 2 and 3, 8 cores
# tier 3a's CPU references take ~2 h 20 min; FLUX2_REF_CACHE=<dir> keeps them for later runs
# tier 3e: run examples/flux2/run.py with FLUX2_STEP_DUMP=<dir> (rank 0 saves steps 0/24/49), then on CPU:
python examples/flux2/step_check.py --model-path <FLUX.2-dev dir> --out steps.json <dir>   # ~15 min, ~125 GB RAM
```

The text encoder follows diffusers / transformers. Positions run 0..S-1 over the padded prompt, as
`Mistral3ForConditionalGeneration` does. Upstream vLLM-Omni's `MistralEncoderModel` derives positions from the
attention mask (`cumsum`), which gives the padded rows different rotary positions; those rows feed the DiT
through the stacked hidden states.

## Performance

trn2.48xlarge, BF16, default prompt, `guidance_scale` 4.0, prompt-embedding cache off, no CPU reference work
on the host (1-minute load average below 20 before every timed request). Warm latency is the median of three
requests after one uncounted warm-up request at the same resolution and step count, on the same process. Per
step is the median DiT call of the timed 50-step requests, timed to device completion. HBM is the per-core
peak from `neuron-monitor` (earlier runs). Accuracy is the tier-2 single-step check (256 px, rel-L2 vs CPU
fp32; bar 0.050 = 2x the CPU bf16 floor 0.0227 + 0.5%).

| Layout | Cores | 1024 px, 4 steps | 1024 px, 50 steps | DiT per step (1024 / 2048) | VAE 1024 / 2048 | HBM per core (1024 / 2048) | 2048 px, 4 / 50 steps | Tier 2 rel-L2 |
|---|---|---|---|---|---|---|---|---|
| TP=8 | 8 | 3.23 s | 26.1 s | 0.495 / 2.54 s | 1.13 / 2.36 s | 15.3 / 21.4 GiB | 11.6 / 129.9 s | 0.0223 |
| TP=16 ¹ | 16 | 2.03 s | 17.0 s | 0.324 / 1.38 s | 0.60 / 1.24 s | 12.8 GiB / – | 7.0 / 71.2 s | 0.0205 |
| TP=8 x CP=2 | 16 | 1.94 s | 15.8 s | 0.299 / 1.26 s | 0.60 / 1.26 s | 17.2 / 17.6 GiB | 6.6 / 65.4 s | 0.0232 |
| TP=16 x CP=2 ¹ | 32 | 1.62 s | 11.2 s | 0.202 / 0.77 s | 0.65 / 0.79 s | 10.7 / 11.2 GiB | 4.1 / 40.9 s | 0.0172 |
| **TP=8 x CP=4** (recommended) | 32 | **1.57 s** | **10.3 s** | **0.186 / 0.70 s** | 0.65 / 0.73 s | 14.4 / 15.6 GiB | **3.9 / 37.4 s** | 0.0234 |
| TP=8 x CP=8 | 64 | 1.39 s | 7.51 s | 0.124 / 0.45 s | 0.76 / 0.83 s | 14.2 / 14.9 GiB | 2.95 / 25.1 s | 0.0225 |
| TP=16 x CP=4 ¹ | 64 | 1.39 s | 7.87 s | 0.132 / 0.46 s | 0.71 / 0.82 s | 8.8 / 9.7 GiB | 3.1 / 25.5 s | 0.0172 |

¹ Measured for speed only, not accuracy-checked at full size (tier 2 only).

The 8- and 16-core rows were first timed on cores 0-15 beside other jobs and came out up to 1.4x slower, with
the DiT step time growing during each request; a re-run alone on cores 32-47 did not reproduce it, and the rows
above are from that re-run. TP=8 at 2048 px is the one case whose step time still rises within a request (2.27
to 2.58 s over the first timed request, then 2.50-2.58 s), so its three 50-step requests spread 124-131 s.
Every other row's requests agree within 2%. Every layout passes the tier-2 bar.

- **CP beats more TP.** At equal core count, TP=8 x CP=2 is faster than TP=16 (0.299 vs 0.324 s per step at
  1024 px), and TP=8 x CP=4 than TP=16 x CP=2. TP=16 halves the matmuls but keeps the full-sequence all-reduce
  per block on every rank, and its group spans a four-chip row. CP shrinks both the matmuls and the all-reduce
  payload and adds only a K/V gather (about 7 MB per block at 1024 px).
- **2048 px:** TP=8 x CP=4 is 3.6x faster per step than TP=8 (16.9k joint tokens; 4.2k per rank) and
  drops HBM per core from 21.4 to 15.6 GiB.
- First request (graphs compiled, cache warm for the VAE tile graphs): about 50-80 s at 1024 px for every
  layout (TP=8 x CP=4: 40 s, TP=8 x CP=8: 39 s), 2.5-10 min at 2048 px (TP=8 x CP=4: 231 s, TP=8 x CP=8:
  158 s). Process start, including the weight load, adds about 40-60 s.

| Item | Before | After |
|---|---|---|
| VAE decode 1024x1024 | 56.9 s (host CPU) | 1.14 s at TP=8, 0.65 s over 32 ranks (NeuronCores) |
| 1024x1024, 4 steps, warm request | 59.0 s (TP=8, host VAE) | 3.23 s (TP=8), 1.57 s (TP=8 x CP=4) |
| 1024x1024, 50 steps, warm request | 26.1 s (TP=8) | 10.3 s (TP=8 x CP=4), 7.51 s (TP=8 x CP=8) |
| 2048x2048, 50 steps, warm request | 129.9 s (TP=8) | 37.4 s (TP=8 x CP=4), 25.1 s (TP=8 x CP=8) |
| VAE cold compile | 3670 s, 388 GB host RAM (untiled graph, numerically wrong) | 602 s single core (plain tiled); 1796 s for both whole-image-GroupNorm passes on a loaded host |

VAE tile graphs have one shape, so they compile once for every resolution. DiT and text-encoder graphs compile
per resolution and layout.

## Known limitations

- **Untiled device VAE is not used.** Compiled as one 1024 px graph, the VAE decode reaches only 22.9 dB against
  the CPU fp32 decode (CPU bf16 reaches 61.7 dB), and the cold compile takes about an hour. Decode is therefore
  always tiled (`FLUX2_VAE_UNTILED_MAX=0`). Suspected cause: the bottleneck self-attention at 16k tokens.
- **Larger layouts are bounded by head count.** TP must divide the DiT's 48 heads and the text encoder's 32
  query heads, so TP is at most 16 (TP=32 would need head padding); at TP=16 each of the text encoder's 8 KV
  heads is replicated on two ranks. CP must divide 512 and the image-token count. The 64-core layouts
  (TP=8 x CP=8, TP=16 x CP=4) gain little at 1024 px with 4 steps (1.39 vs 1.57 s), because the per-rank
  sequence (576 tokens) is too short to hide the collectives; at 50 steps TP=8 x CP=8 is 1.3x faster than
  TP=8 x CP=4 at 1024 px (7.51 vs 10.3 s) and 1.5x at 2048 px (25.1 vs 37.4 s).
- **Host RAM at start-up** grows with the rank count (each worker process maps the checkpoint): about 220 GB
  summed RSS at 8 ranks, 460 GB at 16, 790 GB at 32, 1460 GB at 64.
- **Cold VAE compile on many ranks:** the two whole-image-GroupNorm VAE graphs take 25-30 min to compile
  on a loaded host. On a cold cache, the ranks that wait for a tile compile can exceed the 30 min host
  (gloo) collective timeout and fail the first request (seen at 32 ranks). Warm the VAE graphs once on one
  core first with `examples/flux2/vae_check.py`; they are shared by every resolution and layout.
- **Above 2048 px** is untested. HBM allows it with CP (15.6 GiB per core at 2048 px with TP=8 x CP=4).
- **Few-step parity:** whole-pipeline parity is gated at one step (see Accuracy Evaluation). Fewer than about 20
  steps also gives visible high-frequency texture on the CPU reference; 50 steps is the model's default.
- **Image editing / reference images** (`image=` inputs) are not ported: the VAE encode stays eager.
- **Online serving** (`vllm serve --omni`) uses the same stage config but was not validated on Trn2 in this
  release.
- `FLUX2_INIT_LATENTS`, `FLUX2_VAE_DUMP`, `FLUX2_RANK_DUMP` and `FLUX2_STEP_DUMP` are test plumbing: they inject
  a fixed fp32 initial noise tensor, dump the rank-0 decoded latents and image, dump every rank's denoised latents
  (all-rank agreement check), and dump the inputs and output of selected DiT steps (`FLUX2_STEP_DUMP_CALLS`,
  default `0,24,49`) for the teacher-forced CPU replay in `examples/flux2/step_check.py`.

## Tutorials

- [Tutorial: Deploy FLUX.2-dev with vLLM Omni Neuron](../tutorials/tutorial-flux2-dev.md)
- [Quickstart: Offline generation with FLUX.2-dev](../getting-started/quickstart-offline-serving-flux2-dev.md)
