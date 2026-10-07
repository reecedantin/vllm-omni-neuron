# HunyuanVideo-1.5 Model Card

<!-- meta: description: Model card for HunyuanVideo-1.5 (480p text-to-video, 720p image-to-video with sparse attention) on AWS Trainium2 with the
vLLM Omni Neuron plugin: supported features, the recommended 16-core TP=8 x CFG-parallel configuration,
three-tier accuracy against the diffusers CPU reference, performance, and known issues. -->
<!-- meta: keywords: HunyuanVideo-1.5, HunyuanVideo15Pipeline, model card, text-to-video, video generation,
diffusion, MMDiT, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16, tensor parallelism, CFG parallelism,
image-to-video, context parallelism, sparse attention, SSTA -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[HunyuanVideo-1.5](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v) is
Tencent Hunyuan's 8.3B-parameter text-to-video diffusion transformer: 54 dual-stream MMDiT blocks (hidden 2048,
16 heads), a Qwen2.5-VL-7B text tower plus a byT5 glyph encoder for in-video text, and a 16x spatial / 4x
temporal causal 3D VAE. It is trained with flow matching and sampled with classifier-free guidance.

HunyuanVideo-1.5 is now supported for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Trainium2
(`trn2`) hardware. The Neuron pipeline subclasses upstream vLLM-Omni's `HunyuanVideo15Pipeline` and replaces
the transformer and the VAE with Neuron components.

**License:** Tencent Hunyuan Community License (see the model repository); the weights are not gated.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| HunyuanVideo-1.5 480p T2V | [hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v) | Trn2 | BF16 |
| HunyuanVideo-1.5 720p I2V, step-distilled + sparse attention (SSTA) | [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) `720p_i2v_distilled_sparse`, converted with `examples/hunyuanvideo15/convert_distilled_sparse.py` (diffusers layout, other components from the 720p I2V checkpoint) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-Video, 848x480 (480p), 25 frames | ✅ |
| | Image-to-Video, 1280x720 (720p), 121 frames (distilled sparse checkpoint) | ✅ |
| | Image-to-Video, 1280x720, 33 frames | ✅ |
| | Text-to-Video at 720p, Image-to-Video with the non-distilled checkpoints | Limited |
| | In-video text (byT5 glyph prompts) | ✅ |
| **Attention** | SSTA sparse attention (top-k tiles + window, block-sparse NKI kernel) | ✅ |
| | Dense attention over the same tile layout (`HV15_ATTN_MODE=dense_tiles`, baseline) | ✅ (33 frames) |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP, video tokens split across ranks) | ✅ (TP=8 x CP=2, TP=8 x CP=8) |
| | Text encoder on NeuronCores (Qwen2.5-VL tower, TP=4) | ✅ |
| | CFG Parallelism | ✅ |
| | VAE Tile Parallelism (tiles dealt across all ranks) | ✅ |
| **Performance** | Classifier-Free Guidance | ✅ |
| | Prompt-embedding cache (`HV15_TEXT_CACHE`) | ✅ |
| | Spatial Tiling (VAE, fixed 176 px tiles) | ✅ |
| | Temporal Chunking (VAE, causal, exact) | ✅ |
| | N-block DiT graphs (`HV15_BLOCKS_PER_GRAPH`) | ✅ |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for HunyuanVideo-1.5.
- Limited: runs through the same code path, but not yet measured end to end on device.
- `-`: not supported.

### Recommended configuration

**Text-to-video, 480p:** four Trn2 chips (16 logical NeuronCores at LNC=2), matching the default stage config
`examples/hunyuanvideo15/hunyuanvideo15_stage.yaml`: `tensor_parallel_size=8` x `cfg_parallel_size=2`. Each
classifier-free-guidance branch runs on its own 8-core TP group (an adjacent chip pair); the two noise predictions
are exchanged on the host (the CFG pairs, rank r and r + 8, have no device-to-device path), and every rank
applies the same combine and scheduler step, so the result equals sequential CFG. On the 8- and 4-core layouts
the CFG pair sits inside one chip pair and the exchange runs on the device inside the DiT's last graph. Smaller
layouts: `hunyuanvideo15_stage_tp4_cfg2.yaml` (8 cores) and,
on one chip, `hunyuanvideo15_stage_tp2_cfg2.yaml` (TP=2 x CFG=2) or `hunyuanvideo15_stage_tp4.yaml` (TP=4,
sequential CFG); see Performance.

**Image-to-video, 720p, 121 frames (distilled sparse checkpoint):** 16 cores,
`examples/hunyuanvideo15/hunyuanvideo15_i2v_stage_tp8_cp2.yaml`: `tensor_parallel_size=8` x context parallel 2
(`ring_degree=2`). The 31x45x80 latent grid (111,600 video tokens) is laid out in 360 tiles of 6x8x8 = 384 tokens
(tile-major, padded to whole tiles); each CP rank owns whole tiles, its K/V are all-gathered for the sparse
attention, and the output tokens are all-gathered on the device inside the last graph. SSTA attention: every query
tile attends a local 3D window plus its top-k most similar tiles and the text tiles, selected in the graph
(no sort / scatter) and executed by the shared block-sparse NKI kernel (`diffusion/attention/block_sparse.py`).
The checkpoint is CFG-distilled (guidance 1, one DiT call per step); 50 steps. The 8-core
`hunyuanvideo15_i2v_stage_tp8.yaml` fits 33 frames; dense attention at 121 frames does not fit the 24 GiB of HBM per
core at TP=8 x CP=2 (see Known limitations).

**Lowest latency, 720p I2V:** the whole trn2.48xlarge (64 cores),
`examples/hunyuanvideo15/hunyuanvideo15_i2v_stage_tp8_cp8.yaml` (TP=8 x CP=8, 45 tiles per CP rank): 2.7x faster
per request than the 16-core layout (see Performance).

The DiT is compiled as 6-block graphs (`HV15_BLOCKS_PER_GRAPH=6`); every full chunk shares one graph. The
Qwen2.5-VL text tower runs on the NeuronCores: the world is split into 4-rank groups (one chip each), each holding
a TP=4 shard by attention head (7 query heads and 1 KV head per rank), only the 26 layers the pipeline reads
(`hidden_states[-3]`) are run, and the embedding lookup reads the needed rows on the host. The positive and the
negative prompt are encoded on different chips at the same time and the embeddings are broadcast from rank 0;
they are cached per prompt pair (`HV15_TEXT_CACHE`, 16 entries). The byT5 glyph encoder (0.2B, only for quoted
text) stays on the host. `HV15_TEXT_ENCODER=host` runs the Qwen2.5-VL tower on the host CPU of rank 0 instead.
The VAE decodes on the NeuronCores: fixed 176 px spatial tiles (one compiled shape) dealt across all ranks and
blended on rank 0, each tile decoded with a whole-clip trunk (`conv_in` + mid block) and the up path in causal
one-latent-frame chunks with the causal-conv context carried between chunks (exact against the whole-clip decoder),
one graph per up block. `HV15_VAE_HOST=1` decodes on the host CPU instead.

## Accuracy Evaluation

**Gate:** the three tiers of the onboarding guide, each a `vllm_neuron.accuracy.testing.assert_close_three_way`
comparison of FP32 CPU (the diffusers `HunyuanVideo15Transformer3DModel`), BF16 CPU (the same module in bf16,
which isolates the dtype error) and BF16 Neuron (this plugin's DiT on one NeuronCore). Real 480p T2V weights,
256x256, 5 frames, CFG 6.

| Tier | Neuron rel-L2 vs FP32 | BF16 CPU rel-L2 vs FP32 | Result |
|---|---|---|---|
| 1. DiT call, conditional branch | 1.20% | 1.56% | pass |
| 1. DiT call, unconditional branch | 0.89% | 1.13% | pass |
| 2. One CFG denoising step (latent) | 0.65% | 0.87% | pass |
| 3. Four CFG steps (final latent) | 17.2% (cos 0.986) | 12.8% | pass |
| 3. Decoded video, worst frame PSNR vs FP32 | 23.1 dB | 26.1 dB | - |
| Text tower on device, prompt embeddings vs FP32 CPU (positive / negative prompt) | 0.85% / 1.82% | 0.89% / 2.01% | pass |
| One CFG step at the serving layouts (TP2xCFG2 / TP4xCFG2 / TP8xCFG2 host text tower / TP8xCFG2), device text tower unless noted | 2.39% / 2.41% / 2.59% / 2.29% | 3.12% | pass (bar 6.75%) |
| VAE decode, 480x848x25f real latents: device vs CPU fp32 same tiles | 62.8 dB (worst frame) | - | pass (>= 35 dB) |
| VAE decode, same, device tiled vs CPU fp32 whole frame | 44.9 dB (worst frame) | CPU fp32 tiled: 44.9 dB | pass, no visible seams |

The CPU unit tests check the Neuron DiT against diffusers at TP=1/2/4 (rel-L2 ~1e-7, fp32), the full pipeline
against the diffusers pipeline (latents ~1e-6), CFG-parallel against sequential CFG (exact), context parallel
(CP 2/3/4/8, with TP and CFG, and over the descending rank groups of the trn2 physical-mesh layouts) against one
rank, and the SSTA selection and attention against the upstream reference (bit-equal masks).

**Image-to-video, 720p (distilled sparse checkpoint).** One denoising step from fixed seed-42 noise and a fixed
image, device BF16 vs the CPU FP32 run of the same pipeline; bar = 2.3 x the CPU BF16 error.

| Check | Neuron rel-L2 vs FP32 | BF16 CPU rel-L2 vs FP32 | Result |
|---|---|---|---|
| 1280x720, 33 frames, SSTA, TP=8 (8 cores) | 1.48% | 1.64% | pass (bar 3.78%) |
| 1280x720, 33 frames, SSTA, TP=8 x CP=2 (16 cores) | 1.48% | 1.64% | pass (bar 3.78%) |

Teacher-forced steps of the 50-step run (1280x720, 33 frames, SSTA, TP=8 x CP=2): the device DiT call's own inputs
replayed on the CPU in FP32 and BF16; bar = 2 x the CPU BF16 error + 0.5%.

| DiT call (timestep) | Neuron rel-L2 vs FP32 | BF16 CPU rel-L2 vs FP32 | Bar | Result |
|---|---|---|---|---|
| 0 (t = 1000) | 0.80% | 1.05% | 2.61% | pass |
| 25 (t = 876) | 0.66% | 0.88% | 2.26% | pass |
| 49 (t = 125) | 2.14% | 2.85% | 6.20% | pass |

At 121 frames, 50 steps, TP=8 x CP=2, the final latents of all 16 ranks are identical at every one of the 50 DiT
calls. The frames decoded by the device VAE match a host FP32 decode of the same latents at 51.9-53.7 dB PSNR
(SSIM 0.998) on every sampled frame, and frame 0 matches the conditioning image at 35.1 dB (SSIM 0.950).

| Layout check, 1280x720, same seed, 50 steps | Latent rel-L2 (frame 0) | Per-frame SSIM, min / mean | Result |
|---|---|---|---|
| 33 frames: TP=8 x CP=2 (16 cores) vs TP=8 (8 cores) | 1.83% (0.43%) | 0.989 / 0.996 | equivalent |
| 121 frames: TP=8 x CP=8 (64 cores) vs TP=8 x CP=2 (16 cores) | 3.98% (0.45%) | 0.972 / 0.986 | equivalent |
| 33 frames: SSTA vs dense attention, TP=8 x CP=2 | 5.74% (0.89%) | 0.971 / 0.980 | same content; the gap equals the one-step CPU sparse-vs-dense difference (5.4%) |

At 64 cores the final latents of all 64 ranks are identical.

**Reproduce:**

```bash
pytest test/unit/test_hunyuanvideo15_*.py -q                    # CPU, tiny random checkpoint, no weights
python test/neuron/test_hunyuanvideo15_accuracy.py              # needs trn2 + weights (HV15_ACC_WEIGHTS)
python test/neuron/test_hunyuanvideo15_device.py                # per-component device parity
```

## Performance

trn2.48xlarge, BF16, 848x480, 25 frames, 50 steps, CFG 6, warm NEFF cache, same prompt every request (embeddings
served by the per-prompt cache); quiet host (1-minute load < 20 before every timed request); warm request =
median of 3 after one uncounted warm-up request.
**The 480p rows are measured for speed only, not accuracy-checked at full size** (848x480 x 25 frames, 50 steps):
there is no full-size CPU reference. The accuracy evidence per row is the one-step CFG check at 256x256 x 5
frames named in the last column (bar 6.75% = 2 x CPU BF16 error + 0.5%; see Accuracy Evaluation). At full size
all ranks produce identical final latents on every row.

| Configuration | Cores | Warm request | Text encoders | DiT (50 steps) | VAE decode | First request | HBM per core | Accuracy evidence (256x256x5, 1 step) |
|---|---|---|---|---|---|---|---|---|
| TP=8 x CFG=2, device text tower (default) | 16 | 60.5 s | 0.12 s (0 s cached) | 48.5 s (0.97 s per step) | 11.6 s | 92 s | 13.6 GiB | 2.29%, pass |
| TP=8 x CFG=2, host text tower (`HV15_TEXT_ENCODER=host`) | 16 | 60.4 s | 2.9 s (0 s cached) | 48.5 s | 11.5 s | 90 s | 10.8 GiB | 2.59%, pass |
| TP=4 x CFG=2 (`_tp4_cfg2.yaml`), device text tower | 8 | 102.3 s | 0 s cached | 85.0 s (1.70 s per step) | 17.0 s | 245 s | - | 2.41%, pass |
| TP=2 x CFG=2 (`_tp2_cfg2.yaml`), device text tower | 4 | 168.2 s | 0 s cached | 134.4 s (2.69 s per step) | 33.4 s | 361 s | - | 2.39%, pass |
| TP=4, sequential CFG (`_tp4.yaml`), device text tower | 4 | 203.5 s | 0 s cached | 169 s (100 calls, 1.69 s each) | 33.7 s | 324 s | - | none at this layout (tier checks at TP=1, CPU unit tests at TP=4) |

Scaling the DiT from 4 to 16 cores is close to linear (2.69 -> 1.70 -> 0.97 s per step), and the VAE tiles (24 at
480p) spread over every rank. "First request" is the first request of a fresh engine with the NEFF cache warm; a
cold DiT compile adds about 60 s at TP=8, and the device text tower compiles once (684 s for that first request).
The text tower adds 2.8 GiB of HBM per core (TP=4),
measured at TP=8 (13.6 GiB total of ~24 GiB).

Image-to-video, 1280x720, 121 frames, 50 steps, distilled sparse checkpoint (guidance 1); warm request = median
of 3 after one uncounted warm-up request, measured on a quiet host (1-minute load < 20 before every timed
request):

| Configuration | Cores | Warm request | DiT (50 steps) | VAE decode | First request |
|---|---|---|---|---|---|
| TP=8 x CP=8 (`hunyuanvideo15_i2v_stage_tp8_cp8.yaml`), lowest latency | 64 | 131.6 s | 82.7 s (1.65 s per step) | 34.6 s | 1016 s |
| TP=8 x CP=2 (`hunyuanvideo15_i2v_stage_tp8_cp2.yaml`), recommended | 16 | 351.4 s | 256.1 s (5.12 s per step) | 85.1 s | 1238 s |

"First request" includes the graph compiles missing from the NEFF cache. Going from 16 to 64 cores speeds the DiT
up 3.1x and the VAE decode 2.5x.

## Known limitations

- VAE tile size is capped at 176 px on trn2: at 192 px the full-resolution 128-channel up block fails to compile
  (`NCC_IBTN020`, access-pattern step out of int16 range).
- Dense attention (`HV15_ATTN_MODE=dense_tiles`) at 720p x 121 frames does not fit at TP=8 x CP=2: the 6-block
  graph needs 27.8 GB of HBM per core at compile time (`NCC_EOOM002`) and the 2-block graph fails to load (DMA
  spill rings, `Allocation Failure`). The sparse (SSTA) path is the supported one at that size.
- Text-to-video at 480p is measured for speed only at full size (848x480 x 25 frames, 50 steps); its accuracy
  checks are one-step checks at 256x256 x 5 frames (see Performance).
- Text-to-video is measured at 480p only; 720p text-to-video and the non-distilled image-to-video checkpoints run
  the same code but are not measured. The super-resolution / meanflow checkpoints are not supported.
- One prompt per request; prompts are padded to the full 1000-token MLLM length (`HV15_TEXT_BUCKETS`).
- The device text tower needs a world size that is a multiple of 4 (one TP=4 group per chip); other worlds fall
  back to the host encoder automatically.
- TP=4 x CP=4/8 x CFG=2 stage configs (non-distilled checkpoints) are provided but not yet measured.

## Tutorials

- [Tutorial: Deploy HunyuanVideo-1.5 with vLLM Omni Neuron](../tutorials/tutorial-hunyuanvideo15.md)
- [Quickstart: Offline text-to-video with HunyuanVideo-1.5](../getting-started/quickstart-offline-serving-hunyuanvideo15.md)
