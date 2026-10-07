# Qwen-Image 2.1 Model Card

<!-- meta: description: Model card for Qwen-Image 2.1 on AWS Trainium with the
vLLM Omni Neuron plugin — architecture, supported features (resolutions, BF16,
tensor parallelism, condition-image prefix KV caching, tiled VAE decode), the
recommended 16-core (TP=16) configuration, device parity, and known issues. -->
<!-- meta: keywords: Qwen-Image, Qwen-Image 2.1, model card, text-to-image,
image generation, diffusion, DiT, Qwen3-VL, vLLM, vLLM Omni, Neuron, Trainium,
trn2, BF16, tensor parallelism, block-causal attention, VAE tiling -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) is a text-to-image diffusion model
from the Qwen team. A single-stream, block-causal DiT (32 layers, 4096 hidden, ~7B params) denoises
image latents jointly with text and condition-image tokens drawn from a
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) vision-language encoder: text and any
condition-image tokens attend causally and are modulated at `t = 0`, while the target image's
tokens attend to everything before them and denoise at the sampled timestep. That timestep
independence makes the prefix (text + condition-image) K/V cacheable across the denoising loop —
upstream fills a KV cache on the first step; this Neuron port instead runs the prefix as its own
compiled graph, once per prompt, decoupled from the per-step target graph. A 16x-downsampling VAE
(64 latent channels) decodes the final latent to pixels.

Qwen-Image 2.1 is now supported for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS
Trainium2 (`trn2`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen-Image 2.1 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Trn2 | BF16 |

## Features

Per-model feature availability for Qwen-Image 2.1. See the
[README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) for
configuration details.

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-Image | ✅ |
| | Condition-image input (block-causal prefix) | Not yet ported |
| | Classifier-Free Guidance (`true_cfg_scale > 1`) | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| **Performance** | Prefix KV precompute (text/condition-image graph decoupled from the per-step target graph) | ✅ |
| | Spatial Tiling (VAE decode) | ✅ |
| | Continuous request batching | Not yet ported |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen-Image 2.1
- Not yet ported: the upstream feature exists but this Neuron port's current scope is
  text-to-image only (see [Known limitations](#known-limitations))

### Recommended configuration

The recommended configuration runs on 16 NeuronCores (one row of four `trn2` chips at `LNC=2`),
matching the
[stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage.yaml):
`tensor_parallel_size=16`, 7.2 s warm per 1024x1024 / 40-step image. The DiT's 32 attention heads
split 2 per rank; the Qwen3-VL encoder's 32 query heads split 2 per rank while each of its 8 KV
heads is replicated on 2 ranks (TP above the KV-head count). Two smaller layouts are provided as
alternatives:
[`qwen_image21_stage_tp8.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage_tp8.yaml)
(8 cores, an adjacent chip pair, 10.5 s) and
[`qwen_image21_stage_tp4.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage_tp4.yaml)
(one chip, 16.7 s). A single-core (`tensor_parallel_size=1`) config is also provided for
bring-up and small checkpoints
([`qwen_image21_stage_tp1.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage_tp1.yaml));
the full BF16 model (7B DiT + 8B text encoder, ~31 GB) does not fit the ~24 GB HBM of one `LNC=2`
core. Tensor-parallel groups must be torus-legal: 4 cores inside one chip, 8 on an adjacent chip
pair, 16 on one row of chips.

### Prefix/target DiT split

Block-causal attention plus `causal_condition` modulation means the prefix's (text and any
condition-image tokens') keys and values never depend on the denoising step. This port exploits
that directly rather than through upstream's step-1 KV cache: `forward_prefix` runs once per
prompt (and per CFG branch) as its own fixed-shape graph bucketed by prefix token count, and
`forward_target` — the one invoked every denoising step — only recomputes the target image's
tokens against the cached prefix K/V. The two graphs compile independently and the prefix graph's
cost is amortized to one call per request regardless of step count.

### Tiled VAE decode

The VAE decodes the image in fixed `QWEN_IMAGE_VAE_TILE` latent-pixel tiles (default `16,4`: 256px
tiles with a 4-latent / 64px overlap) with a linear-ramp feather blend, so one compiled graph
serves every output resolution instead of a new graph (and a new compile) per shape. The VAE is
not sharded, so every TP rank holds a full copy; the tiles are dealt round-robin across all TP
ranks (the shared `run_tiles` helper, `QWEN_IMAGE_VAE_PARALLEL=1`, the default) and gathered to
rank 0 for the blend, so 25 tiles at 1024x1024 take 7 rounds at TP=4 and 4 at TP=8 instead of 25.
The result is bit-identical to a rank-0-only decode. `QWEN_IMAGE_VAE_PARALLEL=0` restores the
rank-0-only decode (the other ranks then skip the VAE and its compile).

## Accuracy Evaluation

**Method:** a full-size gate at every published layout: the device BF16 run vs a CPU fp32 run of the
same Neuron pipeline from the same seed, judged against the CPU-BF16 band (the CPU BF16 run's own
error vs fp32, i.e. what the BF16 dtype alone costs). Pass bar: device rel-L2 ≤ 2 × CPU-BF16
band + 0.5%. Two tensors are gated: the step-0 velocity (one DiT forward from identical noise,
teacher-forced) and the final latents after 40 steps. Every rank's outputs are compared bit for
bit (`vllm_omni_neuron.testing.check_rank_agreement`), not just rank 0's. Pixels are reported,
not gated: the VAE decode is not bit-reproducible, so latents are the closer check on the DiT
(see [Optimizing offline video generation](../model-dev/optimizing-offline-video-generation.md)
for the method on another model).

**Qwen-Image 2.1, BF16, 1024x1024, 40 steps, seed 7** (CPU-BF16 band: step-0 velocity 0.71%,
final latents 4.16%; bar 1.92% / 8.82%):

| Layout | Step-0 velocity rel-L2 | Final-latent rel-L2 | Final-latent cosine | All ranks bit-identical | Device VAE vs host fp32 VAE (same latents) | Image PSNR vs fp32 golden |
|---|---|---|---|---|---|---|
| TP=4 | 0.51% | 4.25% | 0.9991 | ✅ 4/4 | 38.4 dB | 32.3 dB |
| TP=8 | 0.52% | 3.39% | 0.9994 | ✅ 8/8 | 38.4 dB | 33.6 dB |
| **TP=16** (recommended) | 0.54% | 4.83% | 0.9988 | ✅ 16/16 | 38.4 dB | 32.0 dB |

**2048x2048, 40 steps** (a 40-step CPU fp32 trajectory at 16,384 tokens is impractical, so the
step-0 velocity is the numeric gate; CPU-BF16 band 0.84%, bar 2.18%):

| Layout | Step-0 velocity rel-L2 | Cosine | All ranks bit-identical | Device VAE vs host fp32 VAE (same latents) |
|---|---|---|---|---|
| TP=8 | 0.74% | 0.99998 | ✅ 8/8 | 36.6 dB |
| **TP=16** | 0.77% | 0.99997 | ✅ 16/16 | 36.6 dB |

The TP=8 and TP=16 2048 final latents agree to 1.5% rel-L2, and both have the same per-row
statistics as the 1024 output (no banding).

Every row lands at or near its CPU-BF16 band, so the Neuron execution path adds no measurable
error beyond the BF16 dtype. Pixel PSNR against the golden image is lower than the latent numbers
suggest because 40 steps of BF16 drift (a CPU BF16 run drifts just as far) move fine texture.

**Qualitative check (BF16, TP=16, 1024x1024, 40 steps):** a served request for *"A cozy bookshop
window on a rainy evening, warm light, a hand-lettered sign that reads 'Open Late'"* produced a
coherent image with a legible hand-lettered sign, consistent warm lighting and glass reflections,
and no visible seam at the VAE's tile boundaries.

**Reproduce:**

```bash
# CPU fp32 + BF16 reference (no NeuronCores); add --probe-only for the step-0 velocity alone
python -m test.neuron.test_qwen_image_gate --mode oracle --model <qwen-image-21 checkpoint> --height 1024 --width 1024 --steps 40 --out ./gate/oracle-1024
# device run at a layout: every rank saves its outputs
torchrun --nproc_per_node 16 -m test.neuron.test_qwen_image_gate --mode device --model <qwen-image-21 checkpoint> --height 1024 --width 1024 --steps 40 --out ./gate/tp16-1024
# compare on the host
python -m test.neuron.test_qwen_image_gate --mode compare --model <qwen-image-21 checkpoint> --oracle ./gate/oracle-1024 --device-dir ./gate/tp16-1024
```

**Regression tests** (`test/neuron/test_qwen_image_21_accuracy.py`, one NeuronCore, random-weight
tiny checkpoint from `test/unit/test_qwen_image_tiny.py`):

1. Component three-way (`assert_close_three_way`, fp32 CPU / BF16 CPU / BF16 Neuron) for the text
   encoder, the DiT prefix + target graphs, and the VAE decode.
2. Single denoising step on Neuron vs the diffusers `QwenImage21Pipeline` CPU fp32 latent.
3. End-to-end 8-step image vs the CPU fp32 golden image (SSIM), plus a same-seed determinism check.

```bash
QWEN_IMAGE21_TINY=<tiny checkpoint> pytest test/neuron/test_qwen_image_21_accuracy.py
```

## Performance

Warm per-stage latency, BF16, TP=16 (recommended), 1024x1024, 40 steps, median of 3 requests
after one uncounted warm-up request (NEFF cache warm; every stage timed to device completion;
measured on a quiet host, 1-minute load average below 20 before every timed request):

| Stage | Time | Share |
|---|---|---|
| Text encoder | 0.02 s | 0.3% |
| DiT prefix | 0.01 s | 0.2% |
| DiT target (40 steps, 0.157 s/step) | 6.26 s | 87% |
| VAE decode (25 tiles dealt over 16 ranks) | 0.89 s | 12% |
| **Total** | **7.2 s** | |

**Layout sweep** (1024x1024, 40 steps, warm median of 3, quiet host; rel-L2 = the full-size
final-latent gate above at the same layout, bar 8.82%; HBM = runtime-reported per core; first
request = load done to first image with the NEFF cache already holding every graph):

| Layout | Cores | Warm total | DiT step | VAE decode | True CFG (`true_cfg_scale=4`)¹ | Rel-L2 vs fp32 | HBM / core | First request |
|---|---|---|---|---|---|---|---|---|
| TP=4, VAE on rank 0 only (`QWEN_IMAGE_VAE_PARALLEL=0`)² | 4 | 23.9 s | 0.344 s | 10.11 s | — | 4.25% | 8.9 GB | — |
| TP=4, VAE tiles over 4 ranks | 4 | 16.7 s | 0.343 s | 2.87 s | 30.5 s | 4.25% | 8.9 GB | 110 s |
| TP=8, VAE tiles over 8 ranks | 8 | 10.5 s | 0.218 s | 1.68 s | 19.2 s | 3.39% | 5.4 GB | 89 s |
| **TP=16, VAE tiles over 16 ranks** | 16 | **7.2 s** | 0.157 s | 0.89 s | 13.5 s | 4.83% | 3.9 GB | 336 s³ |

¹ True CFG is measured for speed only, not accuracy-checked (the gates above run without CFG).
² Same DiT as the gated TP=4 row; its image is bit-identical to the tile-parallel decode (max
absolute pixel difference 0, same process).
³ The first engine of the timing job; no compile was reported, so the extra time is graph and
runtime loading on cores that had not run this model before. Earlier TP=16 runs measured 95 s.

**2048x2048, 40 steps** (warm median of 3, quiet host; step-0 rel-L2 = the 2048 gate above, bar 2.18%):

| Layout | Cores | Warm total | DiT step | VAE decode | Step-0 rel-L2 vs fp32 | HBM / core | First request |
|---|---|---|---|---|---|---|---|
| TP=8 | 8 | 75.3 s | 1.709 s | 6.78 s | 0.74% | 13.7 GB | 441 s |
| **TP=16** | 16 | **44.5 s** | 1.019 s | 3.56 s | 0.77% | 8.8 GB | 265 s |

Both first-request figures are with the 2048 graphs already in the NEFF cache (no compile
reported); the cold first request (gate run, compile included) was 477 s at TP=8 and 249 s at
TP=16. Cache-warm first requests at 2048 are dominated by loading the large unblocked target
graph and vary between runs (TP=16: 99 s and 265 s).

The tile-parallel VAE decode is bit-identical to the rank-0-only decode (max absolute pixel
difference 0) and 3.5x faster at TP=4. The DiT step scales 1.58x from 4 to 8 cores and 1.39x from
8 to 16. True CFG doubles the DiT time (the conditional and unconditional branches run one after
the other on the same cores); it is off unless a request sets `true_cfg_scale > 1` with a negative
prompt, as upstream.

Cold-compile latency at TP=4 for a new shape, first served request: 445 s (peak host RSS 55 GB). A
second independent process issuing the same seeded request produced a bit-identical image (max
absolute pixel difference 0), confirming determinism across processes.

**Reproduce:**

```bash
torchrun --nproc_per_node 16 -m test.neuron.test_qwen_image_m3_perf --model <qwen-image-21 checkpoint> --height 1024 --width 1024 --steps 40 --repeats 3 --cfg 4.0 --parity <m1 parity dir>
# TP=4 with the rank-0-only VAE comparison:
torchrun --nproc_per_node 4 -m test.neuron.test_qwen_image_m3_perf --model <qwen-image-21 checkpoint> --height 1024 --width 1024 --steps 40 --repeats 3 --vae-compare
```

## Known limitations

- **Text-to-image only.** Qwen-Image 2.1's condition-image input path (image-conditioned
  generation through the same block-causal prefix) is not yet ported — `img_shapes` /
  `assemble_prefix_inputs` support it internally and are unit-tested on CPU, but the served
  pipeline's `forward()` raises `NotImplementedError` on `multi_modal_data`.
- **Batch size is limited to one.** Concurrent requests run serially rather than as a batched
  forward pass (`supports_request_batch = False`).
- **VAE decode uses 256px tiles.** A 512px tile (`QWEN_IMAGE_VAE_TILE=32,4`, 9 instead of 25 tile
  launches at 1024x1024) did not finish compiling within a 30-minute budget: the single-tile VAE
  graph's compile alone ran for over 28 minutes at ~52 GB host RSS. The 256px default compiles in
  a few minutes and is what the performance numbers above use.
- **2048x2048 (upstream's native resolution) runs at TP=8 and TP=16 only.** Above 4,096 query rows
  the DiT target graph's attention runs unblocked (`QWEN_IMAGE_ATTN_QBLOCK_MAX_ROWS`, default 4096):
  with the 1,024-row query blocking used at 1024x1024, neuronx-cc 2.27 miscompiles the 16,384-token
  target graph from its second layer on (every query block after the first comes out wrong, at any
  block size), which produced banded, incoherent 2048 images in earlier builds. TP=4 at 2048 does
  not compile (`NCC_IBIR229`, state-buffer overflow in a collective). The unblocked 2048 graph takes
  44.5 s warm at TP=16 (75.3 s at TP=8); see the 2048 table in [Performance](#performance).
- **No CFG-parallel or context-parallel layout.** True CFG runs both branches on the same TP group
  one after the other; splitting them over two groups would need a second process-group axis in
  this package. Context parallelism is not implemented: at 1024x1024 a pass is ~4,100 tokens
  (4,096 target + the text prefix), so CP=2/4 would leave ~2,050/~1,025 tokens per rank while
  every rank still reads the TP-sharded weights, and TP up to 16 is still available and measured
  above.
- **Qwen-Image 2.1 prompt enhancer (PE-T2I) and Qwen-Image-Edit-2511 are not ported.** This PR
  covers the base text-to-image pipeline only.

## Tutorials

- [Tutorial: Deploy Qwen-Image 2.1 with vLLM Omni Neuron](../tutorials/tutorial-qwen-image21.md)
