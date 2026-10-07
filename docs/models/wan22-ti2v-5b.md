# Wan2.2-TI2V-5B and FastWan2.2-TI2V-5B (DMD2) Model Card

<!-- meta: description: Model card for Wan2.2-TI2V-5B and its 3-step DMD2 distillation
FastWan2.2-TI2V-5B on AWS Trainium2 with the vLLM Omni Neuron plugin: supported features,
the recommended 64-core configuration (with 32- and 16-core alternatives), accuracy on Neuron,
performance, and known limitations. -->
<!-- meta: keywords: Wan2.2, Wan2.2-TI2V-5B, FastWan, DMD2, few-step, model card, text-to-video,
image-to-video, video generation, diffusion, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16,
tensor parallelism, context parallelism, Megatron sequence parallelism, VAE tiling -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) is a 5B-parameter
text-and-image-to-video diffusion transformer from Wan-AI. It pairs a 30-layer DiT (24 heads of
128, FFN 14336) with the high-compression Wan2.2 VAE (48 latent channels, 16x spatial and 4x
temporal compression, patchified 2x2) and the UMT5-XXL text encoder. Its native output is
1280x704 at 24 fps, 121 frames. When an image is given it conditions the first latent frame
through per-token timesteps (`expand_timesteps`), so one checkpoint serves text-to-video and
image-to-video.

[FastWan2.2-TI2V-5B](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) is the
same architecture distilled by FastVideo with DMD2 to three denoising steps (timesteps 1000, 757,
522) with classifier-free guidance distilled away.

Both models are now supported for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on Trainium2
hardware.

**License:** Apache-2.0 for both checkpoints; neither is gated.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Wan2.2-TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Trn2 | BF16 |
| FastWan2.2-TI2V-5B (DMD2, 3 steps) | [FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video (TI2V-5B, 50-step UniPC, CFG) | ✅ |
| | Text-to-video (FastWan DMD2, 3 steps, no CFG) | ✅ |
| | Image-to-video (TI2V-5B, image conditions the first latent frame) | ✅ (host condition encode) |
| | 1280x704x121 (native) and 832x480x81 | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) with Megatron sequence parallelism | ✅ |
| | Context Parallelism (CP), including sequence lengths that do not divide by the CP degree | ✅ |
| | CFG Parallelism | ✅ (TI2V-5B; DMD2 has no CFG) |
| | VAE Patch Parallelism | ✅ |
| **Compilation** | torch.compile (native Lite backend) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Wan2.2-TI2V-5B / FastWan2.2-TI2V-5B.
- Limited: accepted, with the caveat noted under [Known limitations](#known-limitations).
- `-`: not supported.

### Recommended configuration

**TI2V-5B (text- and image-to-video): the whole trn2.48xlarge, 64 logical NeuronCores at LNC=2,
TP=4 x CP=8 x CFG-parallel 2** (`wan22_ti2v_stage_64c.yaml`). TP=4 inside each chip with sequence
parallelism (24 heads, 6 per rank), CP=8 across the eight chips (two rows) of each half (`ring_degree: 8`),
and CFG-parallel 2: the conditional and unconditional branches run at the same time, one per half of
the box. CP splits the 1280x704x121 sequence (27,280 tokens) into 3,410 tokens per rank, and the VAE
decode tiles are spread over all 64 ranks. Generate at the native 1280x704x121 for best quality.

```yaml
# engine_args in examples/wan2_2/wan22_ti2v_stage_64c.yaml
model_config:
  tp_sequence_parallel: true
parallel_config:
  tensor_parallel_size: 4
  ring_degree: 8
  cfg_parallel_size: 2
  vae_patch_parallel_size: 64   # VAE decode tiles spread over all 64 ranks
```

```bash
python examples/wan2_2/run_ti2v.py --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --stage-config examples/wan2_2/wan22_ti2v_stage_64c.yaml
# image-to-video: --stage-config examples/wan2_2/wan22_ti2v_i2v_stage_64c.yaml --image <first frame>
```

With fewer cores: one row of 16 cores, TP=4 x CP=4 with sequential CFG (`wan22_ti2v_stage_16c.yaml`,
68.4 s), or 32 cores, TP=4 x CP=4 x CFG2 (`wan22_ti2v_stage.yaml`; not re-timed since the
context-parallel attention change, see Performance). On 64 cores TP=8 x CP=4 x CFG2 is as fast as the
recommended layout (23.6 s both).

**FastWan DMD2: 64 cores, TP=8 x CP=8** (`wan22_dmd2_stage_64c.yaml`, 7.8 s). It has no CFG branch, and
its 3-step denoise (0.7 s at 1280x704x121 on 64 cores) is small next to the 4.4 s VAE decode, so cores
past 32 buy little. The 16-core `wan22_dmd2_stage.yaml` (TP4 x CP4) also runs it; it has not been
re-timed since the attention change. The checkpoint is selected automatically from its
`model_index.json` (`WanDMDPipeline`):

```bash
python examples/wan2_2/run_ti2v.py --model-path FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
    --stage-config examples/wan2_2/wan22_dmd2_stage_64c.yaml
```

A single chip (`--tp 4 --cp 1 --devices 0,1,2,3`) also runs the model at smaller shapes.

## Accuracy Evaluation

Accuracy follows the three tiers of
[Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md). Tiers 1
and 2 compare BF16 Neuron against diffusers' own CPU FP32 and CPU BF16 runs on identical inputs
with `vllm_neuron.accuracy.testing.assert_close_three_way`. Tier 3 runs the whole served generation
(text encode, three DMD2 steps, VAE patch-parallel decode) against an independent diffusers CPU
FP32 reference driven by FastVideo's DMD2 rule, with diffusers' CPU BF16 run as the floor, plus an
end-to-end repeatability check against a reviewed Neuron golden.

| Tier | Check | Trn2 result | Gate |
|---|---|---|---|
| 1 | One DiT call, positive branch (t=999, 832x480 geometry reduced to 128x128x9) | L2 ratio 0.32, L-inf ratio 0.49 | L2 < 3.0, L-inf < 5.0 |
| 1 | One DiT call, unconditional (negative-prompt) branch | L2 ratio 0.16, L-inf ratio 0.16 | L2 < 3.0, L-inf < 5.0 |
| 1 | CFG combine, guidance 5 | L2 ratio 0.17 | L2 < 3.0 |
| 2 | One full denoising step through the served pipeline vs diffusers `WanPipeline`, guidance 1 | rel L2 vs FP32 3.0% (diffusers BF16: 3.8%); L2 ratio 0.80, L-inf ratio 0.98 | three-way default rule |
| 2 | Same, guidance 5 | rel L2 vs FP32 6.7% (diffusers BF16: 8.6%); L2 ratio 0.78, L-inf ratio 0.99 | three-way default rule |
| 3 | FastWan DMD2 1280x704x17 e2e (16 cores, VAE patch parallel 16) vs diffusers CPU FP32, measured with the earlier ring-attention kernel | latent rel L2 41.7% (diffusers BF16: 32.3%); frame SSIM mean 0.724 (diffusers BF16: 0.825) | rel L2 and 1 - SSIM <= 2x BF16 floor + 0.005 |
| 3 | FastWan DMD2 832x480x81 e2e (16 cores) vs reviewed golden | per-frame SSIM mean 1.0000, min 1.0000 | per-frame SSIM >= 0.90, mean >= 0.95 |

### Full-size gate (1280x704x121)

Every check runs from shared injected noise (seed 42), TI2V-5B at 50 steps and guidance 5. The 50-step
models are checked step by step instead of against a full 50-step CPU run: at steps 0, 24 and 49 the
DiT input the device actually saw is re-run through diffusers' transformer on CPU in FP32 (reference)
and BF16 (floor). Bar everywhere: error <= 2 x the CPU BF16 error + 0.005.

**Context-parallel attention.** With CP > 1 the self-attention all-gathers K/V over the CP group and
runs flash attention with the true row maximum and an FP32 softmax. This is the default. An earlier
version used a ring-attention kernel that keeps K/V local and shifts every softmax row by a
Cauchy-Schwarz bound (`scale * |q_i| * max_j |k_j|`) instead of the row maximum. At the late,
low-noise steps of this model the bound overshoots the row maximum by more than about 85 (in
exponent units), every probability of the row underflows, and the kernel returns an all-zero
attention row. A CPU emulation of the kernel at step 49 zeroed about 34,000 query rows and
reproduced the device error (positive-branch prediction 5.79% vs the device's 5.69% from CPU FP32;
first latent frame 13.2% vs 13.1%), while the same emulation with the true row maximum gave 2.11%
(diffusers BF16: 2.03%). The ring kernel is still available with `WAN22_CP_RING_ATTENTION=1`; it is
about 10% faster end to end at 16 cores (warm 59.0 s vs 65.1 s, denoise 49.9 vs 55.8 s) but fails
the step check, so it is not recommended.

Teacher-forced step checks on the default path, 16 cores (TP4 x CP4, CFG sequential), text-to-video:

| Step | Positive-branch prediction | Its first latent frame | CFG-combined prediction |
|---|---|---|---|
| 0 | 1.21% (bar 2.62%) | 1.07% (BF16 1.08%) | 3.65% (bar 7.48%) |
| 24 | 1.07% (bar 2.51%) | 1.15% (BF16 1.13%) | 4.95% (bar 10.28%) |
| 49 | 2.16% (bar 4.61%) | 2.42% (BF16 2.29%) | 11.26% (bar 22.09%) |

All 16 ranks hold the same final latent (SHA-256), and the device error now tracks diffusers' own
BF16 error at every checked step, including the first latent frame.

The same checks at the recommended 64-core layout (TP4 x CP8 x CFG2):

| Step | T2V positive branch | T2V first latent frame | T2V CFG-combined | I2V positive branch | I2V first latent frame | I2V CFG-combined |
|---|---|---|---|---|---|---|
| 0 | 1.17% (bar 2.62%) | 1.08% (BF16 1.08%) | 3.60% (bar 7.48%) | 0.55% (bar 2.57%) | 3.41% (BF16 3.68%) | 2.77% (bar 11.13%) |
| 24 | 1.07% (bar 2.51%) | 1.16% (BF16 1.12%) | 4.96% (bar 10.30%) | 1.01% (bar 3.22%) | 9.13% (BF16 9.34%) | 5.68% (bar 15.16%) |
| 49 | 2.14% (bar 4.59%) | 2.42% (BF16 2.40%) | 11.17% (bar 21.97%) | 1.12% (bar 5.04%) | 1.54% (BF16 2.70%) | 5.56% (bar 24.39%) |

In image-to-video the first latent frame is the encoded condition image; its error is the same on
the device and in diffusers' BF16 run.

At the recommended 64-core layouts (VAE patch parallel 64), same noise and seed:

| Check | TI2V-5B T2V, TP4 x CP8 x CFG2 | TI2V-5B I2V, TP4 x CP8 x CFG2 | FastWan DMD2, TP8 x CP8 | Bar |
|---|---|---|---|---|
| All 64 ranks hold the same final latent (SHA-256) | 64/64 identical | 64/64 identical | 64/64 identical | identical |
| Same inputs at the 16-core layout checked above, every frame | SSIM mean 0.915, min 0.846 (frame 29), largest frame-to-frame drop 0.030; PSNR 28.6 / 26.0 dB; latent rel L2 10.0% | - | - | smooth decline, no frame < 0.6, mean >= 0.8 |
| Full e2e vs diffusers CPU FP32 (3 steps + decode) | - | - | latent 31.6% (BF16 26.8%); SSIM mean 0.826 (BF16 0.876), min 0.750 | 2x BF16 + 0.5% |
| Every frame vs host FP32 tiled decode of the same latent (256 px tiles, 192 px stride) | SSIM mean 0.9988, min 0.9975; PSNR mean 49.8, min 45.5 dB | SSIM mean 0.9993, min 0.9985; PSNR mean 51.4, min 45.8 dB | - | 1 - SSIM <= 2x host BF16 + 0.005 |
| For information: every frame vs host FP32 untiled decode | SSIM mean 0.991, min 0.984; PSNR mean 41.1, min 39.2 dB | SSIM mean 0.997, min 0.995; PSNR mean 44.3, min 40.9 dB | - | - |

The 64-core versus 16-core comparison is a bug detector, not an accuracy measure: the two layouts
reduce in different orders, and guidance 5 over 50 steps amplifies that rounding into a slowly
drifting, equally valid sample; an ordering bug instead shows as a gross error from frame 0. The
other 64-core layouts in Performance also hold one final latent on all 64 ranks. The decode rows
score the VAE alone (the decode does not use CP attention); they were run on latents from the
earlier ring-kernel denoise.

The decode is scored against a host FP32 decode with the same tiling, because tiling itself costs
accuracy: a host FP32 tiled decode differs from the host FP32 untiled decode of the same latent by
40.9 dB mean PSNR (SSIM 0.990) on the text-to-video clip and 44.0 dB (SSIM 0.997) on the
image-to-video clip, which is the whole of the device's distance from the untiled decode (41.1 /
44.3 dB). The device decode is within 49.8 / 51.4 dB of the host tiled decode; the remaining
difference includes the device's tile layout (full-size tiles ending at the frame edge, a
start-aware blend) where the host reference uses narrower edge tiles. The BF16 floor for this check
is the untiled host BF16 decode (1 - SSIM 0.0005).

The tier-3 reference floor is wide because the DMD2 student is ill-conditioned between its fixed
timesteps: at t=757 a 1e-3 relative input perturbation moves its FP32 prediction by about 1% at
1280x704 (6% at 448x256), and shifting the timestep by one unit moves it by about 45%. Ordinary BF16
rounding in the first step therefore grows into a visibly different, equally valid sample (same
scene and layout, different pose). Small off-native shapes make this worse: at 448x256x17 diffusers'
own BF16 run is as far from FP32 as an unrelated sample (rel L2 0.77, SSIM 0.28), so tier 3 uses the
native resolution.

BF16 Neuron is closer to FP32 than diffusers' own BF16 CPU run at every tier-1 check (ratios below
1.0). Over a full trajectory, guidance amplifies ordinary BF16 rounding: with identical initial
noise, a 4-step guidance-5 run differs from CPU FP32 by 38% relative L2 on Neuron and by 41% on
diffusers' own CPU BF16 run, while the same 4 steps at guidance 1 differ by 4% on both. The
Neuron trajectory is never further from FP32 than the reference's own BF16 trajectory.

**Reproduce (on a Trn2 instance, weights downloaded):**

```bash
WAN22_MODEL=<checkpoint dir> pytest -s test/neuron/test_wan2_2_dit_accuracy.py
WAN22_MODEL=<checkpoint dir> WAN22_GUIDANCE=5 pytest -s test/neuron/test_wan2_2_ti2v_pipeline_accuracy.py
WAN22_MODEL=<dmd2 checkpoint dir> WAN22_REF_WORKDIR=<work dir> pytest -s test/neuron/test_wan2_2_ti2v_e2e_reference_accuracy.py
WAN22_MODEL=<dmd2 checkpoint dir> WAN22_GOLDEN=<golden.npy> pytest -s test/neuron/test_wan2_2_ti2v_e2e_accuracy.py
```

CPU unit tests (no device): `pytest test/unit/test_wan2_2_*.py` checks the Neuron DiT against
diffusers on tiny random-weight checkpoints that keep the real layout, under TP, sequence
parallelism and CP (including a padded, non-divisible sequence), plus the DMD2 sampler math.

## Performance

Warm latency per clip, trn2.48xlarge, BF16: the median of three warm requests after one that loads
the model, measured on a quiet host (1-minute load below 20 before every timed request, no other
jobs). Each stage is timed to its device completion; "e2e" is the synced in-pipeline forward (text +
denoise + VAE) of the last request. "First request" is the request after model load with the compiled
graphs cached; "cold" adds the graph compile.

| Model | Layout | Cores | Warm (median) | e2e: denoise / VAE | First request (cold) | Checked |
|---|---|---|---|---|---|---|
| TI2V-5B T2V, 50 steps CFG 5 | **TP4 x CP8 x CFG2, VAE pp 64** (recommended) | 64 | **23.6 s** | 16.8 / 4.3 s | 133 s (502 s) | 64/64 ranks; step checks; vs 16 cores SSIM 0.915 |
| TI2V-5B T2V, 50 steps CFG 5 | TP8 x CP4 x CFG2, VAE pp 64 | 64 | 23.6 s | 16.7 / 4.3 s | 148 s (460 s) | 64/64 ranks |
| TI2V-5B T2V, 50 steps CFG 5 | TP4 x CP16, CFG sequential, VAE pp 64 | 64 | 25.9 s | 19.1 / 4.3 s | 125 s (451 s) | 64/64 ranks |
| TI2V-5B T2V, 50 steps CFG 5 | TP8 x CP8, CFG sequential, VAE pp 64 | 64 | 28.9 s | 21.9 / 4.4 s | 142 s (472 s) | 64/64 ranks |
| TI2V-5B I2V, 50 steps CFG 5 | **TP4 x CP8 x CFG2, VAE pp 64** (recommended) | 64 | **27.2 s** | 17.9 / 4.3 s, encode 2.4 s | 129 s (421 s) | 64/64 ranks; step checks |
| FastWan DMD2, 3 steps | **TP8 x CP8, VAE pp 64** (recommended) | 64 | **7.8 s** | 0.68 / 4.35 s | 122 s (463 s) | 64/64 ranks; e2e vs CPU FP32 |
| TI2V-5B T2V, 50 steps CFG 5 | TP4 x CP4, CFG sequential, VAE pp 16 | 16 | 68.4 s (66.0 / 68.4 / 72.3) | 61.3 / 8.2 s | 168 s (538 s) | 16/16 ranks; step checks |

Warm requests repeat to within 0.3 s on 64 cores. The 16-core warm requests grew from 66.0 to 72.3 s
within the run (the same layout gave 65.0 / 65.2 s in an earlier run on another row of the box), so
its median carries about 5% spread. Sources (fleet job ids on the development instance): timings
`1006-114157` (64 cores) and `1006-114224` (16 cores); cold first requests and rank checks
`1006-103210` (64 cores) and `1006-074101` (16 cores); step checks `1006-124752` (64 cores) and
`1006-081203` (16 cores).

CFG-parallel is the best use of extra cores for TI2V-5B: it halves the DiT work per step with one
noise-prediction exchange per step. TP8 x CP4 x CFG2 ties the recommended TP4 x CP8 x CFG2; layouts
without CFG-parallel are slower. FastWan DMD2's 3-step denoise is 0.7 s, so its latency is mostly the
VAE decode, which does not get faster past 28 ranks: 1280x704 has 28 decode tiles, and smaller tiles
that give every rank work (208x192 px, 50 tiles) were slower (decode 6.8 s vs 4.4 s), since each tile's
frame-by-frame decode costs about the same and the gather and blend grow with the tile count.

**832x480x81, 16 cores (TP4 x CP4).** 8190 tokens do not divide by CP=4, so these shapes always took
the all-gather attention path and the numbers below are unaffected by the attention change:

| Configuration | VAE pp 1 | VAE pp 16 | Breakdown with VAE pp 16 | First request |
|---|---|---|---|---|
| TI2V-5B 832x480x81, 50 steps, CFG 5 | 27.9 s | 15.6 s | text 0.03 s, denoise 13.8 s (0.28 s/step), VAE 1.78 s | 528 s |
| FastWan DMD2 832x480x81, 3 steps | 14.4 s | 2.24 s | text 0.02 s, denoise 0.44 s (0.15 s/step), VAE 1.78 s | 86 s |

Patch-parallel decode is bit-identical to the single-rank decode on these clips (per-frame SSIM
1.000). These first-request times assume the text encoder and VAE graphs are already in the compile
cache from an earlier shape.

**Superseded until re-measured.** These 1280x704x121 timings were measured with the earlier
ring-attention kernel (see [Full-size gate](#full-size-gate-1280x704x121)), which is no longer the
default; their layouts still run, but have not been re-timed:

| Model | Layout | Cores | Warm (ring kernel) |
|---|---|---|---|
| TI2V-5B T2V | TP4 x CP4 x CFG2, VAE pp 32 / VAE pp 16 | 32 | 35.9 / 35.7 s |
| TI2V-5B T2V | TP8 x CP4, CFG sequential | 32 | 40.6 s |
| TI2V-5B I2V | TP4 x CP4 x CFG2, VAE pp 32 | 32 | 40.8 s (one warm request) |
| FastWan DMD2 | TP4 x CP4 / TP8 x CP4 / TP4 x CP8 (VAE pp 16 / 32) | 16 / 32 / 32 | 10.4 / 10.2 / 9.3 / 8.6 s |
| TI2V-5B T2V | TP4 x CP4, CFG sequential, VAE pp 1 | 16 | 103.5 s |

## Known limitations

- **Tiled VAE decode.** Decoding 1280x704 in 256 px tiles costs about 41 dB PSNR (SSIM 0.990) against
  an untiled decode of the same latent; the host reference decoder with the same tiling shows the same
  cost.
- **832x480 output is softer than the native 1280x704.** TI2V-5B is trained at 1280x704; at
  832x480 late frames can look soft as a subject moves toward the camera (no decode collapse:
  per-frame sharpness stays flat). Use 1280x704 for quality.
- **Image-to-video condition encode runs on the host.** The compiled Wan2.2 VAE encoder does not compile
  above 192 px on Trn2, so the single condition frame is VAE-encoded on the output rank's host in FP32
  and broadcast to the other ranks: 2.4 s per request at 1280x704 on 64 cores (`vae_encode_seconds` in
  the perf metrics), included in the image-to-video latency under Performance (27.2 s warm on 64
  cores). Measured against the same pipeline on CPU (1280x704x17, 4 steps, guidance 5, same noise):
  video rel-L2 6.8% at TP4 and 8.3% at TP4 x CP4, condition frame 47.7 dB.
- **Context-parallel attention all-gathers K/V.** Every rank gathers the whole sequence's keys and
  values for each self-attention layer, then masks out pad tokens when the token count is not a
  multiple of the CP degree (for example 832x480x81 = 8190 tokens on CP=4). The faster ring-attention
  kernel (`WAN22_CP_RING_ATTENTION=1`) zeroes attention rows at the late denoising steps of this model
  (see [Full-size gate](#full-size-gate-1280x704x121)) and cannot mask pad tokens, so it is opt-in
  and used only for evenly divisible sequences.
- **DMD2 sampling controls.** FastWan DMD2 ignores `num_inference_steps`, guidance scale and the
  negative prompt (the distilled contract is 3 steps without CFG); a stage config may override the
  timesteps with `model_config.dmd_denoising_steps`.
- **VAE decode tiles.** The decode tile split always ends in a narrower last row and column. Keep
  those edge tiles at 128 px or wider (the default 256/192 px tiles do at 1280x704); thinner device-decoded
  edge tiles have shown corrupted borders in long clips. Decode tiles are set with
  `model_config.vae_tile_sample: [min_h, min_w, stride_h, stride_w]` (pixels).

## Tutorials

- [Tutorial: Deploy Wan2.2-TI2V-5B and FastWan DMD2 with vLLM Omni Neuron](../tutorials/tutorial-wan22-ti2v-5b.md)
- [Quickstart: Offline generation with Wan2.2-TI2V-5B](../getting-started/quickstart-offline-serving-wan22-ti2v-5b.md)
