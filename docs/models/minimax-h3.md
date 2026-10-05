# MiniMax-H3 / FastH3 Model Card

<!-- meta: description: Model card for MiniMax-H3 and its distilled FastH3 students (4-step Dense-DataFree and
8-Step-V2) on AWS Trainium2 with the vLLM Omni Neuron plugin: text-to-video-and-audio generation, the recommended
TP=8 x CP=8 trn2 configuration on 64 cores, accuracy against a CPU fp32 reference, performance, and known limitations. -->
<!-- meta: keywords: MiniMax-H3, FastH3, FastVideo, model card, text-to-video, text-to-audio, joint audio-video,
diffusion, DMD2, video sparse attention, VSA, vLLM, vLLM Omni, Neuron, Trainium2, trn2, BF16, tensor parallelism,
context parallelism, VAE tile parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is MiniMax's 33B-parameter joint audio-video diffusion
transformer. A single stack of 50 transformer blocks (56 heads × 128, hidden 5376, plus 2 text-refiner blocks) runs
full self-attention over **one packed sequence** that holds the text conditioning, the stereo audio latents and the
video latents. Per-row AdaLN modulation selects one of three modalities. The text conditioner is Qwen3-VL-32B read at
`hidden_states[50]`. The video VAE has a 36-layer ViT decoder (16× spatial, 4× temporal), and the audio VAE is a
BigVGAN decoder at 32 kHz.

[FastH3](https://huggingface.co/collections/FastVideo/fastvideo-fasth3) (FastVideo) are DMD2-distilled students of
MiniMax-H3 that generate a clip in a few transformer forwards with guidance distilled away:

- **4-step Dense-DataFree**: 4 forwards with dense attention.
- **8-Step-V2**: 8 forwards over a trained step ladder, trained with Video Sparse Attention (VSA-H3, 80% sparsity)
  and a learned compression gate. Its sparse attention is part of the model, not an optional speed-up.

Both are supported for text-to-video-and-audio (t2va) inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Trainium2 (`trn2`).

**License:** [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), which
applies to the FastH3 students too. Read it before use: it restricts the territories and uses it covers.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| FastH3 4-step (dense) | [FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree) | Trn2 | BF16 |
| FastH3 8-Step-V2 (VSA) | [FastVideo/FastVideo-FastH3-8-Step-V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video-and-audio (t2va), stereo 32 kHz soundtrack | ✅ |
| | 256x256, 384x640, 704x1280 and 1344x768, 124 frames (5.17 s at 24 fps) | ✅ |
| | First/last-frame (FL2VA), reference (Ref2VA), base 50-step MiniMax-H3 | - |
| **Attention** | Dense (NKI `attention_cte` on NeuronCore-v3) | ✅ |
| | VSA-H3 sparse attention (8-Step-V2) | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP=8) | ✅ |
| | Context parallelism: CP=2, 4, 8 (dense); CP=2, 4 (VSA, at the sizes in Known limitations) | ✅ |
| | Video VAE tile parallelism across all stage ranks | ✅ |
| | CFG parallelism | n/a (guidance-distilled) |
| **AdaLN** | On device (default) or host-precomputed modulation tables | ✅ |
| **Text encoder** | Qwen3-VL-32B on device (TP=8, prompt-length buckets) or on the host CPU | ✅ |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested on trn2.
- `-`: not supported yet.

### Recommended configuration

On a trn2.48xlarge, use [`examples/minimax_h3/minimax_h3_stage.yaml`](../../examples/minimax_h3/minimax_h3_stage.yaml)
(the run script's default): `tensor_parallel_size: 8` × `ring_degree: 8` (context parallelism) on all 16 Trainium2
chips (64 logical NeuronCores at LNC=2). This is the layout of the 1344x768 headline numbers and accuracy gate below.
It has four parts:

- **DiT:** column-parallel over the heads (7 per rank), row-parallel output and FFN, with the AdaLN projection
  all-gathered per block. Each of the 8 context-parallel ranks holds an eighth of the packed sequence and the same
  TP=8 weight shard, and all-gathers K/V per layer. On trn2 the worker builds the groups on the physical mesh: each
  TP group is one adjacent chip pair, each CP group one core per pair. 8-Step-V2 keeps its sparse attention exact
  under CP: the sequence is laid out in VSA tile order, so each rank owns whole query tiles. For the dense
  checkpoints the TP ranks also split their CP slice between them outside attention and the FFN (sequence-parallel
  TP): norms, AdaLN modulation and the residual stream run on 1 / TP of the rows, with an all-gather before the
  q/k/v and FFN projections and a reduce-scatter after the output and down projections, in place of the all-reduce
  (`MINIMAX_H3_TP_SP=0` turns it off). The AdaLN projection weights are sharded over the CP ranks too
  (`MINIMAX_H3_ADALN_CP=0` turns it off).
- **Text encoder:** the first 50 Qwen3-VL-32B layers, TP=8 on the device (8 query heads and 1 KV head per rank,
  ~6 GiB per core), compiled once per prompt-length bucket (64 / 128 / 256 / 512 tokens). Every CP replica runs it on
  its own TP group, so no broadcast is needed. The last 16 prompts are cached.
- **Video VAE:** on device, with diffusers' spatial tiling kept on (256x256 tiles, 64 px overlap). The (temporal
  chunk × tile) decoder calls are dealt across all 64 ranks (one batched call per rank), and every rank composes
  its share of the output frames into one shared uint8 clip with the same stitch and cross-fade as diffusers.
- **Audio VAE:** on device, fp32, in 8 fixed time windows of the latent sequence (16 latent frames of context on each
  side), one window per rank on the stage's last 8 ranks after their video tiles; the waveform pieces go to rank 0.
  `model_config: {audio_vae: host}` decodes it on the host CPU of one rank instead (32 threads), overlapped with the
  video decode; that is also the default when the video VAE is not tile-parallel.

Smaller layouts: [`minimax_h3_stage_tp8cp4.yaml`](../../examples/minimax_h3/minimax_h3_stage_tp8cp4.yaml) (TP=8 × CP=4,
32 cores), [`minimax_h3_stage_tp8cp2.yaml`](../../examples/minimax_h3/minimax_h3_stage_tp8cp2.yaml) (TP=8 × CP=2,
16 cores; use it for 8-Step-V2 at 384x640, see Known limitations) and
[`minimax_h3_stage_tp8.yaml`](../../examples/minimax_h3/minimax_h3_stage_tp8.yaml) (TP=8, 8 cores).
`run.py --tp N --cp M` overrides either degree. `model_config: {text_encoder: host}` moves the text encoder to the
host CPU.

The checkpoint's `fastvideo_inference.json` selects the sampling contract automatically:

- forwards and step ladder;
- video and audio scheduler shifts (12/3 for 4-step, 10/3 for 8-Step-V2);
- VSA sparsity, if the checkpoint uses it.

## Accuracy Evaluation

**Full-size gate at the recommended configuration** (FastH3 4-step, 1344x768x124, seed 0, fixed prompt
embeddings, TP=8 × CP=8 on 64 cores). The reference is diffusers' `MiniMaxH3Transformer3DModel` run end to end on the
host CPU in fp32 with identical noise, layout and schedulers; the same run in bf16 is the floor. The bar is
**device error ≤ 2 × the CPU-bf16 error + 0.5%** (k = 2). Video is compared frame by frame: the reference, floor and
device latents decoded by one CPU fp32 VAE (isolates the DiT), and the device's own uint8 output from the device VAE.

| Metric | Trn2 (TP=8 × CP=8) | CPU bf16 floor | Bar | Result |
|---|---|---|---|---|
| first-step video velocity rel-L2 / cos | **4.29% / 0.9992** | 4.22% / 0.9993 | ≤ 8.94% | pass |
| final video latents rel-L2 / cos | **36.9% / 0.932** | 35.5% / 0.937 | ≤ 71.5% | pass |
| final audio latents rel-L2 / cos | 10.1% / 0.995 | 15.4% / 0.989 | - | - |
| video SSIM vs fp32, all 124 frames (mean / worst frame) | **0.657 / 0.601** | 0.658 / 0.607 | 1 - SSIM ≤ 0.690 | pass, every frame |
| video PSNR vs fp32 (mean / worst frame) | 19.2 / 18.5 dB | 19.6 / 18.3 dB | - | - |
| device-decoded video (device VAE, uint8) SSIM vs fp32 (mean / worst) | **0.656 / 0.601** | 0.658 / 0.607 | 1 - SSIM ≤ 0.690 | pass, every frame |
| device VAE vs CPU fp32 VAE on the same latents (PSNR mean / worst) | 53.2 / 50.3 dB | - | - | - |
| device waveform vs fp32 waveform, rel-L2 | **36.1%** | 30.5% | ≤ 61.6% | pass |
| device audio VAE vs CPU fp32 audio VAE on the same latents | 87.7 dB SNR | - | - | - |
| all 64 ranks hold bit-identical final latents (every request) | yes | - | - | pass |
| 4 requests in one process and a second process bit-equal | yes | - | - | pass |

At 768p the device run sits on the bf16 floor on every metric: bf16 rounding moves the 4-step trajectory to a
different, equally plausible clip (same scene and motion), and the device does it by the same amount as CPU bf16.
The decoded clip and the 704x1280 clip from the prompt text were checked by eye: a golden retriever running
through surf, no tile seams, flicker or colour shifts. Job 1005-142737 (device), CPU references on the host.

**Small-canvas gate** (256x256, 124 frames, seed 0, fixed prompt embeddings, same references and bar; also covers
8-Step-V2). The bar is applied to the
first-step video velocity, to the final video latents, and to the decoded video (all three runs decoded by one CPU
fp32 VAE, per-frame SSIM to the reference, as `1 - SSIM`). Two separate device runs must also be bit-equal.

| Checkpoint | Metric | Trn2 (TP=8) | CPU bf16 floor | Bar | Result |
|---|---|---|---|---|---|
| 4-step | first-step video velocity rel-L2 | **3.0%** | 3.9% | ≤ 8.3% | pass |
| 4-step | final video latents rel-L2 / cos | **20.0% / 0.980** | 25.6% / 0.967 | ≤ 51.8% | pass |
| 4-step | final audio latents rel-L2 | 13.9% | 8.1% | - | - |
| 4-step | decoded video SSIM vs fp32 (mean / min frame) | **0.686 / 0.572** | 0.634 / 0.536 | 1 - SSIM ≤ 0.737 | pass |
| 4-step | two device runs bit-equal | yes | - | - | pass |
| 8-Step-V2 | first-step video velocity rel-L2 / cos | **33.3% / 0.949** | 23.2% / 0.973 | ≤ 46.9% | pass |
| 8-Step-V2 | final video latents rel-L2 / cos | **53.7% / 0.855** | 39.3% / 0.922 | ≤ 79.1% | pass |
| 8-Step-V2 | final audio latents rel-L2 | 42.0% | 23.2% | - | - |
| 8-Step-V2 | decoded video SSIM vs fp32 (mean / min frame) | **0.473 / 0.352** | 0.556 / 0.418 | 1 - SSIM ≤ 0.893 | pass |
| 8-Step-V2 | two device runs bit-equal | yes | - | - | pass |

**Context parallelism, same gate.** CP changes the order of the attention reductions, so the run moves within the
bf16 band rather than matching TP=8 bit for bit:

| Layout | 4-step step-0 / final video | 8-Step-V2 step-0 / final video |
|---|---|---|
| TP=8 × CP=2 | 3.19% / 24.8% (bars 8.3% / 51.8%), pass | not measured (256p compile, see Known limitations) |
| TP=8 × CP=4 | 3.19% / 24.8% (bars 8.3% / 51.8%), pass | 26.5% / 44.7% (bars 46.9% / 79.2%), pass |

**Device text encoder.** On the gate prompt (34 tokens) and three 22-26-token prompts, the device embeddings are
1.47% (rel-L2) from a CPU fp32 encode, the same as CPU bf16 (1.47%; bar 2 × bf16 + 0.5% = 3.44%), cosine 1.000. The
256p 4-step gate run from the prompt text instead of fixed embeddings passes at TP=8 (step 0 3.17%, final video
27.5%) and at TP=8 × CP=2 (3.15%, 27.6%; bars 8.3% / 51.8%).

The absolute SSIM values are low for the CPU bf16 run as well: at four to eight distilled steps, bf16 rounding moves
the sampled trajectory to a different, equally plausible clip (same scene and motion, different fine texture). The
4-step device run is closer to fp32 than CPU bf16 is on every video metric. The 4-step fp32 reference reproduces the
earlier FastH3 port's independent fp32 reference to within 1e-5 on the device-vs-reference error (20.0% against both).

**VSA precision sensitivity (8-Step-V2).** On CPU in fp32, the plugin's sparse attention (the code the device runs)
matches the token-mask reference to 1.7e-5 at the first step. In bf16 it is 21.3% off, the same as the CPU-bf16
floor. VSA chooses, per head and per query tile, the top 20% of key tiles by pooled score, and bf16 rounding flips
some of those choices. The flipped fraction grows with depth: 0.1% of selected tiles in the first layer, about 10% in
the last. Computing the fine-stage scores in fp32 does not change this. **8-Step-V2 outputs therefore differ from
fp32 by about 20-50% in latent L2, while remaining visually coherent:** same scene, subject and motion.

**Video VAE:** the device tile-parallel decode is identical to the serial device decode (168 dB PSNR). It is
35.9 dB PSNR against a CPU fp32 decode of the same latents.

**Reproduce** (the reference and gate scripts are in `examples/minimax_h3/eval/`):

```bash
python examples/minimax_h3/eval/reference_cpu.py --model-path <checkpoint> --prompt-embeds embeds.pt \
  --height 256 --width 256 --num-frames 124 --seed 0 --dtype fp32 --out ref_fp32.pt
python examples/minimax_h3/eval/reference_cpu.py --model-path <checkpoint> --prompt-embeds embeds.pt \
  --height 256 --width 256 --num-frames 124 --seed 0 --dtype bf16 --out ref_bf16.pt
MINIMAX_H3_DUMP_LATENTS=device.pt python examples/minimax_h3/run.py --model-path <checkpoint> --tp 8 --cp 1 \
  --height 256 --width 256 --num-frames 124 --seed 0 --prompt-embeds embeds.pt --output clip.mp4
MINIMAX_H3_DUMP_LATENTS=device2.pt python examples/minimax_h3/run.py --model-path <checkpoint> --tp 8 --cp 1 \
  --height 256 --width 256 --num-frames 124 --seed 0 --prompt-embeds embeds.pt --output clip2.mp4
python examples/minimax_h3/eval/tier3_compare.py --model-path <checkpoint> --ref ref_fp32.pt --floor ref_bf16.pt \
  --run device.pt --run2 device2.pt
```

The full-size gate is the same with `--height 768 --width 1344 --tp 8 --cp 8`, plus
`MINIMAX_H3_RANK_CHECK=1 MINIMAX_H3_SAVE_VIDEO=video.pt MINIMAX_H3_SAVE_AUDIO=audio.pt` on the device run (each rank's
final latents through the shared all-rank agreement check; the device frames and waveform) and
`--video video.pt --audio audio.pt --decode-cache <dir>` on `tier3_compare.py`. At this size the CPU references
take about 15-25 minutes per forward each and one CPU fp32 video decode about 30-40 minutes;
`tier3_compare.py --decode-only <latents.pt> --decode-cache <dir>` runs the decodes as separate processes.

`embeds.pt` is a file with a `prompt_embeds` tensor of shape `(1, tokens, 5120)`, the Qwen3-VL `hidden_states[50]`
of the prompt.

Device tests in the three tiers are in `test/neuron/test_minimax_h3_accuracy_device.py`:

- component three-way (`assert_close_three_way`), for the DiT (dense and VSA) and the video VAE;
- single step;
- end to end, against an independent CPU reference.

They run on tiny random-weight checkpoints. `test_tier3_real_weights` applies the tier-3 checks above to saved
real-weight runs.

## Performance

trn2.48xlarge, bf16, 124 frames. Warm means a later request in the same process. Unless a row says otherwise, the
timings were measured on a quiet host (no other CPU-heavy work; the 1-minute load average was logged before every
timed request and was below 20 for each one), device work only, after one uncounted warm-up request, as the median
of 3 timed requests.

**Headline: 1344x768 on the whole box, recommended configuration** (FastH3 4-step, 1344x768x124, TP=8 × CP=8 on
64 cores, [`minimax_h3_stage.yaml`](../../examples/minimax_h3/minimax_h3_stage.yaml), prompt text through the
device text encoder, prompt cached):

| Stage | Seconds |
|---|---|
| DiT, 4 forwards (sequence-parallel TP) | 4 × 1.05 s = 4.20 s |
| video VAE decode on device (tiles dealt across all 64 ranks) + composition into one shared uint8 clip | 1.72 s |
| audio VAE decode on device (8 windows on the last 8 ranks, after their video tiles) | overlapped |
| decode stage, wall | 1.73 s |
| clip hand-off to the engine and the rest of the engine's request handling | 0.16 s |
| **whole request** (runs: 6.49 / 6.39 / 6.38 s; load average 12.0 / 16.2 / 8.4) | **6.39 s** |
| first request, new process, compiled graphs cached | 136 s |
| first request, empty compile cache (every graph compiles; earlier measurement on a busy host) | 693 s |
| HBM per core (peak, device text encoder) | 20.3 GiB |

The same request was run in a second process with `MINIMAX_H3_RANK_CHECK=1`: all 64 ranks ended every request (first
and 4 warm) with bit-identical final video and audio latents, and its clip is byte-identical to the timing process's
clip. Decoded frame by frame against the CPU fp32 reference of the accuracy gate above, the clip is at SSIM 0.668 mean /
0.618 worst frame (CPU bf16 floor 0.658; 124 of 124 frames within the gate's bar; through the H.264 mp4, so slightly
below the raw-pixel numbers of the gate). The rank check itself adds about 0.3 s per request (6.70 s median with it).

| 704x1280x124, same layout, prompt text through the device text encoder | Seconds |
|---|---|
| DiT, 4 forwards | 4 × 0.955 s = 3.82 s |
| decode stage, wall (video VAE 1.71 s, audio windows overlapped) | 1.71 s |
| **whole request** (runs: 5.93 / 5.92 / 5.94 s; load average 15.6 / 8.6 / 13.1) | **5.93 s** |
| first request, new process, compiled graphs cached | 132 s |

704x1280 is measured for speed only, not accuracy-checked against a CPU reference at this size. All 64 ranks hold
bit-identical final latents on every request, and the clip is byte-identical to an earlier run of the same request.

**First request.** It compiles whatever the compile cache does not already hold, and in every case it loads the
NEFFs on 64 ranks and builds the device text encoder. 1344x768, same prompt as the table above:

| Stage of the first request | Empty compile cache (earlier, busy host) | Graphs cached (new process) |
|---|---|---|
| text encoder (shard load + prompt-bucket graph) | 188 s | 36 s |
| DiT, first forward | 168 s | 52 s |
| video VAE decode | 22 s | 22 s |
| audio VAE decode (its graph compiles in about 5 minutes) | 312 s | 23 s |
| **whole first request** | **693 s** | **136 s** |

Stage init (model load, before the first request) adds about 55 s in both cases. The compile-cache key includes the
source locations of the traced code, so a different checkout path or an edit of a traced module recompiles those
graphs; an earlier run that recompiled only the audio decoder took 507 s.

With the TP all-reduce instead of sequence-parallel TP (`MINIMAX_H3_TP_SP=0`) the four forwards took 4.52 s against
4.24 s with SP in one earlier A/B job (the whole request with device audio was 6.98 s). The 256p accuracy gate passes at this layout with SP
(first-step video 4.0%, final video 24.9%; bars 8.3% / 51.8%), and so does the audio gate (85.2 dB from the CPU fp32
decode).

| Audio VAE placement, same 768p request and job | Audio decode | Decode stage, wall | Whole request |
|---|---|---|---|
| host CPU, last rank, 32 threads, overlapped with the video decode (`audio_vae: host`) | 2.62 s | 2.60 s | 7.88 s |
| **device windows** (default) | **0.10 s per window** | **1.66 s** | **6.98 s** |

(This A/B predates sequence-parallel TP.)

The first request compiles the audio decoder graph (about 5 minutes on an empty cache; later processes hit the
compile cache).
The device waveform is 90.9 dB SNR from a CPU fp32 decode of the same final latents (relative error 2.9e-5; the CPU
bf16 decode is 40.5 dB), well inside the parity bar.

The video comes out of the worker as uint8 pixels (`video_output: uint8`, the default), a quarter of the float
clip's bytes. The composed clip is already a file in `/dev/shm`, so the worker hands that file to the engine as
vLLM-Omni's shared-memory tensor handle instead of copying the 384 MB clip into a new segment: the time outside the
worker (mostly the clip transfer) drops from 0.44 s to 0.17 s at 768p, 6.72 s → 6.49 s per
request (`MINIMAX_H3_VIDEO_HANDOFF=0` keeps the copy).
The composed blend is the same linear map as diffusers' tile stitch and temporal cross-fade, computed from the same
uint8 tiles: within one 8-bit level on fewer than 0.1% of pixels in the unit test. Against the previous rank-0 blend on
the same 768p request, the two encoded clips are 36.1 dB PSNR apart (minimum 34.9 dB per frame), the level of the
H.264 encode noise itself.

**Smaller layouts, 384x640** (prompt embeddings precomputed; TP=8 throughout, more context-parallel ranks; measured
for speed only, not accuracy-checked at 384x640: these layouts pass the 256x256 gate above):

| Layout | Cores | 4-step DiT | 4-step decode stage | 4-step warm | 8-Step-V2 DiT | 8-Step-V2 warm |
|---|---|---|---|---|---|---|
| TP=8 | 8 | 4 × 1.34 s | 1.80 s | 7.24 s | 8 × 4.19 s | 35.40 s |
| TP=8 × CP=2 | 16 | 4 × 0.52 s | 1.01 s | 3.17 s | 8 × 2.15 s | 18.31 s |
| TP=8 × CP=4 | 32 | 4 × 0.31 s | 0.73 s | 2.06 s | does not compile | - |

Runs (load average before each): TP=8 4-step 7.22 / 7.24 / 7.24 s (9.2 / 9.5 / 9.7), 8-Step-V2 35.40 / 35.37 /
35.62 s (16.8 / 14.9 / 14.0); TP=8 × CP=2 4-step 3.16 / 3.17 / 3.17 s (15.7 / 16.1 / 16.1), 8-Step-V2 18.31 / 18.32
/ 18.29 s (17.4 / 17.5 / 11.4); TP=8 × CP=4 2.08 / 2.06 / 2.05 s (19.8 / 13.8 / 13.8). TP=8 × CP=8 was not measured at
384x640. The first request at TP=8 × CP=4 at 384x640 in a new process with the graphs cached took 72 s, 34 s of it
the DiT's first forward.

**Earlier 720p / 768p pipeline** (rank-0 gather and blend, float video output, audio on rank 0; same layout):

| Stage | 1344x768 (37.7k tokens) | 704x1280 (33.0k tokens) |
|---|---|---|
| text encoder | device, < 0.01 s (cached) | host, < 0.01 s (cached) |
| DiT, 4 forwards | 4 × 1.14 s = 4.56 s | 4 × 1.01 s = 4.03 s |
| video VAE decode on device (4 tiles per rank, one batched call) | 1.00 s | 0.99 s |
| video VAE tile gather to rank 0 (uint8, shared memory) | 1.03 s | 1.04 s |
| video VAE blend on rank 0 (host) | 1.68 s | 1.33 s |
| video VAE decode, total | 3.81 s | 3.49 s |
| audio VAE decode (host, overlapped with the video decode) | 1.90 s | 2.05 s |
| **whole request** (runs: 10.13 / 10.12 / 10.34 s; 9.06 / 9.27 / 9.03 s) | **10.13 s** | **9.06 s** |
| first request (cold compile cache) | 370 s | 216 s |
| HBM per core (peak) | 20.0 GiB | 14.1 GiB (host text encoder) |

The 720p run used the host text encoder: its device-encoder run failed in Neuron runtime initialization (a 4 MB
host allocation for the DMA rings, `ENOMEM`) while the previous 64-rank process was still releasing the devices; it is
not a device-memory limit (the device encoder fits at 768p with 20.0 GiB per core, and 720p is the smaller canvas).

The rest of the request (about 1.5 s at 768p) is host-side: inputs, scheduler steps and the transfer of the
124 × 768 × 1344 RGB clip out of the worker. The 256p accuracy gate passes at this layout (first-step video 3.43%,
final video 26.1%; bars 8.3% / 51.8%).

**Text encoder** (FastH3 4-step, TP=8, in the worker, per prompt of 22-26 tokens):

| Placement | New prompt | Cached prompt | First prompt in the process | HBM per core (256p run) |
|---|---|---|---|---|
| host CPU, 1 thread (the worker's default before this release) | 6.5-7.0 s | 0 s | 9.4 s | 13.2 GiB |
| host CPU, 32 threads (`MINIMAX_H3_TEXT_THREADS`, now the default for `host`) | ~0.7 s | 0 s | - | 13.2 GiB |
| **device, TP=8** (default) | **0.03 s** | 0 s | 34 s (shard load + bucket compile) | 18.9 GiB |

The device encoder adds about 5.7 GiB per core; TP=8 × CP=2 measured 18.95 GiB per core at 256p.

**Optimization history**, warm request, each step measured when it landed (on a host that was often busy, so the
absolute values are not comparable with the quiet-host tables above):

| Checkpoint | Step | Warm |
|---|---|---|
| 4-step | baseline | 35 s |
| 4-step | tile-parallel video decode (22.4 s → 7.9 s) | 27.3 s |
| 4-step | audio decode on 32 host threads (12.4 s → 1.2 s) | 14.3 s |
| 8-Step-V2 | baseline | 49.3 s |
| 8-Step-V2 | audio decode on 32 host threads (9.0 s → 1.1 s) | 41.4 s |
| 4-step | context parallelism, TP=8 × CP=4 on 32 cores | 7.8 s |
| 8-Step-V2 | context parallelism, TP=8 × CP=2 on 16 cores | 24.1 s |
| 4-step | audio VAE on device, time-folded activations, TP=8 × CP=4 on 32 cores (fixed embeddings) | 3.15 s → 2.26 s |
| 4-step | 1344x768, TP=8 × CP=8 on 64 cores: composed video VAE, then device audio VAE | 10.13 s → 7.14 s → 6.98 s |
| 4-step | sequence-parallel TP inside the CP slices, TP=4 × CP=2 on 8 cores, 256x448 (DiT, per forward) | 0.51 s → 0.41 s |
| 4-step | sequence-parallel TP, 1344x768, TP=8 × CP=8 on 64 cores (DiT, 4 forwards / whole request) | 4.52 → 4.24 s / 6.98 → 6.73 s |
| 4-step | composed clip handed to the engine as a shared-memory file, 1344x768 on 64 cores | 6.72 s → 6.49 s |

The audio thread count is `MINIMAX_H3_AUDIO_THREADS` (default 32); the decode restores the process's thread count
afterwards.

The first request at a new shape compiles its graphs. At TP=8 at 384x640 the whole first request with the graphs
compiling took 287 s for the 4-step checkpoint and 460 s for 8-Step-V2. Later runs hit the compile cache.

## Known limitations

- **t2va only.** FL2VA, Ref2VA, the base 50-step MiniMax-H3 and the LightX2V Turbo LoRAs are not wired in yet.
- **8-Step-V2 is slower per forward than dense.** The sparse attention is a masked full-score attention (exact, static
  shapes): 4.18 s per forward versus 1.34 s for dense. An additive tile-bias formulation of the same mask measured
  18.3 s per forward and is not shipped. A gather-based block-sparse kernel would be needed to profit from
  the sparsity; earlier Neuron ports measured compiler-generated gathers slower than dense.
- **8-Step-V2 hits a compiler internal error at some CP slices.** `neuronx-cc` fails with `NCC_ILIN901`
  (LowerIntrinsics) for the VSA graph at TP=8 × CP=4 at 384x640 and at TP=8 × CP=2 at 256x256. 8-Step-V2 compiles at
  TP=8 × CP=2 at 384x640 (24.1 s warm) and at TP=8 × CP=4 at 256x256; use `minimax_h3_stage_tp8cp2.yaml` for it at
  384x640. The 4-step checkpoint is not affected.
- **8-Step-V2 is bf16-sensitive** (see Accuracy Evaluation). Outputs are coherent, but not bit-close to fp32.
- **Audio VAE on device needs its own graph compile.** The weight-norm hooks are folded into plain weights before
  compile, and the alias-free SnakeBeta activations run with the time axis folded into rows: as written, the narrow
  late stages (8 channels × 46k samples per window) ran at 1/50 of the host's speed (3.8 s per window). The fold
  brings a window to 0.10 s. Replacing the replicate pads with a concat of the edge samples
  (`MINIMAX_H3_AUDIO_RPAD=1`) speeds up a single activation but makes the whole decoder graph 7.5× larger and 16×
  slower, so it is off.
- **Text encoder HBM.** The device text encoder takes about 6 GiB per core next to the DiT and the VAE decoder
  (18.9 GiB per core measured at 256p, TP=8). Its HBM at TP=8 alone at 384x640 has not been measured; if a layout runs
  out of device memory, set `text_encoder: host` (about 0.7 s per new prompt on 32 host threads, and about 50 GB of
  host RAM on the output rank, loaded on first use).
- **Host memory fragmentation at runtime start.** A 64-rank stage can fail to initialize with `Failed to allocate HOST
  memory (4194304 bytes)` for the DMA rings when the host has no free 4 MB contiguous pages (common after large
  compiles). `echo 1 > /proc/sys/vm/compact_memory` before the start fixes it.
- **Prompts longer than 512 tokens** compile one text-encoder graph per exact length.
- **One DiT graph per prompt token count.** The text tokens are part of the packed DiT sequence, so a prompt whose
  token count has not been seen before compiles a new DiT graph for that sequence length. At 1344x768 on 64 cores, a
  first request with a new 25-token prompt in a process warm on a 34-token prompt took 202 s (its text encode 0.69 s);
  later prompts with the same token count reuse the graph from the compile cache.
- **The 4-step checkpoint samples on the linspace grid** (first rung 1.0). Current FastVideo uses its trained ladder
  instead (first rung 0.999). Pass `model_config: {schedule: contract}` to match. Later exports, such as 8-Step-V2,
  always use their trained ladder.
- **Batch size one.**

## Tutorials

- [Quickstart: Offline video-and-audio generation with FastH3 on Trainium2](../getting-started/quickstart-offline-serving-minimax-h3.md)
- [Tutorial: Deploy FastH3 with vLLM Omni Neuron](../tutorials/tutorial-minimax-h3.md)
