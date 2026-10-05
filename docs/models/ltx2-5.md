# LTX-2.5 Model Card

<!-- meta: description: Model card for Lightricks LTX-2.5 (distilled text-to-video with synchronized audio) on AWS
Trainium2 with the vLLM Omni Neuron plugin: supported features, recommended configuration, accuracy on Neuron,
performance, and known issues. -->
<!-- meta: keywords: LTX-2.5, LTX-2, model card, text-to-video, text-to-audio-video, diffusion, DiT, vLLM,
vLLM Omni, Neuron, trn2, NeuronCore-v3, BF16, tensor parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) is Lightricks' joint audio-video diffusion
transformer. One 19B-parameter DiT denoises video latents and 48 kHz stereo audio latents together, with
audio-to-video and video-to-audio cross-attention in every block. A Gemma text encoder and per-modality connectors
condition it; a causal-convolution VAE decodes the video, and an audio VAE plus a bandwidth-extension vocoder decode
the audio. The distilled checkpoint generates in 8 steps without classifier-free guidance.

LTX-2.5 is now supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
using the Neuron SDK on Trn2 hardware (distilled text-to-video with audio).

**License:** [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md). The Hugging Face
repository is gated; accept the license on the model page before downloading.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| LTX-2.5 distilled (`transformer/`) | [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-video with synchronized 48 kHz stereo audio | ✅ |
| | 512x768, 121 frames, 8 steps (distilled) | ✅ |
| | Image-to-video, two-stage (latent upsample + refine), Full/SFT 30-step | - |
| **Guidance** | Positive-only (the distilled default) | ✅ |
| | CFG / spatio-temporal guidance / modality-isolation guidance | - |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ (TP=4 on one chip; TP=4 or 8 combined with CP below) |
| | Context Parallelism (CP) over the video tokens | ✅ (TP=4 x CP=4 on 16 cores, TP=8 x CP=4 on 32, TP=8 x CP=8 on 64) |
| | CFG Parallelism | n/a (guidance-distilled) |
| | VAE decode on device (fixed-shape spatial tiles, one compiled graph, tiles spread over all ranks) | ✅ |
| | Vocoder as time spans on several ranks' host CPUs, overlapped with the video decode | ✅ |
| **Serving** | Prompt cache (repeated prompts skip the text encoder and the text connectors) | ✅ |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for LTX-2.5.
- Limited: accepted, with the caveat noted.
- `-`: not supported.

**Other layouts (experimental):** TP=8 alone (8 cores), TP=4 x CP=2 (8 cores) and TP=8 x CP=2 (16 cores) load and
run, and each passed only the single-step parity check (1 step at 192x256x25 against the CPU reference: video
5.6% / 5.5% / 5.9% rel-L2, bar 9.1%; jobs 1004-064242 and 1004-073254, 2026-10-04). They were not checked at full
size, not checked on every rank, and not re-timed on the current code; use the supported layouts above.

### Recommended configuration

**Four chips** at LNC=2 (16 logical NeuronCores, one full row of the torus: cores 0-15, 16-31, 32-47 or 48-63),
`examples/ltx2/ltx2_stage_tp4cp4.yaml`: TP=4 inside each chip x context parallelism (CP) 4 over the video tokens.
6.41 s per 512x768x121 clip warm, 7.68 s with a new prompt; this is the layout of the full-size accuracy gate
below. Alternatives, each gated at full size on every rank too: `examples/ltx2/ltx2_stage_tp8cp4.yaml` (TP=8 x
CP=4, 32 cores: 6.56 s, no faster than 16 cores at 512x768; the layout for large canvases, 1024x1536x121 in
17.7 s) and `examples/ltx2/ltx2_stage.yaml` (TP=4, one chip, 10.85 s).

- **Transformer:** 4.4 GB (TP=8) / 8.7 GB (TP=4) of bf16 weights per core. The 48 blocks run as 12 calls of one
  compiled 4-block graph (`blocks_per_graph: 4`); timestep (AdaLN) conditioning is computed on the host in fp32
  per step, the RoPE tables once per request. A single core does not fit the transformer.
- **Context parallelism:** each CP rank holds `1 / CP` of the video tokens; the audio tokens and the text stay
  replicated, and the K/V of the video self-attention and of the video-to-audio attention are all-gathered over
  the CP group. Set `ring_degree` (the CP degree) so that it divides the video token count
  `((frames - 1) / 8 + 1) x (height / 32) x (width / 32)`.
- **Text encoder:** the Gemma text tower runs on the NeuronCores, tensor-parallel over rank 0's TP group (one
  compiled six-layer graph per prompt-length bucket, 5.1 GB per core at TP=4); the text connectors run on the host.
  Both run once per new prompt; the conditioning is cached and broadcast to the other ranks.
  `LTX25_TEXT_ENCODER_DEVICE=0` runs the text encoder on the host instead.
- **Video VAE:** decoded on device as fixed-shape spatial tiles (16x8 latents, 4 latents of overlap, diffusers'
  linear blend) through one compiled graph, the tiles spread over all ranks and merged on rank 0. The full
  512x768 frame graph exceeds the compiler's instruction limit.
- **Audio:** the audio VAE and the vocoder run on the host while the video VAE decodes; the vocoder runs as four
  time spans (48 mel frames of context each side) on four ranks' host CPUs.

The stage is multi-process: select devices with `NEURON_VISIBLE_DEVICES` and list them in `devices:` as a comma
list.

## Accuracy Evaluation

**Gate:** three-way parity against the diffusers CPU pipeline with the reference transformer (the same weights,
prompt and seed): CPU fp32 (reference), CPU bf16 (dtype-only error band) and Neuron bf16. Pass:
`device rel-L2 <= 2 x (CPU bf16 rel-L2) + 0.5%`. The VAE is checked separately against the untiled CPU decode of
the same latent.

| Tier | What | Trn2 (TP=4) | Reference / threshold |
|---|---|---|---|
| 1. Component | Transformer, one call (random-weight structure model, 256 video tokens) | rel-L2 0.61% video / 0.57% audio, cos 0.99998 | CPU bf16: 0.68% / 0.62% |
| 1. Component | Text encoder (Gemma text tower, TP=4), 49 stacked hidden states, two prompts | rel-L2 0.43% / 0.41% | CPU bf16 0.43% / 0.34%; bar 1.4% / 1.2% |
| 1. Component | VAE decode, device tiles vs untiled CPU (512x768x121) | PSNR 47.4 dB mean, 45.2 min, seam columns 45.7 | > 35 dB |
| 2. Single step | Pipeline, 1 denoising step, latents (192x256x25), device text encoder | rel-L2 5.2% video (cos 0.9986) / 1.6% audio | CPU bf16 4.3% / 1.6%; bar 9.1% / 3.7% |
| 3. End to end | Pipeline, 4 steps, latents (192x256x25) | rel-L2 24.1% video (cos 0.971) / 4.5% audio | CPU bf16 22.8% / 3.2%; bar 46.2% / 6.9% |
| 3. End to end | **Full size, TP=4 x CP=4 (16 cores)**: 512x768x121, 8 steps, decoded video + audio, served pipeline class (device text encoder, device VAE tiles, host vocoder) | every one of 121 frames inside its per-frame bar; clip rel-L2 14.3% (PSNR 25.5 dB mean / 24.8 min, SSIM 0.81 / 0.78); waveform 15.6%; latents 29.1% video / 2.4% audio; all 16 ranks' latents bit-identical | CPU bf16: clip 13.4% (PSNR 26.0 / 25.4, SSIM 0.83 / 0.80), waveform 15.8%, latents 27.3% / 2.1%; bar 2 x band + 0.5% per frame and per output |
| 3. End to end | **Full size, TP=8 x CP=8 (64 cores)**: same gate | every one of 121 frames inside its per-frame bar; clip rel-L2 14.7% (PSNR 25.3 dB mean / 24.4 min, SSIM 0.80 / 0.76); waveform 14.4%; latents 30.4% video / 2.1% audio; all 64 ranks' latents bit-identical; served output bit-identical to the gate run. Against the TP=4 x CP=4 output: per-frame SSIM 0.86 mean / 0.84 min, declining smoothly with frame index (0.91 -> 0.85) | same bars as the TP=4 x CP=4 row |
| 3. End to end | **Full size, TP=8 x CP=4 (32 cores)**: same gate | 121 / 121 frames inside the bar; clip 14.9% (PSNR 25.1 / 24.4 dB, SSIM 0.80 / 0.76); waveform 11.9%; latents 30.4% / 2.2%; all 32 ranks bit-identical; served output bit-identical to the gate run | same bars as the TP=4 x CP=4 row |
| 3. End to end | **Full size, TP=4 (one chip, 4 cores)**: same gate | 121 / 121 frames inside the bar; clip 14.5% (PSNR 25.4 / 24.7 dB, SSIM 0.81 / 0.78); waveform 19.0%; latents 29.1% / 2.5%; all 4 ranks bit-identical; served output bit-identical to the gate run | same bars as the TP=4 x CP=4 row |
| 3. End to end | **1024x1536x121, TP=8 x CP=4 (32 cores)**, teacher-forced steps (no CPU pipeline run: one fp32 call of the 19B transformer at 24576 + 126 tokens takes ~10 min on the host): the device transformer's inputs at denoise steps 0, 3 and 7 of the served request, replayed through the reference transformer in fp32 and bf16 | every step inside its bar: video rel-L2 3.0% / 8.8% / 3.8% at steps 0 / 3 / 7 (CPU bf16 2.1% / 6.1% / 3.5%), audio 1.2% / 1.5% / 0.8% (CPU bf16 1.2% / 1.5% / 0.8%); all 32 ranks' latents bit-identical; served output bit-identical; frames checked by eye (fox, snow, sunset light, no tile seams) | bar 2 x (CPU bf16 per-call rel-L2) + 0.5% per step and modality |
| 3. End to end | Served 512x768x121 video + audio (the Omni stage, TP=4 x CP=4) | frames and waveform bit-identical to the full-size gate run; frames checked by eye (fox, snow, sunset light, no tile seams) | same bars |

The reference transformer is diffusers' `LTX2VideoTransformer3DModel` at
`huggingface/diffusers@8b33bfc`, vendored unmodified in `vllm_omni_neuron/diffusion/models/ltx2/_vendor/`
(the installed diffusers predates LTX-2.5's transformer options). Run in fp32 on the CPU, the Neuron transformer
code reproduces it to rel-L2 1.5e-7 per call and 0.02% over a full 4-step pipeline.

**Reproduce:**

```bash
pytest test/unit/test_ltx2_*.py -q                                          # CPU, no weights
python -m test.neuron.test_ltx2_pipeline_parity_device --model <dir> --mode reference --out parity --steps 4
torchrun --nproc_per_node 4 -m test.neuron.test_ltx2_pipeline_parity_device --model <dir> --mode device \
    --out parity --steps 4
LTX2_VAE_DUMP_DIR=vae python examples/ltx2/run.py --height 512 --width 768 --num-frames 121 --steps 8
python -m test.neuron.test_ltx2_vae_parity_device --dump vae --model <dir>
python -m test.neuron.test_ltx2_text_encoder_device --model <dir> --mode reference --out text
torchrun --nproc_per_node 4 -m test.neuron.test_ltx2_text_encoder_device --model <dir> --mode device --out text
python -m test.neuron.test_ltx2_fullsize_gate_device --model <dir> --mode reference --out gate   # ~55 min CPU
torchrun --nproc_per_node 16 -m test.neuron.test_ltx2_fullsize_gate_device --model <dir> --mode device \
    --out gate --cp 4
```

The full-size gate (a torchrun process per rank) and the served stage (the Omni engine's workers) produce
bit-identical frames and waveform for the same prompt and seed. This depends on the host stages running with
fixed torch thread counts: the CPU kernels' results depend on the thread count (the text connectors differ by
0.2-0.3% rel-L2 between 1 and 4 threads, the timestep embeddings between 1 and 24), and the distilled denoise
amplifies such a difference to ~9% on the decoded video, still inside the bf16 band. The DiT's host
conditioning runs with `LTX2_HOST_MATH_THREADS` (default 8) threads, the connectors, audio VAE and vocoder with
exactly `LTX25_HOST_THREADS` (default 24); `LTX25_HOST_THREADS_ADAPTIVE=1` caps the latter at the idle cores,
faster on a loaded host but then the output depends on the host load.

## Performance

Distilled text-to-video with audio, 8 steps, trn2.48xlarge at LNC=2, device text encoder, default host thread
counts (`LTX25_HOST_THREADS=24`, `LTX2_HOST_MATH_THREADS=8`). Every row was timed on 2026-10-06 in a device-only
job with no CPU reference work running: one uncounted warm-up request, then the median of three warm requests
with the same prompt (the prompt cache skips the text encoder) and the median of three requests with new prompts.
The host's 1-minute load average was below 20 before every timed request. Every layout in this table passes the
full-size gate on every rank (see the accuracy section); 768x1152x121 is measured for speed only.

| Layout | Cores | Shape | Warm latency | New prompt | First request (warm cache) | HBM per core (peak) | Job |
|---|---|---|---|---|---|---|---|
| TP=4 | 4 | 512x768x121 | 10.85 s | 12.64 s | 50.0 s | 18.1 GB | 1006-144356-A7-Q-tp4-quiet |
| **TP=4 x CP=4 (recommended)** | 16 | 512x768x121 | **6.41 s** | **7.68 s** | 46.8 s | 17.2 GB | 1006-144356-A7-Q-tp4cp4-quiet |
| TP=8 x CP=4 | 32 | 512x768x121 | 6.56 s | 8.26 s | 40.5 s | 10.2 GB | 1006-144356-A7-Q-32c-quiet |
| TP=8 x CP=8 | 64 | 512x768x121 | 11.14 s | 12.28 s | 45.1 s | 10.1 GB | 1006-144356-A7-Q-64c-quiet |
| TP=4 x CP=4 (speed only) | 16 | 768x1152x121 | 10.68 s | - | - | 12.7 GB | 1006-144356-A7-Q-32c-quiet |
| TP=8 x CP=4 | 32 | 1024x1536x121 | 17.66 s | 16.86 s | 73.7 s | 11.4 GB | 1006-144356-A7-Q-32c-quiet |

Individual requests (warm; new prompt), with the 1-minute load before each warm request: TP=4 10.85 / 10.51 /
10.90 s (load 11.5-13.3; new 12.78 / 11.82 / 12.64 s); TP=4 x CP=4 6.61 / 6.37 / 6.41 s (load 19.4-19.6; new
7.88 / 7.68 / 7.54 s); TP=8 x CP=4 6.67 / 6.56 / 6.32 s (load 14.9-18.3; new 8.80 / 8.26 / 7.43 s); 1024x1536x121
22.78 / 17.66 / 16.58 s (load 16.1-19.3; new 18.32 / 16.83 / 16.86 s); 768x1152x121 10.85 / 10.39 / 10.68 s (load
13.7-15.4); TP=8 x CP=8 10.38 / 11.14 / 11.47 s (load 9.5-16.4; new 13.11 / 12.28 / 12.19 s). At 64 ranks the
host load rises to 50-80 during each request (64 worker processes running their host stages) and falls back
before the next. Engine start with a warm compile cache takes 60-120 s.

The end-to-end (tier 3, 4 steps) gate was also run at TP=4 (video 21.2% / audio 3.3%) and TP=4 x CP=4
(25.5% / 4.1%), bar 46.2% / 6.9%.

### 2026-10-04 layout sweep (single-step parity only)

The rows below are the layout sweep that shaped the current design. Each passed only the single-step (tier 2)
parity check (1 step at 192x256x25 against the CPU reference). None was checked at full size or on every rank,
and none was re-timed on the current code or on a quiet host. Rows marked "host text encoder" ran the text encoder
on the host, where it costs 4-11 s per new prompt (on device: 0.15 s plus ~1.3 s of host connectors). End-to-end
latencies of the full-size-gated layouts are in the table above and are left out here.

| Layout | Cores | Shape | Warm latency | Transformer (8 steps) | New prompt | HBM per core (peak) | Tier 2 video / audio rel-L2 (bar 9.1% / 3.7%) |
|---|---|---|---|---|---|---|---|
| TP=4 (before this sweep) | 4 | 512x768x121 | 31.7 s | 9.3 s | 53.8 s | 13.0 GB | 5.4% / 1.7% |
| TP=4, host text encoder | 4 | 512x768x121 | 12.4 s | 9.2 s | 19.9 s | 13.0 GB | 5.4% / 2.2% |
| TP=4, device text encoder | 4 | 512x768x121 | - | 7.7-7.8 s | - | 18.1 GB | 5.9% / 1.4% |
| TP=8 (experimental) | 8 | 512x768x121 | 11.5 s | 7.2-8.1 s | 16.8 s | 8.4 GB | 5.6% / 1.8% |
| TP=4 x CP=2 (experimental) | 8 | 512x768x121 | 10.2 s | 6.1-6.6 s | - | 12.3 GB | 5.5% / 2.4% |
| TP=4 x CP=4, device text encoder | 16 | 512x768x121 | - | 3.8-4.1 s | - | 17.2 GB | 6.1% / 2.0% |
| TP=4 x CP=4, host text encoder | 16 | 512x768x121 | 7.9 s | 4.8-6.2 s | 19.6 s | 12.1 GB | 6.0% / 2.3% |
| TP=4 x CP=4, ring attention | 16 | 512x768x121 | 10.0 s | 5.6-6.2 s | - | 12.0 GB | - |
| TP=8 x CP=2 (experimental) | 16 | 512x768x121 | 8.6 s | 5.4-5.8 s | - | 7.9 GB | 5.9% / 1.5% |
| TP=8 x CP=4, host text encoder | 32 | 512x768x121 | 7.85 s | 5.2-5.6 s | 12.2 s | 7.6 GB | 5.5% / 1.6% |
| TP=8 x CP=4, device text encoder | 32 | 512x768x121 | - | 4.0-4.2 s | - | 10.2 GB | 5.8% / 1.9% |
| TP=4 x CP=4 | 16 | 768x1152x121 | - | 9.1-9.5 s | - | 12.7 GB | - |
| TP=8 x CP=4, device text encoder | 32 | 1024x1536x121 | - | 9.3-9.6 s | - | 11.4 GB | - |

Where the time went at TP=4 (31.7 s -> 12.4 s): the text connectors (3.2B parameters, ~14 s on the worker's single
host thread) are cached with the prompt, the RoPE tables and text embeddings are uploaded once per request instead
of once per step, the VAE tiles run on all ranks (7 s -> 1.2-1.7 s), and the audio decode (audio VAE + vocoder,
host) runs while the video VAE decodes, the vocoder as four time spans on four ranks. Tokens per transformer pass:
6144 video + 126 audio at 512x768x121, 13824 + 126 at 768x1152x121, 24576 + 126 at 1024x1536x121; under CP each
rank holds `video / CP` tokens (1536 at TP=4 x CP=4) and the full audio and text sequence. Past 16 cores the
512x768 transformer stops speeding up (~5.2 s for 8 steps, 14 graph calls per step) while the per-rank work keeps
shrinking, so the larger canvases are where more cores pay off. Ring attention (the plugin's
`collective_permute` kernel) was no faster than the K/V all-gather at these sequence lengths and stays off by
default (`LTX2_CP_RING=1` enables it).

TP=4 x CP=4 is the recommendation because the 32-core TP=8 x CP=4 layout is no faster at 512x768x121
(6.56 s vs 6.41 s) and TP=8 x CP=8 on 64 cores is slower (11.14 s): it is correct at full size on every rank, but at this canvas the transformer does not get
faster past 16 cores (see above), so the extra ranks only add per-request overhead.

The first request at a new shape compiles the transformer graphs and one VAE tile graph per frame count (~20 min
cold at 512x768x121 on 4 cores). Host stages (text connectors, audio VAE, vocoder) slow down when the host CPU is
oversubscribed by other work.

## Known limitations

- **The distilled video denoise is bf16-sensitive at the model level.** Over 4 deterministic Euler steps the
  reference transformer in bf16 on the CPU already differs from fp32 by ~23% rel-L2 on the video latent (audio:
  ~3%), because a rounding difference at step 1 shifts the sample the later steps start from. The Neuron error stays
  inside that band; single-call parity is tight. Pixel-level comparisons below the VAE are not a parity signal.
- Only distilled text-to-video with positive-only guidance. CFG, STG and modality-isolation guidance, image-to-video,
  the two-stage pipeline and the Full/SFT 30-step schedule are not supported yet.
- The default 8-step schedule is the scheduler's own sigma schedule, not the official distilled sigma table.
- The VAE tile graph is specific to the latent frame count: each distinct `--num-frames` compiles its own graph
  (~17 min cold at 121 frames).
- The vocoder runs on the host CPU (as time spans on several ranks): as one graph it exceeds the per-core
  instruction limit.
- Context parallelism splits the video tokens only: the CP degree must divide
  `((frames - 1) / 8 + 1) x (height / 32) x (width / 32)`.
- Resolution: 1024x1536x121 is the largest shape measured (TP=8 x CP=4, 11.4 GB per core peak); its accuracy is
  checked per transformer call (teacher-forced steps), not against a full CPU pipeline run. 768x1152x121 is
  measured for speed only. The limits past 1024x1536 are
  untested; the VAE tile graph is fixed-size, so the VAE is not one, and per-core HBM leaves room at TP=8.
- LTX-2.5's native diffusion decoder (DiffVAE) needs a NATTEN kernel from the Hub; this port uses the convolutional
  VAE.

## Tutorials

- [Tutorial: Deploy LTX-2.5 with vLLM Omni Neuron](../tutorials/tutorial-ltx2-5.md)
- [Quickstart: Offline text-to-video with LTX-2.5](../getting-started/quickstart-offline-serving-ltx2-5.md)
