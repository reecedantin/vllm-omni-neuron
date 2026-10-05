# HunyuanVideo-1.5 Model Card

<!-- meta: description: Model card for HunyuanVideo-1.5 (480p text-to-video) on AWS Trainium2 with the
vLLM Omni Neuron plugin: supported features, the recommended 16-core TP=8 x CFG-parallel configuration,
three-tier accuracy against the diffusers CPU reference, performance, and known issues. -->
<!-- meta: keywords: HunyuanVideo-1.5, HunyuanVideo15Pipeline, model card, text-to-video, video generation,
diffusion, MMDiT, vLLM, vLLM Omni, Neuron, Trainium, trn2, BF16, tensor parallelism, CFG parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-04 -->

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

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-Video | ✅ |
| | 848x480 (480p), 25 frames | ✅ |
| | Longer clips (up to 121 frames) | Limited |
| | Image-to-Video, 720p | - |
| | In-video text (byT5 glyph prompts) | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Text encoder on NeuronCores (Qwen2.5-VL tower, TP=4) | ✅ |
| | Context Parallelism (CP) | - |
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
- Limited: accepted, but not yet measured end to end (compile time grows with the frame count).
- `-`: not supported.

### Recommended configuration

Four Trn2 chips (16 logical NeuronCores at LNC=2), matching the default stage config
`examples/hunyuanvideo15/hunyuanvideo15_stage.yaml`: `tensor_parallel_size=8` x `cfg_parallel_size=2`. Each
classifier-free-guidance branch runs on its own 8-core TP group (an adjacent chip pair); the two noise predictions
are exchanged over the host process group once per step and every rank applies the same combine and scheduler
step, so the result equals sequential CFG. Smaller layouts: `hunyuanvideo15_stage_tp4_cfg2.yaml` (8 cores) and,
on one chip, `hunyuanvideo15_stage_tp2_cfg2.yaml` (TP=2 x CFG=2) or `hunyuanvideo15_stage_tp4.yaml` (TP=4,
sequential CFG); see Performance.

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
| One CFG step at the serving layouts (TP2xCFG2 / TP4xCFG2 / TP8xCFG2 / TP8xCFG2 + device text tower) | 2.49% / 2.44% / 2.59% / 2.29% | 3.12% | pass (bar 6.75%) |
| VAE decode, 480x848x25f real latents: device vs CPU fp32 same tiles | 62.8 dB (worst frame) | - | pass (>= 35 dB) |
| VAE decode, same, device tiled vs CPU fp32 whole frame | 44.9 dB (worst frame) | CPU fp32 tiled: 44.9 dB | pass, no visible seams |

The CPU unit tests check the Neuron DiT against diffusers at TP=1/2/4 (rel-L2 ~1e-7, fp32), the full pipeline
against the diffusers pipeline (latents ~1e-6), and CFG-parallel against sequential CFG (exact).

**Reproduce:**

```bash
pytest test/unit/test_hunyuanvideo15_*.py -q                    # CPU, tiny random checkpoint, no weights
python test/neuron/test_hunyuanvideo15_accuracy.py              # needs trn2 + weights (HV15_ACC_WEIGHTS)
python test/neuron/test_hunyuanvideo15_device.py                # per-component device parity
```

## Performance

trn2.48xlarge, BF16, 848x480, 25 frames, 50 steps, CFG 6, warm NEFF cache; warm request = median of 3.

| Configuration | Cores | Warm request | Text encoders | DiT (50 steps) | VAE decode | First request | HBM per core |
|---|---|---|---|---|---|---|---|
| TP=8 x CFG=2, device text tower (default) | 16 | 60.4 s | 0.12 s (0 s cached) | 48.4 s (0.97 s per step) | 11.5 s | 90 s | 13.6 GiB |
| TP=8 x CFG=2, host text tower (`HV15_TEXT_ENCODER=host`) | 16 | 63.8 s | 3.3 s | 48.4 s | 11.6 s | 152 s | 10.8 GiB |
| TP=4 x CFG=2 (`_tp4_cfg2.yaml`), host text tower | 8 | 104.9 s | 2.8 s | 84.7 s (1.70 s per step) | 17.0 s | 240 s | - |
| TP=2 x CFG=2 (`_tp2_cfg2.yaml`), host text tower | 4 | 171.7 s | 3.6 s | 134.2 s (2.69 s per step) | 33.4 s | 203 s | - |
| TP=4, sequential CFG (`_tp4.yaml`), host text tower, host VAE | 4 | 448 s | 46 s | 169 s | 233 s | - | - |

Scaling the DiT from 4 to 16 cores is close to linear (2.69 -> 1.70 -> 0.97 s per step), and the VAE tiles (24 at
480p) spread over every rank. "First request" is the first request of a fresh engine and includes any graph
compile missing from the NEFF cache: 152 s at TP=8 with the 480p DiT graphs compiled cold; the device text tower
compiles once (684 s for that first request), then 90 s. The text tower adds 2.8 GiB of HBM per core (TP=4),
measured at TP=8 (13.6 GiB total of ~24 GiB).

## Known limitations

- VAE tile size is capped at 176 px on trn2: at 192 px the full-resolution 128-channel up block fails to compile
  (`NCC_IBTN020`, access-pattern step out of int16 range).
- The VAE trunk (mid-block attention) runs over all latent frames of a tile at once, so its graph grows with the
  clip length; 121-frame clips are not yet measured.
- Text-to-video at 480p only; image-to-video, 720p and the distilled / super-resolution (meanflow) checkpoints
  are not supported.
- One prompt per request; prompts are padded to the full 1000-token MLLM length (`HV15_TEXT_BUCKETS`).
- The device text tower needs a world size that is a multiple of 4 (one TP=4 group per chip); other worlds fall
  back to the host encoder automatically.
- Context parallelism is not yet part of the recommended configurations; layouts beyond 16 cores are under
  evaluation.

## Tutorials

- [Tutorial: Deploy HunyuanVideo-1.5 with vLLM Omni Neuron](../tutorials/tutorial-hunyuanvideo15.md)
- [Quickstart: Offline text-to-video with HunyuanVideo-1.5](../getting-started/quickstart-offline-serving-hunyuanvideo15.md)
