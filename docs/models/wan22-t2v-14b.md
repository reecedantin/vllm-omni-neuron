# Wan2.2-T2V-A14B Model Card

<!-- meta: description: Model card for Wan2.2-T2V-A14B on AWS Trainium with the
vLLM Omni Neuron plugin — supported features (resolutions, frame counts, BF16,
TP/CP/CFG parallelism, VAE tiling), FP8 quantization, the recommended 64-core
configuration, VBench accuracy on Neuron, and known issues. -->
<!-- meta: keywords: Wan2.2, Wan2.2-T2V-A14B, model card, text-to-video, video
generation, diffusion, MoE, vLLM, vLLM Omni, Neuron, Trainium, trn2, trn3, BF16,
FP8, fp8_row_mx, quantization,
tensor parallelism, context parallelism, Megatron sequence parallelism, CFG parallelism, VAE tiling, VBench -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-09-23 -->

## Introduction

[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) is a text-to-video diffusion model developed by Wan-AI. It uses a Mixture-of-Experts (MoE) architecture with two experts — a high-noise expert for overall layout during early denoising stages and a low-noise expert for detail refinement during later stages. The model has ~27B total parameters but only 14B active parameters per inference step, keeping computation and memory roughly equivalent to a single 14B dense model. It generates 5-second videos (81 frames at 16fps) at 480P and 720P resolutions with cinematic-level aesthetics and complex motion.

Wan2.2-T2V-A14B is now supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Wan2.2-T2V-A14B | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Trn2, Trn3 | BF16 |
| Wan2.2-T2V-A14B (FP8 DiT) | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | Trn3 | FP8 (`fp8_row_mx`) |

> FP8 loads a ComfyUI FP8-scaled DiT checkpoint. See [Quantization](#quantization) for setup.

## Features

Per-model feature availability for Wan2.2-T2V-A14B. See the [README](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/README.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-Video | ✅ |
| | 832x480 resolution (480p) | ✅ |
| | 1280x720 resolution (720p) | ✅ |
| | Up to 81 frames | ✅ |
| **Quantization** | BF16 | ✅ |
| | FP8 (DiT) | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP) | ✅ |
| | Megatron Sequence Parallelism (SP) | ✅ |
| | CFG Parallelism | ✅ |
| | [VAE Patch Parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/vae_parallel/) | ✅ |
| **Performance** | Classifier-Free Guidance | ✅ |
| | Spatial Tiling (VAE) | ✅ |
| | Temporal Chunking (VAE) | ✅ |
| | Continuous request batching | Limited |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Wan2.2-T2V-A14B
- Limited: accepted, but concurrent requests run serially rather than as a batched forward pass

### Recommended configuration

The recommended configuration runs on 64 NeuronCores (e.g., `trn3`), matching the [stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml): `tensor_parallel_size=4` × `ring_degree=8` (context parallelism) × `cfg_parallel_size=2` = 64.

On **Trn2** (trn2.48xlarge, 64 logical NeuronCores at LNC=2, ~24 GB HBM each), the recommended **720P** configuration is
[`wan22_stage_tp8cp4cfg2_720p.yaml`](../../examples/wan2_2/wan22_stage_tp8cp4cfg2_720p.yaml):
`tensor_parallel_size=8` × `ring_degree=4` × `cfg_parallel_size=2` = 64, with both experts resident (~7.3 GB of
weights per core). 1280x720x81, 40 steps: **157.8 s for a single warm request** (one request after compilation,
not a median of repeated runs).

The TP4 × CP8 × CFG2 720P config (`wan22_stage_tp4cp8cfg2_720p.yaml`) **does not fit Trn2 HBM with the current CP
self-attention default and is not supported on Trn2**. Each core holds 14.7 GB of weights (both experts at TP4),
1.5 GB of scratchpad, ~0.55 GB of code and runtime buffers, and 3.7 GB of DMA-ring spill per resident DiT graph;
the pipeline keeps two DiT graphs resident (the first step computes the text cross-attention K/V, later steps reuse
it), so loading the second graph needs ~24.4 GB against ~24 GB available, about 0.5 GB per core over. The full-sequence
K/V all-gather of the current default adds ~0.4 GB per layer of transient scratch at this shape, compared with
~0.1 GB for the ring kernel.

CFG parallelism (size=2) runs the conditional and unconditional denoising passes across separate replicas in parallel, reducing wall-clock time per diffusion step.

Context parallelism shards the sequence dimension (temporal-spatial latent tokens) across up to 8 ranks, enabling generation of longer videos within per-device HBM limits. The CP degree is configured with **`ring_degree`, not `sequence_parallel_size`**: vLLM Omni enforces `sequence_parallel_size = ulysses_degree * ring_degree`, so the stage config sets `ring_degree` and leaves `ulysses_degree` (default 1) and `sequence_parallel_size` unset — vLLM Omni then derives `sequence_parallel_size = 1 * ring_degree`. `ring_degree` sizes the sequence-parallel group; the CP self-attention all-gathers K/V across the group and runs flash attention with the true row maximum. The const-max ring-attention kernel (K/V stay sharded and are streamed around the CP group) is opt-in with `WAN22_CP_RING_ATTENTION=1`, because its static softmax bound can zero attention rows at the late, low-noise denoising steps. See the [context parallelism design doc](../design/context_parallelism.md#configuration) for details.

Megatron sequence parallelism (SP) is layered **within** the TP group (orthogonal to CP): the normalization and MLP regions that TP would otherwise replicate are instead sharded along the sequence dimension across the TP ranks, cutting activation memory at large resolutions. It is enabled with `model_config.tp_sequence_parallel: true` (see the 720P config [`wan22_stage_tp4cp8cfg2_720p.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml)) and requires `tensor_parallel_size > 1`. Leave it unset (the default) at 480P, where activations already fit.

## Quantization

The DiT can run in **FP8** to cut weight memory and speed up the projection GEMMs. The mode is
`fp8_row_mx`: it loads a [ComfyUI per-tensor FP8-scaled checkpoint](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) and runs the attention (QKV and output) **and** FFN projections in FP8.

Enable it either from `run.py`:

```bash
python examples/wan22/run.py \
  --quantization fp8_row_mx \
  --comfyui-fp8-model-path Comfy-Org/Wan_2.2_ComfyUI_Repackaged
```

or in the [stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml) under `engine_args.model_config`:

```yaml
    engine_args:
      model_config:
        quantization: fp8_row_mx
        comfyui_fp8_model_path: Comfy-Org/Wan_2.2_ComfyUI_Repackaged
```

`comfyui_fp8_model_path` points at the repo (or local root) holding the ComfyUI high-noise and
low-noise DiT checkpoints.

## Accuracy Evaluation

### Correctness on the current CP default

The CP self-attention default changed from the const-max ring-attention kernel to K/V all-gather + flash
attention with the true row maximum (see [Recommended configuration](#recommended-configuration)). Evidence
for the current default, per layout:

| Layout | Full-size run on the current default |
|--------|--------------------------------------|
| 480P, TP4 × CP8 × CFG2 (64 cores) | 81 frames, 40 steps: all 64 ranks produce the same final latent (one SHA-256), finite; frames visually clean |
| 720P, TP8 × CP4 × CFG2 (64 cores, Trn2 recommended) | 81 frames, 40 steps: all 64 ranks produce the same final latent (one SHA-256), finite; frames visually clean (no zeroed rows or blocks, smooth frame-to-frame change) |
| 720P, TP4 × CP8 × CFG2 (64 cores) | not supported on Trn2: does not fit HBM on the current default (see [Recommended configuration](#recommended-configuration)) |

There is **no full-size CPU reference for the 14B models**, so these runs show that every rank agrees and the
output is sound, not that it matches a reference. The evidence that the current default is the more accurate
one is indirect, from Wan2.2-TI2V-5B (same Wan CP attention code): a CPU emulation of the ring kernel reproduces
its device error and shows whole attention rows zeroed when the static softmax bound overshoots the true row
maximum at late, low-noise steps; with the true row maximum the late-step error falls to the bf16 level, and
the TI2V-5B teacher-forced step checks pass at 16 and 64 cores. On I2V-A14B 720P, the same seed on the two
paths gives a smooth difference (per-frame SSIM mean 0.857), consistent with that per-step effect accumulating.

**Benchmark:** [VBench](https://github.com/Vchitect/VBench) is a comprehensive benchmark suite for video generation models, evaluating across 16 dimensions including subject consistency, motion smoothness, temporal flickering, aesthetic quality, and imaging quality.

See the [VBench paper](https://arxiv.org/abs/2311.17982) for the benchmark methodology.

**Wan2.2-T2V-A14B-Diffusers (BF16)**: measured with the previous ring-attention CP default; not re-measured with the current default (see [Correctness on the current CP default](#correctness-on-the-current-cp-default)).

| Subtask and metric | Trn2 |
|--------------------|------|
| Subject Consistency | 87.60% |
| Background Consistency | 85.70% |
| Motion Smoothness | 96.50% |
| Dynamic Degree | 61.04% |
| Appearance Style | 27.11% |
| Scene | 47.82% |
| MLPerf-aligned 6-dimension average | 67.63% |

For externally published results, see the
[Wan2.2 VBench evaluation in *Compositional Video Generation via Inference-Time Guidance* (Table 5)](https://arxiv.org/pdf/2605.14988)
and the [VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard).
Results from different evaluation configurations are not directly comparable.

**Reproduce:** Serve the model following the [quickstart](../tutorials/tutorial-wan22-14b.md), then run VBench evaluation:

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench && pip install . && cd ..
python VBench/evaluate.py \
    --videos_path wan_output_dir \
    --dimension subject_consistency background_consistency motion_smoothness \
        dynamic_degree appearance_style scene \
    --mode=custom_input
```

## Known limitations

- **Batch size is limited to one.** The current Wan2.2 pipeline generates one video per request; concurrent requests run serially rather than as a batched forward pass.
- **FP8 accuracy validation is in progress.** The VBench numbers above are for BF16. FP8 (`fp8_row_mx`) is functionally supported, but its accuracy has not yet been validated against the BF16 reference.

## Tutorials

- [Tutorial: Deploy Wan2.2-A14B with vLLM Omni Neuron](../tutorials/tutorial-wan22-14b.md)
- [Optimizing High-Quality Offline Video Generation](../model-dev/optimizing-offline-video-generation.md)
