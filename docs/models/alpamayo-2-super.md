# Alpamayo 2 Super Model Card

<!-- meta: description: Model card for NVIDIA Alpamayo 2 Super (34B) on AWS Trainium2 with the vLLM Omni Neuron
plugin: supported features, recommended configuration, accuracy on Neuron, performance and known issues. -->
<!-- meta: keywords: Alpamayo, Alpamayo 2 Super, autonomous driving, vision-language-action, VLA, chain of
causation, trajectory prediction, flow matching, Qwen3-VL, model card, vLLM, vLLM Omni, Neuron, trn2, BF16,
tensor parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-05 -->

## Introduction

[Alpamayo 2 Super](https://huggingface.co/nvidia/Alpamayo2-Super) is NVIDIA's 34B-parameter autonomous-driving
foundation model: a 32B Qwen3-VL-shaped vision-language backbone (64 decoder layers, hidden 5,120, 64 query / 8
KV heads) writes a Chain-of-Causation explanation of the driving decision from multi-camera frames and the ego
history, then a 2B flow-matching expert (64 layers, hidden 1,536) that attends to the backbone's KV cache denoises
64 future waypoints (6.4 s at 10 Hz) in 10 Euler steps. It has the same two-stage structure as
[Alpamayo 1.5](alpamayo-1-5.md) and runs on the same Neuron implementation (`diffusion/models/alpamayo/`); the
checkpoint's `config.json` selects the variant.

Alpamayo 2 Super is supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
using the Neuron SDK on AWS Trainium2 (`trn2`), tensor-parallel across eight NeuronCores (an adjacent chip pair).

**License:** see the model page; `nvidia/Alpamayo2-Super` is a gated Hugging Face repo. The checkpoint ships its
own tokenizer and processor files.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Alpamayo 2 Super (34B) | [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Chain-of-Causation reasoning text (greedy, up to 256 tokens, text EOS masked as upstream) | ✅ |
| | 64-waypoint trajectory (flow matching, 10 Euler steps) + unicycle decode to xyz / rotation | ✅ |
| | Sampled reasoning (upstream default top-p 0.98 / temperature 0.6) | - (greedy only) |
| | Multiple trajectory samples per request, navigation classifier-free guidance | - |
| | VQA, meta-action, auto-labeling and grounding text tasks | - |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ (TP=8) |
| | Context / CFG Parallelism | n/a |
| **Compilation** | torch.compile, static shapes (one prefill graph per prompt bucket, one decode graph, one expert graph) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Alpamayo 2 Super.
- `-`: not supported.

### Recommended configuration

Eight NeuronCores of an adjacent chip pair (cores `8d..8d+7`), `tensor_parallel_size: 8`
([`alpamayo2_super_stage_trn2.yaml`](../../examples/alpamayo/alpamayo2_super_stage_trn2.yaml)). The checkpoint is
35.8B parameters (71.6 GB in BF16). The vision tower, the 64-layer text decoder (8 query / 1 KV head per rank)
and the expert shard across the ranks; the LM head is vocab-parallel; the token embedding (155,776 x 5,120) and
the vision mergers are replicated. Weights per rank: about 10.7 GB at TP=8. Four cores would need about 19.4 GB of
the 24 GB per core for weights alone, before the KV cache, activations and graphs, so TP=4 is not offered. Each
rank reads only its own rows / columns of every sharded weight from the checkpoint, so host memory during load
is about 1/8 of the checkpoint per rank.

The default prompt bucket is 4,608 tokens (`ALPAMAYO_TEXT_BUCKETS`): six cameras x four frames of 16:9 video
is about 4,580 tokens (180 per frame). The KV cache is a fixed buffer of the largest bucket + 256 generated
tokens per layer. Six cameras is upstream's trajectory input profile (camera IDs `[0, 1, 2, 3, 5, 6]`); inputs
with all seven cameras would need a larger bucket and are not measured on device.

The request and result formats are the same as for Alpamayo 1.5: the processor outputs (`input_ids`,
`attention_mask`, `pixel_values`, `image_grid_thw`) and the ego history (`ego_history_xyz`, `ego_history_rot`,
16 waypoints) in `sampling_params.extra_args["robot_obs"]`; `pred_xyz` `[1, 64, 3]`, `pred_rot`, `actions`, the
generated token ids and the reasoning text `cot` in `multimodal_output["actions"]`.

## Accuracy Evaluation

**Benchmark / gate:** parity with upstream [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2)
(`Alpamayo2Super.sample_trajectories_from_data`) in FP32 on CPU with greedy decoding (`top_k=1`) and the same
flow-matching noise (seed 0), on the same inputs: six cameras x four frames through upstream's own
`helper.prepare_model_inputs` (synthetic frames, 16-waypoint ego history; 4,580-token prompt). Gates: CPU FP32
port vs upstream, every component within 1e-4 relative L2 and identical greedy tokens; device trajectory error
<= 2x the CPU BF16 error + 0.5%; every tensor-parallel rank returns the same output (SHA-256 of each rank's
trajectory, raw action, rotations and tokens on every request; `ALPAMAYO_RANK_DIGEST_DIR`).

| Metric | Trn2 (BF16, TP=8) | CPU BF16 (port) | CPU FP32 (port) | Gate |
|---|---|---|---|---|
| Reasoning text ("Maintain lane due to no obstruction ahead") | identical (10/10 tokens) | identical (10/10 tokens) | identical (10/10 tokens) | identical |
| Trajectory relative L2 vs upstream FP32 | 1.33e-3 | 3.8e-4 | 7.7e-8 | <= 5.8e-3 (device) |
| Mean / final waypoint error | 0.029 m / 0.082 m | — | ~0 | — |
| Text-layer hidden states / logits relative L2 | — | <= 0.11 / 2.9e-2 | <= 1.1e-5 / 1.6e-6 | <= 1e-4 (FP32) |
| Expert velocity per Euler step relative L2 | — | <= 8.9e-3 | <= 4.1e-7 | <= 1e-3 (FP32) |
| All 8 ranks agree (6 requests) | bit-identical | — | — | bit-identical |

The device column is job `1005-152805-A13-T-gate-t4-s6` (PR tree, an adjacent chip pair). The CPU FP32 column
shows the port reproduces upstream (all 64 text layers, vision + DeepStack features, every
Euler-step velocity). Upstream's FP32 model keeps the action encoder's Fourier frequencies in FP32 (Alpamayo 1.5
rounds them to BF16); the port follows the FP32 reference.

**Reproduce:**

```bash
# upstream FP32 greedy reference (environment with the alpamayo2_super package)
python examples/alpamayo/parity_ref_super.py --model <ckpt> --out parity_super_fp32.pt
# port on CPU, layer by layer (FP32 gate; --dtype bfloat16 for the CPU BF16 column)
VLLM_NEURON_CPU_MODE=1 python examples/alpamayo/parity_check.py --model <ckpt> --ref parity_super_fp32.pt
# port on Trn2, end to end (eight cores of an adjacent chip pair)
python examples/alpamayo/run.py --model-path <ckpt> --reference parity_super_fp32.pt \
  --stage-config examples/alpamayo/alpamayo2_super_stage_trn2.yaml --repeat 5
```

Tiny random-weight checkpoints built from upstream's own model class
(`test/unit/test_alpamayo_super_upstream_tiny.py`) and from the port's module tree
(`test/unit/test_alpamayo_super_tiny.py`) cover the weight names, the Super-specific rollout rules and TP=2 on CPU.

## Performance

| Configuration | Hardware | Warm request latency | First request |
|---|---|---|---|
| 24 images (6 cameras x 4 frames), 4,580-token prompt (bucket 4,608), 10 generated tokens, 10 Euler steps | trn2, 8 NeuronCores = adjacent chip pair (TP=8) | 1.91 s (median of 7) | ~390-450 s cold compile |

Per-stage breakdown (rank 0, median of 7 warm requests after one uncounted warm-up request, timed to device
completion, job `1006-134608-A13-T-quiet-timing-r2` on a quiet host: 1-minute load average 16 on 192 vCPUs before
every timed request; a busy host, load ~100, measured up to 27% slower): host prep 14 ms, vision
tower 194 ms, text prefill 962 ms, decode 38.1 ms per token (9 tokens after the prefill's first), expert 199 ms
(10 steps), engine and request transfer ~200 ms. Device memory: 13.1 GB per NeuronCore (weights ~10.7 GB, KV
cache, graphs). Repeated requests with the same seed return bit-identical trajectories.

Decode is weight-bandwidth-bound (each rank streams its 7.8 GB of decoder layers plus its 0.2 GB LM-head shard
per token) and pays 128 tensor-parallel all-reduces per token across the chip pair; the text prefill (4,608
tokens through 64 layers) is the largest stage. Context parallelism could only split the prefill and vision
tower; it is not used. An optimization pass that fed the attention matmuls bf16 operands instead of FP32
(scores and softmax still FP32) was measured and not adopted: 2.02 s vs 1.95 s within one A/B job, same accuracy.

## Known limitations

- Greedy reasoning only; upstream's default is nucleus sampling (top-p 0.98, temperature 0.6).
- One trajectory sample per request; batch size 1. The text-only tasks (VQA, meta-action, auto-labeling,
  grounding) and the two-GPU navigation-CFG demo are not served.
- Prompts up to 4,608 tokens with the default bucket (six cameras x four frames is ~4,580 tokens). Seven-camera
  inputs are not measured on device (no device run, no accuracy gate).
- All frames of one request must share a size (the vision graph is built for one patch grid).
- TP=8 on an adjacent chip pair only (TP must divide the 8 KV heads; TP=4 does not fit in HBM).
- Inputs are the checkpoint processor's outputs: build them with upstream's `helper.prepare_model_inputs`
  (see `examples/alpamayo/parity_ref_super.py`).

## Tutorials

- [Quickstart: Offline trajectory prediction with Alpamayo 2 Super](../getting-started/quickstart-offline-serving-alpamayo-2-super.md)
- [Alpamayo 1.5 model card](alpamayo-1-5.md) (same implementation, smaller checkpoint)
