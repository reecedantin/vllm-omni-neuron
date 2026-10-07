# Alpamayo 1.5 Model Card

<!-- meta: description: Model card for NVIDIA Alpamayo 1.5 (10B) on AWS Trainium2 with the vLLM Omni Neuron
plugin: supported features, recommended configuration, accuracy on Neuron, performance and known issues. -->
<!-- meta: keywords: Alpamayo, Alpamayo 1.5, autonomous driving, vision-language-action, VLA, chain of causation,
trajectory prediction, flow matching, Cosmos-Reason2, Qwen3-VL, model card, vLLM, vLLM Omni, Neuron, trn2, BF16,
tensor parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-05 -->

## Introduction

[Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) is NVIDIA's 10B-parameter vision-language-action
model for autonomous driving. Given multi-camera video frames and the ego vehicle's recent trajectory, it first
writes a short natural-language Chain-of-Causation explanation (for example "Keep lane since the lane is clear
ahead") with an 8B vision-language backbone ([Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B),
architecturally Qwen3-VL-8B-Instruct), then a 2B flow-matching "expert" transformer that attends to the backbone's
KV cache denoises 64 future waypoints (6.4 s at 10 Hz) in 10 Euler steps. The waypoints are (acceleration,
curvature) controls integrated by a unicycle model into world-frame positions and rotations.

Alpamayo 1.5 is now supported for inference serving with [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
using the Neuron SDK on AWS Trainium2 (`trn2`), tensor-parallel across the four NeuronCores of one chip
(two- and eight-core layouts are provided as alternatives).

**License:** see the model page; `nvidia/Alpamayo-1.5-10B` is a gated Hugging Face repo. The backbone's tokenizer
and processor files are taken from the public `Qwen/Qwen3-VL-8B-Instruct` repo (byte-identical to
Cosmos-Reason2-8B's).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Alpamayo 1.5 (10B) | [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Chain-of-Causation reasoning text (greedy, up to 128 tokens, upstream's own cap) | ✅ |
| | 64-waypoint trajectory (flow matching, 10 Euler steps) + unicycle decode to xyz / rotation | ✅ |
| | Sampled reasoning (upstream default top-p 0.98 / temperature 0.6) | - (greedy only) |
| | Multiple trajectory samples per request (`num_traj_samples > 1`) | - |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ (TP=2, 4, 8) |
| | Context / CFG Parallelism | n/a |
| **Compilation** | torch.compile, static shapes (one prefill graph per prompt bucket, one decode graph, one expert graph) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Alpamayo 1.5.
- `-`: not supported.

### Recommended configuration

The four NeuronCores of one Trainium2 chip, `tensor_parallel_size: 4`
([`alpamayo_stage_trn2.yaml`](../../examples/alpamayo/alpamayo_stage_trn2.yaml)); choose four cores of the same
chip. Weights are 11.08B parameters (22 GB in BF16); attention heads and MLPs of the vision tower, VLM and expert
shard across the ranks, and the LM head is vocab-parallel (each rank holds 1/TP of the vocabulary rows, zero-padded
to 512-row shards, and the logits are all-gathered). The token embedding (155,697 x 4,096) is replicated. Weights
per rank: about 6.7 GB at TP=4. The KV cache is a fixed 3,200-slot buffer (largest prompt bucket 3,072 + 128
generated tokens) per layer.

Alternatives: [`alpamayo_stage_trn2_tp8.yaml`](../../examples/alpamayo/alpamayo_stage_trn2_tp8.yaml) (eight cores,
an adjacent chip pair) for the lowest latency, and
[`alpamayo_stage_trn2_tp2.yaml`](../../examples/alpamayo/alpamayo_stage_trn2_tp2.yaml) (two cores).

The served request carries the backbone processor's outputs (`input_ids`, `attention_mask`, `pixel_values`,
`image_grid_thw`) and the ego history (`ego_history_xyz`, `ego_history_rot`) in
`sampling_params.extra_args["robot_obs"]`; the result is returned in `multimodal_output["actions"]` as a dict
with `pred_xyz` `[1, 64, 3]`, `pred_rot` `[1, 64, 3, 3]`, the raw action `actions` `[1, 64, 2]`, the generated
token ids and the decoded reasoning text `cot`.

## Accuracy Evaluation

**Benchmark / gate:** parity with upstream [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) running in
FP32 on CPU with greedy decoding and the same flow-matching noise (seed 0), on the same inputs (4 cameras x 4
frames, 16-step ego history; 2,926-token prompt). Gates: CPU FP32 port vs upstream, every component within 1e-4
relative L2 and identical greedy tokens; device trajectory error <= 2x the CPU BF16 error + 0.5%; every
tensor-parallel rank returns the same output (SHA-256 of each rank's trajectory, raw action, rotations and tokens
on every request; `ALPAMAYO_RANK_DIGEST_DIR`).

| Metric | Trn2 (BF16, TP=4, recommended) | CPU BF16 (port) | CPU FP32 (port) | Gate |
|---|---|---|---|---|
| Reasoning text | identical | identical | identical (11/11 tokens) | identical |
| Trajectory relative L2 vs upstream FP32 | 4.5e-3 | 1.47e-3 | 2.3e-7 | <= 7.9e-3 (device) |
| Mean / final waypoint error | 0.67 m / 2.47 m | 0.25 m / 0.76 m | ~0 | — |
| Raw action (accel, curvature) relative L2 | 7.3e-3 | — | 1.8e-6 | — |
| Text-layer hidden states / logits relative L2 | — | — | <= 5.4e-6 / 1.4e-6 | <= 1e-4 |
| All 4 ranks agree (6 requests) | bit-identical | — | — | bit-identical |

The device column is job `1005-152805-A13-T-gate-t4-s6` (PR tree, cores of one chip). The CPU FP32 rows show
the port reproduces upstream exactly (all 36 text layers, vision + DeepStack features, every Euler-step
velocity); the device error is BF16 rounding, about 3x the CPU BF16 error at TP=4. It varies with the layout:
TP=2 measures 2.4e-3 and TP=8 2.1e-3 trajectory relative L2, with identical reasoning tokens (see the layout
sweep under Performance). All three are under the gate; why TP=4 rounds about twice as far as TP=2 and TP=8 was
not isolated. Every other layout in the sweep passed the same gate with all ranks compared (job
`1006-144312-A13-T-gate-rows`): every rank bit-identical on all 6 requests, served output equal to the rank
digests, greedy tokens identical to upstream; the replicated and vocab-parallel LM heads give bit-identical
trajectories at each TP size.
Determinism: repeated requests with the same seed return bit-identical trajectories.

**Reproduce:**

```bash
# upstream FP32 greedy reference (in an environment with the alpamayo1_5 package)
python examples/alpamayo/parity_ref.py --model <ckpt> --backbone-config <Qwen3-VL-8B-Instruct dir> \
  --inputs ref_bf16_cpu.pt --out parity_fp32.pt
# port on CPU, layer by layer
VLLM_NEURON_CPU_MODE=1 python examples/alpamayo/parity_check.py --model <ckpt> --ref parity_fp32.pt
# port on Trn2, end to end
ALPAMAYO_WEIGHTS=<ckpt> ALPAMAYO_PARITY_REF=parity_fp32.pt pytest test/neuron/test_alpamayo_device.py -q
```

## Performance

| Configuration | Hardware | Warm request latency | First request |
|---|---|---|---|
| 16 images (4 cameras x 4 frames), 2,926-token prompt, 10 generated tokens, 10 Euler steps | trn2, 4 NeuronCores = one chip (TP=4, recommended) | 0.76 s | ~210 s cold compile |
| Same | trn2, 8 NeuronCores = adjacent chip pair (TP=8) | 0.71 s | ~220 s cold compile |
| Same | trn2, 2 NeuronCores (TP=2) | 1.18 s | 55 s with a warm NEFF cache; ~300-400 s cold compile |
| 28 images (7 cameras x 4 frames), 5,052-token prompt (`ALPAMAYO_TEXT_BUCKETS=1024,2048,3072,4096,5120`) | trn2, TP=4 | 1.32 s | ~250 s cold compile |
| Same | trn2, TP=8 | 1.23 s | ~280 s cold compile |

**Layout sweep** (warm median of 5 requests after one uncounted warm-up request, per-stage times on rank 0 timed
to device completion, all rows from job `1006-134608-A13-T-quiet-timing-r2` on a quiet host: 1-minute load
average 3-20 on 192 vCPUs before every timed request; a busy host, load ~100, measured up to 23% slower;
accuracy = trajectory relative L2 vs upstream FP32 greedy on the same inputs and noise, gate 2x the CPU BF16
error + 0.5%, which is 7.9e-3 for the 4-camera input and 1.8e-2 for the 7-camera input; every row is
all-rank gated: TP=4 vocab-parallel 4-camera in job `1005-152805-A13-T-gate-t4-s6`, the others in job
`1006-144312-A13-T-gate-rows`):

| Layout | Input | Warm | Vision | Text prefill | Decode / token | Expert (10 steps) | Traj. rel-L2 | Tokens |
|---|---|---|---|---|---|---|---|---|
| TP=2 (previous default) | 4 cam | 1,180 ms | 195 ms | 444 ms | 34.1 ms | 132 ms | 2.4e-3 | identical |
| TP=2, vocab-parallel LM head | 4 cam | 1,175 ms | 202 ms | 446 ms | 33.0 ms | 133 ms | 2.4e-3 | identical |
| TP=4, replicated LM head | 4 cam | 815 ms | 140 ms | 269 ms | 20.6 ms | 111 ms | 4.5e-3 | identical |
| **TP=4, vocab-parallel LM head** | 4 cam | **759 ms** | 136 ms | 265 ms | 17.5 ms | 101 ms | 4.5e-3 | identical |
| TP=8, replicated LM head | 4 cam | 723 ms | 108 ms | 193 ms | 15.5 ms | 117 ms | 2.1e-3 | identical |
| TP=8, vocab-parallel LM head | 4 cam | 705 ms | 144 ms | 188 ms | 12.7 ms | 120 ms | 2.1e-3 | identical |
| TP=4, vocab-parallel | 7 cam | 1,320 ms | 252 ms | 601 ms | 19.6 ms | 115 ms | 1.0e-2 | identical |
| TP=8, vocab-parallel | 7 cam | 1,232 ms | 256 ms | 444 ms | 14.4 ms | 147 ms | 2.1e-3 | identical |

TP=4 on one chip is the recommended layout: 36% faster than TP=2 at the same accuracy bar. TP=8 is a further 7%
for twice the cores (its all-reduces cross the chip-to-chip link, and the expert step gets slower than at TP=4).
Decode is weight-bandwidth-bound: per token each rank reads its share of the 13.9 GB of decoder layers plus its
LM-head rows (8.2 GB per rank at TP=2, 3.8 GB at TP=4, 1.9 GB at TP=8), which is why it scales with TP and why
the vocab-parallel LM head helps most at TP=4 and TP=8. TP=16 would need the 8 KV heads replicated and pays more
cross-chip all-reduces per token than the bytes it saves; TP=32/64 do not divide the heads. Context parallelism
could only split the prefill and vision tower (decode and the expert need TP) and is not used: at TP=8 the text
prefill is 27% of the request, so CP=2 could save at most ~95 ms before collectives. There is no classifier-free
guidance branch to parallelize.

At TP=2, an earlier optimization pass built the decode step's RoPE angles, one-hot cache-write mask and causal
mask inside the decode graph (from a two-element position tensor) instead of on the host, cutting the warm request
from 1.31 s to 1.21 s (decode 39.5 -> 34.8 ms per token). A tighter 2,944-token prompt bucket and an FP32 LM head
were measured and gave no significant change.

## Known limitations

- Greedy reasoning only; upstream's default is nucleus sampling (top-p 0.98, temperature 0.6), so served
  reasoning text matches upstream's greedy output, not a particular sampled one.
- One trajectory sample per request; batch size 1.
- Prompts up to 3,072 tokens with the default buckets (4 cameras x 4 frames is 2,926 tokens). All seven
  cameras x 4 frames (5,052 tokens) is measured with `ALPAMAYO_TEXT_BUCKETS=1024,2048,3072,4096,5120`; the largest
  bucket sets the KV-cache length for every request, so add the 5,120 bucket only when such inputs are served.
  Longer prompts are untested (each bucket is one more prefill graph to compile).
- All frames of one request must share a size (the vision graph is built for one patch grid).
- Tensor-parallel layouts: TP must divide the 8 KV heads (TP=2, 4 or 8); TP=4 must use the four cores of one
  chip and TP=8 an adjacent chip pair.
- Inputs are the backbone processor's outputs: images must be preprocessed with the Qwen3-VL processor and
  upstream's message template (see `examples/alpamayo/reference_1_5.py`) before the request.
- Alpamayo 2 Super is not covered by this card.

## Tutorials

- [Tutorial: Deploy Alpamayo 1.5 with vLLM Omni Neuron](../tutorials/tutorial-alpamayo-1-5.md)
- [Quickstart: Offline trajectory prediction with Alpamayo 1.5](../getting-started/quickstart-offline-serving-alpamayo-1-5.md)
