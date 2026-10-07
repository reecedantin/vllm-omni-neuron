# π0 (pi0, base) Model Card

<!-- meta: description: Model card for LeRobot pi0 (base) Vision-Language-Action policy
on AWS Trainium with the vLLM Omni Neuron plugin -- flow-matching action head with a
continuous state token, BF16, single-core serving, LeRobot parity. -->
<!-- meta: keywords: pi0, PI0, VLA, vision-language-action, LeRobot, PaliGemma, Gemma,
flow matching, action expert, state projection, vLLM, vLLM Omni, Neuron, Trainium, trn2 -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[π0](https://www.physicalintelligence.company/download/pi0.pdf) is the original Physical
Intelligence Vision-Language-Action (VLA) policy: a PaliGemma (SigLIP vision + Gemma-2B
language model) prefix feeds a Gemma-300M action expert that predicts a continuous action
chunk via flow matching, same family as [π0.5 / π0.52](./pi05.md). π0 differs from π0.5 in
exactly three places: the robot's continuous **state** is projected (`state_proj`) into its
own suffix token rather than discretized into the language prompt; the flow-matching
**timestep** is concatenated onto the action embedding and fused through
`action_time_mlp_{in,out}` rather than driving an AdaRMS condition; and the action expert's
norms are **plain, unconditioned** `GemmaRMSNorm` (no AdaRMS). π0 has no hierarchical-language
step -- it is action-only.

The checkpoint is a [LeRobot](https://github.com/huggingface/lerobot) export
(`lerobot/pi0_base`); this port shares the same vendoring strategy as π0.5 --
`vllm_omni_neuron/diffusion/models/pi0/_vendor/pi0/`, byte-identical to upstream vLLM-Omni
apart from import paths and edits marked `# neuron:` (the same `variant_dims` / `vision_dims`
override hooks π0.5's vendored code carries, needed to build a shrunk M-tiny checkpoint --
upstream hardcodes the SigLIP projector output at 2048 regardless of the PaliGemma variant
width, which breaks once that width is shrunk below it).

π0 is supported for inference serving with vLLM Omni using the Neuron SDK on AWS Trainium2
(`trn2`).

The checkpoint derives from PaliGemma and is distributed under the
[Gemma license](https://ai.google.dev/gemma/terms); the PaliGemma tokenizer repo
(`google/paligemma-3b-pt-224`) is gated on Hugging Face.

**Compatible model checkpoints:**

| Model | Hardware | Quantization |
|-------|----------|---------------|
| lerobot/pi0_base | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Continuous action chunk (flow matching) | ✅ |
| **Quantization** | BF16 | ✅ |
| | FP8 | Not implemented |
| **Parallelism** | Tensor Parallelism (TP) | ✅ TP=2 supported, not recommended (see below) |
| **Compilation** | torch.compile | ✅ |

### Recommended configuration

The `pi0_base` checkpoint stores 3.50B parameters (SigLIP + projector 0.41B, Gemma-2B LM 1.98B,
token embedding / tied `lm_head` 0.53B, action expert 0.31B, plus the action expert's unused
`lm_head` 0.26B). It fits one NeuronCore's ~24 GB (LNC=2) in bf16, so TP=1 on a single core per
request is the default. TP=2 over the PaliGemma LM is supported
([`examples/pi0/pi0_stage_tp2.yaml`](../../examples/pi0/pi0_stage_tp2.yaml)) but is slower for π0
(served 92.9 vs 90.7 ms): the 48-token prompt leaves too little LM work to amortize the all-reduces.

## Quantization

BF16 only (fits one NeuronCore; quantization is not implemented). `PI0_VISION_DTYPE`
(`bf16` / `fp32`) overrides the vision tower dtype.

## Architecture notes

- **Shares `Pi0PrefixGraph` with π0.5's `Pi05PrefixGraph`** (identical bidirectional
  `[images, language]` layout -- `graphs_pi0.py` subclasses it directly, adding nothing).
- **`Pi0DenoiseGraph`** (`vllm_omni_neuron/diffusion/models/pi0/graphs_pi0.py`) differs from
  π0.5's denoise graph in exactly the three ways listed in Introduction: `state_proj` builds an
  extra suffix token (so the suffix is `[state, action_tokens x H]` with causal mask
  `[1, 1, 0, ..., 0]`, vs. π0.5's `[action_tokens x H]` / `[1, 0, ..., 0]`), the sinusoidal
  timestep embedding (float64 host math, matching the reference bit-for-bit) is concatenated
  onto the action embedding before `action_time_mlp_{in,out}` rather than driving an AdaRMS
  modulation, and every norm in the expert is the shared `_gemma_norm` helper (no conditioning
  path at all, simpler than π0.5's `_ada_norm`).
- The suffix attention mask is NOT fully bidirectional: with AR marks `[1, 1, 0, ...]` the
  state token attends only to the prefix and itself, while the action tokens attend to
  everything. An earlier fully-bidirectional suffix passed the tiny-checkpoint test under a
  loosened bound but was 66% off on the real weights; the test bound is back at 1e-5.
- `NeuronPi0ActionModel` (`model_pi0.py`) mirrors `NeuronPi05ActionModel`'s lifecycle
  (`load_checkpoint` / `to` / `compile` / `sample_actions`) exactly, with `state` as an explicit
  `sample_actions` argument instead of being folded into the language tokens.

## Accuracy Evaluation

Same three-tier methodology as π0.5 ([onboarding guide](../model-dev/onboarding-models.md)).

**Tier 1 (component, three-way):** `test/neuron/test_pi0_base_components_accuracy.py` --
`Pi0PrefixGraph` and `Pi0DenoiseGraph` against the vendored upstream CPU model via
`vllm_neuron.accuracy.testing.assert_close_three_way`, on the shrunk M-tiny checkpoint
(`test/unit/test_pi0_base_tiny.py`).

**Tier 2 (single-step and full-model parity):** `test/unit/test_pi0_base_tiny.py` proves the
CPU legs (fp32 graphs vs. upstream fp32, rel-L2 < 1e-5; bf16 vs. fp32) without a device;
`examples/pi0/device_check_base.py` runs the device leg on the real `lerobot/pi0_base` weights
and `examples/pi0/lerobot_parity_base.py` compares it with LeRobot's own `PI0Pytorch` (10 steps,
fixed seeded observation, 12-token prompt):

| Comparison | rel-L2 | cos | MSE |
|---|---|---|---|
| Our CPU fp32 graphs vs. upstream fp32 | 1.5e-6 | 1.0000 | 6.8e-14 |
| CPU bf16 graphs vs. fp32 (dtype error alone) | 0.54% | 0.99999 | 1.4e-6 |
| Device bf16 vs. CPU fp32 (job 1005-163628) | 0.58% | 0.99998 | 1.6e-6 |
| Device bf16 vs. LeRobot fp32 (2026-10-03) | 0.59% | 0.99998 | 1.7e-6 |
| Served pipeline on device vs. CPU fp32 pipeline (`pipeline_device_smoke_base.py`) | 0.78% | 0.99998 | 1.8e-6 |

The device error (0.58%) is within the bar: 2 × the pure-bf16 CPU error (0.54%) + 0.5% = 1.58%.

**Tier 3 (end-to-end):** the served pipeline (`NeuronPi0Pipeline.forward`, raw observation in)
on device against a CPU fp32 pipeline on the same observation and noise, last row above.

Bar (same as π0.5): action rel-L2 vs. CPU fp32 ≤ 2 × (CPU bf16 rel-L2 on the same input) + 0.5%,
which `device_check_base.py` computes per run. The comparisons of our CPU fp32 path with LeRobot are
fp32-vs-fp32 implementation checks, gated at action MSE ≤ 1e-3 and cos ≥ 0.999.

## Performance

Served through [`examples/pi0/run.py`](../../examples/pi0/run.py) `--profile --repeat 10`, bf16,
10 flow steps, warm median of 10 after one uncounted warm-up request, on a quiet host (1-minute
load average 13-17 before every timed request; job 1006-134101):

| Layout | Served request | Device forward | First request, warm cache |
|---|---|---|---|
| **TP=1, 1 core (default)** | **90.7 ms** | 86.5 ms | 15.9 s |
| TP=2, 2 cores | 92.9 ms | 89.5 ms | -- |

Both TP=2 ranks returned identical action chunks on every request.

Optimization levers (2026-10-04), one core, measured by
[`examples/pi0/latency_levers.py`](../../examples/pi0/latency_levers.py) on the same loaded pipeline:

| Stage | Before | After |
|---|---|---|
| Prefix (3 cameras + 48-token prompt) | 47.1 ms | 47.1 ms |
| Flow-matching loop, 10 steps | 43.6 ms (host Euler update) | 38.2 ms (per-step device graph) |
| `pipeline.forward()` | 91.6 ms | 87.9 ms |
| Served request | 94.3 ms | 88.5 ms |

The Euler update now stays on the device (`denoise_mode: device`). Unrolling all 10 steps into
one graph compiles but returns non-finite actions for π0 on Trn2, so it is not offered as a
default. First request (cold compile of both graphs): ~65 s; host weight load 36 s; peak host RSS 22 GB.

## Known limitations

- **No NKI kernel path / `model_config` fallback field yet** -- the prefix and denoise graphs
  run the SDPA-equivalent torch path from `graphs.py` under `torch.compile`, same as π0.5.
- **Batch size 1.**

## Tutorials

- [Quickstart: Offline action generation with pi0 / pi0.5 on Neuron](../getting-started/quickstart-offline-serving-pi0.md)
- [Tutorial: Deploy pi0 / pi0.5 / pi0.52 with vLLM Omni Neuron](../tutorials/tutorial-pi0.md)
