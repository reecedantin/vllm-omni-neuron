# InternVLA-A1.5 Model Card

<!-- meta: description: Model card for InternRobotics InternVLA-A1.5 on AWS Inferentia2 / Trainium with the
vLLM Omni Neuron plugin: supported features, recommended configuration, accuracy on Neuron, performance,
and known issues. -->
<!-- meta: keywords: InternVLA, InternVLA-A1.5, InternVLA-A1, model card, robot policy, vision-language-action,
VLA, flow matching, Qwen3.5, Gated DeltaNet, vLLM, vLLM Omni, Neuron, inf2, trn1, trn2, BF16 -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-03 -->

## Introduction

[InternVLA-A1.5](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) is a vision-language-action
(VLA) robot policy from InternRobotics. It attaches a lightweight unified action expert to a native
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) vision-language backbone (a hybrid of Gated-DeltaNet
linear-attention and gated full-attention layers, repeating 3 linear + 1 full), sharing full-attention
layers between the two towers so the action expert can attend to the VLM's key/value cache. Given camera
views, a language instruction and the robot's state, it predicts a 50-step action chunk via flow matching
(10 Euler steps). During training it is also supervised by a frozen WAN2.2-5B video model through
learnable "foresight" tokens that query future dynamics; that video branch is discarded at inference, so
serving needs no VAE or video decoder.

InternVLA-A1.5 is now supported for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Inferentia2
(`inf2`), Trainium1 (`trn1`) and Trainium2 (`trn2`) hardware. One NeuronCore is enough: the whole model is
~5.4 GB in BF16 (2.68B parameters).

**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| InternVLA-A1.5-base | [InternRobotics/InternVLA-A1.5-base](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) | Inf2, Trn1, Trn2 | BF16 |
| InternVLA-A1.5-RoboTwin / -Libero / -DOMINO | fine-tunes of -base, same architecture | Inf2, Trn1, Trn2 | BF16 |

> **InternVLA-A1** (the earlier 3B generation, checkpoint `type: qwena1`, a Qwen3-VL 28-layer backbone) is
> a *different* architecture and is **not** supported by this port — the reference implementation this
> plugin was built against (`InternRobotics/InternVLA-A-series`) only has code for A1.5. The A1 checkpoint
> on this host (`internvla-a1`) has no corresponding Neuron pipeline.

The Qwen3.5-2B VLM backbone's own `config.json` is not inside the InternVLA-A1.5 checkpoint (upstream
resolves it from the Hub); see [Known limitations](#known-limitations) for how this port resolves it offline.

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Action-chunk prediction (flow matching, 10 Euler steps) | ✅ |
| | Multiple camera views, variable count | ✅ |
| | Future-video foresight generation (training-only WAN2.2 branch) | n/a at inference |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | n/a (1 NeuronCore is enough; see [Performance](#performance)) |
| **Serving** | Offline standalone runner (`examples/internvla/run.py`) | ✅ |
| | Served through vLLM Omni (`examples/internvla/serve.py`) | ✅ |
| **Compilation** | torch.compile | ✅ |
| | N-block graph fusion (`BlockGraphRunner`) for the Gated-DeltaNet layers | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for InternVLA-A1.5.
- `n/a`: does not apply to this model / configuration.

### Recommended configuration

One NeuronCore, `tensor_parallel_size: 1` (see
[`internvla_stage.yaml`](../../examples/internvla/internvla_stage.yaml)). At 2.68B parameters / 5.4 GiB
BF16 there is no memory or latency reason to shard: the dominant cost (the 24-layer VLM prefix, see
Performance) is dispatch-bound on many small sequential ops, and tensor parallelism would add a
cross-core collective to each of those small ops rather than remove any of them.

## Accuracy Evaluation

**Benchmark / gate:** the three accuracy tiers of the
[onboarding guide](../model-dev/onboarding-models.md#step-4--validate-accuracy), in
[`test/neuron/test_internvla_a15_accuracy_device.py`](../../test/neuron/test_internvla_a15_accuracy_device.py).
Tiers 1 and 2 use `vllm_neuron.accuracy.testing.assert_close_three_way` (FP32 CPU baseline, BF16 CPU
expected, BF16 Neuron actual; pass when the Neuron error distribution matches the BF16 one, BC ≥ 0.99, or
is tighter, σ-ratio ≤ 1). Tier 3 is a golden-actions regression against upstream
`InternVLAA15.sample_actions` in FP32 on the CPU, same seeded noise and request, gated on cosine and
relative L2 within 2x the CPU BF16-vs-FP32 error (floors: cos ≥ 0.999, rel ≤ 2%).

Measured on trn2, InternVLA-A1.5-base, BF16, a 302-token request (3 x 224x224 images):

| Tier | Graph / scope | Rel. L2, Neuron | Rel. L2, CPU BF16 | σ-ratio | BC | Result |
|---|---|---|---|---|---|---|
| 1 | Vision tower | 1.88% | 1.47% | 1.28 | 0.995 | pass |
| 1 | 3 fused Gated-DeltaNet layers | 0.46% | 0.49% | 0.94 | 0.999 | pass |
| 1 | Gated full-attention layer (hidden / K) | 0.32% / 0.30% | 0.32% / 0.30% | 0.99 / 1.00 | 1.000 | pass |
| 1 | One action-expert denoise step | 0.047% | 0.048% | 0.98 | 0.986 | pass |
| 2 | Whole policy, first Euler step | 0.047% | 0.048% | 0.98 | 0.985 | pass |
| 3 | 10-step action chunk vs upstream FP32 golden | 0.520% (cos 0.999987) | 0.582% (cos 0.999985) | — | — | pass |

The Neuron error matches the BF16 rounding error at every tier, and the final action chunk is slightly
closer to upstream FP32 than the CPU BF16 run is. Tier 1 caught one real bug during the port: the flow-matching
state `x_t` and step `dt` were rounded to BF16 on the host-to-device copy (upstream keeps that Euler
accumulator in FP32), which gave the denoise step 3.7x the BF16 error (σ-ratio 3.73) while the end-to-end
chunk still sat inside the band. On a tiny random-weight checkpoint
([`test/unit/test_internvla_a15_tiny_helper.py`](../../test/unit/test_internvla_a15_tiny_helper.py)) the
FP32 port matches upstream to 1.6e-6 relative L2 (1 image) / 1.1e-6 (3 images), checked on the CPU by
[`test/unit/test_internvla_a15_cpu.py`](../../test/unit/test_internvla_a15_cpu.py) (7 tests, no device).

**Reproduce:**

```bash
# golden actions from upstream (CPU, FP32 + BF16); INTERNVLA_REF_SRC = the InternVLA-A-series src dir
python -m test.unit.test_internvla_a15_make_golden_helper --model /path/to/InternVLA-A1.5-base \
  --vlm-config /path/to/qwen3.5-config/config.json --out golden.pt
# the three tiers on a Neuron device
INTERNVLA_A15_WEIGHTS=/path/to/InternVLA-A1.5-base INTERNVLA_VLM_CONFIG=/path/to/qwen3.5-config/config.json \
INTERNVLA_A15_GOLDEN=golden.pt pytest -q test/neuron/test_internvla_a15_accuracy_device.py
pytest -q test/unit/test_internvla_a15_cpu.py   # CPU only
```

## Performance

| Configuration | Hardware | Prefix (once/request) | Denoise (10 steps) | Warm total | First request (cold compile) |
|---|---|---|---|---|---|
| 302-token prompt (3 x 224x224 images), 10-step action chunk | 1x trn2 NeuronCore | 406.5 ms | 170.0 ms | 579.0 ms | ~143 s served / ~830 s standalone (NEFF cache miss) |

Means of 10 warm requests on an otherwise quiet host. The prefix includes host-side work (image-token
placement and per-layer weight binding), so it is sensitive to host CPU load: with other compile jobs
running on the same instance (load average ~100) it measured 409 ms minimum / 454 ms mean over 20
requests, while the denoise loop stayed at 170 ms.

The 24-layer VLM prefix (18 Gated-DeltaNet + 6 gated full-attention layers, run once per request) is the
dominant cost — about 70% of warm latency — because the Gated-DeltaNet chunk rule is dispatch-bound (a
handful of small sequential device ops per layer), not FLOP-bound: 18 separate per-layer calls measured
99 ms versus the 6 full-attention layers' 6.3 ms for the same count. The denoise loop is real device time,
about 17 ms per Euler step (10 steps = 170 ms, the same whether or not the host synchronizes after each
step, and the 10 per-step time-embedding host-to-device copies add only 0.08 ms in total). An earlier
measurement of under 6 ms for the 10 steps timed unsynchronized asynchronous dispatches and was wrong:
queuing 200 such calls without a sync overflows the device execution queue, which only happens because
each call does real work. Two optimizations were tried:

- **Larger GDN chunk size (64 → 128 tokens, halving the chunk-rule's internal loop):** rejected. The
  algorithm is chunk-size-invariant (bit-identical result), but the larger fused chunk graph more than
  doubled compile time and did not finish within a 1806 s budget.
- **Fusing each contiguous run of Gated-DeltaNet layers into one compiled graph** (the checkpoint's layer
  pattern is `[linear, linear, linear, full] x 6`, so each run of 3 becomes one
  [`BlockGraphRunner`](../../vllm_omni_neuron/diffusion/layers/block_graphs.py) call instead of 3 separate
  device dispatches): **adopted**. Cut the 18-call GDN total from 99 ms to 56 ms, and the end-to-end warm
  total from 642.0 ms to 579.0 ms (−9.8%), with no change to compile time or numerical result (verified
  bit-identical on CPU before the device test).

int8 weight-only quantization and tensor parallelism were considered and not pursued: the model is not
memory-bound (5.4 GiB of 24 GiB per core) so int8 would not speed up a dispatch-bound workload, and TP
would add a cross-core collective to each of the many small per-layer ops rather than removing any.

## Known limitations

- **The Qwen3.5-2B VLM `config.json` is not inside the InternVLA-A1.5 checkpoint.** Upstream resolves it
  from the Hugging Face Hub (`Qwen/Qwen3.5-2B`). This port resolves it offline, in order: an explicit
  `vlm_config` path, `$INTERNVLA_VLM_CONFIG`, `<model>/vlm/config.json`, or the local HF cache of
  `Qwen/Qwen3.5-2B` / `-Base`. See [`config.py`](../../vllm_omni_neuron/diffusion/models/internvla/config.py).
- **vLLM Omni's `OmniDiffusionConfig.enrich_config` has no built-in `model_type` mapping for a third-party
  policy** (unlike its hardcoded GR00T case). Its generic fallback reads `config.json["architectures"]`
  (a single-element list) against the pipeline registry, so a served checkpoint needs that key added.
  [`stage_for_serving.py`](../../vllm_omni_neuron/diffusion/models/internvla/stage_for_serving.py) stages a
  symlinked, non-destructive copy with the key patched in, so the original (possibly shared, read-only)
  checkpoint directory is never modified.
- **The InternVLA-A1 (3B, `qwena1`) fallback checkpoint has no Neuron pipeline.** It is a different
  architecture (Qwen3-VL backbone, no Gated-DeltaNet) with no reference implementation available at port
  time.
- **No real tokenizer wired into the served request path by default.** `pipeline_internvla.py`'s
  `_robot_obs_to_batch` uses a byte-fallback placeholder tokenization unless `$INTERNVLA_TOKENIZER` points
  at a real Qwen3.5 tokenizer directory; the offline `run.py` path (used for all accuracy/performance
  numbers above) builds requests directly from token ids and bypasses this.
- Served one request at a time (`max_batch_size: 1`).

## Tutorials

- [Tutorial: Deploy InternVLA-A1.5 with vLLM Omni Neuron](../tutorials/tutorial-internvla-a15.md)
- [Quickstart: Offline action-chunk generation with InternVLA-A1.5](../getting-started/quickstart-offline-serving-internvla-a15.md)
