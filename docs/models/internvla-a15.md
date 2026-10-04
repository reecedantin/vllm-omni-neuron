# InternVLA-A1.5 Model Card

<!-- meta: description: Model card for InternRobotics InternVLA-A1.5 on AWS Trainium2 with the
vLLM Omni Neuron plugin: supported features, recommended configuration, accuracy on Neuron, performance,
and known issues. -->
<!-- meta: keywords: InternVLA, InternVLA-A1.5, InternVLA-A1, model card, robot policy, vision-language-action,
VLA, flow matching, Qwen3.5, Gated DeltaNet, vLLM, vLLM Omni, Neuron, trn2, BF16 -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

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
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK. It is measured on
Trainium2 (`trn2`); every number on this page is from one trn2 NeuronCore. One NeuronCore is enough: the
whole model is ~5.4 GB in BF16 (2.68B parameters).

**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial).

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| InternVLA-A1.5-base | [InternRobotics/InternVLA-A1.5-base](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) | Trn2 (measured) | BF16 |
| InternVLA-A1.5-RoboTwin / -Libero / -DOMINO | fine-tunes of -base, same architecture (not run) | Trn2 | BF16 |

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
| | Multiple camera views, variable count (1, 2 and 3 measured) | ✅ |
| | Upstream inference prompt (Qwen3.5 chat template, state as text, real tokenizer) | ✅ |
| | Future-video foresight generation (training-only WAN2.2 branch) | n/a at inference |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | n/a (1 NeuronCore is enough; see [Recommended configuration](#recommended-configuration)) |
| **Serving** | Offline standalone runner (`examples/internvla/run.py`) | ✅ |
| | Served through vLLM Omni (`examples/internvla/serve.py`) | ✅ |
| **Compilation** | torch.compile | ✅ |
| | N-block graph fusion (`BlockGraphRunner`) for the Gated-DeltaNet layers | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for InternVLA-A1.5.
- `n/a`: does not apply to this model / configuration.

### Recommended configuration

One trn2 NeuronCore, `tensor_parallel_size: 1` (see
[`internvla_stage.yaml`](../../examples/internvla/internvla_stage.yaml)), with the Qwen3.5 tokenizer
configured (`model_config.tokenizer` or `$INTERNVLA_TOKENIZER`). This is the layout every number below
was measured on. The headline request is 3 camera views at 256x256 (upstream's default `num_views=3`),
386-388 prompt tokens, padded to the 448-token bucket. At 2.68B parameters / 5.4 GiB BF16 there is no memory
reason to shard; the dominant cost is the 10-step denoise loop (about 17 ms of device time per step),
whose per-layer ops are small enough that a tensor-parallel collective would add to them. TP was not
measured.

## Accuracy Evaluation

**Benchmark / gate:** the three accuracy tiers of the
[onboarding guide](../model-dev/onboarding-models.md#step-4--validate-accuracy), in
[`test/neuron/test_internvla_a15_accuracy_device.py`](../../test/neuron/test_internvla_a15_accuracy_device.py).
Tiers 1 and 2 use `vllm_neuron.accuracy.testing.assert_close_three_way` (FP32 CPU baseline, BF16 CPU
expected, BF16 Neuron actual; pass when the Neuron error distribution matches the BF16 one, BC ≥ 0.99, or
is tighter, σ-ratio ≤ 1). Tier 3 is a golden-actions regression against upstream
`InternVLAA15.sample_actions` in FP32 on the CPU, same seeded noise and request, gated on cosine and
relative L2 within 2x the CPU BF16-vs-FP32 error (floors: cos ≥ 0.999, rel ≤ 2%).

**Request:** a real observation sent through the served path. LIBERO demo episode 0, step 20
(agent-view and wrist cameras, 256x256 RGB; the third view, when used, is the agent view 10 steps
earlier), the episode's instruction, the 8-value robot state z-scored with the dataset's statistics.
The pipeline turns it into the model's input exactly as upstream's inference transform does: Qwen3.5
chat template with the system message, one vision block per camera, then
`Task: ...; Control Mode: <joint>; State: <32 values in 256 bins>; Output: <Subtask, Action>`, tokenized
with the real Qwen3.5 tokenizer. A CPU test checks that request against upstream's own
`Qwen3VLProcessor` call (token ids identical; pixel patches within 1e-3 relative). The golden is
upstream FP32 on that same request. trn2, one NeuronCore, BF16, seed 0:

| Views (prompt tokens, bucket) | Tier 1: 5 graphs, σ-ratio | Tier 2: first Euler step | Tier 3: rel. L2 / cos vs upstream FP32 | Upstream BF16 rel. L2 | Result |
|---|---|---|---|---|---|
| 1 (254, 256) | 0.84-0.98, pass | pass | 1.124% / 0.999946 | 1.217% | pass |
| 2 (320, 384) | 0.91-1.00, pass | pass | 0.522% / 0.999987 | 1.191% | pass |
| **3 (386, 448)** | 0.90-1.00, pass | σ 1.012, BC 0.977: **fail** (L∞ 1.03x, L2 1.01x) | 1.039% / 0.999946 | 0.728% | tier 3 pass (1.43x the BF16 error) |

The served actions are bit-identical to the direct runner's and to the tier-3 device run for 1, 2 and 3
views, and repeated requests with the same seed are bit-identical. **All ranks agree:** the model serves at
TP=1, so each of the 4 NeuronCores of the lease ran its own vLLM Omni engine on the same 1-, 2- and
3-view requests; the actions are bit-identical on all 4 cores (SHA-256 digests compared with
`vllm_omni_neuron.testing.compare_rank_digest_files`) and equal to the direct runner's.

**Bar, and how far from it:** k = 2, the device error at most 2x the CPU BF16 error. A single seed is a
noisy measure of that ratio, because the BF16 error itself swings with the noise draw (0.6% to 2.7% on
these requests). Over 6 noise seeds, comparing against this port's FP32 and BF16 CPU runs (FP32 port
equals upstream FP32 to ~1e-6): the device error is **0.76x** (1 view), **0.61x** (2 views) and **0.89x**
(3 views) the CPU BF16 error, ratio of means; the worst single seed is 0.94x / 0.89x / 1.78x.

**Tier 2 over the same 6 seeds** (whole policy, first Euler step, strict three-way rule):

| Views | Pass | σ-ratio (pass seeds) | Misses |
|---|---|---|---|
| 1 | 5/6 | 0.96-0.99 | seed 2: σ 1.003, BC 0.982 (L2 1.00x, L∞ 0.83x the BF16 error) |
| 2 | 6/6 | 0.78-0.84 | none |
| 3 | 6/6 | 0.93-0.96 | none |

The rule fails only when σ-ratio is above 1 and BC below 0.99 together; every miss seen is at σ 1.00-1.01,
with the device error equal to the BF16 error. The 3-view seed-0 miss in the table above (σ 1.012) is
the same device output as the 6-seed run's pass (σ 0.951): the device result is deterministic, and what
moves is the CPU BF16 reference, whose rounding depends on the CPU thread count (default threads in the
test, 32 in the sweep). That is noise in the reference at the rule's boundary, not a device deviation.

**A real deviation, found and fixed.** The device error used to be 1.35x / 1.42x the CPU BF16 error (per
seed up to 2.02x), although every component on its own was at or below the BF16 error. Swapping the
three stages (vision + embedding, 24-layer prefix, 10-step denoise) between FP32 CPU, BF16 CPU and the
device over the same 6 seeds located it: the vision rope tables (cos/sin) were rounded to BF16 on the
host-to-device copy, while upstream rotates the image queries and keys in FP32 from FP32 tables. The
rounding is systematic, so it hardly changes the size of the image-token error (1.49% vs 1.52% for CPU
BF16) but costs about 1.5x at the actions: the same rounding on the CPU gives 1.54x / 1.63x the BF16
error, and the device with FP32 tables gives 0.61x / 0.89x. The tables now stay FP32
(`InternVLAA15Runner.FP32_TABLES`, with a CPU test). Text and suffix rope stay BF16, as upstream casts them.

On a tiny random-weight checkpoint
([`test/unit/test_internvla_a15_tiny_helper.py`](../../test/unit/test_internvla_a15_tiny_helper.py)) the
FP32 port matches upstream to 1.6e-6 relative L2 (1 image) / 1.1e-6 (3 images), checked on the CPU by
[`test/unit/test_internvla_a15_cpu.py`](../../test/unit/test_internvla_a15_cpu.py) (no device). During
the port, tier 1 also caught a real bug: the flow-matching state `x_t` and step `dt` were rounded to BF16 on
the host-to-device copy (upstream keeps that Euler accumulator in FP32), which gave the denoise step 3.7x
the BF16 error.

**Reproduce:**

```bash
# a served request: build it with pipeline_internvla._robot_obs_to_batch(robot_obs, cfg, tokenizer)
# and torch.save the dict; golden actions from upstream for it (CPU, FP32 + BF16);
# INTERNVLA_REF_SRC = the InternVLA-A-series src dir
python -m test.unit.test_internvla_a15_make_golden_helper --model /path/to/InternVLA-A1.5-base \
  --vlm-config /path/to/qwen3.5-config/config.json --batch request.pt --out golden.pt
# the three tiers on a Neuron device, on that request
INTERNVLA_A15_WEIGHTS=/path/to/InternVLA-A1.5-base INTERNVLA_VLM_CONFIG=/path/to/qwen3.5-config/config.json \
INTERNVLA_A15_REQUEST=request.pt INTERNVLA_A15_GOLDEN=golden.pt \
  pytest -q test/neuron/test_internvla_a15_accuracy_device.py
# CPU only; INTERNVLA_TOKENIZER enables the request-vs-upstream-processor test
INTERNVLA_TOKENIZER=/path/to/qwen3.5-tokenizer pytest -q test/unit/test_internvla_a15_cpu.py
```

## Performance

Real LIBERO observations replayed through both paths, **the same requests and the same view count in
each column pair**: every request is a new episode step (new pixels and a new state in the prompt, so the
per-prompt device table cache misses, as on a robot). 10-step action chunk, one trn2 NeuronCore, warm
median of 30 (served) / 34 (direct) requests. "Direct" is `InternVLAA15Runner.sample_actions` in-process
plus the same request building (`_robot_obs_to_batch`: chat template, tokenizer, image patches); "served"
adds the vLLM Omni engine round trip.

| Views (tokens → bucket) | Prefix | Denoise (10 steps) | Direct, end to end | **Served, warm** |
|---|---|---|---|---|
| 1 (254 → 256) | 50.4 ms | 168.9 ms | 223.2 ms | **228.3 ms** |
| 2 (320 → 384) | 72.6 ms | 169.4 ms | 246.9 ms | **251.8 ms** |
| **3 (386 → 448), recommended** | 85.7 ms | 171.0 ms | 262.3 ms | **266.9 ms** |
| 3 (386 → 512, without the 448 bucket, before the FP32 vision-rope fix) | 94.3 ms | 171.3 ms | 272.9 ms | 281.3 ms |

The first three rows were timed in one run on a quiet host (1-minute load average 13-19 before each
batch, no CPU reference work running). Building the request costs 2-3 ms on the host; the engine round
trip adds 4-5 ms. Engine start-up is 18-20 s. The first request after start-up takes 77-80 s with a warm
NEFF cache (tracing and loading the
graphs); compiling the graphs for a new prefix bucket from scratch took 500 s (the 448 bucket: vision +
embedding, the prefix layers and the denoise step at that length).

Earlier numbers for this port (554 → 252 ms direct, 230.7 ms served) came from different requests: the
direct runner used a synthetic 302-token request with 3 views, and the served path used 1 view with a
placeholder prompt that left out the chat template and the robot state. Upstream's real prompt carries the
32-value state as text, which is why a 3-view request is 386-388 tokens and needed its own bucket.

Optimization history, same synthetic 3-view request, direct runner:

| Stage | Original | Now |
|---|---|---|
| Prefix embedding lookup + image-token placement | 324.3 ms | 0 (inside the vision graph) |
| Prefix total | 381.9 ms | 73.3 ms |
| Denoise, 10 Euler steps | 169.7 ms | 171.0 ms (178-179 ms while K/V were passed as 12 inputs, see below) |

Where the old prefix went: the token-embedding lookup against the 250k-row table ran as an
uncompiled op on the device (324 ms), and the image tokens took a device-to-host round trip for a host
scatter. The vision graph now takes the host-looked-up token embeddings (image rows zeroed, cached per
prompt) and a one-hot placement matrix, and assembles the prefix embeddings itself
(`text_emb + place @ image_tokens`, exact in any dtype). Request tables (rope, biases, vision taps,
time embeddings, the embeddings and placement) stay on the device in an LRU keyed by the prompt and
image layout (`INTERNVLA_TABLE_CACHE`, default 16), and the full-attention layers' weights are bound once at
load. Because the state is part of the prompt, a robot's consecutive requests do not hit that cache; the
table rebuild is inside the numbers above. In the served path, cameras can be sent as raw bytes
(`{"data": bytes, "shape": [H, W, C]}`, `serve.py --camera-format raw`), and request preprocessing runs
with `INTERNVLA_HOST_THREADS` (default 8) torch threads instead of the worker's one.

**Prefix K/V go to the denoise graph stacked.** Passing the 6 layers' K and V as 12 separate graph inputs
instead of one stacked K and one stacked V made the same denoise graph 0.9 ms slower per step (10 steps:
179.3 ms vs 171.0 ms, same K/V, synchronized). Re-uploading the list tensors from the host did not change
it (179.5 ms), so it is the graph's input signature, not the layout of the tensors the prefix produced.
The two stacks cost about 2 ms once per request.

The denoise loop is device time, not dispatch: each step dispatches in 0.74 ms and completes in
about 17 ms. Measured and rejected:

- **Several Euler steps unrolled in one graph:** 2 steps per graph gave 198 ms for the loop (slower);
  5 steps per graph did not finish compiling within 40 minutes.
- **Computing the 50 learnable suffix tokens once per request** (they never attend to the action
  tokens and carry no time embedding, so their states are the same at every step) and running each
  step over the 50 action tokens only, continuing the Gated-DeltaNet recurrent state: exact on CPU and
  within the accuracy tiers on device, but the Neuron compiler rejects a carried state on a 50-token
  block (`NCC_IBIR297`), and padded to 64 tokens it compiles to 124 ms per step (1242 ms for the loop).
- **Larger GDN chunk size (64 → 128 tokens):** bit-identical, but compile time more than doubled and did
  not finish within 1806 s.

Adopted earlier: fusing each run of 3 Gated-DeltaNet layers into one
[`BlockGraphRunner`](../../vllm_omni_neuron/diffusion/layers/block_graphs.py) graph (prefix GDN 99 ms → 56 ms).
int8 weight-only quantization and tensor parallelism were not pursued: the model uses 5.4 of 24 GiB per
core, and the remaining cost is small per-layer device work that a collective would add to.

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
- **Served requests need the Qwen3.5 tokenizer** (`model_config.tokenizer` or `$INTERNVLA_TOKENIZER`;
  the checkpoint ships none). Without it requests are refused. Validated with the `Qwen/Qwen3.5-2B-Base`
  tokenizer files (same vocabulary and chat template as `Qwen/Qwen3.5-2B`, which upstream names).
- **The robot state must arrive normalised.** Upstream normalises it with the checkpoint's per-robot
  statistics before building the prompt; the pipeline does not, and writes the state into the prompt as
  sent. The base checkpoint's `stats.json` has no LIBERO entry, so the measurements above z-score the
  LIBERO state with the dataset's own statistics. The control mode defaults to `joint`
  (`robot_obs["control_mode"]` overrides it).
- **Tier 2's strict three-way rule sits at its boundary on this model**: misses at σ-ratio 1.00-1.01
  (3 views seed 0 with default CPU threads, 1 view seed 2), passes 17/18 seed x view cases at 32 threads;
  see [Accuracy Evaluation](#accuracy-evaluation).
- **Each prefix bucket compiles its own graphs** (about 500 s from scratch). The buckets are 256, 384,
  448, 512, 768 and 1024 tokens (`INTERNVLA_PREFIX_BUCKETS`); 512 and above were not timed except the
  3-view request at 512.
- Served one request at a time (`max_batch_size: 1`). Only trn2 was measured.

## Tutorials

- [Tutorial: Deploy InternVLA-A1.5 with vLLM Omni Neuron](../tutorials/tutorial-internvla-a15.md)
- [Quickstart: Offline action-chunk generation with InternVLA-A1.5](../getting-started/quickstart-offline-serving-internvla-a15.md)
