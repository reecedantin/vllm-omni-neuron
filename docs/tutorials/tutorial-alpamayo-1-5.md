# Tutorial: Deploy Alpamayo 1.5 with vLLM Omni Neuron

<!-- meta: description: How NVIDIA Alpamayo 1.5 runs on AWS Trainium2 with the vLLM Omni Neuron plugin: the two
inference stages, static-shape graphs, tensor parallelism, request format and how to validate accuracy. -->
<!-- meta: keywords: Alpamayo 1.5, vLLM Omni, Neuron, trn2, autonomous driving, VLA, tutorial -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: tutorial -->

This tutorial walks through how the Neuron port of [Alpamayo 1.5](../models/alpamayo-1-5.md) is organized, how
a request flows through it, and how to check its accuracy against upstream. For the shortest path to a first
result, see the [quickstart](../getting-started/quickstart-offline-serving-alpamayo-1-5.md).

## 1. What runs where

A request runs two stages on the same KV cache:

1. **Reasoning.** The Qwen3-VL vision tower encodes the camera frames; the 36-layer text decoder prefills the
   prompt (images, ego-history trajectory tokens, instruction) and then decodes greedily until one token after
   `<|traj_future_start|>`, writing the Chain-of-Causation text.
2. **Trajectory.** The expert transformer runs 10 flow-matching Euler steps. Each step embeds the current noisy
   (acceleration, curvature) sequence for 64 waypoints, attends non-causally over the VLM's cached keys/values
   up to the end of the reasoning plus its own 64 tokens, and predicts a velocity. The final action is integrated
   by the unicycle model into world-frame waypoints on the host.

On the NeuronCores there are four compiled graphs, all with static shapes:

| Graph | Contents | Shape key |
|---|---|---|
| `alpamayo_vision` | patch embed, 27 ViT blocks, merger + 3 DeepStack mergers | number of image patches |
| `alpamayo_prefill` | 36 text layers over the right-padded prompt, LM head at the last real token | prompt bucket |
| `alpamayo_decode` | embedding lookup, 36 layers against the full fixed-length KV cache, LM head | one graph total |
| `alpamayo_expert` | action projection, 36 expert layers, output projection | one graph total |

The KV cache is a fixed `[1, kv_heads, 3200, 128]` buffer per layer. Each decode step reads all of it with an
additive mask (finite `-30000`) for slots not yet filled and writes its own K/V through a one-hot mask, so the
decode graph never changes shape. Masked attention is written as `softmax(q @ k^T * scale + bias) @ v`; plain SDPA
is only used where there is no mask. The host does the argmax, the stop test and the FP32 Euler update.

## 2. Tensor parallelism

The default stage config sets `tensor_parallel_size: 4` (one chip); TP=2 and TP=8 configs are provided too, and TP
must divide the 8 KV heads. Q/K/V/gate/up projections use `ColumnParallelLinear`, output and down projections
`RowParallelLinear`, and the TP group is registered with `register_replica_groups` so the collectives lower at
compile time. Three details are specific to this model:

- The vision tower's fused `qkv` weight is reordered at load time into per-rank `[q_r | k_r | v_r]` blocks before
  the column split, so each rank holds complete heads.
- The vision tower's biased row-parallel layers keep their bias in FP32 (required under TP) and cast the reduced
  output back to the activation dtype, so the residual stream stays BF16.
- The LM head is vocab-parallel: the 155,697-row weight is zero-padded to a multiple of `512 x TP` rows, each rank
  keeps one shard, and the per-rank logits (padding masked to -inf) are all-gathered inside the prefill and decode
  graphs. Decode reads every LM-head row per token, so this removes 1.1-1.2 GB of per-token weight traffic per
  rank at TP=4/8. A shard size that is not a multiple of 512 rows failed in the runtime's collective at TP=4.

## 3. Request and response

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<Alpamayo-1.5-10B dir>", stage_configs_path="examples/alpamayo/alpamayo_stage_trn2.yaml")
robot_obs = {  # numpy arrays from the Qwen3-VL processor + ego history
    "input_ids": ..., "attention_mask": ..., "pixel_values": ..., "image_grid_thw": ...,
    "ego_history_xyz": ...,  # [1, 1, 16, 3]
    "ego_history_rot": ...,  # [1, 1, 16, 3, 3]
}
out = omni.generate({"prompt": ""}, OmniDiffusionSamplingParams(seed=0, extra_args={"robot_obs": robot_obs}))
result = out[0].multimodal_output["actions"]
print(result["cot"][0], result["pred_xyz"].shape)  # reasoning text, (1, 64, 3)
```

The history trajectory is tokenized into the prompt's `<|traj_history|>` placeholders inside the pipeline, as
upstream does. `seed` selects the flow-matching noise; the same seed gives the same trajectory.

## 4. Validate accuracy

Three tiers, cheapest first:

1. **CPU unit tests** (`test/unit/test_alpamayo_tiny.py`, no weights): a random-weight checkpoint with the real
   tensor names checks weight mapping, that the static decode equals a full recompute, prompt-bucket padding,
   TP=2 vs TP=1 (gloo), the served output contract, and that every graph traces without recompiling.
2. **Port vs upstream, layer by layer** (`examples/alpamayo/parity_ref.py` in the upstream environment, then
   `examples/alpamayo/parity_check.py` or `test/unit/test_alpamayo_upstream_tiny.py` with
   `ALPAMAYO_PARITY_MODEL` / `ALPAMAYO_PARITY_REF`): vision, every text layer, logits, every Euler-step velocity
   and the trajectory, FP32.
3. **End to end on Trainium** (`test/neuron/test_alpamayo_device.py`): a served request vs the upstream FP32
   reference — trajectory relative L2 and identical reasoning tokens.

## 5. Tune

- `ALPAMAYO_TEXT_BUCKETS` controls the prompt buckets (each is one prefill graph; the largest also sizes the KV
  cache).
- `ALPAMAYO_PROFILE=1` adds a vision-tower timing to the per-request `timing_ms` breakdown.
- `ALPAMAYO_DECODE_IN_GRAPH=0` falls back to building the decode RoPE/masks on the host (slower; for A/B).
