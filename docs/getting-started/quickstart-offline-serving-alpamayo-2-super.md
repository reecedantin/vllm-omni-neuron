# Quickstart: Offline trajectory prediction with Alpamayo 2 Super on Neuron

<!-- meta: description: Predict an autonomous-driving trajectory and its reasoning text offline with NVIDIA
Alpamayo 2 Super (34B) on AWS Trainium2 using the vLLM Omni Neuron plugin. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Alpamayo, Alpamayo 2 Super, autonomous driving, VLA,
trajectory, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-05 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to run one Alpamayo 2 Super request offline on a Trainium2 instance with the vLLM
Omni Neuron plugin. When you finish, you have a 64-waypoint trajectory and the model's Chain-of-Causation
reasoning text produced by [Alpamayo 2 Super](../models/alpamayo-2-super.md) on eight NeuronCores (an adjacent
pair of Trainium2 chips).

## Prerequisites

- One `trn2.48xlarge` instance (the model uses eight NeuronCores of two adjacent chips).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- The `nvidia/Alpamayo2-Super` checkpoint (gated; request access, then `hf auth login`), downloaded locally. It
  ships its own tokenizer and processor files.
- A separate Python 3.12 environment with the upstream [alpamayo2_super](https://github.com/NVlabs/alpamayo2)
  package, used only to build the processor inputs (Step 1).

> **Note:** `examples/alpamayo/run.py` uses the vLLM Omni entrypoint (`Omni`). Pass
> `--stage-config examples/alpamayo/alpamayo2_super_stage_trn2.yaml` (TP=8, `devices: "0,1,2,3,4,5,6,7"`). With
> multi-process TP, select cores with `NEURON_VISIBLE_DEVICES` (eight core ids of an adjacent chip pair,
> `8d..8d+7`) and leave `NEURON_RT_VISIBLE_CORES` unset.

## Step 1: Build the request inputs

The model consumes the checkpoint processor's outputs for upstream's trajectory prompt. In the upstream
environment:

```bash
python examples/alpamayo/parity_ref_super.py --model <Alpamayo2-Super dir> --out alpamayo2_inputs.pt
```

This runs upstream's own `helper.prepare_model_inputs` on a six-camera x four-frame sample and saves the tokenized
prompt, pixel values and ego history, plus upstream's FP32 greedy result for comparison (the full FP32 34B model
on CPU needs about 150 GB of host memory and several minutes; pass `--layers none` to skip the per-layer dump).
The camera frames are synthetic; replace `synthetic_sample` with real frames (for example from
`alpamayo2_super.load_physical_aiavdataset`) for a meaningful prediction.

## Step 2: Run the request on Trainium

```bash
export NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python examples/alpamayo/run.py --model-path <Alpamayo2-Super dir> \
  --stage-config examples/alpamayo/alpamayo2_super_stage_trn2.yaml \
  --reference alpamayo2_inputs.pt --output alpamayo2_super.npz --repeat 5
```

Expected: the first request compiles the vision, prefill, decode and expert graphs (a cold NEFF cache takes
several minutes) and the script prints `[run] {...}` with `first_request_s`, `warm_request_ms`, `cot` (the
reasoning text) and a per-stage `warm_breakdown_ms` (set `ALPAMAYO_PROFILE=1` to also time the vision tower
separately). `alpamayo2_super.npz` holds `pred_xyz` `[1, 64, 3]`, `pred_rot`, the raw `actions` and the generated
token ids.

### Options

| Flag / variable | Default | Meaning |
|---|---|---|
| `--model-path` | `$ALPAMAYO_WEIGHTS` | checkpoint directory |
| `--reference` | (required) | `.pt` with `model_inputs` from Step 1 |
| `--seed` | `0` | flow-matching initial-noise seed (same seed, same trajectory) |
| `--repeat` | `1` | extra timed requests after the first |
| `--output` | `alpamayo_actions.npz` | output path (`.json` alongside has the summary) |
| `ALPAMAYO_TEXT_BUCKETS` | `4608` | prompt-length buckets (one prefill graph each; the largest sets the KV-cache length) |

## Common issues

| Symptom | Fix |
|---|---|
| `RuntimeError: NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Unset it; use `NEURON_VISIBLE_DEVICES` with eight core ids (`"0,1,2,3,4,5,6,7"` in the stage config). |
| `prompt is N tokens; the largest bucket is M` | Add a bucket >= N to `ALPAMAYO_TEXT_BUCKETS` (prompts above 4,608 tokens, e.g. seven cameras, are not measured on device). |
| Out of device memory at load | Use eight cores; four do not hold the 34B weights plus cache. |
| Cold-start handshake timeout | Raise `ALPAMAYO_INIT_TIMEOUT_S` (default 3600 s). |

## Clean up

Compiled NEFFs live under the Neuron compiler cache (`$NEURON_LIBTORCH_CACHE_ROOT`); remove it to force a clean
recompile. Outputs are plain `.npz` / `.json` files at the `--output` path.

## Next steps

- [Alpamayo 2 Super model card](../models/alpamayo-2-super.md)
- [Alpamayo 1.5 quickstart](quickstart-offline-serving-alpamayo-1-5.md)
