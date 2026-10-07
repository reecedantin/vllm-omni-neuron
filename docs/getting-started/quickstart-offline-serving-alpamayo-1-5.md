# Quickstart: Offline trajectory prediction with Alpamayo 1.5 on Neuron

<!-- meta: description: Predict an autonomous-driving trajectory and its reasoning text offline with NVIDIA
Alpamayo 1.5 on AWS Trainium2 using the vLLM Omni Neuron plugin. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Alpamayo, Alpamayo 1.5, autonomous driving, VLA,
trajectory, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to run one Alpamayo 1.5 request offline on a Trainium2 instance with the vLLM
Omni Neuron plugin. When you finish, you have a 64-waypoint trajectory and the model's Chain-of-Causation
reasoning text produced by [Alpamayo 1.5](../models/alpamayo-1-5.md) on the four NeuronCores of one Trainium2 chip.

## Prerequisites

- One `trn2.48xlarge` instance (the model uses the four NeuronCores of one chip).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- The `nvidia/Alpamayo-1.5-10B` weights (gated; request access, then `hf auth login`) and the
  `Qwen/Qwen3-VL-8B-Instruct` tokenizer / processor files, downloaded locally.
- A separate Python 3.12 environment with the upstream [alpamayo1_5](https://github.com/NVlabs/alpamayo1.5)
  package, used only to build the processor inputs (Step 1).

> **Note:** `examples/alpamayo/run.py` uses the vLLM Omni entrypoint (`Omni`) and reads
> `examples/alpamayo/alpamayo_stage_trn2.yaml` (TP=4, `devices: "0,1,2,3"`). With multi-process TP, select cores
> with `NEURON_VISIBLE_DEVICES` (four core ids of one chip, `4d..4d+3`) and leave `NEURON_RT_VISIBLE_CORES` unset.
> `alpamayo_stage_trn2_tp8.yaml` (eight cores of an adjacent chip pair) and `alpamayo_stage_trn2_tp2.yaml` (two
> cores) are alternatives; pass them with `--stage-config`.

## Step 1: Build the request inputs

The model consumes the Qwen3-VL processor's outputs for upstream's driving prompt. In the upstream environment:

```bash
python examples/alpamayo/reference_1_5.py --model <Alpamayo-1.5-10B dir> \
  --backbone-config <Qwen3-VL-8B-Instruct dir> --out alpamayo_inputs.pt
```

This writes the tokenized prompt, pixel values and ego history (and upstream's own sampled CPU output). By
default the camera frames are seeded random images; replace them in the script with real frames for a
meaningful prediction.

## Step 2: Run the request on Trainium

```bash
export NEURON_VISIBLE_DEVICES=0,1,2,3
export ALPAMAYO_TOKENIZER_DIR=<Qwen3-VL-8B-Instruct dir>
python examples/alpamayo/run.py --model-path <Alpamayo-1.5-10B dir> \
  --reference alpamayo_inputs.pt --output alpamayo.npz --repeat 5
```

Expected: the first request compiles the vision, prefill, decode and expert graphs (several minutes on a cold
NEFF cache; under a minute once cached) and the script prints `[run] {...}` with `first_request_s`,
`warm_request_ms` (about 0.77 s), `cot` (the reasoning text) and a per-stage `warm_breakdown_ms` (set
`ALPAMAYO_PROFILE=1` to also time the vision tower separately). `alpamayo.npz` holds `pred_xyz` `[1, 64, 3]`,
`pred_rot`, the raw `actions` and the generated token ids.

### Options

| Flag / variable | Default | Meaning |
|---|---|---|
| `--model-path` | `$ALPAMAYO_WEIGHTS` | checkpoint directory |
| `--reference` | (required) | `.pt` with `model_inputs` from Step 1 |
| `--seed` | `0` | flow-matching initial-noise seed (same seed, same trajectory) |
| `--repeat` | `1` | extra timed requests after the first |
| `--output` | `alpamayo_actions.npz` | output path (`.json` alongside has the summary) |
| `ALPAMAYO_TEXT_BUCKETS` | `1024,2048,3072` | prompt-length buckets (one prefill graph each) |

## Common issues

| Symptom | Fix |
|---|---|
| `RuntimeError: NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Unset it; use `NEURON_VISIBLE_DEVICES` with one core id per stage-config `devices:` entry (`"0,1,2,3"` by default). |
| `prompt is N tokens; the largest bucket is M` | Add a bucket >= N to `ALPAMAYO_TEXT_BUCKETS`. |
| Tokenizer download fails offline | Set `ALPAMAYO_TOKENIZER_DIR` to a local Qwen3-VL-8B-Instruct (or Cosmos-Reason2-8B) directory. |
| Cold-start handshake timeout | Raise `ALPAMAYO_INIT_TIMEOUT_S` (default 3600 s). |

## Clean up

Compiled NEFFs live under the Neuron compiler cache (`$NEURON_LIBTORCH_CACHE_ROOT`); remove it to force a clean
recompile. Outputs are plain `.npz` / `.json` files at the `--output` path.

## Next steps

- [Alpamayo 1.5 model card](../models/alpamayo-1-5.md)
- [Tutorial: Deploy Alpamayo 1.5](../tutorials/tutorial-alpamayo-1-5.md)
