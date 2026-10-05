# Quickstart: Offline policy inference with FLUX 3 Action on Trainium2

<!-- meta: description: Predict a robot action chunk offline with Black Forest Labs' FLUX 3 Action DROID policy on
AWS Trainium2 using the vLLM Omni Neuron plugin. Covers building an observation, a single-core run and the
served (vLLM Omni stage) path. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, FLUX 3 Action, DROID, robot policy, action chunk, offline,
Trainium2, trn2, NeuronCore-v3, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to predict a robot action chunk offline on a Trainium2 instance with the vLLM Omni
Neuron plugin. When you finish, you will have a `(1, 32, 8)` DROID action chunk from the
[FLUX 3 Action](../models/flux3-action.md) world action model, produced on one NeuronCore.

## Prerequisites

- One SSH-accessible `trn2` instance.
- The environment prepared with the [setup guide](setup-guide.md).
- Network access to download two repositories from Hugging Face: `black-forest-labs/flux-3-action-droid` (the
  policy, about 14 GB for the BF16 root package) and `black-forest-labs/flux-3-action-base` (the shared video VAE
  and Qwen3-VL text encoder, about 10 GB). Both are under the FLUX Kommunity License; accept it on the model pages
  first.

Download both once (skip the optimized `variants/`):

```bash
hf download black-forest-labs/flux-3-action-droid --exclude 'variants/*' --local-dir weights/droid
hf download black-forest-labs/flux-3-action-base --include 'video_vae.safetensors' --include 'text_encoder/*' \
    --local-dir weights/base
export FLUX3_ACTION_BASE=$PWD/weights/base
```

## Step 1: Build an observation

A DROID observation is an NPZ with three 360x640 RGB cameras (`images.wrist`, `images.left`, `images.right`) and
an 8-value `state` (seven joint positions in radians, then the gripper closed fraction). Without a robot or a
recorded episode, crop one from the demo clip shipped with the policy:

```bash
python examples/flux3_action/make_observation.py --policy weights/droid --output obs.npz
```

This writes `obs.npz` and `obs.json` (the task instruction). For a recorded DROID episode, use the upstream
`examples/droid/make_observation.py`; the format is the same.

## Step 2: Run the policy on one NeuronCore

```bash
python examples/flux3_action/run_policy.py --policy weights/droid --base weights/base \
    --obs obs.npz --out-dir out/ --repeat 2
```

Expected: the last line is a JSON summary with `"actions_shape": [1, 32, 8]` and `"ok": true`, and
`out/outputs.pt` holds the actions and the predicted video latents. The first prediction compiles the DiT graphs
(several minutes on a cold cache); the warm predictions that follow take seconds.

## Step 3: Serve one request through vLLM Omni

```bash
python examples/flux3_action/serve_policy.py --policy weights/droid --base weights/base \
    --obs obs.npz --out-dir out-served/ --warm 2
```

The script runs the `Flux3ActionPipeline` stage from `examples/flux3_action/flux3_action_stage.yaml`. Expected:
`out-served/actions.npy` of shape `(1, 32, 8)` and `out-served/served_report.json` with the warm latency
(about 10 s on Trn2) and the per-stage timing.

### Change the run

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `0` | noise seed; the same seed and observation give the same actions |
| `--decode` (`run_policy.py`) | off | also decode the predicted video frames on the host (slow; see the model card) |
| `--reference` | none | a CPU reference file to report action parity against |
| `--stage-config` (`serve_policy.py`) | TP1 stage | `flux3_action_stage_tp4_cfg2.yaml` (recommended, 8 cores: 1.0 s warm), `flux3_action_stage_tp4.yaml` (4 cores: 1.6 s) |

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | The served stage is multi-process even at TP1. Select cores with `NEURON_VISIBLE_DEVICES` and unset `NEURON_RT_VISIBLE_CORES`. |
| `model_index.json not found` | Pass the downloaded policy directory; `serve_policy.py` adds the discovery files in a symlinked serving directory under `--out-dir`. |
| `video_vae ... is not local` | Set `FLUX3_ACTION_BASE` (or `--base`) to the `flux-3-action-base` download, or allow Hub access. |
| First request takes minutes | Cold compile; later runs reuse the NEFF cache (`TORCH_NEURONX_NEFF_CACHE_DIR`). |

## Clean up

Remove `out/`, `out-served/` and, if you no longer need it, the NEFF cache directory.

## Next steps

- [FLUX 3 Action model card](../models/flux3-action.md)
- [Tutorial: Serve FLUX 3 Action with vLLM Omni Neuron](../tutorials/tutorial-flux3-action.md)
