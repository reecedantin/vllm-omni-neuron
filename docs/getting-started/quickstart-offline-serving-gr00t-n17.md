# Quickstart: Offline action-chunk generation with GR00T N1.7 on Neuron

<!-- meta: description: Generate a robot action chunk offline with NVIDIA GR00T N1.7 on AWS Inferentia2 /
Trainium using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full run with real camera frames. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, GR00T, GR00T N1.7, robot policy, VLA, offline, inf2,
trn2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate one robot action chunk offline with GR00T N1.7 on an
Inferentia2 or Trainium instance using the vLLM Omni Neuron plugin. When you finish, you have a 40-step
action chunk produced offline by the [GR00T N1.7](../models/gr00t-n17.md) model.

## Prerequisites

- One SSH-accessible `inf2.xlarge` (or larger / `trn1` / `trn2`) instance.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`nvidia/GR00T-N1.7-3B`, gated) and the processor
  (`Qwen/Qwen3-VL-2B-Instruct`) from Hugging Face. Request access to the GR00T repo and run `hf auth login`
  (or set `HF_TOKEN`) first.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/gr00t/gr00t_stage.yaml` (identical to `gr00t_stage_trn2.yaml`; GR00T needs no per-hardware
> tuning). Set `GR00T_VLM_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct` to use a local processor instead of the Hub.

## Step 1: Run a quick smoke test

```bash
python examples/gr00t/run.py --model-path nvidia/GR00T-N1.7-3B --output /tmp/gr00t_smoke.npz
```

Expected: the first call compiles 3 graphs (vision tower, text decoder, action head) and takes roughly
30-60s on a fresh NEFF cache; the script prints `[run] {"first_request_s": ..., ...}` and writes
`/tmp/gr00t_smoke.npz` with one 40-step action chunk per modality key (`eef_9d`, `gripper_position`,
`joint_position` for the default DROID embodiment). Without `--image-dir`, camera frames are seeded random
noise — enough to prove the pipeline runs, not a meaningful policy output.

## Step 2: Run a full offline generation with real camera frames

```bash
python examples/gr00t/run.py --model-path nvidia/GR00T-N1.7-3B \
  --image-dir /path/to/frames --prompt "pick up the red cube and put it in the bowl" \
  --repeat 20 --output gr00t_actions.npz
```

`--image-dir` must contain `<camera>_<t>.png` for `t` in `0, 1` (the two cameras `exterior_image_1_left` and
`wrist_image_left`, 180x320 RGB, matching the DROID embodiment's `video.delta_indices=[-15, 0]`). `--repeat`
runs additional warm (post-compile) requests and reports their latency.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` | `nvidia/GR00T-N1.7-3B` | checkpoint directory or Hub id |
| `--prompt` | a pick-and-place instruction | the language instruction |
| `--image-dir` | unset (random frames) | directory with the two camera PNGs |
| `--seed` | `0` | seeds both the random frames (if used) and the action head's initial noise |
| `--repeat` | `1` | extra timed requests after the first, to measure warm latency |
| `--output` | `gr00t_actions.npz` | where to save the decoded action chunk (`.json` alongside has a summary) |

## Common issues

| Symptom | Fix |
|---|---|
| `GatedRepoError` / 401 downloading `nvidia/GR00T-N1.7-3B` | Request access at the model page, then `hf auth login` or set `HF_TOKEN`. |
| `RuntimeError: NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM` | Don't set `NEURON_RT_VISIBLE_CORES`; pick the core via the stage config's `devices:` field instead. |
| Backbone config / processor errors offline | Set `GR00T_VLM_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct` to a local checkout; the backbone architecture itself needs no download. |

## Clean up

Compiled NEFFs live under the Neuron compiler cache directory (`$NEURON_LIBTORCH_CACHE_ROOT`, or
`~/.cache/neuron_libtorch` by default); remove it to force a clean recompile. Output files are plain
`.npz`/`.json` under the path you passed to `--output`.

## Next steps

- [GR00T N1.7 model card](../models/gr00t-n17.md)
- [Tutorial: Deploy GR00T N1.7](../tutorials/tutorial-gr00t-n17.md)
