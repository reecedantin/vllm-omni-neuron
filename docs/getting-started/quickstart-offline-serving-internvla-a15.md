# Quickstart: Offline action-chunk generation with InternVLA-A1.5 on Neuron

<!-- meta: description: Generate a robot action chunk offline with InternRobotics InternVLA-A1.5 on AWS
Trainium2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and a full served run
with real camera frames. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, InternVLA, InternVLA-A1.5, robot policy, VLA,
offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-05 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate one robot action chunk offline with InternVLA-A1.5 on a
Trainium2 instance using the vLLM Omni Neuron plugin. When you finish, you have a 50-step
action chunk produced by the [InternVLA-A1.5](../models/internvla-a15.md) model, either standalone or
served through the vLLM Omni engine.

## Prerequisites

- One SSH-accessible `trn2` instance (the only hardware measured).
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`InternRobotics/InternVLA-A1.5-base`) and, separately, the
  Qwen3.5-2B VLM `config.json` the checkpoint does not carry — either let it download from the Hub, or
  point `--vlm-config` / `$INTERNVLA_VLM_CONFIG` at a local copy (see
  [Known limitations](../models/internvla-a15.md#known-limitations)).
- For served requests, the Qwen3.5 tokenizer (`tokenizer.json` + `tokenizer_config.json` of
  `Qwen/Qwen3.5-2B`), e.g.
  `huggingface-cli download Qwen/Qwen3.5-2B --include "tokenizer*" --local-dir /path/to/qwen3.5-tokenizer`.

## Step 1: Run a quick smoke test (standalone)

```bash
python examples/internvla/run.py --model /path/to/InternVLA-A1.5-base --device neuron
```

Expected: the first call compiles 3 graphs (vision tower, VLM prefix, action-expert denoise step) and
takes several minutes on a fresh NEFF cache (~830 s measured on trn2); the script prints a one-line JSON
summary (`first_call_s`, `warm_s`, a `warm_breakdown` of host-prep/prefix/denoise time, `deterministic`,
`finite`) and saves the action chunk + the request under `--out-dir`. Without real images this is a
plumbing check (random pixels, the real token layout), not a meaningful policy output — add
`--compare-cpu` to also check the device result against a CPU run of the same request.

## Step 2: Serve a full request through vLLM Omni

```bash
python examples/internvla/serve.py --model-path /path/to/InternVLA-A1.5-base \
  --vlm-config /path/to/qwen3.5-config --tokenizer /path/to/qwen3.5-tokenizer \
  --image-dir /path/to/frames --prompt "pick up the red cube and put it in the bowl" \
  --repeat 5 --output internvla_actions.npz
```

`--image-dir` must contain `image_<i>.png` for `i` in `0..n_images-1` (RGB; `--n-images` controls
the count, default 1). `--repeat` runs additional warm (post-compile) requests through the engine and
reports their latency and whether the same-seed result is deterministic. `--obs-npz` replays recorded
observations instead (`images [T,V,H,W,3]` uint8, `state [T,S]` already normalised, `task`), and
`--vary` sends a new recorded step with every timed request.

The pipeline builds each request the way upstream's inference transform does: the Qwen3.5 chat
template with the system message, one vision block per camera, then
`Task: <prompt>; Control Mode: <joint>; State: <256-level bins>; Output: <Subtask, Action>`. The
checkpoint reads the robot state from that text (`tokenize_state`), so send it normalised with your
robot's statistics. Without a tokenizer the request is refused.

> **Note:** the script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/internvla/internvla_stage.yaml`. The Omni engine is always multi-process, even at
> `tensor_parallel_size: 1` — if you drive it from your own script, source
> `fleet/bin/multirank.sh`-equivalent environment (unset `NEURON_RT_VISIBLE_CORES`, set
> `NEURON_VISIBLE_DEVICES`) first; see [Common issues](#common-issues).

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` | `InternRobotics/InternVLA-A1.5-base` | checkpoint directory or Hub id |
| `--vlm-config` | unset (Hub download) | path to the Qwen3.5-2B `config.json` the checkpoint doesn't carry |
| `--tokenizer` | `$INTERNVLA_TOKENIZER` | Qwen3.5 tokenizer directory (required) |
| `--prompt` | a pick-and-place instruction | the language instruction |
| `--image-dir` | unset (random frames) | directory with `image_<i>.png` camera views |
| `--obs-npz` / `--obs-index` / `--vary` | unset | replay recorded observations; `--vary` = a new step per timed request |
| `--n-images` | `1` | number of camera views |
| `--seed` | `0` | seeds both the random frames (if used) and the action expert's initial noise |
| `--repeat` | `1` | extra timed requests after the first, to measure warm latency |
| `--output` | `internvla_actions.npz` | where to save the decoded action chunk (`.json` alongside has a summary) |

## Common issues

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Qwen3.5 VLM config ... not found` | Pass `--vlm-config` to a local `config.json`, or set `$INTERNVLA_VLM_CONFIG`. |
| `... needs the Qwen3.5 tokenizer` (request error) | Pass `--tokenizer`, set `$INTERNVLA_TOKENIZER`, or set `model_config.tokenizer` in the stage config. |
| `RuntimeError: NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM` | Don't set `NEURON_RT_VISIBLE_CORES`; the Omni engine is always multi-process. Pick the core via the stage config's `devices:` field and set `NEURON_VISIBLE_DEVICES` instead. |
| `RuntimeError: Orchestrator initialization failed: model_index.json not found` | The checkpoint's `config.json` needs an `architectures: ["InternVLAA15Pipeline"]` key for the engine's generic pipeline-registry lookup. Use [`stage_for_serving.py`](../../vllm_omni_neuron/diffusion/models/internvla/stage_for_serving.py) to stage a patched, symlinked copy without touching the original checkpoint. |
| `torch._dynamo.exc.Unsupported: ... torch._C._nn.gelu` | Already fixed in this port (hand-written GELU in `qwen3_5.py`); if you see this in your own changes, avoid `F.gelu(..., approximate=...)` inside a `fullgraph=True` region. |

## Clean up

Compiled NEFFs live under the Neuron compiler cache directory (`$TORCH_NEURONX_NEFF_CACHE_DIR`); remove it
to force a clean recompile. Output files are plain `.npz`/`.json` under the path you passed to `--output`.

## Next steps

- [InternVLA-A1.5 model card](../models/internvla-a15.md)
- [Tutorial: Deploy InternVLA-A1.5](../tutorials/tutorial-internvla-a15.md)
