# Tutorial: Deploy InternVLA-A1.5 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving InternRobotics InternVLA-A1.5 (a vision-language-action
robot policy) on AWS Trainium2 with the vLLM Omni Neuron plugin: environment, model download,
stage configuration, offline and served inference, and troubleshooting. -->
<!-- meta: keywords: InternVLA, InternVLA-A1.5, tutorial, vLLM Omni, Neuron, trn2, robot policy, VLA -->
<!-- meta: date_updated: 2026-10-05 -->
<!-- meta: content_type: tutorial -->

You will serve InternRobotics' InternVLA-A1.5 robot policy on one NeuronCore and get a 50-step action
chunk from camera views, a robot state and a language instruction — offline first, then through the
vLLM Omni engine. The whole model is ~5.4 GiB in BF16 (2.68B parameters), so one trn2 NeuronCore
is enough (the only hardware measured); the first request compiles the graphs (minutes on a cold NEFF cache — see
[Performance](../models/internvla-a15.md#performance)).

## Step 1: Set up your environment

Follow the [setup guide](setup-guide.md) for your instance.

## Step 2: Download the model and resolve the VLM config

```bash
huggingface-cli download InternRobotics/InternVLA-A1.5-base --local-dir /opt/models/internvla-a15-base
```

The checkpoint does not carry the Qwen3.5-2B VLM backbone's own `config.json` (upstream resolves it from
the Hub at `Qwen/Qwen3.5-2B`). Either let it download automatically, or fetch just the config offline:

```bash
huggingface-cli download Qwen/Qwen3.5-2B --include "config.json" --local-dir /opt/models/qwen3.5-2b-config
huggingface-cli download Qwen/Qwen3.5-2B --include "tokenizer*" --local-dir /opt/models/qwen3.5-tokenizer
```

Point `--vlm-config /opt/models/qwen3.5-2b-config` (or `$INTERNVLA_VLM_CONFIG`) at it for every command
below. Served requests also need the tokenizer: `--tokenizer /opt/models/qwen3.5-tokenizer`,
`$INTERNVLA_TOKENIZER`, or `model_config.tokenizer` in the stage config.

## Step 3: Review the stage configuration

```yaml
# examples/internvla/internvla_stage.yaml (excerpt)
stage_args:
  - stage_id: 0
    stage_type: diffusion
    final_output_type: actions
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_class_name: InternVLAA15Pipeline
      dtype: bfloat16
      model_config:
        _placeholder: true   # see note below
      parallel_config:
        tensor_parallel_size: 1
```

`model_config` must keep at least one real key: a bare `model_config:` with no sub-key parses as YAML
`null`, and the engine's own tensor-capture setup calls `.get()` on it unconditionally. Set
`vlm_config: /path/to/qwen3.5-2b-config` here instead of the placeholder once you have Step 2's path.

## Step 4: Stage the checkpoint for serving

The engine's generic pipeline lookup (`OmniDiffusionConfig.enrich_config`) needs the checkpoint's
`config.json` to carry `architectures: ["InternVLAA15Pipeline"]`; InternVLA-A1.5's released checkpoint
doesn't have it. Stage a non-destructive, symlinked copy with the key patched in rather than editing the
original (possibly shared, read-only) directory:

```bash
python -m vllm_omni_neuron.diffusion.models.internvla.stage_for_serving \
  /opt/models/internvla-a15-base /opt/models/internvla-a15-base-served
```

Use the `-served` path as `--model-path` from here on.

## Step 5: Run offline (standalone, no engine)

```bash
python examples/internvla/run.py --model /opt/models/internvla-a15-base-served \
  --vlm-config /opt/models/qwen3.5-2b-config --device neuron --compare-cpu
```

This drives the three device graphs directly (no vLLM engine, no multiprocessing) and is the fastest way
to confirm the port and your weights are correct — `--compare-cpu` also runs a CPU reference and reports
the relative error, which should land inside the BF16 rounding band (~0.6%, see the
[model card](../models/internvla-a15.md#accuracy-evaluation)).

## Step 6: Serve through vLLM Omni

```bash
python examples/internvla/serve.py --model-path /opt/models/internvla-a15-base-served \
  --vlm-config /opt/models/qwen3.5-2b-config --tokenizer /opt/models/qwen3.5-tokenizer \
  --image-dir /path/to/frames --prompt "pick up the red cube and put it in the bowl" --repeat 5
```

The Omni engine always runs its diffusion stage as a multi-process worker, even at
`tensor_parallel_size: 1`. If `NEURON_RT_VISIBLE_CORES` is set in your shell, unset it first — the engine
refuses to start with it present and expects `NEURON_VISIBLE_DEVICES` (set via the stage config's
`devices:` field) instead.

Expected: engine startup (~18-20 s), then a first request of ~80 s with a warm NEFF cache (several
minutes more on a cold one: each prefix bucket compiles its graphs), then warm requests of about 228 ms
(1 view), 252 ms (2 views) or 267 ms (3 views) on trn2. The script writes the decoded action chunk and a
timing summary to `--output`.

## Step 7: Clean up

Compiled NEFFs live under `$TORCH_NEURONX_NEFF_CACHE_DIR`; remove it to force a clean recompile. The
staged checkpoint directory from Step 4 is a thin symlink tree — safe to delete; it does not touch the
original weights.

## Troubleshooting

See [Common issues](../getting-started/quickstart-offline-serving-internvla-a15.md#common-issues) in the
quickstart.

## Next steps

- [InternVLA-A1.5 model card](../models/internvla-a15.md)
- [Quickstart: Offline action-chunk generation](../getting-started/quickstart-offline-serving-internvla-a15.md)
