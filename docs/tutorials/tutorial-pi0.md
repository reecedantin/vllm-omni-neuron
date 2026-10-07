# Tutorial: Deploy pi0 / pi0.5 / pi0.52 with vLLM Omni Neuron

<!-- meta: description: Deploy the LeRobot pi0, pi0.5 and pi0.52 Vision-Language-Action policies on
AWS Trainium2 with the vLLM Omni Neuron plugin: environment, stage configuration, offline inference,
the pi0.52 subtask step, accuracy checks and tuning. -->
<!-- meta: keywords: vLLM Omni, Neuron, Trainium2, pi0, pi0.5, pi0.52, LeRobot, VLA, robot policy, tutorial -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

This tutorial deploys the π0 family on one Trainium2 NeuronCore and walks through what each part
of the port does. For the shortest path, use the
[quickstart](../getting-started/quickstart-offline-serving-pi0.md).

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). Each request runs on one NeuronCore;
at LNC=2 a trn2 logical core has about 24 GB of HBM and the model needs about 6 GB.

## Step 2: Download the model (optional)

```bash
hf download lerobot/pi052_base --local-dir pi052_base
hf download google/paligemma-3b-pt-224 --local-dir paligemma-3b-pt-224 --include "tokenizer*"
```

The LeRobot checkpoints ship no tokenizer; the pipelines load PaliGemma's.

## Step 3: Review the stage configuration

```yaml
# examples/pi0/pi052_stage.yaml (excerpt)
    runtime:
      devices: "0"            # one NeuronCore
    engine_args:
      model_class_name: Pi05Pipeline     # Pi0Pipeline in pi0_stage.yaml
      dtype: bfloat16
      model_config:
        tokenizer: google/paligemma-3b-pt-224
        max_new_subtask_tokens: 48       # pi0.52 only
      parallel_config:
        tensor_parallel_size: 1
```

The pipeline compiles a few fixed-shape graphs: the prefix (SigLIP over every camera slot plus the
PaliGemma language model, producing a per-layer K/V cache) and one flow-matching denoise step
over the action expert, which the host replays for each of the 10 steps. π0.52 adds the subtask
graphs: image embedding once per request, one language-model prefill over the image + prompt
prefix, then one KV-cached decode step per generated token (the plugin's shared decode-attention
layer). Set `subtask_kv_cache: false` in `model_config` to fall back to re-running the language
model over the whole sequence for every token.

## Step 4: Run inference

### π0.5 / π0.52

```bash
python examples/pi0/run.py --model pi052_base --tokenizer paligemma-3b-pt-224 \
  --task "pick up the red cube and place it in the bowl" --image base_0_rgb=base.png --profile
```

π0.52 first generates a low-level subtask from `--task` with PaliGemma's own language-model head
(greedy, `"User: {task}\nAssistant:"`), then builds the action prompt
`"User: {subtask}, State: {bins};\n"`. When the model generates nothing, the task string is used,
as LeRobot does.

### π0 (base)

```bash
python examples/pi0/run.py --model lerobot/pi0_base --stage-config examples/pi0/pi0_stage.yaml \
  --task "pick up the red cube" --state state.json
```

π0 takes the continuous state as its own token next to the action tokens; it has no subtask step.

### Online serving

The pipelines register as `Pi05Pipeline` / `pi05` / `pi052` and `Pi0Pipeline` / `pi0`, so the
same stage configs work with `vllm serve` and the OpenPI realtime serving layer, which delivers
the observation as `sampling_params.extra_args["robot_obs"]`.

## Optional: check accuracy on your host

```bash
# CPU unit tests (shrunk random-weight checkpoints, no device needed)
pytest test/unit/test_pi0_tiny.py test/unit/test_pi0_base_tiny.py
# device vs. CPU fp32 on the real weights
python examples/pi0/device_check.py --model pi052_base --tokenizer paligemma-3b-pt-224 --out run_pi052
python examples/pi0/device_check_base.py --model pi0_base --tokenizer paligemma-3b-pt-224 --out run_pi0
```

The model cards list the measured numbers and the LeRobot-parity scripts.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Gated-repo error for the tokenizer | Accept the PaliGemma terms and log in, or pass `--tokenizer <local dir>` |
| Long first request | The graphs compile cold; set `TORCH_NEURONX_NEFF_CACHE_DIR` to a persistent directory |
| π0.52 subtask is always the task | The generated subtask was empty, so the task is used (LeRobot's fallback); expected for the base checkpoint |

## Conclusion

You ran the π0 family on one NeuronCore: π0.5/π0.52 at about 0.18 s per request including the
subtask step, π0 at about 0.1 s per action chunk.

## Next steps

- [π0.5 / π0.52 model card](../models/pi05.md)
- [π0 model card](../models/pi0.md)
- [Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md)
