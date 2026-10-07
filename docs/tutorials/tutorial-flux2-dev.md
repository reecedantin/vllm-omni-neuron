# Tutorial: Deploy FLUX.2-dev with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving FLUX.2-dev on AWS Trainium2 with the vLLM Omni Neuron
plugin: environment, model download, stage configuration, offline inference, accuracy checks, troubleshooting. -->
<!-- meta: keywords: FLUX.2-dev, tutorial, vLLM Omni, Neuron, trn2, Trainium2, text-to-image -->
<!-- meta: date_updated: 2026-10-06 -->
<!-- meta: content_type: tutorial -->

You will serve FLUX.2-dev (a 32B DiT with a 24B Mistral text encoder) on 8 NeuronCores of a trn2.48xlarge
and generate 1024x1024 images.

- **First run:** about 40 minutes, mostly the one-time VAE tile compiles (weight load is about 25 s).
- **Later images:** 26 s each at 50 steps (10.3 s on 32 cores with the default TP=8 x CP=4 stage config).

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). Two FLUX.2-specific settings:

- **Core selection:** select the cores with `NEURON_VISIBLE_DEVICES` (a comma list such as `0,1,2,3,4,5,6,7`),
  not `NEURON_RT_VISIBLE_CORES`.
- **Queue depth:** keep `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63`. `examples/flux2/run.py` sets both for you.

## Step 2: Download the model (optional)

The repository is gated: accept the FLUX Non-Commercial License on the
[model page](https://huggingface.co/black-forest-labs/FLUX.2-dev) first.

```bash
huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir <dir>
```

## Step 3: Review the stage configuration

```yaml
# examples/flux2/flux2_stage.yaml (excerpt)
    runtime:
      devices: "0,1,2,...,31"     # 32 logical cores
    engine_args:
      model_class_name: Flux2Pipeline
      dtype: bfloat16
      model_config:
        double_group: 1     # double-stream blocks per compiled graph
        single_group: 6     # single-stream blocks per compiled graph
        te_group: 10        # text-encoder layers per compiled graph
      parallel_config:
        tensor_parallel_size: 8
        ring_degree: 4      # context parallelism: 4 TP groups split the tokens
```

**Parallelism.** TP=8 shards the DiT and the text encoder over attention heads and the MLP hidden dimension
inside each pair of chips. Context parallelism (`ring_degree: 4`) splits the text and image tokens over four
such groups, which all-gather K/V once per block; each group holds a full TP=8 copy of the weights. There is no
CFG pass to parallelise: FLUX.2-dev is guidance-distilled. On fewer cores use `flux2_stage_tp8_cp2.yaml`
(16 cores) or `flux2_stage_tp8.yaml` (8 cores); the model card compares the layouts.

**Graph grouping.** Blocks of a kind share one compiled graph, with the weights passed as graph inputs, so the
compile cost does not grow with depth.

**VAE.** The VAE decodes on all cores as 512 px tiles. It runs two passes so the tiles agree:

1. Gather each GroupNorm layer's statistics over the whole image, from all tiles.
2. Normalise every tile with those statistics.

Plain per-tile tiling would leave visible tile rectangles (24 dB against the untiled decode instead of
40.9 dB). Tune it with `FLUX2_VAE_TILE`, `FLUX2_VAE_OVERLAP` and `FLUX2_VAE_GN_PASSES`.

## Step 4: Run inference

### Text-to-image

```bash
python examples/flux2/run.py --model-path <dir> \
    --prompt "A lighthouse on a cliff at dawn, mist over the sea, cinematic" \
    --height 1024 --width 1024 --steps 50 --output lighthouse.png
```

### Check accuracy on your instance

```bash
pytest test/unit/test_flux2_*.py -q
FLUX2_WEIGHTS=<dir> pytest test/neuron/test_flux2_pipeline_device.py -q
```

The device test checks two things:

- one denoising step (text encoder, DiT and scheduler) matches the same pipeline on CPU, within the bf16 band;
- the device VAE matches a CPU fp32 decode at 35 dB or better.

### Online serving

```bash
vllm serve black-forest-labs/FLUX.2-dev --omni --stage-configs-path examples/flux2/flux2_stage.yaml --port 8000
```

Online serving uses the same pipeline and stage config, but it was not validated on Trn2 for this release (see
the model card's known limitations).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Execution Queue Full` | the runtime's default hardware queue is shallower than the per-step block launches | `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63` |
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | vllm-neuron workers select cores themselves | use `NEURON_VISIBLE_DEVICES` |
| Grid-like texture in the image | too few steps for the distilled schedule | use 28-50 steps |
| Visible tile rectangles | `FLUX2_VAE_GN_PASSES=1` (per-tile statistics) | leave the default (2) |
| First request is slow | cold compile of the VAE tile graphs and the per-resolution DiT graphs | later requests hit the NEFF cache |

## Conclusion

FLUX.2-dev runs end to end on 8 Trainium2 NeuronCores:

- the text encoder, the DiT and the VAE all run on device;
- a 50-step 1024x1024 image takes 26 s warm.

## Next steps

- [FLUX.2-dev model card](../models/flux2-dev.md)
- [Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md)
