# Tutorial: Deploy HunyuanVideo-1.5 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving HunyuanVideo-1.5 text-to-video on AWS Trainium2 with the
vLLM Omni Neuron plugin: environment, model download, stage configuration, offline and online inference,
troubleshooting. -->
<!-- meta: keywords: HunyuanVideo-1.5, tutorial, vLLM Omni, Neuron, trn2, text-to-video -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: tutorial -->

You will serve HunyuanVideo-1.5 480p text-to-video on one Trainium2 chip (four logical NeuronCores), generate a
25-frame 848x480 clip offline, and serve the same pipeline online. Allow about 45 minutes for the first compile
at a new geometry; warm requests take about 3 minutes.

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). The pipeline needs no extra packages. Set:

```bash
export HV15_BLOCKS_PER_GRAPH=6  # DiT compiled as 6-block graphs that share one NEFF
```

## Step 2: Download the model (optional)

```bash
huggingface-cli download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --local-dir HunyuanVideo-1.5-480p_t2v
```

Pass `--model-path HunyuanVideo-1.5-480p_t2v` to the scripts below to use the local copy.

## Step 3: Review the stage configuration

```yaml
# examples/hunyuanvideo15/hunyuanvideo15_stage.yaml (excerpt)
    runtime:
      devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    engine_args:
      model_class_name: HunyuanVideo15Pipeline
      dtype: bfloat16
      parallel_config:
        tensor_parallel_size: 8
        cfg_parallel_size: 2
```

`tensor_parallel_size=8` shards each of the 54 DiT blocks by attention head over an adjacent chip pair (8 cores);
`cfg_parallel_size=2` runs the conditional and unconditional guidance branches on separate chip pairs at the same
time. The Qwen2.5-VL text tower runs on the NeuronCores, sharded TP=4 over each chip, with the positive and
negative prompts encoded on different chips; the VAE decode is split into fixed-shape tiles shared by all 16 ranks
and merged on rank 0. Smaller layouts: `hunyuanvideo15_stage_tp4_cfg2.yaml` (8 cores) and, on one chip,
`hunyuanvideo15_stage_tp2_cfg2.yaml` (TP=2 x CFG-parallel 2) or `hunyuanvideo15_stage_tp4.yaml` (TP=4, sequential CFG).

## Step 4: Run inference

### Text-to-video, offline

```bash
python examples/hunyuanvideo15/run.py \
  --prompt "A golden retriever runs across a sunlit meadow, slow motion, cinematic." \
  --height 480 --width 848 --num-frames 25 --steps 50 --output retriever.mp4
```

Text in double quotes inside the prompt (for example `a sign reads "OPEN"`) is routed to the byT5 glyph encoder.

### Online serving

```bash
vllm serve hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --omni \
  --stage-configs-path examples/hunyuanvideo15/hunyuanvideo15_stage.yaml --port 8000
```

Send requests in the vLLM-Omni video generation format with `height`, `width`, `num_frames` and
`num_inference_steps` matching a geometry you have already compiled, or the first request compiles it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing` | vLLM's Neuron worker selects cores itself | unset it; set `NEURON_VISIBLE_DEVICES` to a comma list of cores |
| Very long first request | cold compile of the DiT graphs for the geometry | wait; later requests at the same shape reuse the NEFF cache |
| `NCC_IBTN020` compiling the VAE | tile too large for the compiler's access patterns | keep `HV15_VAE_TILE` <= 11 |
| Out of host memory | text tower (~15 GB) | keep 60 GB free |

## Conclusion

You served HunyuanVideo-1.5 on one Trainium2 chip with tensor and CFG parallelism, offline and online.

## Next steps

- [HunyuanVideo-1.5 model card](../models/hunyuanvideo15.md): features, accuracy and performance.
- [Quickstart: Offline text-to-video](../getting-started/quickstart-offline-serving-hunyuanvideo15.md).
