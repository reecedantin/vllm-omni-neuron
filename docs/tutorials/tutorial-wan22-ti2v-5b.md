# Tutorial: Deploy Wan2.2-TI2V-5B and FastWan DMD2 with vLLM Omni Neuron

<!-- meta: description: Deploy Wan2.2-TI2V-5B and the 3-step DMD2 distillation FastWan2.2-TI2V-5B for
offline text-to-video generation on AWS Trainium2 with the vLLM Omni Neuron plugin: environment,
stage configuration, parallel layout, generation, accuracy checks and troubleshooting. -->
<!-- meta: keywords: vLLM Omni, Neuron, Wan2.2, Wan2.2-TI2V-5B, FastWan, DMD2, text-to-video,
Trainium2, trn2, tutorial, context parallelism, tensor parallelism -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

This tutorial deploys [Wan2.2-TI2V-5B](../models/wan22-ti2v-5b.md) and its few-step distillation
FastWan2.2-TI2V-5B on one `trn2.48xlarge` instance, explains the stage configuration and parallel
layout, and shows how to verify accuracy on your own deployment.

**What you will learn:**

- How the TI2V-5B and DMD2 pipelines map onto vLLM Omni stages.
- How to lay the model out over 32 NeuronCores (TP=4 x CP=4 x CFG-parallel 2), or 16 (TP=4 x CP=4).
- How to generate at the native resolution and measure warm latency.
- How to run the three accuracy tiers.

## Step 1: Set up your environment

Prepare the instance with the [setup guide](../getting-started/setup-guide.md). Restrict the
process to the cores you want to use with `NEURON_VISIBLE_DEVICES` (a comma list); the stage
`devices` field indexes into it:

```bash
export NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset NEURON_RT_VISIBLE_CORES   # the multi-process worker refuses it
```

## Step 2: Download the model (optional)

The runner downloads from Hugging Face on first use. To pre-fetch:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
huggingface-cli download FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers --local-dir ./FastWan2.2-TI2V-5B
```

## Step 3: Review the stage configuration

Both checkpoints use one diffusion stage. The pipeline class comes from the checkpoint's
`model_index.json`: `WanPipeline` (TI2V-5B) maps to the Neuron Wan2.2 pipeline, and
`WanDMDPipeline` (FastWan) maps to the DMD2 sampler, which runs the three distilled timesteps
(1000, 757, 522) without CFG, re-noising the predicted clean latent between steps.

```yaml
# examples/wan2_2/wan22_ti2v_stage.yaml (excerpt; wan22_ti2v_stage_16c.yaml drops cfg_parallel_size)
runtime:
  devices: "0,1,2,...,31"
engine_args:
  model_class_name: Wan22Pipeline      # WanDMDPipeline in wan22_dmd2_stage.yaml
  dtype: bfloat16
  flow_shift: 5.0
  vae_use_tiling: true
  model_config:
    tp_sequence_parallel: true         # Megatron SP inside the TP group
  parallel_config:
    tensor_parallel_size: 4            # inside one chip: 24 heads -> 6 per rank
    ring_degree: 4                     # context parallelism across the 4 chips
    cfg_parallel_size: 2               # cond / uncond branches at the same time, one row each
    vae_patch_parallel_size: 32        # VAE decode tiles over all 32 ranks
```

TP sits inside each chip, where the per-layer all-reduces are cheapest; CP spans the four chips of
one row. CP requires the token count to split evenly, so a shape whose count does not divide by 4
is padded and the pad tokens are masked out of attention. CP self-attention all-gathers K/V and
runs flash attention with the true row maximum; see the model card's Full-size gate section for why
the ring-attention kernel is opt-in (`WAN22_CP_RING_ATTENTION=1`).

## Step 4: Run inference

### Text-to-video with TI2V-5B (native 1280x704)

```bash
python examples/wan2_2/run_ti2v.py \
    --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --height 704 --width 1280 --num-frames 121 --steps 50 --guidance-scale 5 \
    --repeat 2 --output ti2v_704p.mp4
```

The final `SUMMARY` line lists both latencies: the first includes compilation, the second is warm
(about 68 s on 16 cores with `--stage-config examples/wan2_2/wan22_ti2v_stage_16c.yaml`; about 23.6 s
on all 64 cores with `--stage-config examples/wan2_2/wan22_ti2v_stage_64c.yaml`).

### Few-step text-to-video with FastWan DMD2

```bash
python examples/wan2_2/run_ti2v.py \
    --model-path FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
    --height 480 --width 832 --num-frames 81 --repeat 2 --output fastwan_480p.mp4
```

Step count, guidance and negative prompt are fixed by the distillation and ignored if passed.

### Stage timing

Set `WORKLOAD_OUTPUT_RW=<dir>` to write `<dir>/metrics/pipeline_perf_metrics.json` with text
encode, denoise and VAE decode seconds per request. Each stage is timed to its device completion.

## Step 5: Check accuracy on your deployment

The tests under `test/neuron/` implement the three tiers of
[Evaluating and debugging model accuracy](../model-dev/accuracy-evaluation-debugging.md):

```bash
# Tier 1: one DiT call (positive, unconditional) and the CFG combine, three-way vs diffusers
WAN22_MODEL=./Wan2.2-TI2V-5B-Diffusers pytest -s test/neuron/test_wan2_2_dit_accuracy.py
# Tier 2: one full denoising step through the served pipeline vs diffusers WanPipeline
WAN22_MODEL=./Wan2.2-TI2V-5B-Diffusers WAN22_GUIDANCE=5 \
    pytest -s test/neuron/test_wan2_2_ti2v_pipeline_accuracy.py
# Tier 3: create a golden, review the video, then regression-check against it
WAN22_MODEL=./FastWan2.2-TI2V-5B python test/neuron/test_wan2_2_ti2v_e2e_accuracy.py --regen golden.npy
WAN22_MODEL=./FastWan2.2-TI2V-5B WAN22_GOLDEN=golden.npy pytest -s test/neuron/test_wan2_2_ti2v_e2e_accuracy.py
```

For a step-level investigation, `run_ti2v.py --dump-noise-pred <file.pt>` saves the first DiT
calls' inputs and outputs (and the first CFG combine) from rank 0, and `--latents-in <noise.pt>`
injects one saved initial latent so two runs start from identical noise.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Use `NEURON_VISIBLE_DEVICES` and the stage `devices` list. |
| `Allocation Failure` while loading | The layout has too few cores for the shape; use 16 cores or a smaller resolution. |
| Sequence-length error with CP | Not expected any more: non-divisible sequences are padded. Check that you run this plugin version. |
| Soft late frames at 832x480 | Generate at the native 1280x704. |
| Image-to-video first request is slow | The condition frame is encoded on the host (about 2-4 s) and the first request also compiles; see the model card. |

## Conclusion

You deployed Wan2.2-TI2V-5B and FastWan DMD2 on Trainium2 (16 to 64 cores), generated at the native
resolution, timed each stage, and ran the accuracy tiers.

## Next steps

- [Wan2.2-TI2V-5B model card](../models/wan22-ti2v-5b.md)
- [Quickstart: Offline generation with Wan2.2-TI2V-5B](../getting-started/quickstart-offline-serving-wan22-ti2v-5b.md)
- [Optimizing high-quality offline video generation](../model-dev/optimizing-offline-video-generation.md)
