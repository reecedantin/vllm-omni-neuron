# Quickstart: Offline video generation with Wan2.2-TI2V-5B on Neuron

<!-- meta: description: Generate a video offline with Wan2.2-TI2V-5B or its 3-step DMD2 distillation
FastWan2.2-TI2V-5B on AWS Trainium2 using the vLLM Omni Neuron plugin. Covers a quick smoke test and
a full run at the native resolution. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Wan2.2, Wan2.2-TI2V-5B, FastWan, DMD2,
text-to-video, offline, trn2, quickstart -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate a video on Trainium2 with the vLLM Omni Neuron plugin.
When you finish, you have an MP4 produced offline by the
[Wan2.2-TI2V-5B](../models/wan22-ti2v-5b.md) model, or by its 3-step DMD2 distillation
FastWan2.2-TI2V-5B.

## Prerequisites

- One SSH-accessible `trn2.48xlarge` instance. The recommended configurations use the whole instance
  (64 logical NeuronCores; `wan22_ti2v_stage_64c.yaml`, `wan22_ti2v_i2v_stage_64c.yaml`,
  `wan22_dmd2_stage_64c.yaml`). The default stage configs use 32 cores for TI2V-5B (two rows of four
  chips, CFG-parallel) and 16 for FastWan DMD2; 16 cores (one row) also run every shape, and a single
  chip (4 cores) runs smaller shapes.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
- Network access to download the weights (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, about 32 GB, or
  `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`, about 23 GB) from Hugging Face. Neither
  checkpoint is gated.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`). It reads its configuration from
> `examples/wan2_2/wan22_ti2v_stage.yaml` (or `wan22_dmd2_stage.yaml` for the DMD2 checkpoint,
> chosen from the checkpoint's `model_index.json`). Command-line flags override the parallel layout
> without editing the YAML.

## Step 1: Run a quick smoke test

FastWan DMD2 is the fastest path (3 steps, no CFG). The `--dev` preset generates 448x256 with 17
frames:

```bash
python examples/wan2_2/run_ti2v.py \
    --model-path FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
    --dev --output fastwan_dev.mp4
```

Expected: `Saved ... fastwan_dev.mp4` and a final `SUMMARY {...}` line with `"finite": true`. The
first run compiles the NEFFs for this shape (several minutes); later runs reuse the cache.

## Step 2: Run a full offline generation

Wan2.2-TI2V-5B at its native 1280x704, 121 frames, 50 steps, guidance 5:

```bash
python examples/wan2_2/run_ti2v.py \
    --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --height 704 --width 1280 --num-frames 121 --steps 50 --guidance-scale 5 \
    --output ti2v_704p.mp4
```

On all 64 cores (`--stage-config examples/wan2_2/wan22_ti2v_stage_64c.yaml`) a warm request takes
about 23.6 s; on 16 cores (`--stage-config examples/wan2_2/wan22_ti2v_stage_16c.yaml`) about 68 s. The
32-core default `wan22_ti2v_stage.yaml` has not been re-timed since the attention change. The first
request also compiles
(about 3-10 minutes).
Add `--repeat 2` to print the warm latency of a second request.

### Change the output shape and sampling controls

| Flag | Default | Meaning |
|---|---|---|
| `--height` / `--width` / `--num-frames` | 704 / 1280 / 121 (`--dev`: 256 / 448 / 17) | Output shape. Height and width are rounded down to multiples of 32. |
| `--steps` | 50 (DMD2: 3, fixed) | Denoising steps. |
| `--guidance-scale` | 5.0 (DMD2: ignored) | Classifier-free guidance. |
| `--prompt` / `--negative-prompt` | a cat on a garden path / none | Conditioning text. |
| `--seed` | 42 | Initial-noise seed. |
| `--tp` / `--cp` / `--cfg-parallel` / `--devices` | 4 / 4 / 2 / `0,...,31` (DMD2: 4 / 4 / 1 / `0,...,15`) | Tensor and context parallelism and the cores used (indices into `NEURON_VISIBLE_DEVICES`). Use `--tp 4 --cp 1 --devices 0,1,2,3` for one chip. |
| `--latents` | off | Return the final latents (no VAE decode) as `<output>.latents.pt`. |
| `--save-frames` | off | Also save the decoded frames as `<output>.npy`. |

## Common issues

| Symptom | Fix |
|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Unset `NEURON_RT_VISIBLE_CORES` and select cores with `NEURON_VISIBLE_DEVICES` plus `--devices` (the runner converts a `NEURON_RT_VISIBLE_CORES` pin automatically). |
| `Allocation Failure` during load | Too few cores for the shape: use the 16-core layout, or a smaller resolution on one chip. |
| The first request appears to hang | It is compiling; a cold 704p compile takes several minutes. Watch for `neuronx-cc` processes. |
| Soft frames at 832x480 | TI2V-5B is trained at 1280x704; generate at the native resolution. |

## Clean up

Delete the NEFF cache directory set by `NEURON_COMPILE_CACHE_URL` / `TORCH_NEURONX_NEFF_CACHE_DIR`
and the generated `.mp4` / `.npy` files when you no longer need them.

## Next steps

- [Wan2.2-TI2V-5B model card](../models/wan22-ti2v-5b.md)
- [Tutorial: Deploy Wan2.2-TI2V-5B and FastWan DMD2](../tutorials/tutorial-wan22-ti2v-5b.md)
