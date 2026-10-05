# Tutorial: Deploy Qwen-Image 2.1 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving Qwen-Image 2.1 on AWS
Trainium2 with the vLLM Omni Neuron plugin: environment, model download, the
tensor-parallel stage configuration, offline text-to-image generation, sampling
controls, the prefix/target DiT split, the tiled VAE, and troubleshooting. -->
<!-- meta: keywords: Qwen-Image, Qwen-Image 2.1, tutorial, vLLM Omni, Neuron,
Trainium2, trn2, text-to-image, diffusion, DiT, Qwen3-VL, tensor parallelism,
block-causal attention, tiled VAE -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: tutorial -->

In this tutorial you deploy Qwen-Image 2.1 on a `trn2.48xlarge` (16 NeuronCores at `LNC=2`, one
row of four Trainium2 chips; 8- and 4-core layouts are provided too) and generate images offline
from text prompts. Budget about ten minutes plus the one-time graph compilation for each new
output shape (a few minutes at 1024x1024).

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). The native Lite backend is enabled
by the container / env; a manual install sets:

```bash
export VLLM_NEURON_BACKEND=neuron_native VLLM_NEURON_LIBTORCH_NEURONX_LITE=1 VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
```

## Step 2: Download the model (optional)

```bash
huggingface-cli download Qwen/Qwen-Image-2.1 --local-dir ~/models/Qwen-Image-2.1
export QWEN_IMAGE21_WEIGHTS=~/models/Qwen-Image-2.1   # run.py's default is the HF repo id
```

## Step 3: Review the stage configuration

```yaml
# examples/qwen_image/qwen_image21_stage.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    final_output: true
    final_output_type: image
    runtime:
      process: true
      devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"   # one row of four trn2 chips at LNC=2
      max_batch_size: 1
    engine_args:
      model_class_name: QwenImage21Pipeline   # the registry model_arch key, auto-discovered
      model_stage: diffusion
      dtype: bfloat16
      parallel_config:
        tensor_parallel_size: 16
```

`tensor_parallel_size: 16` shards the DiT's 32 attention heads (2 per rank) and the Qwen3-VL
encoder's 32 query heads (2 per rank; each of its 8 KV heads is replicated on 2 ranks). The VAE is
not sharded: every rank holds a copy and decodes its share of the image tiles, which rank 0 blends.
The `devices` list must be a comma list of logical core indices, not a range. On fewer cores use
`qwen_image21_stage_tp8.yaml` (an adjacent chip pair) or `qwen_image21_stage_tp4.yaml` (one chip)
with `--stage-config`.

## Step 4: Generate an image

```bash
python examples/qwen_image/run.py \
  --prompt "A cozy bookshop window on a rainy evening, warm light, a hand-lettered sign that reads \"Open Late\"" \
  --output qwen_image.png --profile
```

`--profile` runs the request twice and prints the cold-compile and warm latencies. The first run for
a new shape compiles four graph families — the Qwen3-VL text encoder, the DiT prefix graph, the
per-step DiT target graph, and the VAE decode tile — and caches their NEFFs; later runs of the same
shape hit the cache.

### How a request runs

1. **Text encoder** (once): the Qwen3-VL decoder stack produces the prompt's last-layer hidden
   states. The token-embedding lookup runs on the host; the decoder runs on device, TP-sharded.
2. **DiT prefix** (once per prompt, per CFG branch): because Qwen-Image 2.1's attention is
   block-causal and its text / condition-image tokens are modulated at `t = 0`, the prefix's keys
   and values do not depend on the denoising step. This port computes them in a dedicated graph,
   bucketed by prefix token count, instead of refilling a KV cache every step.
3. **DiT target** (every denoising step): only the target image's tokens are recomputed, attending
   to the cached prefix K/V. This is the one graph that runs 40 times for a 40-step request.
4. **VAE decode** (once): the 16x-upsampling decoder runs on fixed 256px latent tiles with a
   feathered overlap, so one compiled graph serves every output resolution; the tiles are dealt
   across all TP ranks and gathered to rank 0 for the blend.

### Sampling controls

Qwen-Image 2.1 is meant to be sampled without classifier-free guidance (`--true-cfg-scale 1.0`, the
default). To enable guidance:

```bash
python examples/qwen_image/run.py \
  --prompt "A detailed botanical illustration of a sunflower" \
  --negative-prompt "blurry, low quality" --true-cfg-scale 4.0 \
  --output sunflower.png
```

Guidance runs a second DiT branch per step (its own prefix graph plus 40 target calls), roughly
doubling the denoising cost.

Override the output shape with `--height`, `--width` (both multiples of 32), and the step count with
`--steps`. More steps increase latency approximately linearly.

### Tune the VAE tiling

The VAE tile size is set by `QWEN_IMAGE_VAE_TILE="<tile>,<overlap>"` in latent pixels (default
`16,4` — 256px tiles with a 64px overlap). Smaller tiles compile faster and use less HBM but add
launches; a 1024x1024 image is 25 tiles at the default, spread over the TP ranks
(`QWEN_IMAGE_VAE_PARALLEL=0` decodes them all on rank 0 instead, 10.1 s instead of 2.9 s at TP=4).

## Performance (BF16, 1024x1024, 40 steps)

| Stage | TP=16 | TP=8 | TP=4 |
|---|---|---|---|
| Text encoder | 0.02 s | 0.02 s | 0.03 s |
| DiT prefix | 0.01 s | 0.01 s | 0.01 s |
| DiT target (40 steps) | 6.26 s | 8.73 s | 13.74 s |
| VAE decode (tiles over all ranks) | 0.89 s | 1.68 s | 2.87 s |
| **Total** | **7.2 s** | **10.5 s** | **16.7 s** |

2048x2048 runs at TP=16 (44.5 s) and TP=8 (75.3 s); TP=4 does not compile at that size (see the
model card's known limitations). See the [model card](../models/qwen-image21.md) for device
parity and the full method.

## Troubleshooting

- **`Stage 0 requires 4 device(s) ... 1 device(s) available`:** the stage `devices` field or
  `NEURON_VISIBLE_DEVICES` names fewer cores than `tensor_parallel_size`. Pass as many
  comma-separated core indices as the TP degree (16, 8 or 4), or pick the stage config that fits.
- **`NEURON_RT_VISIBLE_CORES cannot be used with multi-processing`:** vLLM Omni's multi-process
  stage refuses `NEURON_RT_VISIBLE_CORES`; set the cores through the stage `devices` field (or
  `NEURON_VISIBLE_DEVICES`) instead.
- **Very long first-run latency:** expected — the Neuron compiler builds NEFF graphs for each new
  output shape on the first request.
- **Out of HBM on a single core:** the full BF16 model (~31 GB) needs `tensor_parallel_size: 4` or more; the
  TP=1 config is for reduced-size checkpoints only.

## Known limitations

- Text-to-image only in this release: the condition-image input path, the Qwen-Image 2.1 prompt
  enhancer (PE-T2I), and Qwen-Image-Edit-2511 are not yet ported.
- One image per request (`supports_request_batch = False`); concurrent requests run serially.

## Next steps

- [Qwen-Image 2.1 model card](../models/qwen-image21.md)
- [Offline quickstart](../getting-started/quickstart-offline-serving-qwen-image21.md)
