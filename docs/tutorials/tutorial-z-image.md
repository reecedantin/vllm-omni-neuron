# Tutorial: Deploy Z-Image with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving Z-Image and Z-Image-Turbo on AWS Trainium2 with the vLLM Omni
Neuron plugin: environment, model download, stage configuration, offline text-to-image, accuracy checks and
troubleshooting. -->
<!-- meta: keywords: Z-Image, Z-Image-Turbo, tutorial, vLLM Omni, Neuron, Trainium2, trn2, text-to-image,
tensor parallelism, classifier-free guidance -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: tutorial -->

In this tutorial you deploy Z-Image-Turbo and Z-Image on a trn2.48xlarge, using four logical NeuronCores (one chip). You
generate 1024 x 1024 images offline and run the accuracy checks. Budget about 30 minutes: most of it is the one-time
graph compilation for each new resolution.

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). Trn2 runs with two physical cores per logical core:

```bash
export NEURON_LOGICAL_NC_CONFIG=2
```

vLLM Omni starts its diffusion workers with multiprocessing, which does not accept `NEURON_RT_VISIBLE_CORES`. To pin
the stage to specific cores, set `NEURON_VISIBLE_DEVICES` to a comma-separated core list and unset
`NEURON_RT_VISIBLE_CORES`; the stage config's `devices` field then indexes into that list.

## Step 2: Download the model (optional)

```bash
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir ~/models/Z-Image-Turbo
huggingface-cli download Tongyi-MAI/Z-Image --local-dir ~/models/Z-Image
```

Both checkpoints are Apache-2.0 and not gated. They share the architecture (6B single-stream DiT, Qwen3-4B text
encoder, 16-channel Flux VAE) and differ in sampling: Turbo is guidance-distilled (9 scheduler steps, no CFG), the
base model uses CFG (28 to 50 steps, guidance 3 to 5).

## Step 3: Review the stage configuration

```yaml
# examples/z_image/z_image_stage.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      devices: "0,1,2,3"
      max_batch_size: 1
    engine_args:
      model_class_name: ZImagePipeline
      dtype: bfloat16
      engine_backend: vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image.ZImageDiffusionEngine
      model_config:
        block_split: 5
        text_encoder_on_host: false
        vae_tile_lat: 64
        dummy_run_height: 1024
        dummy_run_width: 1024
      parallel_config:
        tensor_parallel_size: 2
        cfg_parallel_size: 2
```

- `tensor_parallel_size: 2` shards the DiT's 30 attention heads across two cores. At 1024 x 1024 one logical core
  cannot load the served DiT graph.
- `cfg_parallel_size: 2` runs the two classifier-free-guidance branches of Z-Image at the same time, each on its own
  TP 2 group, so the stage uses four cores. Z-Image-Turbo has no guidance branch: serve it with
  `examples/z_image/z_image_stage_tp4.yaml` (TP 4 on the same four cores) instead. The model card lists the other
  layouts (`z_image_stage_tp4_cfg2.yaml` and `z_image_stage_tp8.yaml` on eight cores are the fastest for Z-Image
  and Z-Image-Turbo).
- `block_split: 5` compiles one 5-block graph and replays it for the 30 main blocks, which bounds compile time and
  compiler host memory.
- `text_encoder_on_host: false` runs the Qwen3 encoder on the NeuronCores, tensor-parallel, with an fp32 residual
  stream and fp32 norms around bf16 matmuls (about 0.02 s per new prompt; repeated prompts are cached). `true` runs
  it on the CPU in bf16 instead.
- `vae_tile_lat: 64` decodes the latent in 512 x 512 pixel tiles with 1/8 overlap, blended on the host, with the
  tiles dealt across the stage's cores. The decoder runs in fp32, as the VAE config's `force_upcast` asks.
- `engine_backend` and `dummy_run_height` / `dummy_run_width` make the engine's warmup request use the served
  resolution, so the warmup compiles the same graphs production uses.

## Step 4: Run inference

### Text-to-image with Z-Image-Turbo

```bash
python examples/z_image/run.py --model-path ~/models/Z-Image-Turbo --height 1024 --width 1024 \
    --steps 9 --guidance-scale 0 --stage-config examples/z_image/z_image_stage_tp4.yaml --profile --output turbo.png
```

### Text-to-image with Z-Image (CFG)

```bash
python examples/z_image/run.py --model-path ~/models/Z-Image --height 1024 --width 1024 \
    --steps 50 --guidance-scale 4 --negative-prompt "blurry, low quality" --profile --output base.png
```

`--profile` sends the request again after the first one and prints the warm latency (`--warm-runs 3` prints the
median of three): about 6 s for Turbo and 40 s for the base model at 1024 x 1024.

### Online serving

`vllm serve <model> --omni --stage-configs-path examples/z_image/z_image_stage.yaml` uses the same stage config. It has
not been validated for Z-Image yet; the offline entrypoint above is the tested path.

## Optional: check accuracy on your instance

`examples/z_image/device_check.py` runs the same generation on the NeuronCore and as a pure-diffusers fp32 run on the
CPU, and reports the final-latent rel-L2, cosine and image PSNR. `examples/z_image/bf16_cpu_floor.py` measures the CPU
bf16 error against the same fp32 latent, the floor the Neuron run is judged against:

```bash
python examples/z_image/device_check.py --model ~/models/Z-Image --height 512 --width 512 --steps 20 \
    --guidance 4 --te-on-host --out ./check
python examples/z_image/bf16_cpu_floor.py --model ~/models/Z-Image --height 512 --width 512 --steps 20 \
    --guidance 4 --ref-latent ./check/oracle_latent.pt --out ./check/floor
```

With CFG 4 a single Z-Image base sample can land far from fp32 on either bf16 path, the CPU one included: the
trajectory is chaotic and may settle on a different composition. Compare against the CPU bf16 floor over several
seeds rather than one; see [the model card](../models/z-image.md#accuracy-evaluation).

The device tests in `test/neuron/test_z_image_accuracy.py` cover components, a single denoising step and the full
generation against a CPU reference. Run one test per process. The end-to-end base tests need about 2.5 hours of CPU
reference runs the first time; set `Z_IMAGE_REF_DIR` to a persistent directory to cache them.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not load the model status=4 message=Allocation Failure` | The DiT graph does not fit one logical core at 1024 px | Keep `tensor_parallel_size` at 2 or more (one of the shipped stage configs) |
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | vLLM Omni multiprocessing | Use `NEURON_VISIBLE_DEVICES` and unset `NEURON_RT_VISIBLE_CORES` |
| `Logical devices must be non-negative integers` | `devices` given as a range (`"0-1"`) | Use a comma list (`"0,1"`) |
| Visible grid lines at 512 px boundaries | VAE tile overlap set to 0 | Leave `Z_IMAGE_VAE_TILE_OVERLAP` at its default (0.125) |
| First request times out | Cold compile | Raise `Z_IMAGE_HANDSHAKE_TIMEOUT_S` (default 3600) |

## Conclusion

You served Z-Image-Turbo and Z-Image at 1024 x 1024 on two Trainium2 logical cores from the offline entrypoint, and
checked the output against a CPU fp32 reference.

## Next steps

- [Z-Image model card](../models/z-image.md)
- [Quickstart: Offline generation with Z-Image](../getting-started/quickstart-offline-serving-z-image.md)
