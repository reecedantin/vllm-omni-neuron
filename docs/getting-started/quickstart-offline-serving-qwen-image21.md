# Quickstart: Offline image generation with Qwen-Image 2.1 on Neuron

<!-- meta: description: Generate an image offline from a text prompt with the
Qwen-Image 2.1 model on AWS Trainium using the vLLM Omni Neuron plugin. Covers a
quick single-core smoke test and a full 1024x1024 tensor-parallel generation run. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Qwen-Image, Qwen-Image 2.1,
text-to-image, image generation, offline, diffusion, AWS Trainium, trn2,
NeuronCores, tensor parallelism, quickstart -->
<!-- meta: date_updated: 2026-10-04 -->
<!-- meta: content_type: procedural-quickstart -->

This quickstart shows you how to generate an image from a text prompt on AWS
Trainium with the vLLM Omni Neuron plugin. When you finish, you have a PNG file
produced offline by the [Qwen-Image 2.1](../models/qwen-image21.md) text-to-image
model.

## Prerequisites

Before you start, make sure that you have the following:

- One SSH-accessible `trn2.48xlarge` instance.
- The environment prepared with either flow in the [setup guide](setup-guide.md).
  The DLC commands below use the `vllm-omni-neuron` container, with device
  access, persistent caches, and `--shm-size=2g`.
- Network access to download the Qwen-Image 2.1 weights
  (`Qwen/Qwen-Image-2.1`) from Hugging Face.

Run the following commands in an SSH session on the instance host. Each command uses
`docker exec` to run the example script inside the local `vllm-omni-neuron` container.

For manual installation, run the Python commands in the activated environment,
omit `sudo docker exec vllm-omni-neuron`, and replace `/workspace` with
`$VLLM_OMNI_HOME`.

> **Note:** The script uses the vLLM Omni entrypoint (`Omni`), not the
> `vllm.LLM` API. It reads its configuration from
> `/workspace/plugin/examples/qwen_image/qwen_image21_stage.yaml`
> (tensor-parallel over 16 NeuronCores; `qwen_image21_stage_tp8.yaml` and
> `qwen_image21_stage_tp4.yaml` use 8 and 4).

## Step 1: Run a full offline generation

To generate the default image, run the script with an output path. The default run
produces a 1024x1024 image over 40 denoising steps on 16 NeuronCores.

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/qwen_image/run.py \
  --output /workspace/output/qwen_image.png
```

The command writes the image to `/workspace/output/qwen_image.png`, which is in
the host-mounted output directory created by the setup guide.

To set your own prompt and output file, add the `--prompt` and `--output` flags:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/qwen_image/run.py \
  --prompt "A red hot air balloon rising over green hills at sunrise" \
  --output /workspace/output/my_image.png
```

### Change the output shape and sampling controls

`examples/qwen_image/run.py` sets the default height, width, and number of denoising
steps. Override them with `--height`, `--width`, and `--steps`:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/qwen_image/run.py \
  --height 1328 --width 1328 --steps 40 \
  --output /workspace/output/qwen_1328.png
```

Height and width must be multiples of 32 (the VAE's 16x downsampling times the DiT's
2x packing); the runner rounds down to the nearest multiple otherwise. Changing the
output shape triggers compilation for that shape.

Qwen-Image 2.1 is designed to be sampled **without** classifier-free guidance
(`--true-cfg-scale 1.0`, the default). To enable guidance, pass a scale above 1 and a
negative prompt:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/qwen_image/run.py \
  --prompt "A detailed botanical illustration of a sunflower" \
  --negative-prompt "blurry, low quality" \
  --true-cfg-scale 4.0 \
  --output /workspace/output/sunflower.png
```

### Run a single-core smoke test

For bring-up on a small checkpoint, the single-core stage config
([`qwen_image21_stage_tp1.yaml`](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage_tp1.yaml))
runs the whole pipeline on one NeuronCore. The full BF16 model (7B DiT + 8B text
encoder, ~31 GB) does not fit one `LNC=2` core's ~24 GB HBM, so use this only with a
reduced-size checkpoint:

```bash
sudo docker exec vllm-omni-neuron python /workspace/plugin/examples/qwen_image/run.py \
  --stage-config /workspace/plugin/examples/qwen_image/qwen_image21_stage_tp1.yaml \
  --model-path <small checkpoint> \
  --height 256 --width 256 --steps 8 \
  --output /workspace/output/smoke.png
```

To copy the image to the host, use `docker cp`:

```bash
sudo docker cp vllm-omni-neuron:/workspace/output/qwen_image.png .
```

> **Note:** The first run for a new output shape is slow because Neuron compiles
> shape-specific NEFF graphs and loads the model weights. Measured cold-compile
> latency at 1024x1024 / 40 steps / TP=4 was ~445 s; the warm per-image latency
> after the cache is built is 7.2 s at TP=16 (10.5 s at TP=8, 16.7 s at TP=4).

## Common issues

- **Silent crash or `SIGBUS` on output:** The container `/dev/shm` is too small.
  Start the container with `--shm-size=2g`, as shown in the setup guide.
- **`NRT_FAILURE` in `nrt_init()`, or Neuron cores not available:** The host
  driver or device access is incompatible with the runtime. Follow the host
  prerequisites and device checks in the setup guide.
- **Very long first-run latency:** The Neuron compiler builds NEFF graphs on the
  first run for each new output shape. This is expected.
- **Weights download fails:** The run cannot reach the Qwen-Image 2.1 weights.
  Confirm network access and Hugging Face permission for `Qwen/Qwen-Image-2.1`.

## Clean up

To stop and remove the container, run the following commands:

```bash
sudo docker stop vllm-omni-neuron
sudo docker rm vllm-omni-neuron
```

## Next steps

- [Tutorial: Deploy Qwen-Image 2.1 with vLLM Omni Neuron](../tutorials/tutorial-qwen-image21.md) —
  the stage config, parallelism, and sampling controls in depth.
- [Qwen-Image 2.1 model card](../models/qwen-image21.md) — architecture, feature
  support, device parity, and performance.
- [Entrypoint script (`run.py`)](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/run.py) —
  the offline entrypoint and its flags.
- [Stage config (`qwen_image21_stage.yaml`)](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/qwen_image/qwen_image21_stage.yaml) —
  the configuration the run reads.
