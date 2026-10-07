# Tutorial: Deploy LTX-2.5 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving Lightricks LTX-2.5 (distilled text-to-video with
synchronized audio) on AWS Trainium2 with the vLLM Omni Neuron plugin: environment, model download, stage
configuration, offline inference, performance switches and troubleshooting. -->
<!-- meta: keywords: LTX-2.5, tutorial, vLLM Omni, Neuron, trn2, text-to-video, audio -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

You will generate 512x768, 121-frame videos with 48 kHz stereo audio from text prompts on one Trainium2 chip
(four logical NeuronCores). Plan on ~25 minutes for the first request at a new resolution (transformer and VAE
compilation) and about one minute per warm request.

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). The stage runs four worker processes, so select
devices with `NEURON_VISIBLE_DEVICES` (comma list) rather than `NEURON_RT_VISIBLE_CORES`:

```bash
unset NEURON_RT_VISIBLE_CORES
export NEURON_VISIBLE_DEVICES=0,1,2,3
export TORCH_NEURONX_NEFF_CACHE_DIR=$HOME/neff-cache   # keep compiled graphs across runs
```

## Step 2: Download the model (optional)

The repository is gated (LTX-2.x Community License). Accept it on the model page, log in, then:

```bash
hf download Lightricks/LTX-2.5-Diffusers --local-dir ltx25 \
    --exclude "prompt_enhancer/*" "transformer_full/*" "diffusion_decoder/*" "*latent_upsampler/*" "*lora*"
```

The distilled pipeline uses `transformer/`, `text_encoder/`, `connectors/`, `vae/`, `audio_vae/`, `vocoder/`,
`tokenizer/` and `scheduler/`.

## Step 3: Review the stage configuration

```yaml
# examples/ltx2/ltx2_stage.yaml (excerpt)
runtime:
  devices: "0,1,2,3"            # logical indices into NEURON_VISIBLE_DEVICES
engine_args:
  model_class_name: LTX2Pipeline
  model_config:
    blocks_per_graph: 4          # 48 DiT blocks run as 12 calls of one compiled 4-block graph
  parallel_config:
    tensor_parallel_size: 4
```

How the work is placed:

- **Transformer (19B):** tensor-parallel over the four cores, 8.7 GB of weights per core. Timestep and RoPE
  conditioning is computed on the host in fp32 and passed in.
- **Video VAE decode:** on core 0, as spatial tiles of one fixed shape (16x8 latents, 4 latents of overlap)
  through a single compiled graph. The full-frame graph exceeds the compiler's instruction limit.
- **Text encoder (Gemma) and vocoder:** on the host, on rank 0 only, multi-threaded. The prompt embedding is
  computed once on rank 0, broadcast to the other ranks and cached, so a repeated prompt skips the encoder.

## Step 4: Run inference

### Text-to-video with audio

```bash
python examples/ltx2/run.py --height 512 --width 768 --num-frames 121 --steps 8 \
    --prompt "A red fox walking through a snowy forest at dawn, the camera tracking alongside." \
    --output fox.mp4 --profile
```

### A quick shape check

```bash
python examples/ltx2/run.py --height 64 --width 96 --num-frames 9 --steps 2 --output smoke.mp4
```

## Optional: performance switches

| Switch | Default | Effect |
|---|---|---|
| `LTX2_VAE_DEVICE` | `1` | `0` decodes on the host (reference path, ~5 min at 512x768x121) |
| `LTX25_PROMPT_CACHE` | `1` | `0` re-encodes every prompt (text encoder and connectors) |
| `LTX25_TEXT_ENCODER_DEVICE` | `1` | `0` runs the Gemma text encoder on the host (rank 0) instead of the NeuronCores |
| `LTX25_TEXT_BUCKETS` | `128,256,512,1024` | Prompt-length buckets of the device text encoder (one compiled graph each) |
| `LTX2_VAE_PARALLEL` | `1` | `0` decodes every VAE tile on rank 0's core |
| `LTX25_VOCODER_PARALLEL` / `LTX25_VOCODER_SPANS` | `1` / `4` | Vocoder as time spans on that many ranks' host CPUs |
| `LTX25_OVERLAP_AUDIO` | `1` | `0` runs the audio decode after the video decode instead of alongside it |
| `LTX2_CP_RING` | `0` | `1` uses ring attention for the context-parallel video self-attention |
| `LTX25_EMB_CACHE_DIR` | unset | Persist prompt embeddings on disk across processes |
| `LTX25_HOST_THREADS` | `24` | Torch threads for the host stages (text connectors, audio VAE, vocoder) |
| `LTX25_HOST_THREADS_ADAPTIVE` | unset | `1` caps `LTX25_HOST_THREADS` at the idle cores (faster on a loaded host; output then depends on the load) |
| `LTX25_TIME_STAGES` | unset | `1` prints per-stage timings for each request |

For four or more chips use `examples/ltx2/ltx2_stage_tp4cp4.yaml` (TP=4 x CP=4, 16 cores) and set
`NEURON_VISIBLE_DEVICES` to 16 cores; see the model card for the measured layouts.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution` | Multi-process stage | Unset it, set `NEURON_VISIBLE_DEVICES` |
| `NCC_EVRF007 ... exceeds the typical limit` while compiling the VAE | Tile too large | Keep the default tile (`LTX2_VAE_TILE_W=8`) |
| Host stages much slower than expected | Host CPU oversubscribed | Lower `LTX25_HOST_THREADS`, or set `LTX25_HOST_THREADS_ADAPTIVE=1` to cap it at the free cores |
| A new `--num-frames` recompiles the VAE | The latent frame count is part of the tile shape | Pre-compile the frame counts you serve |

## Conclusion

You served LTX-2.5 text-to-video with audio on one Trainium2 chip, with the transformer sharded across four
cores and the VAE decoded on device.

## Next steps

- [LTX-2.5 model card](../models/ltx2-5.md)
- [Quickstart: Offline text-to-video with LTX-2.5](../getting-started/quickstart-offline-serving-ltx2-5.md)
