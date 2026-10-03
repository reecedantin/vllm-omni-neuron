# Tutorial: Deploy Cosmos3-Edge with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving NVIDIA Cosmos3-Edge on AWS Inferentia2 with the vLLM Omni
Neuron plugin: environment, model download, stage configuration, offline generation for all six modalities, online
serving, the optional step cache, and troubleshooting. -->
<!-- meta: keywords: Cosmos3-Edge, tutorial, vLLM Omni, Neuron, Inferentia2, inf2, text-to-image, image-to-video,
robot policy, world model, CFG parallelism, VAE patch parallelism -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

In this tutorial you deploy NVIDIA Cosmos3-Edge on an inf2.8xlarge (one Inferentia2 chip, two NeuronCore-v2). You run
all six modalities offline and serve text-to-image over HTTP. Budget about an hour: most of it is the one-time graph
compilation for each new shape.

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). Then set the NeuronCore-v2 configuration:

```bash
export NEURON_LOGICAL_NC_CONFIG=1
export VLLM_NEURON_BACKEND=neuron_native VLLM_NEURON_LIBTORCH_NEURONX_LITE=1 VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
```

## Step 2: Download the model (optional)

```bash
huggingface-cli download nvidia/Cosmos3-Edge --local-dir ~/models/Cosmos3-Edge
export COSMOS3_EDGE_WEIGHTS=~/models/Cosmos3-Edge   # run.py's default is the HF repo id
```

## Step 3: Review the stage configuration

```yaml
# examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml
stage_args:
- engine_args:
    model_class_name: Cosmos3OmniPipeline
    dtype: bfloat16
    model_config:
      compile_vae_encoder: true   # I2V and the action modes encode a conditioning image or video
      guardrails: false
    parallel_config:
      cfg_parallel_size: 2        # conditional / unconditional pass on separate cores
      tensor_parallel_size: 1
      vae_patch_parallel_size: 2  # the VAE decode is split into 2 tiles, one per core
  runtime:
    devices: 0,1
```

CFG parallelism beats TP=2 on inf2 because each attention call no longer needs a cross-core reduction: 176.9 s against
206.7 s for 480p/121-frame I2V. On NeuronCore-v2 the GEN tower's attention runs the NC-v2 NKI kernel
(`nki_attention_nc2.py`), which `nc_generation.py` selects automatically.

## Step 4: Run inference

All examples use `FAST=examples/cosmos3_edge/cosmos3_edge_stage_inf2_fast.yaml`.

### Text-to-image and text-to-video

```bash
python examples/cosmos3_edge/run.py --mode t2i --stage-config $FAST --output edge_t2i.png
python examples/cosmos3_edge/run.py --mode t2v --height 480 --width 832 --num-frames 121 \
  --prompt "A robot arm stacks two red cubes on a wooden table" --stage-config $FAST --output edge_t2v.mp4
```

### Image-to-video

```bash
python examples/cosmos3_edge/run.py --mode i2v --image frame0.png --height 480 --width 832 --num-frames 121 \
  --stage-config $FAST --output edge_i2v.mp4
```

### Robot world-model modes

```bash
# Bridge (WidowX) embodiment settings, as used by NVIDIA's golden tests
BRIDGE="--domain bridge_orig_lerobot --raw-action-dim 10 --action-chunk 16 --action-fps 5 --fps 5 \
  --height 544 --width 736 --num-frames 17 --resolution 480"

# policy: image + instruction -> actions (+ predicted video unless --action-only)
python examples/cosmos3_edge/run.py --mode policy --image view.png --prompt "Put the pot to the left of the purple item." \
  $BRIDGE --stage-config $FAST --output policy

# forward dynamics: image + actions (JSON, chunk x raw_action_dim) -> video
python examples/cosmos3_edge/run.py --mode forward_dynamics --image view.png --actions actions.json \
  $BRIDGE --stage-config $FAST --output fwd

# inverse dynamics: video -> actions
python examples/cosmos3_edge/run.py --mode inverse_dynamics --video clip.mp4 --action-only \
  $BRIDGE --stage-config $FAST --output inv
```

`--action-only` (equivalent to `extra_args: {"action_only": true}`) skips the VAE video decode. The actions are
bit-identical to a full run. Policy drops from 11.3 s to 7.1 s, and inverse dynamics from 13.0 s to 8.0 s.

### Online serving (text-to-image)

```bash
vllm serve nvidia/Cosmos3-Edge --omni --host 127.0.0.1 --port 8091 \
  --stage-configs-path examples/cosmos3_edge/cosmos3_edge_stage.yaml --no-guardrails
python examples/cosmos3_edge/t2i_request.py --port 8091 --out served_t2i.png
```

The client posts to `/v1/images/generations`. The server has no authentication, so bind it to localhost or put it
behind your own access control.

## Optional: step cache

```bash
export COSMOS3_EDGE_STEP_CACHE=1e9 COSMOS3_EDGE_STEP_CACHE_MAX_SKIP=1 \
       COSMOS3_EDGE_STEP_CACHE_TMIN=100 COSMOS3_EDGE_STEP_CACHE_TMAX=900
```

This skips every other GEN forward in the middle of the trajectory. I2V goes from 122.0 s to 82.9 s, at 35.8 dB
against the full run.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A long first request | Per-shape graph compilation | Expected once per shape. Keep the compile cache directory. |
| A second identical request recompiles | A new text-length bucket or a new frame count | Use the same shapes. Prompts are padded to text buckets. |
| Out of device memory at TP=1 with large videos | 16 GB per NeuronCore-v2 | Use the fast stage (CFG parallel, VAE patch parallel), or reduce frames or resolution |

## Conclusion

You ran every Cosmos3-Edge modality on a single Inferentia2 chip and served text-to-image over HTTP.

## Next steps

- [Cosmos3-Edge model card](../models/cosmos3-edge.md): accuracy, performance, limitations
- [Optimizing High-Quality Offline Video Generation](../model-dev/optimizing-offline-video-generation.md)
