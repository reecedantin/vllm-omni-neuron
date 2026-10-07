# Tutorial: Deploy FastH3 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for generating video with audio from FastH3 (MiniMax-H3 distilled) on
AWS Trainium2 with the vLLM Omni Neuron plugin: environment, model download, stage configuration, offline inference,
accuracy checks, troubleshooting. -->
<!-- meta: keywords: FastH3, MiniMax-H3, tutorial, vLLM Omni, Neuron, trn2, text-to-video, text-to-audio -->
<!-- meta: date_updated: 2026-10-05 -->
<!-- meta: content_type: tutorial -->

You will generate 5-second videos with a stereo soundtrack from FastH3 on a trn2.48xlarge, using all 16 Trainium2
chips with tensor and context parallelism. Expect about an hour, most of it the model download and the first-request
compiles.

## Step 1: Set up your environment

Follow the [setup guide](../getting-started/setup-guide.md). The run script imports `vllm_omni_neuron.bootstrap`
first and converts an inherited `NEURON_RT_VISIBLE_CORES` into `NEURON_VISIBLE_DEVICES`, which vLLM Omni's
multiprocess workers expect.

## Step 2: Download the model (optional)

```bash
huggingface-cli download FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree --local-dir fasth3-4step
huggingface-cli download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir fasth3-8step-v2
```

The FastH3 releases ship `modular_model_index.json` (diffusers modular pipelines) rather than `model_index.json`;
the plugin reads either.

## Step 3: Review the stage configuration

```yaml
# examples/minimax_h3/minimax_h3_stage.yaml (excerpt)
    runtime:
      devices: "0,1,2,...,63"   # 64 logical cores, written out as a comma list
    engine_args:
      model_class_name: MiniMaxH3ModularPipeline
      dtype: bfloat16
      model_config:
        adaln: device
        text_encoder: device
      parallel_config:
        tensor_parallel_size: 8
        ring_degree: 8
```

- **TP=8** splits the 56 attention heads 7 per rank and the 14336-wide FFN 1792 per rank. TP must divide both.
- **`ring_degree: 8`** is the context-parallel degree: each of 8 TP groups holds an eighth of the packed sequence,
  and K/V are all-gathered per layer. The sequence is padded to a multiple of the degree, so any degree works; the
  model card measured 2, 4 and 8. `minimax_h3_stage_tp8cp4.yaml` (32 cores), `minimax_h3_stage_tp8cp2.yaml`
  (16 cores) and `minimax_h3_stage_tp8.yaml` (8 cores) are the smaller layouts.
- **`devices`** is a comma list of logical indices into the visible cores, never a range.
- **`model_config`** keys:
  - `adaln`: `device` (default) or `host` (modulation tables computed on the host);
  - `text_encoder`: `device` (default; TP over the stage's TP group, prompt-length buckets) or `host`;
  - `prompt_cache_size`: prompts whose embeddings are kept (default 16);
  - `video_output`: `uint8` (default, 8-bit pixels) or `float` (`[0, 1]`);
  - `audio_vae`: `device` (default when the video VAE is tile-parallel: fixed time windows on the stage's last
    ranks) or `host` (one rank's host CPU, overlapped with the video decode; the default otherwise);
  - `vae`: `device` (default) or `cpu`;
  - `vae_tile_parallel`: default on when the stage has more than one rank;
  - `schedule`: `auto`, `contract` or `linspace`;
  - `vsa`: `auto`, or `off` to run a VSA checkpoint densely, for diagnostics only.

## Step 4: Run inference

### Text-to-video-and-audio, 4-step

```bash
python examples/minimax_h3/run.py --model-path fasth3-4step --height 768 --width 1344 --num-frames 124 \
  --prompt "A golden retriever runs through the surf at sunset." --profile --repeat 3 --output fasth3.mp4
```

The JSON next to the MP4 reports the first and warm per-stage timings: text encode, each DiT forward, video decode,
audio decode, and the number of video-VAE tile calls.

### Text-to-video-and-audio, 8-Step-V2

```bash
python examples/minimax_h3/run.py --model-path fasth3-8step-v2 --steps 9 --cp 2 --height 384 --width 640 \
  --num-frames 124 --output fasth3v2.mp4
```

`--cp 2` runs it at TP=8 × CP=2 on 16 cores: its sparse-attention graph does not compile at every context-parallel
slice (see the model card's known limitations).

8-Step-V2 runs its trained sparse attention: per head and query tile, the top 20% of video key tiles plus all text
and audio keys. It also runs its trained step ladder and video shift 10.

### Check accuracy against a CPU reference

```bash
python examples/minimax_h3/eval/reference_cpu.py --model-path fasth3-4step --prompt-embeds embeds.pt \
  --height 256 --width 256 --num-frames 124 --out ref_fp32.pt
MINIMAX_H3_DUMP_LATENTS=device.pt python examples/minimax_h3/run.py --model-path fasth3-4step \
  --height 256 --width 256 --num-frames 124 --prompt-embeds embeds.pt --output check.mp4
python examples/minimax_h3/eval/gate_compare.py --ref ref_fp32.pt --run device.pt
```

The fp32 CPU reference of the 33B DiT takes about 5 minutes per forward on a trn2.48xlarge host.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Fine vertical streaks in the decoded video | VAE tiling disabled | Keep `use_tiling` on (the default): this VAE's output depends on the tile geometry |
| `Could not find config.json or model_index.json` | An old plugin without modular-index support | Update the plugin |
| A job is killed for host RSS | Host AdaLN at TP=8 streams the 13B AdaLN weights on every rank | Keep `adaln: device` (the default) |
| `Failed to allocate HOST memory (4194304 bytes)` ... `dma rings` at stage start | No free 4 MB contiguous host pages (fragmentation) | `echo 1 > /proc/sys/vm/compact_memory`, then restart |

## Conclusion

You served FastH3 on all 16 Trainium2 chips: the DiT is tensor- and context-parallel, the text encoder runs
tensor-parallel on the device, the video VAE decodes on device in parallel tiles, and the audio VAE decodes on device
in parallel time windows.

## Next steps

- [MiniMax-H3 / FastH3 model card](../models/minimax-h3.md)
- [Quickstart: Offline video-and-audio generation with FastH3](../getting-started/quickstart-offline-serving-minimax-h3.md)
