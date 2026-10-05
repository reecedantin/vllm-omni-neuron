# Tutorial: Deploy GR00T N1.7 with vLLM Omni Neuron

<!-- meta: description: End-to-end tutorial for serving NVIDIA GR00T N1.7 (a vision-language-action robot
policy) on AWS Inferentia2 / Trainium with the vLLM Omni Neuron plugin: environment, model download, stage
configuration, offline and online inference, the GR00T-H variant, and troubleshooting. -->
<!-- meta: keywords: GR00T, GR00T N1.7, tutorial, vLLM Omni, Neuron, inf2, trn2, robot policy, VLA -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

You will serve NVIDIA's GR00T N1.7-3B robot policy on one NeuronCore and get a 40-step action chunk from a
camera observation, state and a language instruction — offline first, then as an OpenPI-compatible online
server. The whole model is ~5.9 GiB in BF16, so one Inferentia2 core is enough; the first request compiles
3 graphs (roughly 30-60s on a cold NEFF cache).

## Step 1: Set up your environment

Follow the [setup guide](setup-guide.md) for your instance. GR00T needs no extra environment variables
beyond the processor path if you want it fully offline (Step 2).

## Step 2: Download the model (optional)

```bash
hf auth login   # nvidia/GR00T-N1.7-3B is a gated repo
huggingface-cli download nvidia/GR00T-N1.7-3B --local-dir /opt/models/gr00t-n17
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir /opt/models/qwen3-vl-2b-instruct
```

The second download is the VLM processor (tokenizer + image processor): GR00T's backbone is a
Cosmos-Reason2-2B post-train of Qwen3-VL-2B, and Cosmos-Reason2-2B does not ship its own processor files,
so upstream GR00T falls back to Qwen3-VL-2B-Instruct's. Point `GR00T_VLM_PROCESSOR` at the local copy to
avoid a Hub call at serve time.

## Step 3: Review the stage configuration

```yaml
# examples/gr00t/gr00t_stage_trn2.yaml (excerpt)
stage_args:
  - stage_id: 0
    stage_type: diffusion
    final_output_type: actions
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_class_name: Gr00tN1d7Pipeline
      dtype: bfloat16
      model_config:
        embodiment_tag: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
      parallel_config:
        tensor_parallel_size: 1
```

GR00T needs no parallelism: `tensor_parallel_size: 1`, one core. `devices:` picks which NeuronCore the
engine uses — change it if core 0 is in use by something else on the host. `embodiment_tag` must match a
tag the checkpoint's processor config declares (see `nvidia/GR00T-N1.7-3B`'s `processor_config.json` for
the full list); it decides which state/action keys the model expects and how they are normalized.
`gr00t_stage.yaml` (no hardware suffix) is identical — GR00T needs no per-generation tuning, so there is
only one recommended configuration.

## Step 4: Run inference

### Offline

```bash
python examples/gr00t/run.py --model-path /opt/models/gr00t-n17 \
  --image-dir /path/to/frames --prompt "pick up the red cube and put it in the bowl" \
  --output gr00t_actions.npz
```

See the [offline quickstart](../getting-started/quickstart-offline-serving-gr00t-n17.md) for the full flag
reference and what to put in `--image-dir`.

### Online serving

```bash
vllm serve /opt/models/gr00t-n17 --omni --stage-configs-path examples/gr00t/gr00t_stage_trn2.yaml --port 8000
```

GR00T's single diffusion stage is served through vLLM Omni's OpenPI-compatible endpoint. A client sends an
observation (video frames, state, language) and a session id; the server returns the decoded action
dictionary. See [`vllm_omni.entrypoints.openpi`](https://github.com/aws-neuron/vllm-omni-neuron) for the
request/response schema — it is unchanged by the Neuron port.

## Optional: GR00T-H-N1.7 (surgical robots)

[GR00T-H-N1.7](https://huggingface.co/nvidia/GR00T-H-N1.7) is NVIDIA's surgical-robotics post-train of
N1.7. The same `Gr00tN1d7Pipeline` loads and runs it — point `--model-path` at a GR00T-H checkpoint and the
pipeline adapts automatically (it has no VL self-attention block and a 50-step action horizon instead of
40). What it does **not** have out of the box is GR00T-H's own processor: its `processor_config.json` uses
representations and surgical embodiment tags that upstream vLLM-Omni's processor does not implement, so
serving GR00T-H end to end needs [NVIDIA-Medtech/GR00T-H](https://github.com/NVIDIA-Medtech/GR00T-H)'s own
processor code. See [GR00T-H vs N1.7](../models/gr00t-n17.md#gr00t-h-vs-n17) in the model card for the full
comparison and what is and is not ported.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GatedRepoError` downloading the backbone | `nvidia/GR00T-N1.7-3B` is gated | Request access, then `hf auth login` / `HF_TOKEN` |
| `NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM` | vLLM Omni manages core assignment itself | Remove the env var; use the stage config's `devices:` field |
| `ModalityConfig.__init__() got an unexpected keyword argument 'min_max_embedding_keys'` | Trying to load GR00T-H's processor config with upstream's processor | Expected — see the GR00T-H section above; not fixable from this plugin alone |
| First request takes 30-60s | Cold NEFF cache (3 graphs compiling) | Expected once per cache; subsequent requests are warm (~86 ms end to end) |

## Conclusion

You served GR00T N1.7 on one NeuronCore, offline and online, and saw where the GR00T-H surgical variant
diverges. The model-level port is parity-tested against upstream's CPU reference (action MSE 4.66e-4,
cosine 0.99980 — see the [model card](../models/gr00t-n17.md#accuracy-evaluation)) and runs end to end at
~86 ms per request warm.

## Next steps

- [GR00T N1.7 model card](../models/gr00t-n17.md)
- [Offline quickstart](../getting-started/quickstart-offline-serving-gr00t-n17.md)
