# Tutorial: Serve FLUX 3 Action with vLLM Omni Neuron

<!-- meta: description: Serve Black Forest Labs' FLUX 3 Action DROID policy on AWS Trainium2 with the vLLM Omni Neuron
plugin: environment, weights, stage configuration, a served policy request, tensor parallelism and accuracy
checks against the upstream reference. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, FLUX 3 Action, DROID, robot policy, world action model,
Trainium2, trn2, tensor parallelism, tutorial -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- meta: content_type: tutorial -->

FLUX 3 Action is a 7B world action model: from three camera frames, the robot state and an instruction it
denoises the next 32 actions jointly with the next video frames. This tutorial serves the released DROID policy
on Trainium2 through a vLLM Omni stage, then checks its actions against the upstream reference. See the
[model card](../models/flux3-action.md) for the architecture, accuracy and performance numbers.

## Step 1: Set up your environment

Prepare the plugin environment with the [setup guide](setup-guide.md). The served stage runs in its own worker
process, so select NeuronCores with `NEURON_VISIBLE_DEVICES` and leave `NEURON_RT_VISIBLE_CORES` unset.

## Step 2: Download the model

```bash
hf download black-forest-labs/flux-3-action-droid --exclude 'variants/*' --local-dir weights/droid
hf download black-forest-labs/flux-3-action-base --include 'video_vae.safetensors' --include 'text_encoder/*' \
    --local-dir weights/base
export FLUX3_ACTION_BASE=$PWD/weights/base
```

The policy package's config pins the shared encoders to a revision of `flux-3-action-base`. Without
`FLUX3_ACTION_BASE` the pipeline resolves them from the Hugging Face Hub (or its offline cache) at that revision.

## Step 3: Review the stage configuration

```yaml
# examples/flux3_action/flux3_action_stage.yaml (excerpt)
    runtime:
      process: true
      devices: "0"
    engine_args:
      model_class_name: Flux3ActionPipeline
      dtype: bfloat16
      parallel_config:
        tensor_parallel_size: 1
```

`Flux3ActionPipeline` wraps the policy: the request carries the cameras, state and instruction in
`sampling_params.extra_args`, and the response carries the action chunk. The released packages ship no
`model_index.json`; `examples/flux3_action/make_serving_dir.py` builds a directory of symlinks to the package
plus the two discovery files vLLM Omni reads, and `serve_policy.py` does this for you.

## Step 4: Run inference

### A served policy request

```bash
python examples/flux3_action/make_observation.py --policy weights/droid --output obs.npz
python examples/flux3_action/serve_policy.py --policy weights/droid --base weights/base \
    --obs obs.npz --out-dir out-served/ --warm 2
```

`out-served/served_report.json` reports the warm latency and the per-stage timing (`vae_encode_s`,
`denoise_forward_s`, `num_steps`, `cfg_branches`) and a `request_breakdown_s` split of the whole request
(client build, client to worker, parse, encode, denoise, return). On Trn2 a warm request takes about 1.6 s at
TP4 and 1.0 s at TP4 x CFG2 (below). Cameras travel as raw bytes (`observation_extra_args` /
`observation.encode_camera`); nested lists are accepted but cost ~3 s per request in transport.

From Python, the request is:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni_neuron.diffusion.models.flux3_action.observation import observation_extra_args

omni = Omni(model="out-served/serving", stage_configs_path="examples/flux3_action/flux3_action_stage.yaml")
extra = observation_extra_args("obs.npz")  # images.*, state, task
result = omni.generate({"prompt": extra["task"]}, OmniDiffusionSamplingParams(seed=0, extra_args=extra))
# the action envelope {"actions": (1, 32, 8), ...} is in the request output; serve_policy.py shows the lookup
```

### Tensor and CFG parallelism

`examples/flux3_action/flux3_action_stage_tp4.yaml` shards the DiT's heads across the four cores of one chip
(`devices: "0,1,2,3"`, `tensor_parallel_size: 4`): DiT denoise 1.36 s, warm request 1.65 s. The VAE encoder runs
on rank 0's NeuronCore and the latent is broadcast to the other ranks.

The recommended `examples/flux3_action/flux3_action_stage_tp4_cfg2.yaml` adds `cfg_parallel_size: 2` on eight
cores (an adjacent chip pair): two TP4 groups each run one guidance branch per step, at once, and exchange the
outputs. Same actions as TP4 bit for bit; DiT denoise 0.73 s, warm request 1.01 s on a quiet host (1.12 s on a
busy one). Add `--rank-check` to confirm that all eight ranks return identical actions and predicted latents on
every request (`rank_check` in `served_report.json`; about 0.1 s per request):

```bash
python examples/flux3_action/serve_policy.py --policy weights/droid --base weights/base \
    --obs obs.npz --out-dir out-tp4cfg2/ --warm 3 --rank-check \
    --stage-config examples/flux3_action/flux3_action_stage_tp4_cfg2.yaml
```

## Optional: compare against the upstream reference

```bash
export FLUX_ACTION_SRC=<checkout of black-forest-labs/flux-action>/src
python -m test.unit.test_flux3_action_utils.reference --policy weights/droid --base weights/base \
    --obs obs.npz --out ref_bf16.pt
python examples/flux3_action/run_policy.py --policy weights/droid --base weights/base \
    --obs obs.npz --out-dir out/ --reference ref_bf16.pt
```

The summary's `parity` block reports action MSE and rel-L2 and the predicted-latent cosine against the
reference. The upstream reference runs on CPU with a NATTEN stand-in, since NATTEN needs CUDA.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Dummy run failed` at startup | The engine warms the graphs with an empty request; make sure you run this plugin's pipeline (the stage YAML's `model_class_name: Flux3ActionPipeline`). |
| `replica id #N not seen in replica groups` at TP > 1 | Use the plugin's DiT; it registers its TP replica groups at construction. |
| Predicted frames show a mesh texture | Expected at 4 steps on out-of-distribution observations; see Known limitations in the model card. |

## Conclusion

You served the FLUX 3 Action DROID policy on Trainium2 through vLLM Omni and checked its action chunk against the
upstream reference.

## Next steps

- [FLUX 3 Action model card](../models/flux3-action.md)
- [Quickstart: Offline policy inference with FLUX 3 Action](../getting-started/quickstart-offline-serving-flux3-action.md)
