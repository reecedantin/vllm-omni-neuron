# Cosmos3-Nano / Cosmos3-Super Model Card

<!-- meta: description: Model card for NVIDIA Cosmos3-Nano and Cosmos3-Super (base + 4-step distilled)
on AWS Trainium2 with the vLLM Omni Neuron plugin. Covers the Qwen3-VL backbone, TP layouts (TP=4 Nano,
TP=16 Super with KV-head replication), accuracy against a CPU fp32 oracle, performance, and known issues. -->
<!-- meta: keywords: Cosmos3-Nano, Cosmos3-Super, NVIDIA Cosmos, model card, text-to-image, Qwen3-VL,
diffusion, vLLM, vLLM Omni, Neuron, Trainium2, trn2, NeuronCore-v3, BF16, KV-head replication, tensor
parallelism -->
<!-- meta: content_type: model-card -->
<!-- meta: date_updated: 2026-10-06 -->

## Introduction

[Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (15.2B params, Qwen3-VL-8B backbone) and
[Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) (65B params, Qwen3-VL-32B backbone; also
the Super 4-step I2V/T2I distilled variants) are NVIDIA's larger Cosmos3 world models. They share
Cosmos3-Edge's Mixture-of-Transformers design -- a causal understanding (UND) tower encoding text and
conditioning, a bidirectional generation (GEN) tower cross-attending to the UND tower's cached K/V at
every denoising step, and a Wan2.2-5B VAE -- with a Qwen3-VL backbone instead of Edge's Nemotron dense
backbone: per-head QK RMSNorm in the UND attention and SiLU-gated MLPs in both towers. One checkpoint
serves text-to-image, text-to-video, image-to-video, and the robot world-model (action) modes.

Cosmos3-Nano / Super are supported for inference serving with
[vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/) using the Neuron SDK on AWS Trainium2
(`trn2`, NeuronCore-v3, LNC=2).

**License:** [OpenMDW 1.1](https://openmdw.ai/license/1-1/) (NVIDIA). Checkpoints are not gated.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|---------------|
| Cosmos3-Nano | [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | Trn2 | BF16 |
| Cosmos3-Super | [nvidia/Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) | Trn2 | BF16 |
| Cosmos3-Super 4-step I2V | nvidia/Cosmos3-Super-i2v-4step | Trn2 | BF16 |
| Cosmos3-Super 4-step T2I | nvidia/Cosmos3-Super-t2i-4step | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Generation** | Text-to-image (up to 832x480 / 640x640) | ✅ |
| | Image-to-video | ✅ |
| | Robot action modes | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor Parallelism (TP) | ✅ |
| | Context Parallelism (CP) | ✅ (GEN video tokens, TP=8 x CP) |
| | CFG Parallelism | ✅ |
| | VAE Patch Parallelism | ✅ (decode; encode tiles dealt across TP ranks) |
| **Compilation** | torch.compile | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Cosmos3-Nano / Super.
- `-`: not supported.

### Recommended configuration

| Model | Layout | Stage config |
|---|---|---|
| Cosmos3-Nano | TP=4 on one Trainium2 chip (4 logical cores at LNC=2; 32 query heads / 8 KV heads -> 8/2 per core) | `examples/cosmos3_edge/cosmos3_nano_stage_trn2_tp4.yaml` |
| Cosmos3-Super (base, 4-step I2V/T2I) | TP=16 across one torus row (4 chips, 16 logical cores; 64 query heads / 8 KV heads -> KV-head replication, one full KV head per rank, 2 ranks per head) | `examples/cosmos3_edge/cosmos3_super_stage_trn2_tp16.yaml`, `..._super4step_i2v_stage_trn2_tp16.yaml` |

**Cosmos3-Super 4-step at 832x480 (the NVIDIA model-card workload)**, best measured layouts:

| Workload | Cores (chips) | Layout | Stage config |
|---|---|---|---|
| T2I 832x480 | 16 (4) | TP=16, VAE patch-parallel 16 | `examples/cosmos3_edge/cosmos3_super4step_t2i_stage_trn2_tp16.yaml` |
| I2V 832x480x189f | 32 (8) | TP=8 x CP=4, VAE patch-parallel 32, host tile gather | `examples/cosmos3_edge/cosmos3_super4step_i2v_stage_trn2_tp8cp4.yaml` |
| I2V 832x480x189f, lowest latency | 64 (16) | TP=8 x CP=8, VAE patch-parallel 64, host tile gather | `examples/cosmos3_edge/cosmos3_super4step_i2v_stage_trn2_tp8cp8.yaml` |

**Robot action modes** (Nano-Policy-DROID, Edge-Policy-DROID; DROID recipe 832x480x33f, 4 steps,
guidance 3.0): TP=4 x CFG-parallel 2 on 8 cores, `examples/cosmos3_edge/cosmos3_nano_policy_stage_trn2_tp4cfg2.yaml`
with `COSMOS3_EDGE_VAE_TILE=256,256,240,208` (see Robot action modes).

At 832x480 set `COSMOS3_EDGE_VAE_TILE=256,256,224,192` (the decode runs as 2 x 4 fixed 256 px tiles;
the untiled 832x480 decoder graph does not compile on Trn2, `NCC_IBIR229`), and for long clips
`COSMOS3_EDGE_VAE_HOST_GATHER_T=12` (see Known limitations). Context parallelism splits
the GEN tower's patchified video tokens over the CP ranks; each rank attends with its local queries
over [UND text K/V | every rank's GEN K/V], all-gathered per layer over the CP device group (GEN
attention is bidirectional and RoPE is applied per token before the gather). TP=8 x CP uses the
plugin's Trn2 physical-mesh groups. CP applies to video GEN calls only; the UND prefill and action
calls run unsplit, and the token count must divide by the CP degree.

Super's KV-head count (8) is smaller than its TP degree (16), so weight loading replicates whole KV
heads across the ranks that share one (`EdgeTextConfig.kv_heads_local`); this is a strict extension of
Nano's plain per-rank KV shard, verified bit-identical at TP<=KV-heads (Nano TP=1/2/4) in
`test/unit/test_cosmos3_edge_qwen3.py::test_tp_matches`.

The I2V / action conditioning frame is VAE-encoded on the NeuronCore (the moments are broadcast from
the TP group's first rank to the others; tiled encodes deal their tiles across the TP ranks,
`COSMOS3_VAE_ENCODE_TP=0` keeps them on one rank), with `compile_encoder=True` (stage config `compile_vae_encoder: true`).
On Trn2 frames above 192 px are encoded as fixed-shape 192 px tiles with 96 px overlap through the
plugin's shared VAE tiling: one compiled encoder graph serves every resolution. Overrides:
`COSMOS3_VAE_ENCODE_TILE=<tile>,<overlap>` (`0` = untiled), `COSMOS3_VAE_ENCODE=host` (fp32 CPU encode).

## Accuracy Evaluation

**Gate:** teacher-forced parity (UND + one GEN call) against a full fp32 CPU oracle, bf16 on device.
Super's checkpoint (128 GB bf16) does not fit one 24 GB logical core, so its device side runs the same
TP=16 layout the stage config uses, each rank pinned to its own core before Neuron-runtime init
(`test/neuron/test_cosmos3_edge_super_tp_parity.py`); Nano runs single-rank
(`test/neuron/test_cosmos3_edge_qwen3_device.py`). Threshold: device rel-L2 <= 2x the pure-bf16-on-CPU
rel-L2, cosine >= 0.999, bit-identical across repeated runs.

| Metric | Nano, TP=4 | Super, TP=16 | Reference / threshold |
|---|---|---|---|
| T2I GEN rel-L2 (640x640, 400 tok) | 1.59% (pure-bf16 CPU: 1.84%) | 1.49% | <= 2x pure-bf16 CPU |
| I2V GEN rel-L2 | 1.46% (pure-bf16 CPU: 1.63%) | 1.50% | <= 2x pure-bf16 CPU |
| action GEN rel-L2 | 1.52% (pure-bf16 CPU: 1.70%) | 1.52% | <= 2x pure-bf16 CPU |
| cosine (all cases) | >= 0.9999 | >= 0.9999 | >= 0.999 |
| deterministic (2 runs) | yes, all cases | yes, all cases | bit-identical |
| UND rel-L2 (T2I) | 0.72% (pure-bf16 CPU: 0.86%) | -- | <= 2x pure-bf16 CPU |

**Context-parallel GEN (Super 4-step I2V, step-0 parity).** The first GEN call of a device request is
dumped (`COSMOS3_EDGE_DUMP_GEN0=<file>`) and re-run unsharded on the CPU at fp32 and bf16. Same bars as
above.

| Layout | Call | device rel-L2 vs fp32 | CPU bf16 rel-L2 vs fp32 | cosine | verdict |
|---|---|---|---|---|---|
| TP=8 x CP=4 | 256 GEN tokens (64 per CP rank), 12 text tokens | 2.48% | 2.47% | 0.9997 | pass |
| TP=8 x CP=8 | 832x480x189f first step, 18,720 GEN tokens (2,340 per CP rank) | 7.88% | 6.41% | 0.9969 (CPU bf16: 0.9980) | pass vs bf16 (1.23x rel, 1.52x 1-cos); below the absolute 0.999 cosine bar, which pure bf16 on the CPU misses too at this length |

At 18,720 tokens the bf16 rounding error itself grows (6.4% on the CPU vs 1.5% at 400 tokens), so
the device is judged against pure bf16 of the same call, as for the shorter cases. The CPU test
`test/unit/test_cosmos3_edge_qwen3.py::test_cp_matches_unsplit` checks CP=2/4 against the
unsplit call (rel < 1e-5). The 189-frame device videos are temporally coherent over all frames, with no
tile seams from the 2 x 4 decode grid and no flicker or banding (12-frame contact sheets at 16, 32 and
64 cores).

**Conditioning-frame encode (I2V / action), device tiled** (`test_cosmos3_edge_vae_encode_device.py`):
the Wan encoder runs as fixed-shape 192 px tiles with 96 px overlap (see Recommended configuration).
Three-way with the SAME tiling in all three isolates the Neuron error; the untiled fp32 CPU encode (the
earlier host path) measures what tiling itself changes.

| Frame | device vs fp32 tiled (bf16 CPU) | three-way | device vs untiled fp32, latent | after decode, vs untiled | VAE's own recon PSNR | warm |
|---|---|---|---|---|---|---|
| 256 px | 0.41% (0.41%) | pass | 3.4% | 47.1 dB | 29.6 dB | 0.20 s |
| 480 px | 0.42% (0.43%) | pass | 7.9% | 39.8 dB | 32.8 dB | 0.79 s |
| 640 px | 0.44% (0.46%) | pass | 8.9% | 41.2 dB | 40.3 dB | 1.73 s |

The tiling difference stays at or below the VAE's own reconstruction error, and end to end it is not
visible: Nano I2V 640x640x9f (35 steps, seed 1) against the same request with the untiled host encode
scores per-frame SSIM 0.940-0.955 (mean 0.946; gate 0.90), with no seams.

**Test tiers** (`test/neuron/`, auto-skipped without a Neuron device or weights):

| Tier | Test | Scope | Gate |
|---|---|---|---|
| 1. Component | `test_cosmos3_edge_qwen3_device.py` | UND tower (per-layer K/V) and one GEN call, T2I / I2V / action | `assert_close_three_way` (fp32 CPU / bf16 CPU / bf16 Neuron) |
| 1. Component | `test_cosmos3_edge_vae_encode_device.py` | tiled VAE encode of one conditioning frame | `assert_close_three_way`, and decoded PSNR vs the untiled encode >= 35 dB |
| 2. Single step | `test_cosmos3_edge_pipeline_accuracy.py` | upstream `diffuse()` for one step (`num_steps=1`): both CFG branches, guidance combine, scheduler step, no VAE decode; Nano TP=4 | `assert_close_three_way` against upstream's own fp32 transformer |
| 3. End to end | `test_cosmos3_edge_e2e_accuracy.py` | full 50-step T2I and 35-step I2V through `examples/cosmos3_edge/run.py` | SSIM against a cached golden image / video (`COSMOS3_E2E_GOLDEN`, `COSMOS3_E2E_I2V_GOLDEN`) |

Super's 128 GB checkpoint does not fit one logical core, so its tier-1 coverage is the TP=16 harness
`test_cosmos3_edge_super_tp_parity.py`.

**Reproduce:**

```bash
# CPU parity (no device needed), real-weight device parity, Super TP parity
python -m pytest test/unit/test_cosmos3_edge_qwen3.py
COSMOS3_QWEN3_WEIGHTS=/path/to/Cosmos3-Nano python -m pytest test/neuron/test_cosmos3_edge_qwen3_device.py
COSMOS3_QWEN3_WEIGHTS=/path/to/Cosmos3-Super python -m pytest test/neuron/test_cosmos3_edge_super_tp_parity.py

# T2I at the checkpoint's own default steps (see "Known limitations" -- do not lower this for the base checkpoints)
python examples/cosmos3_edge/run.py --mode t2i --model-path /path/to/Cosmos3-Nano \
  --stage-config examples/cosmos3_edge/cosmos3_nano_stage_trn2_tp4.yaml --height 640 --width 640 --steps 50
```

## Performance

**How the numbers are measured.** Every warm latency on this card is the median of 3 back-to-back
requests end to end through `examples/cosmos3_edge/run.py`, after the first request and one more
uncounted warm-up request, on a quiet host (1-minute load average below 20 before every timed
request; device-only runs, no CPU reference work alongside). Stages come from `COSMOS3_EDGE_PROFILE=1`
(seconds per request on the slowest rank). Every timed run also digests the final latent on every
rank (`COSMOS3_EDGE_RANK_DIGEST`) and checks them with `vllm_omni_neuron.testing.compare_rank_digest_files`:
all ranks were bit-identical on every request of every row. Each row states its accuracy evidence;
rows marked *speed only* were not accuracy-checked at that layout.

| Configuration | Hardware | Warm latency | Notes | Accuracy evidence |
|---|---|---|---|---|
| T2I 640x640, 50 steps | Nano, TP=4 | 3.50 s | GEN 2.89 s; VAE decode 0.50 s | output bit-identical to the tier-3 golden |
| I2V 640x640x9f, 35 steps | Nano, TP=4 | 8.18 s | device tiled VAE encode 0.52 s, GEN 6.14 s, decode 1.26 s | per-frame SSIM 0.946 (min 0.940) vs the host-encode golden (tier 3, gate 0.90) |
| action (policy, 256), 30 steps | Nano, TP=4 | 1.28 s | encode 0.07 s, GEN 0.86 s, decode 0.29 s | *speed only* (GEN-call parity at 1 core, see Accuracy Evaluation) |
| T2I 640x640, 50 steps | Super, TP=16 | 5.86 s | GEN 5.12 s; VAE decode 0.58 s | TP=16 teacher-forced GEN parity (Accuracy Evaluation) |
| I2V 640x640x9f, 4 steps | Super 4-step, TP=16 | 2.06 s | stage config as shipped: encode 0.22 s (tiles dealt across the 16 TP ranks), GEN 4 x 150 ms, decode without patch parallelism 1.10 s | TP=16 teacher-forced GEN parity |
| I2V 640x640x9f, 4 steps, decode patch-parallel 16 | Super 4-step, TP=16 | 1.27 s | as above plus `vae_patch_parallel_size: 16` and `COSMOS3_EDGE_VAE_TILE=256,256,192,192`: decode 0.31 s | vs the untiled decode per-frame SSIM 0.957 (min 0.953), PSNR 36.2 dB |

**Super 4-step at 832x480 vs NVIDIA's model-card numbers.** Warm = median of 3 back-to-back requests
on a quiet host (see the measurement note above) end to end through `examples/cosmos3_edge/run.py`
(bf16, batch 1, 4 fixed steps, guidance 1.0); stage columns are seconds per request on the slowest
rank; HBM from the earlier sweep at the same layouts. 1 Trainium2 chip = 4 logical cores (LNC=2).
Super's 128 GB of weights do not fit at 1, 2 or 4 cores (~24 GB HBM per core); 189 frames do not fit
at TP=8 without CP (HBM). The first request after engine start takes 28-31 s for T2I and about 3 min
for the 189-frame I2V (32 cores; compile cache warm).

T2I, single frame (390 GEN tokens):

| Cores (chips) | Layout | Warm | GEN 4 steps | VAE decode | HBM / core max | Accuracy evidence |
|---|---|---|---|---|---|---|
| 8 (2) | TP=8 | 0.52 s | 0.32 s | 0.16 s | 19.2 GB | first GEN call of the timed configuration vs CPU: rel-L2 2.93% (pure-bf16 CPU 2.36%, 1.24x), cosine 0.9996 -- pass |
| 16 (4) | TP=16 | 0.45 s | 0.23 s | 0.18 s | 12.0 GB | first GEN call of the timed run vs CPU: rel-L2 3.66% (pure-bf16 CPU 2.36%, 1.55x), cosine 0.9993 -- pass |
| 32 (8) | TP=32 | 0.48 s | 0.22 s | 0.22 s | 8.5 GB | first GEN call of the timed configuration vs CPU: rel-L2 2.88% (pure-bf16 CPU 2.36%, 1.22x), cosine 0.9996 -- pass |
| 64 (16) | TP=64 | 0.48 s | 0.17 s | 0.28 s | 6.7 GB | first GEN call of the timed configuration vs CPU: rel-L2 2.65% (pure-bf16 CPU 2.36%, 1.12x), cosine 0.9997 -- pass |

I2V, 189 frames at 24 fps (48 latent frames, 18,720 GEN tokens):

| Cores (chips) | Layout | Warm | VAE encode | GEN 4 steps | VAE decode | HBM / core max |
|---|---|---|---|---|---|---|
| 16 (4) | TP=8 x CP=2 | 25.94 s | 0.29 s | 19.15 s | 4.50 s | 21.2 GB |
| 16 (4) | TP=16 | 27.24 s | 0.18 s | 20.37 s | 4.51 s | 14.6 GB |
| 32 (8) | TP=8 x CP=4, host point-to-point tile gather | 16.34 s | 0.30 s | 9.64 s | 4.43 s | 20.6 GB |
| 64 (16) | TP=8 x CP=8, host point-to-point tile gather | 12.49 s | 0.30 s | 5.47 s | 4.64 s | 20.4 GB |

All rows use decode tiles 256/256/224/192. Accuracy evidence: the first GEN call at TP=8 x CP=8 (the
full 18,720-token shape) passes the CPU fp32 / bf16 check (Accuracy Evaluation), and every other
layout is checked against it through TP=16 (no context parallelism), frame by frame: TP=8 x CP=2 vs
TP=16 per-frame SSIM 0.954 (min 0.935), PSNR 32.8 dB; TP=8 x CP=4 SSIM 0.953 (min 0.928), PSNR 32.8
dB; TP=8 x CP=8 SSIM 0.950 (min 0.925), PSNR 31.9 dB, with no frame below 0.92 and no drift over the
clip; the right and bottom 64 px strips agree as well (SSIM 0.95 / 0.89-0.91). The TP=8 x CP=2,
TP=8 x CP=4 and TP=8 x CP=8 videos of the timing runs are bit-identical to those of the accuracy runs. The latent each
rank decodes is bit-identical on all 16, 32 and 64 ranks, and identical across repeated requests.

NVIDIA's published figures for the same checkpoints are estimates, not measurements: the base model's
measured latency divided by 17.5 (I2V: 35 steps x 2 CFG passes / 4) or 25 (T2I: 50 x 2 / 4). They
therefore cover the transformer only, with no text encode, VAE encode / decode or output encode:

| | T2I 832x480 | I2V 832x480x189f |
|---|---|---|
| NVIDIA, vLLM-Omni, B200 | 0.132 s (1 GPU), 0.124 s (4), 0.364 s (8) | 6.581 s (1), 2.114 s (4), 1.329 s (8) |
| NVIDIA, vLLM-Omni, H200 | 0.228 s (1), 0.150 s (4), 0.286 s (8) | 12.611 s (1), 3.731 s (4), 2.109 s (8) |
| NVIDIA, PyTorch, B200 | 0.191 s (1), 0.165 s (4), 0.179 s (8) | 6.423 s (1), 2.040 s (4), 1.214 s (8) |
| NVIDIA, PyTorch, H100 NVL | 0.791 s (4) | 5.689 s (4), 3.667 s (8) |
| Trn2, GEN 4 steps only (comparable column) | 0.17-0.23 s (4-16 chips) | 9.64 s (8 chips), 5.47 s (16 chips) |
| Trn2, end to end | 0.45 s (4 chips) | 16.34 s (8 chips), 12.49 s (16 chips) |

T2I is latency-floor-bound above 16 cores (390 tokens, 40-60 ms per GEN step; the VAE decode is ~40-55%
of the request). I2V GEN scales with CP (19.2 -> 9.6 -> 5.5 s at CP 2 / 4 / 8). The 189-frame VAE
decode does not: the 2 x 4 tile grid gives 8 tiles whatever the world size, each streaming 48 latent
frames through the decoder's causal cache (~4.5 s per tile), so decode stays at 4.4-4.6 s from 16 to
64 cores and is about 37% of the 64-core request.

HBM / first request (real weights, T2I 640x640):

| | Nano | Super |
|---|---|---|
| HBM / core | 10.13 GB (of ~24) | 11.42 GB, uniform across all 16 cores (of ~24) |
| first request, cold cache (incl. compile; 4 steps) | 399 s | 393 s |
| first request, warm compile cache (50 steps) | 29.9 s | 38.5 s |

**Optimization applied: the conditioning-frame encode runs once, on the NeuronCore.** The host-side
pipeline math runs independently on every TP rank's worker process, so before any fix every rank
re-encoded the identical conditioning frame on the host CPU (4x ~15 s for Nano TP=4). Two steps:

1. Only TP rank 0 encodes; the others receive the small moments tensor over the TP group's existing
   gloo `cpu_group` (the channel CFG-parallel combine uses). Nano I2V TP=4, same geometry and seed:
   23.72 s -> 20.96 s, and 4x less redundant host CPU.
2. Rank 0 encodes on the device instead of the host: the compiled Wan encoder as fixed-shape 192 px
   tiles (the untiled graph does not compile above 192 px on Trn2). Warm VAE encode 12.2 s -> 1.87 s;
   Nano I2V warm 19.69 s -> 9.42 s (same job, same request: host vs device encode).

## Robot action modes

The action head serves three modes through the generic action path (`extra_args["action_mode"]`):
`policy` (image + instruction -> action chunk and predicted video), `forward_dynamics` (image + action
chunk -> video; the actions are the condition) and `inverse_dynamics` (video -> actions; every latent
frame is a clean condition, so the whole clip is VAE-encoded). The checkpoints with an action head are
Cosmos3-Nano / Super (base), Cosmos3-Nano-Policy-DROID and Cosmos3-Edge-Policy-DROID. The 4-step
distilled checkpoints reject action requests (upstream behaviour). The OpenPI / RoboLab policy-server
path of upstream (`extra_args["robot_obs"]`) needs the `cosmos_framework` package and is not covered
here; the generic path takes the same settings directly.

**Accuracy** (one GEN call per mode, DROID domain, 33 frames = 9 latent frames + a 32-row action chunk,
condition masks from the pipeline's own rules; rel-L2 vs fp32 CPU, pure-bf16 CPU in brackets):

| Checkpoint | Layout | Mode | video | action | cosine (video / action) | three-way |
|---|---|---|---|---|---|---|
| Nano-Policy-DROID | 1 core | policy | 1.50% (1.72%) | 1.01% (1.30%) | 0.99989 / 0.99995 | pass |
| | | forward_dynamics | 1.54% (1.71%) | 1.37% (1.74%) | 0.99988 / 0.99991 | pass |
| | | inverse_dynamics | 1.37% (1.44%) | 0.64% (0.72%) | 0.99991 / 0.99998 | pass |
| Edge-Policy-DROID | 1 core | policy | 1.28% (1.41%) | 0.35% (0.38%) | 0.99992 / 0.99999 | pass |
| | | forward_dynamics | 1.17% (1.23%) | 0.49% (0.64%) | 0.99993 / 0.99999 | pass |
| | | inverse_dynamics | 1.11% (1.13%) | 0.34% (0.38%) | 0.99994 / 0.99999 | pass |
| Super (base) | TP=16 | policy (17 frames, 16 rows) | 1.53% (1.55%) | 0.75% (0.84%) | 0.99988 / 0.99997 | pass |
| | | forward_dynamics | 1.54% (1.55%) | 0.95% (1.07%) | 0.99988 / 0.99995 | pass |
| | | inverse_dynamics | 1.50% (1.50%) | 0.80% (0.83%) | 0.99989 / 0.99997 | pass |

All deterministic across two runs. Tests: `test/unit/test_cosmos3_edge_qwen3.py::test_gen_action_modes_match_upstream`
(CPU, vs the upstream transformer, rel < 1e-4), `test/neuron/test_cosmos3_edge_qwen3_device.py::test_qwen3_action_mode_device_parity`
(single rank, Nano and Edge), `test/neuron/test_cosmos3_edge_super_tp_parity.py` (Super, TP=16).

**Latency** with NVIDIA's DROID serving settings: 832x480, 33 frames (32-row chunk), 4 steps,
guidance 3.0, decode tiles `COSMOS3_EDGE_VAE_TILE=256,256,240,208`, warm median of 3 on a quiet host
(see Performance), seconds. *Speed only*: accuracy is checked per GEN call at 1 core (Nano / Edge)
and TP=16 (Super) above; at the served TP layouts below every rank decodes a bit-identical latent
and TP=4 x CFG 2 is bit-identical to sequential CFG at TP=4, but the end-to-end output is not
compared with a CPU reference.

| Checkpoint | Cores | Layout | Mode | warm | VAE encode | GEN (4 steps x 2 CFG passes) | VAE decode |
|---|---|---|---|---|---|---|---|
| Nano-Policy-DROID | 8 | TP=4 x CFG 2 | policy | **2.63** | 0.48 | 1.17 | 0.65 |
| | 8 | TP=4 x CFG 2 | policy, actions only | **1.68** | 0.47 | 1.16 | -- |
| | 8 | TP=4 x CFG 2 | forward_dynamics | **2.64** | 0.48 | 1.17 | 0.67 |
| | 8 | TP=4 x CFG 2 | inverse_dynamics | 4.93 | 1.75 | 1.17 | 0.67 |
| | 8 | TP=4 x CFG 2 | inverse_dynamics, actions only | **3.98** | 1.77 | 1.19 | -- |
| | 8 | TP=8 | inverse_dynamics | **4.64** | 1.02 | 1.57 | 0.66 |
| Edge-Policy-DROID | 8 | TP=4 x CFG 2 | policy | **1.89** | 0.48 | 0.40 | 0.65 |
| | 8 | TP=4 x CFG 2 | forward_dynamics | **1.95** | 0.47 | 0.40 | 0.67 |
| | 8 | TP=4 x CFG 2 | inverse_dynamics | **4.18** | 1.71 | 0.40 | 0.65 |

On 4 cores (TP=4, Nano-Policy-DROID, 256/224/192 decode tiles): policy 4.38 s (actions only 2.83 s),
forward_dynamics 4.33 s, inverse_dynamics 6.48 s (10.89 s with the encode on one rank). TP=8 for policy:
2.93 s. Best layout on 8 cores: TP=4 x CFG-parallel 2
(`examples/cosmos3_edge/cosmos3_nano_policy_stage_trn2_tp4cfg2.yaml`), whose actions and video are
bit-identical to sequential CFG at TP=4. Inverse dynamics encodes the whole clip, so its encode is
dealt over the TP ranks (6.17 -> 1.66 s at TP=4, bit-identical); at TP=8 the encode halves again and
TP=8 is the faster inverse-dynamics layout. `extra_args["action_only"]` skips the VAE decode. Context
parallelism does not apply: action calls run unsplit (3,542 tokens here).

## Known limitations

- **Base checkpoints need their own default step count.** Cosmos3-Nano / Super (non-distilled) use a
  50-step UniPC multistep scheduler by default (`guidance_scale=7.0`). Running them at a handful of
  steps -- for example borrowing the 4-step distilled checkpoints' schedule -- visibly undersamples the
  scheduler: moire/fine striping over the whole frame and horizontal banding on sharp-color regions.
  This is NOT a device or VAE-decode defect; it reproduces purely from step-count undersampling and
  disappears entirely at the checkpoint's own default. The 4-step distilled checkpoints
  (`cosmos3-super-{i2v,t2i}-4step`) use a different scheduler (`FlowMatchEulerDiscreteScheduler`, SDE
  sampling, a fixed 4-entry `t_list`) and are correct at exactly 4 steps -- do not raise their step
  count or borrow the base checkpoints' 50. Verified: 640x640 T2I at each checkpoint's correct step
  count is visually clean for all three, with the device VAE decode cross-checked against a CPU fp32
  decode of the identical latent (PSNR 53.3-53.7 dB, max abs channel diff <= 11/255).
- **The untiled Wan VAE encoder graph does not compile on Trn2 above 192 px** (`NCC_IDDT901`).
  Handled: on NeuronCore-v3+ the conditioning frame is encoded as fixed-shape 192 px tiles (see
  Recommended configuration), which compile at every resolution. The tiled latent differs from an
  untiled encode by 3-9% (seams) -- at or below the VAE's own reconstruction error, not visible end to
  end (see Accuracy Evaluation). `COSMOS3_VAE_ENCODE=host` restores the untiled fp32 CPU encode
  (about 12 s per 640 px frame).
- **Inverse dynamics at TP=8 hung once** on its second warm request (gloo timeout in the
  tile-parallel encode's gather over the TP group); a re-run served four warm requests cleanly.
  `COSMOS3_EDGE_ENCODE_WATCHDOG_S=<s>` dumps every thread's stack if an encode exceeds `<s>` seconds.
- **Super T2I-4step at 1024x1024 shows a hard seam about 640 px down** (the image reads as two
  stacked panels). 640x640 and 832x480 are clean. Not addressed here; suspected in the GEN RoPE /
  position handling for heights above 640 px. Use 832x480 or 640x640 for T2I.
- **832x480 needs 256 px decode tiles** (see Recommended configuration); the default 480 px tile
  graph did not finish compiling in 19 minutes there. Prefer strides 240/208 (a 2 x 4 latent grid
  with no thin edge tile): strides 224/192 leave 2-row / 4-column edge slivers, which the shared
  device plane gather decodes wrong at 32 ranks on long clips (measured before the CP gather-order fix;
  not re-checked since). At 33 frames both match an fp32 CPU
  decode equally (SSIM 0.913 / 0.914, edge strips no worse) and 240/208 decodes faster (0.82 vs 1.09 s).
  The Super 4-step numbers above were measured with 224/192 and the host tile gather.
- **Long clips decode through the host tile gather** (`COSMOS3_EDGE_VAE_HOST_GATHER_T=12`: clips longer
  than 12 latent frames send each rank's decoded tiles to rank 0 as raw tensors over point-to-point
  gloo, overlapped with rank 0's own tile; bit-identical to the single-process tiled decode). The
  shared device plane gather fits HBM for a 189-frame 832x480 decode at 32 ranks (21.0 GB max) but was
  slower there (7.27 s vs 4.97 s); its accuracy at that length has not been re-measured since the CP
  gather-order fix. `vae_patch_parallel_size` must equal the stage's world size.
- Sound generation is not ported; a sound-generation request fails with the pipeline's own
  capability error, same as Cosmos3-Edge.

## Tutorials

- [Quickstart: Offline generation with Cosmos3-Nano / Super](../getting-started/quickstart-offline-serving-cosmos3-nano-super.md)
