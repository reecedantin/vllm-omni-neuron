# Plan: Cosmos3 (Edge/Nano/Super), MiniMax-H3, and FastH3 on vLLM-Omni Neuron — including Trn1/Inf2

**Status**: planning document — synthesizes prior art from independent ports, not yet implemented in this repo.

> **Update (2026-10-03), superseding parts of §3.3 and §4:**
>
> - **NKI on NeuronCore-v2.** Only the *bundled* `nkilib` kernels are NC-v3+ (trn2/trn3). Hand-written NKI kernels
>   that use the NC-v2 ISA (tensor-engine transposes, no DMA transpose) run on Inferentia2 and Trainium1. Cosmos3-Edge
>   now ships one for its attention: `diffusion/models/cosmos3_edge/nki_attention_nc2.py`, selected by
>   `vllm_omni_neuron/nc_generation.py`.
> - **Cosmos3-Edge on inf2** is implemented in this repo. See `docs/models/cosmos3-edge.md`.
> - **FastH3 / MiniMax-H3.** Upstream vLLM-Omni now has a `minimax_h3` package, including a FastH3 adapter. The 33B
>   FastH3 DiT has been run on a single inf2.8xlarge (2 × 16 GB) by computing the AdaLN modulation on the host and
>   storing the block linears as int8 weight-only. Plugin integration is tracked separately.
**Scope requested**: Cosmos3-Edge, Cosmos3-Nano, Cosmos3-Super (all modalities: T2I, T2V, I2V, forward dynamics,
inverse dynamics, action/policy), MiniMax-H3 (full model, T2V + I2V), FastH3 (the MiniMax-H3 distilled variant,
T2V only, both single-node and multi-node). Also: extend `vllm-omni-neuron`'s current Trn2/Trn3-only plugin to
cover **Trn1 and Inf2**.

This repo (`vllm-omni-neuron`) today supports exactly one model family (Wan2.2 T2V/I2V) on Trn2/Trn3. Everything
below is a plan to generalize the plugin to the requested models, grounded in substantial prior-art ports found
in `~/dev/cuda-cracking/` that were built *outside* this repo's architecture (mostly as vLLM-Neuron / vLLM-Omni
integrations, or bespoke serving stacks) and must be re-shaped to fit this plugin's conventions.

---

## 1. What this repo expects from a model integration (recap)

Per `docs/design/vllm_omni_neuron_overview.md` and `docs/model-dev/onboarding-models.md`, a model is added as:

1. A package under `vllm_omni_neuron/diffusion/models/<name>/` exposing a module-level `PIPELINE_REGISTRY` list
   (`model_arch`, `class_name`, `pre_process_func_name`, `post_process_func_name`). Auto-discovered and registered
   by `vllm_omni_neuron/__init__.py::register_neuron_pipelines()` via `vllm_omni.diffusion.registry.register_diffusion_model`.
2. A **Pipeline** class subclassing the matching upstream `vllm_omni` pipeline, overriding `__init__` (device/weight
   setup) and `load_weights()` (TP-sharded loading), reusing upstream `forward()`/`encode_prompt()`/`prepare_latents()`.
3. **Model components** (transformer/DiT, VAE, text/vision encoder) rebuilt with raw `nn.Parameter` + weight-loader
   closures (the vLLM-Neuron LLaMA3 pattern) — NOT `diffusers` modules — so each layer is TP/CP-shardable and
   NKI-kernel-backed.
4. Registration of TP/CP process groups via `register_replica_groups()` so collectives legalize under
   `torch.compile`'s StableHLO lowering (Lite's mesh registry).
5. Execution is single-node, shared-memory multiproc (`MultiprocDiffusionExecutor`); parallelism (TP, CP/ring,
   CFG-parallel, VAE-patch-parallel, pipeline-parallel-as-config-field) all composes within one instance's core
   count. There is **no multi-node executor** in this repo or upstream `vllm_omni` today (`"ray"` and
   `"external_launcher"` backends are both `NotImplementedError` stubs).

Every model below needs items 1-4 rebuilt from scratch in this idiom. None of the prior-art code is a drop-in —
it was built against different plugin surfaces (bare vLLM-Neuron model packages, bare vLLM-Omni Neuron platform
code outside this repo, or no vLLM/vLLM-Omni integration at all for FastH3).

---

## 2. Model architecture summary (what we're porting)

| Model | Params | Reasoner/text backbone | Generator | Modalities in scope |
|---|---|---|---|---|
| Cosmos3-Edge | ~4B (2B dense ×2 towers) | Nemotron (SigLIP2 vision + Llama-derived decoder, mRoPE) | MoT dual-tower (UND causal self-attn + GEN cross-attn to UND K/V), Wan-style VAE | T2I, T2V, I2V, forward dynamics, inverse dynamics, action/policy |
| Cosmos3-Nano | 15.2B | Qwen3-VL-8B | Same MoT shape as Edge, scaled | same 6 modalities |
| Cosmos3-Super | 65B | Qwen3-VL-32B | Same MoT shape, scaled | same 6 modalities (deployed set validated: T2I/T2V/I2V; action modalities validated for Edge/Nano, Super's action path not separately re-verified in the trn1/trn2 work found — verify before shipping) |
| MiniMax-H3 (full) | ~33B omni DiT (FastVideo "FastH3" base) | Qwen3-VL-32B text encoder (not fused into the DiT — separate encoder) | 50-layer, 56×128-head single-stream DiT (video+text+audio one token stream), 3D causal VAE (36 attn layers) + audio decoder, Video Sparse Attention (VSA) option | T2V, I2V |
| FastH3 | same 33B DiT, **4-step distilled** (5 scheduler steps, no CFG) | same Qwen3-VL-32B encoder | same DiT/VAE, distilled scheduler | T2V only; single-node AND multi-node (2× trn2.48xlarge over EFA, TP spanning nodes) |

Key structural difference from Wan2.2 (today's only supported model): Cosmos3's generator is **not a plain DiT** —
it is a dual-stream Mixture-of-Transformers where a causal "understanding" (UND) tower's K/V (computed once,
cached) is cross-attended by a bidirectional "generation" (GEN) tower per denoising step. MiniMax-H3/FastH3 is
single-stream (closer to Wan2.2's shape) but much larger (33B vs ~14B) and introduces block-sparse attention
(VSA) as an optional path.

---

## 3. Prior art inventory — where proven code and numbers already exist

All paths relative to `~/dev/cuda-cracking/`. None of this is in the `vllm-omni-neuron` idiom; all of it is either
(a) a vLLM-Neuron model package + vLLM-Omni Neuron platform code built in **separate submodule repos**
(`vllm-neuron`, `vllm-neuron-0.28`), or (b) a bespoke non-vLLM serving stack (FastH3). Treat as reference
implementations and validated numbers, not code to import directly.

### 3.1 Cosmos3-Edge
- `cosmos-other/` (trn1 original port) + `cosmos-other/progress/trn2-*.md` (trn2 bring-up).
- Reasoner: `vllm-neuron/vllm_neuron/model/cosmos3_edge/` (config/model/weight_loaders/factory, Nemotron decoder +
  SigLIP2 vision + PatchMerger). On trn1: generic `torch.compile`-lowered attention only (NKI requires gen3+).
  On trn2: **NKI kernels still not wired for Edge** — Edge's attention is hand-rolled torch and bypasses the NKI
  import path entirely (an obsolete trn1-era `CPCollectiveMode` import workaround); this is a known, scoped gap
  (`cosmos-other/progress/trn2-phase4-bringup.md` Result 2, backlog item #2).
- Generator: `vllm-neuron-0.28/vllm_neuron/omni/` (`neuron_transformer.py`, `attention.py`, `neuron_vae.py`,
  `nc_dispatch.py`, `nki_flash_fwd.py`). This is a **host-driven compiled-graph** design (`und_pass` + `gen_step`
  graphs, geometry-set warm-up, one server per geometry set + a router) — structurally different from this repo's
  TP/CP-parameter-layer + single-executor pattern. `attention_cte_gen3` (the NKI kernel path) had a real bug on
  trn2 (padded keys not sorted to the tail) — found, fixed via host-side valid-first K/V reorder.
  Numbers: full real-time T2V/I2V pipeline validated on trn1 (one quad) and trn2 (beats H20/DGX Spark on NVIDIA's
  own I2V 480p×121f benchmark cell on just 2 trn2 devices; see `trn2-generator.md`).
- Action/forward-dynamics/inverse-dynamics modalities: validated via `DomainAwareLinear` adapter (per-embodiment
  weight selection) — confirmed **zero attention-core changes needed per modality** (`M4.10-phase4-vllm-hosting-gaps.md`
  Gap 5); this is the cleanest modality to generalize since it's purely host-side packing + one small adapter module.
- SFT/training reference (not inference, but documents the real weight-key map and model internals in detail):
  `cosmos-neuron-training/`.

### 3.2 Cosmos3-Nano / Cosmos3-Super
- `cosmos-nano-super/` (trn1 port, README/RUNBOOK/TASK.md) + `cosmos-other/progress/trn2-*.md` (trn2 bring-up,
  shared log with Edge — same branch `port/nano-super`, same two plugin repos).
- Reasoner: `vllm_neuron/model/cosmos3_omni/` — subclasses the plugin's existing `qwen3_vl` model (Nano/Super share
  Qwen3-VL backbones, unlike Edge's Nemotron), with a weight-key map UND-tower → Qwen3-VL names.
- Generator: same `vllm_neuron/omni/neuron_transformer.py`, generalized from `Cosmos3EdgeVFMTransformer` to
  `Cosmos3VFMTransformer` (one class now covers Edge/Nano/Super — this generalization already happened in prior
  art and should be preserved as the template).
- **Key new mechanism for Super**: `vllm_neuron/omni/kv_replication.py` — Omni's Cosmos3 attention shards KV heads
  as `num_kv_heads // tp_size`, capping useful TP at `num_kv_heads` (8 for Nano/Super) unless each rank gets a
  *replicated* full KV head. This module implements that replication, unlocking TP=16/32/64 for Super. Required to
  fit Super's ~65B/~130B-bf16 generator in bounded per-core HBM. **This is almost certainly needed for this repo's
  own TP machinery too** if Super's generator is to run at TP > 8 on Trn2/Trn3.
- **Reasoner-side sampling ceiling (hardware, not model-specific)**: on-device argmax/top-k kernels require
  `vocab/(4×TP) ≤ 16384`; for Nano/Super's 151936-vocab, that's `TP ≥ 4` minimum when on-device sampling is used.
  Document as a known constraint for any future on-device-sampled reasoner path.
- Numbers (trn2, real hardware, vs NVIDIA GPUs on identical AIPerf workloads): Nano/Super both beat the RTX PRO
  6000 on TTFT at concurrency 1 for every modality measured; Super beats RTX PRO 6000 and H100 NVL on batch
  throughput by 1.8× at concurrency 64; both trail B200/B300. Full matrices in
  `cosmos-other/progress/trn2-phase4-bringup.md` Results 6/7, `trn2-super-remaining.md`.
- Generator FP8 experiment (`cosmos-other/progress/trn2-p4-fp8.md`): STATIC and ROW per-tensor/per-channel fp8 for
  the generator's MLP gemms, 8-24% faster, but **failed the project's own numerics bar** (ROW: 1.68% rel/cos 0.9998
  vs. accepted ~1%/0.9999) — shipped as a labeled non-default tier, not a default. Useful groundwork, not yet a
  shippable optimization; needs either keeping the down-projection in bf16, e5m2 for outlier layers, or
  SmoothQuant-style folding before it clears the bar.
- Super action/policy modality: **not separately validated** in the material found — the Edge/Nano action work
  (DomainAwareLinear) should generalize, but confirm against a Super checkpoint before claiming support.

### 3.3 MiniMax-H3 / FastH3
- `MiniMax-H3/` — the actual model checkout (`repo/transformer`, `repo/vae`, `repo/text_encoder`, `repo/scheduler`),
  plus `vsa-trn2-kernel-handoff.md` (VSA kernel research/handoff, **not yet implemented against real FastH3 source
  in this environment** — flagged in that doc as its own open item).
- `fasth3/` — the actual working Neuron port (`fasth3_neuron/`), **not built on vLLM-Omni or vLLM-Neuron at all** —
  a from-scratch torch-xla/NxD application: `tp_model.py`, `runner.py`, `attention.py` (NKI flash kernels
  `nki_flash_rowbias.py`/`nki_flash_v2.py`/skew kernel), `vae_decoder.py`, `audio_vae.py`, `vsa.py` (block-sparse
  VSA, including a real **custom gen3 NKI kernel** `nki_bsa_gen3.py` getting to 2.62ms/layer — VSA was found to be
  *slower than dense* with the compiler-path kernel but faster with the hand-written one; still not fully wired
  into the production path at verdict time), a resident serving daemon (`fasth3_neuron/serve/`), web UI, and an
  HLS livestream publisher.
- **Real-time results, confirmed by direct measurement** (`fasth3/notes/claude-memory/fasth3-port-status.md`):
  - trn1 (32 cores): 256p ~3.0s/clip, 384×640 ~4.5-5.2s/clip (both under real-time for a 5s clip).
  - trn2 single node (64 logical cores, LNC=2): 720p 6.4s/clip (1.24× real-time) via `nkilib.core.attention.attention_cte`.
  - **trn2 × 2 nodes over EFA (TP=8×CP=16 spanning both nodes — NOT independent replicas)**: 768p real-time
    confirmed at 5s (~5.9s end-to-end for a 5.17s clip) and 10s (~10.3-10.5s end-to-end, i.e. *faster* than
    real-time); 15s clips land at ~1.27× real-time (attention's quadratic cost catching up at longer lengths).
  - EFA setup gotcha that blocked this for a while: SRD frames never match a security-group CIDR rule — needs an
    **outbound self-referencing SG rule**, not just inbound.
  - VSA verdict as of the last check: 1.6× *slower* than dense on trn2 at 768p/5s with the compiler-path kernel;
    the hand-written gen3 NKI kernel (`nki_bsa_gen3.py`) measured faster in isolation (2.62ms/layer) but wasn't
    fully validated end-to-end in the two-node run. Treat VSA as **not production-ready**; ship dense attention
    first, revisit VSA as a follow-on optimization once the kernel is finished and gated.
- **This is the biggest architectural gap to close**: unlike Cosmos3 (which has existing vLLM-Neuron/vLLM-Omni
  model packages to adapt), FastH3/MiniMax-H3 has **no vLLM/vLLM-Omni integration of any kind** to start from.
  The transformer, attention kernels, and VAE math are proven; the entire `PIPELINE_REGISTRY`/pipeline-class/
  weight-loader layer for this repo's conventions has to be built from scratch, informed by (not copied from)
  `fasth3_neuron/`.
- VSA kernel handoff doc (`MiniMax-H3/vsa-trn2-kernel-handoff.md`) is a separate, more theoretical NKI-primitive
  mapping exercise (gpsimd_topk, fused gather-then-attend modeled on `mla_sparse_attention_cte_kernel`) that
  predates/overlaps the actually-working `nki_bsa_gen3.py` in `fasth3_neuron/vsa.py` — reconcile the two before
  starting any VSA kernel work; the working kernel in `fasth3_neuron/` is the more authoritative source since it's
  measured on hardware.

---

## 4. Trn1 / Inf2 support — gap analysis

This repo (`vllm_omni_neuron/platform.py`, `lite_compat.py`) is Trn2/Trn3-only today. Key facts, confirmed by direct
read of this repo and by the trn1 prior-art ports:

- **NKI (the kernel toolkit this repo's attention/MLP/norm kernels under `kernels/nkilib/` are built on) does not
  support Trainium1 or Inferentia2 at all.** Confirmed independently by two separate prior-art investigations
  (Cosmos3-Edge's trn1 port and FastH3's trn1 port): NKI kernel compilation on trn1 fails with a documented,
  non-fixable `nc-version >= gen3` assertion. This is a hard platform wall, not a version/flag issue.
- Therefore **every NKI-kernel-backed component in this repo (ring-attention CP, the Wan2.2 attention/MLP/norm
  kernels under `kernels/nkilib/`) cannot run on Trn1/Inf2 as written.** A Trn1/Inf2 path needs a **generic
  `torch.compile`-lowered fallback attention/MLP/norm path**, parallel to the existing NKI path, selected by
  platform detection — not a port of the NKI kernels themselves.
- Prior art already proves this fallback works and is the *only* viable attention path on trn1: both the
  Cosmos3-Edge port (`vllm-neuron-0.28/vllm_neuron/omni/attention.py`'s `torch` attention impl, the
  `COSMOS3_OMNI_ATTN_IMPL` dispatch) and FastH3 (`fasth3_neuron/attention.py`'s hand-written skew/rowbias NKI
  kernels — those *are* trn1-legal since they're hand-authored NKI kernels targeting gen2 ISA, not the bundled
  `nkilib` which the above wall applies to) ran real, correct, real-time-adjacent inference on trn1 this way.
  Note the distinction: **the *bundled* `nkilib` kernels are gen3+-only; hand-authored NKI kernels targeting gen2
  ISA (no `nisa.dma_transpose`, PE-only transposes) remain legal on trn1** — FastH3's own kernel work is the
  existence proof. A trn1 path in this repo could reuse that same two-tier approach: generic `torch.compile`
  fallback for correctness everywhere, with hand-authored gen2 NKI kernels as a trn1-specific optimization layer
  if attention throughput becomes the bottleneck (as it did for FastH3 on trn1).
- **Memory/topology differences to encode in platform detection** (`NeuronOmniPlatform.get_device_total_memory`,
  `get_platform_target`): trn1 is 16GB/NeuronCore-v2, no LNC concept, aligned-quad collective topology rules
  (1, 4, 8, or 16 aligned devices only); trn2 is 96GB/chip with LNC=1/2 and a different, also-constrained quad/mesh
  rule (`override_groups_with_physical_mesh` in `diffusion/worker/diffusion_worker.py` already encodes the trn2
  8×8 mesh case — trn1's rule needs its own branch). Inf2's HBM/topology profile has not been characterized in any
  of the prior art found; treat as unverified and gate behind explicit testing, not inference from trn1/trn2.
- **vLLM-Neuron's own version support matters here too**: the reasoner-serving half of Cosmos3 depends on
  vLLM-Neuron, whose *current* release line is Trn2/Trn3-only; Trn1/Inf2 support exists only on vLLM-Neuron's
  legacy `0.5.3` "Maintenance" branch (older NxD Inference dependency). Any Trn1/Inf2 reasoner path in this
  ecosystem is constrained by whichever vLLM-Neuron line is actually installed — this is an external dependency
  risk, not something this repo's plugin code can route around.
- Compile cost is categorically worse on trn1 for large graphs (longer wall-clock per geometry; the FastH3 783 note
  about per-geometry 4-25min compiles pre-dates the trn2 move, which didn't change the order of magnitude much
  either) — plan for cache warm-up time, not something to optimize away.

### Proposed Trn1/Inf2 integration shape

1. Add a **capability-detection layer** parallel to `lite_compat.get_platform_target()`: a small helper that
   reports NKI-kernel availability (not just "what chip," since the same chip generation gate recurs across every
   attention/MLP/norm kernel call site) so model code can branch once, not per-kernel.
2. For every NKI-kernel call site this repo's Wan2.2 code and the new Cosmos3/MiniMax-H3 code introduce
   (`ring_attention_const_max_fwd`, the QKV/MLP/norm `nkilib` kernels, any new Cosmos3 `attention_cte`-style
   kernel), add a `torch.compile`-only fallback branch, gated on the capability check from (1) — mirroring
   `attention/backends/sdpa.py`'s existing `NeuronSDPABackend` pattern, which is already platform-generic.
3. Extend `NeuronOmniPlatform` (`platform.py`) with trn1/Inf2 entries in `get_hbm_memory_gb`'s target table and a
   trn1-specific collective-topology branch alongside the existing trn2 8×8-mesh override.
4. Treat Inf2 as **unverified by this plan** — none of the prior art tested Inf2 directly (only trn1 and trn2 were
   actually run). Scope Inf2 as "should work once trn1's generic fallback path exists, since Inf2 and trn1 share
   the gen2 NKI/no-NKI-kernel constraint," but require an explicit hardware validation pass before claiming
   support, not just extrapolation from trn1 numbers.
5. Sequencing: build the generic-fallback attention/MLP/norm paths as part of the *first* model's integration
   (whichever lands first — see §5 priority order) rather than as a separate retrofit; every model's integration
   plan below should include "does this run through the generic path on trn1" as an explicit checklist item, not
   a follow-up.

---

## 5. Proposed integration plan

### 5.1 Package layout (per this repo's existing convention)

```
vllm_omni_neuron/diffusion/models/
├── cosmos3/                     # shared base: Cosmos3VFMTransformer (MoT core), shared by Edge/Nano/Super
│   ├── pipeline_cosmos3.py      # one Pipeline class, backbone-parameterized (mirrors upstream vllm_omni's
│   │                            #   single-pipeline-many-backbones shape — see upstream reference below)
│   ├── transformer_cosmos3.py   # Cosmos3VFMTransformer: UND tower (causal) + GEN tower (cross-attn), TP/CP-sharded
│   ├── reasoner_nemotron.py     # Edge's Nemotron+SigLIP2 UND backbone
│   ├── reasoner_qwen3vl.py      # Nano/Super's Qwen3-VL UND backbone (reuses upstream vllm_omni Qwen3-VL math where possible)
│   ├── kv_replication.py        # ported mechanism: TP > num_kv_heads support (needed for Super)
│   ├── modality_adapters.py     # action (DomainAwareLinear), forward/inverse-dynamics adapters
│   └── vae_wan.py                # Cosmos3's Wan-style VAE (reuse/extend wan2_2's VAE code where the format matches)
├── minimax_h3/
│   ├── pipeline_minimax_h3.py   # full model: T2V + I2V
│   ├── transformer_minimax_h3.py # 50-layer single-stream DiT, 56x128 heads
│   ├── vae_minimax_h3.py        # 3D causal VAE, 36 attn layers
│   ├── audio_vae.py              # audio decoder (full model only — not required for FastH3's T2V-only scope)
│   └── attention_vsa.py          # optional block-sparse path — ship OFF by default per the "VSA not production
│                                  #   ready" finding in §3.3; dense attention is the default and the initial target
└── fasth3/
    ├── pipeline_fasth3.py        # T2V only; thin subclass/config of minimax_h3's pipeline (4-step distilled scheduler)
    └── multinode/                 # multi-node launch/topology helpers (see 5.3)
```

Reuse `minimax_h3`'s transformer/VAE/attention code for `fasth3` via config (distilled scheduler, no-CFG, 4-step)
rather than forking the model code — the two are the same architecture, FastH3 is MiniMax-H3 distilled.

### 5.2 Suggested build order

1. **Cosmos3-Edge, T2I/T2V/I2V first.** Smallest model (easiest to iterate compile cycles on), has the most
   complete trn1+trn2 reference (reasoner, generator, and the NKI-bug-fix precedent to follow), and forces solving
   the hardest *new* architectural problem (the UND/GEN dual-tower + cross-attention-to-cached-K/V shape) at the
   smallest scale before scaling to Nano/Super.
2. **Edge action/forward-dynamics/inverse-dynamics modalities.** Low incremental cost once T2I/T2V/I2V works
   (confirmed zero attention-core changes needed) — do this before moving to Nano/Super so the modality adapter
   package is validated once and reused.
3. **Cosmos3-Nano**, reusing the dual-tower core from step 1, swapping in the Qwen3-VL UND backbone and the six
   modalities already proven for Edge.
4. **Cosmos3-Super**, same transformer code, adding `kv_replication.py` for TP>8 and validating the heavier
   compile/memory budget. Re-verify action-family modalities specifically for Super (flagged unverified in §3.2).
5. **Trn1/Inf2 generic-fallback path**, built against whichever of steps 1-4 is in progress at the time — do not
   defer this to the end; land it alongside Edge (step 1) per §4 item 5, since Edge has the most direct trn1
   precedent to validate against.
6. **MiniMax-H3 (T2V, single node)**, standing up the net-new pipeline/transformer/VAE package from scratch,
   informed by `fasth3_neuron/`'s proven math but written in this repo's parameter/weight-loader idiom. Ship dense
   attention only (no VSA) at this stage.
7. **MiniMax-H3 I2V.**
8. **FastH3 (T2V, single node)**, as a configuration of step 6/7's transformer (distilled 4-step scheduler).
9. **FastH3 multi-node (2× trn2.48xlarge, TP spanning nodes over EFA)** — this is the one capability with **no
   analog anywhere else in this repo or upstream `vllm_omni`**, since both are single-node-executor-only today
   (§1). See §5.3 for what's actually required.
10. **VSA (block-sparse attention) for MiniMax-H3/FastH3**, only after the dense path on both is shipped and only
    if the kernel work (`nki_bsa_gen3.py`-style, gen3-targeted) clears both a correctness gate and a real speedup
    over dense — not before, per the "VSA is currently a performance regression" finding.

### 5.3 Multi-node support — the one genuinely new capability

FastH3's two-node result spans a TP×CP group across two physical hosts over EFA — something neither this repo's
`MultiprocDiffusionExecutor` nor upstream `vllm_omni`'s executor abstraction supports today (§1: only the
single-host `"mp"` backend is implemented; `"ray"`/`"external_launcher"` are stubs).

Two paths, in increasing order of integration depth:

- **(a) Minimal — document as an external launch pattern, not a plugin capability.** FastH3's own two-node setup
  does not use vLLM-Omni's executor/scheduler at all; it is `torchrun`-launched processes with manually-managed
  `NEURON_RT_ROOT_COMM_ID`/`MASTER_ADDR`/EFA env vars, outside any framework's distributed abstraction. The
  equivalent minimal path here would be: run this repo's existing single-node worker code unchanged, but launch it
  via an external multi-node bootstrap (torchrun `--nnodes=2` or similar) that sets up the cross-host process group
  before `NeuronDiffusionWorker.init_device()` runs its (currently always-localhost) distributed init. This needs
  the smallest code change — primarily, stop hardcoding `MASTER_ADDR=localhost` and `nnodes=1` in
  `diffusion_worker.py`, and parameterize them from the launch environment.
- **(b) Deeper — a real multi-node `DiffusionExecutor` backend.** Implement the `"ray"` or `"external_launcher"`
  backend stub in upstream `vllm_omni` (or a repo-local subclass passed via the `distributed_executor_backend`
  dotted-path mechanism `DiffusionExecutor.get_class()` already supports) that can spawn/coordinate workers across
  hosts, handling EFA bootstrap and the SG self-reference gotcha from §3.3 as setup prerequisites, not per-request
  logic.

Recommend starting with (a) for FastH3 specifically (matches the proven, measured prior art exactly, lowest risk)
and treating (b) as a separate, larger follow-on effort if multi-node becomes a repeated need across models rather
than a FastH3-specific one-off.

---

## 6. Open risks and unresolved questions (carry into implementation)

1. **VSA correctness/perf is unresolved.** Ship dense attention first for MiniMax-H3/FastH3; do not promise VSA
   speedups in any model card until the gen3 kernel (`nki_bsa_gen3.py`-style) clears both a numerics gate and a
   measured win over dense on the target hardware.
2. **Super's action-family modalities are unverified.** Confirm against a real Super checkpoint before listing
   forward/inverse-dynamics/action as supported for Super specifically.
3. **FP8 for the Cosmos3 generator fails this project's own numerics bar as currently implemented** (§3.2) — do
   not ship as default; either fix (bf16 down-proj / outlier-layer e5m2 / SmoothQuant) or ship as an explicitly
   labeled, opt-in lower-precision tier only.
4. **Edge gets no NKI kernel benefit on trn2 today** — its attention path needs the `CPCollectiveMode` re-wiring
   mentioned in §3.1 before Edge reasoner serving is kernel-accelerated like Nano/Super already are. Decide whether
   this blocks the Edge milestone (step 1) or ships as a fast-follow once Edge's correctness path is validated.
5. **Inf2 is unverified by any prior art found** — do not claim Inf2 support without an explicit hardware pass; the
   plan in §4 is an informed extrapolation from trn1, not a tested result.
6. **Multi-node is a genuinely new capability for this plugin and for upstream `vllm_omni`.** Scope and effort for
   §5.3 should be estimated and agreed separately before committing to the FastH3 multi-node milestone's timeline.
7. **The 1:1-aspect-ratio generator slowdown anomaly** found during Super's trn2 video sweep (640×640 at 1.43× the
   expected per-call time, unexplained) should be re-investigated if Cosmos3's T2I/T2V modalities are exercised at
   square aspect ratios — note the risk, don't block on root-causing it before this plan's build-out starts.
8. **Compile-cache budgeting**: several of the prior-art geometries (Super's 720p×189-frame video, 480p+ Edge I2V)
   had multi-minute-to-tens-of-minutes cold compiles. Any CI/validation plan for these models needs to budget for
   this explicitly (persistent compile caches, not fresh-cache gating on every run).

---

## 7. References

- This repo: `docs/design/vllm_omni_neuron_overview.md`, `docs/model-dev/onboarding-models.md`,
  `docs/design/context_parallelism.md`.
- Upstream `vllm_omni`'s own (CUDA-only) Cosmos3 pipeline — the closest architectural reference for the
  `PIPELINE_REGISTRY`/pipeline-class shape, even though it has no Neuron code:
  `vllm_omni/diffusion/models/cosmos3/{pipeline_cosmos3.py,transformer_cosmos3.py,transformer_cosmos3_edge.py}`.
- Cosmos3-Edge: `~/dev/cuda-cracking/cosmos-other/` (trn1), `~/dev/cuda-cracking/cosmos-other/progress/trn2-*.md`
  (trn2 bring-up), `~/dev/cuda-cracking/vllm-neuron/vllm_neuron/model/cosmos3_edge/`,
  `~/dev/cuda-cracking/vllm-neuron-0.28/vllm_neuron/omni/`.
- Cosmos3-Nano/Super: `~/dev/cuda-cracking/cosmos-nano-super/` (README/RUNBOOK/TASK.md), same two plugin repos on
  branch `port/nano-super`, `~/dev/cuda-cracking/cosmos-other/progress/trn2-phase4-bringup.md`,
  `trn2-generator.md`, `trn2-super-remaining.md`, `trn2-p4-fp8.md`.
- MiniMax-H3/FastH3: `~/dev/cuda-cracking/MiniMax-H3/` (model checkout + `vsa-trn2-kernel-handoff.md`),
  `~/dev/cuda-cracking/fasth3/` (`fasth3_neuron/`, `TRN2-HANDOFF.md`, `RESTORE.md`,
  `notes/claude-memory/fasth3-port-status.md` and sibling notes).
