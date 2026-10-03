# PoC: Cosmos3-Edge MoT generator serving through vLLM-Omni Neuron — Inf2 first, Trn2 alongside

**Status**: planning document for a proof-of-concept, not yet implemented.
**Goal**: prove that this repo's *actual* architecture (single-node `MultiprocDiffusionExecutor`,
`Pipeline` + raw-parameter `nn.Module` transformer, TP/CP replica-group registration) can serve a
Mixture-of-Transformers (MoT) model — Cosmos3-Edge's dual-tower generator — not just Wan2.2's
single-stream DiT. Starting hardware: **1× inf2.8xlarge** (1 Inferentia2 chip, 2 NeuronCore-v2, no
LNC, 32 GB HBM total / 16 GB per core, TP=2 ceiling). Trn2 support is built into the same code from
day one (not bolted on after); Inf2 is the harder, more memory-constrained target we validate first
because it forces the generic fallback path to exist and forces every memory decision to be
deliberate. Once Inf2 is correct, move to Trn2 to validate the kernel-accelerated path and compare
against the already-measured Trn2 numbers from the non-`vllm-omni-neuron` prior art.

This document assumes the reader has `docs/design/planned-cosmos3-minimax-h3-fasth3-integration.md`
(the full five-model plan) — this is a narrower, sequenced first slice of it: Cosmos3-Edge only,
generator only (no reasoner — see §1), T2I first.

---

## 1. Scope: generator only, on purpose

`vllm-omni-neuron` is a **diffusion-only** plugin — it has no LLM/text-generation stage, no
`StageType.LLM_GENERATION`, no chat-completion surface. Confirmed by direct inspection: the repo has
exactly two subpackages (`diffusion/`, `kernels/`), and `NeuronOmniPlatform` only implements
diffusion-worker/model-runner extension points. Cosmos3-Edge's **reasoner** (text/VQA) is served
through the separate `vllm-neuron` plugin in every piece of prior art found (`vllm_neuron/model/cosmos3_edge/`,
`cosmos3_omni/`) — that is a different repo, a different serving stack, and out of scope here.

**This PoC is about the generator tower only**: the diffusion transformer that this repo's
`PIPELINE_REGISTRY` / `Pipeline` / weight-loader idiom is built for. Where the generator's UND
(understanding) sub-tower needs to run — because Cosmos3's architecture couples a causal text tower
to the diffusion tower via cross-attention to cached K/V, not a separate reasoner call — see §3.

---

## 2. Why Cosmos3-Edge is the right first target, and why Inf2 first

- **Edge is the smallest Cosmos3 variant** (~4B combined, two 2B-dense towers) — fastest compile
  iteration, smallest memory footprint, most forgiving on a 16 GB/core budget.
- **Edge is architecturally representative of the hard part**: Cosmos3's generator is a genuine
  Mixture-of-Transformers — a causal UND tower (self-attention, computed once, K/V cached) and a
  bidirectional GEN tower (cross-attends to the cached UND K/V, run once per denoising step). This is
  structurally different from Wan2.2 (today's only supported model: a single-stream DiT with plain
  cross-attention to a frozen text encoder's output, no causal tower, no step-persistent KV cache).
  Proving MoT serving on Edge first means every later model (Nano, Super, and eventually anything
  else with a cached-KV-plus-cross-attention shape) inherits a validated pattern instead of
  reinventing it at larger, slower-to-iterate scale.
- **Inf2 before Trn2, deliberately**: Inf2 and Trn1 share NeuronCore-v2 — the *same* "NKI requires
  nc-version ≥ gen3" wall documented in the Trn1/Inf2 gap analysis. Starting on Inf2 forces the
  **generic `torch.compile`-only attention/MLP/norm path** to exist and be correct before any NKI
  kernel work happens, and forces the UND/GEN cross-attention-to-cached-KV mechanism to be expressed
  in a way that doesn't implicitly depend on an NKI-only primitive. Validating MoT serving on the
  *harder, more constrained* platform first is the stronger proof; Trn2 then mostly needs the kernel
  dispatch added, not a redesign.
- **Inf2's TP=2 ceiling is a feature for this PoC, not just a constraint**: it is the smallest
  non-trivial TP degree this repo's `register_replica_groups`/mesh code supports, so it's the cheapest
  real multi-rank collective test. Trn2's physical-mesh TP/CP/CFG group logic
  (`diffusion/distributed/parallel_state.py`) is validated only for its 8×8 fabric cases today —
  Inf2's single-chip, 2-core case exercises the *other* (arithmetic, non-mesh) code path, which is
  worth confirming works correctly regardless of which platform ships first.

---

## 3. Where does the UND tower run?

Cosmos3's architecture requires the UND tower's K/V, computed from the text prompt (and, for Edge,
SigLIP2 vision input), to be available to the GEN tower's cross-attention for every denoising step.
Two prior-art designs exist, both outside this repo's architecture, and neither is what we want to
copy structurally — only the *math* and *weight maps* are reusable:

- **vllm-neuron-0.28/omni's design** (`~/dev/cuda-cracking/vllm-neuron-0.28/vllm_neuron/omni/`): a
  host-driven pair of compiled graphs, `und_pass` (UND prefill → K/V) and `gen_step` (GEN cross-attn,
  called once per denoising step), with the UND pass's output held in Python between calls. This is
  architecturally a *different* execution model from `vllm-omni-neuron`'s single `Pipeline.forward()`
  call per request.
- **The Cosmos3-Nano/Super M4.12 investigation's finding** (`cosmos3/docs/epics/M4.12-...md` §1.1,
  found during the five-model plan's research): vLLM-Omni's own diffusion stage type has **no**
  built-in mechanism for "run an LLM-shaped causal prefill once, then reuse its KV across many
  diffusion forward passes" — that pattern belongs to `StageType.LLM_GENERATION`'s scheduler, not the
  diffusion path this plugin implements.

**Proposed approach for this PoC, consistent with how this repo's Wan2.2 pipeline already works**:
run the UND prefill **inside** `Pipeline.forward()`, as a first sub-step before the denoising loop —
exactly analogous to how `NeuronWanPipeline.forward()` runs the UMT5 text encoder once before its
denoising loop and holds the result (`encoder_hidden_states`) for every step. The UND tower is just a
"text/vision encoder" from this pipeline's point of view; the only difference from UMT5 is that its
output is a per-layer K/V cache (one pair per GEN-tower cross-attention layer) rather than a single
hidden-state tensor, and it is causal rather than bidirectional. This fits the existing
`Pipeline.__init__`/`forward()` shape with no new engine-level mechanism required — the UND tower is
simply another model component the pipeline owns and calls once, like the VAE and text encoder
already are.

This is deliberately the opposite choice from vllm-neuron-0.28/omni's external, host-driven
`und_pass`/`gen_step` split: that split exists because *their* design serves the UND tower through a
live, scheduler-managed vLLM `LLM` instance with paged KV cache shared across many unrelated
requests. This PoC's UND tower has no such requirement — it is private to one diffusion request, so
it can be a plain, un-paged forward pass inside the pipeline, same as any other encoder. If a future
need arises to serve Cosmos3's reasoner and generator from one shared KV cache (as the M4.12 doc's
project did), that is a different, larger integration decision — out of scope here.

---

## 4. Hardware budget check (Inf2.8xlarge)

| | value |
|---|---|
| HBM total | 32 GB (16 GB / NeuronCore-v2, 2 cores) |
| TP | 2 (fixed — one chip, two cores, no LNC) |
| Edge combined weights (UND + GEN towers, bf16) | ~8 GB |
| Per-core weights at TP=2 | ~4 GB |
| Host RAM | 128 GB — plenty for CPU-side fallback work (VAE, calibration, oracle generation) |

Headroom at TP=2 (~4 GB/core weights, 16 GB/core budget) is generous relative to the trn1 precedent
(Edge reasoner alone needed real care to fit at TP=2 on a 16 GB core; the *generator* tower here is
comparable in size to the reasoner, so expect a similar, survivable budget) — but the generator adds
KV cache for the UND tower (sized to prompt length, small for T2I) and denoising-loop activation
memory, which the reasoner-only prior art doesn't have to budget for. **VAE decode is the open
question**: the Trn2 prior art found that even 480p VAE *encode* blows the compiler's SBUF budget on
a 24 GB/core Trn2 chip (`cosmos-other/progress/trn2-generator.md`'s root-cause entry) — on a 16 GB/
core Inf2 chip, on-device VAE for anything beyond a small T2I frame is very likely infeasible without
spatial tiling. **Plan**: run VAE decode (and encode, if I2V is attempted later) on the **host CPU**
for this PoC, exactly as the Edge generator prior art did as its first, deliberately-simple choice
(`M4.10-phase4-vllm-hosting-gaps.md` Gap 2's "real, first choice") — defer on-device/tiled VAE to a
follow-up once correctness is established.

**Modality scope for this PoC: T2I only.** T2I is the smallest token count (single frame, no video
VAE temporal causality, shortest denoising-relevant sequence), making it the cheapest way to validate
the UND/GEN cross-attention mechanism end-to-end before adding video's longer sequences or I2V's
conditioning-image VAE encode. T2V, I2V, and the action-family modalities (forward dynamics, inverse
dynamics, policy) are explicitly deferred past this PoC — they are additive once the core mechanism
works (per the five-model plan's finding that action-family modalities need no attention-core
changes, only host-side packing + a small adapter module).

---

## 5. Attention/MLP/norm kernel path

Per the Trn1/Inf2 gap analysis: NKI (the kernel toolkit backing this repo's `kernels/nkilib/`) is
gen3+-only. Inf2's NeuronCore-v2 cannot use it — not a flag, a hard compiler wall. This PoC's
transformer code must therefore be written with **capability-dispatched attention/MLP/norm**, exactly
per the gap analysis's proposed shape:

- A small capability check (parallel to `lite_compat.get_platform_target()`) that reports whether
  NKI kernels are usable on the current platform.
- Every attention/MLP/norm call site in the new Cosmos3 transformer code branches on that check:
  generic `torch.compile`-lowered path (works everywhere, including Inf2 and Trn1) vs. NKI-kernel
  path (Trn2/Trn3 only). Model this on the existing `attention/backends/sdpa.py`'s
  `NeuronSDPABackend` pattern, which is already platform-generic.
- **On Inf2 this PoC only exercises the generic path — there is no NKI fallback-of-a-fallback tier
  available**, unlike Trn1, which at least has FastH3's proven "hand-authored gen2 NKI kernel" tier as
  a future option. That tier is unverified on Inf2 and out of scope for this PoC; if Inf2 generic-path
  performance turns out to matter, treat a gen2 NKI kernel port as a separate, later effort.
- **On Trn2, the same code should additionally dispatch to NKI kernels** where available — this is
  the "Trn2 alongside" half of this PoC's scope. The specific kernel to target is
  `attention_cte` (gen3's dispatch target in the vllm-neuron-0.28/omni prior art,
  `nc_dispatch.attention_impl()`), with the **padded-keys-must-sort-valid-first bug already found and
  fixed in that prior art** (`cosmos-other/progress/trn2-generator.md`'s root-cause entry) carried
  forward as a known pitfall to avoid re-discovering. Do not copy that fix's code (different package
  shape), but do copy the understanding: **padded/masked keys passed to `attention_cte` must be
  sorted or packed so valid keys form a contiguous prefix** — the kernel's `bound_max` masking
  assumes this and silently drops valid keys / attends padding otherwise.

---

## 6. Package layout (within this PoC's reduced scope)

Following `docs/design/planned-cosmos3-minimax-h3-fasth3-integration.md` §5.1's proposed layout, but
scoped to exactly what this PoC needs — no Nano/Super backbone abstraction yet, no modality adapters
beyond T2I:

```
vllm_omni_neuron/diffusion/models/cosmos3/
├── pipeline_cosmos3_edge.py     # PIPELINE_REGISTRY entry; Pipeline.__init__ builds UND+GEN towers +
│                                #   host-side VAE wrapper; forward() = UND prefill once, then the
│                                #   denoising loop calling the GEN tower per step, then host VAE decode
├── transformer_cosmos3_edge.py  # Cosmos3EdgeUndTower (Nemotron+SigLIP2, causal, KV-cache-emitting)
│                                #   and Cosmos3EdgeGenTower (bidirectional, cross-attends cached K/V),
│                                #   both built from raw nn.Parameter + weight-loader closures (the
│                                #   vLLM-Neuron LLaMA3 pattern this repo's Wan2.2 code already uses)
├── attention_dispatch.py        # the capability check from §5 + the generic torch.compile path;
│                                #   NKI/attention_cte dispatch added once Trn2 validation starts
└── vae_host.py                  # thin host-CPU VAE decode wrapper (§4) — not TP/CP-sharded, runs on
                                  #   rank 0 only, matching this repo's existing Wan2.2 VAE rank-0 convention
```

Weight-key mapping (checkpoint → this package's parameter names) and the Nemotron/SigLIP2/mRoPE
details (no QK-norm, ReLU² activation, interleaved mRoPE `[24,20,20]` theta 1e8, alternating
attention-only/MLP-only blocks) should be taken directly from the already-validated prior art
(`vllm-neuron/vllm_neuron/model/cosmos3_edge/` and the trn2 NKI port notes in
`cosmos-other/progress/trn2-edge-nki.md`) — these are checkpoint facts, not design choices, and
re-deriving them from scratch would be redundant and error-prone against already-correct references.

---

## 7. Validation gates (project convention, carried forward)

Every piece of prior art examined for this effort used the same discipline — **CPU oracle → device
parity → served output**, never promoting a number before its gate. Apply it here:

1. **CPU oracle**: run the real Cosmos3-Edge T2I path (HF `transformers`/`diffusers`-level, matching
   whichever reference the checkpoint ships, e.g. `Cosmos3EdgeForConditionalGeneration`-style modeling
   code or the `cosmos-framework` reference) on CPU, fp32, capturing step-0 velocity and the final
   image for a fixed prompt/seed.
2. **Device parity, teacher-forced**: run this PoC's ported UND+GEN pipeline on Inf2 at TP=2, with the
   same inputs forced at each denoising step (not free-run), compare per-call velocity error against
   the CPU oracle. Use the same acceptance class every prior-art port used: roughly 1-3% relative
   error / cosine ≥ 0.999 for a correct bf16 kernel-vs-fp32-CPU comparison; investigate anything
   outside that band rather than accepting it, per the Edge Stage 4 precedent where a 5.4% gap was
   root-caused rather than shrugged off.
3. **Served output, free-run**: run the full pipeline through `Omni.generate()` end-to-end (no
   teacher forcing), confirm a coherent image, deterministic across repeated runs with the same seed
   (md5-identical final latents/image, the convention every prior-art port used to confirm no
   nondeterminism crept in).
4. Repeat all three gates on Trn2 once the Inf2 path is correct, additionally confirming the NKI
   kernel dispatch path matches the generic path's parity class (same discipline the trn2 Edge NKI
   port used when comparing kernel-attention vs. torch-attention token streams).

---

## 8. Sequencing

1. Build the UND tower (Nemotron text-only path first — no vision encoder yet) with the generic
   attention/MLP path, get CPU-oracle parity on a plain forward pass (no diffusion yet — just confirm
   the UND tower alone reproduces the reference's causal LM hidden states / cached K/V correctly).
2. Add the SigLIP2 vision encoder + PatchMerger to the UND tower, re-gate parity with an image input
   (mirrors the Edge reasoner prior art's own Stage 1 → Stage 2 sequencing).
3. Build the GEN tower (bidirectional cross-attention to the UND tower's cached K/V), wire the T2I
   denoising loop and host-side VAE decode inside `Pipeline.forward()`, gate teacher-forced parity.
4. Validate TP=2 on real Inf2 hardware: weight sharding, replica-group registration, collective
   legality for the arithmetic (non-mesh) 2-rank case.
5. Free-run served T2I on Inf2, confirm determinism, declare the PoC's Inf2 half done.
6. Add the capability dispatch's NKI branch, targeting `attention_cte` on Trn2, carrying forward the
   padded-keys-sort-valid-first fix from §5. Re-run all three gates on Trn2.
7. Compare Trn2 served-output timing against the already-measured, non-`vllm-omni-neuron` prior art
   numbers (`cosmos-other/progress/trn2-generator.md`'s Edge T2I-class figures) as a sanity check that
   this repo's architecture isn't leaving obvious performance on the table relative to the external-
   orchestration design — not a hard requirement to match exactly (different execution models), but a
   useful signal if this PoC's numbers are far worse.
8. Only after this PoC's T2I path is validated on both platforms: hand off to the five-model plan's
   broader sequencing (Edge's remaining modalities, then Nano, then Super, etc.) per
   `docs/design/planned-cosmos3-minimax-h3-fasth3-integration.md` §5.2.

---

## 9. Open risks specific to this PoC

1. **Inf2 is unverified by any prior art examined for this effort.** Every number and every fix
   referenced above (NKI gen3 wall, the `attention_cte` padding bug, VAE SBUF limits) was found on
   Trn1 or Trn2 hardware, never Inf2. Treat every Inf2-specific claim in this document as a
   reasoned extrapolation, not a confirmed result, until actually run.
2. **No prior art validates a diffusion MoT model through `vllm-omni-neuron`'s specific
   `Pipeline.forward()`-owns-the-encoder shape** — the UND-tower-as-encoder design in §3 is this
   document's own proposal, not a port of something already proven to work in this exact shape.
   Budget real design/debug time for this, not just a mechanical port.
3. **Host-CPU VAE decode latency is unmeasured for this specific checkpoint/shape.** The Edge
   generator prior art measured CPU VAE decode cost as "a real, separate, additive latency term" but
   didn't commit a specific number into any document reviewed; expect to measure this fresh.
4. **TP=2's arithmetic (non-mesh) replica-group path in this repo is exercised today only by
   Wan2.2's smaller-TP configurations**, if at all — confirm it is actually reachable and correct for
   a from-scratch model before assuming it "just works" because the code exists.
5. **Compile time on Inf2 is unknown.** Trn1 (same NeuronCore-v2 generation) saw cold compiles of
   single-digit minutes for comparable reasoner graphs; budget similarly for the generator here, but
   don't assume parity without measuring — the diffusion graph shape differs from a reasoner's.

---

## 10. References

- This repo: `docs/design/vllm_omni_neuron_overview.md`, `docs/model-dev/onboarding-models.md`,
  `docs/design/context_parallelism.md`, `docs/design/planned-cosmos3-minimax-h3-fasth3-integration.md`
  (the full five-model plan this PoC is a slice of).
- Edge reasoner weight/architecture facts and the Trn2 NKI kernel port (attention_cte bug, FFN
  SquaredReLU library limitation, norm fusion, TP-dependent kernel eligibility):
  `~/dev/cuda-cracking/vllm-neuron/vllm_neuron/model/cosmos3_edge/`,
  `~/dev/cuda-cracking/cosmos-other/progress/trn2-edge-nki.md`.
- Edge generator design precedent (host-driven und_pass/gen_step split — reference for *math*, not
  structure) and the attention_cte padding bug + VAE SBUF root-cause:
  `~/dev/cuda-cracking/vllm-neuron-0.28/vllm_neuron/omni/`,
  `~/dev/cuda-cracking/cosmos-other/progress/trn2-generator.md`.
- M4.12's finding that vLLM-Omni's diffusion stage has no built-in cached-KV-across-steps mechanism
  (motivating §3's design choice): `~/dev/cuda-cracking/cosmos3/docs/epics/M4.12-production-serving-vllm-omni-dp8.md` §1.1.
- Trn1/Inf2 NKI wall and the generic-fallback precedent: this repo's
  `docs/design/planned-cosmos3-minimax-h3-fasth3-integration.md` §4, and the trn1 Edge/FastH3 ports
  cited there.
