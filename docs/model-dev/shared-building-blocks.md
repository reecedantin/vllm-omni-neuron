# Shared building blocks for porting diffusion models

Model-agnostic utilities that a new port can reuse instead of re-deriving. Each one is unit tested on
CPU (`test/unit/test_shared_utils.py`) and the device-facing ones are exercised by
`test/neuron/smoke_platform_trn2.py`.

## NeuronCore generation and kernel dispatch

`vllm_omni_neuron.nc_generation` is the single gate for NKI kernels:

- `neuron_core_generation()`: 2 (Inf2/Trn1), 3 (Trn2), 4 (Trn3). Detection order: the
  `VLLM_OMNI_NEURON_CORE_GEN` override, the Lite platform target, the Neuron driver's sysfs
  architecture record, else 2 with a warning.
- `supports_nki()`: NKI needs NeuronCore-v3+. `use_nki_kernels(tensor)` also requires a device
  tensor and honours `VLLM_OMNI_NEURON_ATTN_IMPL=torch`.

Branch every attention / MLP / norm kernel call site on these, never on an instance name. The result
is a compile-time constant inside `torch.compile` graphs.

## N-block graph splitting

`vllm_omni_neuron.diffusion.layers.block_graphs.BlockGraphRunner` runs a stack of structurally
identical blocks as compiled N-block graphs that share one graph. It traces a weightless template
block with `torch.func.functional_call` and passes each chunk's weights and per-layer side inputs as
graph inputs, so all full chunks lower to the same graph (one compile, one resident NEFF). A shorter
trailing chunk adds one more graph.

```python
runner = BlockGraphRunner(
    model.blocks, group_size=5,
    compile_fn=lambda f: torch.compile(f, backend=backend, fullgraph=True, dynamic=False, options=opts),
    block_call=lambda blk, x, shared, layer, kw: blk(x, *shared, table=layer[0]),
)
with torch.no_grad():
    x = runner(x, rope, per_layer=[(t,) for t in tables])
```

Use it when the full graph overflows the compiler (SBUF / instruction limit) or its compile host RAM
is too high. Raise `group_size` until it compiles: each chunk boundary costs a graph launch and loses
cross-block fusion.

## Host-precomputed modulation tables

`vllm_omni_neuron.diffusion.layers.modulation_tables` bakes each block's AdaLN / timestep modulation
(`proj(act(temb))`) on the host for a fixed step schedule, so the modulation weights never go to the
device. `bake_linear_modulation` reproduces the device rounding (act in fp32, bf16 input, fp32
accumulation, bf16 output, bf16 bias add). `ModulationTables` caches per step in memory and
optionally on disk, can stream weights layer by layer from a checkpoint, and `device_tables()`
returns one device tensor per layer. Pass those per layer (for example through `BlockGraphRunner`'s
`per_layer`); indexing one stacked tensor inside a graph bakes the offset in and splits the graph.
Tables are only valid for the schedule they were built for.

## int8 weight-only linear (W8A16)

`vllm_omni_neuron.diffusion.quantization.int8_weight_only` provides `Int8WeightOnlyLinear` and
`quantize_linears_(module, include=...)`: per-output-channel symmetric int8 weights, matmul in the
activation dtype, fp32 per-channel scale on the output. It halves weight HBM. It is not a speedup
on NeuronCore-v2, and on Trn2 it is only worth it when the model is memory-bound. Keep modulation
projections out of it (bake them to tables instead).

## Text-encoder phase and prompt-embedding cache

`vllm_omni_neuron.diffusion.layers.embedding_cache`: `TextEncoderPhase` serves prompt embeddings from
a host LRU (`PromptEmbeddingCache`, optional disk layer), encodes only the misses in one batch, and
can release the encoder after a precompute pass (reloading lazily on a miss). The key covers the
encoder identity, the prompt and any setting that changes the output (max length, template).

## Shared autoregressive decode-attention layer

`vllm_omni_neuron.diffusion.attention.decode_attention` is for text towers embedded in a diffusion
pipeline that decode autoregressively (Cosmos3's Qwen3-VL reasoner, Qwen-Image 2.1's
prompt-enhancer LLM, pi0.5.2's text subtask) -- model-agnostic over hidden size, head count and
cache length, with a static-shape KV cache sized for one request (no block-table paging: a diffusion
request runs one sequence through, not a multi-tenant LLM server).

- `DecodeAttentionConfig`: head counts (already TP-sharded by the caller), head_dim, max_len, dtype.
- `DecodeWeights.from_separate(q_proj, k_proj, v_proj, out_proj)`: packs a tower's existing
  `nn.Linear`-style QKV/output weights into the layout `attention_decode` expects (a reshape/concat,
  no retraining).
- `StaticKVCache(cfg, device)`: the `[1, kv_heads, max_len, head_dim]` cache. The fill position
  `pos` is an `int32[1]` DEVICE tensor (read `fill`, a host int, only outside compiled code): with
  a Python int Dynamo specialized the compiled `decode_step` on it, so every decode position was a
  different graph -- a retrace plus a NEFF compile/lookup per token (0.3 s warm, 2.4 s on a cache
  miss in rounds 15-18) and a `fullgraph=True` recompile-limit failure on the ninth token. Since
  round 19 the step is ONE graph for every position: the cache write is a one-hot selection built
  from `pos` (exact), attention runs over the whole `max_len` cache under a position mask (unfilled
  slots masked out like the causal mask), and nothing slices at the position. Overflow past
  `max_len` cannot raise inside the graph; the write is dropped (size the cache for the request).
- `decode_prefill(cfg, cache, q, k, v, cos=, sin=)`: the whole prompt, causal, through SDPA with an
  explicit additive causal mask (default; an opt-in NKI `attention_cte` path exists). Writes the cache.
- `decode_step(cfg, weights, cache, hidden, cos=, sin=)`: one call (or a small `S_tkg` block) of new
  tokens. DEFAULT: the torch path (project, RoPE, write the new K/V, attend against the masked
  full cache, GQA-expand, project out) -- device-verified on Trn2 (smoke rounds 15-18: decode rel
  0.0037-0.0059 vs the CPU fp32 oracle at the Qwen3-VL 16/2-head shape, cache read back correct).
  OPT-IN
  (`VLLM_OMNI_NEURON_DECODE_STEP_NKI=1`): `vllm_neuron`'s `attention_decode` -- the production fused
  `attention_block_tkg` kernel (RMSNorm -> QKV -> RoPE -> GQA attention -> KV-cache update -> output
  projection in one NEFF) on NC-v3+. It was off because on device it returned an output uncorrelated
  with the oracle (rel 1.03) at that GQA shape while reading the same verified cache. Smoke round 17's
  bisect (`decode_attention_kernel_diag`) found the cause: the kernel addresses a 4D cache
  `[num_blocks, kv_heads, block_len, d]` as a flat `[num_blocks * kv_heads]` pool, so a per-head block
  table entry must be `block * kv_heads + head`, and this layer's table was all zeros -- every query
  head read KV head 0 (MHA 1.28 and real GQA 0.95 wrong, GQA with identical KV heads 0.0067 right).
  `StaticKVCache.active_blocks_table()` now returns `arange(kv_heads)`; **round 18 confirmed it on
  device: MHA 0.0067, GQA 0.0070 (CPU-bf16 band 0.0063)**. The kernel stays opt-in for now (the torch
  path is the verified default every adopter is on; flipping the default is a separate decision).
  `can_use_decode_kernel` checks the kernel's shape constraints (`batch * S_tkg * q_heads <= 128`,
  `H` a multiple of 128, even head_dim, `max_len` a multiple of 128) ahead of the call.

Decode one token at a time for any tower with more than 128 query heads (`cfg.max_decode_tokens()`
reports the limit); none of the three target towers need more than one call per block in practice.


`python -m vllm_omni_neuron.tiny_models SRC OUT --layers 2` writes a copy of a real checkpoint with
the same folder layout, parameter names, dtypes and non-weight files (tokenizers, schedulers,
processors), random weights, and every block list cut to `--layers`. It proves loading, name
mapping, sharding, compile and a device forward in minutes; it says nothing about quality.

- `synthesize` (default) reads only the source safetensors headers and needs no model code, so it
  works for classes the installed libraries lack. Widths stay real. Layer lists are cut when their
  length matches a layer-count field (`num_layers`, `num_hidden_layers`, `depth`, ...) of the
  folder's own `config.json`, or of the nearest ancestor's when the folder has none.
- `--mode <component>=instantiate --set <component>.<key>=<value>` builds the class (diffusers
  `_class_name` / transformers `architectures[0]`) from a shrunk config, which lets you shrink widths.
  Each tensor keeps the dtype of the same-named source tensor.
- Every `*.safetensors.index.json`, including root-level ones that point into component folders,
  is rewritten to point at the tiny files. `tiny_manifest.json` records what was done, and
  `tiny_models.compare_layout(tiny, real, component)` checks names and dtypes against the source.

Example (Cosmos3-Edge, 13 s, 1.7 GB): `--layers 2 --mode vae=instantiate --set vae.base_dim=16
--set vae.decoder_base_dim=32` gives a 2-layer UND tower and vision encoder plus a 10.5M-param VAE.
The plugin's UND tower loads it and runs a forward on CPU. Write tiny checkpoints outside the
repository and never commit them; commit the command or script that generates them.

## Neighborhood (windowed / NATTEN / Swin) attention

`vllm_omni_neuron.diffusion.attention.neighborhood_attention.neighborhood_attention_tiled` is a
device-legal replacement for NATTEN-style neighborhood attention (each token attends over a fixed
`kernel`-sized window), used by Swin3D / NATTEN VAEs (FLUX-3-Action's video VAE, Wan's). The obvious
formulations all fail `neuronx-cc` on NeuronCore-v3: an `index_select` window gather
(`NCC_EBIR026`), a dense `masked_fill` band (`NCC_IMPR902`), and a dense additive band in a query
loop (`NCC_IMPR902` too).

It uses **halo tiling**: partition the grid into fixed `tile`-sized query blocks; each block attends
to a fixed `tile + 2*(kernel-1)` key window around it (halo = `kernel - 1` per side covers NATTEN's
inward border clamp), zero-padded at the edges. Every tile is then an identical small dense attention
-- one static shape, pure `matmul` + `softmax`, no gather, no `[N, N]` mask. The only mask is a small
per-tile additive bias (`[tile_tokens, window_tokens]`), a host constant shared by all tiles.

```python
# q/k/v: [B, *axes, heads, head_dim]  (heads-last, NATTEN layout)
out = neighborhood_attention_tiled(q, k, v, kernel=[5, 5], causal=[False, False])  # tile=None
```

`kernel`/`causal`/`tile` are per spatial axis (1D/2D/3D). `causal=True` on an axis uses NATTEN's
causal window `(i-kernel, i]`; non-causal axes clamp the window inward at the borders. Semantics are
bit-exact against the dense masked reference. The per-tile key-window token count is
`prod(tile + 2*(kernel-1))`, independent of the grid size -- that bound is what makes it compile
where a full `N^2` band does not. Leave `tile=None`: the default (`pad_free_tile`) picks, per axis,
the largest divisor of the axis `<= 16`, so no axis has a partially padded last tile (FLUX-3-Action's
136x184 grid -> 8x8), which keeps the final crop out of the graph.

**Device status (Trn2, smoke rounds 13-18).** The interior of the op has matched the CPU reference
to the bf16 band (0.0036) since round 13, but the fused single graph was wrong in a tiling-dependent
region (16x16: the last tile row; 8x8: everything but the last tile column) with the original
`narrow` + `stack` key/value windowing. Round 17's single-graph bisect showed the same graph correct
with host-windowed operands, with fp32 inputs and with a host crop -- i.e. the compiler mis-fuses
the overlapping strided bf16 window reads with their consumer, not the arithmetic. The default
windowing is therefore `window_impl="select"`: each overlapping key/value window axis is produced by
ONE matmul against a 0/1 selection matrix (exact in any dtype; no overlapping slice in the graph;
~25 GFLOP at the 136x184 grid). Round 18 confirmed it on device: **0.0036 overall, border rows
0.0035, pad-free 8x8 0.0032** (CPU-bf16 band 0.0032). The selection matrices are built in-graph
from `arange` + compare (a few static integer ops; no module-level cache -- round 18 found a dict
cache consulted inside the trace makes Dynamo recompile on every call, 58-75 s per call), or passed
as host constants: `select=neighborhood_select_matrices(axes, kernel, tile, dtype, device)`, the
same pattern as `bias=`. Round 19 times both (gate: one Dynamo graph, warm < 100 ms at 136x184).
`window_impl="slice"` keeps the old formulation (correct on CPU and as a graph of its own; wrong
once fused on Trn2 in bf16) and `upcast_first=True` casts q/k/v to fp32 at the graph entry; both
are diagnostic knobs, not device paths (round 18: `select`+upcast fails to compile `NCC_ILSA902`,
`slice`+upcast is wrong in the first tile column). Until round 19's timing is in, keep a host (CPU)
neighborhood attention as the fallback in your pipeline, the way FLUX-3-Action does.

## All-rank agreement check for gates

A multi-rank layout is only verified when every rank ends with the same output: a host gather in the
wrong order (the unsorted physical-mesh CP / CFG groups, see `host_all_gather` in
`diffusion/distributed/parallel_state.py`) leaves rank 0 right at the first call and the other
ranks wrong, so a rank-0 check passes. `vllm_omni_neuron.testing.rank_agreement` digests each
rank's output (shape, dtype, SHA-256 of the bytes, non-finite count, fp64 sum / L2 / max-abs, a
fixed strided sample of 4096 values) and reports every rank that disagrees with rank 0.

In-process (every rank of the group calls it at the same point; every rank gets the same report):

```python
from vllm_omni_neuron.testing import check_rank_agreement

rep = check_rank_agreement({"latents": latents, "video": video})  # world group (vLLM cpu_group)
rep = check_rank_agreement(latents, coord=get_tp_group())         # one coordinator, its list order
if rep.rank == 0:
    print(rep.summary())  # "rank agreement OK: 16 ranks, [latents,video] bit-exact" / which ranks, why
summary["rank_agreement"] = rep.to_json()  # ok, disagreeing_ranks, max_rel, rank-0 sha256 per output
rep.raise_if_failed()
```

From separate worker processes the gate cannot call into (served path), write one file per rank and
compare after the run; a rank with no file fails the check:

```python
write_rank_digest(out_dir, rank, {"latents": latents})           # in each worker
rep = compare_rank_digest_files(out_dir, world_size=16)         # in the gate, after the run
```

Digest only tensors that should be identical on every rank: replicated outputs, or a CP shard
after it is gathered. `rtol=0` (default) requires bit-identical bytes, which replicated TP / CP / CFG
outputs give when every rank runs the same graphs; pass a small `rtol` only for an output reduced in
a rank-dependent order, and state it in the gate. Device tensors are copied to the host before any
cast. Unit test: `test/unit/test_rank_agreement.py` (4 gloo ranks, one drifting).
