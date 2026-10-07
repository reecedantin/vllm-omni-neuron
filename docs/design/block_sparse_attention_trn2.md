# Block-sparse attention on Trainium2 (streaming kernel)

`vllm_omni_neuron/kernels/bsa_stream.py` is a block-sparse flash-attention kernel for NeuronCore-v3 (NKI 0.6);
`vllm_omni_neuron/diffusion/attention/block_sparse.py` holds the plan, the operand builders, the fp32 reference and
the entry point `attend_stream`. It serves per-(head, query block) key-block selections such as Turbo-SLA's
top-15% of 128-row blocks and HunyuanVideo-1.5's SSTA (384-token tiles, 3x3x3 window + top-k, text tiles kept).

## How it works

- One item = one head x one query block (128 or 384 rows). Each listed key block is fetched once per item and
  applied to every 128-row sub-tile of the block.
- K^T and V of a block are stored packed (`[K^T | V]` per partition row), so a listed block is one DMA with a
  dynamic start address (hardware descriptor generation), alternating between two DMA queues; K/V passes are
  prefetched ahead of compute.
- Per pass of 1-4K keys the arithmetic is the dense `attention_cte` loop: QK^T in 512-key PSUM chunks, a masked
  copy with the chunk max, one online-softmax rescale per pass, exp with fused row sums, P^T by DMA transposes,
  P^T V accumulated in PSUM.
- Masking is by a per-pass key count: lists hold full blocks first, the single partial block last, then pads
  (which point at a real block and are masked).

## Using it from a model

Entry points in `vllm_omni_neuron/diffusion/attention/block_sparse.py`; nothing in the kernel needs touching:

```python
from vllm_omni_neuron.diffusion.attention import block_sparse as BS

# once per geometry (host, CPU): static shapes for the compiled graph
sp = BS.stream_plan(plan)              # plan: BlockSparsePlan of ONE representative selection
# per layer (device, compile-friendly, static shapes): the selection -> kernel lists
lists, bounds = BS.stream_lists(ids, counts, k_block, n_k_blocks, sp.pb, n_tail_invalid=...)
out = BS.attend_stream(q, k, v, sp.with_lists(lists, bounds), key_valid=key_valid)   # [H, Lq, 128]
# or, with a host-side plan (tests, fixed selections): BS.attend_stream(q, k, v, plan)
```

- **Shapes**: `q` `[H, Lq, 128]` = this rank's query rows (Lq = n_q_blocks * q_block); `k`, `v` `[H, Lk, 128]` = the
  CP-gathered keys (Lk = n_k_blocks * k_block, the tail block zero-padded). bf16; q is scaled by `1/sqrt(128)`
  inside (`scale=` to override). Off the device / in CPU mode `attend_stream(plan=BlockSparsePlan)` returns the fp32
  masked reference, so model CPU tests run the same plan logic.
- **Index lists** (`stream_lists`): `ids` `[H, nQ, K]` key-block ids per (head, query block), **ascending**, real
  entries first, `-1` pads after them; `counts` `[H, nQ]`. SSTA's `kept_tile_lists` output is already this format;
  SLA's top-k must be sorted (`torch.sort` of `topk` indices; counts all = topk, no pads). K is padded to a multiple of
  the pass width internally. Lists are per head (SSTA `attn_mask_share_within_head=0` works as is).
- **Validity / padding**: SLA's sequence tail: pass `n_tail_invalid = n_k_blocks * k_block - L_real` to
  `stream_lists` and `key_valid` (`[Lk]` bool) to `attend_stream`, which zeroes those rows; only the LAST key block
  may be partial (ascending order then puts it last in any list). SSTA's zero-padded tile rows are valid keys that
  stay in the softmax (upstream `pad_type="zero"`): leave `key_valid=None`, `n_tail_invalid=0`, and zero the pad
  rows of q/k/v before the call.
- **Prefix / text blocks**: no special path. SSTA: put the text tiles in every video query tile's list (the
  "always attended" rule); SLA: text/audio blocks are ordinary blocks. Do NOT route SSTA's text QUERY tiles (which
  attend everything) through the same call: the list width is the max over all items, so one all-tiles list makes
  every item cost a dense pass. Run them with `attend_dense` / attention_cte (6 tiles x 2 heads) or as a second
  `attend_stream` call with their own StreamPlan.
- **Query blocks**: 128 rows (SLA) or 384 (SSTA: one item = 3 sub-tiles sharing each K/V fetch). Items are
  (head, query block) pairs; the LNC2 pair splits them, an odd count is padded with a duplicate (handled).
- **Key blocks**: 128 or 384 rows (packed path, recommended); 64 works unpacked at ~1/4 the throughput.
- **CP**: each rank selects for its own query blocks against the full key set; nothing rank-specific is in the
  plan except which query blocks it owns. Static per geometry: `q_block`, `k_block`, `n_k_blocks`, heads, list
  width K; one compiled graph per geometry.
- **Exactness bar**: rel-L2 vs the fp32 masked reference = 1.2x the CPU-bf16 error at both rank shapes.

## Measured (one LNC=2 logical core, d = 128, bf16)

| shape (per rank, TP8 x CP8) | dense attention_cte | streaming kernel | per-key efficiency vs dense | error vs fp32 |
|---|---|---|---|---|
| HunyuanVideo-1.5 720p 121 f: 2 heads, 45 x 384 query rows, 366 x 384 key tiles, 66 kept | 36.2 ms | 9.9 ms (3.65x) | 0.62-0.66 | 1.2x the CPU-bf16 error |
| MiniMax-H3 Turbo-SLA 768p: 7 heads, 37 x 128 query rows, 296 x 128 key blocks, 44 kept | 10.1 ms | 4.1 ms (2.5x) | 0.33-0.37 | 1.2x |
| 64-row key blocks (88 of 590 kept), same rank shape | 10.0 ms | 22.2 ms | 0.07 | 1.2x |
| VSA real geometry: 7 heads, 42 pairs of 64-row query tiles, 672 x 64-row tiles, per-half masks, union 74-140 | 12.8 ms | 16.3-29.0 ms | 0.05-0.09 | 1.2x |

Blocks of 128 rows or more are recommended: the kernel is bound by DMA throughput when a block moves fewer than
~1 KB per partition (a 64-row block cannot be packed and needs two DMAs).

Tests: `test/unit/test_block_sparse_attention.py` (`NKI_SIMULATOR=1` runs the kernel on the NKI CPU simulator).
Device probe: `test/neuron/smoke_bsa_stream_trn2.py`.
