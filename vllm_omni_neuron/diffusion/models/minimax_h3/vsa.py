# SPDX-License-Identifier: Apache-2.0
"""Video Sparse Attention (VSA-H3) for MiniMax-H3 students distilled with it (FastH3 8-Step-V2, sparsity 0.8).

Semantics follow FastVideo's ``fastvideo/attention/backends/video_sparse_attn_h3.py`` ("exempt" mode, tile 64):

* The packed ``[text | audio | video]`` sequence is tiled into **segment-pure prefix tiles** (each dense segment in
  64-row chunks, the last chunk partial) followed by **3-D video tiles** of ``(4, 4, 4)`` tokens over the
  ``(T, H/2, W/2)`` token grid, in ``(t-tile, h-tile, w-tile)`` order, partial at the edges. Pad slots are zero.
* **Coarse stage**: per-tile masked mean of Q and K (fp32), tile scores ``Qc Kc^T / sqrt(d)`` per head.
* **Mask**: prefix query tiles attend every key; video query tiles attend every prefix tile plus their top-k video
  tiles, ``k = max(1, min(ceil((1 - sparsity) * n_video_tiles), n_video_tiles))``. Pad key slots are masked.
* **Fine stage**: attention restricted to the selected tiles.
* **Compression branch**: ``softmax(tile scores) @ pooled V`` broadcast to each tile's rows and multiplied by the
  learned per-token gate ``to_gate_compress(attention input)``, added to the fine output.

Two implementations share the host-built :class:`VSAGeometry`:

* :func:`vsa_attention_reference` -- explicit token-level mask over the whole slot sequence (CPU oracle).
* :func:`vsa_attention_cp` -- the same for context parallelism: the sequence is already in slot order (the DiT's
  residual stream is permuted into it), each rank holds whole query tiles and attends to the all-gathered keys.
  Pad tiles appended to make the tile count divisible by the CP degree are never selected.
* :func:`vsa_attention` -- static-shape, query-blocked masked attention for the compiled graph. Top-k is ``k``
  argmax passes (neuronx-cc does not reliably compile the sort ``topk`` lowers to). It computes the full score matrix
  and masks it, so it is exact but not faster than dense attention -- VSA is a correctness requirement for this
  checkpoint here, not an optimisation (the earlier Neuron ports measured every sparse formulation slower than dense).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

TILE = (4, 4, 4)
TILE_ELEMS = 64
NEG = -1e9
QBLOCK = int(
    os.environ.get("MINIMAX_H3_VSA_QBLOCK", "2048")
)  # query slots per block on the compiled path
FINE_FP32 = (
    os.environ.get("MINIMAX_H3_VSA_FINE_FP32", "0") == "1"
)  # fine-stage Q.K^T in fp32 (default: compute dtype)


def compute_topk(sparsity: float, num_blocks: int) -> int:
    return max(1, min(math.ceil((1 - sparsity) * num_blocks), num_blocks))


@dataclass
class VSAGeometry:
    n_tiles: int
    n_prefix: int
    n_video: int
    k_vid: int
    slot_to_row: torch.Tensor  # (n_slots,) long; pad slots -> L (an appended zero row)
    valid: torch.Tensor  # (n_slots,) float 1/0
    tile_sizes: torch.Tensor  # (n_tiles,) float
    row_to_slot: torch.Tensor  # (L,) long
    tile_of_slot: torch.Tensor  # (n_slots,) long
    seq_len: int

    @property
    def n_slots(self) -> int:
        return self.n_tiles * TILE_ELEMS

    @staticmethod
    def build(
        dense_segments: tuple[int, ...],
        video_grid: tuple[int, int, int],
        sparsity: float,
        tile: tuple[int, int, int] = TILE,
    ) -> VSAGeometry:
        """``dense_segments``: row counts of the leading dense segments in packed order (t2va: text, audio);
        ``video_grid``: the generated video's ``(T, H/2, W/2)`` token grid, packed last, frame-major."""
        te = math.prod(tile)
        rows: list[torch.Tensor] = []
        cursor = 0
        for seg in dense_segments:
            for s in range(0, seg, te):
                rows.append(torch.arange(cursor + s, cursor + min(s + te, seg)))
            cursor += seg
        n_prefix = len(rows)
        t, h, w = video_grid
        grid = torch.arange(t * h * w).reshape(t, h, w) + cursor
        ts, hs, ws = tile
        for ti in range(math.ceil(t / ts)):
            for hi in range(math.ceil(h / hs)):
                for wi in range(math.ceil(w / ws)):
                    rows.append(
                        grid[
                            ti * ts : ti * ts + ts, hi * hs : hi * hs + hs, wi * ws : wi * ws + ws
                        ].flatten()
                    )
        seq_len = cursor + t * h * w
        n_tiles = len(rows)
        n_video = n_tiles - n_prefix
        slot_to_row = torch.full((n_tiles * te,), seq_len, dtype=torch.long)
        valid = torch.zeros(n_tiles * te)
        sizes = torch.zeros(n_tiles)
        row_to_slot = torch.zeros(seq_len, dtype=torch.long)
        for i, r in enumerate(rows):
            n = r.numel()
            slot_to_row[i * te : i * te + n] = r
            valid[i * te : i * te + n] = 1.0
            sizes[i] = n
            row_to_slot[r] = torch.arange(i * te, i * te + n)
        assert bool((sizes > 0).all()) and int(valid.sum()) == seq_len
        tile_of_slot = torch.arange(n_tiles).repeat_interleave(te)
        return VSAGeometry(
            n_tiles,
            n_prefix,
            n_video,
            compute_topk(sparsity, n_video),
            slot_to_row,
            valid,
            sizes,
            row_to_slot,
            tile_of_slot,
            seq_len,
        )

    def pad_tiles(self, multiple: int) -> VSAGeometry:
        """Append empty tiles (size 0, every slot invalid) so ``n_tiles`` is a multiple of ``multiple``; the real
        tiles, their order and ``k_vid`` are unchanged (pad tiles sit after the last video tile)."""
        extra = -self.n_tiles % multiple
        if extra == 0:
            return self
        n = extra * TILE_ELEMS
        return VSAGeometry(
            self.n_tiles + extra,
            self.n_prefix,
            self.n_video,
            self.k_vid,
            torch.cat([self.slot_to_row, torch.full((n,), self.seq_len, dtype=torch.long)]),
            torch.cat([self.valid, torch.zeros(n)]),
            torch.cat([self.tile_sizes, torch.zeros(extra)]),
            self.row_to_slot.clone(),
            torch.arange(self.n_tiles + extra).repeat_interleave(TILE_ELEMS),
            self.seq_len,
        )

    def to(self, device) -> VSAGeometry:
        return VSAGeometry(
            self.n_tiles,
            self.n_prefix,
            self.n_video,
            self.k_vid,
            self.slot_to_row.to(device),
            self.valid.to(device),
            self.tile_sizes.to(device),
            self.row_to_slot.to(device),
            self.tile_of_slot.to(device),
            self.seq_len,
        )


def _to_slots(x: torch.Tensor, g: VSAGeometry) -> torch.Tensor:
    """(L, h, d) rows -> (n_slots, h, d) slot order, pad slots zero."""
    xz = torch.cat([x, x.new_zeros((1, *x.shape[1:]))], dim=0)
    return xz.index_select(0, g.slot_to_row) * g.valid.to(x.dtype)[:, None, None]


def _pool(slots: torch.Tensor, g: VSAGeometry) -> torch.Tensor:
    """(n_slots, h, d) -> (h, n_tiles, d) fp32 masked mean."""
    pooled = (
        slots.float().view(g.n_tiles, TILE_ELEMS, *slots.shape[1:]).sum(dim=1)
        / g.tile_sizes[:, None, None]
    )
    return pooled.permute(1, 0, 2)


def _tile_scores(qs, ks, g):
    d = qs.shape[-1]
    return torch.matmul(_pool(qs, g), _pool(ks, g).transpose(-1, -2)) / math.sqrt(
        d
    )  # (h, nt, nt) fp32


def _tile_mask(scores: torch.Tensor, g: VSAGeometry, use_topk: bool) -> torch.Tensor:
    """(h, nt, nt) fp32 scores -> bool tile mask, FastVideo's exempt mode."""
    if g.k_vid >= g.n_video:
        return torch.ones_like(scores, dtype=torch.bool)
    vid = scores[..., g.n_prefix :]
    if use_topk:
        sel = torch.zeros_like(vid, dtype=torch.bool).scatter(
            -1, vid.topk(g.k_vid, dim=-1).indices, True
        )
    else:  # k argmax passes: static shapes, no sort
        ar = torch.arange(g.n_video, device=scores.device)
        sel = torch.zeros_like(vid, dtype=torch.bool)
        s = vid
        for _ in range(g.k_vid):
            hit = ar == s.argmax(dim=-1, keepdim=True)
            sel = sel | hit
            s = s.masked_fill(hit, float("-inf"))
    mask = torch.cat([torch.ones_like(scores[..., : g.n_prefix], dtype=torch.bool), sel], dim=-1)
    is_prefix_q = (torch.arange(g.n_tiles, device=scores.device) < g.n_prefix)[None, :, None]
    return mask | is_prefix_q


def _compress(scores, vs, g, gate_s):
    out_c = torch.matmul(torch.softmax(scores, dim=-1), _pool(vs, g))  # (h, nt, d) fp32
    return (
        out_c.index_select(1, g.tile_of_slot).permute(1, 0, 2).to(gate_s.dtype) * gate_s
    )  # (n_slots, h, d)


@torch.no_grad()
def vsa_attention_reference(q, k, v, gate, g: VSAGeometry) -> torch.Tensor:
    """CPU oracle. q/k/v/gate ``(L, h, d)`` (any float dtype; computed in fp32) -> ``(L, h, d)`` fp32."""
    q, k, v = q.float(), k.float(), v.float()
    qs, ks, vs = _to_slots(q, g), _to_slots(k, g), _to_slots(v, g)
    scores = _tile_scores(qs, ks, g)
    tmask = _tile_mask(scores, g, use_topk=True)
    tok = (
        tmask.index_select(1, g.tile_of_slot).index_select(2, g.tile_of_slot)
        & g.valid.bool()[None, None, :]
    )
    d = q.shape[-1]
    s = torch.matmul(qs.permute(1, 0, 2), ks.permute(1, 2, 0)) / math.sqrt(d)  # (h, ns, ns)
    out = torch.matmul(
        torch.softmax(s.masked_fill(~tok, NEG), dim=-1), vs.permute(1, 0, 2)
    ).permute(1, 0, 2)
    if gate is not None:
        out = out + _compress(scores, vs, g, _to_slots(gate.float(), g))
    return out.index_select(0, g.row_to_slot)


def vsa_attention(q, k, v, gate, g: VSAGeometry) -> torch.Tensor:
    """Compiled-graph path. q/k/v/gate ``(L, h, d)`` in the compute dtype -> ``(L, h, d)`` in that dtype.
    Scores and softmax in fp32, P@V in the compute dtype; query slots processed in blocks of ``QBLOCK``."""
    dt, d = q.dtype, q.shape[-1]
    qs, ks, vs = _to_slots(q, g), _to_slots(k, g), _to_slots(v, g)
    scores = _tile_scores(qs, ks, g)
    tmask = _tile_mask(
        scores, g, use_topk=q.device.type == "cpu" and not torch.compiler.is_compiling()
    )
    key_valid = g.valid.bool()[None, None, :]
    qh, kt, vh = (
        qs.permute(1, 0, 2),
        ks.permute(1, 2, 0),
        vs.permute(1, 0, 2),
    )  # (h, ns, d), (h, d, ns), (h, ns, d)
    h, ns = qh.shape[0], qh.shape[1]
    outs = []
    for s0 in range(0, ns, QBLOCK):  # static loop, unrolled into the graph
        s1 = min(s0 + QBLOCK, ns)
        tq = g.tile_of_slot[s0:s1]
        m = tmask.index_select(1, tq)  # (h, qb, nt)
        m = (
            m.unsqueeze(-1).expand(h, s1 - s0, g.n_tiles, TILE_ELEMS).reshape(h, s1 - s0, ns)
            & key_valid
        )
        if FINE_FP32:
            sc = torch.matmul(qh[:, s0:s1].float(), kt.float()) / math.sqrt(d)
        else:
            sc = torch.matmul(qh[:, s0:s1], kt).float() / math.sqrt(d)
        p = torch.softmax(sc.masked_fill(~m, NEG), dim=-1)
        outs.append(torch.matmul(p.to(dt), vh))
    out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)).permute(1, 0, 2)  # (ns, h, d)
    if gate is not None:
        out = out + _compress(scores, vs, g, _to_slots(gate, g))
    return out.index_select(0, g.row_to_slot)


def vsa_attention_cp(
    q, k, v, gate, g: VSAGeometry, valid_local, sizes_local, prefix_q
) -> torch.Tensor:
    """Context-parallel VSA. ``q`` / ``gate`` ``(n_local, h, d)``: this rank's slots (whole tiles, slot order);
    ``k`` / ``v`` ``(n_slots, h, d)``: every rank's slots (all-gathered, slot order) of the tile-padded geometry ``g``.
    ``valid_local`` / ``sizes_local`` / ``prefix_q``: this rank's slot validity, tile sizes (pad tiles clamped to 1)
    and prefix-tile flags (float 1 / 0). Same math as :func:`vsa_attention` restricted to this rank's query
    tiles; pad tiles (size 0) are excluded from the top-k and the compression softmax. Returns ``(n_local, h, d)``
    in slot order."""
    dt, d = q.dtype, q.shape[-1]
    ntl, nt, h = q.shape[0] // TILE_ELEMS, g.n_tiles, q.shape[1]
    vl, vk = valid_local.to(dt)[:, None, None], g.valid.to(dt)[:, None, None]
    qs, ks, vs = q * vl, k * vk, v * vk
    sizes = g.tile_sizes.clamp(min=1.0)[:, None, None]
    pq = (qs.float().view(ntl, TILE_ELEMS, h, d).sum(dim=1) / sizes_local[:, None, None]).permute(
        1, 0, 2
    )
    pk = (ks.float().view(nt, TILE_ELEMS, h, d).sum(dim=1) / sizes).permute(1, 0, 2)
    pv = (vs.float().view(nt, TILE_ELEMS, h, d).sum(dim=1) / sizes).permute(1, 0, 2)
    scores = torch.matmul(pq, pk.transpose(-1, -2)) / math.sqrt(d)  # (h, ntl, nt) fp32
    scores = scores.masked_fill(
        (g.tile_sizes <= 0)[None, None, :], NEG
    )  # pad tiles: never selected, no weight
    if g.k_vid >= g.n_video:
        tmask = torch.ones_like(scores, dtype=torch.bool)
    else:
        vid = scores[..., g.n_prefix :]
        if q.device.type == "cpu" and not torch.compiler.is_compiling():
            sel = torch.zeros_like(vid, dtype=torch.bool).scatter(
                -1, vid.topk(g.k_vid, dim=-1).indices, True
            )
        else:  # k argmax passes: static shapes, no sort
            ar = torch.arange(vid.shape[-1], device=scores.device)
            sel = torch.zeros_like(vid, dtype=torch.bool)
            s = vid
            for _ in range(g.k_vid):
                hit = ar == s.argmax(dim=-1, keepdim=True)
                sel = sel | hit
                s = s.masked_fill(hit, float("-inf"))
        tmask = torch.cat(
            [torch.ones_like(scores[..., : g.n_prefix], dtype=torch.bool), sel], dim=-1
        )
        tmask = tmask | (prefix_q > 0.5)[None, :, None]
    key_valid = g.valid.bool()[None, None, :]
    qh, kt, vh = qs.permute(1, 0, 2), ks.permute(1, 2, 0), vs.permute(1, 0, 2)
    nq, ns = qh.shape[1], kt.shape[-1]
    tile_q = torch.arange(ntl, device=q.device).repeat_interleave(TILE_ELEMS)
    outs = []
    for s0 in range(0, nq, QBLOCK):  # static loop, unrolled into the graph
        s1 = min(s0 + QBLOCK, nq)
        m = tmask.index_select(1, tile_q[s0:s1])  # (h, qb, nt)
        m = m.unsqueeze(-1).expand(h, s1 - s0, nt, TILE_ELEMS).reshape(h, s1 - s0, ns) & key_valid
        if FINE_FP32:
            sc = torch.matmul(qh[:, s0:s1].float(), kt.float()) / math.sqrt(d)
        else:
            sc = torch.matmul(qh[:, s0:s1], kt).float() / math.sqrt(d)
        p = torch.softmax(sc.masked_fill(~m, NEG), dim=-1)
        outs.append(torch.matmul(p.to(dt), vh))
    out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)).permute(1, 0, 2)  # (nq, h, d)
    if gate is not None:
        out_c = torch.matmul(torch.softmax(scores, dim=-1), pv)  # (h, ntl, d) fp32
        out = out + out_c.index_select(1, tile_q).permute(1, 0, 2).to(gate.dtype) * (gate * vl)
    return out


class VSAReferenceProcessor:
    """diffusers attention processor for ``MiniMaxH3Transformer3DModel`` blocks implementing VSA-H3 with
    :func:`vsa_attention_reference` (fp32). Used for the CPU oracle; ``holder["geom"]`` is set per layout."""

    def __init__(self, gate_weight: torch.Tensor | None, holder: dict):
        self.gate_weight = gate_weight  # (inner, hidden) or None
        self.holder = holder

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from ._vendor.transformer_minimax_h3 import _apply_rotary_emb

        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        gate = None
        if self.gate_weight is not None:
            w = self.gate_weight.to(hidden_states.dtype)
            gate = torch.nn.functional.linear(hidden_states, w).unflatten(-1, (attn.heads, -1))[0]
        out = vsa_attention_reference(q[0], k[0], v[0], gate, self.holder["geom"])
        return attn.to_out[0](out.to(q.dtype).unsqueeze(0).flatten(2, 3))


def install_reference_processors(model, gate_weights: list[torch.Tensor | None]) -> dict:
    """Swap every transformer block's processor for :class:`VSAReferenceProcessor`; returns the shared holder
    (set ``holder["geom"]`` to a :class:`VSAGeometry` before each forward). Refiner blocks stay dense."""
    holder: dict = {"geom": None}
    for blk, gw in zip(model.transformer_blocks, gate_weights, strict=True):
        blk.attn.set_processor(VSAReferenceProcessor(gw, holder))
    return holder
