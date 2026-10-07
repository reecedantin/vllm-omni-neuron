# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape compute graphs for the pi0.5 / pi0.52 action path on Neuron.

The vendored upstream model (``_vendor/pi05/modeling_pi05.py``) stays the parameter owner and
the CPU reference. These modules hold references to its sub-modules (so they share the same
parameters and checkpoint names) and re-express the two hot paths as plain tensor math with
static shapes, which is what ``torch.compile`` on Neuron wants:

* :class:`Pi05PrefixGraph`: SigLIP over every camera slot + projector, the language
  embedding, and the PaliGemma LM over the whole prefix. Returns the post-RoPE K/V of every
  layer stacked into two ``[L, B, Hk, P, D]`` tensors, which stay on the device for the whole
  denoising loop. The last layer's MLP is skipped: only its K/V is consumed downstream.
* :class:`Pi05DenoiseGraph`: one flow-matching step of the AdaRMS action expert over the
  ``chunk_size`` action tokens, attending to the cached prefix. Returns ``v_t`` in fp32.

Numerics: residual streams, norms, RoPE and softmax are fp32; linear layers run in their
weight dtype (bf16 on device). Masks are additive fp32 biases, so fully padded rows stay finite.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large finite negative for masked keys (OpenPI's constant; finite so no NaN rows).
MASK_VALUE = -2.3819763e38


def _lin(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """Linear in the layer's weight dtype; result returned in fp32."""
    w = layer.weight
    y = F.linear(x.to(w.dtype), w, None if layer.bias is None else layer.bias.to(w.dtype))
    return y.float()


def _all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    """Sum ``x`` over the tensor-parallel ``group`` (in fp32); identity without a group."""
    if group is None:
        return x
    import torch.distributed as dist

    x = x.float()
    dist.all_reduce(x, group=group)
    return x


def _row_parallel(x: torch.Tensor, layer: nn.Linear, group) -> torch.Tensor:
    """``layer(x)`` (fp32) for a layer whose INPUT features are sharded over ``group``: the local
    partial product, summed over the group."""
    return _all_reduce(_lin(x, layer), group)


def _lm_logits(p, x: torch.Tensor, lm_head: nn.Linear) -> torch.Tensor:
    """LM-head logits (fp32). Under tensor parallelism ``lm_head`` holds this rank's vocabulary
    rows; each rank places its slice in a zero full-vocabulary row (``p.tp_slot`` one-hot, the
    same graph on every rank) and one all-reduce assembles the full logits."""
    local = _lin(x, lm_head)
    if p.tp_group is None:
        return local
    full = torch.cat([local * p.tp_slot[r] for r in range(p.tp_size)], dim=-1)
    return _all_reduce(full, p.tp_group)


def shard_lm_tp(p, rank: int, size: int, group, lm_head: nn.Linear | None = None) -> None:
    """Shard the PaliGemma LM read by prefix graph ``p`` over a tensor-parallel group of ``size``
    ranks: query heads and MLP columns split (``q_proj``/``gate_proj``/``up_proj`` output rows,
    ``o_proj``/``down_proj`` input columns, one all-reduce after each), K/V replicated (one KV
    head), the vocabulary of ``lm_head`` split (one all-reduce on the logits). The vision tower,
    embeddings, norms and the action expert stay replicated. Call before building the decode
    graph (its packed weights copy the sharded projections)."""

    def keep(lin: nn.Linear, dim: int) -> None:
        w = lin.weight.detach()
        n = w.shape[dim] // size
        if w.shape[dim] % size:
            raise ValueError(f"{tuple(w.shape)} does not split over TP={size} on dim {dim}")
        lin.weight = nn.Parameter(w.narrow(dim, rank * n, n).contiguous(), requires_grad=False)
        if lin.bias is not None and dim == 0:
            lin.bias = nn.Parameter(lin.bias.detach().narrow(0, rank * n, n).contiguous(), False)

    if p.n_heads % size:
        raise ValueError(f"{p.n_heads} query heads do not split over TP={size}")
    for layer in p.lm.layers:
        at, m = layer.self_attn, layer.mlp
        for lin in (at.q_proj, m.gate_proj, m.up_proj):
            keep(lin, 0)
        for lin in (at.o_proj, m.down_proj):
            keep(lin, 1)
    if lm_head is not None:
        keep(lm_head, 0)
    p.n_heads //= size
    p.tp_group, p.tp_size, p.tp_rank = group, size, rank
    slot = torch.zeros(size, dtype=torch.float32)
    slot[rank] = 1.0
    p.tp_slot = slot.to(p.inv_freq.device)


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


def _gemma_norm(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    """transformers ``GemmaRMSNorm``: ``rms(x) * (1 + w)``, fp32."""
    return _rms(x, norm.eps) * (1.0 + norm.weight.float())


def _ada_norm(x: torch.Tensor, norm: nn.Module, cond: torch.Tensor):
    """``Pi05AdaRMSNorm`` with conditioning: returns ``(normed, gate)`` in fp32."""
    return _ada_norm_mod(x, norm, _lin(cond, norm.dense))


def _ada_norm_mod(x: torch.Tensor, norm: nn.Module, mod: torch.Tensor):
    """``Pi05AdaRMSNorm`` from its precomputed modulation ``mod = dense(cond)`` ``[B, 3W]``."""
    scale, shift, gate = mod.float()[:, None, :].chunk(3, dim=-1)
    return _rms(x, norm.eps) * (1.0 + scale) + shift, gate


def _key_bias(keep: torch.Tensor) -> torch.Tensor:
    """Additive fp32 bias: 0 where ``keep``, MASK_VALUE elsewhere. Built from tensors, not two
    Python scalars: ``torch.where(c, 0.0, x)`` traces its scalars as f64, which neuronx-cc rejects."""
    zero = torch.zeros((), dtype=torch.float32, device=keep.device)
    neg = torch.full((), MASK_VALUE, dtype=torch.float32, device=keep.device)
    return torch.where(keep, zero, neg)


def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """Gemma default RoPE inverse frequencies (fp32), computed on the host."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor):
    """Gemma default RoPE tables ``[B, 1, S, D]`` (fp32) for integer positions ``[B, S]``."""
    freqs = position_ids.float()[:, :, None] * inv_freq.float()[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[:, None], emb.sin()[:, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * cos + _rotate_half(x) * sin


def _attention(q, k, v, bias, n_rep: int, scale: float, mm_dtype: torch.dtype) -> torch.Tensor:
    """softmax(q k^T * scale + bias) v. q ``[B,H,Sq,D]``, k/v ``[B,Hk,Sk,D]``, bias fp32
    broadcastable to ``[B,1,Sq,Sk]``. Matmuls in ``mm_dtype``, softmax in fp32."""
    if n_rep > 1:
        b, hk, s, d = k.shape
        k = k[:, :, None].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)
        v = v[:, :, None].expand(b, hk, n_rep, s, d).reshape(b, hk * n_rep, s, d)
    scores = torch.matmul(q.to(mm_dtype), k.to(mm_dtype).transpose(2, 3)).float() * scale
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(mm_dtype), v.to(mm_dtype)).float()


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # Written out: the Lite runtime's gelu override rejects ``approximate=`` (FX ``apply()`` kwargs).
    # Same formula as ``F.gelu(x, approximate="tanh")``.
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class Pi05PrefixGraph(nn.Module):
    """Images + language tokens -> per-layer prefix K/V (see module docstring).

    Inputs (all static shapes):
      ``pixel_values`` ``[B*ncam, 3, R, R]`` in [-1, 1], camera-major within a sample;
      ``image_valid`` ``[B, ncam]`` float (1 real, 0 padded slot);
      ``lang_tokens`` ``[B, T]`` int64; ``lang_valid`` ``[B, T]`` float.
    Outputs: ``k``, ``v`` ``[L, B, Hk, P, D]`` (mm dtype) and ``prefix_valid`` ``[B, P]`` float.

    Also exposes the pieces a text-generation graph over the same prefix needs: embedding (images
    + language) and the per-layer self-attention/MLP math, factored out so
    :class:`Pi05TextPrefixGraph` (bidirectional prefix -> LM-head logits at every position, no
    action expert) can share them without duplicating the Gemma layer math.
    """

    def __init__(self, model: nn.Module, num_cameras: int):
        super().__init__()
        pwe = model.paligemma_with_expert
        self.vision = pwe.paligemma.model.vision_tower
        self.projector = pwe.paligemma.model.multi_modal_projector
        self.lm = pwe.paligemma.model.language_model
        self.num_cameras = int(num_cameras)
        vc = pwe.paligemma.config.vision_config
        tc = pwe.paligemma.config.text_config
        self.v_heads = vc.num_attention_heads
        self.v_eps = vc.layer_norm_eps
        self.patch = vc.patch_size
        self.head_dim = tc.head_dim
        self.n_heads = tc.num_attention_heads
        self.n_kv = tc.num_key_value_heads
        self.rope_theta = float(
            (getattr(tc, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        )
        self.mm_dtype = self.lm.layers[0].self_attn.q_proj.weight.dtype
        self.register_buffer(
            "inv_freq", rope_inv_freq(self.head_dim, self.rope_theta), persistent=False
        )
        # Tensor parallelism over the LM (shard_lm_tp); n_heads is then this rank's share.
        self.tp_group, self.tp_size, self.tp_rank, self.tp_slot = None, 1, 0, None

    def _vision_model(self):
        return getattr(self.vision, "vision_model", self.vision)

    def embed_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vm = self._vision_model()
        emb = vm.embeddings
        n, c, h, w = pixel_values.shape
        p = self.patch
        # Patch conv as an unfold + matmul (stride == kernel).
        x = pixel_values.float().reshape(n, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(n, (h // p) * (w // p), c * p * p)
        pw = emb.patch_embedding.weight
        x = F.linear(
            x.to(pw.dtype), pw.reshape(pw.shape[0], -1), emb.patch_embedding.bias.to(pw.dtype)
        ).float()
        x = x + emb.position_embedding.weight.float()[None]
        heads = self.v_heads
        for layer in vm.encoder.layers:
            res = x
            y = F.layer_norm(
                x,
                (x.shape[-1],),
                layer.layer_norm1.weight.float(),
                layer.layer_norm1.bias.float(),
                self.v_eps,
            )
            at = layer.self_attn
            b, s, d = y.shape
            hd = d // heads
            q = _lin(y, at.q_proj).view(b, s, heads, hd).transpose(1, 2)
            k = _lin(y, at.k_proj).view(b, s, heads, hd).transpose(1, 2)
            v = _lin(y, at.v_proj).view(b, s, heads, hd).transpose(1, 2)
            o = _attention(q, k, v, None, 1, hd**-0.5, at.q_proj.weight.dtype)
            x = res + _lin(o.transpose(1, 2).reshape(b, s, d), at.out_proj)
            res = x
            y = F.layer_norm(
                x,
                (d,),
                layer.layer_norm2.weight.float(),
                layer.layer_norm2.bias.float(),
                self.v_eps,
            )
            x = res + _lin(_gelu_tanh(_lin(y, layer.mlp.fc1)), layer.mlp.fc2)
        pl = vm.post_layernorm
        x = F.layer_norm(x, (x.shape[-1],), pl.weight.float(), pl.bias.float(), self.v_eps)
        return _lin(x, self.projector.linear)

    def embed_language(self, tokens: torch.Tensor) -> torch.Tensor:
        et = self.lm.embed_tokens
        e = F.embedding(tokens, et.weight)
        scale = getattr(et, "embed_scale", None)
        if scale is None:  # transformers <= 5.3 applies the normalizer inside GemmaModel.forward
            return e.float() * math.sqrt(e.shape[-1])
        return (e * scale.to(e.dtype)).float()

    def forward(self, pixel_values, image_valid, lang_tokens, lang_valid):
        return self.kv_from_embeddings(
            self.embed_images(pixel_values), image_valid, lang_tokens, lang_valid
        )

    def kv_from_embeddings(self, img, image_valid, lang_tokens, lang_valid):
        """The LM half of :meth:`forward`: image token embeddings ``[B*ncam, N, W]`` (fp32, as
        :meth:`embed_images` returns them) + language -> per-layer K/V."""
        b = lang_tokens.shape[0]
        ncam = self.num_cameras
        n_img = img.shape[1]
        img = img.float().reshape(b, ncam * n_img, img.shape[-1])
        x = torch.cat([img, self.embed_language(lang_tokens)], dim=1)  # [B, P, W] fp32
        valid = torch.cat(
            [
                image_valid[:, :, None].expand(b, ncam, n_img).reshape(b, ncam * n_img),
                lang_valid.float(),
            ],
            dim=1,
        )  # [B, P]
        # Bidirectional prefix: attend iff both query and key are real tokens.
        bias = _key_bias((valid[:, None, :, None] * valid[:, None, None, :]) > 0.5)
        pos = torch.cumsum(valid, dim=1) - 1.0
        cos, sin = rope_cos_sin(pos, self.inv_freq)
        hd, nh, nk = self.head_dim, self.n_heads, self.n_kv
        s = x.shape[1]
        ks, vs = [], []
        layers = self.lm.layers
        for i, layer in enumerate(layers):
            at = layer.self_attn
            res = x
            y = _gemma_norm(x, layer.input_layernorm)
            q = _apply_rope(_lin(y, at.q_proj).view(b, s, nh, hd).transpose(1, 2), cos, sin)
            k = _apply_rope(_lin(y, at.k_proj).view(b, s, nk, hd).transpose(1, 2), cos, sin)
            v = _lin(y, at.v_proj).view(b, s, nk, hd).transpose(1, 2)
            ks.append(k.to(self.mm_dtype))
            vs.append(v.to(self.mm_dtype))
            if i == len(layers) - 1:
                break  # the last layer's output is never consumed, only its K/V
            o = _attention(q, k, v, bias, nh // nk, hd**-0.5, self.mm_dtype)
            x = res + _row_parallel(
                o.transpose(1, 2).reshape(b, s, nh * hd), at.o_proj, self.tp_group
            )
            res = x
            y = _gemma_norm(x, layer.post_attention_layernorm)
            m = layer.mlp
            x = res + _row_parallel(
                _gelu_tanh(_lin(y, m.gate_proj)) * _lin(y, m.up_proj), m.down_proj, self.tp_group
            )
        return torch.stack(ks), torch.stack(vs), valid


class Pi05PrefixFromEmbeddingsGraph(nn.Module):
    """:class:`Pi05PrefixGraph` minus the vision tower: ``(img_emb [B*ncam, N, W] fp32,
    image_valid, lang_tokens, lang_valid) -> (k, v, prefix_valid)``. pi0.52 runs SigLIP once per
    request for the subtask decode (:class:`Pi05EmbedImagesGraph`); the action prefix reuses that
    embedding instead of recomputing it (same math, same fp32 embedding)."""

    def __init__(self, prefix: Pi05PrefixGraph):
        super().__init__()
        self.prefix = prefix

    def forward(self, img_emb, image_valid, lang_tokens, lang_valid):
        return self.prefix.kv_from_embeddings(img_emb, image_valid, lang_tokens, lang_valid)


class EulerLoopGraph(nn.Module):
    """``num_steps`` flow-matching Euler steps inside one graph: ``x <- x + dt * step(x, cond_s)``.

    ``step`` is a one-step velocity graph (:class:`Pi05DenoiseGraph`, or pi0's denoise graph)
    whose positional arguments are ``ctx`` with ``x`` and the step's condition inserted at
    ``x_pos`` / ``x_pos + 1``. ``conds`` is ``[num_steps, B, C]`` (fp32). The update is fp32, as
    upstream's ``x_t + dt * v_t``, so the state never leaves the device between steps.
    ``num_steps=1`` with ``dt=-1/n`` is one device-resident step of an ``n``-step schedule.
    """

    def __init__(self, step: nn.Module, num_steps: int, dt: float, x_pos: int = 0):
        super().__init__()
        self.step = step
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.x_pos = int(x_pos)

    def forward(self, x, conds, *ctx):
        x = x.float()
        for s in range(self.num_steps):
            args = list(ctx)
            args.insert(self.x_pos, x)
            args.insert(self.x_pos + 1, conds[s])
            x = x + self.dt * self.step(*args).float()
        return x


class Pi05TextPrefixGraph(nn.Module):
    """Full bidirectional PaliGemma LM forward over [images, language] -> LM-head logits at
    every position (fp32), for pi0.52's greedy subtask-generation re-prefill decode.

    One call recomputes the whole prefix (no KV cache: ``max_new_tokens`` is small and this
    keeps the graph a single fixed shape per sequence-length bucket, which is what Neuron wants).
    Shares its embedding and per-layer math with :class:`Pi05PrefixGraph` via composition, so a
    checkpoint's weights are identical for both. Input/output contract matches
    :class:`Pi05PrefixGraph` apart from running every layer's attention+MLP (not just K/V) and
    projecting the final norm through ``lm_head``.
    """

    def __init__(self, prefix: Pi05PrefixGraph, lm_head: nn.Linear):
        super().__init__()
        self.prefix = prefix
        self.lm_head = lm_head

    def forward(self, pixel_values, image_valid, lang_tokens, lang_valid, lang_causal=None):
        p = self.prefix
        b = lang_tokens.shape[0]
        ncam = p.num_cameras
        img = p.embed_images(pixel_values)
        n_img = img.shape[1]
        img = img.reshape(b, ncam * n_img, img.shape[-1])
        return _text_prefix_core(
            p, self.lm_head, img, image_valid, lang_tokens, lang_valid, ncam, n_img, lang_causal
        )


def _prefix_inputs(p, img, image_valid, lang_tokens, lang_valid, ncam, n_img, lang_causal=None):
    """Embedded [image, language] prefix ``x`` ``[B, P, W]`` (fp32), its validity ``[B, P]``, the
    key bias and the cumsum RoPE tables, shared by every text-prefix graph.

    The bias is bidirectional over the real tokens, except that ``lang_causal`` ``[B, T]`` (1.0 on
    generated tokens) opens a new causal block per marked token, LeRobot's ``make_att_2d_masks``:
    key ``j`` is visible to query ``i`` iff ``cumsum(att)[j] <= cumsum(att)[i]``. The prompt never
    sees generated tokens and generated tokens attend causally (upstream ``select_message``)."""
    b = lang_tokens.shape[0]
    x = torch.cat([img, p.embed_language(lang_tokens)], dim=1)
    valid = torch.cat(
        [
            image_valid[:, :, None].expand(b, ncam, n_img).reshape(b, ncam * n_img),
            lang_valid.float(),
        ],
        dim=1,
    )
    keep = (valid[:, None, :, None] * valid[:, None, None, :]) > 0.5
    if lang_causal is not None:
        att = torch.cat([torch.zeros_like(valid[:, : ncam * n_img]), lang_causal.float()], dim=1)
        c = torch.cumsum(att, dim=1)
        keep = keep & (c[:, None, None, :] <= c[:, None, :, None])
    bias = _key_bias(keep)
    pos = torch.cumsum(valid, dim=1) - 1.0
    cos, sin = rope_cos_sin(pos, p.inv_freq)
    return x, valid, bias, cos, sin


def _lm_layers(p, x, bias, cos, sin, collect_kv: bool = False):
    """The full PaliGemma LM (every layer's attention + MLP) over ``x`` ``[B, S, W]``; returns the
    final-norm hidden state (fp32) and, when ``collect_kv``, every layer's post-RoPE K and V
    ``[B, Hk, S, D]`` (fp32)."""
    b, s = x.shape[0], x.shape[1]
    hd, nh, nk = p.head_dim, p.n_heads, p.n_kv
    ks, vs = [], []
    for layer in p.lm.layers:
        at = layer.self_attn
        res = x
        y = _gemma_norm(x, layer.input_layernorm)
        q = _apply_rope(_lin(y, at.q_proj).view(b, s, nh, hd).transpose(1, 2), cos, sin)
        k = _apply_rope(_lin(y, at.k_proj).view(b, s, nk, hd).transpose(1, 2), cos, sin)
        v = _lin(y, at.v_proj).view(b, s, nk, hd).transpose(1, 2)
        if collect_kv:
            ks.append(k)
            vs.append(v)
        o = _attention(q, k, v, bias, nh // nk, hd**-0.5, p.mm_dtype)
        x = res + _row_parallel(o.transpose(1, 2).reshape(b, s, nh * hd), at.o_proj, p.tp_group)
        res = x
        y = _gemma_norm(x, layer.post_attention_layernorm)
        m = layer.mlp
        x = res + _row_parallel(
            _gelu_tanh(_lin(y, m.gate_proj)) * _lin(y, m.up_proj), m.down_proj, p.tp_group
        )
    return _gemma_norm(x, p.lm.norm), ks, vs


def _text_prefix_core(
    p, lm_head, img, image_valid, lang_tokens, lang_valid, ncam, n_img, lang_causal=None
):
    """Shared body of the subtask text-prefix forward: given the already-embedded image tokens
    ``img`` ``[B, ncam*n_img, W]``, run embed-language + the full bidirectional LM + lm_head.
    Factored out so :class:`Pi05TextPrefixGraphCached` can skip the vision tower (identical image
    embedding across every decode step)."""
    x, _, bias, cos, sin = _prefix_inputs(
        p, img, image_valid, lang_tokens, lang_valid, ncam, n_img, lang_causal
    )
    x, _, _ = _lm_layers(p, x, bias, cos, sin)
    return _lm_logits(p, x, lm_head)


class Pi05EmbedImagesGraph(nn.Module):
    """Just SigLIP + projector over the camera stack -> image token embeddings ``[B*ncam, N, W]``
    (fp32, as the prefix graph consumes them). Run once per request: the subtask decode and the
    action prefix (:class:`Pi05PrefixFromEmbeddingsGraph`) both reuse it."""

    def __init__(self, prefix: Pi05PrefixGraph):
        super().__init__()
        self.prefix = prefix

    def forward(self, pixel_values):
        return self.prefix.embed_images(pixel_values).float()


class Pi05TextPrefixGraphCached(nn.Module):
    """Subtask text-prefix forward over PRECOMPUTED image embeddings: ``(img_emb, image_valid,
    lang_tokens, lang_valid) -> lm_head logits``. Identical math to :class:`Pi05TextPrefixGraph`
    minus the vision tower, which :class:`Pi05EmbedImagesGraph` ran once for the request."""

    def __init__(self, prefix: Pi05PrefixGraph, lm_head: nn.Linear):
        super().__init__()
        self.prefix = prefix
        self.lm_head = lm_head

    def forward(self, img_emb, image_valid, lang_tokens, lang_valid, lang_causal=None):
        p = self.prefix
        b = lang_tokens.shape[0]
        ncam = p.num_cameras
        n_img = img_emb.shape[1]
        img = img_emb.reshape(b, ncam * n_img, img_emb.shape[-1]).float()
        return _text_prefix_core(
            p, self.lm_head, img, image_valid, lang_tokens, lang_valid, ncam, n_img, lang_causal
        )


class Pi05SubtaskPrefillGraph(nn.Module):
    """KV-cached subtask decode, prefill half: ONE bidirectional LM pass over the precomputed image
    embedding + the subtask prompt (same math as :class:`Pi05TextPrefixGraphCached`) returning

    * the LM-head logits ``[1, V]`` (fp32) at the last real prompt token (``last`` ``[1, P]`` is a
      one-hot row, so there is no data-dependent index): the first generated token's logits;
    * every layer's post-RoPE K and V COMPACTED to the real prefix tokens, in the
      :class:`~vllm_omni_neuron.diffusion.attention.decode_attention.StaticKVCache` layout
      ``[1, Hk, max_len, D]``. ``sel`` ``[max_len, P]``: row ``j`` is one-hot on the ``j``-th real
      prefix token, rows past the fill are zero. Compaction lets the shared decode layer's position
      mask (``slot <= pos``) stand in for the prefix padding mask: padded camera slots and padded
      prompt positions never reach the cache. Attention is order-free over keys and every K
      already carries its own RoPE position, so this is exact.

    Returns ``(logits, token, k_0 .. k_{L-1}, v_0 .. v_{L-1})`` as separate base tensors, one per
    cache (a stacked output would hand the decode graph slices of a device tensor as inputs).
    ``token`` ``[1, 1]`` int64 is the greedy pick over ``logits + bias`` (:func:`greedy_token`;
    ``bias`` ``[1, V]`` fp32 carries the FAST/``<loc>``/special-id suppression). Batch 1.
    """

    def __init__(self, prefix: Pi05PrefixGraph, lm_head: nn.Linear, cache_dtype: torch.dtype):
        super().__init__()
        self.prefix = prefix
        self.lm_head = lm_head
        self.cache_dtype = cache_dtype

    def forward(self, img_emb, image_valid, lang_tokens, lang_valid, sel, last, bias):
        p = self.prefix
        ncam = p.num_cameras
        n_img = img_emb.shape[1]
        img = img_emb.reshape(1, ncam * n_img, img_emb.shape[-1]).float()
        x, _, attn_bias, cos, sin = _prefix_inputs(
            p, img, image_valid, lang_tokens, lang_valid, ncam, n_img
        )
        x, ks, vs = _lm_layers(p, x, attn_bias, cos, sin, collect_kv=True)
        logits = _lm_logits(p, torch.matmul(last, x[0]), self.lm_head)  # [1, V]
        hk, d, n = p.n_kv, p.head_dim, sel.shape[0]

        def compact(t):  # [1, Hk, P, D] fp32 -> [1, Hk, max_len, D] cache dtype
            return torch.matmul(sel[None], t[0]).reshape(1, hk, n, d).to(self.cache_dtype)

        token = greedy_token(logits, bias)
        return (logits, token, *[compact(k) for k in ks], *[compact(v) for v in vs])


class Pi05SubtaskDecodeGraph(nn.Module):
    """KV-cached subtask decode, step half: one new token through every PaliGemma LM layer with
    the shared decode-attention layer (``decode_step``: QKV projection, RoPE at the cache position,
    cache append, position-masked attention over the static cache, output projection), the Gemma
    norms and MLP around it, the final norm and ``lm_head``. Position-static (the position is the
    caches' ``pos`` device tensor), so ONE compiled graph serves every generated token.

    ``forward(token [1, 1] int64, bias [1, V] fp32, caches) -> (logits [1, V] fp32, next token
    [1, 1] int64)``; ``caches`` (one ``StaticKVCache`` per layer) advance in place. The RoPE
    tables for the new token come from the caches' fill position inside the graph, and the
    greedy pick over ``logits + bias`` (:func:`greedy_token`) is in the graph too, so a decode
    loop can feed each step's ``token`` output straight into the next step with no host round
    trip (``logits`` is only read by diagnostics / teacher forcing).
    """

    def __init__(self, prefix: Pi05PrefixGraph, lm_head: nn.Linear, cfg):
        super().__init__()
        from vllm_omni_neuron.diffusion.attention.decode_attention import DecodeWeights

        self.prefix = prefix
        self.lm_head = lm_head
        self.cfg = cfg
        # Packed [in, out] QKV / output projections (a copy: the prefix graphs keep reading the
        # nn.Linear weights). Built from the checkpoint dtype, cast to the cache dtype.
        self.weights = [
            DecodeWeights.from_separate(
                layer.self_attn.q_proj.weight.detach(),
                layer.self_attn.k_proj.weight.detach(),
                layer.self_attn.v_proj.weight.detach(),
                layer.self_attn.o_proj.weight.detach(),
            ).to(cfg.dtype)
            for layer in prefix.lm.layers
        ]

    def move_weights(self, device) -> None:
        self.weights = [w.to(device) for w in self.weights]

    def forward(self, token, bias, caches):
        from vllm_omni_neuron.diffusion.attention.decode_attention import decode_step

        p = self.prefix
        # RoPE at the new token's position (= the real tokens before it = the fill position).
        freqs = caches[0].pos.float()[:, None] * p.inv_freq.float()[None]  # [1, D//2]
        cos, sin = freqs.cos().to(self.cfg.dtype), freqs.sin().to(self.cfg.dtype)
        x = p.embed_language(token)  # [1, 1, W] fp32
        for layer, w, cache in zip(p.lm.layers, self.weights, caches):
            res = x
            y = _gemma_norm(x, layer.input_layernorm).to(self.cfg.dtype)
            o = decode_step(self.cfg, w, cache, y, cos=cos, sin=sin)  # [1, W]
            x = res + _all_reduce(o.float(), p.tp_group).reshape(res.shape)
            res = x
            y = _gemma_norm(x, layer.post_attention_layernorm)
            m = layer.mlp
            x = res + _row_parallel(
                _gelu_tanh(_lin(y, m.gate_proj)) * _lin(y, m.up_proj), m.down_proj, p.tp_group
            )
        x = _gemma_norm(x, p.lm.norm)
        logits = _lm_logits(p, x[0], self.lm_head)  # [1, V]
        return logits, greedy_token(logits, bias)


def greedy_token(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``argmax(logits + bias)`` as a ``[1, 1]`` int64 token (in-graph greedy decode). ``bias`` is
    0 for allowed ids and :data:`MASK_VALUE` for suppressed ones, the device form of the host
    path's ``-inf`` fills."""
    return torch.argmax(logits.float() + bias, dim=-1, keepdim=True).to(torch.long)


class Pi05DenoiseGraph(nn.Module):
    """One flow-matching step: ``(x_t, time_cond, prefix K/V) -> v_t`` (fp32).

    Inputs: ``x_t`` ``[B, H, A]`` fp32; ``time_cond`` ``[B, W_e]`` fp32 (the AdaRMS condition,
    ``Pi05ForActionPrediction.embed_timestep`` on the host); ``k``/``v`` from the prefix graph;
    ``prefix_valid`` ``[B, P]`` float.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        pwe = model.paligemma_with_expert
        self.expert = pwe.gemma_expert.model
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        ec = pwe.gemma_expert.config
        self.head_dim = ec.head_dim
        self.n_heads = ec.num_attention_heads
        self.n_kv = ec.num_key_value_heads
        self.rope_theta = float(
            (getattr(ec, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        )
        self.mm_dtype = self.expert.layers[0].self_attn.q_proj.weight.dtype
        self.register_buffer(
            "inv_freq", rope_inv_freq(self.head_dim, self.rope_theta), persistent=False
        )
        # True: forward takes precomputed AdaRMS modulations (modulation_table), not the condition.
        self.modulation_tables = False
        self._mod_weights = None

    def forward(self, x_t, time_cond, k_cache, v_cache, prefix_valid):
        """``time_cond``: the AdaRMS condition ``[B, W_e]``, or (``modulation_tables``) the
        precomputed per-norm modulations ``[B, 2L+1, 3W]`` from :meth:`modulation_table`."""
        b, h, _ = x_t.shape
        x = _lin(x_t, self.action_in_proj)  # [B, H, W]
        tables = self.modulation_tables

        def norm(x, idx, mod_norm):
            if tables:
                return _ada_norm_mod(x, mod_norm, time_cond[:, idx])
            return _ada_norm(x, mod_norm, time_cond)

        # Suffix attention: every action token sees all real prefix tokens and every action token.
        key_valid = torch.cat([prefix_valid, prefix_valid.new_ones(b, h)], dim=1)
        bias = _key_bias(key_valid[:, None, None, :] > 0.5)
        pos = (
            prefix_valid.sum(dim=1, keepdim=True)
            + torch.arange(h, device=x_t.device, dtype=torch.float32)[None]
        )
        cos, sin = rope_cos_sin(pos, self.inv_freq)
        hd, nh, nk = self.head_dim, self.n_heads, self.n_kv
        for i, layer in enumerate(self.expert.layers):
            at = layer.self_attn
            res = x
            y, gate = norm(x, 2 * i, layer.input_layernorm)
            q = _apply_rope(_lin(y, at.q_proj).view(b, h, nh, hd).transpose(1, 2), cos, sin)
            k = _apply_rope(_lin(y, at.k_proj).view(b, h, nk, hd).transpose(1, 2), cos, sin)
            v = _lin(y, at.v_proj).view(b, h, nk, hd).transpose(1, 2)
            k = torch.cat([k_cache[i].float(), k], dim=2)
            v = torch.cat([v_cache[i].float(), v], dim=2)
            o = _attention(q, k, v, bias, nh // nk, hd**-0.5, self.mm_dtype)
            x = res + gate * _lin(o.transpose(1, 2).reshape(b, h, nh * hd), at.o_proj)
            res = x
            y, gate = norm(x, 2 * i + 1, layer.post_attention_layernorm)
            m = layer.mlp
            x = res + gate * _lin(
                _gelu_tanh(_lin(y, m.gate_proj)) * _lin(y, m.up_proj), m.down_proj
            )
        y, _ = norm(x, 2 * len(self.expert.layers), self.expert.norm)
        return _lin(y, self.action_out_proj)

    def adarms_norms(self) -> list[nn.Module]:
        """Every AdaRMS norm in forward order: per layer (input, post-attention), then final."""
        out = []
        for layer in self.expert.layers:
            out += [layer.input_layernorm, layer.post_attention_layernorm]
        return out + [self.expert.norm]

    def snapshot_modulation_weights(self) -> None:
        """Host fp32 copies of the AdaRMS ``dense`` projections, for :meth:`modulation_table`
        (taken once after loading, while the weights are still on the host)."""
        self._mod_weights = [
            (
                n.dense.weight.detach().float().cpu().clone(),
                n.dense.bias.detach().float().cpu().clone(),
            )
            for n in self.adarms_norms()
        ]

    @torch.no_grad()
    def modulation_table(self, conds: torch.Tensor) -> torch.Tensor:
        """``conds`` ``[n, B, W_e]`` (fp32 host AdaRMS conditions of an Euler schedule) -> every
        norm's ``(scale, shift, gate)`` modulation ``[n, B, 2L+1, 3W]`` (fp32 host). The same
        ``dense(cond)`` the graph would compute per step, done once per schedule instead (as the
        GR00T port bakes its AdaLN tables), so a step reads no modulation weights."""
        if getattr(self, "_mod_weights", None) is None:
            self.snapshot_modulation_weights()
        return torch.stack([F.linear(conds.float(), w, b) for w, b in self._mod_weights], dim=2)
