# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5's Gemma text encoder (the 48-layer text tower of ``Gemma4UnifiedForConditionalGeneration``)
on NeuronCores, tensor-parallel over the stage's TP group.

The LTX-2.5 pipeline uses every hidden state of the text tower (the embeddings, the outputs of
layers 0-46 and the final-normed output of layer 47, stacked to ``3840 x 49`` features per token), so
this module returns exactly diffusers' ``_get_gemma_prompt_embeds`` tensor. The math is
transformers' ``Gemma4UnifiedTextModel`` for a text-only prompt:

* 48 decoder layers in a fixed period of six (five sliding-window layers, one global layer).
  Sliding layers: 16 query heads / 8 KV heads of 256; global layers: 16 query heads of 512 and one
  KV head whose value is the un-normed key (``attention_k_eq_v``). Scale 1 (QK-norm), RMSNorm
  ``x * w`` in fp32, a per-layer scalar, gelu-tanh MLP 15360.
* **TP:** query heads, KV heads, MLP columns are sharded (Q/K/V/gate/up column-parallel,
  o/down row-parallel with an all-reduce); the single global KV head is replicated. Legal TP:
  1, 2, 4, 8 (TP must divide the 8 sliding KV heads).
* **Graphs:** one compiled graph runs one six-layer period with the layer weights as inputs, so
  eight calls of one NEFF (per prompt bucket) serve the 48 layers. The embedding lookup,
  the RoPE tables (transformers' own rotary module, fp32) and the final norm run on the host.
* **Prompt buckets:** the prompt is left-padded to 1024 tokens; only the last ``B`` positions are run,
  ``B`` the smallest bucket that holds the prompt. Padding is masked out of every attention, the
  positions stay the ones of the 1024-token sequence, and the hidden states at padded positions
  (which the text connectors drop) are returned as zeros. The sliding window (1024) covers every
  bucket, so sliding and global layers share one causal mask.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import (
    _all_reduce,
    _SafetensorsIndex,
    tp_state,
)

BUCKETS = tuple(
    int(x) for x in os.environ.get("LTX25_TEXT_BUCKETS", "128,256,512,1024").split(",") if x
)
PREFIX = "model.language_model."


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``Gemma4UnifiedRMSNorm``: fp32 ``x * (mean(x^2) + eps)^-0.5 [* w]``, cast back."""
    xf = x.float()
    out = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        out = out * w.float()
    return out.to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x [B, S, H, D]``, ``cos/sin [B, S, D]`` fp32 (rotate-half RoPE)."""
    xf = x.float()
    h = xf.shape[-1] // 2
    rot = torch.cat([-xf[..., h:], xf[..., :h]], dim=-1)
    return (xf * cos[:, :, None] + rot * sin[:, :, None]).to(x.dtype)


def layer_forward(cfg, w: dict, x, cos, sin, mask, valid, glob: bool, tp: int, group):
    """One decoder layer on this rank's heads. ``mask [1, 1, S, S]`` additive fp32,
    ``valid [1, S, 1]`` (0 at padded positions: their hidden states are kept at zero)."""
    eps = cfg["eps"]
    b, s, _ = x.shape
    hd = cfg["global_head_dim"] if glob else cfg["head_dim"]
    hq = cfg["heads"] // tp
    hkv = 1 if glob else cfg["kv_heads"] // tp
    n = rms_norm(x, w["ln_in"], eps)
    q = rms_norm(F.linear(n, w["q_w"]).view(b, s, hq, hd), w["q_n"], eps)
    kr = F.linear(n, w["k_w"]).view(b, s, hkv, hd)
    vr = kr if glob else F.linear(n, w["v_w"]).view(b, s, hkv, hd)
    k = rms_norm(kr, w["k_n"], eps)
    v = rms_norm(vr, None, eps)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    rep = hq // hkv
    qh = q.transpose(1, 2).float()  # [B, Hq, S, D]
    kh = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vh = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    p = torch.softmax(torch.matmul(qh, kh.transpose(-2, -1)) + mask, dim=-1)  # scale 1 (QK-norm)
    o = torch.matmul(p, vh).to(x.dtype).transpose(1, 2).reshape(b, s, hq * hd)
    a = _all_reduce(F.linear(o, w["o_w"]), tp, group)
    x = x + rms_norm(a, w["ln_post_attn"], eps)
    n = rms_norm(x, w["ln_pre_ff"], eps)
    h = F.gelu(F.linear(n, w["gate_w"]), approximate="tanh") * F.linear(n, w["up_w"])
    m = _all_reduce(F.linear(h, w["down_w"]), tp, group)
    x = x + rms_norm(m, w["ln_post_ff"], eps)
    return (x * w["scalar"].to(x.dtype)) * valid.to(x.dtype)


class NeuronGemmaTextEncoder:
    """The LTX-2.5 Gemma text tower, TP-sharded on the NeuronCores of this rank's TP group."""

    def __init__(self, text_encoder_dir: str, device, dtype=torch.bfloat16, buckets=BUCKETS):
        from transformers import AutoConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextRotaryEmbedding,
        )

        tc = AutoConfig.from_pretrained(text_encoder_dir).text_config
        self.tp, self.rank, self.group = tp_state()
        self.device, self.dtype = torch.device(device), dtype
        self.layer_types = list(tc.layer_types)
        g0 = self.layer_types.index("full_attention")
        sl, gl = tc.per_layer_config[0], tc.per_layer_config[g0]  # sliding / global geometry
        self.cfg = dict(
            eps=tc.rms_norm_eps,
            heads=tc.num_attention_heads,
            kv_heads=sl.num_key_value_heads,
            head_dim=sl.head_dim,
            global_head_dim=gl.head_dim,
            hidden=tc.hidden_size,
        )
        if self.cfg["kv_heads"] % self.tp or self.cfg["heads"] % self.tp:
            raise ValueError(f"TP={self.tp} must divide the text encoder's KV heads (8)")
        if gl.num_key_value_heads != 1 or not tc.attention_k_eq_v:
            raise NotImplementedError("expected one shared K=V head in the global layers")
        if getattr(tc, "num_kv_shared_layers", 0) or getattr(tc, "hidden_size_per_layer_input", 0):
            raise NotImplementedError("KV-shared / per-layer-input Gemma variants")
        period = self.layer_types.index("full_attention") + 1
        if self.layer_types != self.layer_types[:period] * (len(self.layer_types) // period):
            raise NotImplementedError("layer types are not a repeated period")
        self.period = period
        self.max_len = 1024
        self.buckets = tuple(sorted(b for b in buckets if b <= self.max_len)) or (self.max_len,)
        self.rotary = Gemma4UnifiedTextRotaryEmbedding(tc)
        self._fn = None
        self._tables: dict = {}
        self._load(text_encoder_dir)

    # -- weights ----------------------------------------------------------------------------
    def _layer_specs(self, glob: bool):
        """(local key, checkpoint suffix, shard dim)"""
        attn = [
            ("q_w", "self_attn.q_proj.weight", 0),
            ("q_n", "self_attn.q_norm.weight", None),
            ("k_w", "self_attn.k_proj.weight", None if glob else 0),
            ("k_n", "self_attn.k_norm.weight", None),
        ]
        if not glob:
            attn.append(("v_w", "self_attn.v_proj.weight", 0))
        attn.append(("o_w", "self_attn.o_proj.weight", 1))
        return attn + [
            ("ln_in", "input_layernorm.weight", None),
            ("ln_post_attn", "post_attention_layernorm.weight", None),
            ("ln_pre_ff", "pre_feedforward_layernorm.weight", None),
            ("ln_post_ff", "post_feedforward_layernorm.weight", None),
            ("gate_w", "mlp.gate_proj.weight", 0),
            ("up_w", "mlp.up_proj.weight", 0),
            ("down_w", "mlp.down_proj.weight", 1),
            ("scalar", "layer_scalar", None),
        ]

    def _load(self, path: str) -> None:
        ckpt = _SafetensorsIndex(path, prefix="model")
        self.embed = ckpt.get(PREFIX + "embed_tokens.weight").to(self.dtype)  # host (2 GB)
        self.final_norm = ckpt.get(PREFIX + "norm.weight").float()
        self.layers = []
        for i, lt in enumerate(self.layer_types):
            glob = lt == "full_attention"
            w = {}
            for key, suffix, dim in self._layer_specs(glob):
                t = ckpt.get(f"{PREFIX}layers.{i}.{suffix}", dim, self.rank, self.tp)
                w[key] = t.to(self.dtype).contiguous().to(self.device)
            self.layers.append(w)
        ckpt.close()
        self.keys = {g: [k for k, _, _ in self._layer_specs(g)] for g in (False, True)}

    def num_local_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for w in self.layers for t in w.values())

    # -- graph ------------------------------------------------------------------------------
    def period_fn(self, x, cos_s, sin_s, cos_g, sin_g, mask, valid, *weights):
        """One period of layers (weights flattened in layer order). Returns each layer's output."""
        outs, it = [], iter(weights)
        for lt in self.layer_types[: self.period]:
            glob = lt == "full_attention"
            w = {k: next(it) for k in self.keys[glob]}
            c, s = (cos_g, sin_g) if glob else (cos_s, sin_s)
            x = layer_forward(self.cfg, w, x, c, s, mask, valid, glob, self.tp, self.group)
            outs.append(x)
        return tuple(outs)

    def compile(self, backend: str) -> None:
        self._fn = torch.compile(
            self.period_fn,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options={
                "model_name": "ltx2_gemma_period",
                "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
            },
        )

    def _bucket_tables(self, bucket: int):
        """Device RoPE tables (sliding, global) for the last ``bucket`` positions of 1024."""
        key = (self.max_len, bucket)
        if key not in self._tables:
            pos = torch.arange(self.max_len - bucket, self.max_len).unsqueeze(0)
            x = torch.zeros(1, 1, dtype=torch.float32)
            t = []
            for lt in ("sliding_attention", "full_attention"):
                cos, sin = self.rotary(x, pos, lt)
                t += [c.float().contiguous().to(self.device) for c in (cos, sin)]
            self._tables[key] = t
        return self._tables[key]

    # -- encode ----------------------------------------------------------------------------
    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``input_ids / attention_mask [1, 1024]`` (left-padded) -> ``[1, 1024, hidden * 49]``,
        laid out as diffusers' ``torch.stack(hidden_states, -1).flatten(2, 3)``."""
        if input_ids.shape[0] != 1:
            raise NotImplementedError("one prompt per call")
        total = input_ids.shape[1]
        n_valid = int(attention_mask.sum())
        bucket = next((b for b in self.buckets if b >= n_valid), total)
        bucket = min(bucket, total)
        ids, am = input_ids[:, -bucket:], attention_mask[:, -bucket:].bool()
        h0 = (self.embed[ids] * torch.tensor(self.cfg["hidden"] ** 0.5).to(self.dtype)).to(
            self.dtype
        )
        causal = torch.tril(torch.ones(bucket, bucket, dtype=torch.bool))
        allowed = causal & am[0][None, :]
        mask = torch.where(allowed, 0.0, -1e9).to(torch.float32)[None, None]
        valid = am[..., None].float()
        dev = self.device
        tables = self._bucket_tables(bucket)
        fn = self._fn or self.period_fn
        x = h0.to(dev)
        mask_d, valid_d = mask.to(dev), valid.to(dev)
        outs = []
        for p in range(0, len(self.layers), self.period):
            flat = []
            for li in range(p, p + self.period):
                glob = self.layer_types[li] == "full_attention"
                flat += [self.layers[li][k] for k in self.keys[glob]]
            res = fn(x, *tables, mask_d, valid_d, *flat)
            outs.extend(res)
            x = res[-1]
        states = [h0 * valid.to(self.dtype)] + [o.cpu() for o in outs]
        states[-1] = rms_norm(states[-1], self.final_norm, self.cfg["eps"]) * valid.to(self.dtype)
        stacked = torch.stack(states, dim=-1).flatten(2, 3)  # [1, bucket, hidden * 49]
        if bucket < total:
            stacked = F.pad(stacked, (0, 0, total - bucket, 0))
        return stacked.to(self.dtype)

    def encode_prompt(self, tokenizer, prompt: str, max_sequence_length: int = 1024):
        """diffusers ``LTX2Pipeline._get_gemma_prompt_embeds`` for one prompt:
        ``(prompt_embeds [1, L, hidden * 49], attention_mask [1, L])``."""
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        t = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.max_len = max_sequence_length
        return self.hidden_states(t.input_ids, t.attention_mask), t.attention_mask
