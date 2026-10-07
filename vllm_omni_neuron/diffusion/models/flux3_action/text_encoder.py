# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-4B text encoder of FLUX Action (NeuronCore, with a host fallback).

Upstream (``flux_action/models/text_encoder.py``) builds the DiT text context by stacking the
hidden states of layers 4, 8, ..., 32 of the Qwen3-VL language model along channels (8 x 2560 =
20480 for the released encoder). Prompts go through the chat template and are right-padded to
the next multiple of 80 tokens; the padding positions are part of the context the DiT attends to.

Tokenisation, chat template, padding, the embedding lookup, the attention mask and the mRoPE
cos/sin tables always come from ``transformers`` itself on the host (captured at the input of the
first decoder layer), so they stay bit-identical to upstream. The 32 decoder layers then run either
on the host (``transformers``, the fallback) or, after :meth:`Qwen3VLTextEncoder.to_device`, on the
NeuronCore: one compiled graph per padded length, replayed for every layer with that layer's
weights, optionally head-sharded over a tensor-parallel group (GQA: TP must divide the 8 KV heads).
The pipeline caches the result per caption either way.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

TEXT_PAD_MULTIPLE = 80
TEXT_PAD_MAX_LENGTH = 8192
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def padded_text_token_count(
    real_len: int, multiple: int = TEXT_PAD_MULTIPLE, cap: int = TEXT_PAD_MAX_LENGTH
) -> int:
    return min(math.ceil(real_len / multiple) * multiple, cap)


class _LayerInputs(Exception):
    """Raised by the first decoder layer's pre-hook to stop the host forward at the layer stack."""


def _hf_rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """transformers' Qwen3 RMSNorm: fp32 statistics, cast back, then scale in the input dtype."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def decoder_layer(
    h: Tensor,
    cos: Tensor,
    sin: Tensor,
    bias: Tensor,
    ln1: Tensor,
    w_qkv: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    w_o: Tensor,
    ln2: Tensor,
    w_gu: Tensor,
    w_down: Tensor,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    ffn: int,
    eps: float,
    tp_group=None,
) -> Tensor:
    """One Qwen3 decoder layer (transformers' math) on this rank's shard: ``heads`` / ``kv_heads``
    / ``ffn`` are local, and the attention and MLP outputs are all-reduced over ``tp_group``.
    ``bias`` is the additive fp32 ``[1, 1, L, L]`` form of transformers' causal + padding mask."""
    b, length, _ = h.shape
    x = _hf_rms(h, ln1, eps)
    q, k, v = F.linear(x, w_qkv).split(
        (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1
    )
    q = _hf_rms(q.reshape(b, length, heads, head_dim), q_norm, eps).transpose(1, 2)
    k = _hf_rms(k.reshape(b, length, kv_heads, head_dim), k_norm, eps).transpose(1, 2)
    v = v.reshape(b, length, kv_heads, head_dim).transpose(1, 2)
    c, s = cos[:, None], sin[:, None]
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s
    rep = heads // kv_heads
    if rep > 1:  # GQA: query head i uses KV head i // rep (transformers' repeat_kv)
        shape = (b, kv_heads, rep, length, head_dim)
        k = k[:, :, None].expand(shape).reshape(b, heads, length, head_dim)
        v = v[:, :, None].expand(shape).reshape(b, heads, length, head_dim)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * head_dim**-0.5 + bias
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    attn = torch.matmul(probs, v).transpose(1, 2).reshape(b, length, heads * head_dim)
    o = F.linear(attn, w_o)
    if tp_group is not None:
        dist.all_reduce(o, group=tp_group)
    h = h + o
    gate, up = F.linear(_hf_rms(h, ln2, eps), w_gu).split((ffn, ffn), dim=-1)
    d = F.linear(F.silu(gate) * up, w_down)
    if tp_group is not None:
        dist.all_reduce(d, group=tp_group)
    return h + d


class Qwen3VLTextEncoder:
    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16, threads: int | None = None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.dtype = dtype
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype)
        # Only the language model is used, up to the deepest selected layer. Keep ONE layer more:
        # the last entry of ``hidden_states`` is the final-normed output, so truncating to exactly
        # 32 layers would turn hidden_states[32] into a normed tensor.
        lm = model.model.language_model
        keep = max(OUTPUT_LAYERS) + 1
        if len(lm.layers) > keep:
            lm.layers = lm.layers[:keep]
            lm.config.num_hidden_layers = keep
        self.lm = lm.eval().requires_grad_(False)
        self.model = model
        self.processor = AutoProcessor.from_pretrained(path)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the text encoder requires a right-padding tokenizer")
        self.threads = threads
        self.device: torch.device | None = None  # set by to_device: the layer stack runs there
        self._layer_w: list[tuple[Tensor, ...]] = []
        self._layer_fn = None

    # -- NeuronCore layer stack ---------------------------------------------------------------
    def to_device(
        self, device, *, tp_size: int = 1, tp_rank: int = 0, tp_group=None, keep_host_layers=False
    ) -> None:
        """Move this rank's shard of decoder layers 1..32 to ``device``, head-sharded over
        ``tp_group``. The host keeps the embedding, the rotary tables and layer 0's module (its
        pre-hook captures the layer inputs); the other host layers are dropped unless
        ``keep_host_layers``."""
        c = self.lm.config
        heads, kv, ffn = c.num_attention_heads, c.num_key_value_heads, c.intermediate_size
        hd = getattr(c, "head_dim", None) or c.hidden_size // heads
        if kv % tp_size or heads % tp_size or ffn % tp_size:
            raise ValueError(f"text encoder: tp_size={tp_size} must divide the {kv} KV heads")
        hl, kl, fl, r = heads // tp_size, kv // tp_size, ffn // tp_size, tp_rank
        dev = torch.device(device)

        def rows(w: Tensor, n: int) -> Tensor:  # this rank's n-row block
            return w[r * n : (r + 1) * n]

        self._layer_w = []
        for layer in self.lm.layers[: max(OUTPUT_LAYERS)]:
            a, m = layer.self_attn, layer.mlp
            ws = (
                layer.input_layernorm.weight,
                torch.cat(
                    (
                        rows(a.q_proj.weight, hl * hd),
                        rows(a.k_proj.weight, kl * hd),
                        rows(a.v_proj.weight, kl * hd),
                    )
                ),
                a.q_norm.weight,
                a.k_norm.weight,
                a.o_proj.weight[:, r * hl * hd : (r + 1) * hl * hd],
                layer.post_attention_layernorm.weight,
                torch.cat((rows(m.gate_proj.weight, fl), rows(m.up_proj.weight, fl))),
                m.down_proj.weight[:, r * fl : (r + 1) * fl],
            )
            self._layer_w.append(tuple(t.detach().to(self.dtype).contiguous().to(dev) for t in ws))
        self._shape = dict(
            heads=hl,
            kv_heads=kl,
            head_dim=hd,
            ffn=fl,
            eps=c.rms_norm_eps,
            tp_group=tp_group if tp_size > 1 else None,
        )
        self.device = dev
        self._layer_fn = self._layer
        if not keep_host_layers:
            self.lm.layers = self.lm.layers[:1]  # layer 0's module only hosts the input hook

    def _layer(self, h, cos, sin, bias, *w):
        return decoder_layer(h, cos, sin, bias, *w, **self._shape)

    def compile(self, backend: str, options: dict | None = None) -> None:
        """One graph per padded caption length, replayed for all 32 layers."""
        if self.device is None:
            return
        self._layer_fn = torch.compile(
            self._layer,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options={
                **(options or {}),
                "model_name": "flux3_action_text_layer",
                "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
            },
        )

    def device_param_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for ws in self._layer_w for t in ws)

    def _layer_inputs(self, toks) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run transformers up to the first decoder layer and return what it would receive: the
        embeddings, the mRoPE cos/sin tables and the additive form of its causal + padding mask."""
        captured = {}

        def hook(_mod, args, kwargs):
            captured["h"] = args[0] if args else kwargs["hidden_states"]
            captured["mask"] = kwargs.get("attention_mask")
            captured["pe"] = kwargs["position_embeddings"]
            raise _LayerInputs

        handle = self.lm.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            self.model.model(
                input_ids=toks["input_ids"], attention_mask=toks["attention_mask"], use_cache=False
            )
        except _LayerInputs:
            pass
        finally:
            handle.remove()
        h, (cos, sin), mask = captured["h"], captured["pe"], captured["mask"]
        length = h.shape[1]
        if mask is None:  # no padding: plain causal
            mask = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
        elif mask.dtype != torch.bool:
            mask = mask > -1  # additive (eager) form -> allowed
        if not bool(mask.any(-1).all()):
            raise ValueError("text encoder mask has a fully masked row")
        bias = torch.where(mask, 0.0, -30000.0).to(torch.float32)
        return h, cos, sin, bias

    def _device_forward(self, toks) -> Tensor:
        h, cos, sin, bias = self._layer_inputs(toks)
        dev = self.device
        h, cos, sin, bias = (t.to(dev).contiguous() for t in (h, cos, sin, bias))
        picked = []
        for i, ws in enumerate(self._layer_w, start=1):
            h = self._layer_fn(h, cos, sin, bias, *ws)
            if i in OUTPUT_LAYERS:
                picked.append(h.to("cpu"))
        return torch.cat(picked, dim=-1)

    @property
    def context_dim(self) -> int:
        return len(OUTPUT_LAYERS) * self.lm.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str], *, fixed_length: int | None = None) -> list[Tensor]:
        """Captions -> ``[(1, L_i, 8 * hidden)]`` in input order (one forward per padded length bucket)."""
        tok = self.processor.tokenizer
        formatted = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
            )
            for t in texts
        ]
        buckets: dict[int, list[int]] = {}
        for i, prompt in enumerate(formatted):
            if fixed_length is not None:
                buckets.setdefault(fixed_length, []).append(i)
                continue
            n = tok(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=TEXT_PAD_MAX_LENGTH,
            )["input_ids"].shape[1]
            buckets.setdefault(padded_text_token_count(n), []).append(i)
        out: list[Tensor | None] = [None] * len(texts)
        prev_threads = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            for length, members in sorted(buckets.items()):
                toks = tok(
                    [formatted[i] for i in members],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=length,
                    padding_side="right",
                )
                if self.device is not None:
                    stacked = self._device_forward(toks).to(self.dtype)
                else:
                    res = self.model.model(
                        input_ids=toks["input_ids"],
                        attention_mask=toks["attention_mask"],
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    stacked = torch.cat([res.hidden_states[k] for k in OUTPUT_LAYERS], dim=-1)
                    stacked = stacked.to(self.dtype)
                for j, i in enumerate(members):
                    out[i] = stacked[j : j + 1].clone()
        finally:
            torch.set_num_threads(prev_threads)
        return out  # type: ignore[return-value]
