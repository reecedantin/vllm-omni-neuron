# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape compute graphs for the pi0 (base) action path on Neuron.

Same architecture as pi0.5's graphs (``.graphs``) with π0's differences: a continuous ``state``
projected through ``state_proj`` into the suffix as its own token (pi0.5 discretizes state into
the language prompt instead), ``action_time_mlp_{in,out}`` concatenating the time embedding onto
the action embedding (2W -> W -> W) rather than feeding an AdaRMS condition, and a PLAIN Gemma
action expert (no AdaRMS norms — reuses the same ``_gemma_norm`` helper as the prefix).

Suffix layout: ``[state_token, action_tokens x H]``, causal mask ``[1, 1, 0, ..., 0]`` (both the
state token and the first action token open a new causal block; the rest of the action tokens
attend to each other bidirectionally).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import (
    Pi05PrefixGraph,
    _apply_rope,
    _attention,
    _gelu_tanh,
    _gemma_norm,
    _key_bias,
    _lin,
    rope_cos_sin,
    rope_inv_freq,
)


class Pi0PrefixGraph(Pi05PrefixGraph):
    """Identical to :class:`.graphs.Pi05PrefixGraph`: π0's prefix layout
    (``[img_cam_0, ..., lang_tokens]``, bidirectional) is the same as π0.5's."""


class Pi0DenoiseGraph(nn.Module):
    """One flow-matching step for π0: ``(state, x_t, time_sincos, prefix K/V) -> v_t`` (fp32).

    Unlike pi0.5's :class:`.graphs.Pi05DenoiseGraph`, the time conditioning is NOT an AdaRMS
    vector — it is concatenated onto the action embedding before ``action_time_mlp_{in,out}`` —
    and the suffix carries an extra ``state`` token (``state_proj``). The action expert's norms
    are plain ``GemmaRMSNorm`` (``_gemma_norm``), not conditioned.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        pwe = model.paligemma_with_expert
        self.expert = pwe.gemma_expert.model
        self.state_proj = model.state_proj
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.action_time_mlp_in = model.action_time_mlp_in
        self.action_time_mlp_out = model.action_time_mlp_out
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

    def forward(self, state, x_t, time_sincos, k_cache, v_cache, prefix_valid):
        b, h, _ = x_t.shape
        state_emb = _lin(state, self.state_proj)[:, None, :]  # [B, 1, W]

        action_emb = _lin(x_t, self.action_in_proj)  # [B, H, W]
        time_emb = time_sincos[:, None, :].expand(-1, h, -1).float()
        at = torch.cat([action_emb, time_emb], dim=2)
        at = F.silu(_lin(at, self.action_time_mlp_in))
        action_time_emb = _lin(at, self.action_time_mlp_out)

        x = torch.cat([state_emb, action_time_emb], dim=1)  # [B, 1+H, W]
        suffix_len = x.shape[1]

        key_valid = torch.cat(
            [prefix_valid, prefix_valid.new_ones(b, suffix_len)], dim=1
        )  # [B, P+S]
        # Suffix AR mask [1, 1, 0, ..., 0] -> cumsum [1, 2, 2, ...]: the state token (query 0) sees
        # the prefix and itself only; the action tokens see everything. Built from tensors (no
        # Python-scalar where -> no f64 in the graph).
        p_len = prefix_valid.shape[1]
        k_idx = torch.arange(p_len + suffix_len, device=x_t.device)
        q_idx = torch.arange(suffix_len, device=x_t.device)
        block_ok = (k_idx[None, :] <= p_len) | (q_idx[:, None] >= 1)  # [S, P+S]
        keep = (key_valid[:, None, None, :] > 0.5) & block_ok[None, None, :, :]
        bias = _key_bias(keep)  # [B, 1, S, P+S]
        pos = (
            prefix_valid.sum(dim=1, keepdim=True)
            + torch.arange(suffix_len, device=x_t.device, dtype=torch.float32)[None]
        )
        cos, sin = rope_cos_sin(pos, self.inv_freq)
        hd, nh, nk = self.head_dim, self.n_heads, self.n_kv
        for i, layer in enumerate(self.expert.layers):
            at_ = layer.self_attn
            res = x
            y = _gemma_norm(x, layer.input_layernorm)
            q = _apply_rope(
                _lin(y, at_.q_proj).view(b, suffix_len, nh, hd).transpose(1, 2), cos, sin
            )
            k = _apply_rope(
                _lin(y, at_.k_proj).view(b, suffix_len, nk, hd).transpose(1, 2), cos, sin
            )
            v = _lin(y, at_.v_proj).view(b, suffix_len, nk, hd).transpose(1, 2)
            k = torch.cat([k_cache[i].float(), k], dim=2)
            v = torch.cat([v_cache[i].float(), v], dim=2)
            o = _attention(q, k, v, bias, nh // nk, hd**-0.5, self.mm_dtype)
            x = res + _lin(o.transpose(1, 2).reshape(b, suffix_len, nh * hd), at_.o_proj)
            res = x
            y = _gemma_norm(x, layer.post_attention_layernorm)
            m = layer.mlp
            x = res + _lin(_gelu_tanh(_lin(y, m.gate_proj)) * _lin(y, m.up_proj), m.down_proj)
        y = _gemma_norm(x, self.expert.norm)
        suffix_out = y[:, -h:]  # drop the state token, keep the action tokens
        return _lin(suffix_out, self.action_out_proj)
