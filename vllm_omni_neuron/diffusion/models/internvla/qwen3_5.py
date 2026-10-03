# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 layers (text + vision) for InternVLA-A1.5, written for fixed-shape Neuron graphs.

Module and parameter names mirror upstream's ``modeling_qwen3_5`` (the copy vendored by
InternVLA-A1.5 under ``transformers_replace``) so checkpoints load without renaming. The math
follows the upstream *eager* path op for op, with three graph-friendly rewrites that are exact
up to fp32 rounding:

* Gated DeltaNet's intra-chunk forward substitution (a 63-step Python loop of slice updates)
  is the inverse of a unit lower-triangular matrix; we compute it as a product of
  ``log2(chunk)`` factors ``(I + A^(2^k))`` (``A`` is nilpotent), i.e. a handful of batched
  64x64 matmuls instead of 63 sequential dynamic-update-slices.
* The depthwise causal ``conv1d`` (kernel 4) is four shifted multiply-adds in fp32.
* The vision encoder's per-image variable-length attention is a batched attention over
  equal-size images (all camera views share one resolution).

Host-only bookkeeping (mRoPE position ids, vision rotary / position-embedding interpolation
tables, attention biases) lives in :mod:`.preprocess` and enters the graphs as tensors.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextConfig, VisionConfig

# -- norms ------------------------------------------------------------------------------------


class Qwen35RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32, cast back (zero-init weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """Gated DeltaNet output norm: plain-weight RMSNorm, then ``* silu(gate)`` in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        out = self.weight * xf.to(dtype)
        out = out * F.silu(gate.float())
        return out.to(dtype)


# -- rotary -----------------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q/k ``[B, H, S, D]``; cos/sin ``[B, S, R]`` with ``R <= D`` (Qwen3.5: R = D/4)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    r = cos.shape[-1]
    q_rot, q_pass = q[..., :r], q[..., r:]
    k_rot, k_pass = k[..., :r], k[..., r:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def eager_attention(q, k, v, bias, scaling):
    """Upstream ``eager_attention_forward``: model-dtype scores, fp32 bias add + softmax.

    q ``[B, H, Sq, D]``, k/v ``[B, Hkv, Sk, D]``, bias additive ``[B, 1, Sq, Sk]`` (fp32) or None.
    Returns ``[B, Sq, H, D]``.
    """
    n_rep = q.shape[1] // k.shape[1]
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    w = torch.matmul(q, k.transpose(2, 3)) * scaling
    if bias is not None:
        w = w + bias
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, v).transpose(1, 2)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``F.gelu(x, approximate="tanh")`` written by hand: the Neuron XLA build monkey-patches
    ``torch._C._nn.gelu``, and Dynamo's fullgraph tracer cannot step into that patched builtin
    (``torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped``) once it is
    reached from inside a larger compiled graph (a lone call, as in the Wan2.2 port, is fine)."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf-based) GELU, same reasoning as :func:`gelu_tanh`."""
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


# -- Gated DeltaNet ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """``(I - A)^-1`` for strictly lower-triangular ``A`` ``[..., C, C]`` (C a power of two).

    Block-recursive forward substitution, halving the block size each step: split ``A`` into
    quadrants ``[[A11,0],[A21,A22]]``, solve the two half-size diagonal blocks recursively, then
    ``T21 = T22 @ A21 @ T11``. This is the textbook stable triangular-inverse recursion -- each
    step composes bounded sub-inverses, so it never forms the intermediate powers ``A^2, A^4, ...``
    a repeated-squaring Neumann series would. Those powers are mathematically forced to cancel back
    down to ``A^64 = 0``, but on the way there they blow past 1e12 for a real (checkpoint, not
    random-init) Gated-DeltaNet gate -- measured on InternVLA-A1.5-base layer 8: step magnitudes
    0.85 -> 32 -> 9.8e3 -> 3.2e7 -> 3.8e11 -> NaN. The block recursion has no such blowup because
    every intermediate value it ever forms is itself a bounded sub-block of the final answer.
    """
    c = a.shape[-1]
    if c == 1:
        return torch.ones_like(a)
    h = c // 2
    a11, a21, a22 = a[..., :h, :h], a[..., h:, :h], a[..., h:, h:]
    t11 = unit_lower_inverse(a11)
    t22 = unit_lower_inverse(a22)
    t21 = torch.matmul(torch.matmul(t22, a21), t11)
    zeros_tr = torch.zeros(*a.shape[:-2], h, c - h, dtype=a.dtype, device=a.device)
    top = torch.cat([t11, zeros_tr], dim=-1)
    bottom = torch.cat([t21, t22], dim=-1)
    return torch.cat([top, bottom], dim=-2)


# M3 perf: profiled the prefix's ~400ms warm latency on real device weights (job
# m3-prefix-layers2): the 18 Gated-DeltaNet layers cost 99ms (5.5ms/call) vs the 6 full-attention
# layers' 6.3ms (1.05ms/call) -- dispatch-bound (6 sequential chunks of small ops/layer), not
# FLOP-bound. Tried chunk_size=128 (3 chunks instead of 6, bit-identical result): the LARGER fused
# chunk graph more than doubled compile time and still hadn't finished at the 1806s job timeout
# (job m3-chunk128-device, killed). Reverted to 64. The real lever is sharing ONE compiled graph
# across the 18 layers (SplitPrefix already does this architecturally) and reducing per-call
# Python/host overhead, not changing the per-layer math.
GDN_CHUNK_SIZE = int(os.environ.get("INTERNVLA_GDN_CHUNK_SIZE", "64"))


def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size: int = 64):
    """Upstream ``torch_chunk_gated_delta_rule`` (no initial state, qk l2-norm in kernel).

    q/k ``[B, S, H, Dk]``, v ``[B, S, H, Dv]``, g/beta ``[B, S, H]``. Returns ``[B, S, H, Dv]``
    in q's dtype. Internally fp32, like upstream.
    """
    dtype = q.dtype
    q = l2norm(q)
    k = l2norm(k)
    q, k, v, beta, g = (t.transpose(1, 2).contiguous().float() for t in (q, k, v, beta, g))
    b, h, s, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk_size - s % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    n = (s + pad) // chunk_size
    q = q * (1.0 / math.sqrt(dk))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v_beta, k_beta = (t.reshape(b, h, n, chunk_size, t.shape[-1]) for t in (q, k, v_beta, k_beta))
    g = g.reshape(b, h, n, chunk_size).cumsum(dim=-1)

    lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device))
    strict = torch.tril(torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device), diagonal=-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    # Multiplicative float masks, not torch.where(bool_mask, x, 0): the Neuron compiler rejects a
    # broadcast where/select here (NCC_IINAR001 TensorScalarAffineSelect, invalid element count).
    # exp() is clamped to the lower triangle first so the masked-out upper entries can't overflow.
    decay = torch.exp(diff * lower) * lower
    a = -(torch.matmul(k_beta, k.transpose(-1, -2)) * decay) * strict
    t_inv = unit_lower_inverse(a)
    value = torch.matmul(t_inv, v_beta)
    k_cumdecay = torch.matmul(t_inv, k_beta * g.exp().unsqueeze(-1))

    state = torch.zeros(b, h, dk, dv, dtype=torch.float32, device=q.device)
    outs = []
    for i in range(n):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], value[:, :, i], g[:, :, i]
        attn = (torch.matmul(q_i, k_i.transpose(-1, -2)) * decay[:, :, i]) * lower
        v_new = v_i - torch.matmul(k_cumdecay[:, :, i], state)
        inter = torch.matmul(q_i * g_i.exp().unsqueeze(-1), state)
        outs.append(inter + torch.matmul(attn, v_new))
        g_last = g_i[..., -1:]
        state = state * g_last.exp().unsqueeze(-1) + torch.matmul(
            (k_i * (g_last - g_i).exp().unsqueeze(-1)).transpose(-1, -2), v_new
        )
    out = torch.stack(outs, dim=2).reshape(b, h, n * chunk_size, dv)[:, :, :s]
    return out.transpose(1, 2).contiguous().to(dtype)


class _Conv1dWeight(nn.Module):
    """Holds ``conv1d.weight`` ``[C, 1, K]`` (the depthwise causal conv is applied by hand)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel))


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, cfg: TextConfig, hidden_size: int):
        super().__init__()
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.kernel = cfg.linear_conv_kernel_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = _Conv1dWeight(conv_dim, self.kernel)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)

    def causal_conv_silu(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv over the sequence, ``x`` ``[B, S, C]``; fp32 accumulate."""
        s, kk = x.shape[1], self.kernel
        w = self.conv1d.weight[:, 0, :].float()  # [C, K]
        xp = F.pad(x.float(), (0, 0, kk - 1, 0))
        acc = xp[:, 0:s] * w[:, 0]
        for j in range(1, kk):
            acc = acc + xp[:, j : j + s] * w[:, j]
        return F.silu(acc.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        mixed = self.causal_conv_silu(self.in_proj_qkv(x))
        z = self.in_proj_z(x).reshape(b, s, -1, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, s, -1, self.head_k_dim)
        k = k.reshape(b, s, -1, self.head_k_dim)
        v = v.reshape(b, s, -1, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        core = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=GDN_CHUNK_SIZE)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(b, s, -1))


# -- gated full attention ---------------------------------------------------------------------


class Qwen35Attention(nn.Module):
    def __init__(self, cfg: TextConfig, hidden_size: int):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def project(self, x, cos, sin):
        """-> q ``[B,H,S,D]`` (roped), k ``[B,Hkv,S,D]`` (normed+roped), v, gate ``[B,S,H*D]``."""
        b, s, _ = x.shape
        qg = self.q_proj(x).view(b, s, -1, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(b, s, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, -1, self.head_dim).transpose(1, 2)
        q, k = apply_partial_rope(q, k, cos, sin)
        return q, k, v, gate

    def finish(self, attn, gate):
        b, s = attn.shape[:2]
        attn = attn.reshape(b, s, -1) * torch.sigmoid(gate)
        return self.o_proj(attn)


class Qwen35MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(cfg, hidden_size)
        else:
            self.self_attn = Qwen35Attention(cfg, hidden_size)
        self.mlp = Qwen35MLP(hidden_size, intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=cfg.rms_norm_eps)

    def ffn(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward_linear(self, x):
        return self.ffn(x + self.linear_attn(self.input_layernorm(x)))

    def forward_full(self, x, cos, sin, bias, past_kv=None, kv_only: bool = False):
        """Full-attention layer. ``past_kv`` = prefix (K, V) prepended to this layer's own.

        Returns ``(x_out, (k, v))`` with this layer's own roped K / V; with ``kv_only`` the
        attention and MLP are skipped (``x_out`` is None) -- the prefix's last layer only has to
        hand its K/V to the action expert.
        """
        h = self.input_layernorm(x)
        q, k, v, gate = self.self_attn.project(h, cos, sin)
        if kv_only:
            return None, (k, v)
        kk, vv = (k, v) if past_kv is None else (torch.cat([past_kv[0], k], 2), torch.cat([past_kv[1], v], 2))
        attn = eager_attention(q, kk, vv, bias, self.self_attn.scaling)
        x = x + self.self_attn.finish(attn, gate)
        return self.ffn(x), (k, v)


class Qwen35TextModel(nn.Module):
    def __init__(self, cfg: TextConfig, hidden_size: int | None = None, intermediate_size: int | None = None,
                 with_embeddings: bool = True):
        super().__init__()
        hidden_size = hidden_size or cfg.hidden_size
        intermediate_size = intermediate_size or cfg.intermediate_size
        self.cfg = cfg
        if with_embeddings:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(cfg, i, hidden_size, intermediate_size) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(hidden_size, eps=cfg.rms_norm_eps)


# -- vision -----------------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        k = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size, kernel_size=k, stride=k, bias=True)

    def forward(self, x):  # [N, C*T*P*P] -> [N, hidden]: the stride==kernel Conv3d as a matmul
        w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
        return F.linear(x.to(w.dtype), w, self.proj.bias)


class _VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.scaling = cfg.head_dim**-0.5

    def forward(self, x, cos, sin):  # x [N, P, D] (N equal-size images), cos/sin [P, head_dim] fp32
        n, p, d = x.shape
        qkv = self.qkv(x).reshape(n, p, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # [3,N,H,P,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        dtype = q.dtype
        c, s = cos[None, None], sin[None, None]
        q = (q.float() * c + rotate_half(q.float()) * s).to(dtype)
        k = (k.float() * c + rotate_half(k.float()) * s).to(dtype)
        o = eager_attention(q, k, v, None, self.scaling)  # [N, P, H, hd]
        return self.proj(o.reshape(n, p, d))


class _VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))


class _VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = _VisionAttention(cfg)
        self.mlp = _VisionMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _PatchMerger(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.hidden = cfg.hidden_size * cfg.spatial_merge_size**2
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.linear_fc2 = nn.Linear(self.hidden, cfg.out_hidden_size)

    def forward(self, x):
        x = self.norm(x).reshape(-1, self.hidden)
        return self.linear_fc2(gelu_exact(self.linear_fc1(x)))


class Qwen35VisionModel(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = _PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([_VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = _PatchMerger(cfg)

    def forward(self, patches, pe_idx, pe_w, cos, sin):
        """patches ``[N, P, C*T*p*p]`` (N images of one size, P patches each, merge-block order);
        pe_idx/pe_w ``[4, P]`` bilinear position-embedding taps; cos/sin ``[P, head_dim]``.
        Returns merged image tokens ``[N * P / merge^2, out_hidden]``."""
        n, p, k = patches.shape
        x = self.patch_embed(patches.reshape(n * p, k)).reshape(n, p, -1)
        w = pe_w.to(self.pos_embed.weight.dtype)
        taps = self.pos_embed(pe_idx) * w[..., None]  # [4, P, D], model dtype as upstream
        pe = taps[0] + taps[1] + taps[2] + taps[3]
        x = x + pe[None]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.merger(x.reshape(-1, x.shape[-1]))
