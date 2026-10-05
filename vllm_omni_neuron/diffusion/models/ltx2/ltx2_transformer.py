# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 / LTX-2.3 audio-video DiT for Neuron.

One denoiser call maps patchified video latents ``[B, N, 128]`` and audio latents
``[B, Na, 128]`` (plus the connector text embeddings) to velocities. The math is upstream's
``LTX2VideoTransformer3DModel.forward`` (vendored in ``_vendor/transformer_ltx2.py``); the
split across host and device is the Neuron part:

* **Host (fp32):** every timestep-only quantity, per step -- the six AdaLN-single MLPs
  (``time_embed``, ``audio_time_embed``, the four a2v/v2a modulation embedders and the two
  prompt AdaLNs) -- and the four RoPE tables, once per request (they depend only on the token
  coordinates; the device copies are reused across steps, as are the text embeddings). This
  keeps ~0.34 B parameters off the device. Upstream computes these in the model dtype; here
  they are exact fp32 and cast once.
* **Device:** three graph kinds. ``head`` (``proj_in`` + the LTX-2.5 keyframe marker),
  ``chunk`` (``blocks_per_graph`` transformer blocks) and ``tail`` (output norm, modulation,
  ``proj_out``). Block weights enter ``chunk`` as graph *inputs*, so the 48 blocks reuse one
  compiled graph; three distinct NEFFs serve the whole model.
* **TP:** every attention (video/audio self, video/audio text cross, a2v, v2a) shards its
  heads; Q/K/V and the per-head gate are column-parallel, the output projection is
  row-parallel with an all-reduce. ``rms_norm_across_heads`` QK-norm needs the full-width
  mean square, so the local sums of squares are all-reduced (Q and K in one collective).
  The feed-forwards are column/row parallel. Hidden states stay replicated.

The text cross-attention is unmasked: LTX-2.3/2.5 connectors replace padding with learnable
registers and return an all-ones mask (vLLM-Omni drops it the same way).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

ATTN_NAMES = (
    "attn1",
    "audio_attn1",
    "attn2",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
ATTN_FIELDS = ("q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "nq", "nk", "g_w", "g_b", "o_w", "o_b")
TABLE_NAMES = (
    "scale_shift_table",
    "audio_scale_shift_table",
    "prompt_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "video_a2v_cross_attn_scale_shift_table",
    "audio_a2v_cross_attn_scale_shift_table",
)


@dataclass
class LTX2DiTConfig:
    num_layers: int = 48
    heads: int = 32
    head_dim: int = 128
    audio_heads: int = 32
    audio_head_dim: int = 64
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048
    in_channels: int = 128
    out_channels: int = 128
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    norm_eps: float = 1e-6
    ff_bias: bool = True
    audio_ff_bias: bool = True
    gated_attn: bool = True
    audio_gated_attn: bool = True
    cross_attn_mod: bool = True
    audio_cross_attn_mod: bool = True
    use_prompt_adaln_single: bool = True
    use_keyframes_abs_pos_embedding: bool = False
    timestep_scale_multiplier: float = 1000.0
    cross_attn_timestep_scale_multiplier: float = 1000.0
    rope_type: str = "split"
    raw: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def audio_dim(self) -> int:
        return self.audio_heads * self.audio_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> LTX2DiTConfig:
        if cfg.get("rope_type", "interleaved") != "split":
            raise NotImplementedError("only rope_type='split' (LTX-2.3 / LTX-2.5) is supported")
        if cfg.get("use_prompt_embeddings", True):
            # LTX-2.0 projects captions inside the DiT; LTX-2.3+ does it in the connectors.
            raise NotImplementedError("LTX-2.0 (use_prompt_embeddings=True) is not supported")
        if cfg.get("patch_size", 1) != 1 or cfg.get("patch_size_t", 1) != 1:
            raise NotImplementedError("patch_size 1 only")
        return cls(
            num_layers=cfg["num_layers"],
            heads=cfg["num_attention_heads"],
            head_dim=cfg["attention_head_dim"],
            audio_heads=cfg["audio_num_attention_heads"],
            audio_head_dim=cfg["audio_attention_head_dim"],
            cross_attention_dim=cfg["cross_attention_dim"],
            audio_cross_attention_dim=cfg["audio_cross_attention_dim"],
            in_channels=cfg["in_channels"],
            out_channels=cfg.get("out_channels") or cfg["in_channels"],
            audio_in_channels=cfg["audio_in_channels"],
            audio_out_channels=cfg.get("audio_out_channels") or cfg["audio_in_channels"],
            norm_eps=cfg.get("norm_eps", 1e-6),
            ff_bias=cfg.get("ff_bias", True),
            audio_ff_bias=cfg.get("audio_ff_bias", True),
            gated_attn=cfg.get("gated_attn", False),
            audio_gated_attn=cfg.get("audio_gated_attn", False),
            cross_attn_mod=cfg.get("cross_attn_mod", False),
            audio_cross_attn_mod=cfg.get("audio_cross_attn_mod", False),
            use_prompt_adaln_single=cfg.get("use_prompt_adaln_single", True),
            use_keyframes_abs_pos_embedding=cfg.get("use_keyframes_abs_pos_embedding", False),
            timestep_scale_multiplier=cfg.get("timestep_scale_multiplier", 1000),
            cross_attn_timestep_scale_multiplier=cfg.get(
                "cross_attn_timestep_scale_multiplier", 1000
            ),
            rope_type=cfg["rope_type"],
            raw=dict(cfg),
        )

    @classmethod
    def from_dir(cls, transformer_dir: str) -> LTX2DiTConfig:
        with open(os.path.join(transformer_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))

    def attn_geometry(self, name: str) -> tuple[int, int, int, int, int]:
        """(heads, head_dim, query_dim, kv_dim, out_dim) of one block attention."""
        d, da = self.dim, self.audio_dim
        return {
            "attn1": (self.heads, self.head_dim, d, d, d),
            "audio_attn1": (self.audio_heads, self.audio_head_dim, da, da, da),
            "attn2": (self.heads, self.head_dim, d, self.cross_attention_dim, d),
            "audio_attn2": (
                self.audio_heads,
                self.audio_head_dim,
                da,
                self.audio_cross_attention_dim,
                da,
            ),
            "audio_to_video_attn": (self.audio_heads, self.audio_head_dim, d, da, d),
            "video_to_audio_attn": (self.audio_heads, self.audio_head_dim, da, d, da),
        }[name]

    def gated(self, name: str) -> bool:
        video_q = name in ("attn1", "attn2", "audio_to_video_attn")
        return self.gated_attn if video_q else self.audio_gated_attn


# ---------------------------------------------------------------------------------------------
# TP helpers
# ---------------------------------------------------------------------------------------------
def tp_state() -> tuple[int, int, object]:
    """(tp_size, tp_rank, tp_group); (1, 0, None) when vLLM's TP group is not initialized."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError):
        return 1, 0, None
    cp = cp_state()[0]
    if size > 1 or cp > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=cp)
    return size, rank, group


def cp_state() -> tuple[int, int, object]:
    """(cp_size, cp_rank, cp GroupCoordinator); (1, 0, None) without a sequence-parallel group.

    CP is vLLM-Omni's sequence-parallel group (stage ``parallel_config.ring_degree``)."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        g = get_sp_group()
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    if g is None or g.world_size <= 1:
        return 1, 0, None
    return g.world_size, g.rank_in_group, g


class CPContext:
    """Context parallelism over the video tokens: each CP rank holds ``N / cp`` video tokens;
    the audio tokens (126 for a 5 s clip) and the text stay replicated on every CP rank.

    Only two attentions see other ranks' tokens: the video self-attention (local Q over all
    video K/V) and video-to-audio (audio Q over all video K/V). Their K/V are projected,
    QK-normed and RoPE'd locally (all per-token) and then all-gathered over the CP group; the
    video self-attention can use the plugin's ring-attention kernel instead
    (``LTX2_CP_RING=1``, NeuronCore-v3+), which never materializes the full K/V."""

    def __init__(self, size: int, rank: int, group, tp: int):
        self.size, self.rank, self.group = size, rank, group
        self.replica_groups = None
        self.ring = size > 1 and os.environ.get("LTX2_CP_RING", "0") == "1"
        if self.ring:
            from vllm_omni_neuron.diffusion.distributed.parallel_state import (
                get_cp_replica_groups,
            )

            self.replica_groups = get_cp_replica_groups(tp, size)

    def gather(self, t: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return self.group.all_gather(t.contiguous(), dim=dim)

    def self_attention(self, q, k, v, scale):
        """``q/k/v [B, S/cp, H, D]`` (local tokens) -> ``[B, S/cp, H, D]`` over all tokens."""
        if self.ring and q.device.type != "cpu":
            from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
                can_run_kernel,
                wan_cp_self_attention,
            )

            if can_run_kernel(v):
                o = wan_cp_self_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    scale,
                    self.size,
                    self.group,
                    self.replica_groups,
                )  # [B, H, D, S/cp]
                return o.permute(0, 3, 1, 2)
        return sdpa(q, self.gather(k), self.gather(v), scale)


def _all_reduce(x: torch.Tensor, tp: int, group) -> torch.Tensor:
    if tp > 1:
        dist.all_reduce(x, group=group)
    return x


@contextmanager
def host_math_threads(n: int | None = None):
    """Run the host conditioning (timestep embeddings, RoPE tables) with a fixed torch thread count
    (``LTX2_HOST_MATH_THREADS``, default 8: 26 ms per call on the host conditioning vs 137 ms with
    1 thread). The fp32 CPU kernels' results depend on the thread count (1 vs 24 threads differ in
    the timestep embeddings), so a fixed count makes the DiT inputs the same in every process: the
    served worker, a torchrun gate, a standalone script."""
    n = n or int(os.environ.get("LTX2_HOST_MATH_THREADS", "8"))
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# ---------------------------------------------------------------------------------------------
# Math (pure functions over explicit tensors, so one compiled graph serves every block)
# ---------------------------------------------------------------------------------------------
def rms_norm_noaffine(x: torch.Tensor, eps: float) -> torch.Tensor:
    """diffusers ``RMSNorm(elementwise_affine=False)``: fp32 statistics, cast back."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def apply_split_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Upstream ``apply_split_rotary_emb`` on ``x [B, S, H, D]`` with ``cos/sin [B, S, H, D/2]``."""
    xf = x.float()
    r = x.shape[-1] // 2
    x1, x2 = xf[..., :r], xf[..., r:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def _torch_attention(q, k, v, scale):
    """softmax(scale QK^T) V over ``[B, H, S, D]``; fp32 softmax (attention_cte's math)."""
    if q.device.type == "cpu":
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    scores = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """``q [B, Sq, H, D]``, ``k/v [B, Sk, H, D]`` -> ``[B, Sq, H, D]``.

    NeuronCore-v3+: the plugin's ``attention_cte`` NKI kernel (Wan2.2's wrapper); else torch.
    """
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if q.device.type != "cpu" and os.environ.get("LTX2_ATTN_IMPL", "auto") != "torch":
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
            _can_use_wan_attention_kernel,
            _nki_attend,
        )

        if _can_use_wan_attention_kernel(qh, kh, vh):
            return _nki_attend(qh.contiguous(), kh.contiguous(), vh.contiguous(), scale).permute(
                0, 3, 1, 2
            )
    return _torch_attention(qh, kh, vh, scale).transpose(1, 2)


def attention(
    w: dict,
    xq,
    xkv,
    q_rope,
    k_rope,
    heads: int,
    head_dim: int,
    eps: float,
    tp: int,
    group,
    cp: CPContext | None = None,
    cp_mode: str | None = None,
):
    """One LTX2 attention (TP-sharded heads). ``w`` holds this rank's slices; ``heads`` is local.

    ``cp_mode``: None (all tokens local), ``"self"`` (Q and K/V are this rank's video tokens and
    attend over all CP ranks' tokens) or ``"kv"`` (only K/V are video tokens: gather them).
    """
    b, sq, _ = xq.shape
    sk = xkv.shape[1]
    q = F.linear(xq, w["q_w"], w["q_b"])
    k = F.linear(xkv, w["k_w"], w["k_b"])
    v = F.linear(xkv, w["v_w"], w["v_b"])
    # rms_norm_across_heads: the mean square runs over ALL heads -> all-reduce the local sums.
    full = heads * head_dim * tp
    qf, kf = q.float(), k.float()
    if tp > 1 and sq == sk:
        ss = torch.cat([qf.pow(2).sum(-1, keepdim=True), kf.pow(2).sum(-1, keepdim=True)], dim=-1)
        ss = _all_reduce(ss, tp, group)
        ssq, ssk = ss[..., :1], ss[..., 1:]
    else:
        ssq = _all_reduce(qf.pow(2).sum(-1, keepdim=True), tp, group)
        ssk = _all_reduce(kf.pow(2).sum(-1, keepdim=True), tp, group)
    q = (qf * torch.rsqrt(ssq / full + eps)).to(q.dtype) * w["nq"]
    k = (kf * torch.rsqrt(ssk / full + eps)).to(k.dtype) * w["nk"]
    q = q.view(b, sq, heads, head_dim)
    k = k.view(b, sk, heads, head_dim)
    v = v.view(b, sk, heads, head_dim)
    if q_rope is not None:
        q = apply_split_rope(q, *q_rope)
        k = apply_split_rope(k, *(k_rope if k_rope is not None else q_rope))
    if cp is not None and cp_mode == "self":
        o = cp.self_attention(q, k, v, head_dim**-0.5).to(xq.dtype)
    elif cp is not None and cp_mode == "kv":
        o = sdpa(q, cp.gather(k), cp.gather(v), head_dim**-0.5).to(xq.dtype)
    else:
        o = sdpa(q, k, v, head_dim**-0.5).to(xq.dtype)  # [B, Sq, H, D]
    if "g_w" in w:
        gates = 2.0 * torch.sigmoid(F.linear(xq, w["g_w"], w["g_b"]))  # [B, Sq, H]
        o = o * gates.unsqueeze(-1)
    out = _all_reduce(F.linear(o.reshape(b, sq, heads * head_dim), w["o_w"]), tp, group)
    return out + w["o_b"] if w.get("o_b") is not None else out


def feed_forward(w: dict, x, tp: int, group):
    h = F.gelu(F.linear(x, w["up_w"], w.get("up_b")), approximate="tanh")
    out = _all_reduce(F.linear(h, w["down_w"]), tp, group)
    return out + w["down_b"] if w.get("down_b") is not None else out


def _mods(table: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Upstream ``get_mod_params``: ``table [n, D] + temb [B, T, n*D]`` -> n x ``[B, T, D]``."""
    b, t = temb.shape[:2]
    return (table[None, None] + temb.reshape(b, t, table.shape[0], -1)).unbind(dim=2)


def block_forward(
    cfg: LTX2DiTConfig,
    w: dict,
    x,
    ax,
    tv,
    ta,
    ctx: dict,
    tp: int,
    group,
    a2v: bool = True,
    cp: CPContext | None = None,
):
    """One ``LTX2VideoTransformerBlock`` (no STG perturbation; masks are all-ones). With ``cp``,
    ``x`` holds this CP rank's video tokens (see :class:`CPContext`)."""
    cp = cp if cp is not None and cp.size > 1 else None
    eps = cfg.norm_eps
    hl, hd = cfg.heads // tp, cfg.head_dim
    ahl, ahd = cfg.audio_heads // tp, cfg.audio_head_dim
    vrope, arope = ctx["video_rope"], ctx["audio_rope"]
    cvrope, carope = ctx["ca_video_rope"], ctx["ca_audio_rope"]

    vm = _mods(w["scale_shift_table"], ctx["temb"])
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vm[:6]
    am = _mods(w["audio_scale_shift_table"], ctx["temb_audio"])
    a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = am[:6]

    # 1. self-attention
    n = rms_norm_noaffine(x, eps) * (1 + scale_msa) + shift_msa
    o = attention(w["attn1"], n, n, vrope, None, hl, hd, eps, tp, group, cp, "self")
    x = x + o * gate_msa
    n = rms_norm_noaffine(ax, eps) * (1 + a_scale_msa) + a_shift_msa
    ax = ax + attention(w["audio_attn1"], n, n, arope, None, ahl, ahd, eps, tp, group) * a_gate_msa

    # 2. text cross-attention (K/V modulation by the prompt AdaLN / static table)
    if cfg.cross_attn_mod or cfg.audio_cross_attn_mod:
        if ctx.get("temb_prompt") is not None:
            shift_kv, scale_kv = _mods(w["prompt_scale_shift_table"], ctx["temb_prompt"])
            a_shift_kv, a_scale_kv = _mods(
                w["audio_prompt_scale_shift_table"], ctx["temb_prompt_audio"]
            )
        else:
            shift_kv, scale_kv = w["prompt_scale_shift_table"][None, None].unbind(2)
            a_shift_kv, a_scale_kv = w["audio_prompt_scale_shift_table"][None, None].unbind(2)
        tv = tv * (1 + scale_kv) + shift_kv
        ta = ta * (1 + a_scale_kv) + a_shift_kv
    n = rms_norm_noaffine(x, eps)
    if cfg.cross_attn_mod:
        n = n * (1 + vm[7]) + vm[6]
    o = attention(w["attn2"], n, tv, None, None, hl, hd, eps, tp, group)
    x = x + (o * vm[8] if cfg.cross_attn_mod else o)
    n = rms_norm_noaffine(ax, eps)
    if cfg.audio_cross_attn_mod:
        n = n * (1 + am[7]) + am[6]
    o = attention(w["audio_attn2"], n, ta, None, None, ahl, ahd, eps, tp, group)
    ax = ax + (o * am[8] if cfg.audio_cross_attn_mod else o)

    # 3. audio<->video cross-attention
    if a2v:
        nv = rms_norm_noaffine(x, eps)
        na = rms_norm_noaffine(ax, eps)
        vt = w["video_a2v_cross_attn_scale_shift_table"]
        at = w["audio_a2v_cross_attn_scale_shift_table"]
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = _mods(vt[:4], ctx["ca_ss"])
        (a2v_gate,) = _mods(vt[4:], ctx["ca_gate"])
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = _mods(at[:4], ctx["ca_ss_audio"])
        (v2a_gate,) = _mods(at[4:], ctx["ca_gate_audio"])
        qv = nv * (1 + v_a2v_scale) + v_a2v_shift
        kva = na * (1 + a_a2v_scale) + a_a2v_shift
        x_new = x + a2v_gate * attention(
            w["audio_to_video_attn"], qv, kva, cvrope, carope, ahl, ahd, eps, tp, group
        )
        qa = na * (1 + a_v2a_scale) + a_v2a_shift
        kvv = nv * (1 + v_v2a_scale) + v_v2a_shift
        ax = ax + v2a_gate * attention(
            w["video_to_audio_attn"], qa, kvv, carope, cvrope, ahl, ahd, eps, tp, group, cp, "kv"
        )
        x = x_new

    # 4. feed-forward
    n = rms_norm_noaffine(x, eps) * (1 + scale_mlp) + shift_mlp
    x = x + feed_forward(w["ff"], n, tp, group) * gate_mlp
    n = rms_norm_noaffine(ax, eps) * (1 + a_scale_mlp) + a_shift_mlp
    ax = ax + feed_forward(w["audio_ff"], n, tp, group) * a_gate_mlp
    return x, ax


CTX_KEYS = (
    "temb",
    "temb_audio",
    "ca_ss",
    "ca_ss_audio",
    "ca_gate",
    "ca_gate_audio",
    "temb_prompt",
    "temb_prompt_audio",
)
ROPE_KEYS = ("video_rope", "audio_rope", "ca_video_rope", "ca_audio_rope")


# ---------------------------------------------------------------------------------------------
# Weight bookkeeping
# ---------------------------------------------------------------------------------------------
def block_weight_specs(cfg: LTX2DiTConfig, tp: int) -> list[tuple[str, str, int | None, tuple]]:
    """Per-block ``(local_key, checkpoint_suffix, shard_dim, local_shape)`` in a fixed order.

    ``shard_dim`` is the dim split across TP ranks (None = replicated). The order defines the
    flat argument list of the compiled chunk graph.
    """
    specs = []
    for name in ATTN_NAMES:
        h, hd, dq, dkv, dout = cfg.attn_geometry(name)
        inner, hl = h * hd, h // tp
        il = hl * hd
        p = f"{name}."
        specs += [
            (f"{name}.q_w", p + "to_q.weight", 0, (il, dq)),
            (f"{name}.q_b", p + "to_q.bias", 0, (il,)),
            (f"{name}.k_w", p + "to_k.weight", 0, (il, dkv)),
            (f"{name}.k_b", p + "to_k.bias", 0, (il,)),
            (f"{name}.v_w", p + "to_v.weight", 0, (il, dkv)),
            (f"{name}.v_b", p + "to_v.bias", 0, (il,)),
            (f"{name}.nq", p + "norm_q.weight", 0, (il,)),
            (f"{name}.nk", p + "norm_k.weight", 0, (il,)),
        ]
        if cfg.gated(name):
            specs += [
                (f"{name}.g_w", p + "to_gate_logits.weight", 0, (hl, dq)),
                (f"{name}.g_b", p + "to_gate_logits.bias", 0, (hl,)),
            ]
        specs += [
            (f"{name}.o_w", p + "to_out.0.weight", 1, (dout, il)),
            (f"{name}.o_b", p + "to_out.0.bias", None, (dout,)),
        ]
        del inner
    for ff, dim, bias in (
        ("ff", cfg.dim, cfg.ff_bias),
        ("audio_ff", cfg.audio_dim, cfg.audio_ff_bias),
    ):
        hid = 4 * dim
        specs.append((f"{ff}.up_w", f"{ff}.net.0.proj.weight", 0, (hid // tp, dim)))
        if bias:
            specs.append((f"{ff}.up_b", f"{ff}.net.0.proj.bias", 0, (hid // tp,)))
        specs.append((f"{ff}.down_w", f"{ff}.net.2.weight", 1, (dim, hid // tp)))
        if bias:
            specs.append((f"{ff}.down_b", f"{ff}.net.2.bias", None, (dim,)))
    n_v = 9 if cfg.cross_attn_mod else 6
    n_a = 9 if cfg.audio_cross_attn_mod else 6
    specs += [
        ("scale_shift_table", "scale_shift_table", None, (n_v, cfg.dim)),
        ("audio_scale_shift_table", "audio_scale_shift_table", None, (n_a, cfg.audio_dim)),
    ]
    if cfg.cross_attn_mod or cfg.audio_cross_attn_mod:
        specs += [
            ("prompt_scale_shift_table", "prompt_scale_shift_table", None, (2, cfg.dim)),
            (
                "audio_prompt_scale_shift_table",
                "audio_prompt_scale_shift_table",
                None,
                (2, cfg.audio_dim),
            ),
        ]
    specs += [
        (
            "video_a2v_cross_attn_scale_shift_table",
            "video_a2v_cross_attn_scale_shift_table",
            None,
            (5, cfg.dim),
        ),
        (
            "audio_a2v_cross_attn_scale_shift_table",
            "audio_a2v_cross_attn_scale_shift_table",
            None,
            (5, cfg.audio_dim),
        ),
    ]
    return specs


def unflatten_block(keys: list[str], flat) -> dict:
    """Flat tensor list -> nested ``{attn_name: {field: t}, "ff": {...}, table: t}``."""
    w: dict = {}
    for key, t in zip(keys, flat, strict=True):
        if "." in key:
            mod, fld = key.split(".", 1)
            w.setdefault(mod, {})[fld] = t
        else:
            w[key] = t
    return w


class _SafetensorsIndex:
    """name -> file for a (possibly sharded) diffusers safetensors directory."""

    def __init__(self, path: str, prefix: str = "diffusion_pytorch_model"):
        self.path = path
        idx = os.path.join(path, f"{prefix}.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open

            single = os.path.join(path, f"{prefix}.safetensors")
            with safe_open(single, "pt") as f:
                wm = {k: f"{prefix}.safetensors" for k in f.keys()}
        self.weight_map = wm
        self._handles: dict = {}

    def _open(self, fname):
        from safetensors import safe_open

        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.path, fname), "pt")
        return self._handles[fname]

    def get(
        self, name: str, shard_dim: int | None = None, rank: int = 0, tp: int = 1
    ) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"{name} not in checkpoint {self.path}")
        f = self._open(self.weight_map[name])
        if shard_dim is None or tp == 1:
            return f.get_tensor(name)
        sl = f.get_slice(name)
        shape = sl.get_shape()
        n = shape[shard_dim] // tp
        idx = [slice(None)] * len(shape)
        idx[shard_dim] = slice(rank * n, (rank + 1) * n)
        return sl[tuple(idx)]

    def close(self):
        self._handles.clear()


# ---------------------------------------------------------------------------------------------
# Host conditioning (fp32): AdaLN-single MLPs + RoPE tables
# ---------------------------------------------------------------------------------------------
class LTX2HostConditioning(nn.Module):
    """Timestep / position quantities of one denoiser call, computed on the host in fp32."""

    def __init__(self, cfg: LTX2DiTConfig):
        super().__init__()
        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2AdaLayerNormSingle,
            LTX2AudioVideoRotaryPosEmbed,
        )

        raw = cfg.raw
        self.cfg = cfg
        d, da = cfg.dim, cfg.audio_dim
        self.time_embed = LTX2AdaLayerNormSingle(d, num_mod_params=9 if cfg.cross_attn_mod else 6)
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            da, num_mod_params=9 if cfg.audio_cross_attn_mod else 6
        )
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(d, num_mod_params=4)
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(da, num_mod_params=4)
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(d, num_mod_params=1)
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(da, num_mod_params=1)
        self.prompt_modulation = cfg.cross_attn_mod or cfg.audio_cross_attn_mod
        if self.prompt_modulation and cfg.use_prompt_adaln_single:
            self.prompt_adaln = LTX2AdaLayerNormSingle(d, num_mod_params=2)
            self.audio_prompt_adaln = LTX2AdaLayerNormSingle(da, num_mod_params=2)
        common = dict(
            patch_size=1,
            patch_size_t=1,
            base_height=raw.get("base_height", 2048),
            base_width=raw.get("base_width", 2048),
            theta=raw.get("rope_theta", 10000.0),
            causal_offset=raw.get("causal_offset", 1),
            double_precision=raw.get("rope_double_precision", True),
            rope_type=cfg.rope_type,
        )
        audio_kw = dict(
            sampling_rate=raw.get("audio_sampling_rate", 16000),
            hop_length=raw.get("audio_hop_length", 160),
        )
        vmax, amax = raw.get("pos_embed_max_pos", 20), raw.get("audio_pos_embed_max_pos", 20)
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=d,
            base_num_frames=vmax,
            scale_factors=tuple(raw.get("vae_scale_factors", (8, 32, 32))),
            modality="video",
            num_attention_heads=cfg.heads,
            **common,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=da,
            base_num_frames=amax,
            scale_factors=[raw.get("audio_scale_factor", 4)],
            modality="audio",
            num_attention_heads=cfg.audio_heads,
            **audio_kw,
            **common,
        )
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=cfg.audio_cross_attention_dim,
            base_num_frames=max(vmax, amax),
            modality="video",
            num_attention_heads=cfg.heads,
            **common,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=cfg.audio_cross_attention_dim,
            base_num_frames=max(vmax, amax),
            modality="audio",
            num_attention_heads=cfg.audio_heads,
            **audio_kw,
            **common,
        )
        self.requires_grad_(False)

    _PREFIXES = (
        "time_embed.",
        "audio_time_embed.",
        "av_cross_attn_video_scale_shift.",
        "av_cross_attn_audio_scale_shift.",
        "av_cross_attn_video_a2v_gate.",
        "av_cross_attn_audio_v2a_gate.",
        "prompt_adaln.",
        "audio_prompt_adaln.",
    )

    def load_from(self, ckpt: _SafetensorsIndex) -> None:
        sd = {}
        own = self.state_dict()
        for k in own:
            sd[k] = ckpt.get(k).float()
        self.load_state_dict(sd, strict=True)
        self.float()

    @staticmethod
    def _rope_local(freqs, rank: int, tp: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream ``(B, H, T, r)`` cos/sin -> this rank's heads as ``(B, T, H/tp, r)`` fp32."""
        out = []
        for f in freqs:
            h = f.shape[1] // tp
            out.append(f[:, rank * h : (rank + 1) * h].transpose(1, 2).float().contiguous())
        return out[0], out[1]

    @torch.no_grad()
    def rope_tables(
        self,
        b,
        num_frames,
        height,
        width,
        audio_num_frames,
        fps,
        video_coords,
        audio_coords,
        rank: int,
        tp: int,
    ) -> dict:
        """The four RoPE ``(cos, sin)`` tables of this rank's heads (fp32, upstream double
        precision). A function of the token coordinates only."""
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                b, num_frames, height, width, "cpu", fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(b, audio_num_frames, "cpu")
        return {
            "video_rope": self._rope_local(self.rope(video_coords, device="cpu"), rank, tp),
            "audio_rope": self._rope_local(self.audio_rope(audio_coords, device="cpu"), rank, tp),
            "ca_video_rope": self._rope_local(
                self.cross_attn_rope(video_coords[:, 0:1, :], device="cpu"), rank, tp
            ),
            "ca_audio_rope": self._rope_local(
                self.cross_attn_audio_rope(audio_coords[:, 0:1, :], device="cpu"), rank, tp
            ),
        }

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None,
        sigma: torch.Tensor | None,
        audio_sigma: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        audio_num_frames: int,
        fps: float = 24.0,
        use_cross_timestep: bool = True,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        rank: int = 0,
        tp: int = 1,
        with_rope: bool = True,
    ) -> dict:
        """Upstream forward steps 1 and 3 (RoPE, timestep embeddings) for one call.

        ``with_rope=False`` skips the four RoPE tables (see :meth:`rope_tables`): they depend only
        on the token coordinates, which are the same for every step of a request.

        ``timestep`` / ``audio_timestep`` are ``[B]`` (already scaled by
        ``timestep_scale_multiplier``, as upstream expects); per-token timesteps are not
        supported yet. ``sigma`` / ``audio_sigma`` default to ``timestep`` (the LTX-2.3/2.5
        pipelines pass ``sigma=timestep``).
        """
        cfg = self.cfg
        b = batch_size
        if timestep.ndim != 1:
            if timestep.ndim == 2 and bool((timestep == timestep[:, :1]).all()):
                timestep = timestep[:, 0]
            else:
                raise NotImplementedError(
                    "per-token timesteps (I2V conditioning) are not supported yet"
                )
        timestep = (
            timestep.float().reshape(-1).expand(b) if timestep.numel() == 1 else timestep.float()
        )
        audio_timestep = (
            timestep if audio_timestep is None else audio_timestep.float().reshape(-1).expand(b)
        )
        sigma = timestep if sigma is None else sigma.float().reshape(-1).expand(b)
        audio_sigma = sigma if audio_sigma is None else audio_sigma.float().reshape(-1).expand(b)

        ctx = {}
        if with_rope:
            ctx.update(
                self.rope_tables(
                    b,
                    num_frames,
                    height,
                    width,
                    audio_num_frames,
                    fps,
                    video_coords,
                    audio_coords,
                    rank,
                    tp,
                )
            )
        f32 = torch.float32

        def emb(mod, t):
            out, embedded = mod(t, batch_size=b, hidden_dtype=f32)
            return out.view(b, -1, out.shape[-1]), embedded.view(b, -1, embedded.shape[-1])

        gate_scale = cfg.cross_attn_timestep_scale_multiplier / cfg.timestep_scale_multiplier
        ctx["temb"], ctx["embedded_timestep"] = emb(self.time_embed, timestep)
        ctx["temb_audio"], ctx["audio_embedded_timestep"] = emb(
            self.audio_time_embed, audio_timestep
        )
        if self.prompt_modulation and cfg.use_prompt_adaln_single:
            ctx["temb_prompt"] = emb(self.prompt_adaln, sigma)[0]
            ctx["temb_prompt_audio"] = emb(self.audio_prompt_adaln, audio_sigma)[0]
        else:
            ctx["temb_prompt"] = ctx["temb_prompt_audio"] = None
        v_ca_t = audio_sigma if use_cross_timestep else timestep
        a_ca_t = sigma if use_cross_timestep else audio_timestep
        ctx["ca_ss"] = emb(self.av_cross_attn_video_scale_shift, v_ca_t)[0]
        ctx["ca_gate"] = emb(self.av_cross_attn_video_a2v_gate, v_ca_t * gate_scale)[0]
        ctx["ca_ss_audio"] = emb(self.av_cross_attn_audio_scale_shift, a_ca_t)[0]
        ctx["ca_gate_audio"] = emb(self.av_cross_attn_audio_v2a_gate, a_ca_t * gate_scale)[0]
        return ctx


# ---------------------------------------------------------------------------------------------
# The device model
# ---------------------------------------------------------------------------------------------
class NeuronLTX2Transformer(nn.Module):
    """LTX-2.3/2.5 AV DiT: host conditioning + ``head`` / ``chunk`` x L/K / ``tail`` graphs.

    ``forward`` has upstream's keyword interface (the subset LTX-2.3/2.5 T2V uses) and returns
    ``(video_velocity, audio_velocity)``.
    """

    def __init__(
        self, cfg: LTX2DiTConfig, dtype: torch.dtype = torch.bfloat16, blocks_per_graph: int = 4
    ):
        super().__init__()
        tp, rank, group = tp_state()
        cp_size, cp_rank, cp_group = cp_state()
        self.cp = CPContext(cp_size, cp_rank, cp_group, tp)
        if cfg.heads % tp or cfg.audio_heads % tp:
            raise ValueError(f"TP={tp} must divide heads ({cfg.heads}, {cfg.audio_heads})")
        if cfg.num_layers % blocks_per_graph:
            raise ValueError(
                f"blocks_per_graph={blocks_per_graph} must divide num_layers={cfg.num_layers}"
            )
        self.cfg, self.dtype = cfg, dtype
        self.tp, self.rank, self.group = tp, rank, group
        self.blocks_per_graph = blocks_per_graph
        self.specs = block_weight_specs(cfg, tp)
        self.block_keys = [s[0] for s in self.specs]
        self.host = LTX2HostConditioning(cfg)
        self.block_weights: list[list[torch.Tensor]] = []
        self.top: dict[str, torch.Tensor] = {}
        self.device = torch.device("cpu")
        self._head = self._chunk = self._tail = None
        self._step_cache: dict = {}  # per-request device tensors reused across steps
        # Minimal `.config` surface: the diffusers LTX2Pipeline reads exactly these three fields
        # off `self.transformer.config` (prepare_latents' patch sizes, in_channels for the noise
        # draw) -- SimpleNamespace rather than a real dataclass/ConfigMixin since nothing else
        # touches it.
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            patch_size=cfg.raw.get("patch_size", 1),
            patch_size_t=cfg.raw.get("patch_size_t", 1),
            in_channels=cfg.in_channels,
        )
        # The diffusers LTX2Pipeline precomputes video/audio coords itself (so the same tensor
        # is reused across CFG/STG branches) by calling `self.transformer.rope.prepare_*_coords`
        # directly, not through `forward`. Expose the host conditioning module's own rope
        # objects as plain attributes (object.__setattr__, not nn.Module's: these are references
        # to self.host's own submodules, and letting nn.Module re-register them under a second
        # name would duplicate them in state_dict()/named_parameters()).
        object.__setattr__(self, "rope", self.host.rope)
        object.__setattr__(self, "audio_rope", self.host.audio_rope)

    @classmethod
    def from_dir(
        cls, transformer_dir: str, dtype=torch.bfloat16, blocks_per_graph: int = 4, device="cpu"
    ):
        m = cls(
            LTX2DiTConfig.from_dir(transformer_dir), dtype=dtype, blocks_per_graph=blocks_per_graph
        )
        m.load_weights(transformer_dir, device)
        return m

    @contextmanager
    def cache_context(self, name: str):
        """No-op stand-in for diffusers' ``CacheMixin.cache_context``.

        The real one scopes a registered cache hook (e.g. a step-cache/KV-cache for CFG/STG
        branch reuse) to a named context; this port never registers one (no ``enable_cache``
        call), so on the real `CacheMixin` this would ALSO be a no-op here -- only the attribute
        needs to exist for the diffusers pipeline's ``with self.transformer.cache_context(...)``
        call sites.
        """
        yield

    # -- weights ------------------------------------------------------------------------------
    def load_weights(self, transformer_dir: str, device="cpu") -> None:
        """This rank's shard; device tensors are created directly (no full replica on device)."""
        dev = torch.device(device)
        ckpt = _SafetensorsIndex(transformer_dir)
        cfg = self.cfg
        self.host.load_from(ckpt)
        blocks = []
        for i in range(cfg.num_layers):
            flat = []
            for key, suffix, shard_dim, shape in self.specs:
                t = ckpt.get(f"transformer_blocks.{i}.{suffix}", shard_dim, self.rank, self.tp)
                if tuple(t.shape) != tuple(shape):
                    raise ValueError(
                        f"block {i} {suffix}: shape {tuple(t.shape)} != expected {shape}"
                    )
                flat.append(t.to(self.dtype).contiguous().to(dev))
            blocks.append(flat)
        self.block_weights = blocks
        top = {
            "proj_in.weight": None,
            "proj_in.bias": None,
            "audio_proj_in.weight": None,
            "audio_proj_in.bias": None,
            "proj_out.weight": None,
            "proj_out.bias": None,
            "audio_proj_out.weight": None,
            "audio_proj_out.bias": None,
            "scale_shift_table": None,
            "audio_scale_shift_table": None,
        }
        if cfg.use_keyframes_abs_pos_embedding:
            top["keyframes_abs_pos_embedding"] = None
        for k in top:
            top[k] = ckpt.get(k).to(self.dtype).contiguous().to(dev)
        self.top = top
        ckpt.close()
        self.device = dev

    def num_local_params(self) -> int:
        return sum(t.numel() for b in self.block_weights for t in b) + sum(
            t.numel() for t in self.top.values()
        )

    # -- graphs -------------------------------------------------------------------------------
    def head_fn(self, latents, audio_latents, keyframes_mask, pi_w, pi_b, api_w, api_b, kf):
        x = F.linear(latents, pi_w, pi_b)
        if kf is not None and keyframes_mask is not None:
            x = x + (keyframes_mask > 0).to(x.dtype) * kf
        return x, F.linear(audio_latents, api_w, api_b)

    def chunk_fn(self, x, ax, tv, ta, ctx_flat, rope_flat, *weights):
        ctx = dict(zip(CTX_KEYS, ctx_flat, strict=True))
        it = iter(rope_flat)
        for k in ROPE_KEYS:
            ctx[k] = (next(it), next(it))
        n = len(self.block_keys)
        for j in range(len(weights) // n):
            w = unflatten_block(self.block_keys, weights[j * n : (j + 1) * n])
            x, ax = block_forward(self.cfg, w, x, ax, tv, ta, ctx, self.tp, self.group, cp=self.cp)
        return x, ax

    @staticmethod
    def _out(x, table, emb, w, b):
        ss = table[None, None] + emb[:, :, None]
        shift, scale = ss[:, :, 0], ss[:, :, 1]
        n = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)
        return F.linear(n * (1 + scale) + shift, w, b)

    def tail_fn(self, x, ax, emb, aemb, sst, asst, po_w, po_b, apo_w, apo_b):
        v = self._out(x, sst, emb, po_w, po_b)
        if self.cp.size > 1:  # every rank returns the full video velocity
            v = self.cp.gather(v, dim=1)
        return v, self._out(ax, asst, aemb, apo_w, apo_b)

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        base = dict(options or {})
        kw = {"fullgraph": True, "dynamic": False, **kwargs}
        args = ["--model-type=transformer", "--auto-cast=none", "-O1"]

        def opts(name):
            return {**base, "model_name": name, "compiler_args": base.get("compiler_args", args)}

        self._head = torch.compile(
            self.head_fn, backend=backend, options=opts("ltx2_dit_head"), **kw
        )
        self._chunk = torch.compile(
            self.chunk_fn, backend=backend, options=opts("ltx2_dit_chunk"), **kw
        )
        self._tail = torch.compile(
            self.tail_fn, backend=backend, options=opts("ltx2_dit_tail"), **kw
        )

    # -- forward ------------------------------------------------------------------------------
    def _dev(self, t, dtype=None):
        if t is None:
            return None
        return t.to(dtype=dtype or t.dtype).contiguous().to(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        use_cross_timestep: bool = True,
        video_keyframes_mask: torch.Tensor | None = None,
        **unused,
    ):
        for k in ("isolate_modalities", "spatio_temporal_guidance_blocks", "perturbation_mask"):
            if unused.get(k):
                raise NotImplementedError(f"{k} is not supported yet")
        dt = self.dtype
        b = hidden_states.shape[0]
        audio_n = audio_num_frames or audio_hidden_states.shape[1]
        vc = None if video_coords is None else video_coords.cpu()
        ac = None if audio_coords is None else audio_coords.cpu()
        with host_math_threads():
            ctx = self.host(
                b,
                timestep.cpu(),
                None if audio_timestep is None else audio_timestep.cpu(),
                None if sigma is None else sigma.cpu(),
                None if audio_sigma is None else audio_sigma.cpu(),
                num_frames,
                height,
                width,
                audio_n,
                fps=fps,
                use_cross_timestep=use_cross_timestep,
                video_coords=vc,
                audio_coords=ac,
                rank=self.rank,
                tp=self.tp,
                with_rope=False,
            )
        ctx_flat = [self._dev(ctx[k], dt) if ctx[k] is not None else None for k in CTX_KEYS]
        # RoPE tables and the text embeddings are the same for every step of a request: build /
        # upload them once and reuse the device copies while the host inputs are unchanged (the
        # video RoPE alone is ~0.15 s of single-threaded host math plus ~40 MB of upload per step).
        n_video = hidden_states.shape[1]
        cps, cpr = self.cp.size, self.cp.rank
        if n_video % cps:
            raise ValueError(
                f"{n_video} video tokens are not divisible by the CP degree {cps}; pick a "
                "resolution / frame count whose latent token count is"
            )
        lo, hi = cpr * n_video // cps, (cpr + 1) * n_video // cps

        def build_rope():
            with host_math_threads():
                tables = self.host.rope_tables(
                    b, num_frames, height, width, audio_n, fps, vc, ac, self.rank, self.tp
                )
            if cps > 1:  # the video tables follow this rank's tokens; audio stays whole
                for k in ("video_rope", "ca_video_rope"):
                    tables[k] = tuple(t[:, lo:hi].contiguous() for t in tables[k])
            return [self._dev(t) for k in ROPE_KEYS for t in tables[k]]

        rope_flat = self._step_cached(
            "rope", (b, num_frames, height, width, audio_n, float(fps)), (vc, ac), build_rope
        )
        top = self.top
        head = self._head or self.head_fn
        chunk = self._chunk or self.chunk_fn
        tail = self._tail or self.tail_fn
        if cps > 1:
            hidden_states = hidden_states[:, lo:hi]
            if video_keyframes_mask is not None:
                video_keyframes_mask = video_keyframes_mask[:, lo:hi]
        kmask = (
            self._dev(video_keyframes_mask, dt)
            if self.cfg.use_keyframes_abs_pos_embedding
            else None
        )
        x, ax = head(
            self._dev(hidden_states, dt),
            self._dev(audio_hidden_states, dt),
            kmask,
            top["proj_in.weight"],
            top["proj_in.bias"],
            top["audio_proj_in.weight"],
            top["audio_proj_in.bias"],
            top.get("keyframes_abs_pos_embedding"),
        )
        tv, ta = self._step_cached(
            "text",
            (),
            (encoder_hidden_states, audio_encoder_hidden_states),
            lambda: (
                self._dev(encoder_hidden_states, dt),
                self._dev(audio_encoder_hidden_states, dt),
            ),
        )
        k = self.blocks_per_graph
        for i in range(0, self.cfg.num_layers, k):
            flat = [t for blk in self.block_weights[i : i + k] for t in blk]
            x, ax = chunk(x, ax, tv, ta, ctx_flat, rope_flat, *flat)
            self._queue_backpressure(x)
        vout, aout = tail(
            x,
            ax,
            self._dev(ctx["embedded_timestep"], dt),
            self._dev(ctx["audio_embedded_timestep"], dt),
            top["scale_shift_table"],
            top["audio_scale_shift_table"],
            top["proj_out.weight"],
            top["proj_out.bias"],
            top["audio_proj_out.weight"],
            top["audio_proj_out.bias"],
        )
        # Return host tensors: the diffusers denoise loop runs its scheduler/guidance math on the
        # pipeline's execution device, which for this port is the CPU (text encoder, VAE, vocoder
        # all CPU) -- and it immediately calls `.float()` on the output, which fails as an in-place
        # dtype cast on a Lite device tensor. Bringing the velocities to host here matches the
        # Cosmos3-Edge facade (its transformer returns host tensors too).
        return vout.cpu(), aout.cpu()

    def _step_cached(self, name: str, key: tuple, host: tuple, build):
        """Device tensors derived from ``host`` (host tensors or None) and ``key`` (scalars),
        rebuilt only when either changes. The host tensors are compared by value against a
        private copy (a few ms), not by identity. ``LTX2_STEP_CACHE=0`` rebuilds every call."""
        if os.environ.get("LTX2_STEP_CACHE", "1") != "1":
            return build()
        host = tuple(None if t is None else t.detach().cpu() for t in host)
        hit = self._step_cache.get(name)
        if hit is not None and hit[0] == key and len(hit[1]) == len(host):
            if all(
                (a is None and b is None)
                or (
                    a is not None
                    and b is not None
                    and a.shape == b.shape
                    and a.dtype == b.dtype
                    and torch.equal(a, b)
                )
                for a, b in zip(hit[1], host, strict=True)
            ):
                return hit[2]
        value = build()
        self._step_cache[name] = (key, tuple(None if t is None else t.clone() for t in host), value)
        return value

    @staticmethod
    def _queue_backpressure(t: torch.Tensor) -> None:
        """Optional per-chunk Neuron-queue drain (a one-element device-to-device copy that waits
        on the source future).

        OFF by default: with ``NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63`` set (recommended), which
        on its own holds the ~N+2 graph executions this ``forward`` fires without the
        "status=7 Execution Queue Full" error (verified 2026-10-03: a full TP=4 T2V run with the
        drain disabled completed clean). Set ``LTX2_BACKPRESSURE=1`` to re-enable the drain as a
        fallback if a future shape/queue-size combination brings the error back (the trick Wan2.2
        uses once per scheduler step)."""
        if os.environ.get("LTX2_BACKPRESSURE") != "1":
            return
        if t.device.type != "cpu":
            src = t.view(-1)[:1]
            torch.empty_like(src).copy_(src)


def first_frame_keyframes_mask(
    batch: int, num_tokens: int, latent_num_frames: int, dtype=torch.float32
):
    """Official LTX-2.5 marks the first causal latent frame (vLLM-Omni ``_first_frame_keyframes_mask``)."""
    per_frame, rem = divmod(num_tokens, latent_num_frames)
    if rem:
        raise ValueError(f"{num_tokens} tokens not divisible by {latent_num_frames} latent frames")
    m = torch.zeros(batch, num_tokens, 1, dtype=dtype)
    m[:, :per_frame] = 1
    return m


__all__ = [
    "LTX2DiTConfig",
    "LTX2HostConditioning",
    "NeuronLTX2Transformer",
    "block_forward",
    "block_weight_specs",
    "first_frame_keyframes_mask",
    "tp_state",
]
