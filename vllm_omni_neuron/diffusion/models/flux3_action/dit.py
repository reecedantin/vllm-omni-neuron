# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action DiT (the 7B joint video + action transformer) for Neuron.

Math is upstream's prepared BF16 inference DiT (``flux_action/models/transformer_inf_bf16.py``)
with the dimensions read from the checkpoint config instead of hard-coded, so the same code runs
the released 7B policies and the random-weight ``tiny`` structure model:

    text phase   (once per caption): vector/time MLPs -> static stream modulations,
                                     txt_in -> ``depth`` text mode blocks
    observation  (once per request): ``depth`` mode blocks over the video_cond / action_cond streams
    step         (once per solver step): timestep embedding -> video/action modulations
    forward      (per denoiser call): emb_in + ``depth`` mode blocks over video / action,
                                      ``depth_single_blocks`` joint blocks over
                                      [txt | video | video_cond | action | action_cond],
                                      final layers of video and action

Every phase is a pure function of tensors with the weights passed in as arguments. That is what
lets one compiled graph serve every block of a group: all joint blocks share shapes, so the
joint-block graph (``JOINT_GROUP`` blocks per call) is compiled once and replayed, which bounds
the compiler's host RAM and the NEFF count.

Tensor parallelism shards heads: Q/K/V and the gated-MLP input are column-parallel, the attention
and MLP output projections row-parallel, and the two partial outputs are summed before ONE
all-reduce per block. Modulations, embedders and final layers are replicated.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .attention import dense_attention

VIDEO = "video"
VIDEO_COND = "video_cond"
TXT = "txt"
LATENT_CHANNELS = 96

# One joint graph for all single-stream blocks by default: the blocks share shapes, so compiling them
# as a single NEFF (not groups) keeps the resident-NEFF count low enough that a 2-branch CFG denoise
# does not overrun the Neuron runtime execution queue (status=7 QUEUE_FULL). Override for debugging.
# One joint graph for all single-stream blocks by default: the blocks share shapes, so compiling them
# as a single NEFF (not groups) keeps the resident-NEFF count and the per-request exec-queue depth low.
# (With NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=63, even JOINT_GROUP=4 runs a 2-branch CFG
# denoise without an explicit queue drain; one graph is still fewer NEFFs and gave the tightest parity.)
JOINT_GROUP = int(os.environ.get("FLUX3_ACTION_JOINT_GROUP", "28"))


@dataclass(frozen=True)
class DiTDims:
    """``JointSingleSeqParams`` fields that shape the inference DiT (upstream defaults = the 7B)."""

    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    context_in_dim: int = 20480
    vec_in_dim: int = 768
    mlp_ratio: float = 3.0
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 10000
    modality: str = "action_prediction_droid"
    action_dim: int = 8
    cond_channels: int = 8
    video_channels: int = LATENT_CHANNELS

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @property
    def action(self) -> str:
        return self.modality

    @property
    def action_cond(self) -> str:
        return f"{self.modality}_cond"

    @property
    def streams(self) -> tuple[str, str, str, str]:
        return (VIDEO, VIDEO_COND, self.action, self.action_cond)

    @classmethod
    def from_policy_config(cls, cfg: dict) -> DiTDims:
        """From a policy package's ``config.native.json`` / ``config.json`` dict."""
        dit = dict(cfg.get("dit_config") or {})
        unsupported = {k for k in dit if k not in _DIT_CONFIG_KEYS}
        if unsupported:
            raise ValueError(
                f"unsupported dit_config keys for the Neuron DiT: {sorted(unsupported)}"
            )
        if dit.get("depth_late_blocks", 0) or dit.get("gate_type") or dit.get("qkv_bias"):
            raise ValueError(
                "late blocks, gated outputs and qkv bias are not used by released FLUX Action DiTs"
            )
        if tuple(cfg.get("content_streams") or ("video", "video_cond")) != ("video", "video_cond"):
            raise ValueError("only the video/video_cond content streams are supported")
        action_dim = int(cfg.get("action_dim", 8))
        past = bool(cfg.get("condition_on_past_actions", False))
        kw = {
            k: dit[k]
            for k in (
                "hidden_size",
                "num_heads",
                "depth",
                "depth_single_blocks",
                "context_in_dim",
                "vec_in_dim",
                "mlp_ratio",
                "theta",
            )
            if k in dit
        }
        if "axes_dim" in dit:
            kw["axes_dim"] = tuple(dit["axes_dim"])
        return cls(
            modality=str(cfg.get("action_modality", "action")),
            action_dim=action_dim,
            cond_channels=action_dim * (2 if past else 1),
            **kw,
        )


_DIT_CONFIG_KEYS = {
    "in_channels",
    "sequence",
    "vec_in_dim",
    "context_in_dim",
    "hidden_size",
    "num_heads",
    "depth",
    "depth_single_blocks",
    "depth_late_blocks",
    "axes_dim",
    "theta",
    "mlp_ratio",
    "qkv_bias",
    "gate_type",
    "attn_mode",
}


# --------------------------------------------------------------------------------------------
# Host-side position tables (fp32, as upstream: float64 angles cast to fp32)
# --------------------------------------------------------------------------------------------
def rope_table(ids: Tensor, axes_dim: Sequence[int], theta: int) -> Tensor:
    """Position ids ``[B, L, 4]`` -> rotation table ``[B, 1, L, head_dim / 2, 2, 2]`` fp32."""
    parts = []
    for axis, dim in enumerate(axes_dim):
        pos = ids[..., axis].to(torch.float64)
        scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        omega = 1.0 / (theta**scale)
        ang = torch.einsum("...n,d->...nd", pos, omega)
        m = torch.stack((torch.cos(ang), -torch.sin(ang), torch.sin(ang), torch.cos(ang)), dim=-1)
        parts.append(m.reshape(*m.shape[:-1], 2, 2).float())
    return torch.cat(parts, dim=-3).unsqueeze(1).contiguous()


# --------------------------------------------------------------------------------------------
# Pure building blocks (weights as arguments)
# --------------------------------------------------------------------------------------------
def _layer_norm(x: Tensor) -> Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)


def _rms(x: Tensor, scale: Tensor) -> Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)).to(x.dtype) * scale


def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = rope[..., 0] * pairs[..., 0] + rope[..., 1] * pairs[..., 1]
    return out.reshape(x.shape).to(x.dtype)


def _modulation(vec: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    out = F.linear(F.silu(vec), w)
    if out.ndim == 2:
        out = out[:, None, :]
    return out.chunk(3, dim=-1)


def _timestep_embedding(t: Tensor, dim: int = 256) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = (t * 1000.0)[:, None].float() * freqs[None]
    return torch.cat((torch.cos(ang), torch.sin(ang)), dim=-1)


def _mlp_embed(x: Tensor, w_in: Tensor, w_out: Tensor) -> Tensor:
    return F.linear(F.silu(F.linear(x, w_in)), w_out)


@dataclass(frozen=True)
class _BlockShape:
    heads: int  # local (per TP rank)
    head_dim: int
    mlp: int  # local gated-MLP hidden


def _block_core(
    modulated: Tensor, rope: Tensor, w_in, w_ao, w_mo, qn, kn, bs: _BlockShape, tp_group
) -> Tensor:
    b, length, _ = modulated.shape
    loc = bs.heads * bs.head_dim
    q, k, v, mlp = F.linear(modulated, w_in).split((loc, loc, loc, 2 * bs.mlp), dim=-1)
    q = q.reshape(b, length, bs.heads, bs.head_dim).transpose(1, 2)
    k = k.reshape(b, length, bs.heads, bs.head_dim).transpose(1, 2)
    v = v.reshape(b, length, bs.heads, bs.head_dim).transpose(1, 2)
    q, k = _rms(q, qn).to(v.dtype), _rms(k, kn).to(v.dtype)
    q, k = _apply_rope(q, rope), _apply_rope(k, rope)
    gate, value = mlp.chunk(2, dim=-1)
    out = F.linear(dense_attention(q, k, v), w_ao) + F.linear(F.silu(gate) * value, w_mo)
    if tp_group is not None:
        dist.all_reduce(out, group=tp_group)
    return out


def mode_blocks(
    x: Tensor,
    rope: Tensor,
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
    weights: Sequence[Tensor],
    bs: _BlockShape,
    tp_group,
) -> Tensor:
    """``len(weights) / 5`` mode blocks of one stream, all modulated by the same (shift, scale, gate)."""
    for i in range(0, len(weights), 5):
        w_in, w_ao, w_mo, qn, kn = weights[i : i + 5]
        out = _block_core(
            (1 + scale) * _layer_norm(x) + shift, rope, w_in, w_ao, w_mo, qn, kn, bs, tp_group
        )
        x = x + gate * out
    return x


def joint_blocks(
    seq: Tensor,
    rope: Tensor,
    lengths: tuple[int, ...],
    mods: Sequence[Tensor],
    weights: Sequence[Tensor],
    bs: _BlockShape,
    tp_group,
) -> Tensor:
    """Joint blocks over the concatenated segments; ``mods`` = 3 tensors (shift, scale, gate) per segment."""
    n_seg = len(lengths)
    for i in range(0, len(weights), 5):
        w_in, w_ao, w_mo, qn, kn = weights[i : i + 5]
        normed = torch.split(_layer_norm(seq), lengths, dim=1)
        modulated = torch.cat(
            [(1 + mods[3 * j + 1]) * normed[j] + mods[3 * j] for j in range(n_seg)], dim=1
        )
        out = _block_core(modulated, rope, w_in, w_ao, w_mo, qn, kn, bs, tp_group)
        outs = torch.split(out, lengths, dim=1)
        seq = seq + torch.cat([mods[3 * j + 2] * outs[j] for j in range(n_seg)], dim=1)
    return seq


# --------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------
_BLOCK_PARTS = ("w_in", "w_ao", "w_mo", "qn", "kn")


@dataclass
class _Prepared:
    """Per-request device state (one CFG branch)."""

    txt_len: int
    txt: Tensor
    video_cond: Tensor
    action_cond: Tensor
    rope: Tensor  # joint rope [txt | video | video_cond | action | action_cond]
    video_rope: Tensor
    action_rope: Tensor
    single_txt: tuple[Tensor, Tensor, Tensor]
    single_video_cond: tuple[Tensor, Tensor, Tensor]
    single_action_cond: tuple[Tensor, Tensor, Tensor]
    vector_embedding: Tensor
    video_len: int = 0
    action_len: int = 0


@dataclass
class _Step:
    early_video: tuple[Tensor, Tensor, Tensor]
    early_action: tuple[Tensor, Tensor, Tensor]
    single_video: tuple[Tensor, Tensor, Tensor]
    single_action: tuple[Tensor, Tensor, Tensor]
    final: tuple[
        Tensor, Tensor, Tensor, Tensor
    ]  # video shift, video scale(+1), action shift, action scale(+1)


@dataclass
class Flux3ActionDiT:
    """Weights (this TP rank's shard) + the compiled phase graphs."""

    dims: DiTDims
    dtype: torch.dtype = torch.bfloat16
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: object = None
    w: dict[str, Tensor] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _fns: dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        d = self.dims
        if d.num_heads % self.tp_size or d.mlp_hidden % self.tp_size:
            raise ValueError(
                f"tp_size={self.tp_size} must divide heads={d.num_heads} and mlp={d.mlp_hidden}"
            )
        self.bs = _BlockShape(d.num_heads // self.tp_size, d.head_dim, d.mlp_hidden // self.tp_size)
        if self.tp_size == 1:
            self.tp_group = None
        elif os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
            # Register the TP group's full partition before the first collective (the joint-block
            # all_reduce), or the compiler legalizes it against only this rank's singleton group and
            # fails "replica id #N not seen in replica groups" (onboarding-models.md §1b). A compile-time
            # hint for the Neuron backend only; CPU-mode gloo runs skip it.
            from vllm_omni_neuron.diffusion.distributed.parallel_state import (
                register_replica_groups,
            )

            register_replica_groups(tp_size=self.tp_size, cp_size=1)
        for name in ("text", "obs", "step", "head", "joint", "final"):
            self._fns.setdefault(name, getattr(self, f"_g_{name}"))

    # -- block-name helpers --------------------------------------------------------------
    def _block_names(self) -> list[str]:
        d = self.dims
        names = [f"txt_mode_blocks.{i}" for i in range(d.depth)]
        for s in sorted(d.streams):
            names += [f"content_mode_blocks.{s}.{i}" for i in range(d.depth)]
        names += [f"single_blocks.{i}" for i in range(d.depth_single_blocks)]
        return names

    def _blocks(self, prefixes: Sequence[str]) -> list[Tensor]:
        return [self.w[f"{p}.{part}"] for p in prefixes for part in _BLOCK_PARTS]

    # -- loading -------------------------------------------------------------------------
    def load(self, path: str, *, prefix: str = "dit.", device: torch.device | str = "cpu") -> None:
        """Load this rank's shard from a policy ``model.safetensors`` (``prefix`` + upstream DiT names)."""
        from safetensors import safe_open

        d, tp, r = self.dims, self.tp_size, self.tp_rank
        hid, mlp = d.hidden_size, d.mlp_hidden
        hl, ml = hid // tp, mlp // tp
        w: dict[str, Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            def get(name: str) -> Tensor:
                key = prefix + name
                if key not in keys:
                    raise KeyError(f"{path}: missing {key}")
                return f.get_tensor(key)

            def rows(name: str, spans: Sequence[tuple[int, int]]) -> Tensor:
                sl = f.get_slice(prefix + name)
                return torch.cat([sl[a:b] for a, b in spans], dim=0)

            def cols(name: str, a: int, b: int) -> Tensor:
                return f.get_slice(prefix + name)[:, a:b]

            for blk in self._block_names():
                q = rows(f"{blk}.q_proj.weight", [(r * hl, (r + 1) * hl)])
                k = rows(f"{blk}.k_proj.weight", [(r * hl, (r + 1) * hl)])
                v = rows(f"{blk}.v_proj.weight", [(r * hl, (r + 1) * hl)])
                m = rows(
                    f"{blk}.mlp_in.weight",
                    [(r * ml, (r + 1) * ml), (mlp + r * ml, mlp + (r + 1) * ml)],
                )
                w[f"{blk}.w_in"] = torch.cat((q, k, v, m), dim=0)
                w[f"{blk}.w_ao"] = cols(f"{blk}.attn_out.weight", r * hl, (r + 1) * hl)
                w[f"{blk}.w_mo"] = cols(f"{blk}.mlp_out.weight", r * ml, (r + 1) * ml)
                w[f"{blk}.qn"] = get(f"{blk}.norm.query_norm.scale")
                w[f"{blk}.kn"] = get(f"{blk}.norm.key_norm.scale")
            for s in d.streams:
                w[f"emb_in.{s}"] = get(f"emb_in.{s}.weight")
            for s in (*d.streams, TXT):
                w[f"early_mod.{s}"] = get(f"early_stream_modulations.{s}.lin.weight")
                w[f"single_mod.{s}"] = get(f"single_stream_modulations.{s}.lin.weight")
            for s in (VIDEO, d.action):
                w[f"final.{s}.linear"] = get(f"final_layer.{s}.linear.weight")
                w[f"final.{s}.adaln"] = get(f"final_layer.{s}.adaLN_modulation.1.weight")
            w["txt_in"] = get("txt_in.weight")
            for e in ("time_in", "vector_in"):
                w[f"{e}.in"] = get(f"{e}.in_layer.weight")
                w[f"{e}.out"] = get(f"{e}.out_layer.weight")
        dev = torch.device(device)
        self.w = {k: v.to(self.dtype).contiguous().to(dev) for k, v in w.items()}
        self.device = dev

    def param_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.w.values())

    # -- compile -------------------------------------------------------------------------
    def compile(self, backend: str, options: dict | None = None) -> None:
        base = dict(options or {})
        args = [
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        for name in list(self._fns):
            fn = getattr(self, f"_g_{name}")
            self._fns[name] = torch.compile(
                fn,
                backend=backend,
                fullgraph=True,
                dynamic=False,
                options={**base, "model_name": f"flux3_action_{name}", "compiler_args": list(args)},
            )

    # -- phase graphs (pure tensor functions; weights as arguments) ------------------------
    def _g_text(
        self,
        ctx,
        text_rope,
        vector,
        zero_t,
        w_vin,
        w_vout,
        w_tin,
        w_tout,
        w_txt_in,
        m_early_txt,
        m_early_vc,
        m_early_ac,
        m_single_txt,
        m_single_vc,
        m_single_ac,
        *blocks,
    ):
        vec_emb = _mlp_embed(vector, w_vin, w_vout)
        static = (
            _mlp_embed(_timestep_embedding(zero_t).to(w_tin.dtype), w_tin, w_tout).to(ctx.dtype)
            + vec_emb
        )
        early_txt = _modulation(static, m_early_txt)
        txt = mode_blocks(
            F.linear(ctx, w_txt_in), text_rope, *early_txt, blocks, self.bs, self.tp_group
        )
        return (
            txt,
            *_modulation(static, m_early_vc),
            *_modulation(static, m_early_ac),
            *_modulation(static, m_single_txt),
            *_modulation(static, m_single_vc),
            *_modulation(static, m_single_ac),
            vec_emb,
        )

    def _g_obs(
        self,
        video_cond,
        action_cond,
        vc_rope,
        ac_rope,
        vc_sh,
        vc_sc,
        vc_ga,
        ac_sh,
        ac_sc,
        ac_ga,
        w_emb_vc,
        w_emb_ac,
        *blocks,
    ):
        n = 5 * self.dims.depth
        vc = mode_blocks(
            F.linear(video_cond, w_emb_vc),
            vc_rope,
            vc_sh,
            vc_sc,
            vc_ga,
            blocks[:n],
            self.bs,
            self.tp_group,
        )
        ac = mode_blocks(
            F.linear(action_cond, w_emb_ac),
            ac_rope,
            ac_sh,
            ac_sc,
            ac_ga,
            blocks[n:],
            self.bs,
            self.tp_group,
        )
        return vc, ac

    def _g_step(
        self, vec_emb, t_video, t_action, w_tin, w_tout, m_ev, m_ea, m_sv, m_sa, w_fv, w_fa
    ):
        def vec(t):
            return (
                _mlp_embed(_timestep_embedding(t).to(w_tin.dtype), w_tin, w_tout).to(vec_emb.dtype)
                + vec_emb
            )

        vv, va = vec(t_video), vec(t_action)
        out = [
            *_modulation(vv, m_ev),
            *_modulation(va, m_ea),
            *_modulation(vv, m_sv),
            *_modulation(va, m_sa),
        ]
        for v, wf in ((vv, w_fv), (va, w_fa)):
            act = F.silu(v)
            sh_w, sc_w = wf.chunk(2)
            out += [F.linear(act, sh_w).unsqueeze(1), (F.linear(act, sc_w) + 1).unsqueeze(1)]
        return tuple(out)

    def _g_head(
        self,
        video,
        action,
        video_rope,
        action_rope,
        ev_sh,
        ev_sc,
        ev_ga,
        ea_sh,
        ea_sc,
        ea_ga,
        txt,
        video_cond,
        action_cond,
        w_emb_v,
        w_emb_a,
        *blocks,
    ):
        n = 5 * self.dims.depth
        v = mode_blocks(
            F.linear(video, w_emb_v),
            video_rope,
            ev_sh,
            ev_sc,
            ev_ga,
            blocks[:n],
            self.bs,
            self.tp_group,
        )
        a = mode_blocks(
            F.linear(action, w_emb_a),
            action_rope,
            ea_sh,
            ea_sc,
            ea_ga,
            blocks[n:],
            self.bs,
            self.tp_group,
        )
        return torch.cat((txt, v, video_cond, a, action_cond), dim=1)

    def _g_joint(self, seq, rope, lengths, *rest):
        mods, blocks = rest[:15], rest[15:]
        return joint_blocks(seq, rope, lengths, mods, blocks, self.bs, self.tp_group)

    def _g_final(self, seq, lengths, v_shift, v_scale, a_shift, a_scale, w_lin_v, w_lin_a):
        t, nv, nvc, na, _ = lengths
        vh = _layer_norm(seq[:, t : t + nv]) * v_scale + v_shift
        ah = _layer_norm(seq[:, t + nv + nvc : t + nv + nvc + na]) * a_scale + a_shift
        return F.linear(vh, w_lin_v), F.linear(ah, w_lin_a)

    # -- host API ------------------------------------------------------------------------
    def _dev(self, x: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        x = x.detach()
        if dtype is not None:
            x = x.to(dtype)
        return x.contiguous().to(self.device)

    @torch.no_grad()
    def prepare(
        self,
        ctx: Tensor,
        ctx_ids: Tensor,
        *,
        video_ids: Tensor,
        video_cond: Tensor,
        video_cond_ids: Tensor,
        action_ids: Tensor,
        action_cond: Tensor,
        action_cond_ids: Tensor,
    ) -> _Prepared:
        """Text + observation phases for one CFG branch. Host tensors in, device state out."""
        d, w = self.dims, self.w
        rope = {
            k: rope_table(ids, d.axes_dim, d.theta)
            for k, ids in (
                ("t", ctx_ids),
                ("v", video_ids),
                ("vc", video_cond_ids),
                ("a", action_ids),
                ("ac", action_cond_ids),
            )
        }
        b = ctx.shape[0]
        vector = torch.zeros(b, d.vec_in_dim, dtype=self.dtype)
        zero_t = torch.zeros(b, dtype=torch.float32)
        txt_blocks = self._blocks([f"txt_mode_blocks.{i}" for i in range(d.depth)])
        out = self._fns["text"](
            self._dev(ctx, self.dtype),
            self._dev(rope["t"]),
            self._dev(vector),
            self._dev(zero_t),
            w["vector_in.in"],
            w["vector_in.out"],
            w["time_in.in"],
            w["time_in.out"],
            w["txt_in"],
            w["early_mod.txt"],
            w[f"early_mod.{VIDEO_COND}"],
            w[f"early_mod.{d.action_cond}"],
            w["single_mod.txt"],
            w[f"single_mod.{VIDEO_COND}"],
            w[f"single_mod.{d.action_cond}"],
            *txt_blocks,
        )
        txt, mods, vec_emb = out[0], out[1:16], out[16]
        obs_blocks = self._blocks(
            [f"content_mode_blocks.{VIDEO_COND}.{i}" for i in range(d.depth)]
            + [f"content_mode_blocks.{d.action_cond}.{i}" for i in range(d.depth)]
        )
        vc, ac = self._fns["obs"](
            self._dev(video_cond, self.dtype),
            self._dev(action_cond, self.dtype),
            self._dev(rope["vc"]),
            self._dev(rope["ac"]),
            *mods[0:6],
            w[f"emb_in.{VIDEO_COND}"],
            w[f"emb_in.{d.action_cond}"],
            *obs_blocks,
        )
        joint = torch.cat((rope["t"], rope["v"], rope["vc"], rope["a"], rope["ac"]), dim=2)
        return _Prepared(
            txt_len=ctx.shape[1],
            txt=txt,
            video_cond=vc,
            action_cond=ac,
            rope=self._dev(joint),
            video_rope=self._dev(rope["v"]),
            action_rope=self._dev(rope["a"]),
            single_txt=tuple(mods[6:9]),
            single_video_cond=tuple(mods[9:12]),
            single_action_cond=tuple(mods[12:15]),
            vector_embedding=vec_emb,
            video_len=video_ids.shape[1],
            action_len=action_ids.shape[1],
        )

    @torch.no_grad()
    def prepare_step(self, req: _Prepared, t_video: float, t_action: float) -> _Step:
        d, w = self.dims, self.w
        b = req.vector_embedding.shape[0]
        tv = self._dev(torch.full((b,), float(t_video), dtype=torch.float32))
        ta = self._dev(torch.full((b,), float(t_action), dtype=torch.float32))
        o = self._fns["step"](
            req.vector_embedding,
            tv,
            ta,
            w["time_in.in"],
            w["time_in.out"],
            w[f"early_mod.{VIDEO}"],
            w[f"early_mod.{d.action}"],
            w[f"single_mod.{VIDEO}"],
            w[f"single_mod.{d.action}"],
            w[f"final.{VIDEO}.adaln"],
            w[f"final.{d.action}.adaln"],
        )
        return _Step(tuple(o[0:3]), tuple(o[3:6]), tuple(o[6:9]), tuple(o[9:12]), tuple(o[12:16]))

    @torch.no_grad()
    def forward(
        self, req: _Prepared, step: _Step, video: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One denoiser call: host ``video [B, Nv, 96]`` / ``action [B, Na, D]`` -> host velocities (model dtype)."""
        d, w = self.dims, self.w
        early = self._blocks(
            [f"content_mode_blocks.{VIDEO}.{i}" for i in range(d.depth)]
            + [f"content_mode_blocks.{d.action}.{i}" for i in range(d.depth)]
        )
        seq = self._fns["head"](
            self._dev(video, self.dtype),
            self._dev(action, self.dtype),
            req.video_rope,
            req.action_rope,
            *step.early_video,
            *step.early_action,
            req.txt,
            req.video_cond,
            req.action_cond,
            w[f"emb_in.{VIDEO}"],
            w[f"emb_in.{d.action}"],
            *early,
        )
        lengths = (
            req.txt_len,
            video.shape[1],
            req.video_cond.shape[1],
            action.shape[1],
            req.action_cond.shape[1],
        )
        mods = (
            *req.single_txt,
            *step.single_video,
            *req.single_video_cond,
            *step.single_action,
            *req.single_action_cond,
        )
        n = d.depth_single_blocks
        for start in range(0, n, JOINT_GROUP):
            blocks = self._blocks(
                [f"single_blocks.{i}" for i in range(start, min(start + JOINT_GROUP, n))]
            )
            seq = self._fns["joint"](seq, req.rope, lengths, *mods, *blocks)
        v, a = self._fns["final"](
            seq, lengths, *step.final, w[f"final.{VIDEO}.linear"], w[f"final.{d.action}.linear"]
        )
        return v.to("cpu"), a.to("cpu")


def load_policy_config(policy_dir: str) -> dict:
    name = (
        "config.native.json"
        if os.path.isfile(os.path.join(policy_dir, "config.native.json"))
        else "config.json"
    )
    with open(os.path.join(policy_dir, name)) as f:
        return json.load(f)
