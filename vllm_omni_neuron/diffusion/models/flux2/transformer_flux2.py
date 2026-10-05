# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 DiT (``Flux2Transformer2DModel``) for NeuronCores.

A drop-in for upstream vLLM-Omni's ``Flux2Transformer2DModel`` inside its ``Flux2Pipeline``: same
``forward`` signature, host tensors in and out. The math is diffusers' (``transformer_flux2.py``):

* prologue: timestep + guidance embedding, the three shared modulation projections
  (double-stream img / txt, single-stream) and the final AdaLN, ``x_embedder``, ``context_embedder``;
* 8 double-stream blocks: separate img / txt weights, joint attention over ``[txt | img]``
  with QK-RMSNorm and 4-axis interleaved RoPE, SwiGLU FFN per stream;
* 48 single-stream blocks: one fused ``[q | k | v | gate | up]`` input projection, joint
  attention + SwiGLU in parallel, one fused output projection;
* epilogue: drop the text tokens, AdaLN-continuous, ``proj_out``.

Tensor parallelism shards attention heads and the FFN hidden dimension (Megatron column /
row split). A row-parallel output is followed by one all-reduce; in the double blocks the img
and txt streams share it (concatenated on the sequence axis), so a double block costs two
all-reduces and a single block one. The modulation / embedding weights are replicated.

Context parallelism (the stage's ``ring_degree``) gives each CP rank a contiguous slice of the text tokens
and of the image tokens (with the matching RoPE rows); every attention all-gathers K/V over the CP group
(the joint attention is unmasked, so key order does not matter) and the epilogue gathers the image tokens.
CP splits activations only: each CP rank holds the full TP shard of the weights.

Execution: one compiled graph per stage *kind* -- prologue, a group of ``double_group`` double
blocks, a group of ``single_group`` single blocks, epilogue -- with the block weights passed as
graph inputs, so the 8 double and 48 single blocks reuse the same NEFFs (distinct compiled
graphs drive compile time, compiler RAM and resident NEFF HBM). The RoPE tables are built on
the host from the position ids (diffusers ``Flux2PosEmbed``, fp64) and cached per geometry.
"""

from __future__ import annotations

import json
import os
import time
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from .ops import (
    TIMING,
    LaunchThrottle,
    all_gather_seq,
    all_reduce,
    attention,
    cp_state,
    layer_norm,
    log_info,
    param,
    rms_norm,
    rope_interleaved,
    set_loader,
    tp_state,
)

logger = init_logger(__name__)

TRANSFORMER_COMPILER_ARGS = [
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]

DOUBLE_KEYS = (
    "q",
    "k",
    "v",
    "add_q",
    "add_k",
    "add_v",
    "norm_q",
    "norm_k",
    "norm_added_q",
    "norm_added_k",
    "out",
    "add_out",
    "ff_in",
    "ff_out",
    "ffc_in",
    "ffc_out",
)
SINGLE_KEYS = ("qkv_mlp", "norm_q", "norm_k", "out")


class Flux2Config(SimpleNamespace):
    @classmethod
    def from_dict(cls, cfg: dict) -> Flux2Config:
        heads, d = int(cfg.get("num_attention_heads", 48)), int(cfg.get("attention_head_dim", 128))
        inner = heads * d
        return cls(
            patch_size=int(cfg.get("patch_size", 1)),
            in_channels=int(cfg.get("in_channels", 128)),
            out_channels=int(cfg.get("out_channels") or cfg.get("in_channels", 128)),
            num_layers=int(cfg.get("num_layers", 8)),
            num_single_layers=int(cfg.get("num_single_layers", 48)),
            attention_head_dim=d,
            num_attention_heads=heads,
            joint_attention_dim=int(cfg.get("joint_attention_dim", 15360)),
            timestep_guidance_channels=int(cfg.get("timestep_guidance_channels", 256)),
            mlp_ratio=float(cfg.get("mlp_ratio", 3.0)),
            axes_dims_rope=tuple(cfg.get("axes_dims_rope", (32, 32, 32, 32))),
            rope_theta=int(cfg.get("rope_theta", 2000)),
            eps=float(cfg.get("eps", 1e-6)),
            guidance_embeds=bool(cfg.get("guidance_embeds", True)),
            inner_dim=inner,
            mlp_hidden=int(inner * float(cfg.get("mlp_ratio", 3.0))),
        )

    @classmethod
    def from_model_dir(cls, model_path: str, subfolder: str = "transformer") -> Flux2Config:
        with open(os.path.join(model_path, subfolder, "config.json")) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------------------------
# Stage functions (traced into the compiled graphs; eager on CPU)
# ---------------------------------------------------------------------------------------------


def _timestep_proj(t: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``: ``[cos | sin]``."""
    half = dim // 2
    exponent = (
        -torch.log(torch.tensor(10000.0))
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def prologue(
    cfg, x, ctx, timestep, guidance, w_x, w_ctx, t1, t2, g1, g2, m_img, m_txt, m_single, m_out
):
    """-> ``img [B,Si,D]``, ``txt [B,St,D]``, ``mod_img [B,1,6D]``, ``mod_txt [B,1,6D]``,
    ``mod_single [B,1,3D]``, ``mod_out [B,2D]``."""
    dtype = x.dtype
    t = timestep.to(dtype) * 1000
    temb = F.linear(
        F.silu(F.linear(_timestep_proj(t, cfg.timestep_guidance_channels).to(dtype), t1)), t2
    )
    if g1 is not None:
        g = guidance.to(dtype) * 1000
        temb = temb + F.linear(
            F.silu(F.linear(_timestep_proj(g, cfg.timestep_guidance_channels).to(dtype), g1)), g2
        )
    s = F.silu(temb)
    mod_img = F.linear(s, m_img).unsqueeze(1)
    mod_txt = F.linear(s, m_txt).unsqueeze(1)
    mod_single = F.linear(s, m_single).unsqueeze(1)
    mod_out = F.linear(s, m_out)
    return F.linear(x, w_x), F.linear(ctx, w_ctx), mod_img, mod_txt, mod_single, mod_out


def _heads(x, n, d):
    return x.unflatten(-1, (n, d))


def _joint_attention(q, k, v, d, cp, cpg):
    """``[B, S, H, D]`` -> ``[B, S, H*D]``. Under CP each rank holds a slice of the tokens: K/V are
    all-gathered over the CP group (attention is unmasked, so the gathered key order is irrelevant)."""
    k, v = all_gather_seq(k, cp, cpg), all_gather_seq(v, cp, cpg)
    return (
        attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), d**-0.5)
        .transpose(1, 2)
        .flatten(2)
    )


def double_block(cfg, nh, tp, group, cp, cpg, img, txt, cos, sin, mod_img, mod_txt, *w):
    d, eps = cfg.attention_head_dim, cfg.eps
    (
        q_w,
        k_w,
        v_w,
        aq_w,
        ak_w,
        av_w,
        nq,
        nk,
        naq,
        nak,
        out_w,
        aout_w,
        ff_in,
        ff_out,
        ffc_in,
        ffc_out,
    ) = w
    sh1, sc1, g1, sh2, sc2, g2 = mod_img.chunk(6, dim=-1)
    csh1, csc1, cg1, csh2, csc2, cg2 = mod_txt.chunk(6, dim=-1)
    st = txt.shape[1]

    ni = (1 + sc1) * layer_norm(img, eps) + sh1
    nt = (1 + csc1) * layer_norm(txt, eps) + csh1
    q = rms_norm(_heads(F.linear(ni, q_w), nh, d), nq, eps)
    k = rms_norm(_heads(F.linear(ni, k_w), nh, d), nk, eps)
    v = _heads(F.linear(ni, v_w), nh, d)
    eq = rms_norm(_heads(F.linear(nt, aq_w), nh, d), naq, eps)
    ek = rms_norm(_heads(F.linear(nt, ak_w), nh, d), nak, eps)
    ev = _heads(F.linear(nt, av_w), nh, d)
    q = rope_interleaved(torch.cat([eq, q], dim=1), cos, sin)
    k = rope_interleaved(torch.cat([ek, k], dim=1), cos, sin)
    v = torch.cat([ev, v], dim=1)
    a = _joint_attention(q, k, v, d, cp, cpg)
    o = torch.cat([F.linear(a[:, :st], aout_w), F.linear(a[:, st:], out_w)], dim=1)
    o = all_reduce(o, tp, group)
    txt = txt + cg1 * o[:, :st]
    img = img + g1 * o[:, st:]

    ni = layer_norm(img, eps) * (1 + sc2) + sh2
    nt = layer_norm(txt, eps) * (1 + csc2) + csh2
    hi_g, hi_u = F.linear(ni, ff_in).chunk(2, dim=-1)
    ht_g, ht_u = F.linear(nt, ffc_in).chunk(2, dim=-1)
    f = torch.cat(
        [F.linear(F.silu(ht_g) * ht_u, ffc_out), F.linear(F.silu(hi_g) * hi_u, ff_out)], dim=1
    )
    f = all_reduce(f, tp, group)
    return img + g2 * f[:, st:], txt + cg2 * f[:, :st]


def single_block(cfg, nh, tp, group, cp, cpg, h, cos, sin, mod, *w):
    d, eps = cfg.attention_head_dim, cfg.eps
    qkv_mlp, nq, nk, out_w = w
    sh, sc, g = mod.chunk(3, dim=-1)
    n = (1 + sc) * layer_norm(h, eps) + sh
    inner, mlp = nh * d, (qkv_mlp.shape[0] - 3 * nh * d) // 2
    q, k, v, gate, up = torch.split(F.linear(n, qkv_mlp), [inner, inner, inner, mlp, mlp], dim=-1)
    q = rope_interleaved(rms_norm(_heads(q, nh, d), nq, eps), cos, sin)
    k = rope_interleaved(rms_norm(_heads(k, nh, d), nk, eps), cos, sin)
    v = _heads(v, nh, d)
    a = _joint_attention(q, k, v, d, cp, cpg)
    o = all_reduce(F.linear(torch.cat([a, F.silu(gate) * up], dim=-1), out_w), tp, group)
    return h + g * o


def double_group(cfg, nh, tp, group, cp, cpg, n, img, txt, cos, sin, mod_img, mod_txt, *w):
    per = len(DOUBLE_KEYS)
    for i in range(n):
        img, txt = double_block(
            cfg,
            nh,
            tp,
            group,
            cp,
            cpg,
            img,
            txt,
            cos,
            sin,
            mod_img,
            mod_txt,
            *w[i * per : (i + 1) * per],
        )
    return img, txt


def single_group(cfg, nh, tp, group, cp, cpg, n, h, cos, sin, mod, *w):
    per = len(SINGLE_KEYS)
    for i in range(n):
        h = single_block(cfg, nh, tp, group, cp, cpg, h, cos, sin, mod, *w[i * per : (i + 1) * per])
    return h


def join_streams(img, txt):
    return torch.cat([txt, img], dim=1)


def epilogue(cfg, cp, cpg, h, mod_out, s_txt, proj_out):
    """Drop this rank's text tokens, final AdaLN + ``proj_out``; under CP gather the image tokens."""
    h = h[:, s_txt:]
    scale, shift = mod_out.chunk(2, dim=1)
    h = layer_norm(h, cfg.eps) * (1 + scale)[:, None, :] + shift[:, None, :]
    return all_gather_seq(F.linear(h, proj_out), cp, cpg)


# ---------------------------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------------------------


def _d(p):
    return None if p is None else p.detach()


class _Block(nn.Module):
    pass


def _resolve_group(explicit: int | None, env_var: str, default: int, n: int) -> int:
    """A group size that divides ``n`` blocks: the explicit arg or env value if given (and
    validated), else the largest divisor of ``n`` not exceeding ``default`` -- so a smaller
    checkpoint (M-tiny) need not match the real model's env / launch-queue tuning."""
    requested = explicit if explicit is not None else os.environ.get(env_var)
    if requested is not None:
        requested = int(requested)
        if n % requested:
            raise ValueError(f"{env_var}={requested} must divide the block count {n}")
        return requested
    for g in range(min(default, n), 0, -1):
        if n % g == 0:
            return g
    return 1


class NeuronFlux2Transformer(nn.Module):
    """Neuron FLUX.2 DiT. Construct with ``model_path`` (the pipeline directory)."""

    def __init__(
        self,
        model_path: str | None = None,
        cfg: Flux2Config | None = None,
        dtype: torch.dtype = torch.bfloat16,
        double_group: int | None = None,
        single_group: int | None = None,
        **_unused,
    ):
        super().__init__()
        if cfg is None:
            cfg = Flux2Config.from_model_dir(model_path)
        self.model_path = model_path
        self.cfg = cfg
        self._dtype = dtype
        self.tp_size, self.tp_rank, self.tp_group = tp_state()
        self.cp_size, self.cp_rank, self.cp_group = cp_state()
        tp = self.tp_size
        if cfg.num_attention_heads % tp or cfg.mlp_hidden % tp:
            raise ValueError(
                f"TP={tp} must divide heads={cfg.num_attention_heads} and mlp={cfg.mlp_hidden}"
            )
        self.nh = cfg.num_attention_heads // tp
        self.double_group = _resolve_group(double_group, "FLUX2_DOUBLE_GROUP", 1, cfg.num_layers)
        self.single_group = _resolve_group(
            single_group, "FLUX2_SINGLE_GROUP", 6, cfg.num_single_layers
        )

        D, M, inner, d = cfg.inner_dim, cfg.mlp_hidden, cfg.inner_dim, cfg.attention_head_dim
        Ir, Mr = inner // tp, M // tp
        p = partial(param, dtype=dtype)
        gch = cfg.timestep_guidance_channels
        self.x_embedder = p((D, cfg.in_channels))
        self.context_embedder = p((D, cfg.joint_attention_dim))
        self.t_lin1, self.t_lin2 = p((D, gch)), p((D, D))
        if cfg.guidance_embeds:
            self.g_lin1, self.g_lin2 = p((D, gch)), p((D, D))
        self.mod_img, self.mod_txt = p((6 * D, D)), p((6 * D, D))
        self.mod_single, self.mod_out = p((3 * D, D)), p((2 * D, D))
        self.proj_out = p((cfg.patch_size**2 * cfg.out_channels, D))

        self.double_blocks = nn.ModuleList()
        for _ in range(cfg.num_layers):
            b = _Block()
            for name in ("q", "k", "v", "add_q", "add_k", "add_v"):
                setattr(b, name, p((Ir, D)))
                set_loader(getattr(b, name), 0, [(0, Ir)])
            for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                setattr(b, name, p((d,)))
            for name in ("out", "add_out"):
                setattr(b, name, p((D, Ir)))
                set_loader(getattr(b, name), 1, [(0, Ir)])
            for name in ("ff_in", "ffc_in"):
                setattr(b, name, p((2 * Mr, D)))
                set_loader(getattr(b, name), 0, [(0, Mr), (M, Mr)])
            for name in ("ff_out", "ffc_out"):
                setattr(b, name, p((D, Mr)))
                set_loader(getattr(b, name), 1, [(0, Mr)])
            self.double_blocks.append(b)
        self.single_blocks = nn.ModuleList()
        for _ in range(cfg.num_single_layers):
            b = _Block()
            b.qkv_mlp = p((3 * Ir + 2 * Mr, D))
            set_loader(
                b.qkv_mlp,
                0,
                [(0, Ir), (inner, Ir), (2 * inner, Ir), (3 * inner, Mr), (3 * inner + M, Mr)],
            )
            b.norm_q, b.norm_k = p((d,)), p((d,))
            b.out = p((D, Ir + Mr))
            set_loader(b.out, 1, [(0, Ir), (inner, Mr)])
            self.single_blocks.append(b)

        # upstream-pipeline facing attributes
        self.config = SimpleNamespace(**{k: getattr(cfg, k) for k in vars(cfg)})
        self.guidance_embeds = cfg.guidance_embeds
        self.in_channels = cfg.in_channels
        self._device = torch.device("cpu")
        self._rope_cache: dict = {}
        self.stats = {"calls": 0, "seconds": 0.0}
        self._bind_stage_fns(None)

    # -- upstream attributes ---------------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # -- weights ------------------------------------------------------------------------------
    def checkpoint_mappings(self) -> dict[str, str]:
        m = {
            "x_embedder": "x_embedder.weight",
            "context_embedder": "context_embedder.weight",
            "t_lin1": "time_guidance_embed.timestep_embedder.linear_1.weight",
            "t_lin2": "time_guidance_embed.timestep_embedder.linear_2.weight",
            "mod_img": "double_stream_modulation_img.linear.weight",
            "mod_txt": "double_stream_modulation_txt.linear.weight",
            "mod_single": "single_stream_modulation.linear.weight",
            "mod_out": "norm_out.linear.weight",
            "proj_out": "proj_out.weight",
        }
        if self.cfg.guidance_embeds:
            m["g_lin1"] = "time_guidance_embed.guidance_embedder.linear_1.weight"
            m["g_lin2"] = "time_guidance_embed.guidance_embedder.linear_2.weight"
        dmap = {
            "q": "attn.to_q",
            "k": "attn.to_k",
            "v": "attn.to_v",
            "add_q": "attn.add_q_proj",
            "add_k": "attn.add_k_proj",
            "add_v": "attn.add_v_proj",
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out": "attn.to_out.0",
            "add_out": "attn.to_add_out",
            "ff_in": "ff.linear_in",
            "ff_out": "ff.linear_out",
            "ffc_in": "ff_context.linear_in",
            "ffc_out": "ff_context.linear_out",
        }
        for i in range(self.cfg.num_layers):
            for ours, theirs in dmap.items():
                m[f"double_blocks.{i}.{ours}"] = f"transformer_blocks.{i}.{theirs}.weight"
        smap = {
            "qkv_mlp": "attn.to_qkv_mlp_proj",
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out": "attn.to_out",
        }
        for i in range(self.cfg.num_single_layers):
            for ours, theirs in smap.items():
                m[f"single_blocks.{i}.{ours}"] = f"single_transformer_blocks.{i}.{theirs}.weight"
        return m

    def load_weights(
        self, model_path: str | None = None, device: torch.device | str | None = None
    ) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        model_path = model_path or self.model_path
        device = torch.device(device) if device is not None else self._device
        t0 = time.time()
        res = SafetensorsCheckpoint(os.path.join(model_path, "transformer")).load_sharded_pipelined(
            self.tp_rank, self.tp_size, self, self.checkpoint_mappings(), device
        )
        self.load_state_dict(res.state_dict, strict=True, assign=True)
        self._device = device
        self.load_seconds = time.time() - t0
        self.param_gb = sum(p.numel() * p.element_size() for p in self.parameters()) / 2**30
        log_info(
            "flux2 DiT: loaded tp_rank %d/%d to %s in %.1fs (%.2f GiB weights on this rank)",
            self.tp_rank,
            self.tp_size,
            device,
            self.load_seconds,
            self.param_gb,
        )

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    # -- compile ------------------------------------------------------------------------------
    def _bind_stage_fns(self, compile_fn):
        # Each stage is its own module-level function (own code object, own dynamo cache) with
        # the static config passed as leading constant args. Compiling functools.partial objects
        # instead would funnel every stage through dynamo's one shared wrapper and exhaust its
        # recompile limit.
        cfg, nh, tp, g, cp, cg = (
            self.cfg,
            self.nh,
            self.tp_size,
            self.tp_group,
            self.cp_size,
            self.cp_group,
        )
        fns = {
            "prologue": (prologue, (cfg,)),
            "double": (double_group, (cfg, nh, tp, g, cp, cg, self.double_group)),
            "single": (single_group, (cfg, nh, tp, g, cp, cg, self.single_group)),
            "join": (join_streams, ()),
            "epilogue": (epilogue, (cfg, cp, cg)),
        }
        self._fns = {
            k: partial(compile_fn(k, f) if compile_fn else f, *lead) for k, (f, lead) in fns.items()
        }

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        base = dict(options or {})
        tag = f"tp{self.tp_size}_d{self.double_group}_s{self.single_group}"
        if self.cp_size > 1:
            tag += f"_cp{self.cp_size}"

        def compile_fn(name, fn):
            opts = {
                **base,
                "model_name": f"flux2_dit_{name}_{tag}",
                "compiler_args": list(TRANSFORMER_COMPILER_ARGS),
            }
            return torch.compile(
                fn,
                backend=backend,
                options=opts,
                fullgraph=kwargs.get("fullgraph", True),
                dynamic=False,
            )

        self._bind_stage_fns(compile_fn)

    # -- host helpers -------------------------------------------------------------------------
    def rope_tables(
        self, txt_ids: torch.Tensor, img_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``[S_txt + S_img, head_dim]`` fp32 for ``[txt | img]`` (diffusers ``Flux2PosEmbed``)."""
        from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

        pe = Flux2PosEmbed(theta=self.cfg.rope_theta, axes_dim=list(self.cfg.axes_dims_rope))
        ct, st = pe(txt_ids.cpu())
        ci, si = pe(img_ids.cpu())
        return torch.cat([ct, ci], 0).float().contiguous(), torch.cat(
            [st, si], 0
        ).float().contiguous()

    def _rope_on_device(self, txt_ids, img_ids):
        txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
        img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        key = (
            tuple(txt_ids.shape),
            tuple(img_ids.shape),
            hash(txt_ids.cpu().numpy().tobytes()),
            hash(img_ids.cpu().numpy().tobytes()),
        )
        if key not in self._rope_cache:
            if len(self._rope_cache) > 8:
                self._rope_cache.clear()
            cos, sin = self.rope_tables(txt_ids, img_ids)
            cp, r = self.cp_size, self.cp_rank
            if cp > 1:  # rows of this rank's [txt slice | img slice] (see forward)
                st, si = txt_ids.shape[0], img_ids.shape[0]
                lt, li = st // cp, si // cp
                rows = torch.cat(
                    [torch.arange(r * lt, (r + 1) * lt), st + torch.arange(r * li, (r + 1) * li)]
                )
                cos, sin = cos[rows].contiguous(), sin[rows].contiguous()
            self._rope_cache[key] = (cos.to(self._device), sin.to(self._device))
        return self._rope_cache[key]

    def _block_weights(self, blocks, keys, start, n):
        out = []
        for b in blocks[start : start + n]:
            # plain (non-Parameter) views: the compiled block graph is keyed on shapes only,
            # so every block reuses it instead of guarding on parameter identity
            out.extend(getattr(b, k).detach() for k in keys)
        return out

    # -- forward (upstream signature) ----------------------------------------------------------
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
        **kwargs,
    ):
        t_start = time.time()
        step_dump = _step_dump_path(self.stats["calls"])
        if step_dump:  # teacher-forced step checks: the call's full (pre-CP-slice) inputs
            dumped = {
                k: v.detach().to("cpu").clone()
                for k, v in dict(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                ).items()
                if isinstance(v, torch.Tensor)
            }
        dev, dt, cfg = self._device, self._dtype, self.cfg
        host = lambda x: x.detach().to("cpu").to(dt).contiguous().to(dev)  # noqa: E731
        cp, r = self.cp_size, self.cp_rank
        if (
            cp > 1
        ):  # context parallel: this rank's contiguous slice of the text and of the image tokens
            st, si = encoder_hidden_states.shape[1], hidden_states.shape[1]
            if st % cp or si % cp:
                raise ValueError(
                    f"CP={cp} must divide the text ({st}) and image ({si}) token counts"
                )
            lt, li = st // cp, si // cp
            encoder_hidden_states = encoder_hidden_states[:, r * lt : (r + 1) * lt]
            hidden_states = hidden_states[:, r * li : (r + 1) * li]
        x, ctx = host(hidden_states), host(encoder_hidden_states)
        ts = timestep.detach().to("cpu").float().reshape(-1).contiguous().to(dev)
        gs = None
        if cfg.guidance_embeds:
            if guidance is None:
                raise ValueError("FLUX.2-dev is guidance-distilled: pass `guidance`")
            gs = guidance.detach().to("cpu").float().reshape(-1).contiguous().to(dev)
        cos, sin = self._rope_on_device(txt_ids, img_ids)
        s_txt = ctx.shape[1]

        f = self._fns
        with torch.no_grad():
            img, txt, m_img, m_txt, m_single, m_out = f["prologue"](
                x,
                ctx,
                ts,
                gs,
                *(
                    _d(getattr(self, n, None))
                    for n in (
                        "x_embedder",
                        "context_embedder",
                        "t_lin1",
                        "t_lin2",
                        "g_lin1",
                        "g_lin2",
                        "mod_img",
                        "mod_txt",
                        "mod_single",
                        "mod_out",
                    )
                ),
            )
            throttle = LaunchThrottle(dev)
            for i in range(0, cfg.num_layers, self.double_group):
                img, txt = f["double"](
                    img,
                    txt,
                    cos,
                    sin,
                    m_img,
                    m_txt,
                    *self._block_weights(self.double_blocks, DOUBLE_KEYS, i, self.double_group),
                )
                throttle.tick(f"double {i}")
            h = f["join"](img, txt)
            for i in range(0, cfg.num_single_layers, self.single_group):
                h = f["single"](
                    h,
                    cos,
                    sin,
                    m_single,
                    *self._block_weights(self.single_blocks, SINGLE_KEYS, i, self.single_group),
                )
                throttle.tick(f"single {i}")
            out = f["epilogue"](h, m_out, s_txt, _d(self.proj_out)).to("cpu")
        if step_dump:
            dumped.update(
                output=out.float(), call=self.stats["calls"], tp=self.tp_size, cp=self.cp_size
            )
            torch.save(dumped, step_dump)
        self.stats["calls"] += 1
        self.stats["seconds"] += time.time() - t_start
        if TIMING:
            log_info(
                "timing dit_call %.4fs (S_img=%d S_txt=%d tp=%d cp=%d)",
                time.time() - t_start,
                out.shape[1],
                s_txt * self.cp_size,
                self.tp_size,
                self.cp_size,
            )
        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=out)


def _step_dump_path(call: int) -> str | None:
    """Test plumbing for teacher-forced step checks. ``FLUX2_STEP_DUMP=dir`` makes rank 0 save the
    full inputs and the output of DiT calls ``FLUX2_STEP_DUMP_CALLS`` (default ``0,24,49``; counted
    from process start, so with one request they are the denoising steps) to ``dir/step_NNN.pt``,
    for replay on CPU from the device trajectory."""
    out_dir = os.environ.get("FLUX2_STEP_DUMP")
    if not out_dir:
        return None
    calls = os.environ.get("FLUX2_STEP_DUMP_CALLS", "0,24,49")
    if call not in {int(c) for c in calls.split(",") if c.strip()}:
        return None
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"step_{call:03d}.pt")
