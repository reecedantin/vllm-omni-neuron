# SPDX-License-Identifier: Apache-2.0
"""Z-Image DiT (``ZImageTransformer2DModel``, Z-Image and Z-Image-Turbo) for Neuron.

A static-shape re-implementation of diffusers' ``ZImageTransformer2DModel`` (basic, non-omni
mode). Parameter names are the checkpoint's, so ``transformer/*.safetensors`` load as-is
(optionally TP-sharded over heads / FFN columns).

The reference builds ragged per-item sequences on the fly. Here the host does that bookkeeping
once per request (:func:`prepare_dit_inputs`): patchify, the SEQ_MULTI_OF=32 pad rows (which
DO take part in attention in the reference), 3-axis RoPE tables and a caption bucket. Caption
rows past the reference's 32-multiple are masked out of every key set, which makes a bucketed
caption numerically identical to the reference's per-batch padding. Inside the graph the
unified sequence is ``[image tokens (Sx), caption tokens (Cb)]``; the valid keys form a
contiguous prefix of it.

Execution modes (``Z_IMAGE_BLOCK_SPLIT``):

* ``0``: one graph for the whole forward.
* ``n > 0``: a prologue graph (timestep / patch / caption embedders and both refiners), the 30
  main blocks as ``30 / n`` calls of ONE compiled n-block graph whose weights are graph inputs,
  and an epilogue graph (final layer). Fewer distinct graphs, far less compiler RAM.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    all_reduce,
    apply_rope_interleaved,
    attention,
    layer_norm_noaffine,
    rms_norm,
    shard,
    tp_state,
)

SEQ_MULTI_OF = 32
ADALN_EMBED_DIM = 256
CAP_BUCKETS = tuple(int(b) for b in os.environ.get("Z_IMAGE_CAP_BUCKETS", "128,256,512").split(","))


@dataclass
class ZImageDiTConfig:
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    in_channels: int = 16
    cap_feat_dim: int = 2560
    norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: list = field(default_factory=lambda: [32, 48, 48])
    axes_lens: list = field(default_factory=lambda: [1536, 512, 512])
    patch_size: int = 2
    f_patch_size: int = 1

    @classmethod
    def from_dict(cls, cfg: dict) -> ZImageDiTConfig:
        if cfg.get("siglip_feat_dim"):
            raise NotImplementedError(
                "Z-Image Omni (SigLIP conditioning) is not supported on Neuron yet"
            )
        kw = {
            k: cfg[k]
            for k in (
                "dim",
                "n_layers",
                "n_refiner_layers",
                "n_heads",
                "n_kv_heads",
                "in_channels",
                "cap_feat_dim",
                "norm_eps",
                "qk_norm",
                "rope_theta",
                "t_scale",
                "axes_dims",
                "axes_lens",
            )
            if k in cfg
        }
        kw["patch_size"] = int(cfg.get("all_patch_size", [2])[0])
        kw["f_patch_size"] = int(cfg.get("all_f_patch_size", [1])[0])
        return cls(**kw)

    @classmethod
    def from_model_dir(cls, model_path: str, subfolder: str = "transformer") -> ZImageDiTConfig:
        with open(os.path.join(model_path, subfolder, "config.json")) as f:
            return cls.from_dict(json.load(f))

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def tp_heads(self, tp: int) -> int:
        """Attention heads after padding to a multiple of ``tp`` (30 -> 32 at TP 4 or 8). The extra
        heads have zero Q/K/V rows and zero output-projection columns, so they contribute exactly 0."""
        return -(-self.n_heads // tp) * tp

    @property
    def ffn_dim(self) -> int:
        return int(self.dim / 3 * 8)

    @property
    def patch_dim(self) -> int:
        return self.f_patch_size * self.patch_size * self.patch_size * self.in_channels

    @property
    def key(self) -> str:
        return f"{self.patch_size}-{self.f_patch_size}"


def _p(*shape, dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class Lin(nn.Module):
    def __init__(self, out_f: int, in_f: int, bias: bool, dtype: torch.dtype):
        super().__init__()
        self.weight = _p(out_f, in_f, dtype=dtype)
        self.bias = _p(out_f, dtype=dtype) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Norm(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.weight = _p(dim, dtype=dtype)


class _Attn(nn.Module):
    def __init__(self, cfg: ZImageDiTConfig, tp: int, dtype):
        super().__init__()
        inner = cfg.tp_heads(tp) * cfg.head_dim // tp
        self.to_q = Lin(inner, cfg.dim, False, dtype)
        self.to_k = Lin(inner, cfg.dim, False, dtype)
        self.to_v = Lin(inner, cfg.dim, False, dtype)
        self.to_out = nn.ModuleList([Lin(cfg.dim, inner, False, dtype)])
        self.norm_q = Norm(cfg.head_dim, dtype)
        self.norm_k = Norm(cfg.head_dim, dtype)


class _FFN(nn.Module):
    def __init__(self, cfg: ZImageDiTConfig, tp: int, dtype):
        super().__init__()
        h = cfg.ffn_dim // tp
        self.w1 = Lin(h, cfg.dim, False, dtype)
        self.w2 = Lin(cfg.dim, h, False, dtype)
        self.w3 = Lin(h, cfg.dim, False, dtype)


class _GroupRef:
    """Holds a process group on a block. ``BlockGraphRunner`` deep-copies a block as its graph template,
    and a ``ProcessGroup`` cannot be copied: the copy shares the same group instead."""

    def __init__(self, group):
        self.group = group

    def __deepcopy__(self, memo):
        return self


class ZBlock(nn.Module):
    """One Z-Image transformer block. For a modulation block the AdaLN projection is done on the
    HOST (``precompute_mod``) and the resulting [B, 4, dim] scales/gates are passed in as ``mod`` —
    so the projection weights never go to the device and every block is one identical graph for
    :class:`~vllm_omni_neuron.diffusion.layers.block_graphs.BlockGraphRunner`."""

    def __init__(self, cfg: ZImageDiTConfig, tp: int, dtype, modulation: bool):
        super().__init__()
        self.cfg = cfg
        self.tp_size, group = tp if isinstance(tp, tuple) else (tp, None)
        self._tp = _GroupRef(group)
        self.n_heads = cfg.tp_heads(self.tp_size) // self.tp_size
        self.head_dim = cfg.head_dim
        self.eps = cfg.norm_eps
        self.attention = _Attn(cfg, self.tp_size, dtype)
        self.feed_forward = _FFN(cfg, self.tp_size, dtype)
        self.attention_norm1 = Norm(cfg.dim, dtype)
        self.attention_norm2 = Norm(cfg.dim, dtype)
        self.ffn_norm1 = Norm(cfg.dim, dtype)
        self.ffn_norm2 = Norm(cfg.dim, dtype)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.ModuleList(
                [Lin(4 * cfg.dim, min(cfg.dim, ADALN_EMBED_DIM), True, dtype)]
            )

    def precompute_mod(self, adaln: torch.Tensor | None) -> torch.Tensor | None:
        """AdaLN projection -> [B, 4, dim] (1+scale_msa, gate_msa.tanh, 1+scale_mlp, gate_mlp.tanh).

        Plain tensor math, usable either traced (inside the compiled prologue, for the 2+2 refiner
        blocks) or eager (for the 30 main blocks, via :meth:`host_mod` below, which is what needs to
        run off-device: see its docstring)."""
        if not self.modulation or adaln is None:
            return None
        mod = self.adaLN_modulation[0](adaln).view(adaln.shape[0], 4, self.cfg.dim)
        s_msa, g_msa, s_mlp, g_mlp = mod.unbind(1)  # each a non-contiguous view/slice of `mod`
        return torch.stack(
            [1.0 + s_msa, g_msa.tanh(), 1.0 + s_mlp, g_mlp.tanh()], dim=1
        ).contiguous()

    def host_mod(self, adaln_cpu: torch.Tensor) -> torch.Tensor:
        """EAGER, HOST-ONLY AdaLN projection for the main-block loop (:meth:`NeuronZImageDiT.main_blocks`).

        Running :meth:`precompute_mod` eagerly on a device tensor (outside any compiled graph) hit a
        Neuron-eager ``torch.stack``/``.contiguous()`` failure the FIRST time this port was wired into
        the served (``Omni``) path -- the standalone ``device_check.py`` harness never exercises eager
        per-block calls this way, which is why it did not catch it. Moving the tiny [B,256] input and
        this block's own modulation weights to CPU first (and the result back after, in
        ``main_blocks``) avoids running Neuron eager math outside torch.compile, and matches the
        design intent: the projection weights never need to be chip-resident."""
        lin = self.adaLN_modulation[0]
        mod = torch.nn.functional.linear(
            adaln_cpu, lin.weight.detach().to("cpu"), lin.bias.detach().to("cpu")
        )
        mod = mod.view(adaln_cpu.shape[0], 4, self.cfg.dim)
        s_msa, g_msa, s_mlp, g_mlp = mod.unbind(1)
        return torch.stack(
            [1.0 + s_msa, g_msa.tanh(), 1.0 + s_mlp, g_mlp.tanh()], dim=1
        ).contiguous()

    def forward(self, x, cos, sin, bias, mod):
        a, f = self.attention, self.feed_forward
        b, s, _ = x.shape
        if mod is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (mod[:, i].unsqueeze(1) for i in range(4))
            h = rms_norm(x, self.attention_norm1.weight, self.eps) * scale_msa
        else:
            h = rms_norm(x, self.attention_norm1.weight, self.eps)
        q = rms_norm(
            F.linear(h, a.to_q.weight).view(b, s, self.n_heads, self.head_dim),
            a.norm_q.weight,
            1e-5,
        )
        k = rms_norm(
            F.linear(h, a.to_k.weight).view(b, s, self.n_heads, self.head_dim),
            a.norm_k.weight,
            1e-5,
        )
        v = F.linear(h, a.to_v.weight).view(b, s, self.n_heads, self.head_dim)
        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)
        o = attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            1.0 / math.sqrt(self.head_dim),
            bias=bias,
        )
        o = F.linear(
            o.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim), a.to_out[0].weight
        )
        o = all_reduce(o, self.tp_size, self._tp.group)
        if mod is not None:
            x = x + gate_msa * rms_norm(o, self.attention_norm2.weight, self.eps)
            h = rms_norm(x, self.ffn_norm1.weight, self.eps) * scale_mlp
        else:
            x = x + rms_norm(o, self.attention_norm2.weight, self.eps)
            h = rms_norm(x, self.ffn_norm1.weight, self.eps)
        ff = F.linear(F.silu(F.linear(h, f.w1.weight)) * F.linear(h, f.w3.weight), f.w2.weight)
        ff = all_reduce(ff, self.tp_size, self._tp.group)
        if mod is not None:
            x = x + gate_mlp * rms_norm(ff, self.ffn_norm2.weight, self.eps)
        else:
            x = x + rms_norm(ff, self.ffn_norm2.weight, self.eps)
        return x


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _TEmb(nn.Module):
    def __init__(self, cfg: ZImageDiTConfig, dtype):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Lin(1024, 256, True, dtype),
                nn.SiLU(),
                Lin(min(cfg.dim, ADALN_EMBED_DIM), 1024, True, dtype),
            ]
        )


class _Final(nn.Module):
    def __init__(self, cfg: ZImageDiTConfig, dtype):
        super().__init__()
        self.linear = Lin(cfg.patch_dim, cfg.dim, True, dtype)
        self.adaLN_modulation = nn.ModuleList(
            [nn.SiLU(), Lin(cfg.dim, min(cfg.dim, ADALN_EMBED_DIM), True, dtype)]
        )


class NeuronZImageDiT(nn.Module):
    """Static-shape Z-Image DiT. See module docstring for the input contract."""

    def __init__(
        self, cfg: ZImageDiTConfig, dtype: torch.dtype = torch.bfloat16, tp: tuple | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.tp_size, self.tp_rank, self.tp_group = tp if tp is not None else tp_state()
        if (
            cfg.ffn_dim % self.tp_size
        ):  # heads are zero-padded to a multiple of TP (ZImageDiTConfig.tp_heads)
            raise ValueError(f"TP={self.tp_size} must divide ffn_dim={cfg.ffn_dim}")
        # Blocks get the TP group itself: with CFG-parallel the world is larger than the TP group, and an
        # all-reduce over the default (world) group would sum the two guidance branches together.
        tpn = (self.tp_size, self.tp_group)
        self.all_x_embedder = nn.ModuleDict({cfg.key: Lin(cfg.dim, cfg.patch_dim, True, dtype)})
        self.all_final_layer = nn.ModuleDict({cfg.key: _Final(cfg, dtype)})
        self.noise_refiner = nn.ModuleList(
            [ZBlock(cfg, tpn, dtype, True) for _ in range(cfg.n_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [ZBlock(cfg, tpn, dtype, False) for _ in range(cfg.n_refiner_layers)]
        )
        self.layers = nn.ModuleList([ZBlock(cfg, tpn, dtype, True) for _ in range(cfg.n_layers)])
        self.t_embedder = _TEmb(cfg, dtype)
        self.cap_embedder = nn.ModuleList(
            [Norm(cfg.cap_feat_dim, dtype), Lin(cfg.dim, cfg.cap_feat_dim, True, dtype)]
        )
        self.x_pad_token = _p(1, cfg.dim, dtype=dtype)
        self.cap_pad_token = _p(1, cfg.dim, dtype=dtype)
        self._local_heads = cfg.tp_heads(self.tp_size) // self.tp_size

    # -- weights ---------------------------------------------------------------------------------
    _COL = (
        "attention.to_q.weight",
        "attention.to_k.weight",
        "attention.to_v.weight",
        "feed_forward.w1.weight",
        "feed_forward.w3.weight",
    )
    _ROW = ("attention.to_out.0.weight", "feed_forward.w2.weight")
    _QKV = ("attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight")

    def _pad_heads(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """Zero-pad the head axis of Q/K/V (rows) and of the output projection (columns) up to
        ``cfg.tp_heads(tp)`` heads. A padded head has q = k = 0 and v = 0, and its output columns are
        0, so the block output is unchanged."""
        extra = (self.cfg.tp_heads(self.tp_size) - self.cfg.n_heads) * self.cfg.head_dim
        if not extra:
            return t
        if key.endswith(self._QKV):
            return torch.cat([t, t.new_zeros(extra, t.shape[1])], dim=0)
        if key.endswith("attention.to_out.0.weight"):
            return torch.cat([t, t.new_zeros(t.shape[0], extra)], dim=1)
        return t

    def load_weights(self, model_path: str, subfolder: str = "transformer") -> None:
        from safetensors import safe_open

        folder = os.path.join(model_path, subfolder)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".safetensors"))
        params = dict(self.named_parameters())
        seen = set()
        for fn in files:
            with safe_open(os.path.join(folder, fn), framework="pt") as f:
                for key in f.keys():
                    if key not in params:
                        raise KeyError(f"unexpected checkpoint key {key}")
                    t = self._pad_heads(key, f.get_tensor(key))
                    if key.endswith(self._COL):
                        t = shard(t, 0, self.tp_size, self.tp_rank)
                    elif key.endswith(self._ROW):
                        t = shard(t, 1, self.tp_size, self.tp_rank)
                    p = params[key]
                    if tuple(t.shape) != tuple(p.shape):
                        raise ValueError(
                            f"{key}: checkpoint {tuple(t.shape)} vs model {tuple(p.shape)}"
                        )
                    p.data = t.to(self.dtype).contiguous()
                    seen.add(key)
        missing = set(params) - seen
        if missing:
            raise KeyError(f"missing checkpoint keys: {sorted(missing)[:8]} ... ({len(missing)})")

    # -- graph pieces ----------------------------------------------------------------------------
    def setup_runner(self, group_size: int, compile_fn=None) -> None:
        """Build the shared N-block graph runner for the 30 main ``layers`` (uniform blocks).

        ``per_layer`` carries each block's host-precomputed modulation [B, 4, dim]; ``cos/sin/bias``
        are shared across blocks. The modulation projection runs on the host (``precompute_mod``), so
        its weights never enter the graph and every chunk is one identical graph.
        """
        from vllm_omni_neuron.diffusion.layers.block_graphs import BlockGraphRunner

        self._runner = BlockGraphRunner(
            self.layers,
            group_size,
            compile_fn=compile_fn,
            block_call=lambda blk, carry, shared, layer, kw: blk(carry, *shared, layer[0]),
        )

    def _refiner(self, blocks, x, cos, sin, bias, adaln):
        for blk in blocks:
            x = blk(x, cos, sin, bias, blk.precompute_mod(adaln))
        return x

    def prologue(self, x_tok, x_pad, x_cos, x_sin, cap, cap_pad, cap_cos, cap_sin, cap_bias, t):
        """-> (unified hidden [B, Sx+Cb, dim], adaln [B, 256])."""
        c = self.cfg
        te = self.t_embedder.mlp
        tf = timestep_embedding(t * c.t_scale, 256).to(self.dtype)
        adaln = te[2](F.silu(te[0](tf)))
        x = self.all_x_embedder[c.key](x_tok)
        x = torch.where(x_pad > 0.5, self.x_pad_token.to(x.dtype), x)
        x = self._refiner(self.noise_refiner, x, x_cos, x_sin, None, adaln)
        y = self.cap_embedder[1](rms_norm(cap, self.cap_embedder[0].weight, c.norm_eps))
        y = torch.where(cap_pad > 0.5, self.cap_pad_token.to(y.dtype), y)
        y = self._refiner(self.context_refiner, y, cap_cos, cap_sin, cap_bias, None)
        return torch.cat([x, y], dim=1), adaln

    def epilogue(self, u, adaln):
        fl = self.all_final_layer[self.cfg.key]
        scale = 1.0 + fl.adaLN_modulation[1](F.silu(adaln))
        return fl.linear(layer_norm_noaffine(u, 1e-6) * scale.unsqueeze(1))

    def main_blocks(self, u, u_cos, u_sin, u_bias, adaln):
        """The 30 main blocks via the shared runner (or a plain loop if no runner was built).

        Modulation is computed on the HOST (:meth:`ZBlock.host_mod`), eagerly, outside any compiled
        graph -- see that method's docstring for why. Results are moved back to ``u``'s device/dtype
        before entering the compiled runner, which is the only thing that touches the NeuronCore here.
        """
        adaln_cpu = adaln.detach().to("cpu")
        dev, dt = u.device, u.dtype
        mods = [
            blk.host_mod(adaln_cpu).to(device=dev, dtype=dt) if blk.modulation else None
            for blk in self.layers
        ]
        if getattr(self, "_runner", None) is not None:
            return self._runner(u, u_cos, u_sin, u_bias, per_layer=[(m,) for m in mods])
        for blk, m in zip(self.layers, mods):
            u = blk(u, u_cos, u_sin, u_bias, m)
        return u

    def forward(
        self,
        x_tok,
        x_pad,
        x_cos,
        x_sin,
        cap,
        cap_pad,
        cap_cos,
        cap_sin,
        cap_bias,
        t,
        u_cos,
        u_sin,
        u_bias,
    ):
        u, adaln = self.prologue(
            x_tok, x_pad, x_cos, x_sin, cap, cap_pad, cap_cos, cap_sin, cap_bias, t
        )
        u = self.main_blocks(u, u_cos, u_sin, u_bias, adaln)
        return self.epilogue(u, adaln)


# ------------------------------------------------------------------------------------------------
# host-side input preparation (the reference's patchify_and_embed / _prepare_sequence bookkeeping)


class RopeTables:
    def __init__(self, cfg: ZImageDiTConfig):
        self.cfg = cfg
        self.cos, self.sin = [], []
        for d, e in zip(cfg.axes_dims, cfg.axes_lens):
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            ang = torch.outer(torch.arange(e, dtype=torch.float64), freqs).float()
            self.cos.append(torch.cos(ang))
            self.sin.append(torch.sin(ang))

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``ids`` [N, 3] long -> (cos, sin) [N, head_dim/2] fp32."""
        cs = [self.cos[i][ids[:, i]] for i in range(3)]
        ss = [self.sin[i][ids[:, i]] for i in range(3)]
        return torch.cat(cs, -1), torch.cat(ss, -1)


def _grid(f, h, w, start):
    a = [torch.arange(s, s + n) for s, n in zip(start, (f, h, w))]
    return torch.stack(torch.meshgrid(*a, indexing="ij"), -1).reshape(-1, 3)


def pick_cap_bucket(n: int) -> int:
    for b in CAP_BUCKETS:
        if n <= b:
            return b
    raise ValueError(
        f"caption is {n} tokens; the largest caption bucket is {CAP_BUCKETS[-1]} (Z_IMAGE_CAP_BUCKETS)"
    )


def prepare_dit_inputs(
    cfg: ZImageDiTConfig,
    rope: RopeTables,
    images: list[torch.Tensor],
    caps: list[torch.Tensor],
    dtype: torch.dtype,
    cap_bucket: int | None = None,
) -> dict:
    """Host bookkeeping for one DiT call. ``images``: per item [C, F, H, W]; ``caps``: per item [L, D]."""
    p, pf = cfg.patch_size, cfg.f_patch_size
    sizes = {tuple(im.shape) for im in images}
    if len(sizes) != 1:
        raise ValueError(f"all images of a call must share one shape, got {sizes}")
    c, fr, hh, ww = images[0].shape
    ft, ht, wt = fr // pf, hh // p, ww // p
    img_len = ft * ht * wt
    sx = img_len + (-img_len) % SEQ_MULTI_OF
    lps = [len(cf) + (-len(cf)) % SEQ_MULTI_OF for cf in caps]
    cb = cap_bucket or pick_cap_bucket(max(lps))
    if max(lps) > cb:
        raise ValueError(f"caption needs {max(lps)} rows > bucket {cb}")
    b = len(images)
    x_tok = torch.zeros(b, sx, cfg.patch_dim)
    x_pad = torch.zeros(b, sx, 1)
    x_pad[:, img_len:] = 1
    cap = torch.zeros(b, cb, cfg.cap_feat_dim)
    cap_pad = torch.ones(b, cb, 1)
    cap_bias = torch.full((b, 1, 1, cb), 0.0)
    hd2 = cfg.head_dim // 2
    x_cos, x_sin = torch.zeros(b, sx, hd2), torch.zeros(b, sx, hd2)
    c_cos, c_sin = torch.zeros(b, cb, hd2), torch.zeros(b, cb, hd2)
    for i, (im, cf) in enumerate(zip(images, caps)):
        tok = (
            im.float()
            .reshape(c, ft, pf, ht, p, wt, p)
            .permute(1, 3, 5, 2, 4, 6, 0)
            .reshape(img_len, -1)
        )
        x_tok[i, :img_len] = tok
        x_tok[i, img_len:] = tok[-1:]
        L, lp = len(cf), lps[i]
        cap[i, :L] = cf.float()
        cap[i, L:] = cf[-1:].float()
        cap_pad[i, :L] = 0
        cap_bias[i, ..., lp:] = -30000.0
        cc, cs = rope(_grid(lp, 1, 1, (1, 0, 0)))
        c_cos[i, :lp], c_sin[i, :lp] = cc, cs
        xc, xs = rope(_grid(ft, ht, wt, (lp + 1, 0, 0)))
        x_cos[i, :img_len], x_sin[i, :img_len] = xc, xs
        zc, zs = rope(torch.zeros(1, 3, dtype=torch.long))
        x_cos[i, img_len:], x_sin[i, img_len:] = zc, zs
    u_cos, u_sin = torch.cat([x_cos, c_cos], 1), torch.cat([x_sin, c_sin], 1)
    u_bias = torch.cat([torch.zeros(b, 1, 1, sx), cap_bias], -1)
    return {
        "args": (
            x_tok.to(dtype),
            x_pad.to(dtype),
            x_cos,
            x_sin,
            cap.to(dtype),
            cap_pad.to(dtype),
            c_cos,
            c_sin,
            cap_bias,
            None,
            u_cos,
            u_sin,
            u_bias,
        ),
        "meta": {"size": (fr, hh, ww), "img_len": img_len, "sx": sx, "cb": cb},
    }


def unpatchify(cfg: ZImageDiTConfig, out: torch.Tensor, meta: dict) -> list[torch.Tensor]:
    """[B, Sx+Cb, patch_dim] -> per item [C, F, H, W]."""
    fr, hh, ww = meta["size"]
    p, pf, c = cfg.patch_size, cfg.f_patch_size, cfg.in_channels
    res = []
    for i in range(out.shape[0]):
        x = out[i, : meta["img_len"]]
        x = (
            x.reshape(fr // pf, hh // p, ww // p, pf, p, p, c)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(c, fr, hh, ww)
        )
        res.append(x)
    return res
