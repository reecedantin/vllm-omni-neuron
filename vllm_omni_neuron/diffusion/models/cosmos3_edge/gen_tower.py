# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Edge GEN (diffusion) tower for Neuron.

Per denoising call the GEN tower maps noisy video latents ``[B, 48, t, h, w]`` (plus an
optional action stream) to a velocity. Each of its 28 layers attends with its own Q/K/V
(QK-norm + mRoPE) over the concatenation ``[UND K/V of the prompt | its own K/V]``; the UND
K/V come from :class:`.und_tower.NeuronCosmos3EdgeUND` and are fixed for the whole request.

This re-implements upstream's GEN path of ``Cosmos3VFMTransformer.forward`` (vendored, T2I /
T2V / I2V / action; no sound or transfer-control streams) in the plugin's raw-parameter style:

* The math order is upstream's: ``patchify -> proj_in -> + timestep embedding (only on noisy
  tokens) -> 28 x GEN layer -> norm_moe_gen -> proj_out -> unpatchify``.
* The text is padded to a fixed bucket so one compiled graph serves every prompt length; the
  padded UND keys are hidden with an additive key bias (upstream instead trims to the real
  length, which would make the graph shape prompt-dependent).
* The action projections are ``DomainAwareLinear`` (one weight table per embodiment domain);
  the requested domain's slice is gathered on the host and passed in, keeping the
  data-dependent lookup out of the graph.
* TP shards heads (Q/K/V column-parallel, output and MLP down row-parallel + all-reduce).
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ._vendor.transformer_cosmos3 import (
    Cosmos3VFMTransformer,
    Qwen3VLTextRotaryEmbedding,
    _apply_rotary_pos_emb,
)
from .attention import edge_attention, key_padding_bias
from .und_tower import EdgeTextConfig, _sharded, _tp_state, rms_norm


class EdgeGenConfig(EdgeTextConfig):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.latent_channel = int(cfg.get("latent_channel", 48))
        self.patch = int(cfg.get("latent_patch_size", 2))
        self.patch_dim = self.patch * self.patch * self.latent_channel
        self.timestep_scale = float(cfg.get("timestep_scale", 0.001))
        self.qk_norm = bool(cfg.get("qk_norm_for_diffusion", True))
        self.action_gen = bool(cfg.get("action_gen", False))
        self.action_dim = int(cfg.get("action_dim", 64))
        self.num_domains = int(cfg.get("num_embodiment_domains", 32))
        self.base_fps = float(cfg.get("base_fps", 24.0))
        self.temporal_compression_factor = int(cfg.get("temporal_compression_factor", 4))
        self.enable_fps_modulation = bool(cfg.get("enable_fps_modulation", True))
        self.temporal_modality_margin = int(cfg.get("unified_3d_mrope_temporal_modality_margin", 15000))
        self.sound_latent_fps = cfg.get("sound_latent_fps", 25)

    @classmethod
    def from_model_dir(cls, model_path: str) -> EdgeGenConfig:
        import json

        with open(os.path.join(model_path, "transformer", "config.json")) as f:
            return cls(json.load(f))


def timestep_embedding_freqs(dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    return torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)


class EdgeGenLayer(nn.Module):
    def __init__(self, cfg: EdgeGenConfig, tp_size: int, dtype: torch.dtype):
        super().__init__()
        h, d = cfg.hidden_size, cfg.head_dim
        self.cfg = cfg
        self.tp_size = tp_size
        self.n_heads = cfg.num_heads // tp_size
        self.n_kv = cfg.num_kv_heads // tp_size
        self.q_weight = _sharded((cfg.num_heads * d, h), 0, tp_size, dtype)
        self.k_weight = _sharded((cfg.num_kv_heads * d, h), 0, tp_size, dtype)
        self.v_weight = _sharded((cfg.num_kv_heads * d, h), 0, tp_size, dtype)
        self.o_weight = _sharded((h, cfg.num_heads * d), 1, tp_size, dtype)
        self.q_norm_weight = nn.Parameter(torch.ones(d, dtype=dtype), requires_grad=False)
        self.k_norm_weight = nn.Parameter(torch.ones(d, dtype=dtype), requires_grad=False)
        self.input_norm_weight = nn.Parameter(torch.ones(h, dtype=dtype), requires_grad=False)
        self.post_norm_weight = nn.Parameter(torch.ones(h, dtype=dtype), requires_grad=False)
        self.up_weight = _sharded((cfg.intermediate_size, h), 0, tp_size, dtype)
        self.down_weight = _sharded((h, cfg.intermediate_size), 1, tp_size, dtype)

    def _all_reduce(self, x, group):
        if self.tp_size > 1:
            dist.all_reduce(x, group=group)
        return x

    def forward(self, x, k_und, v_und, cos, sin, key_bias, group):
        cfg, d = self.cfg, self.cfg.head_dim
        b, s, _ = x.shape
        hn = rms_norm(x, self.input_norm_weight, cfg.rms_norm_eps)
        q = F.linear(hn, self.q_weight).view(b, s, self.n_heads, d)
        k = F.linear(hn, self.k_weight).view(b, s, self.n_kv, d)
        v = F.linear(hn, self.v_weight).view(b, s, self.n_kv, d)
        if cfg.qk_norm:
            q = F.rms_norm(q, (d,), self.q_norm_weight, eps=cfg.rms_norm_eps)
            k = F.rms_norm(k, (d,), self.k_norm_weight, eps=cfg.rms_norm_eps)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        k_all = torch.cat([k_und, k], dim=1).transpose(1, 2)
        v_all = torch.cat([v_und, v], dim=1).transpose(1, 2)
        attn = edge_attention(q.transpose(1, 2), k_all, v_all, d**-0.5, key_bias=key_bias).transpose(1, 2)
        x = x + self._all_reduce(F.linear(attn.reshape(b, s, -1), self.o_weight), group)
        hn = rms_norm(x, self.post_norm_weight, cfg.rms_norm_eps)
        mlp = F.relu(F.linear(hn, self.up_weight))
        return x + self._all_reduce(F.linear(mlp * mlp, self.down_weight), group)


class NeuronCosmos3EdgeGEN(nn.Module):
    """GEN tower. ``forward(latents, timestep, cos, sin, key_bias, noisy_mask, *und_kv)``.

    * ``latents`` ``[B, 48, t, h, w]`` (``h``/``w`` latent pixels, even), ``timestep`` ``[B]`` fp32.
    * ``cos``/``sin`` ``[B, S_gen, 1, head_dim]`` from :meth:`rope_tables`.
    * ``key_bias`` ``[B, 1, 1, S_text + S_gen]`` from :meth:`key_bias` (text bucket padding).
    * ``noisy_mask`` ``[B, S_video, 1]``: 1 for noisy tokens (receive the timestep embedding,
      as upstream), 0 for clean conditioning frames (I2V first frame).
    * ``und_kv`` = ``k_0..k_27, v_0..v_27`` from the UND tower, ``[B, S_text, kv/tp, D]``.

    With an action stream use :meth:`forward_action`, which returns ``(video_v, action_v)``.
    """

    def __init__(self, cfg: EdgeGenConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.tp_size, self.tp_rank, self.tp_group = _tp_state()
        h = cfg.hidden_size
        self.layers = nn.ModuleList(EdgeGenLayer(cfg, self.tp_size, dtype) for _ in range(cfg.num_layers))
        p = lambda *shape, dt=dtype: nn.Parameter(torch.empty(*shape, dtype=dt), requires_grad=False)  # noqa: E731
        self.proj_in_weight, self.proj_in_bias = p(h, cfg.patch_dim), p(h)
        self.proj_out_weight, self.proj_out_bias = p(cfg.patch_dim, h), p(cfg.patch_dim)
        # timestep embedder in fp32 (upstream post_load_weights casts it)
        self.t_lin1_weight, self.t_lin1_bias = p(h, 256, dt=torch.float32), p(h, dt=torch.float32)
        self.t_lin2_weight, self.t_lin2_bias = p(h, h, dt=torch.float32), p(h, dt=torch.float32)
        self.register_buffer("t_freqs", timestep_embedding_freqs(), persistent=False)
        self.norm_out_weight = p(h)
        if cfg.action_gen:
            self.action_modality_embed = p(h)
        # host-side tables / rotary (kept off the module so .to(device) leaves them)
        object.__setattr__(self, "_host", {})
        object.__setattr__(
            self,
            "_rotary_host",
            Qwen3VLTextRotaryEmbedding(head_dim=cfg.head_dim, rope_theta=cfg.rope_theta, mrope_section=cfg.mrope_section),
        )
        self._attach_weight_loaders()

    # -- weights --------------------------------------------------------------------------
    def _attach_weight_loaders(self) -> None:
        if self.tp_size == 1:
            return
        from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

        for layer in self.layers:
            for name, dim in (("q_weight", 0), ("k_weight", 0), ("v_weight", 0), ("o_weight", 1), ("up_weight", 0), ("down_weight", 1)):
                prm = getattr(layer, name)
                set_weight_loader(prm, sharding_weight_loader(shard_dim=dim, shard_size=prm.shape[dim], num_shards=self.tp_size))

    def checkpoint_mappings(self) -> dict[str, str]:
        m = {
            "proj_in_weight": "proj_in.weight",
            "proj_in_bias": "proj_in.bias",
            "proj_out_weight": "proj_out.weight",
            "proj_out_bias": "proj_out.bias",
            "t_lin1_weight": "time_embedder.linear_1.weight",
            "t_lin1_bias": "time_embedder.linear_1.bias",
            "t_lin2_weight": "time_embedder.linear_2.weight",
            "t_lin2_bias": "time_embedder.linear_2.bias",
            "norm_out_weight": "norm_moe_gen.weight",
        }
        if self.cfg.action_gen:
            m["action_modality_embed"] = "action_modality_embed"
        for i in range(self.cfg.num_layers):
            p, c = f"layers.{i}", f"layers.{i}"
            m.update(
                {
                    f"{p}.q_weight": f"{c}.self_attn.add_q_proj.weight",
                    f"{p}.k_weight": f"{c}.self_attn.add_k_proj.weight",
                    f"{p}.v_weight": f"{c}.self_attn.add_v_proj.weight",
                    f"{p}.o_weight": f"{c}.self_attn.to_add_out.weight",
                    f"{p}.q_norm_weight": f"{c}.self_attn.norm_added_q.weight",
                    f"{p}.k_norm_weight": f"{c}.self_attn.norm_added_k.weight",
                    f"{p}.input_norm_weight": f"{c}.input_layernorm_moe_gen.weight",
                    f"{p}.post_norm_weight": f"{c}.post_attention_layernorm_moe_gen.weight",
                    f"{p}.up_weight": f"{c}.mlp_moe_gen.up_proj.weight",
                    f"{p}.down_weight": f"{c}.mlp_moe_gen.down_proj.weight",
                }
            )
        return m

    def load_weights(self, model_path: str, device: torch.device | str = "cpu") -> None:
        from safetensors import safe_open
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tdir = os.path.join(model_path, "transformer")
        result = SafetensorsCheckpoint(tdir).load_sharded_pipelined(
            self.tp_rank, self.tp_size, self, self.checkpoint_mappings(), torch.device(device)
        )
        self.load_state_dict(result.state_dict, strict=False, assign=True)
        if self.cfg.action_gen:  # DomainAwareLinear tables stay on the host (see _domain_weights)
            want = {"action_proj_in.fc.weight": "w_in", "action_proj_in.bias.weight": "b_in",
                    "action_proj_out.fc.weight": "w_out", "action_proj_out.bias.weight": "b_out"}
            for fn in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
                with safe_open(os.path.join(tdir, fn), "pt") as f:
                    for key in f.keys():
                        if key in want:
                            self._host[want[key]] = f.get_tensor(key)
        self.t_freqs = self.t_freqs.to(device)

    # -- host helpers ---------------------------------------------------------------------
    def rope_tables(self, text_mask, t, h, w, fps=None, t_action=0, action_start_frame_offset=1, action_fps=None):
        """GEN cos/sin ``[B, S_gen, 1, D]`` via upstream's own ``_compute_rope_freqs``."""
        cfg = self.cfg
        p = cfg.patch
        hp, wp = (h + p - 1) // p, (w + p - 1) // p
        shim = SimpleNamespace(
            temporal_modality_margin=cfg.temporal_modality_margin,
            base_fps=cfg.base_fps,
            temporal_compression_factor=cfg.temporal_compression_factor,
            enable_fps_modulation=cfg.enable_fps_modulation,
            sound_latent_fps=cfg.sound_latent_fps,
            temporal_compression_factor_sound=1,
            language_model=SimpleNamespace(rotary_emb=self._rotary_host),
        )
        _, (cos, sin) = Cosmos3VFMTransformer._compute_rope_freqs(
            shim, text_mask, t, hp, wp, fps, torch.device("cpu"), self.dtype,
            t_action=t_action, action_start_frame_offset=action_start_frame_offset, action_fps=action_fps,
        )
        return cos.contiguous(), sin.contiguous()

    @staticmethod
    def key_bias(text_mask: torch.Tensor, s_gen: int) -> torch.Tensor:
        # prefix-only (text keys); attention zero-extends it over the s_gen video keys
        del s_gen
        return key_padding_bias(text_mask.bool()).contiguous()

    def domain_weights(self, domain: int, device) -> tuple[torch.Tensor, ...]:
        """Gather one embodiment domain's action in/out projection (host lookup, cached)."""
        cache = self._host.setdefault("domains", {})
        if domain not in cache:
            hsz, ad = self.cfg.hidden_size, self.cfg.action_dim
            hst = self._host
            if not 0 <= domain < hst["w_in"].shape[0]:
                raise ValueError(f"action domain {domain} out of range [0, {hst['w_in'].shape[0]})")
            cache[domain] = tuple(
                x.to(self.dtype).contiguous().to(device)
                for x in (
                    hst["w_in"][domain].reshape(ad, hsz),
                    hst["b_in"][domain],
                    hst["w_out"][domain].reshape(hsz, ad),
                    hst["b_out"][domain],
                )
            )
        return cache[domain]

    # -- forward --------------------------------------------------------------------------
    def _patchify(self, x):
        b, c, t, h, w = x.shape
        p = self.cfg.patch
        x = x.reshape(b, c, t, h // p, p, w // p, p).permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(b, t * (h // p) * (w // p), p * p * c)

    def _unpatchify(self, tok, t, h, w):
        b, p, c = tok.shape[0], self.cfg.patch, self.cfg.latent_channel
        x = tok.reshape(b, t, h // p, w // p, p, p, c).permute(0, 6, 1, 2, 4, 3, 5)
        return x.reshape(b, c, t, h, w)

    def _time_embed(self, timestep, dtype):
        args = (timestep.float() * self.cfg.timestep_scale)[:, None] * self.t_freqs[None]
        tf = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        e = F.linear(F.silu(F.linear(tf, self.t_lin1_weight, self.t_lin1_bias)), self.t_lin2_weight, self.t_lin2_bias)
        return e.to(dtype)

    def _layers(self, hidden, cos, sin, key_bias, und_kv):
        n = len(und_kv) // 2
        for layer, k_und, v_und in zip(self.layers, und_kv[:n], und_kv[n:], strict=True):
            hidden = layer(hidden, k_und, v_und, cos, sin, key_bias, self.tp_group)
        return rms_norm(hidden, self.norm_out_weight, self.cfg.rms_norm_eps)

    def forward(self, latents, timestep, cos, sin, key_bias, noisy_mask, *und_kv):
        _, _, t, h, w = latents.shape
        hidden = F.linear(self._patchify(latents.to(self.dtype)), self.proj_in_weight, self.proj_in_bias)
        hidden = hidden + self._time_embed(timestep, hidden.dtype).unsqueeze(1) * noisy_mask
        hidden = self._layers(hidden, cos, sin, key_bias, und_kv)
        return self._unpatchify(F.linear(hidden, self.proj_out_weight, self.proj_out_bias), t, h, w)

    def forward_action(self, latents, timestep, cos, sin, key_bias, noisy_mask, action, action_noisy_mask,
                       w_in, b_in, w_out, b_out, *und_kv):
        _, _, t, h, w = latents.shape
        hv = F.linear(self._patchify(latents.to(self.dtype)), self.proj_in_weight, self.proj_in_bias)
        s_video = hv.shape[1]
        ha = torch.matmul(action.to(self.dtype), w_in) + b_in + self.action_modality_embed
        temb = self._time_embed(timestep, hv.dtype).unsqueeze(1)
        hidden = torch.cat([hv + temb * noisy_mask, ha + temb * action_noisy_mask], dim=1)
        hidden = self._layers(hidden, cos, sin, key_bias, und_kv)
        video = self._unpatchify(F.linear(hidden[:, :s_video], self.proj_out_weight, self.proj_out_bias), t, h, w)
        return video, torch.matmul(hidden[:, s_video:], w_out) + b_out
