# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-dev text encoder (Mistral-Small-3.x language tower) for NeuronCores.

FLUX.2 conditions on the hidden states after decoder layers 10, 20 and 30 of
``Mistral3ForConditionalGeneration`` (stacked on the channel axis, 3 x 5120 = 15360). Nothing
past layer 30 is used, so only layers ``0 .. max(layers)-1`` are built and loaded (30 of 40,
~3/4 of the tower), and neither the vision tower nor ``lm_head`` is loaded.

Math is HF ``MistralModel``: token embedding, RMSNorm, GQA attention with rotate-half RoPE
(theta from the config, positions ``0..S-1``), causal + key-padding mask, SwiGLU MLP. The
token-embedding gather runs on the host (one 512-row lookup per prompt); the decoder layers run
on the NeuronCores, ``group`` layers per compiled graph with the weights as graph inputs, so with
the default ``group=10`` a single NEFF serves all three 10-layer segments and each segment's
output is exactly one of the requested hidden states.

TP shards query / key-value heads and the MLP hidden dimension (one all-reduce after the
attention output projection, one after the MLP down projection).

The facade mirrors the bits of upstream's ``MistralEncoderModel`` the FLUX.2 pipeline uses:
``forward(input_ids, attention_mask, output_hidden_states=True)`` returning an object whose
``hidden_states[k]`` is the input of decoder layer ``k`` (host tensors), plus
``set_processor`` / ``dtype`` / ``device``.
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
    MASK_VALUE,
    TIMING,
    all_reduce,
    log_info,
    param,
    rms_norm,
    rope_half,
    set_loader,
    torch_attention,
    tp_state,
)

logger = init_logger(__name__)

ENCODER_COMPILER_ARGS = [
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
]
LAYER_KEYS = ("in_norm", "q", "k", "v", "o", "post_norm", "gate_up", "down")
DEFAULT_LAYERS = (10, 20, 30)


class MistralTextConfig(SimpleNamespace):
    @classmethod
    def from_model_dir(cls, model_path: str, subfolder: str = "text_encoder") -> MistralTextConfig:
        with open(os.path.join(model_path, subfolder, "config.json")) as f:
            raw = json.load(f)
        t = raw.get("text_config", raw)
        rope = t.get("rope_parameters") or {}
        hidden, heads = int(t["hidden_size"]), int(t["num_attention_heads"])
        return cls(
            hidden_size=hidden,
            intermediate_size=int(t["intermediate_size"]),
            num_layers=int(t["num_hidden_layers"]),
            num_heads=heads,
            num_kv_heads=int(t.get("num_key_value_heads", heads)),
            head_dim=int(t.get("head_dim") or hidden // heads),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-5)),
            rope_theta=float(t.get("rope_theta", rope.get("rope_theta", 1e9))),
            vocab_size=int(t.get("vocab_size", 131072)),
        )


def encoder_layers(cfg, nh, nkv, tp, group, n, h, cos, sin, bias, *w):
    """``n`` Mistral decoder layers. ``h`` ``[B,S,H]``, ``cos``/``sin`` ``[S,Dh]``, ``bias`` ``[B,1,S,S]`` fp32."""
    d, eps = cfg.head_dim, cfg.rms_norm_eps
    per = len(LAYER_KEYS)
    b, s, _ = h.shape
    for i in range(n):
        in_norm, q_w, k_w, v_w, o_w, post_norm, gate_up, down = w[i * per : (i + 1) * per]
        x = rms_norm(h, in_norm, eps)
        q = rope_half(F.linear(x, q_w).view(b, s, nh, d), cos, sin)
        k = rope_half(F.linear(x, k_w).view(b, s, nkv, d), cos, sin)
        v = F.linear(x, v_w).view(b, s, nkv, d)
        a = torch_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), d**-0.5, bias)
        h = h + all_reduce(F.linear(a.transpose(1, 2).reshape(b, s, nh * d), o_w), tp, group)
        x = rms_norm(h, post_norm, eps)
        g, u = F.linear(x, gate_up).chunk(2, dim=-1)
        h = h + all_reduce(F.linear(F.silu(g) * u, down), tp, group)
    return h


class _Layer(nn.Module):
    pass


class NeuronFlux2TextEncoder(nn.Module):
    """Mistral3 language tower up to the deepest requested hidden state."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        group: int | None = None,
        **_unused,
    ):
        super().__init__()
        self.model_path = model_path
        self.cfg = cfg = MistralTextConfig.from_model_dir(model_path)
        self._dtype = dtype
        self.out_layers = tuple(sorted(set(int(x) for x in layers)))
        self.depth = max(self.out_layers)
        if self.depth > cfg.num_layers:
            raise ValueError(
                f"hidden state {self.depth} requested; the text encoder has {cfg.num_layers} layers"
            )
        self.group = int(group or os.environ.get("FLUX2_TE_GROUP", "10"))
        if any(x % self.group for x in self.out_layers):
            raise ValueError(
                f"FLUX2_TE_GROUP={self.group} must divide every output layer {self.out_layers}"
            )
        self.tp_size, self.tp_rank, self.tp_group = tp_state()
        tp = self.tp_size
        # TP above the KV-head count replicates each KV head over tp // num_kv_heads ranks (GQA: heads
        # are sharded contiguously, so all of a rank's query heads share that one KV head).
        kv_rep = tp // cfg.num_kv_heads if tp > cfg.num_kv_heads else 1
        if cfg.num_heads % tp or cfg.intermediate_size % tp or (cfg.num_kv_heads * kv_rep) % tp:
            raise ValueError(
                f"TP={tp} must divide heads={cfg.num_heads} and mlp={cfg.intermediate_size}, "
                f"and divide or be a multiple of kv={cfg.num_kv_heads}"
            )
        self.nh, self.nkv = cfg.num_heads // tp, cfg.num_kv_heads * kv_rep // tp
        H, d, M = cfg.hidden_size, cfg.head_dim, cfg.intermediate_size
        Qr, KVr, Mr = self.nh * d, self.nkv * d, M // tp
        p = partial(param, dtype=dtype)
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            L = _Layer()
            L.in_norm, L.post_norm = p((H,)), p((H,))
            L.q, L.k, L.v = p((Qr, H)), p((KVr, H)), p((KVr, H))
            set_loader(L.q, 0, [(0, Qr)])
            set_loader(L.k, 0, [(0, KVr)], rank_div=kv_rep)
            set_loader(L.v, 0, [(0, KVr)], rank_div=kv_rep)
            L.o = p((H, Qr))
            set_loader(L.o, 1, [(0, Qr)])
            L.gate_up = p((2 * Mr, H))  # fused [gate | up] from the two checkpoint tensors
            L.down = p((H, Mr))
            set_loader(L.down, 1, [(0, Mr)])
            self.layers.append(L)
        self._fuse_gate_up_loaders(Mr)
        object.__setattr__(self, "_embed", None)  # host-side table, not a module parameter
        self._device = torch.device("cpu")
        self.processor = None
        self.stats = {"calls": 0, "seconds": 0.0}
        self._lead = (cfg, self.nh, self.nkv, tp, self.tp_group, self.group)
        self._fn = partial(encoder_layers, *self._lead)

    def _fuse_gate_up_loaders(self, mr: int) -> None:
        from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

        def transform(slices, rank):
            g, u = slices
            return torch.cat(
                [g[rank * mr : (rank + 1) * mr], u[rank * mr : (rank + 1) * mr]], dim=0
            )

        for L in self.layers:
            set_weight_loader(L.gate_up, SafetensorsWeightLoader(transform=transform))

    # -- upstream MistralEncoderModel facade ------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")  # inputs / outputs are host tensors

    def set_processor(self, processor, **kwargs) -> None:
        self.processor = processor
        self._processor_kwargs = kwargs

    # -- weights ------------------------------------------------------------------------------
    def checkpoint_mappings(self) -> dict:
        m = {}
        pre = "language_model.model.layers"
        for i in range(self.depth):
            m.update(
                {
                    f"layers.{i}.in_norm": f"{pre}.{i}.input_layernorm.weight",
                    f"layers.{i}.post_norm": f"{pre}.{i}.post_attention_layernorm.weight",
                    f"layers.{i}.q": f"{pre}.{i}.self_attn.q_proj.weight",
                    f"layers.{i}.k": f"{pre}.{i}.self_attn.k_proj.weight",
                    f"layers.{i}.v": f"{pre}.{i}.self_attn.v_proj.weight",
                    f"layers.{i}.o": f"{pre}.{i}.self_attn.o_proj.weight",
                    f"layers.{i}.gate_up": [
                        f"{pre}.{i}.mlp.gate_proj.weight",
                        f"{pre}.{i}.mlp.up_proj.weight",
                    ],
                    f"layers.{i}.down": f"{pre}.{i}.mlp.down_proj.weight",
                }
            )
        return m

    def load_weights(
        self, model_path: str | None = None, device: torch.device | str | None = None
    ) -> None:
        from safetensors import safe_open
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        model_path = model_path or self.model_path
        device = torch.device(device) if device is not None else self._device
        t0 = time.time()
        tdir = os.path.join(model_path, "text_encoder")
        res = SafetensorsCheckpoint(tdir).load_sharded_pipelined(
            self.tp_rank, self.tp_size, self, self.checkpoint_mappings(), device
        )
        self.load_state_dict(res.state_dict, strict=True, assign=True)
        key = "language_model.model.embed_tokens.weight"
        index = os.path.join(tdir, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                shards = [json.load(f)["weight_map"][key]]
        else:
            shards = sorted(x for x in os.listdir(tdir) if x.endswith(".safetensors"))
        for shard in shards:
            with safe_open(os.path.join(tdir, shard), "pt") as f:
                if key in f.keys():
                    object.__setattr__(self, "_embed", f.get_tensor(key).to(self._dtype))
                    break
        self._device = device
        self.load_seconds = time.time() - t0
        self.param_gb = sum(p.numel() * p.element_size() for p in self.parameters()) / 2**30
        log_info(
            "flux2 text encoder: %d layers, tp_rank %d/%d on %s in %.1fs (%.2f GiB weights on this rank)",
            self.depth,
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

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> None:
        opts = {
            **dict(options or {}),
            "model_name": f"flux2_te_tp{self.tp_size}_g{self.group}",
            "compiler_args": list(ENCODER_COMPILER_ARGS),
        }
        self._fn = partial(
            torch.compile(
                encoder_layers,
                backend=backend,
                options=opts,
                fullgraph=kwargs.get("fullgraph", True),
                dynamic=False,
            ),
            *self._lead,
        )

    # -- host helpers -------------------------------------------------------------------------
    def rope_tables(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        """HF ``MistralRotaryEmbedding`` for positions ``0..s-1``: ``[s, head_dim]`` fp32."""
        d = self.cfg.head_dim
        inv = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        freqs = torch.arange(s, dtype=torch.float32)[:, None] * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self._dtype), emb.sin().to(self._dtype)

    @staticmethod
    def mask_bias(attention_mask: torch.Tensor) -> torch.Tensor:
        """Causal + key-padding additive bias ``[B, 1, S, S]`` fp32."""
        b, s = attention_mask.shape
        causal = torch.ones(s, s, dtype=torch.bool).tril()
        ok = causal[None] & attention_mask.bool()[:, None, :]
        return torch.where(ok, 0.0, MASK_VALUE).float()[:, None]

    # -- forward ------------------------------------------------------------------------------
    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=True, use_cache=False, **kwargs
    ):
        t0 = time.time()
        dev = self._device
        ids = input_ids.detach().to("cpu").long()
        b, s = ids.shape
        mask = (
            torch.ones(b, s, dtype=torch.long)
            if attention_mask is None
            else attention_mask.detach().to("cpu")
        )
        if self._embed is None:
            raise RuntimeError("text encoder weights are not loaded")
        h = self._embed[ids].contiguous().to(dev)
        cos, sin = (x.contiguous().to(dev) for x in self.rope_tables(s))
        bias = self.mask_bias(mask).contiguous().to(dev)
        hidden: list = [None] * (self.depth + 1)
        hidden[0] = h.to("cpu")
        with torch.no_grad():
            for start in range(0, self.depth, self.group):
                ws = []
                for L in self.layers[start : start + self.group]:
                    ws.extend(getattr(L, k).detach() for k in LAYER_KEYS)
                h = self._fn(h, cos, sin, bias, *ws)
                if start + self.group in self.out_layers:
                    hidden[start + self.group] = h.to("cpu")
        self.stats["calls"] += 1
        self.stats["seconds"] += time.time() - t0
        if TIMING:
            log_info("timing text_encoder %.3fs (S=%d)", time.time() - t0, s)
        return SimpleNamespace(hidden_states=tuple(hidden), last_hidden_state=None)
