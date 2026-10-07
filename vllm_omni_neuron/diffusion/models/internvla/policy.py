# SPDX-License-Identifier: Apache-2.0
"""InternVLA-A1.5 policy (Qwen3.5-2B VLM + unified action expert + flow-matching head).

Inference is upstream ``InternVLAA15.sample_actions``: the VLM encodes the prefix (camera
views + prompt) once, and the action expert -- a 24-layer Qwen3.5 decoder with the VLM's layer
pattern but hidden 1024 -- runs ``num_inference_steps`` Euler steps over a 100-token suffix
(50 learnable foresight tokens + a 50-step action chunk). The two towers only meet in the 6
gated full-attention layers, where suffix queries attend to ``[prefix K/V, suffix K/V]``; the
18 Gated-DeltaNet layers of the expert never see prefix state (upstream overwrites it). So
the prefix hands the expert exactly 6 (K, V) pairs, and nothing else.

On Neuron the prefix and the action expert run as small fixed-shape graphs, each compiled once
per shape bucket:

* :class:`VisionEmbedGraph`   pixels + cached token embeddings -> prefix embeddings
* :class:`SplitPrefix`        24 prefix layers as 6 fused Gated-DeltaNet runs + 6 full-attention
  layers -> 6 x (K, V)
* :class:`DenoiseGraph`       one Euler step ``x_{t+dt} = x_t + dt * v(x_t, t)`` over the 100-token
  suffix

The CPU reference path keeps upstream's prefix structure (:class:`VisionGraph`, host scatter,
:class:`PrefixGraph`).

The WAN video branch is training-only and is not built; its projection in the checkpoint is
skipped. Module names match the checkpoint (``model.`` prefix stripped).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.layers.block_graphs import BlockGraphRunner

from . import preprocess as pp
from .config import InternVLAConfig
from .qwen3_5 import Qwen35DecoderLayer, Qwen35TextModel, Qwen35VisionModel, eager_attention

_SKIP_PREFIXES = ("learnable_to_wan_proj.", "_wan_grid_sizes", "wan_video_model.")
_FP32_ALWAYS = ("action_out_proj.",)  # upstream Policy.to() keeps the action head fp32


class _VLM(nn.Module):
    def __init__(self, cfg: InternVLAConfig):
        super().__init__()
        self.visual = Qwen35VisionModel(cfg.vlm.vision)
        self.language_model = Qwen35TextModel(cfg.vlm.text)


class _Qwen35ForCG(nn.Module):  # mirrors Qwen3_5ForConditionalGeneration's ``model`` attribute
    def __init__(self, cfg: InternVLAConfig):
        super().__init__()
        self.model = _VLM(cfg)


class _WithExpert(nn.Module):
    def __init__(self, cfg: InternVLAConfig):
        super().__init__()
        self.qwen3_5 = _Qwen35ForCG(cfg)
        p = cfg.policy
        self.action_expert = Qwen35TextModel(
            cfg.vlm.text,
            p.action_expert_hidden_size,
            p.action_expert_intermediate_size,
            with_embeddings=False,
        )


class InternVLAA15(nn.Module):
    def __init__(self, cfg: InternVLAConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.policy
        h = p.action_expert_hidden_size
        self.qwen3_5_with_expert = _WithExpert(cfg)
        self.action_in_proj = nn.Linear(p.max_action_dim, h)
        self.action_out_proj = nn.Linear(h, p.max_action_dim)
        if not p.tokenize_state:
            self.state_proj = nn.Linear(p.max_state_dim, h)
        self.action_time_mlp_in = nn.Linear(2 * h, h)
        self.action_time_mlp_out = nn.Linear(h, h)
        self.learnable_tokens = nn.Parameter(torch.zeros(p.num_learnable_tokens, h))
        self.learnable_tokens_in_proj = nn.Linear(h, h)

    @property
    def vlm(self) -> _VLM:
        return self.qwen3_5_with_expert.qwen3_5.model

    @property
    def expert(self) -> Qwen35TextModel:
        return self.qwen3_5_with_expert.action_expert

    # -- weights ----------------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device="cpu",
        vlm_config: str | None = None,
    ) -> InternVLAA15:
        cfg = InternVLAConfig.from_model_dir(model_path, vlm_config)
        with torch.device("meta"):
            model = cls(cfg)
        model.load_checkpoint(os.path.join(model_path, "model.safetensors"), dtype, device)
        model.eval()
        # Inference-only: every param/buffer must be requires_grad=False, or torch.compile traces a
        # backward graph for it and the Neuron Lite backend fails ("doesn't support events") the
        # moment that graph is replayed alongside another compiled graph (e.g. vision + the
        # per-layer prefix loop) in the same process. load_checkpoint already loads with
        # requires_grad=False, but this is the one call site that guarantees it model-wide,
        # independent of how any future module gets constructed.
        model.requires_grad_(False)
        return model

    def load_checkpoint(self, path: str, dtype: torch.dtype, device="cpu") -> dict:
        """Strict load of ``model.safetensors``. Upstream inference runs ``policy.to(bf16)``, so
        every tensor goes to ``dtype`` except the fp32 action head; the tied input embedding
        is read from ``lm_head.weight``. Returns load stats."""
        from safetensors import safe_open

        t0 = time.time()
        expected = dict(self.named_parameters())
        state, skipped = {}, []
        with safe_open(path, "pt") as f:
            for key in f.keys():
                name = key[len("model.") :] if key.startswith("model.") else key
                if name.startswith(_SKIP_PREFIXES):
                    skipped.append(name)
                    continue
                if name == "qwen3_5_with_expert.qwen3_5.lm_head.weight":
                    name = "qwen3_5_with_expert.qwen3_5.model.language_model.embed_tokens.weight"
                if name not in expected:
                    raise KeyError(f"unexpected checkpoint tensor {key}")
                t = f.get_tensor(key)
                want = expected[name].shape
                if tuple(t.shape) != tuple(want):
                    raise ValueError(f"{key}: checkpoint {tuple(t.shape)} vs model {tuple(want)}")
                td = torch.float32 if name.startswith(_FP32_ALWAYS) else dtype
                state[name] = nn.Parameter(
                    t.to(td).to(device), requires_grad=False
                )  # cast on host first
        missing = sorted(set(expected) - set(state))
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
        self.load_state_dict(state, strict=True, assign=True)
        n = sum(t.numel() for t in state.values())
        return {
            "tensors": len(state),
            "params": n,
            "skipped": len(skipped),
            "load_s": round(time.time() - t0, 2),
        }


# -- graphs -----------------------------------------------------------------------------------


class VisionGraph(nn.Module):
    """``(patches [N,P,K], pe_idx [4,P], pe_w [4,P], cos [P,hd], sin) -> [N*P/4, D_text]``."""

    def __init__(self, model: InternVLAA15):
        super().__init__()
        self.visual = model.vlm.visual

    def forward(self, patches, pe_idx, pe_w, cos, sin):
        return self.visual(patches, pe_idx, pe_w, cos, sin)


class PrefixGraph(nn.Module):
    """VLM prefix pass. Inputs: ``input_ids [B,L]``, ``image_slots [B,L,D]`` (each image token's
    embedding placed at its position, zeros elsewhere; the host does the placement with an
    index_copy), ``image_mask [B,L,1]`` (1 at image-token positions), ``cos/sin [B,L,R]`` fp32,
    ``bias [B,1,L,L]`` fp32. Returns ``(k_0, .., k_5, v_0, .., v_5)``, each ``[B, Hkv, L, D]``.

    This is upstream's ``embs[ids == image_token_id] = image_embs`` written as an elementwise
    select (``text*(1-mask) + image_slots``) instead of a gather: a gather over the concatenated
    ``[text, image]`` sequence compiles to a strided access the Neuron compiler rejects
    (``NCC_ITEN406 too many partition dimensions``), while the select is a plain fixed-shape op.
    """

    def __init__(self, model: InternVLAA15):
        super().__init__()
        self.lm = model.vlm.language_model
        self.full = model.cfg.vlm.text.full_attention_layers

    def forward(self, input_ids, image_slots, image_mask, cos, sin, bias):
        x = self.lm.embed_tokens(input_ids)
        m = image_mask.to(x.dtype)
        x = x * (1.0 - m) + image_slots.to(x.dtype) * m
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        ks, vs = [], []
        last = self.full[-1]
        for i, layer in enumerate(self.lm.layers[: last + 1]):
            if layer.layer_type == "linear_attention":
                x = layer.forward_linear(x)
            else:
                # K/V come out as transposed views ([B,Hkv,L,D] over an [B,L,Hkv,D] base); return
                # them contiguous, else the stacked graph output is a strided access the compiler
                # rejects (NCC_ITEN406 too many partition dimensions)
                x, (k, v) = layer.forward_full(x, cos, sin, bias, kv_only=(i == last))
                ks.append(k.contiguous())
                vs.append(v.contiguous())
        # one stacked K and one stacked V, not 12 separate outputs: returning the per-layer tensors
        # individually makes the compiler emit a strided multi-output access it rejects
        # (NCC_ITEN406 too many partition dimensions)
        return torch.stack(ks), torch.stack(vs)  # each [num_full, B, Hkv, L, D]


# -- per-layer prefix graphs (device path) ----------------------------------------------------
# The whole-prefix graph above compiles on CPU but overflows the Neuron compiler: returning the 6
# full-layer K/V keeps every Gated-DeltaNet chunk-rule reshape ([B,H,n,chunk,chunk]) live, and the
# compiler rejects the resulting strided access (NCC_ITEN406). The fix is the LESSONS N-block split
# at group_size=1: each layer is its own small compiled graph and the host loop carries ``x`` and
# collects K/V between calls. A0's BlockGraphRunner only takes a structurally-UNIFORM stack and this
# stack interleaves two layer types, so we compile one graph per type (two graphs total) and reuse
# them across all 24 layers.


class _LinearLayerFwd(nn.Module):
    """Wrapper whose ``forward`` is the linear-attn path, so ``functional_call`` can drive it."""

    def __init__(self, layer: Qwen35DecoderLayer):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.linear_attn = layer.linear_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def ffn(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(self, x):
        return self.ffn(x + self.linear_attn(self.input_layernorm(x)))


class _FullLayerFwd(nn.Module):
    """Wrapper whose ``forward`` is the full-attn path (returns ``(x, k, v)``)."""

    def __init__(self, layer: Qwen35DecoderLayer):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def ffn(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(self, x, cos, sin, bias):
        h = self.input_layernorm(x)
        q, k, v, gate = self.self_attn.project(h, cos, sin)
        attn = eager_attention(q, k, v, bias, self.self_attn.scaling)
        x = x + self.self_attn.finish(attn, gate)
        return self.ffn(x), k.contiguous(), v.contiguous()


class VisionEmbedGraph(nn.Module):
    """Vision tower + prefix embedding assembly in ONE graph (device path).

    ``(patches, pe_idx, pe_w, vcos, vsin, text_emb [B,L,D], place [B,L,M]) -> x [B,L,D]``.
    ``text_emb`` is the token embedding with image positions zeroed (looked up on the host, cached
    per prompt); ``place`` is the one-hot image-token placement matrix (``place[b, l, j] = 1`` when
    sequence position ``l`` holds image token ``j``). ``place @ image_tokens`` is upstream's
    ``embs[ids == image_token_id] = image_embs`` as a fixed-shape matmul: each output row is one
    image token times 1.0 plus zeros, so it is exact in any dtype. This replaces a device->host
    sync of the image tokens, a host scatter, and an eager on-device embedding lookup that cost
    324 ms of a 382 ms prefix (the lookup against the 250k-row table runs uncompiled).
    """

    def __init__(self, model: InternVLAA15):
        super().__init__()
        self.visual = model.vlm.visual

    def forward(self, patches, pe_idx, pe_w, cos, sin, text_emb, place):
        img = self.visual(patches, pe_idx, pe_w, cos, sin)
        img = img.reshape(place.shape[0], place.shape[2], -1).to(text_emb.dtype)
        return text_emb + torch.matmul(place.to(text_emb.dtype), img)


class PrefixFullLayer(nn.Module):
    """One gated full-attention prefix layer: ``(x, cos, sin, bias) -> (x, k, v)`` with contiguous
    ``k/v`` ``[B, Hkv, L, D]``. One compiled graph serves every full-attn layer."""

    def __init__(self, template: _FullLayerFwd):
        super().__init__()
        self.template = template

    def forward(self, x, cos, sin, bias, weights):
        from torch.func import functional_call

        return functional_call(self.template, weights, (x, cos, sin, bias), {}, strict=True)


class SplitPrefix:
    """Device prefix driven layer by layer, from the prefix embeddings ``x`` (built by
    :class:`VisionEmbedGraph`). The GDN layers come in 6 contiguous runs of 3 (the checkpoint's
    [lin,lin,lin,full] x 6 pattern); each run is fused into ONE compiled graph via A0's
    :class:`BlockGraphRunner` (``group_size=3``), so the host dispatches once per run of 3
    instead of 3 separate device calls. Full-attention layers (needed individually for their K/V)
    keep one shared template graph, called once per layer with weights bound once at load.
    Returns the full-layer K and V as two lists (one entry per full-attention layer)."""

    def __init__(self, model: InternVLAA15, compile_fn):
        self.lm = model.vlm.language_model
        self.full = model.cfg.vlm.text.full_attention_layers
        self.last = self.full[-1]
        wrap = compile_fn or (lambda mod, name: mod)
        full_t = next(ly for ly in self.lm.layers if ly.layer_type == "full_attention")
        self.fullg = wrap(PrefixFullLayer(_FullLayerFwd(full_t)), "internvla_prefix_full")

        # Group consecutive linear-attention layers (within the active prefix depth) into runs,
        # each driven by its own BlockGraphRunner (one compiled graph per run, group_size = run
        # length). Every run has the same length (3) on this checkpoint, so in practice this is a
        # single compiled graph reused across all 6 runs -- group_size and the template's structural
        # signature (class + every param's shape/dtype) are what key compilation.
        self.lin_runs: list[tuple[int, BlockGraphRunner]] = []
        layers = self.lm.layers[: self.last + 1]
        i = 0
        while i < len(layers):
            if layers[i].layer_type != "linear_attention":
                i += 1
                continue
            j = i
            while j < len(layers) and layers[j].layer_type == "linear_attention":
                j += 1
            run = [_LinearLayerFwd(ly) for ly in layers[i:j]]
            runner = BlockGraphRunner(
                run, group_size=len(run), compile_fn=lambda f: wrap(f, "internvla_prefix_linear")
            )
            self.lin_runs.append((i, runner))
            i = j
        self._run_by_start = dict(self.lin_runs)
        self.refresh_weights()

    def refresh_weights(self) -> None:
        """Bind each full-attention layer's weight dict once (re-run after moving the model)."""
        self._full_w = {
            i: self._full_weights(ly)
            for i, ly in enumerate(self.lm.layers[: self.last + 1])
            if ly.layer_type == "full_attention"
        }
        for _, runner in self.lin_runs:
            runner.refresh_weights()

    @staticmethod
    def _full_weights(layer):
        sub = (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)
        prefixes = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        out = {}
        for pre, mod in zip(prefixes, sub):
            for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                out[f"{pre}.{n}"] = t.detach()
        return out

    def __call__(self, x, cos, sin, bias):
        ks, vs = [], []
        n = self.last + 1
        i = 0
        while i < n:
            runner = self._run_by_start.get(i)
            if runner is not None:
                x = runner(x)
                i += runner.group_size
            else:
                x, k, v = self.fullg(x, cos, sin, bias, self._full_w[i])
                ks.append(k)
                vs.append(v)
                i += 1
        return ks, vs


class DenoiseGraph(nn.Module):
    """One Euler step of the action expert. Inputs: ``x_t [B,chunk,A]`` fp32, ``time_emb [B,H]``,
    ``dt [1]`` fp32, suffix ``cos/sin [B,S,R]`` fp32, ``bias [B,1,S,L+S]`` fp32, then the stacked
    prefix keys ``ks`` and values ``vs``, each ``[num_full, B, Hkv, L, D]``. Returns
    ``x_t + dt * v_t`` fp32."""

    def __init__(self, model: InternVLAA15):
        super().__init__()
        self.m = model
        self.expert = model.expert
        self.chunk = model.cfg.policy.chunk_size

    def velocity(self, x_t, time_emb, cos, sin, bias, ks, vs):
        m = self.m
        dtype = m.action_in_proj.weight.dtype
        b = x_t.shape[0]
        lt = m.learnable_tokens_in_proj(m.learnable_tokens)[None].expand(b, -1, -1)
        act = m.action_in_proj(x_t.to(dtype))
        te = time_emb.to(dtype)[:, None, :].expand_as(act)
        at = m.action_time_mlp_out(F.silu(m.action_time_mlp_in(torch.cat([act, te], dim=2))))
        x = torch.cat([lt, at], dim=1)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        j = 0
        for layer in self.expert.layers:
            if layer.layer_type == "linear_attention":
                x = layer.forward_linear(x)
            else:
                x, _ = layer.forward_full(x, cos, sin, bias, past_kv=(ks[j], vs[j]))
                j += 1
        x = self.expert.norm(x)[:, -self.chunk :]
        return m.action_out_proj(x.float())

    def forward(self, x_t, time_emb, dt, cos, sin, bias, ks, vs):
        return x_t + dt * self.velocity(x_t, time_emb, cos, sin, bias, ks, vs)


# -- host driver ------------------------------------------------------------------------------


def pad_bucket(n: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"prefix of {n} tokens exceeds the largest bucket {buckets[-1]}")


class InternVLAA15Runner:
    """Host-side ``sample_actions``: builds every table on the CPU and calls the graphs.

    ``device`` is ``cpu`` (eager reference) or a Neuron device; ``compile_fn(module, name)``
    returns the callable to use for each graph (identity for eager).

    The CPU path is the reference: whole-prefix graph, exactly upstream's structure. The device
    path runs one vision+embedding graph, then the split prefix. Request tables that depend only
    on the prompt and image layout are built once and kept on the device (LRU of
    ``INTERNVLA_TABLE_CACHE`` entries). Both paths run ``num_inference_steps`` launches of
    :class:`DenoiseGraph`.
    """

    # 448: a 3-camera request with upstream's prompt (task + 32-value state) is 386-388 tokens;
    # padding it to 448 instead of 512 saves 8 ms of prefix per request (94.3 -> 86.2 ms on trn2)
    PREFIX_BUCKETS = tuple(
        int(b)
        for b in os.environ.get("INTERNVLA_PREFIX_BUCKETS", "256,384,448,512,768,1024").split(",")
    )

    def __init__(self, model: InternVLAA15, device="cpu", compile_fn=None):
        self.model = model
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.dtype = model.action_in_proj.weight.dtype
        wrap = compile_fn or (lambda mod, name: mod)
        self.timings: dict[str, float] = {}
        if self.device.type == "cpu":
            self.vision = wrap(VisionGraph(model), "internvla_vision")
            self.prefix = wrap(PrefixGraph(model), "internvla_prefix")
        else:
            # Whole-prefix single graph overflows the Neuron compiler (NCC_ITEN406); on device drive
            # the prefix layer by layer (one small compiled graph per layer type).
            self.vision_embed = wrap(VisionEmbedGraph(model), "internvla_vision_embed")
            self.prefix = SplitPrefix(model, compile_fn)
        self.denoise = wrap(DenoiseGraph(model), "internvla_denoise")
        self._tables: OrderedDict = OrderedDict()
        self._table_cache = int(os.environ.get("INTERNVLA_TABLE_CACHE", "16"))
        self._embed_host = None  # host copy of the token embedding (device path), made on first use

    # Host tables that must reach the device in fp32 (every other float table goes in the model
    # dtype, see ``_dev``):
    # * ``vcos``/``vsin``: upstream's vision attention applies its rope in fp32 from fp32 cos/sin.
    #   Rounded to bf16 they give the image tokens a systematic (not random) error: with them the
    #   device's end-to-end action error was 1.35x (2 views) / 1.42x (3 views) the CPU-bf16 error
    #   over 6 noise seeds, with fp32 tables it is below the CPU-bf16 error. The text and suffix
    #   rope stay bf16: upstream casts those to the model dtype itself.
    # * ``dt``: the Euler step stays fp32, as upstream (see ``_dev``).
    FP32_TABLES = ("vcos", "vsin", "dt")

    def _dev(self, t, exact: bool = False):
        """Host->device copy. Host tables (text rope cos/sin, additive biases, the time embedding)
        are cast to the model dtype on the host first -- the graphs were compiled for that signature
        and up-cast internally where upstream does. (The Neuron runtime rejects a dtype change
        *during* the copy, ``.to(dev, dtype)``; a plain fp32 copy is fine.) ``exact=True`` keeps the
        dtype, for the tables in ``FP32_TABLES`` and for the flow-matching state ``x_t``: as in
        upstream ``sample_actions``, ``x_t`` and ``dt`` stay fp32 (it casts ``x_t`` to the model
        dtype only for the velocity input and keeps the Euler accumulator in fp32). Rounding them to
        bf16 costs 3.7x the CPU-bf16 error per step (``dt=-0.1`` becomes ``-0.10009766``, a 0.1%
        step-size bias, and ``x_t`` loses ~0.4% per element)."""
        if (
            not exact
            and t.is_floating_point()
            and t.dtype != self.dtype
            and self.device.type != "cpu"
        ):
            t = t.to(self.dtype)
        return t.contiguous().to(self.device)

    def _sync(self, t):
        return t.to("cpu") if self.device.type != "cpu" else t

    def prepare(self, batch: dict, bucket: int | None = None) -> dict:
        """All host tables for one request (upstream ``predict_action_chunk`` inputs)."""
        cfg, vlm = self.cfg, self.cfg.vlm
        ids, mask = batch["input_ids"], batch["attention_mask"]
        grid = batch["image_grid_thw"].view(-1, 3)
        b, n_real = ids.shape
        if (
            not all(int(t) == 1 for t in grid[:, 0])
            or len({(int(h), int(w)) for _, h, w in grid.tolist()}) != 1
        ):
            raise ValueError("all images must be stills (t=1) of one resolution")
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        n_img = grid.shape[0]
        p = gh * gw
        patches = batch["pixel_values"].view(n_img, p, -1)
        pe_idx, pe_w, vcos, vsin = pp.vision_tables(gh, gw, vlm.vision)

        length = bucket or pad_bucket(n_real, self.PREFIX_BUCKETS)
        pad_id = 248044
        ids_p = torch.full((b, length), pad_id, dtype=torch.long)
        ids_p[:, :n_real] = ids
        mask_p = torch.zeros(b, length, dtype=torch.long)
        mask_p[:, :n_real] = mask
        pos = pp.rope_index(ids_p, mask_p, grid.tolist(), vlm)
        cos, sin = pp.text_rope(pos, vlm.text)
        m_img = n_img // b * p // vlm.vision.spatial_merge_size**2
        img_pos = torch.zeros(b, m_img, dtype=torch.long)  # where this row's image tokens land
        image_mask = torch.zeros(b, length, 1)
        for i in range(b):
            where = (ids_p[i] == vlm.image_token_id).nonzero().flatten()
            if where.numel() != m_img:
                raise ValueError(f"row {i}: {where.numel()} image tokens, expected {m_img}")
            img_pos[i] = where
            image_mask[i, where, 0] = 1.0
        pad = mask_p.bool()
        fast = batch.get("fast_token_mask")
        if fast is not None:
            fast = F.pad(fast.bool(), (0, length - fast.shape[1]))
        s = len(pp.suffix_layout(cfg.policy))
        spos = pp.suffix_positions(pos, s)
        scos, ssin = pp.text_rope(spos, vlm.text)
        temb, dt = pp.time_embedding_table(
            cfg.policy, cfg.policy.action_expert_hidden_size, self.dtype, b
        )
        return {
            "patches": patches,
            "pe_idx": pe_idx,
            "pe_w": pe_w,
            "vcos": vcos,
            "vsin": vsin,
            "input_ids": ids_p,
            "img_pos": img_pos,
            "image_mask": image_mask,
            "cos": cos,
            "sin": sin,
            "bias": pp.prefix_bias(pad),
            "scos": scos,
            "ssin": ssin,
            "sbias": pp.suffix_bias(pad, cfg.policy, fast),
            "temb": temb,
            "dt": dt.reshape(1),
            "batch": b,
            "length": length,
            "m_img": m_img,
        }

    def _suffix_tables(self, h: dict) -> dict:
        """Host tables for the denoise loop."""
        steps = h["temb"].shape[0]
        return {
            "dt": h["dt"],
            "scos": h["scos"],
            "ssin": h["ssin"],
            "sbias": h["sbias"],
            "temb": [h["temb"][i] for i in range(steps)],
        }

    def _device_tables(self, batch: dict, bucket: int | None) -> dict:
        """Device-resident request tables (everything except the pixels), cached per prompt/image
        layout: rope, biases, vision taps, the time-embedding table, the host-looked-up token
        embeddings and the image-token placement matrix. A repeated instruction with the same
        camera layout reuses them; only the pixels go host->device per request."""
        fast = batch.get("fast_token_mask")
        key = (
            batch["input_ids"].numpy().tobytes(),
            tuple(batch["input_ids"].shape),
            batch["attention_mask"].numpy().tobytes(),
            tuple(batch["image_grid_thw"].reshape(-1).tolist()),
            None if fast is None else fast.numpy().tobytes(),
            bucket,
        )
        hit = self._tables.get(key)
        if hit is not None:
            self._tables.move_to_end(key)
            return hit
        h = self.prepare(batch, bucket)
        if self._embed_host is None:
            self._embed_host = self.model.vlm.language_model.embed_tokens.weight.detach().to("cpu")
        mask = h["image_mask"].to(self.dtype)
        text_emb = F.embedding(h["input_ids"], self._embed_host) * (1.0 - mask)
        place = torch.zeros(h["batch"], h["length"], h["m_img"], dtype=self.dtype)
        for i in range(h["batch"]):
            place[i, h["img_pos"][i], torch.arange(h["m_img"])] = 1.0
        d = self._dev
        st = {k: h[k] for k in ("batch", "length", "m_img")}
        st.update(
            {
                k: d(h[k], k in self.FP32_TABLES)
                for k in ("pe_idx", "pe_w", "vcos", "vsin", "cos", "sin", "bias")
            }
        )
        st.update(text_emb=d(text_emb), place=d(place))
        for k, v in self._suffix_tables(h).items():
            exact = k in self.FP32_TABLES
            st[k] = [d(t, exact) for t in v] if isinstance(v, list) else d(v, exact)
        self._tables[key] = st
        while len(self._tables) > self._table_cache:
            self._tables.popitem(last=False)
        return st

    def _denoise(self, st: dict, x: torch.Tensor, ks, vs, return_trajectory: bool):
        traj = []
        for te in st["temb"]:
            x = self.denoise(x, te, st["dt"], st["scos"], st["ssin"], st["sbias"], ks, vs)
            if return_trajectory:
                traj.append(self._sync(x).clone())
        return x, traj

    @torch.inference_mode()
    def sample_actions(
        self,
        batch: dict,
        noise: torch.Tensor,
        bucket: int | None = None,
        return_trajectory: bool = False,
    ):
        t0 = time.time()
        d = self._dev
        if self.device.type == "cpu":
            h = self.prepare(batch, bucket)
            st = {**h, **self._suffix_tables(h)}
            t1 = time.time()
            img = self.vision(
                h["patches"].to(self.dtype), h["pe_idx"], h["pe_w"], h["vcos"], h["vsin"]
            )
            img = img.reshape(h["batch"], h["m_img"], -1).to(self.dtype)
            # place each image token's embedding at its sequence position (upstream's masked assign)
            slots = torch.zeros(h["batch"], h["length"], img.shape[-1], dtype=self.dtype)
            for i in range(h["batch"]):
                slots[i].index_copy_(0, h["img_pos"][i], img[i])
            ks, vs = self.prefix(
                h["input_ids"], slots, h["image_mask"], h["cos"], h["sin"], h["bias"]
            )
        else:
            st = self._device_tables(batch, bucket)
            grid = batch["image_grid_thw"].view(-1, 3)
            patches = d(
                batch["pixel_values"]
                .view(grid.shape[0], int(grid[0, 1] * grid[0, 2]), -1)
                .to(self.dtype)
            )
            t1 = time.time()
            x = self.vision_embed(
                patches,
                st["pe_idx"],
                st["pe_w"],
                st["vcos"],
                st["vsin"],
                st["text_emb"],
                st["place"],
            )
            ks, vs = self.prefix(x, st["cos"], st["sin"], st["bias"])
            ks[-1].to("cpu")  # sync point for the stage timing (one 0.4 MB copy)
            # The denoise graph takes ONE stacked K and ONE stacked V. Fed as 12 separate inputs
            # (2 x 6 layers) the same graph runs 0.9 ms slower per step (10 steps: 179.3 vs 171.0 ms);
            # the two stacks cost about 2 ms once per request.
            ks, vs = torch.stack(ks), torch.stack(vs)
        t2 = time.time()
        x = d(noise.float(), exact=True)  # fp32 Euler accumulator, as upstream
        x, traj = self._denoise(st, x, ks, vs, return_trajectory)
        out = self._sync(x)
        t3 = time.time()
        self.timings = {
            "host_prep_s": t1 - t0,
            "prefix_s": t2 - t1,
            "denoise_s": t3 - t2,
            "total_s": t3 - t0,
        }
        out = out[:, :, : self.cfg.policy.action_dim]
        return (out, traj) if return_trajectory else out
