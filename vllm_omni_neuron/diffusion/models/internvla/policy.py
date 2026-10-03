# SPDX-License-Identifier: Apache-2.0
"""InternVLA-A1.5 policy (Qwen3.5-2B VLM + unified action expert + flow-matching head).

Inference is upstream ``InternVLAA15.sample_actions``: the VLM encodes the prefix (camera
views + prompt) once, and the action expert -- a 24-layer Qwen3.5 decoder with the VLM's layer
pattern but hidden 1024 -- runs ``num_inference_steps`` Euler steps over a 100-token suffix
(50 learnable foresight tokens + a 50-step action chunk). The two towers only meet in the 6
gated full-attention layers, where suffix queries attend to ``[prefix K/V, suffix K/V]``; the
18 Gated-DeltaNet layers of the expert never see prefix state (upstream overwrites it). So
the prefix hands the expert exactly 6 (K, V) pairs, and nothing else.

On Neuron that is three fixed-shape graphs, each compiled once per shape bucket:

* :class:`VisionGraph`   pixels -> merged image tokens (all views batched)
* :class:`PrefixGraph`   tokens + image tokens -> 6 x (K, V)
* :class:`DenoiseGraph`  one Euler step ``x_{t+dt} = x_t + dt * v(x_t, t)``

The WAN video branch is training-only and is not built; its projection in the checkpoint is
skipped. Module names match the checkpoint (``model.`` prefix stripped).
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import preprocess as pp
from .config import InternVLAConfig
from .qwen3_5 import Qwen35DecoderLayer, Qwen35TextModel, Qwen35VisionModel, eager_attention
from vllm_omni_neuron.diffusion.layers.block_graphs import BlockGraphRunner

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
            cfg.vlm.text, p.action_expert_hidden_size, p.action_expert_intermediate_size, with_embeddings=False
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
    def from_pretrained(cls, model_path: str, dtype: torch.dtype = torch.bfloat16, device="cpu",
                        vlm_config: str | None = None) -> InternVLAA15:
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
                name = key[len("model."):] if key.startswith("model.") else key
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
                state[name] = nn.Parameter(t.to(td).to(device), requires_grad=False)  # cast on host first
        missing = sorted(set(expected) - set(state))
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
        self.load_state_dict(state, strict=True, assign=True)
        n = sum(t.numel() for t in state.values())
        return {"tensors": len(state), "params": n, "skipped": len(skipped), "load_s": round(time.time() - t0, 2)}


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


class PrefixEmbed(nn.Module):
    """``input_ids [B,L]`` + image slots/mask -> prefix embeddings ``[B,L,D]`` (elementwise select)."""

    def __init__(self, model: InternVLAA15):
        super().__init__()
        self.embed_tokens = model.vlm.language_model.embed_tokens

    def forward(self, input_ids, image_slots, image_mask):
        x = self.embed_tokens(input_ids)
        m = image_mask.to(x.dtype)
        return x * (1.0 - m) + image_slots.to(x.dtype) * m



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
    """Device prefix driven layer by layer. The GDN layers come in 6 contiguous runs of 3 (the
    checkpoint's [lin,lin,lin,full] x 6 pattern); each run is fused into ONE compiled graph via
    A0's :class:`BlockGraphRunner` (``group_size=3``), so the host dispatches once per run of 3
    instead of 3 separate device calls. Full-attention layers (needed individually for their K/V)
    keep one shared template graph, called once per layer. Measured: 18 separate GDN calls cost
    99ms (5.5ms/call, dispatch-bound -- see GDN_CHUNK_SIZE's comment); fusing 3-at-a-time cuts the
    dispatch count 3x without touching the chunk-rule math (so no compile-time blowup, unlike the
    chunk_size=128 attempt). Returns the stacked full-layer K/V exactly like :class:`PrefixGraph`."""

    def __init__(self, model: InternVLAA15, compile_fn):
        self.lm = model.vlm.language_model
        self.full = model.cfg.vlm.text.full_attention_layers
        self.last = self.full[-1]
        wrap = compile_fn or (lambda mod, name: mod)
        # The embedding lookup + select is a single cheap op; run it eagerly on device (compiling it
        # alongside the layer graphs tripped the Lite backend's "doesn't support events" path).
        self.embed = PrefixEmbed(model)
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
            runner = BlockGraphRunner(run, group_size=len(run), compile_fn=lambda f: wrap(f, "internvla_prefix_linear"))
            self.lin_runs.append((i, runner))
            i = j
        self._run_by_start = dict(self.lin_runs)

    @staticmethod
    def _lin_weights(layer):
        sub = (layer.input_layernorm, layer.linear_attn, layer.post_attention_layernorm, layer.mlp)
        prefixes = ("input_layernorm", "linear_attn", "post_attention_layernorm", "mlp")
        out = {}
        for pre, mod in zip(prefixes, sub):
            for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                out[f"{pre}.{n}"] = t.detach()
        return out

    @staticmethod
    def _full_weights(layer):
        sub = (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)
        prefixes = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        out = {}
        for pre, mod in zip(prefixes, sub):
            for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                out[f"{pre}.{n}"] = t.detach()
        return out

    def __call__(self, input_ids, image_slots, image_mask, cos, sin, bias):
        x = self.embed(input_ids, image_slots, image_mask)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        ks, vs = [], []
        layers = self.lm.layers[: self.last + 1]
        i = 0
        while i < len(layers):
            if layers[i].layer_type == "linear_attention":
                runner = self._run_by_start[i]
                x = runner(x)
                i += runner.group_size
            else:
                x, k, v = self.fullg(x, cos, sin, bias, self._full_weights(layers[i]))
                ks.append(k)
                vs.append(v)
                i += 1
        return torch.stack(ks), torch.stack(vs)


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
        x = self.expert.norm(x)[:, -self.chunk:]
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
    """Host-side ``sample_actions``: builds every table on the CPU and calls the three graphs.

    ``device`` is ``cpu`` (eager reference) or a Neuron device; ``compile_fn(module, name)``
    returns the callable to use for each graph (identity for eager).
    """

    PREFIX_BUCKETS = tuple(int(b) for b in os.environ.get("INTERNVLA_PREFIX_BUCKETS", "256,384,512,768,1024").split(","))

    def __init__(self, model: InternVLAA15, device="cpu", compile_fn=None):
        self.model = model
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.dtype = model.action_in_proj.weight.dtype
        wrap = compile_fn or (lambda mod, name: mod)
        self.vision = wrap(VisionGraph(model), "internvla_vision")
        # Whole-prefix single graph overflows the Neuron compiler (NCC_ITEN406); on device drive the
        # prefix layer by layer (one small compiled graph per layer type). On CPU the single graph is
        # fine and simpler, and it is what the parity tests exercise.
        if self.device.type == "cpu":
            self.prefix = wrap(PrefixGraph(model), "internvla_prefix")
        else:
            self.prefix = SplitPrefix(model, compile_fn)
        self.denoise = wrap(DenoiseGraph(model), "internvla_denoise")
        self.timings: dict[str, float] = {}

    def _dev(self, t, exact: bool = False):
        """Host->device copy. Host tables (rope cos/sin, additive biases, the time embedding) are
        cast to the model dtype on the host first -- the graphs were compiled for that signature and
        up-cast internally where they need fp32, as the Cosmos3-Edge device path does for its rope
        tables. (The Neuron runtime rejects a dtype change *during* the copy, ``.to(dev, dtype)``;
        a plain fp32 copy is fine.) ``exact=True`` keeps the dtype: the flow-matching state ``x_t``
        and the step ``dt`` must stay fp32, as in upstream ``sample_actions`` (it casts ``x_t`` to
        the model dtype only for the velocity input and keeps the Euler accumulator in fp32).
        Rounding them to bf16 costs 3.7x the CPU-bf16 error per step (``dt=-0.1`` becomes
        ``-0.10009766``, a 0.1% step-size bias, and ``x_t`` loses ~0.4% per element)."""
        if not exact and t.is_floating_point() and t.dtype != self.dtype and self.device.type != "cpu":
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
        if not all(int(t) == 1 for t in grid[:, 0]) or len({(int(h), int(w)) for _, h, w in grid.tolist()}) != 1:
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
        temb, dt = pp.time_embedding_table(cfg.policy, cfg.policy.action_expert_hidden_size, self.dtype, b)
        return {
            "patches": patches, "pe_idx": pe_idx, "pe_w": pe_w, "vcos": vcos, "vsin": vsin,
            "input_ids": ids_p, "img_pos": img_pos, "image_mask": image_mask, "cos": cos, "sin": sin,
            "bias": pp.prefix_bias(pad),
            "scos": scos, "ssin": ssin, "sbias": pp.suffix_bias(pad, cfg.policy, fast),
            "temb": temb, "dt": dt.reshape(1), "batch": b, "length": length, "m_img": m_img,
        }

    @torch.inference_mode()
    def sample_actions(self, batch: dict, noise: torch.Tensor, bucket: int | None = None,
                       return_trajectory: bool = False):
        t0 = time.time()
        h = self.prepare(batch, bucket)
        d = self._dev
        t1 = time.time()
        img = self.vision(d(h["patches"].to(self.dtype)), d(h["pe_idx"]), d(h["pe_w"]), d(h["vcos"]), d(h["vsin"]))
        img = img.reshape(h["batch"], h["m_img"], -1)
        # place each image token's embedding at its sequence position (upstream's masked assign);
        # done on the host because the compiled graph cannot take a strided gather
        hid = img.shape[-1]
        slots = torch.zeros(h["batch"], h["length"], hid, dtype=self.dtype)
        img_cpu = self._sync(img).to(self.dtype)
        for i in range(h["batch"]):
            slots[i].index_copy_(0, h["img_pos"][i], img_cpu[i])
        ks, vs = self.prefix(d(h["input_ids"]), d(slots), d(h["image_mask"]), d(h["cos"]), d(h["sin"]), d(h["bias"]))
        if self.device.type != "cpu":
            ks.to("cpu")  # sync point for timing only
        t2 = time.time()
        x = d(noise.float(), exact=True)  # fp32 Euler accumulator, as upstream
        dt = d(h["dt"], exact=True)
        scos, ssin, sbias = d(h["scos"]), d(h["ssin"]), d(h["sbias"])
        traj = []
        for i in range(h["temb"].shape[0]):
            x = self.denoise(x, d(h["temb"][i]), dt, scos, ssin, sbias, ks, vs)
            if return_trajectory:
                traj.append(self._sync(x).clone())
        out = self._sync(x)
        t3 = time.time()
        self.timings = {"host_prep_s": t1 - t0, "prefix_s": t2 - t1, "denoise_s": t3 - t2, "total_s": t3 - t0}
        out = out[:, :, : self.cfg.policy.action_dim]
        return (out, traj) if return_trajectory else out
