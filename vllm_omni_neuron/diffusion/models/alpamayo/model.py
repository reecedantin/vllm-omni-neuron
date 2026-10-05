# SPDX-License-Identifier: Apache-2.0
"""The full Alpamayo model (1.5 and 2 Super) for Neuron: VLM prefill -> fixed-length
autoregressive decode -> flow-matching expert, tying ``backbone.py`` and ``action_head.py`` together
via ``prep.py``'s host math.

1. **Stage 1 (VLM rollout).** Fuse the ego-history trajectory tokens into the prompt, prefill it
   (bucket-padded) through :class:`~.backbone.Gr2TextTower` into fixed-length per-layer KV caches,
   then decode greedily one token per call of the same graph, stopping as upstream's ``generate``
   does: one token after ``<traj_future_start>``, on the backbone's EOS ids, or at
   ``max_new_tokens`` (1.5: ``tokens_per_future_traj`` = 128; 2 Super: 256, with the text EOS
   masked so only ``<traj_future_start>`` ends the reasoning). ``offset`` mirrors upstream's
   ``_find_eos_offset``. Greedy decoding (upstream defaults to nucleus sampling) gives a
   deterministic parity target.
2. **Stage 2 (flow-matching expert).** ``num_inference_steps`` (10) Euler steps over
   ``linspace(0, 1, steps+1)``: project the current noisy action through
   :class:`~.action_head.ActionInProj`, run one :class:`~.action_head.Expert` step over the VLM's
   cache (masked as upstream's ``_build_expert_pos_ids_and_attn_mask``) plus its own K/V, project
   through ``action_out_proj``, Euler-update in fp32 on the host.

Weight loading maps the checkpoint's flat ``vlm.model.*`` / ``vlm.lm_head.*`` / ``expert.*`` /
``action_in_proj.*`` / ``action_out_proj.*`` prefixes (Alpamayo 1.5) or ``expert.expert.*`` /
``expert.action_in_proj.*`` / ``expert.action_out_proj.*`` (Alpamayo 2 Super) onto this module tree.
"""

from __future__ import annotations

import glob
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from .action_head import ActionInProj, Expert, fourier_freqs
from .backbone import Gr2TextTower, Gr2VisionTower, _tp_state
from .config import AlpamayoConfig
from .decode_backend import DecodeConfig
from .prep import (
    MASK_VALUE,
    BackbonePrep,
    decode_masks,
    expert_inputs,
    fuse_history_tokens,
    prefill_bias,
    rope_cos_sin,
)

logger = logging.getLogger(__name__)

# Prompt-length buckets (one prefill graph each; the largest also sets the decode / expert cache
# length). Alpamayo 1.5's 4-camera prompt is ~2.9k tokens; Alpamayo 2 Super's 6-camera ~4.6k.
DEFAULT_TEXT_BUCKETS = {"alpamayo1_5": "1024,2048,3072", "alpamayo2_super": "4608"}
TEXT_BUCKETS = tuple(
    int(b) for b in os.environ.get("ALPAMAYO_TEXT_BUCKETS", "1024,2048,3072").split(",")
)


def text_buckets(variant: str) -> tuple[int, ...]:
    spec = os.environ.get("ALPAMAYO_TEXT_BUCKETS") or DEFAULT_TEXT_BUCKETS.get(
        variant, "1024,2048,3072"
    )
    return tuple(sorted(int(b) for b in spec.split(",")))


# the backbone's generation_config.json eos ids (<|im_end|>, <|endoftext|>), honoured by upstream's generate
GENERATION_EOS_IDS = (151645, 151643)


def pick_bucket(n: int, buckets=TEXT_BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(
        f"prompt is {n} tokens; the largest bucket is {buckets[-1]} (ALPAMAYO_TEXT_BUCKETS)"
    )


def _interleave_qkv_for_tp(w: torch.Tensor, tp_size: int) -> torch.Tensor:
    """The vision tower's FUSED qkv rows are ``[q | k | v]`` (each ``dim`` wide).
    ``ColumnParallelLinear`` shards rows contiguously, which would hand rank 0 all of ``q`` plus
    half of ``k``. Reorder to ``[q_0 k_0 v_0 | q_1 k_1 v_1 | ...]`` so rank ``r``'s contiguous shard
    is its own heads' ``[q_r | k_r | v_r]`` -- the local layout ``_VisionAttention`` reshapes to."""
    d3 = w.shape[0]
    rest = w.shape[1:]
    return (
        w.reshape(3, tp_size, d3 // 3 // tp_size, *rest)
        .transpose(0, 1)
        .reshape(d3, *rest)
        .contiguous()
    )


def _read_rank_tensor(fh, key: str, dst: str, modules: dict) -> torch.Tensor:
    """Read only this rank's shard of a column-/row-parallel weight (``safe_open.get_slice``): the
    same rows / columns the layer's own ``_load_from_state_dict`` keeps, so each rank holds ~1/tp
    of the checkpoint in host memory instead of all of it (Alpamayo 2 Super is 72 GB bf16). The
    vision tower's fused qkv is read whole: its rows are re-interleaved per rank before sharding
    (``_interleave_qkv_for_tp``)."""
    from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

    mod_name, _, leaf = dst.rpartition(".")
    mod = modules.get(mod_name)
    if getattr(mod, "tp_size", 1) > 1 and not dst.endswith((".attn.qkv.weight", ".attn.qkv.bias")):
        # weights only: ColumnParallelLinear._load_from_state_dict decides whether to shard the
        # BIAS by comparing the (already-sharded) weight's shape with the bias's, so a pre-sharded
        # bias would be sliced a second time
        if isinstance(mod, ColumnParallelLinear) and leaf == "weight":
            n = mod.out_features_per_rank
            return fh.get_slice(key)[mod.tp_rank * n : (mod.tp_rank + 1) * n]
        if isinstance(mod, RowParallelLinear) and leaf == "weight":
            n = mod.in_features_per_rank
            return fh.get_slice(key)[:, mod.tp_rank * n : (mod.tp_rank + 1) * n].contiguous()
    return fh.get_tensor(key)


def _vllm_tp_group_initialized() -> bool:
    """True inside the vLLM-Omni worker (which builds the TP group register_replica_groups reads);
    False under a bare torch.distributed harness such as the CPU TP=2 check."""
    from vllm.distributed import parallel_state as ps

    return getattr(ps, "_TP", None) is not None


class NeuronAlpamayo1_5(nn.Module):
    def __init__(self, cfg: AlpamayoConfig, dtype: torch.dtype = torch.bfloat16, tp_group=None):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.tp_group = tp_group
        # Make the TP group resolvable to its full replica-group partition so the torch-native
        # Neuron (Lite) backend can legalize the ColumnParallelLinear/RowParallelLinear
        # all-reduce/all-gather collectives at SPMD compile time -- without this the first
        # collective in a forward() raises "replica id #N not seen in replica groups" (see
        # docs/design/vllm_omni_neuron_overview.md "Collectives under compilation" and
        # docs/model-dev/onboarding-models.md Common issues). CPU mode (tp_size==1) has no
        # collectives, which is why the CPU reference path never exercises this. cp_size=1:
        # Alpamayo is TP-only, no context parallelism.
        tp_size = _tp_state(tp_group)[0]
        if tp_size > 1 and _vllm_tp_group_initialized():
            from vllm_omni_neuron.diffusion.distributed.parallel_state import (
                register_replica_groups,
            )

            register_replica_groups(tp_size=tp_size, cp_size=1)
        self.text_buckets = text_buckets(cfg.variant)
        self.vision = Gr2VisionTower(cfg.backbone["vision_config"], tp_group)
        self.text = Gr2TextTower(cfg.backbone["text_config"], tp_group)
        # Vocab-parallel LM head under TP: each rank holds 1/tp of the (zero-padded) vocab rows and
        # the logits are all-gathered. Decode reads every lm_head row per token, so a replicated
        # 155k x 4096 head is 1.27 GB of per-token weight traffic on every rank.
        self.vocab = int(cfg.backbone["text_config"]["vocab_size"])
        self.vocab_parallel = tp_size > 1 and bool(
            int(os.environ.get("ALPAMAYO_VOCAB_PARALLEL", "1"))
        )
        # Shards are rounded up to 512 rows: the runtime's mesh collectives reject odd per-rank
        # payloads ("invalid input block size" in the logits all-gather at TP=4, vocab 155,697).
        self.vocab_shard = (
            -(-self.vocab // (tp_size * 512)) * 512 if self.vocab_parallel else self.vocab
        )
        self.lm_head_weight = nn.Parameter(
            torch.empty(self.vocab_shard, cfg.backbone["text_config"]["hidden_size"])
        )
        ec = dict(cfg.backbone["text_config"])
        # the expert's own widths (1.5: expert_cfg; 2 Super: expert_config.llm_config) over the
        # backbone's text config; K/V heads and head_dim must equal the VLM's (shared cache)
        ec.update(cfg.expert)
        self.expert = Expert(ec, tp_group)
        aip = cfg.head["action_in_proj_cfg"]
        self.action_in_proj = ActionInProj(
            action_dim=2,
            out_dim=ec["hidden_size"],
            num_enc_layers=int(aip.get("num_enc_layers", 4)),
            hidden_size=int(aip.get("hidden_size", 1024)),
            max_freq=float(aip.get("max_freq", 100.0)),
            num_fourier_feats=int(aip.get("num_fourier_feats", 20)),
            bf16_freqs=bool(cfg.head.get("fourier_freqs_bf16", True)),
        )
        self.action_out_proj = nn.Linear(ec["hidden_size"], 2)
        self._action_space: UnicycleAccelCurvatureActionSpace | None = None
        self.n_waypoints = cfg.n_waypoints
        self.num_inference_steps = int(cfg.head["diffusion_cfg"].get("num_inference_steps", 10))
        self.device = torch.device("cpu")
        self._fns: dict = {}
        self.trace = bool(int(os.environ.get("ALPAMAYO_TRACE", "0")))  # keep per-step expert inputs
        self.profile = bool(int(os.environ.get("ALPAMAYO_PROFILE", "0")))  # time vision separately
        # build decode RoPE + masks inside the decode graph (vs per-step on the host)
        self.decode_in_graph = bool(int(os.environ.get("ALPAMAYO_DECODE_IN_GRAPH", "1")))
        self._prep_ms = 0.0
        self._vision_ms = None
        self.stats: dict = {}
        self._tokenizer = None
        self._prep: BackbonePrep | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.cfg.build_tokenizer()
        return self._tokenizer

    @property
    def traj_future_start_id(self) -> int:
        """``<|traj_future_start|>`` (upstream's ``StopAfterEOS`` token), from the checkpoint's own
        ``traj_token_ids`` -- no tokenizer needed to run the rollout."""
        ids = self.cfg.extra.get("traj_token_ids") or {}
        if "future_start" in ids:
            return int(ids["future_start"])
        return int(self.tokenizer.convert_tokens_to_ids("<|traj_future_start|>"))

    @property
    def hf_cfg(self):
        """The backbone's ``Qwen3VLConfig``, built once (constructing it costs ~3 ms)."""
        if getattr(self, "_hf_cfg", None) is None:
            self._hf_cfg = self.cfg.hf_backbone_config()
        return self._hf_cfg

    @property
    def prep(self) -> BackbonePrep:
        if self._prep is None:
            self._prep = BackbonePrep(self.hf_cfg)
        return self._prep

    @property
    def action_space(self) -> UnicycleAccelCurvatureActionSpace:
        """Not a submodule: it has no learned weights and must not participate in the
        meta-device build / load_weights pass that NeuronAlpamayo1_5 itself goes through."""
        if self._action_space is None:
            asc = dict(self.cfg.head["action_space_cfg"])
            self._action_space = UnicycleAccelCurvatureActionSpace(
                accel_mean=float(asc.get("accel_mean", 0.0)),
                accel_std=float(asc.get("accel_std", 1.0)),
                curvature_mean=float(asc.get("curvature_mean", 0.0)),
                curvature_std=float(asc.get("curvature_std", 1.0)),
                accel_bounds=tuple(asc.get("accel_bounds", (-9.8, 9.8))),
                curvature_bounds=tuple(asc.get("curvature_bounds", (-0.33, 0.33))),
                dt=float(asc.get("dt", 0.1)),
                n_waypoints=int(asc.get("n_waypoints", 64)),
                theta_lambda=float(asc.get("theta_lambda", 1e-6)),
                theta_ridge=float(asc.get("theta_ridge", 1e-8)),
                v_lambda=float(asc.get("v_lambda", 1e-6)),
                v_ridge=float(asc.get("v_ridge", 1e-4)),
                a_lambda=float(asc.get("a_lambda", 1e-4)),
                a_ridge=float(asc.get("a_ridge", 1e-4)),
                kappa_lambda=float(asc.get("kappa_lambda", 1e-4)),
                kappa_ridge=float(asc.get("kappa_ridge", 1e-4)),
            )
        return self._action_space

    # -- weights -----------------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls, model_dir: str, dtype: torch.dtype = torch.bfloat16, device="cpu", tp_group=None
    ) -> NeuronAlpamayo1_5:
        with torch.device("meta"):
            m = cls(AlpamayoConfig.from_model_dir(model_dir), dtype=dtype, tp_group=tp_group)
        m.load_weights(model_dir)
        return m.to(device)

    def load_weights(self, model_dir: str) -> None:
        from safetensors import safe_open

        t0 = time.time()
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {model_dir}")
        state: dict[str, torch.Tensor] = {}
        modules = dict(self.named_modules())
        for f in files:
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    dst = self._map_key(k)
                    if dst is not None:
                        state[dst] = _read_rank_tensor(fh, k, dst, modules).to(self.dtype)
        # vllm_neuron's RowParallelLinear requires an fp32 bias under TP>1 (its forward adds the
        # bias after the all-reduce, and the XLA lowering only handles fp32 there -- see
        # backbone._row_parallel_biased, which builds exactly those layers in fp32 under TP).
        # assign=True adopts the checkpoint dtype, so without help the bias would reload as bf16
        # and the layer would reject it. Preserve fp32 for ONLY those layers' biases -- the vision
        # attention output proj and MLP down-proj -- and only under TP; every other bias (e.g. the
        # patch-embed conv, whose weight loads bf16 and whose forward expects a matching bias) stays
        # bf16. Scoped by key suffix so a stray fp32 meta param elsewhere is not accidentally pinned.
        _RPL_BIAS = (".attn.proj.bias", ".mlp.linear_fc2.bias")
        tp_size = _tp_state(self.tp_group)[0]
        if tp_size > 1:
            for key in list(state):
                if key.endswith(_RPL_BIAS):
                    state[key] = state[key].to(torch.float32)
                elif key.startswith("vision.") and key.endswith(
                    (".attn.qkv.weight", ".attn.qkv.bias")
                ):
                    state[key] = _interleave_qkv_for_tp(state[key], tp_size)
        if self.vocab_parallel and "lm_head_weight" in state:
            w = state["lm_head_weight"]
            rank = _tp_state(self.tp_group)[1]
            w = F.pad(w, (0, 0, 0, self.vocab_shard * tp_size - w.shape[0]))
            state["lm_head_weight"] = w[
                rank * self.vocab_shard : (rank + 1) * self.vocab_shard
            ].clone()
        missing, unexpected = self.load_state_dict(state, strict=False, assign=True)
        remaining_missing = [
            m for m in missing if "freqs" not in m
        ]  # RoPE/Fourier: computed, not stored
        if remaining_missing or unexpected:
            raise RuntimeError(
                f"Alpamayo weight load mismatch: missing={remaining_missing[:8]} "
                f"({len(remaining_missing)}), unexpected={unexpected[:8]} ({len(unexpected)})"
            )
        # `assign=True` leaves any buffer NOT in the checkpoint (the Fourier "freqs" tables above,
        # a deterministic function of (dim, max_freq), never saved) on the meta device the module
        # was built on -- rebuild them for real so .to(device) can copy out of them.
        for module in self.modules():
            if hasattr(module, "freqs") and module.freqs.is_meta:
                module.freqs = fourier_freqs(
                    2 * module.freqs.shape[-1], module.max_freq, module.bf16_freqs
                )
        self.stats["load_s"] = time.time() - t0
        self.stats["weights_gb"] = (
            sum(p.numel() * p.element_size() for p in self.parameters()) / 1e9
        )
        logger.info(
            "Alpamayo weights: %d tensors in %.1fs, %.2f GB on this rank (tp=%d)",
            len(state),
            self.stats["load_s"],
            self.stats["weights_gb"],
            tp_size,
        )

    @staticmethod
    def _map_key(key: str) -> str | None:
        if key.startswith("expert.action_space.") or key.endswith(".freqs"):
            return None  # derived buffers (normalization constants from config.json, Fourier freqs)
        for super_prefix, dst in (  # Alpamayo 2 Super nests the expert's parts under "expert."
            ("expert.expert.", "expert."),
            ("expert.action_in_proj.", "action_in_proj."),
            ("expert.action_out_proj.", "action_out_proj."),
        ):
            if key.startswith(super_prefix):
                return dst + key[len(super_prefix) :]
        if key == "vlm.lm_head.weight":
            return "lm_head_weight"
        if key.startswith("vlm.model.visual."):
            return "vision." + key[len("vlm.model.visual.") :]
        if key.startswith("vlm.model.language_model."):
            return "text." + key[len("vlm.model.language_model.") :]
        if key.startswith(("expert.", "action_in_proj.", "action_out_proj.")):
            return key  # already match this module tree's own names
        raise KeyError(f"unexpected Alpamayo checkpoint tensor {key!r}")

    def to(self, *args, **kwargs):
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            super().to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            super().to(device=self.device)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> NeuronAlpamayo1_5:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {
                **base,
                "model_name": name,
                "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
            }

        # One compiled graph per entry point: vision; prefill (+ lm_head at the last real token);
        # decode (embedding + layers + one-hot cache write + lm_head); expert step (action_in_proj
        # + expert + action_out_proj).
        self._fns["vision"] = torch.compile(
            self.vision, backend=backend, options=opts("alpamayo_vision"), **kw
        )
        for name in ("g_prefill", "g_decode", "g_decode_pos", "g_expert"):
            self._fns[name] = torch.compile(
                getattr(self, name), backend=backend, options=opts(f"alpamayo_{name[2:]}"), **kw
            )
        return self

    def _fn(self, name: str, mod):
        fn = self._fns.get(name)
        return mod if fn is None else fn

    # -- stage 1a: prefill ---------------------------------------------------------------------
    @property
    def cache_len(self) -> int:
        """Fixed KV-cache length shared by EVERY request: the largest prompt bucket plus the
        generation cap. One length means one decode graph and one expert graph in total."""
        return max(self.text_buckets) + self.cfg.max_new_tokens

    def fuse_history(
        self, input_ids: torch.Tensor, ego_history_xyz, ego_history_rot
    ) -> torch.Tensor:
        """Upstream ``fuse_traj_tokens``: write the history-trajectory tokens into the prompt's
        ``<|traj_history|>`` placeholders (the processor emits placeholders only)."""
        if ego_history_xyz is None or ego_history_rot is None:
            return input_ids
        return fuse_history_tokens(input_ids, ego_history_xyz, ego_history_rot, self.cfg)

    # -- device graphs ------------------------------------------------------------------------
    # Every op that touches a device tensor runs inside one of these graphs (plus the vision graph):
    # on Neuron an EAGER dtype cast / copy_ / index_put_ on a device tensor fails ("Expected
    # self.dtype() == dst.dtype()") even though CPU accepts it. The host only builds inputs (in their
    # final dtype, then a pure .to(device)) and reads outputs back with .cpu().

    def _lm_head(self, hidden: torch.Tensor, logit_bias: torch.Tensor) -> torch.Tensor:
        """``lm_head(norm(h)) + logit_bias`` (fp32 result); ``logit_bias`` masks the discrete
        trajectory-token range (upstream's ``ExpertLogitsProcessor``). The matmul runs in the model
        dtype like upstream's ``nn.Linear`` lm_head: upcasting the 155k x 4096 weight to fp32 inside
        the graph streamed 3x its bf16 bytes on every decode step. Under vocab parallelism each rank
        computes its own vocab shard and the shards are all-gathered (padding rows dropped)."""
        logits = F.linear(self.text.norm(hidden), self.lm_head_weight).float() + logit_bias
        if not self.vocab_parallel:
            return logits
        from torch.distributed._functional_collectives import all_gather_tensor

        tp_size, _, group = _tp_state(self.tp_group)
        b, s, _ = logits.shape
        full = all_gather_tensor(logits.reshape(1, b * s * self.vocab_shard), 0, group)
        full = full.reshape(tp_size, b * s, self.vocab_shard).transpose(0, 1)
        return full.reshape(b, s, tp_size * self.vocab_shard)[..., : self.vocab]

    def g_prefill(
        self,
        input_ids,
        image_index,
        image_keep,
        image_embeds,
        cos,
        sin,
        bias,
        last_idx,
        logit_bias,
        cfg: DecodeConfig,
        deepstack: tuple,
    ):
        hidden, ks, vs = self.text.do_prefill(
            input_ids, image_index, image_keep, image_embeds, cos, sin, bias, cfg, deepstack
        )
        return self._lm_head(hidden.index_select(1, last_idx), logit_bias)[:, 0], ks, vs

    def g_decode(self, token, cos, sin, ks, vs, write_mask, bias, logit_bias, cfg: DecodeConfig):
        embed = self.text.embed_tokens(token)  # [1, 1, hidden], model dtype
        hidden, ks, vs = self.text.do_decode(embed, cos, sin, ks, vs, write_mask, bias, cfg)
        return self._lm_head(hidden, logit_bias)[:, 0], ks, vs

    def g_decode_pos(self, token, slot_pos, inv_freq, ks, vs, logit_bias, cfg: DecodeConfig):
        """``g_decode`` with the per-step host work moved in-graph: ``slot_pos`` (long ``[2]``) is
        ``[cache slot, mRoPE position]``; the RoPE angles (pure text: all three mRoPE axes equal, so
        Qwen3-VL's rotary reduces to ``cos/sin(pos * cat(inv_freq, inv_freq))``), the one-hot write
        mask and the causal bias are built here. Only two tiny tensors cross the host boundary."""
        dt = self.text.embed_tokens.weight.dtype
        ang = slot_pos[1:2].float()[:, None] * inv_freq[None, :]  # [1, head_dim], fp32
        cos, sin = ang.cos().to(dt)[None], ang.sin().to(dt)[None]  # [1, 1, head_dim]
        cols = torch.arange(cfg.max_len, device=slot_pos.device)
        slot = slot_pos[0]
        write_mask = (cols == slot)[None, None, :, None]
        bias = torch.where(cols <= slot, 0.0, MASK_VALUE).to(torch.float32)[None, None, None]
        return self.g_decode(token, cos, sin, ks, vs, write_mask, bias, logit_bias, cfg)

    def _inv_freq(self) -> torch.Tensor:
        """``cat(inv_freq, inv_freq)`` fp32 ``[head_dim]``, built on the host and moved once."""
        if getattr(self, "_inv_freq_dev", None) is None:
            tc = self.hf_cfg.text_config
            rp = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None) or {}
            theta = float(rp.get("rope_theta", getattr(tc, "rope_theta", 5e6)))
            hd = int(tc.head_dim)
            inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
            self._inv_freq_dev = torch.cat([inv, inv]).to(self.device)
        return self._inv_freq_dev

    def g_expert(self, actions, t, cos, sin, ks, vs, bias):
        """One Euler step's velocity: fp32 actions ``[1, n, 2]`` + time ``[1, 1, 1]`` -> fp32 ``v``."""
        embeds = self.action_in_proj(actions, t)
        out = self.expert(embeds, cos, sin, ks, vs, bias)
        return self.action_out_proj(out).float(), embeds

    def _g(self, name: str):
        return self._fns.get(name) or getattr(self, name)

    def _logit_bias(self) -> torch.Tensor:
        """``[1, vocab]`` fp32 (this rank's ``[1, vocab_shard]`` slice under vocab parallelism),
        built on the host and moved once."""
        if getattr(self, "_logit_bias_dev", None) is None:
            tp_size, rank, _ = _tp_state(self.tp_group)
            n = self.vocab_shard * tp_size if self.vocab_parallel else self.vocab
            b = torch.zeros(1, n, dtype=torch.float32)
            start = int(self.cfg.head["traj_token_start_idx"])
            b[:, start : start + int(self.cfg.head["traj_vocab_size"])] = -1e30
            # Alpamayo 2 Super masks the backbone's text EOS (upstream _append_text_eos_mask)
            for tid in self.cfg.head.get("masked_token_ids", ()):
                b[:, int(tid)] = -1e30
            if self.vocab_parallel:
                b[:, self.vocab :] = -1e30
                b = b[:, rank * self.vocab_shard : (rank + 1) * self.vocab_shard].contiguous()
            self._logit_bias_dev = b.to(self.device)
        return self._logit_bias_dev

    @torch.no_grad()
    def _prefill(self, input_ids, attention_mask, pixel_values, image_grid_thw, bucket: int):
        tp = time.perf_counter()
        bi = self.prep(input_ids, attention_mask, pixel_values, image_grid_thw, bucket=bucket)
        self._prep_ms = 1e3 * (time.perf_counter() - tp)  # host-side mRoPE / vision tables
        dev, dt = self.device, self.dtype

        def d(x, dtype=None):  # host-side cast first, then a pure move to the device
            return (x.to(dtype) if dtype is not None else x).contiguous().to(dev)

        vis = self._fn("vision", self.vision)(
            d(bi.pixels, dt),
            d(bi.pos_index),
            d(bi.pos_weight, torch.float32),
            d(bi.vis_cos, torch.float32),
            d(bi.vis_sin, torch.float32),
            bi.n_images,
        )
        image_embeds, deepstack = vis[0], tuple(vis[1:])
        if self.profile:  # split vision from text prefill (costs one extra device sync)
            image_embeds[:1, :1].cpu()
            self._vision_ms = 1e3 * (time.perf_counter() - tp) - self._prep_ms
        cfg = self.text.decode_config(max_len=self.cache_len, dtype=dt)
        logits, ks, vs = self._g("g_prefill")(
            d(bi.input_ids),
            d(bi.image_index),
            d(bi.image_keep),
            image_embeds,
            d(bi.txt_cos, dt),
            d(bi.txt_sin, dt),
            d(prefill_bias(bi.real_len, bucket)),
            d(torch.tensor([bi.real_len - 1])),
            self._logit_bias(),
            cfg,
            deepstack,
        )
        logits = logits.cpu()
        self._debug = {
            "image_embeds": image_embeds,
            "deepstack": deepstack,
            "prefill_logits_last": logits,
        }
        return cfg, ks, vs, logits, bi

    # -- stage 1b: autoregressive decode ------------------------------------------------------
    @torch.no_grad()
    def _rollout(self, cfg, ks, vs, logits, bi, force_tokens: torch.Tensor | None = None):
        """Greedy decode through ONE fixed-shape graph (the full ``cfg.max_len`` cache, an additive
        mask for unfilled slots, a one-hot write mask). Mirrors upstream's ``generate`` call: stop one
        token AFTER ``<traj_future_start>`` (``StopAfterEOS``), on the backbone's own EOS ids, or at
        ``max_new_tokens``; the cache then holds every generated token except the last.
        ``force_tokens`` (``[N]``) teacher-forces the rollout for parity tests.

        Returns ``(generated [N], offset, cache_valid, ks, vs)``: ``offset`` is upstream's
        ``_find_eos_offset`` (sequence index right after ``<traj_future_start>``, else the sequence
        length) and ``cache_valid`` the number of live cache slots (prompt + N - 1)."""
        traj_eos = self.traj_future_start_id
        stop_ids = set(self.cfg.head.get("stop_token_ids", GENERATION_EOS_IDS))
        n_max = self.cfg.max_new_tokens if force_tokens is None else int(force_tokens.numel())
        real = bi.real_len
        delta = int(bi.rope_deltas.reshape(-1)[0])
        hf_cfg = self.hf_cfg
        logit_bias = self._logit_bias()
        gen: list[int] = []
        tok = int(force_tokens[0]) if force_tokens is not None else int(logits.argmax(-1)[0])
        while True:
            gen.append(tok)
            if len(gen) >= n_max or tok in stop_ids or (len(gen) >= 2 and gen[-2] == traj_eos):
                break
            p = real + len(gen) - 1  # cache slot == sequence index of the token being fed
            token = torch.tensor([[tok]], dtype=torch.long).to(self.device)
            if self.decode_in_graph:
                slot_pos = torch.tensor([p, p + delta], dtype=torch.long).to(self.device)
                logits, ks, vs = self._g("g_decode_pos")(
                    token, slot_pos, self._inv_freq(), ks, vs, logit_bias, cfg
                )
            else:
                cos, sin = rope_cos_sin(
                    hf_cfg, torch.full((3, 1), p + delta, dtype=torch.long), self.dtype, self.device
                )
                write_mask, bias = decode_masks(p, cfg.max_len, self.device)
                logits, ks, vs = self._g("g_decode")(
                    token, cos, sin, ks, vs, write_mask, bias, logit_bias, cfg
                )
            tok = (
                int(force_tokens[len(gen)])
                if force_tokens is not None
                else int(logits.cpu().argmax(-1)[0])
            )
        generated = torch.tensor(gen, dtype=torch.long)
        hits = (generated == traj_eos).nonzero()
        offset = real + (int(hits[0, 0]) + 1 if hits.numel() else len(gen))
        return generated, offset, real + len(gen) - 1, ks, vs

    # -- stage 2: flow-matching expert --------------------------------------------------------
    @torch.no_grad()
    def _denoise(self, ks, vs, offset: int, cache_valid: int, bi, generator=None, noise=None):
        """Euler integration on the HOST in fp32 (upstream's state dtype); each step's velocity comes
        from the expert graph."""
        n_wp = self.n_waypoints
        if noise is None:
            # fp32 on the host CPU generator, as upstream's torch.randn(...) on a CPU device
            noise = torch.randn((1, n_wp, 2), dtype=torch.float32, generator=generator)
        actions = noise.to(torch.float32).cpu()
        steps = self.num_inference_steps
        time_steps = torch.linspace(0.0, 1.0, steps + 1)
        delta = int(bi.rope_deltas.reshape(-1)[0])
        cos, sin, bias = expert_inputs(
            self.hf_cfg, offset, cache_valid, delta, n_wp, self.cache_len, self.dtype, self.device
        )
        trace: dict[str, list] = {"action_in": [], "v": []}
        for i in range(steps):
            t = time_steps[i].reshape(1, 1, 1).to(self.device)  # upstream: [B, 1, 1]
            pred, embeds = self._g("g_expert")(actions.to(self.device), t, cos, sin, ks, vs, bias)
            pred = pred.cpu()
            if self.trace:  # parity tooling only: an extra device->host copy per step
                trace["action_in"].append(embeds.cpu())
            trace["v"].append(pred)
            actions = actions + float(time_steps[i + 1] - time_steps[i]) * pred
        self._debug_steps = trace
        return actions

    @torch.no_grad()
    def get_action(
        self,
        inputs: dict,
        generator: torch.Generator | None = None,
        bucket: int | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        force_tokens: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict:
        real = int(inputs["attention_mask"].sum())
        bucket = bucket or pick_bucket(real, self.text_buckets)
        t0 = time.perf_counter()
        input_ids = self.fuse_history(inputs["input_ids"], ego_history_xyz, ego_history_rot)
        cfg, ks, vs, logits, bi = self._prefill(
            input_ids,
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            bucket=bucket,
        )
        t1 = time.perf_counter()  # prefill ends in logits.cpu(), a device sync
        generated, offset, cache_valid, ks, vs = self._rollout(
            cfg, ks, vs, logits, bi, force_tokens
        )
        t2 = time.perf_counter()  # every decode step syncs on its logits
        actions = self._denoise(ks, vs, offset, cache_valid, bi, generator=generator, noise=noise)
        t3 = time.perf_counter()  # every expert step syncs on its velocity
        n_dec = int(generated.numel()) - 1
        timing = {
            "prep_ms": self._prep_ms,
            "vision_ms": self._vision_ms,
            "prefill_ms": 1e3 * (t1 - t0),
            "decode_ms": 1e3 * (t2 - t1),
            "decode_tokens": n_dec,
            "decode_ms_per_token": 1e3 * (t2 - t1) / max(n_dec, 1),
            "expert_ms": 1e3 * (t3 - t2),
            "expert_steps": self.num_inference_steps,
        }
        out = {
            "actions": actions.float().cpu(),
            "generated": generated,
            "offset": offset,
            "prompt_len": real,
            "n_cot_tokens": offset - real,
            "timing_ms": timing,
        }
        if ego_history_xyz is not None and ego_history_rot is not None:
            # upstream: action_to_traj(sampled_action, ego_history_xyz[:, -1], ego_history_rot[:, -1])
            pred_xyz, pred_rot = self.action_space.action_to_traj(
                actions.float().cpu(),
                ego_history_xyz[:, -1].float(),
                ego_history_rot[:, -1].float(),
            )
            out["pred_xyz"] = pred_xyz
            out["pred_rot"] = pred_rot
        return out
