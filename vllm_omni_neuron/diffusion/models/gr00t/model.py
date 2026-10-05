# SPDX-License-Identifier: Apache-2.0
"""The full GR00T N1.7 model for Neuron: vision graph -> text graph -> action-head graph.

``NeuronGr00tModel.get_action(inputs, noise)`` takes the upstream collator's model inputs
(``input_ids``, ``attention_mask``, ``mm_token_type_ids``, ``pixel_values``,
``image_grid_thw``, ``state``, ``embodiment_id``) and returns the normalised
``action_pred [B, horizon, max_action_dim]``, the same contract as upstream
``Gr00tN1d7.get_action``. The three sub-graphs exchange device tensors only; the host
does tokenisation-side math (:class:`~.backbone.BackbonePrep`) and the noise draw.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from .action_head import Gr00tActionHead
from .backbone import BackbonePrep, Gr00tText, Gr00tVision
from .config import Gr00tConfig

logger = logging.getLogger(__name__)

# checkpoint prefix -> attribute of NeuronGr00tModel
PREFIX_MAP = (
    ("backbone.model.model.visual.", "vision."),
    ("backbone.model.model.language_model.", "text."),
    ("action_head.", "head."),
)
# present in checkpoints, unused at inference: the tied LM head (GR00T reads hidden states) and
# GR00T-H's per-embodiment state-dropout table (training only).
SKIP_KEYS = ("backbone.model.lm_head.weight", "action_head.dropout_prob_by_embodiment")

TEXT_BUCKETS = tuple(
    int(b)
    for b in os.environ.get("GR00T_TEXT_BUCKETS", "160,192,256,320,384,512,768,1024").split(",")
)

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

_DEVICE_CACHE_MAX = 32


class _Backbone(nn.Module):
    """Vision tower + text decoder as one graph (``GR00T_FUSED_BACKBONE=1``): one launch, and the
    image features never leave the graph. Holds references only; not registered on the model."""

    def __init__(self, vision: nn.Module, text: nn.Module):
        super().__init__()
        self.vision, self.text = vision, text

    def forward(
        self,
        pixels,
        pos_index,
        pos_weight,
        vis_cos,
        vis_sin,
        n_images: int,
        input_ids,
        image_index,
        image_keep,
        txt_cos,
        txt_sin,
        txt_bias,
    ):
        vis = self.vision(pixels, pos_index, pos_weight, vis_cos, vis_sin, n_images)
        return self.text(
            input_ids, image_index, image_keep, vis[0], txt_cos, txt_sin, txt_bias, *vis[1:]
        )


def pick_bucket(n: int, buckets=TEXT_BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(
        f"VLM prompt is {n} tokens; the largest bucket is {buckets[-1]} (GR00T_TEXT_BUCKETS)"
    )


def map_checkpoint_key(key: str) -> str | None:
    if key in SKIP_KEYS:
        return None
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    raise KeyError(f"unexpected GR00T checkpoint tensor {key!r}")


class NeuronGr00tModel(nn.Module):
    def __init__(self, cfg: Gr00tConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.vision = Gr00tVision(cfg.backbone["vision_config"])
        self.text = Gr00tText(cfg.backbone["text_config"])
        self.head = Gr00tActionHead(cfg)
        self._prep: BackbonePrep | None = None
        self.device = torch.device("cpu")
        self._fns: dict = {}  # name -> compiled callable (a plain dict, so not registered as submodules)
        self._dev_cache: OrderedDict = (
            OrderedDict()
        )  # (kind, key) -> device tensors that repeat across calls
        self.fused_backbone = os.environ.get("GR00T_FUSED_BACKBONE", "0") == "1"
        self.stage_timing = os.environ.get("GR00T_STAGE_TIMING", "0") == "1"
        self.stats: dict[str, float] = {}

    @property
    def prep(self) -> BackbonePrep:
        if self._prep is None:  # host-side HF helpers; built lazily, off the meta device
            self._prep = BackbonePrep(self.cfg.hf_backbone_config())
        return self._prep

    @classmethod
    def from_pretrained(
        cls, model_dir: str, dtype: torch.dtype = torch.bfloat16, device="cpu"
    ) -> NeuronGr00tModel:
        with torch.device(
            "meta"
        ):  # no random init of ~3B parameters; load_weights assigns real tensors
            m = cls(Gr00tConfig.from_model_dir(model_dir), dtype=dtype)
        m.load_weights(model_dir)
        return m.to(device)

    # -- weights ---------------------------------------------------------------------------
    def load_weights(self, model_dir: str) -> None:
        from safetensors import safe_open

        t0 = time.time()
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {model_dir}")
        state = {}
        for f in files:
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    dst = map_checkpoint_key(k)
                    if dst is not None:
                        state[dst] = fh.get_tensor(k).to(self.dtype)
        missing, unexpected = self.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"GR00T weight load mismatch: missing={missing[:8]} ({len(missing)}), "
                f"unexpected={unexpected[:8]} ({len(unexpected)})"
            )
        self.stats["load_s"] = time.time() - t0
        if os.environ.get("GR00T_ADALN_TABLES", "1") == "1":
            self.head.bake_adaln(
                fingerprint=os.path.basename(os.path.normpath(model_dir)),
                cache_dir=os.environ.get("GR00T_MODULATION_CACHE"),
            )
        if os.environ.get("GR00T_HEAD_FUSE_QKV", "0") == "1":
            self.head.fuse_projections()
        self.pretranspose(os.environ.get("GR00T_PRETRANSPOSE", "all"))
        logger.info("GR00T weights: %d tensors in %.1fs", len(state), self.stats["load_s"])

    def pretranspose(self, which: str) -> None:
        """Store linear weights as contiguous ``[in, out]`` for the listed parts ("head", "text",
        "vision", or "all"; comma-separated; "none" turns it off). Done on the host, before
        ``.to(device)``. Default "all": on trn2 it removes the in-graph weight transposes (~35% of
        TensorE work in the head and text graphs) and cut the 1-core LIBERO request 56.6 -> 51.0 ms."""
        from .layers import pretranspose_linears

        if which in ("", "none", "0"):
            return
        parts = {"head", "text", "vision"} if which == "all" else {p for p in which.split(",") if p}
        for name in sorted(parts):
            n = pretranspose_linears(getattr(self, name))
            logger.info("GR00T: %d %s linears pretransposed", n, name)

    def to(self, *args, **kwargs):
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            super().to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            super().to(device=self.device)
        return self

    def compile(self, backend: str, options: dict | None = None, **kwargs) -> NeuronGr00tModel:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(COMPILER_ARGS)}

        if self.fused_backbone:
            self._fns["backbone"] = torch.compile(
                _Backbone(self.vision, self.text),
                backend=backend,
                options=opts("gr00t_backbone"),
                **kw,
            )
        for name in ("vision", "text", "head"):
            self._fns[name] = torch.compile(
                getattr(self, name), backend=backend, options=opts(f"gr00t_{name}"), **kw
            )
        return self

    def _fn(self, name: str):
        fn = self._fns.get(name)
        return getattr(self, name) if fn is None else fn

    # -- inference -------------------------------------------------------------------------
    def draw_noise(
        self, batch: int, generator: torch.Generator | None = None, dtype=None
    ) -> torch.Tensor:
        """Upstream draws ``torch.randn`` in the model dtype on the model device; we draw on the
        host (CPU generator) in ``dtype`` (default: the model dtype)."""
        h = self.cfg.head
        return torch.randn(
            (batch, int(h["action_horizon"]), int(h["max_action_dim"])),
            dtype=dtype or self.dtype,
            generator=generator,
        )

    def _cached(self, kind: str, key, build):
        """Device copies of host tables that repeat across requests (vision taps/RoPE per grid,
        text RoPE/masks per prompt): uploaded once instead of every call."""
        k = (kind, key)
        hit = self._dev_cache.get(k)
        if hit is None:
            hit = self._dev_cache[k] = build()
            while len(self._dev_cache) > _DEVICE_CACHE_MAX:
                self._dev_cache.popitem(last=False)
        else:
            self._dev_cache.move_to_end(k)
        return hit

    @torch.no_grad()
    def get_action(
        self,
        inputs: dict,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        bucket: int | None = None,
    ) -> dict:
        dev, dt = self.device, self.dtype
        t0 = time.perf_counter()
        real = int(inputs["attention_mask"].sum())
        bi = self.prep(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs.get("mm_token_type_ids"),
            bucket=bucket or pick_bucket(real),
        )

        def d(x, dtype=None):
            x = x.detach().to("cpu")
            return (x.to(dtype) if dtype is not None else x).contiguous().to(dev)

        vis_t = self._cached(
            "vis",
            bi.vis_key,
            lambda: (
                d(bi.pos_index),
                d(bi.pos_weight, torch.float32),
                d(bi.vis_cos, torch.float32),
                d(bi.vis_sin, torch.float32),
            ),
        )
        txt_t = self._cached(
            "txt",
            bi.txt_key,
            lambda: (
                d(bi.input_ids),
                d(bi.image_index),
                d(bi.image_keep),
                d(bi.txt_cos, dt),
                d(bi.txt_sin, dt),
                d(bi.txt_bias),
                d(bi.valid),
                d(bi.image_mask),
            ),
        )
        emb = inputs["embodiment_id"].long().reshape(-1)
        emb_t = self._cached("emb", tuple(emb.tolist()), lambda: d(emb))
        if noise is None:
            noise = self.draw_noise(int(inputs["input_ids"].shape[0]), generator)
        pixels, state, noise_t = d(bi.pixels, dt), d(inputs["state"], dt), d(noise, dt)

        t1 = time.perf_counter()
        if self.fused_backbone:
            hidden = self._fn("backbone")(pixels, *vis_t, bi.n_images, *txt_t[:6])
        else:
            vis = self._fn("vision")(pixels, *vis_t, bi.n_images)
            hidden = self._fn("text")(*txt_t[:3], vis[0], *txt_t[3:6], *vis[1:])
        tb = None
        if self.stage_timing:  # one extra device sync so backbone and head time separately
            hidden[0, 0, :1].to("cpu")
            tb = time.perf_counter()
        actions = self._fn("head")(hidden, txt_t[6], txt_t[7], state, noise_t, emb_t)
        out = actions.to("cpu")
        t2 = time.perf_counter()
        self.stats.update(
            prep_s=t1 - t0,
            device_s=t2 - t1,
            bucket=bi.input_ids.shape[1],
            real_len=real,
            n_images=bi.n_images,
        )
        if tb is not None:
            self.stats.update(backbone_s=tb - t0, head_s=t2 - tb)
        return {"action_pred": out}

    def backbone_features(self, inputs: dict, bucket: int | None = None):
        """Debug/parity helper: (pre-norm hidden ``[B, real, D]`` on host, BackboneInputs)."""
        real = int(inputs["attention_mask"].sum())
        bi = self.prep(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            inputs.get("mm_token_type_ids"),
            bucket=bucket or real,
        )
        dev, dt = self.device, self.dtype

        def d(x, dtype=None):
            return (x.to(dtype) if dtype is not None else x).contiguous().to(dev)

        vis = self._fn("vision")(
            d(bi.pixels, dt),
            d(bi.pos_index),
            d(bi.pos_weight, torch.float32),
            d(bi.vis_cos, torch.float32),
            d(bi.vis_sin, torch.float32),
            bi.n_images,
        )
        hidden = self._fn("text")(
            d(bi.input_ids),
            d(bi.image_index),
            d(bi.image_keep),
            vis[0],
            d(bi.txt_cos, dt),
            d(bi.txt_sin, dt),
            d(bi.txt_bias),
            *vis[1:],
        )
        return hidden.to("cpu")[:, :real], bi


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
