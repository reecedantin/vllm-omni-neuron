# SPDX-License-Identifier: Apache-2.0
"""Run upstream InternVLA-A1.5 (``InternRobotics/InternVLA-A-series``) on the CPU as an oracle.

Upstream's policy lives in a LeRobot fork whose package imports pull in training-only
dependencies (draccus, accelerate, datasets, WAN). This loader execs only the two files that
define the inference math -- ``modeling_internvla_a1_5.py`` and the Qwen3.5 modeling file
upstream installs over transformers (``transformers_replace``) -- with the rest stubbed, then
builds ``InternVLAA15`` with ``action_loss_only`` (no WAN) and loads the checkpoint into it.

Point ``INTERNVLA_REF_SRC`` at the clone's ``src`` directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import torch
import torch.nn as nn

_REF_QWEN = (
    "lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py"
)
_REF_MODEL = "lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py"


def ref_src() -> str | None:
    p = os.environ.get("INTERNVLA_REF_SRC", "")
    return p if p and os.path.isfile(os.path.join(p, _REF_MODEL)) else None


def _exec(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


_LOADED: dict = {}


def load_upstream_module(src: str):
    """Returns ``(modeling_internvla_a1_5, modeling_qwen3_5)`` from the upstream clone."""
    if "mod" in _LOADED:
        return _LOADED["mod"]
    import transformers.models.qwen3_5 as q35

    qwen = _exec(
        "transformers.models.qwen3_5._internvla_upstream_modeling", os.path.join(src, _REF_QWEN)
    )
    _create_causal_mask = qwen.create_causal_mask

    def create_causal_mask(config, inputs_embeds, attention_mask, cache_position=None, **kw):
        # upstream targets transformers 5.2 (``cache_position`` kwarg); InternVLA only ever
        # passes a prepared 4D mask, which every version returns as-is
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            return attention_mask
        return _create_causal_mask(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw
        )

    qwen.create_causal_mask = create_causal_mask
    saved = {k: sys.modules.get(k) for k in ("transformers.models.qwen3_5.modeling_qwen3_5",)}
    sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"] = qwen
    saved_attrs = {
        k: getattr(q35, k, None)
        for k in ("modeling_qwen3_5", "Qwen3_5ForConditionalGeneration", "Qwen3_5TextModel")
    }
    q35.modeling_qwen3_5 = qwen
    q35.Qwen3_5ForConditionalGeneration = qwen.Qwen3_5ForConditionalGeneration
    q35.Qwen3_5TextModel = qwen.Qwen3_5TextModel
    try:
        for pkg in (
            "lerobot",
            "lerobot.policies",
            "lerobot.policies.internvla_a1_5",
            "lerobot.policies.internvla_a1_5.wan",
            "lerobot.policies.internvla_a1_5.wan.modules",
            "lerobot.policies.internvla_a1_5.wan.utils",
            "lerobot.utils",
            "lerobot.transforms",
        ):
            _stub(pkg)
        _exec(
            "lerobot.policies.internvla_a1_5.action_tokens",
            os.path.join(src, "lerobot/policies/internvla_a1_5/action_tokens.py"),
        )
        _stub(
            "lerobot.policies.internvla_a1_5.configuration_internvla_a1_5",
            InternVLAA15Config=object,
        )
        _stub("lerobot.policies.internvla_a1_5.wan_model", WanVideoModel=None)
        _stub("lerobot.policies.internvla_a1_5.wan.modules.model", sinusoidal_embedding_1d=None)
        _stub("lerobot.policies.internvla_a1_5.wan.utils.fm", FlowMatchScheduler=None)
        _stub(
            "lerobot.policies.internvla_a1_5.transform_internvla_a1_5",
            LABEL_MODE_FAST=2,
            LABEL_MODE_NONE=0,
            LABEL_MODE_TEXT=1,
        )
        _stub("lerobot.policies.pretrained", PreTrainedPolicy=nn.Module)
        _stub("lerobot.utils.utils", format_big_number=str)
        _exec("lerobot.utils.constants", os.path.join(src, "lerobot/utils/constants.py"))
        model = _exec(
            "lerobot.policies.internvla_a1_5.modeling_internvla_a1_5", os.path.join(src, _REF_MODEL)
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
        for k, v in saved_attrs.items():
            setattr(q35, k, v)
    _LOADED["mod"] = (model, qwen)
    return model, qwen


def build_upstream(model_path: str, vlm_config_path: str, dtype: torch.dtype, src: str):
    """Upstream ``InternVLAA15`` with the checkpoint loaded, in eval mode on the CPU.

    Weights follow upstream inference (``policy.to(dtype)``, action head fp32)."""
    M, qwen = load_upstream_module(src)
    from transformers.models.qwen3_5 import Qwen3_5Config

    with open(os.path.join(model_path, "config.json")) as f:
        pcfg = json.load(f)
    vcfg = Qwen3_5Config.from_pretrained(os.path.dirname(vlm_config_path))
    from safetensors import safe_open

    st = os.path.join(model_path, "model.safetensors")
    with safe_open(st, "pt") as f:
        vocab = f.get_slice("model.qwen3_5_with_expert.qwen3_5.lm_head.weight").get_shape()[0]
    vcfg.text_config.vocab_size = vocab
    vcfg.vocab_size = vocab

    class _CG(qwen.Qwen3_5ForConditionalGeneration):
        @classmethod
        def from_pretrained(cls, *_a, **_k):
            return qwen.Qwen3_5ForConditionalGeneration(vcfg)

    class _Tok:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return None

    M.Qwen3_5ForConditionalGeneration = _CG
    M.Qwen3_5Tokenizer = _Tok
    M.ensure_qwen35_action_tokens = lambda *a, **k: None
    cfg = types.SimpleNamespace(**pcfg)
    cfg.action_loss_only = True
    cfg.compile_model = False
    cfg.freeze_vision_encoder = False
    cfg.train_expert_only = False
    cfg.freeze_learnable_tokens = False
    cfg.dtype = "float32"
    with torch.device("cpu"):
        model = M.InternVLAA15(cfg)
    state = {}
    with safe_open(st, "pt") as f:
        for k in f.keys():
            name = k[len("model.") :]
            if name.startswith(("learnable_to_wan_proj.", "_wan_grid_sizes")):
                continue
            state[name] = f.get_tensor(k)
    state["qwen3_5_with_expert.qwen3_5.model.language_model.embed_tokens.weight"] = state[
        "qwen3_5_with_expert.qwen3_5.lm_head.weight"
    ]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"upstream load: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(dtype)
    model.action_out_proj.to(torch.float32)
    return model.eval()


@torch.no_grad()
def upstream_sample(model, batch: dict, noise: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Upstream ``predict_action_chunk`` (state cast to the model dtype, as the openloop eval)."""
    state = batch["state"].to(dtype)
    return model.sample_actions(
        batch["pixel_values"].to(dtype),
        batch["image_grid_thw"],
        batch["input_ids"],
        batch["attention_mask"],
        state,
        fast_token_mask=batch.get("fast_token_mask"),
        noise=noise.clone(),
    )
