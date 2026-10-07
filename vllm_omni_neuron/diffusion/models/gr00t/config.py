# SPDX-License-Identifier: Apache-2.0
"""Configuration for the Neuron GR00T N1.7 port.

A GR00T checkpoint directory holds ``config.json`` (the ``Gr00tN1d7`` action-head and wiring
config), the processor files and one ``model*.safetensors`` set with both the VLM backbone
(``backbone.model.*``) and the action head (``action_head.*``). The backbone's own architecture
config is *not* in the checkpoint: upstream fetches it from ``nvidia/Cosmos-Reason2-2B`` on the
Hub. This module resolves it offline, in order:

1. ``<model_dir>/backbone_config.json`` (the tiny test checkpoint writes one);
2. ``$GR00T_BACKBONE_CONFIG`` (a ``config.json`` path or a directory holding one);
3. the built-in Cosmos-Reason2-2B (Qwen3-VL-2B) architecture below.

GR00T only uses the first ``select_layer`` decoder layers of the backbone, and the checkpoint
holds exactly those, so ``text_config.num_hidden_layers`` is overridden with ``select_layer``.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Cosmos-Reason2-2B is a Qwen3-VL-2B post-train; the architecture is Qwen3-VL-2B-Instruct's.
COSMOS_REASON2_2B: dict[str, Any] = {
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "max_position_embeddings": 262144,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "vocab_size": 151936,
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "intermediate_size": 4096,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

# Gr00tN1d7Config defaults (upstream ``configs/gr00t_n1d7.py``) for keys a config.json may omit.
# They are upstream's class defaults, not the released checkpoints: GR00T-N1.7-3B and GR00T-H-N1.7
# both ship ``select_layer: 16`` (16 of the backbone's 28 decoder layers) and a 32-layer DiT.
_HEAD_DEFAULTS: dict[str, Any] = {
    "action_horizon": 40,
    "hidden_size": 1024,
    "input_embedding_dim": 1536,
    "backbone_embedding_dim": 2048,
    "max_state_dim": 132,
    "max_action_dim": 132,
    "max_num_embodiments": 32,
    "max_seq_len": 1024,
    "state_history_length": 1,
    "num_inference_timesteps": 4,
    "num_timestep_buckets": 1000,
    "add_pos_embed": True,
    "use_vlln": True,
    "use_alternate_vl_dit": True,
    "attend_text_every_n_blocks": 2,
    "select_layer": 12,
    "vl_self_attention_cfg": None,
    "use_vl_self_attention": True,
    "diffusion_model_cfg": {
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "num_attention_heads": 32,
        "num_layers": 16,
        "output_dim": 1024,
        "interleave_self_attention": True,
        "positional_embeddings": None,
    },
}


def _read_backbone_config(model_dir: str) -> dict[str, Any]:
    local = os.path.join(model_dir, "backbone_config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    env = os.environ.get("GR00T_BACKBONE_CONFIG")
    if env:
        path = os.path.join(env, "config.json") if os.path.isdir(env) else env
        with open(path) as f:
            return json.load(f)
    return copy.deepcopy(COSMOS_REASON2_2B)


@dataclass
class Gr00tConfig:
    """The GR00T ``config.json`` fields this port uses, plus the resolved backbone config."""

    model_dir: str
    head: dict[str, Any]
    backbone: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model_dir(cls, model_dir: str) -> Gr00tConfig:
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        arch = raw.get("model_type") or (raw.get("architectures") or [""])[0]
        if arch != "Gr00tN1d7":
            raise ValueError(
                f"{model_dir}: expected a Gr00tN1d7 checkpoint, got model_type={arch!r}"
            )
        head = copy.deepcopy(_HEAD_DEFAULTS)
        head.update({k: v for k, v in raw.items() if k in _HEAD_DEFAULTS})
        dm = copy.deepcopy(_HEAD_DEFAULTS["diffusion_model_cfg"])
        dm.update(raw.get("diffusion_model_cfg") or {})
        head["diffusion_model_cfg"] = dm
        backbone = _read_backbone_config(model_dir)
        backbone["text_config"]["num_hidden_layers"] = int(head["select_layer"])
        return cls(
            model_dir=model_dir,
            head=head,
            backbone=backbone,
            extra={k: v for k, v in raw.items() if k not in _HEAD_DEFAULTS},
        )

    # -- derived quantities --------------------------------------------------------------
    @property
    def vl_self_attention_layers(self) -> int:
        cfg = self.head.get("vl_self_attention_cfg") or {}
        return int(cfg.get("num_layers", 0) or 0)

    @property
    def dit(self) -> dict[str, Any]:
        return self.head["diffusion_model_cfg"]

    @property
    def dit_inner_dim(self) -> int:
        return int(self.dit["num_attention_heads"]) * int(self.dit["attention_head_dim"])

    def hf_backbone_config(self):
        """The backbone as a ``transformers`` ``Qwen3VLConfig`` (host-side RoPE/position helpers)."""
        from transformers import Qwen3VLConfig

        return Qwen3VLConfig(**copy.deepcopy(self.backbone))
