# SPDX-License-Identifier: Apache-2.0
"""Configuration for the Neuron Alpamayo 1.5 / Alpamayo 2 Super port.

Both checkpoints are a Qwen3-VL-family VLM backbone (writing Chain-of-Causation reasoning text
autoregressively) plus a narrower Qwen3-VL-text-shaped "expert" transformer that attends to the
VLM's KV cache while flow-matching 64 future trajectory waypoints. ``config.json`` differs between
the two checkpoints in how much of the backbone it carries:

* **Alpamayo 1.5** names its backbone (``vlm_name_or_path: nvidia/Cosmos-Reason2-8B``) without
  shipping its architecture or tokenizer. Cosmos-Reason2-8B's config, tokenizer, chat template and
  processor files are byte-identical to ``Qwen/Qwen3-VL-8B-Instruct``'s (every file matches by MD5;
  ``model.safetensors.index.json`` differs only in shard file names), and every one of the 363
  backbone tensor shapes derived from that config matches the checkpoint. The checkpoint ships no
  tokenizer: running the backbone tokenizer through Alpamayo's own vocabulary extension (4000
  discrete trajectory tokens, then the special tokens) reproduces its ``embed_tokens`` row count
  (155697) and its documented ``traj_token_start_idx`` / ``traj_token_ids`` exactly.
* **Alpamayo 2 Super** ships its full backbone (``vlm_config``) and expert (``expert_config``) inline.

Backbone resolution (offline-first):
1. ``<model_dir>/backbone_config.json`` (a tiny test checkpoint writes one);
2. the checkpoint's own inline ``vlm_config`` (Alpamayo 2 Super);
3. ``$ALPAMAYO_BACKBONE_CONFIG`` (a ``config.json`` path or a directory holding one);
4. the built-in ``COSMOS_REASON2_8B`` (Alpamayo 1.5 only).

Tokenizer resolution: ``<model_dir>`` if it ships tokenizer files, else ``$ALPAMAYO_TOKENIZER_DIR``,
else ``$ALPAMAYO_BACKBONE_CONFIG`` (if a directory), else the Hub id :data:`DEFAULT_TOKENIZER_REPO`.
The tokenizer is only needed to decode the reasoning text; the rollout itself does not use it.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Qwen/Qwen3-VL-8B-Instruct's own public (ungated) config.json. Cosmos-Reason2-8B's model card:
# "post-trained based on Qwen3-VL-8B-Instruct and follows the same model architecture" -- verified
# against the real Alpamayo-1.5-10B checkpoint's tensor shapes (hidden_size, layer count, vision depth
# all match; only vocab_size differs, extended for the trajectory-token vocabulary).
COSMOS_REASON2_8B: dict[str, Any] = {
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "vocab_size": 151936,  # overridden per checkpoint (Alpamayo extends it for trajectory tokens)
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "intermediate_size": 4304,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

# alpamayo1_5.config.Alpamayo1_5Config / alpamayo2_super's analogous config -- the fields this port
# uses, with upstream's defaults for anything a checkpoint's config.json might omit.
_HEAD_DEFAULTS: dict[str, Any] = {
    "tokens_per_future_traj": 128,  # upstream's own generation_config.max_new_tokens cap
    "tokens_per_history_traj": 48,
    "traj_vocab_size": 4000,
    "traj_token_start_idx": 151669,
    "expert_non_causal_attention": True,
    "action_space_cfg": {
        "n_waypoints": 64,
        "dt": 0.1,
        "accel_bounds": [-9.8, 9.8],
        "curvature_bounds": [-0.33, 0.33],
    },
    "action_in_proj_cfg": {
        "hidden_size": 1024,
        "num_enc_layers": 4,
        "max_freq": 100.0,
        "num_fourier_feats": 20,
    },
    "diffusion_cfg": {
        "num_inference_steps": 10,
        "int_method": "euler",
        "use_classifier_free_guidance": False,
        "inference_guidance_weight": 1.0,
    },
}


# Hub id of the backbone's (public, byte-identical to Cosmos-Reason2-8B) tokenizer/processor files.
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-VL-8B-Instruct"

# TRAJ_TOKEN / SPECIAL_TOKENS_KEYS from alpamayo1_5.models.base_model, duplicated here so this
# module (and the tiny-checkpoint builder) can reproduce the real tokenizer's vocabulary extension
# without importing hydra-dependent upstream code. Keep in sync if upstream changes these lists.
TRAJ_TOKEN: dict[str, str] = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
SPECIAL_TOKENS_KEYS: list[str] = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS: dict[str, str] = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}
# alpamayo2_super.models.utils.SPECIAL_TOKENS_KEYS: the same 29 slots (so every trajectory special
# token keeps its id), with the reserved ``_padding_*`` slots renamed.
SUPER_SPECIAL_TOKENS_KEYS: list[str] = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]


def build_tokenizer(
    tokenizer_dir: str,
    traj_vocab_size: int,
    add_special_tokens: bool,
    special_keys: list[str] | None = None,
):
    """Reproduce upstream ``ReasoningVLA._build_tokenizer``'s vocabulary extension on top of a base
    (public) Qwen3-VL tokenizer, WITHOUT depending on hydra/the upstream package. Verified to
    reproduce the real Alpamayo-1.5-10B checkpoint exactly: final vocab size 155697 (== the
    checkpoint's ``embed_tokens`` row count), ``<i0>`` id 151669 (== ``traj_token_start_idx``),
    ``<|traj_future_start|>`` id 155681 (== ``traj_token_ids.future_start``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
    n = tok.add_tokens(discrete_tokens)
    if n != len(discrete_tokens):
        raise RuntimeError(
            f"expected to add {len(discrete_tokens)} discrete trajectory tokens, added {n}"
        )
    tok.traj_token_start_idx = tok.convert_tokens_to_ids("<i0>")
    tok.traj_token_end_idx = tok.convert_tokens_to_ids(f"<i{traj_vocab_size - 1}>")
    if add_special_tokens:
        keys = special_keys or SPECIAL_TOKENS_KEYS
        tok.add_tokens(["<|" + k + "|>" for k in keys], special_tokens=True)
    else:
        tok.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
    tok.traj_token_ids = {k: tok.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()}
    return tok


def _tokenizer_dir(model_dir: str) -> str:
    """The checkpoint's own tokenizer files if it ships any (Alpamayo 2 Super does; 1.5 does not),
    else the same backbone-config fallback chain as ``_read_backbone_config``."""
    if os.path.isfile(os.path.join(model_dir, "tokenizer_config.json")):
        return model_dir
    for env in (
        os.environ.get("ALPAMAYO_TOKENIZER_DIR"),
        os.environ.get("ALPAMAYO_BACKBONE_CONFIG"),
    ):
        if env and os.path.isdir(env):
            return env
    return DEFAULT_TOKENIZER_REPO


def _read_backbone_config(model_dir: str, inline: dict[str, Any] | None) -> dict[str, Any]:
    local = os.path.join(model_dir, "backbone_config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    if inline:
        return copy.deepcopy(inline)
    env = os.environ.get("ALPAMAYO_BACKBONE_CONFIG")
    if env:
        path = os.path.join(env, "config.json") if os.path.isdir(env) else env
        with open(path) as f:
            return json.load(f)
    return copy.deepcopy(COSMOS_REASON2_8B)


@dataclass
class AlpamayoConfig:
    """The fields this port uses from one Alpamayo checkpoint's ``config.json``."""

    model_dir: str
    variant: str  # "alpamayo1_5" | "alpamayo2_super"
    head: dict[str, Any]
    backbone: dict[str, Any]
    expert: dict[str, Any]
    tokenizer_dir: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model_dir(cls, model_dir: str) -> AlpamayoConfig:
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        arch = (raw.get("architectures") or [raw.get("model_type", "")])[0]
        if "Alpamayo2" in arch or raw.get("model_type") == "alpamayo2_super":
            return cls._from_alpamayo2(model_dir, raw)
        if "Alpamayo" in arch or raw.get("model_type") == "alpamayo1_5":
            return cls._from_alpamayo1_5(model_dir, raw)
        raise ValueError(f"{model_dir}: unrecognized Alpamayo architecture {arch!r}")

    @classmethod
    def _from_alpamayo1_5(cls, model_dir: str, raw: dict) -> AlpamayoConfig:
        head = copy.deepcopy(_HEAD_DEFAULTS)
        for k in head:
            if k in raw:
                head[k] = raw[k]
        head["action_space_cfg"].update(raw.get("action_space_cfg", {}))
        head["action_in_proj_cfg"].update(raw.get("action_in_proj_cfg", {}))
        head["diffusion_cfg"].update(raw.get("diffusion_cfg", {}))
        backbone = _read_backbone_config(model_dir, None)
        backbone["text_config"]["vocab_size"] = int(
            raw.get("vocab_size", backbone["text_config"]["vocab_size"])
        )
        expert = dict(raw.get("expert_cfg") or {})
        tok_dir = _tokenizer_dir(model_dir)
        return cls(
            model_dir=model_dir,
            variant="alpamayo1_5",
            head=head,
            backbone=backbone,
            expert=expert,
            tokenizer_dir=tok_dir,
            extra={k: v for k, v in raw.items() if k not in head},
        )

    @classmethod
    def _from_alpamayo2(cls, model_dir: str, raw: dict) -> AlpamayoConfig:
        """Alpamayo 2 Super: backbone + expert inline; trajectory ids under ``traj_ids``
        (``history_id0`` / ``future_id0`` = the first history / future bin token, ``*_pad`` = the
        placeholders). Rollout differences from 1.5, all upstream's own
        ``sample_trajectories_from_data`` defaults: generation runs to ``max(256,
        tokens_per_future_traj)`` new tokens, and the backbone's text EOS is masked to ``-inf`` (only
        ``<|traj_future_start|>`` ends the reasoning)."""
        head = copy.deepcopy(_HEAD_DEFAULTS)
        for k in head:
            if k in raw:
                head[k] = raw[k]
        exp_cfg = raw.get("expert_config", {})
        head["action_space_cfg"].update(exp_cfg.get("action_space_cfg", {}))
        head["action_in_proj_cfg"].update(exp_cfg.get("action_in_proj_cfg", {}))
        head["diffusion_cfg"].update(exp_cfg.get("diffusion_cfg", {}))
        head["expert_non_causal_attention"] = bool(exp_cfg.get("expert_non_causal_attention", True))
        ids = raw["traj_ids"]
        head["traj_vocab_size"] = int(
            raw.get("history_vocab_size", 1000) + raw.get("future_vocab_size", 3000)
        )
        head["traj_token_start_idx"] = min(int(ids["history_id0"]), int(ids["future_id0"]))
        head["hist_token_start_idx"] = int(ids["history_id0"])
        head["max_generation_length"] = max(256, int(head["tokens_per_future_traj"]))
        head["fourier_freqs_bf16"] = False  # FourierEncoderV2.freqs are fp32 in the fp32 model
        backbone = _read_backbone_config(model_dir, raw.get("vlm_config"))
        eos = backbone["text_config"].get("eos_token_id", 151645)
        eos = [int(e) for e in (eos if isinstance(eos, list) else [eos])]
        head["masked_token_ids"] = [e for e in eos if e != int(ids["future_start"])]
        head["stop_token_ids"] = eos
        full_expert = exp_cfg.get("llm_config") or exp_cfg.get("expert_update_cfg") or {}
        expert = {
            k: full_expert[k]
            for k in (
                "head_dim",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "rms_norm_eps",
            )
            if k in full_expert
        }
        tok_dir = _tokenizer_dir(model_dir)
        extra = {k: v for k, v in raw.items() if k not in head}
        extra["traj_token_ids"] = {
            "history": int(ids["history_pad"]),
            "future": int(ids["future_pad"]),
            "history_start": int(ids["history_start"]),
            "future_start": int(ids["future_start"]),
            "history_end": int(ids["history_end"]),
            "future_end": int(ids["future_end"]),
        }
        return cls(
            model_dir=model_dir,
            variant="alpamayo2_super",
            head=head,
            backbone=backbone,
            expert=expert,
            tokenizer_dir=tok_dir,
            extra=extra,
        )

    def build_tokenizer(self):
        return build_tokenizer(
            self.tokenizer_dir,
            int(self.head["traj_vocab_size"]),
            bool(self.extra.get("add_special_tokens", True)),
            SUPER_SPECIAL_TOKENS_KEYS if self.variant == "alpamayo2_super" else None,
        )

    def hf_backbone_config(self):
        from transformers import Qwen3VLConfig

        return Qwen3VLConfig(**copy.deepcopy(self.backbone))

    @property
    def expert_hidden_size(self) -> int:
        return int(self.expert.get("hidden_size", self.backbone["text_config"]["hidden_size"]))

    @property
    def n_waypoints(self) -> int:
        return int(self.head["action_space_cfg"]["n_waypoints"])

    @property
    def max_new_tokens(self) -> int:
        """Upstream's own fixed cap on stage-1 (VLM reasoning) generation length: 1.5 generates at
        most ``tokens_per_future_traj`` (128) tokens, 2 Super ``max(256, tokens_per_future_traj)``."""
        return int(self.head.get("max_generation_length", self.head["tokens_per_future_traj"]))
