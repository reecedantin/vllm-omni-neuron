# SPDX-License-Identifier: Apache-2.0
"""Configuration for InternVLA-A1.5 (policy ``config.json`` + the Qwen3.5 VLM ``config.json``).

An InternVLA-A1.5 checkpoint (``InternRobotics/InternVLA-A1.5-*``) is a LeRobot policy folder:
``config.json`` (policy hyper-parameters, ``type: internvla_a1_5``), ``model.safetensors`` (VLM +
action expert + action head, ``model.`` prefixed) and ``stats.json``. The VLM architecture is not
in the folder: upstream builds it from ``vlm_model_name_or_path`` (``Qwen/Qwen3.5-2B``). We resolve
that config, in order, from an explicit path, ``$INTERNVLA_VLM_CONFIG``, ``<ckpt>/vlm/config.json``,
then the local Hugging Face cache (``Qwen/Qwen3.5-2B`` or its ``-Base`` twin: same architecture).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class TextConfig:
    """``text_config`` of a Qwen3.5 model (hybrid Gated-DeltaNet / gated full attention)."""

    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    layer_types: list[str] = field(default_factory=list)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])

    @classmethod
    def from_dict(cls, d: dict) -> TextConfig:
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        n = int(d.get("num_hidden_layers", 24))
        layer_types = d.get("layer_types")
        if not layer_types:
            every = int(d.get("full_attention_interval", 4))
            layer_types = [
                "full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)
            ]
        return cls(
            hidden_size=int(d.get("hidden_size", 2048)),
            intermediate_size=int(d.get("intermediate_size", 6144)),
            num_hidden_layers=n,
            num_attention_heads=int(d.get("num_attention_heads", 8)),
            num_key_value_heads=int(d.get("num_key_value_heads", 2)),
            head_dim=int(d.get("head_dim", 256)),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            vocab_size=int(d.get("vocab_size", 248320)),
            layer_types=list(layer_types),
            linear_conv_kernel_dim=int(d.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(d.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(d.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(d.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(d.get("linear_num_value_heads", 16)),
            rope_theta=float(rope.get("rope_theta", d.get("rope_theta", 1e7))),
            partial_rotary_factor=float(
                rope.get("partial_rotary_factor", d.get("partial_rotary_factor", 0.25))
            ),
            mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
        )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]


@dataclass
class VisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304

    @classmethod
    def from_dict(cls, d: dict) -> VisionConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class VLMConfig:
    text: TextConfig
    vision: VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    @classmethod
    def from_dict(cls, d: dict) -> VLMConfig:
        return cls(
            text=TextConfig.from_dict(d["text_config"]),
            vision=VisionConfig.from_dict(d["vision_config"]),
            image_token_id=int(d.get("image_token_id", 248056)),
            video_token_id=int(d.get("video_token_id", 248057)),
            vision_start_token_id=int(d.get("vision_start_token_id", 248053)),
            vision_end_token_id=int(d.get("vision_end_token_id", 248054)),
        )


@dataclass
class PolicyConfig:
    """The inference-relevant subset of the upstream ``InternVLAA15Config``."""

    action_expert_hidden_size: int = 1024
    action_expert_intermediate_size: int = 3072
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    num_learnable_tokens: int = 50
    tokenize_state: bool = True
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 48
    action_token_min: int = 248077
    action_token_max: int = 250124
    block_action_attend_fast_tokens: bool = True
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"
    policy_type: str = "internvla_a1_5"

    @classmethod
    def from_dict(cls, d: dict) -> PolicyConfig:
        out = cls()
        for k in cls.__dataclass_fields__:
            if k in d:
                setattr(out, k, d[k])
        out.policy_type = d.get("type", out.policy_type)
        out.image_resolution = tuple(d.get("image_resolution", out.image_resolution))
        act = (d.get("output_features") or {}).get("action", {}).get("shape")
        if act:
            out.action_dim = int(act[0])
        return out


@dataclass
class InternVLAConfig:
    policy: PolicyConfig
    vlm: VLMConfig
    model_path: str = ""

    @classmethod
    def from_model_dir(cls, model_path: str, vlm_config: str | None = None) -> InternVLAConfig:
        with open(os.path.join(model_path, "config.json")) as f:
            pd = json.load(f)
        policy = PolicyConfig.from_dict(pd)
        if policy.policy_type != "internvla_a1_5":
            raise ValueError(
                f"{model_path}: policy type {policy.policy_type!r} is not supported "
                "(InternVLA-A1.5 only; the A1 'qwena1' policy needs its own port)"
            )
        path = resolve_vlm_config(model_path, policy.vlm_model_name_or_path, vlm_config)
        with open(path) as f:
            vlm = VLMConfig.from_dict(json.load(f))
        vlm.text.vocab_size = _checkpoint_vocab(model_path, vlm.text.vocab_size)
        return cls(policy=policy, vlm=vlm, model_path=model_path)


def _checkpoint_vocab(model_path: str, default: int) -> int:
    """Upstream grows the Qwen3.5 vocab with 2048 robot-action tokens; trust the checkpoint."""
    st = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(st):
        return default
    from safetensors import safe_open

    with safe_open(st, "pt") as f:
        for k in f.keys():
            if k.endswith("qwen3_5.lm_head.weight") or k.endswith(
                "language_model.embed_tokens.weight"
            ):
                return int(f.get_slice(k).get_shape()[0])
    return default


def resolve_vlm_config(model_path: str, vlm_name: str, explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("INTERNVLA_VLM_CONFIG"),
        os.path.join(model_path, "vlm", "config.json"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            c = os.path.join(c, "config.json")
        if c and os.path.isfile(c):
            return c
    try:
        from huggingface_hub import try_to_load_from_cache

        for repo in (vlm_name, vlm_name + "-Base"):
            hit = try_to_load_from_cache(repo, "config.json")
            if isinstance(hit, str) and os.path.isfile(hit):
                return hit
    except Exception:  # noqa: BLE001  any hub/cache error means "not cached"
        pass
    raise FileNotFoundError(
        f"Qwen3.5 VLM config for {vlm_name!r} not found: pass vlm_config=, set INTERNVLA_VLM_CONFIG, "
        f"or place it at {model_path}/vlm/config.json"
    )
