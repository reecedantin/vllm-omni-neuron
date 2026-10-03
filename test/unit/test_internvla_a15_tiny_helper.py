# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny random-weight InternVLA-A1.5 checkpoint with the real model's structure.

Same on-disk layout as ``InternRobotics/InternVLA-A1.5-base`` (``config.json``,
``model.safetensors`` with the ``model.``-prefixed upstream tensor names and dtypes,
``stats.json``) plus ``vlm/config.json``, the Qwen3.5 config the real checkpoint takes from the
Hub. Same layer pattern (3 Gated-DeltaNet + 1 gated full-attention layer, repeated), the real
vocabulary (action-token ids must stay valid), real chunk/action/learnable-token sizes; only
widths and depths shrink. It proves loading, name mapping, graph construction and compile;
it says nothing about quality.

    python test/unit/test_internvla_a15_tiny_helper.py OUT_DIR [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

GAIN = float(os.environ.get("INTERNVLA_TINY_GAIN", "0.5"))
VOCAB = 250368  # Qwen3.5 248320 + 2048 robot-action tokens (as in the released checkpoint)

TINY_VLM = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 0,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {
        "model_type": "qwen3_5",
        "deepstack_visual_indexes": [],
        "depth": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "in_channels": 3,
        "intermediate_size": 64,
        "num_heads": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


def tiny_policy_config(real: dict | None = None) -> dict:
    cfg = dict(real or {})
    cfg.update({
        "type": "internvla_a1_5",
        # vLLM-Omni's OmniDiffusionConfig.enrich_config has no built-in model_type mapping for a
        # third-party policy (unlike its hardcoded "Gr00tN1d7" case); its generic fallback is
        # `architectures == [class_name]` matched against the DiffusionModelRegistry, so this is
        # how a checkpoint tells the engine which pipeline class to use.
        "architectures": ["InternVLAA15Pipeline"],
        "n_obs_steps": 1,
        "input_features": {"observation.state": {"type": "STATE", "shape": [32]}},
        "output_features": {"action": {"type": "ACTION", "shape": [32]}},
        "vlm_model_name_or_path": "vlm",
        "action_expert_hidden_size": 32,
        "action_expert_intermediate_size": 64,
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "min_period": 0.004,
        "max_period": 4.0,
        "image_resolution": [224, 224],
        "tokenizer_max_length": 48,
        "tokenize_state": True,
        "action_token_min": 248077,
        "action_token_max": 250124,
        "knowledge_insulation": False,
        "block_action_attend_fast_tokens": True,
        "inference_action_type": "fm",
        "num_learnable_tokens": 50,
        "action_loss_only": False,
        "inference_backend": "standard",
    })
    return cfg


def _tensors(seed: int) -> dict[str, torch.Tensor]:
    """Upstream tensor names/dtypes (``model.``-prefixed), random values."""
    g = torch.Generator().manual_seed(seed)
    t, v = TINY_VLM["text_config"], TINY_VLM["vision_config"]
    eh, ei = 32, 64
    bf, f32 = torch.bfloat16, torch.float32

    def rnd(*shape, std=0.05, dtype=bf):
        # GAIN < 1 keeps the random network contracting: a unit-gain random stack is chaotic
        # (a 1e-6 input change moved the 10-step actions by 6%), which hides real parity bugs.
        return (torch.randn(*shape, generator=g) * std * GAIN).to(dtype)

    out: dict[str, torch.Tensor] = {}

    def text_layers(prefix: str, hidden: int, inter: int):
        kd = t["linear_key_head_dim"] * t["linear_num_key_heads"]
        vd = t["linear_value_head_dim"] * t["linear_num_value_heads"]
        nh, hd, kvh = t["num_attention_heads"], t["head_dim"], t["num_key_value_heads"]
        for i, kind in enumerate(t["layer_types"]):
            p = f"{prefix}.layers.{i}"
            out[f"{p}.input_layernorm.weight"] = rnd(hidden, std=0.1, dtype=f32)
            out[f"{p}.post_attention_layernorm.weight"] = rnd(hidden, std=0.1, dtype=f32)
            out[f"{p}.mlp.gate_proj.weight"] = rnd(inter, hidden, std=1 / math.sqrt(hidden))
            out[f"{p}.mlp.up_proj.weight"] = rnd(inter, hidden, std=1 / math.sqrt(hidden))
            out[f"{p}.mlp.down_proj.weight"] = rnd(hidden, inter, std=1 / math.sqrt(inter))
            if kind == "linear_attention":
                la = f"{p}.linear_attn"
                nv = t["linear_num_value_heads"]
                out[f"{la}.A_log"] = torch.log(torch.rand(nv, generator=g) * 15 + 1).to(bf)
                out[f"{la}.dt_bias"] = (torch.rand(nv, generator=g) * 2 - 1).to(bf)
                out[f"{la}.conv1d.weight"] = rnd(2 * kd + vd, 1, t["linear_conv_kernel_dim"], std=0.3)
                out[f"{la}.in_proj_qkv.weight"] = rnd(2 * kd + vd, hidden, std=1 / math.sqrt(hidden))
                out[f"{la}.in_proj_z.weight"] = rnd(vd, hidden, std=1 / math.sqrt(hidden))
                out[f"{la}.in_proj_a.weight"] = rnd(nv, hidden, std=1 / math.sqrt(hidden))
                out[f"{la}.in_proj_b.weight"] = rnd(nv, hidden, std=1 / math.sqrt(hidden))
                out[f"{la}.norm.weight"] = (1 + rnd(t["linear_value_head_dim"], std=0.1, dtype=f32)).to(bf)
                out[f"{la}.out_proj.weight"] = rnd(hidden, vd, std=1 / math.sqrt(vd))
            else:
                sa = f"{p}.self_attn"
                out[f"{sa}.q_proj.weight"] = rnd(nh * hd * 2, hidden, std=1 / math.sqrt(hidden))
                out[f"{sa}.k_proj.weight"] = rnd(kvh * hd, hidden, std=1 / math.sqrt(hidden))
                out[f"{sa}.v_proj.weight"] = rnd(kvh * hd, hidden, std=1 / math.sqrt(hidden))
                out[f"{sa}.o_proj.weight"] = rnd(hidden, nh * hd, std=1 / math.sqrt(nh * hd))
                out[f"{sa}.q_norm.weight"] = rnd(hd, std=0.1)
                out[f"{sa}.k_norm.weight"] = rnd(hd, std=0.1)

    vlm = "model.qwen3_5_with_expert.qwen3_5"
    text_layers(f"{vlm}.model.language_model", t["hidden_size"], t["intermediate_size"])
    out[f"{vlm}.model.language_model.norm.weight"] = rnd(t["hidden_size"], std=0.1, dtype=f32)
    out[f"{vlm}.lm_head.weight"] = rnd(VOCAB, t["hidden_size"], std=0.5)
    exp = "model.qwen3_5_with_expert.action_expert"
    text_layers(exp, eh, ei)
    out[f"{exp}.norm.weight"] = rnd(eh, std=0.1)

    vis = f"{vlm}.model.visual"
    vh, vi = v["hidden_size"], v["intermediate_size"]
    k = v["in_channels"] * v["temporal_patch_size"] * v["patch_size"] ** 2
    out[f"{vis}.patch_embed.proj.weight"] = rnd(vh, v["in_channels"], v["temporal_patch_size"], v["patch_size"],
                                                v["patch_size"], std=1 / math.sqrt(k))
    out[f"{vis}.patch_embed.proj.bias"] = rnd(vh, std=0.02)
    out[f"{vis}.pos_embed.weight"] = rnd(v["num_position_embeddings"], vh, std=0.2)
    for i in range(v["depth"]):
        bp = f"{vis}.blocks.{i}"
        for n in ("norm1", "norm2"):
            out[f"{bp}.{n}.weight"] = (1 + rnd(vh, std=0.1, dtype=f32)).to(bf)
            out[f"{bp}.{n}.bias"] = rnd(vh, std=0.02)
        out[f"{bp}.attn.qkv.weight"] = rnd(3 * vh, vh, std=1 / math.sqrt(vh))
        out[f"{bp}.attn.qkv.bias"] = rnd(3 * vh, std=0.02)
        out[f"{bp}.attn.proj.weight"] = rnd(vh, vh, std=1 / math.sqrt(vh))
        out[f"{bp}.attn.proj.bias"] = rnd(vh, std=0.02)
        out[f"{bp}.mlp.linear_fc1.weight"] = rnd(vi, vh, std=1 / math.sqrt(vh))
        out[f"{bp}.mlp.linear_fc1.bias"] = rnd(vi, std=0.02)
        out[f"{bp}.mlp.linear_fc2.weight"] = rnd(vh, vi, std=1 / math.sqrt(vi))
        out[f"{bp}.mlp.linear_fc2.bias"] = rnd(vh, std=0.02)
    mh = vh * v["spatial_merge_size"] ** 2
    out[f"{vis}.merger.norm.weight"] = (1 + rnd(vh, std=0.1, dtype=f32)).to(bf)
    out[f"{vis}.merger.norm.bias"] = rnd(vh, std=0.02)
    out[f"{vis}.merger.linear_fc1.weight"] = rnd(mh, mh, std=1 / math.sqrt(mh))
    out[f"{vis}.merger.linear_fc1.bias"] = rnd(mh, std=0.02)
    out[f"{vis}.merger.linear_fc2.weight"] = rnd(t["hidden_size"], mh, std=1 / math.sqrt(mh))
    out[f"{vis}.merger.linear_fc2.bias"] = rnd(t["hidden_size"], std=0.02)

    a = 32
    out["model.action_in_proj.weight"] = rnd(eh, a, std=1 / math.sqrt(a), dtype=f32)
    out["model.action_in_proj.bias"] = rnd(eh, std=0.02, dtype=f32)
    out["model.action_out_proj.weight"] = rnd(a, eh, std=1 / math.sqrt(eh), dtype=f32)
    out["model.action_out_proj.bias"] = rnd(a, std=0.02, dtype=f32)
    out["model.action_time_mlp_in.weight"] = rnd(eh, 2 * eh, std=1 / math.sqrt(2 * eh), dtype=f32)
    out["model.action_time_mlp_in.bias"] = rnd(eh, std=0.02, dtype=f32)
    out["model.action_time_mlp_out.weight"] = rnd(eh, eh, std=1 / math.sqrt(eh), dtype=f32)
    out["model.action_time_mlp_out.bias"] = rnd(eh, std=0.02, dtype=f32)
    out["model.learnable_tokens"] = rnd(50, eh, std=0.5, dtype=f32)
    out["model.learnable_tokens_in_proj.weight"] = rnd(eh, eh, std=1 / math.sqrt(eh), dtype=f32)
    out["model.learnable_tokens_in_proj.bias"] = rnd(eh, std=0.02, dtype=f32)
    # training-only WAN bridge, present in released checkpoints and ignored at inference
    out["model.learnable_to_wan_proj.weight"] = rnd(96, eh, dtype=f32)
    out["model.learnable_to_wan_proj.bias"] = rnd(96, dtype=f32)
    out["model._wan_grid_sizes"] = torch.tensor([2, 7, 7], dtype=torch.long)
    return out


def make_tiny_checkpoint(out_dir: str, seed: int = 0) -> str:
    from safetensors.torch import save_file

    os.makedirs(os.path.join(out_dir, "vlm"), exist_ok=True)
    tensors = _tensors(seed)
    emb = "model.qwen3_5_with_expert.qwen3_5.model.language_model.embed_tokens.weight"
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out_dir, "model.safetensors"),
              metadata={emb: "model.qwen3_5_with_expert.qwen3_5.lm_head.weight"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(tiny_policy_config(), f, indent=2)
    with open(os.path.join(out_dir, "vlm", "config.json"), "w") as f:
        json.dump(TINY_VLM, f, indent=2)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({}, f)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_tiny_checkpoint(a.out_dir, a.seed))
