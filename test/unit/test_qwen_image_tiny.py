# SPDX-License-Identifier: Apache-2.0
"""Random-weight Qwen-Image 2.1 checkpoint with the real structure and shrunk dimensions.

Same ``model_index.json`` + component folders, class names, parameter names, layer types and
wiring as ``Qwen/Qwen-Image-2.1`` (single-stream block-causal DiT, Qwen3-VL text encoder,
16x image VAE), with few layers and small widths. It proves loading, weight-name mapping,
sharding and compilation in minutes; it says nothing about quality.

Usage::

    python test/unit/test_qwen_image_tiny.py --out <dir> [--processor-from <Qwen-Image-2.1 dir>]

The tokenizer/processor files are copied from a real checkpoint (``--processor-from``) because
the prompt template and special-token ids must match; without one, the tests that need a
tokenizer skip.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

# Every head-count divides 4, so the tiny model shards over TP=1, 2 and 4.
TINY_DIT = dict(
    patch_size=1,
    in_channels=16,
    out_channels=16,
    num_layers=2,
    attention_head_dim=32,
    num_attention_heads=8,
    context_in_dim=128,
    mlp_ratio=3,
    axes_dims_rope=(8, 12, 12),
    eps=1e-6,
    causal_condition=True,
)

TINY_VAE = dict(
    base_dim=8,
    decoder_base_dim=12,
    z_dim=16,
    dim_mult=[1, 2, 4, 8, 8],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True, True],
    dropout=0.0,
    in_channels=4,
    out_channels=4,
    is_residual=True,
    scale_factor_spatial=16,
    scale_factor_temporal=8,
)

TINY_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=32,
    rms_norm_eps=1e-6,
    rope_theta=5000000,
    max_position_embeddings=8192,
    vocab_size=151936,
    rope_scaling={"rope_type": "default", "mrope_section": [6, 5, 5], "mrope_interleaved": True},
    tie_word_embeddings=False,
)

TINY_VISION = dict(
    depth=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=2,
    out_hidden_size=128,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    in_channels=3,
    num_position_embeddings=2304,
    deepstack_visual_indexes=[0],
)

SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

MODEL_INDEX = {
    "_class_name": "QwenImage21Pipeline",
    "_diffusers_version": "0.37.0.dev0",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}


def _randomize_(module: torch.nn.Module, gen: torch.Generator) -> None:
    """Random, well-scaled values everywhere (zero-init norms and gates would hide mapping bugs)."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 1:
                base = (
                    0.0 if name.endswith("text_norm.weight") else 1.0
                )  # zero-centred RMSNorm stores scale-1
                p.copy_(base + 0.1 * torch.randn(p.shape, generator=gen))
            else:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=gen) / fan_in**0.5)


def build(out: str, processor_from: str | None = None, seed: int = 0) -> str:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
    )
    from vllm_omni_neuron.diffusion.models.qwen_image._vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    gen = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)

    dit = QwenImage21Transformer2DModel(**TINY_DIT)
    _randomize_(dit, gen)
    dit.to(torch.bfloat16).save_pretrained(os.path.join(out, "transformer"))

    z = TINY_VAE["z_dim"]
    vae = AutoencoderKLQwenImage21(
        **TINY_VAE,
        latents_mean=[round(0.3 * float(x), 4) for x in torch.randn(z, generator=gen)],
        latents_std=[round(1.0 + 0.5 * float(x), 4) for x in torch.rand(z, generator=gen)],
    )
    _randomize_(vae, gen)
    vae.to(torch.bfloat16).save_pretrained(os.path.join(out, "vae"))

    cfg = Qwen3VLConfig(
        text_config=dict(TINY_TEXT),
        vision_config=dict(TINY_VISION),
        image_token_id=151655,
        video_token_id=151656,
        tie_word_embeddings=False,
    )
    te = Qwen3VLForConditionalGeneration(cfg)
    _randomize_(te, gen)
    te.to(torch.bfloat16).save_pretrained(os.path.join(out, "text_encoder"))

    os.makedirs(os.path.join(out, "scheduler"), exist_ok=True)
    with open(os.path.join(out, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(SCHEDULER, f, indent=2)
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2)
    if processor_from:
        src = os.path.join(processor_from, "processor")
        dst = os.path.join(out, "processor")
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--processor-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(build(a.out, a.processor_from, a.seed))


if __name__ == "__main__":
    main()
