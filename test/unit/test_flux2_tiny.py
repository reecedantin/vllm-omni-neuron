# SPDX-License-Identifier: Apache-2.0
"""Random-weight FLUX.2-dev structure model ("tiny-flux2") for plumbing tests.

The checkpoint has the real model's on-disk layout (``model_index.json`` + ``transformer`` /
``text_encoder`` / ``vae`` / ``scheduler`` / ``tokenizer``), the same classes, parameter names,
block types, attention layout (joint double-stream blocks + fused parallel single-stream blocks,
QK-RMSNorm, 4-axis RoPE, head_dim 128) and VAE / text-encoder wiring (Mistral3 text tower whose
hidden states 10/20/30 are stacked into the DiT context, 32-channel VAE latents patchified to
128 channels). Only the widths and depths are shrunk. It says nothing about image quality.

Head counts divide 8 so the same checkpoint exercises TP=1/2/4/8 sharding.

Usage::

    python test/unit/test_flux2_tiny.py --out <dir> [--tokenizer-from <flux2-dev dir>]

The tokenizer and scheduler config are copied from a real FLUX.2-dev checkout when given
(``--tokenizer-from``); generated weights are never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

TINY_TRANSFORMER = dict(
    patch_size=1,
    in_channels=128,
    out_channels=None,
    num_layers=2,
    num_single_layers=6,
    attention_head_dim=128,
    num_attention_heads=8,
    joint_attention_dim=3 * 256,
    timestep_guidance_channels=256,
    mlp_ratio=3.0,
    axes_dims_rope=(32, 32, 32, 32),
    rope_theta=2000,
    eps=1e-6,
)

TINY_TEXT = dict(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=31,  # >= 30: the pipeline reads hidden_states[10, 20, 30]
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=32,
    rms_norm_eps=1e-5,
    rope_theta=1000000000.0,
    max_position_embeddings=131072,
    vocab_size=131072,
    hidden_act="silu",
    sliding_window=None,
)

TINY_VISION = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=2,
    head_dim=32,
    image_size=1540,
    patch_size=14,
    num_channels=3,
    rope_theta=10000.0,
    hidden_act="silu",
)

TINY_VAE = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(32, 32, 32, 32),
    layers_per_block=1,
    act_fn="silu",
    latent_channels=32,
    norm_num_groups=32,
    sample_size=1024,
    force_upcast=True,
    use_quant_conv=True,
    use_post_quant_conv=True,
    mid_block_add_attention=True,
    batch_norm_eps=1e-4,
    batch_norm_momentum=0.1,
    patch_size=(2, 2),
)

SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

MODEL_INDEX = {
    "_class_name": "Flux2Pipeline",
    "_diffusers_version": "0.36.0.dev0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"],
    "tokenizer": ["transformers", "PixtralProcessor"],
    "transformer": ["diffusers", "Flux2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def _randomize(module: torch.nn.Module, seed: int, std: float = 0.02) -> None:
    """Deterministic random init with non-trivial norm weights (ones would hide name swaps)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(module.named_parameters()):
            if p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=g) * min(std, fan_in**-0.5))
        for name, b in module.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.1 * torch.randn(b.shape, generator=g))
            elif name.endswith("running_var"):
                b.copy_(1.0 + 0.2 * torch.rand(b.shape, generator=g))


def build(
    out: str,
    tokenizer_from: str | None = None,
    seed: int = 0,
    num_layers: int | None = None,
    num_single_layers: int | None = None,
) -> str:
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from transformers import (
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        MistralConfig,
        PixtralVisionConfig,
    )

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2)

    tcfg = dict(TINY_TRANSFORMER)
    if num_layers is not None:
        tcfg["num_layers"] = num_layers
    if num_single_layers is not None:
        tcfg["num_single_layers"] = num_single_layers
    tf = Flux2Transformer2DModel(**tcfg)
    _randomize(tf, seed)
    tf.to(torch.bfloat16).save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)

    cfg = Mistral3Config(
        text_config=MistralConfig(**TINY_TEXT, dtype="bfloat16"),
        vision_config=PixtralVisionConfig(**TINY_VISION, dtype="bfloat16"),
        image_token_index=10,
        multimodal_projector_bias=False,
        projector_hidden_act="gelu",
        spatial_merge_size=2,
        vision_feature_layer=-1,
    )
    te = Mistral3ForConditionalGeneration(cfg)
    _randomize(te, seed + 1)
    te.to(torch.bfloat16).save_pretrained(
        os.path.join(out, "text_encoder"), safe_serialization=True
    )

    vae = AutoencoderKLFlux2(**TINY_VAE)
    _randomize(vae, seed + 2, std=0.2)
    vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)

    os.makedirs(os.path.join(out, "scheduler"), exist_ok=True)
    sched_src = tokenizer_from and os.path.join(
        tokenizer_from, "scheduler", "scheduler_config.json"
    )
    if sched_src and os.path.isfile(sched_src):
        shutil.copy(sched_src, os.path.join(out, "scheduler", "scheduler_config.json"))
    else:
        with open(os.path.join(out, "scheduler", "scheduler_config.json"), "w") as f:
            json.dump(SCHEDULER, f, indent=2)

    if tokenizer_from:
        shutil.copytree(
            os.path.join(tokenizer_from, "tokenizer"),
            os.path.join(out, "tokenizer"),
            dirs_exist_ok=True,
        )
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(
            "tiny-flux2: random weights with FLUX.2-dev structure. Plumbing tests only; not a model.\n"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", default=os.environ.get("FLUX2_WEIGHTS"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=None, help="double-stream blocks (default 2)")
    ap.add_argument(
        "--num-single-layers", type=int, default=None, help="single-stream blocks (default 3)"
    )
    a = ap.parse_args()
    print(build(a.out, a.tokenizer_from, a.seed, a.num_layers, a.num_single_layers))


if __name__ == "__main__":
    main()
