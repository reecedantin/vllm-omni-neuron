# SPDX-License-Identifier: Apache-2.0
"""Generate a random-weight "tiny LTX-2.5" checkpoint (structure test model, NOT a real model).

Same classes, module/parameter names, attention layout, connector / VAE / audio-VAE / vocoder
wiring and on-disk layout (``model_index.json`` + component folders) as
``Lightricks/LTX-2.5-Diffusers``; dims shrunk and few layers. Head counts stay divisible by 4
so TP=4 sharding is exercised. Weights are random (std 0.02, AdaLN tables ~N(0,1)/sqrt(dim))
so every path carries signal; the output is meaningless as video/audio.

Usage::

    python -m test.unit.test_ltx2_tiny --out tiny-ltx25 [--real /path/to/LTX-2.5-Diffusers]

``--real`` (default ``$LTX25_WEIGHTS``) is only read for the tokenizer / scheduler / processor
files and the real component configs the tiny ones are derived from (the transformer config is
embedded below, so the CPU unit tests need no weights). Components the Neuron port does not serve
yet (prompt enhancer, diffusion decoder, latent upsamplers, duration head) are left out of
``model_index.json``. This module has no tests of its own; the ``test_`` prefix keeps it inside
the package's test namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

TINY_TRANSFORMER = dict(
    num_layers=4,
    num_attention_heads=8,
    attention_head_dim=32,  # video dim 256
    audio_num_attention_heads=8,
    audio_attention_head_dim=16,  # audio dim 128
    cross_attention_dim=256,
    audio_cross_attention_dim=128,
    caption_channels=64,
)
TINY_TEXT = dict(
    hidden_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    num_global_key_value_heads=1,
    head_dim=32,
    global_head_dim=64,
    intermediate_size=128,
    num_hidden_layers=2,
)


def _randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1] ** 0.5)
            elif p.ndim == 1 and ("norm" in name or name.endswith("weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def _load_cfg(real: str | None, sub: str) -> dict:
    if sub == "transformer" and (real is None or not os.path.isdir(os.path.join(real, sub))):
        return dict(LTX25_TRANSFORMER_CONFIG)
    with open(os.path.join(real, sub, "config.json")) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


# Lightricks/LTX-2.5-Diffusers transformer/config.json, so the DiT tests need no real weights.
LTX25_TRANSFORMER_CONFIG = dict(
    activation_fn="gelu-approximate",
    attention_bias=True,
    attention_head_dim=128,
    attention_out_bias=True,
    audio_attention_head_dim=64,
    audio_cross_attention_dim=2048,
    audio_cross_attn_mod=True,
    audio_ff_bias=True,
    audio_gated_attn=True,
    audio_hop_length=160,
    audio_in_channels=128,
    audio_num_attention_heads=32,
    audio_out_channels=128,
    audio_patch_size=1,
    audio_patch_size_t=1,
    audio_pos_embed_max_pos=20,
    audio_sampling_rate=16000,
    audio_scale_factor=4,
    base_height=2048,
    base_width=2048,
    caption_channels=3840,
    causal_offset=1,
    cross_attention_dim=4096,
    cross_attn_mod=True,
    cross_attn_timestep_scale_multiplier=1000,
    ff_bias=False,
    gated_attn=True,
    in_channels=128,
    norm_elementwise_affine=False,
    norm_eps=1e-06,
    num_attention_heads=32,
    num_layers=48,
    out_channels=128,
    patch_size=1,
    patch_size_t=1,
    perturbed_attn=True,
    pos_embed_max_pos=20,
    qk_norm="rms_norm_across_heads",
    rope_double_precision=True,
    rope_theta=10000.0,
    rope_type="split",
    timestep_scale_multiplier=1000,
    use_keyframes_abs_pos_embedding=True,
    use_prompt_adaln_single=True,
    use_prompt_embeddings=False,
    vae_scale_factors=[8, 32, 32],
)


def build_transformer(real: str, seed: int = 0):
    from vllm_omni_neuron.diffusion.models.ltx2._vendor.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )

    cfg = _load_cfg(real, "transformer")
    cfg.update(TINY_TRANSFORMER)
    m = LTX2VideoTransformer3DModel(**cfg)
    _randomize(m, seed)
    return m


def build_connectors(real: str, seed: int = 1):
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    cfg = _load_cfg(real, "connectors")
    t = TINY_TRANSFORMER
    cfg.update(
        caption_channels=TINY_TEXT["hidden_size"],
        text_proj_in_factor=TINY_TEXT["num_hidden_layers"] + 1,
        video_connector_num_attention_heads=t["num_attention_heads"],
        video_connector_attention_head_dim=t["attention_head_dim"],
        video_hidden_dim=t["num_attention_heads"] * t["attention_head_dim"],
        audio_connector_num_attention_heads=t["audio_num_attention_heads"],
        audio_connector_attention_head_dim=t["audio_attention_head_dim"],
        audio_hidden_dim=t["audio_num_attention_heads"] * t["audio_attention_head_dim"],
        video_connector_num_layers=1,
        audio_connector_num_layers=1,
    )
    m = LTX2TextConnectors(**cfg)
    _randomize(m, seed)
    return m


def build_text_encoder(real: str, seed: int = 2):
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

    with open(os.path.join(real, "text_encoder", "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc.update(TINY_TEXT)
    tc["layer_types"] = ["sliding_attention", "full_attention"]
    for sub in ("vision_config",):
        if cfg.get(sub):
            cfg[sub].update(
                mm_embed_dim=TINY_TEXT["hidden_size"], output_proj_dims=TINY_TEXT["hidden_size"]
            )
    cfg.pop("ltx_source_checkpoint", None)
    config = Gemma4UnifiedConfig(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    torch.manual_seed(seed)
    m = Gemma4UnifiedForConditionalGeneration(config)
    return m


def build_vae(real: str, seed: int = 3):
    from diffusers import AutoencoderKLLTX2Video

    cfg = _load_cfg(real, "vae")
    cfg.update(
        block_out_channels=[8, 16, 32, 32],
        decoder_block_out_channels=[8, 16, 16, 32],
        layers_per_block=[1, 1, 1, 1, 1],
        decoder_layers_per_block=[1, 1, 1, 1, 1],
    )
    m = AutoencoderKLLTX2Video(**cfg)
    _randomize(m, seed)
    return m


def build_audio_vae(real: str, seed: int = 4):
    from diffusers import AutoencoderKLLTX2Audio

    cfg = _load_cfg(real, "audio_vae")
    cfg.update(base_channels=16)
    m = AutoencoderKLLTX2Audio(**cfg)
    _randomize(m, seed)
    return m


def build_vocoder(real: str, seed: int = 5):
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    cfg = _load_cfg(real, "vocoder")
    cfg.update(hidden_channels=32, bwe_hidden_channels=16)
    m = LTX2VocoderWithBWE(**cfg)
    _randomize(m, seed)
    return m


BUILDERS = {
    "transformer": build_transformer,
    "connectors": build_connectors,
    "text_encoder": build_text_encoder,
    "vae": build_vae,
    "audio_vae": build_audio_vae,
    "vocoder": build_vocoder,
}
COPIED = ("tokenizer", "scheduler", "processor")


def generate(out: str, real: str, components=None) -> str:
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(
        real
    ):  # HF repo id: fetch only configs + tokenizer/processor/scheduler files
        from huggingface_hub import snapshot_download

        real = snapshot_download(
            real,
            allow_patterns=[
                "model_index.json",
                "*/config.json",
                "tokenizer/*",
                "processor/*",
                "scheduler/*",
            ],
        )
    components = list(components or BUILDERS)
    with open(os.path.join(real, "model_index.json")) as f:
        index = json.load(f)
    keep = set(BUILDERS) | set(COPIED)
    index = {k: v for k, v in index.items() if k.startswith("_") or k in keep}
    index["_name_or_path"] = "tiny-ltx25 (random weights, structure test only)"
    for name in components:
        m = BUILDERS[name](real)
        m.to(torch.bfloat16).save_pretrained(os.path.join(out, name), safe_serialization=True)
        print(f"[tiny-ltx25] {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")
    for name in COPIED:
        src, dst = os.path.join(real, name), os.path.join(out, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--real", default=os.environ.get("LTX25_WEIGHTS", "Lightricks/LTX-2.5-Diffusers")
    )
    ap.add_argument("--components", nargs="*", default=None)
    a = ap.parse_args()
    generate(a.out, a.real, a.components)


if __name__ == "__main__":
    main()
