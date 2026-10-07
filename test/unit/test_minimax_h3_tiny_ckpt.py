# SPDX-License-Identifier: Apache-2.0
"""Write a random-weight MiniMax-H3 structure checkpoint ("tiny-h3-full") for tests.

Thin wrapper over the shared :mod:`vllm_omni_neuron.tiny_models` tool. The tool's ``instantiate`` mode needs the
model class, and diffusers 0.38 ships no MiniMax-H3 classes, so the ``class_resolver`` here returns the vendored
ones (``_vendor/``). Widths are shrunk per component; Qwen3-VL keeps its real vocabulary so the copied tokenizer /
processor work unchanged, and is read at ``text_encoder_layer`` (set in the written model index) of its 3 layers.

Usage::

    python test/unit/test_minimax_h3_tiny_ckpt.py --out <dir> [--src <real FastH3 / MiniMax-H3 checkout>]

Never commit the generated weights; commit this script. Pytest uses the ``h3_tiny`` fixture below.
"""

from __future__ import annotations

import argparse
import json
import os

TOKENIZER_FROM = os.environ.get(
    "MINIMAX_H3_WEIGHTS", ""
)  # local FastH3 / MiniMax-H3 checkout (configs + tokenizer)

# Per-component config overrides for tiny widths (names are the real configs' keys).
OVERRIDES = {
    "transformer": {
        "num_attention_heads": 8,
        "attention_head_dim": 128,
        "hidden_size": 256,
        "ffn_dim": 512,
        "text_dim": 64,
        "freq_dim": 32,
        "time_embed_hidden_dim": 128,
        "time_embed_dim": 64,
    },
    "vae": {
        "block_out_channels": [8, 16, 16, 16, 16, 32],
        "layers_per_block": 1,
        "norm_num_groups": 4,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 32,
        "decoder_ffn_mult": 2,
    },
    "audio_vae": {"encoder_dim": 8, "latent_dim": 32, "decoder_dim": 256, "num_attention_heads": 2},
    "text_encoder": {
        "text_config.hidden_size": 64,
        "text_config.intermediate_size": 128,
        "text_config.num_attention_heads": 4,
        "text_config.num_key_value_heads": 2,
        "text_config.head_dim": 16,
        "text_config.rope_scaling.mrope_section": [2, 3, 3],
        "vision_config.depth": 2,
        "vision_config.hidden_size": 32,
        "vision_config.intermediate_size": 64,
        "vision_config.num_heads": 2,
        "vision_config.out_hidden_size": 64,
        "vision_config.deepstack_visual_indexes": [0],
    },
}
MODES = {
    "transformer": "instantiate",
    "vae": "instantiate",
    "audio_vae": "instantiate",
    "text_encoder": "instantiate",
}


def _resolver(name: str, cfg: dict):
    """Return the vendored MiniMax-H3 class for a component config (diffusers 0.38 has none); defer the rest."""
    from vllm_omni_neuron.tiny_models import default_class_resolver

    cls_name = cfg.get("_class_name")
    mapping = {
        "MiniMaxH3Transformer3DModel": ("transformer_minimax_h3", "MiniMaxH3Transformer3DModel"),
        "AutoencoderKLMiniMaxH3": ("autoencoder_kl_minimax_h3", "AutoencoderKLMiniMaxH3"),
        "AutoencoderKLMiniMaxH3Audio": (
            "autoencoder_kl_minimax_h3_audio",
            "AutoencoderKLMiniMaxH3Audio",
        ),
    }
    if cls_name in mapping:
        import importlib

        mod = importlib.import_module(
            f"vllm_omni_neuron.diffusion.models.minimax_h3._vendor.{mapping[cls_name][0]}"
        )
        return getattr(mod, mapping[cls_name][1])
    return default_class_resolver(name, cfg)


def write_tiny(out: str, src: str = TOKENIZER_FROM, seed: int = 0, layers: int = 2) -> str:
    from vllm_omni_neuron.tiny_models import make_tiny_checkpoint

    text_encoder_layer = layers - 1  # layers.0..layers-2 kept pre-norm; the DiT reads one of those
    make_tiny_checkpoint(
        src,
        out,
        layers=layers,
        modes=MODES,
        overrides=OVERRIDES,
        seed=seed,
        class_resolver=_resolver,
    )
    # The DiT reads an intermediate hidden state; record which, so the pipeline / text encoder keep layer+1 layers.
    for name in ("model_index.json", "modular_model_index.json"):
        path = os.path.join(out, name)
        if os.path.isfile(path):
            with open(path) as f:
                index = json.load(f)
            index["text_encoder_layer"] = text_encoder_layer
            index["_structure_test_checkpoint"] = "random weights; tests only"
            with open(path, "w") as f:
                json.dump(index, f, indent=2)
    return out


try:
    import pytest

    @pytest.fixture(scope="session")
    def h3_tiny(tmp_path_factory) -> str:
        path = os.environ.get("MINIMAX_H3_TINY", "")
        if path and os.path.isdir(os.path.join(path, "transformer")):
            return path
        if not os.path.isdir(os.path.join(TOKENIZER_FROM, "tokenizer")):
            pytest.skip("set MINIMAX_H3_WEIGHTS to a local FastH3 / MiniMax-H3 checkout")
        return write_tiny(str(tmp_path_factory.mktemp("h3") / "tiny-h3-full"))

except ImportError:
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=TOKENIZER_FROM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2)
    a = ap.parse_args()
    print(write_tiny(a.out, a.src, a.seed, a.layers))
