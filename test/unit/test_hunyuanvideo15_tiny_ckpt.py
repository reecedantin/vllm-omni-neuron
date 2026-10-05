# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight HunyuanVideo-1.5 checkpoint (M-tiny) + its layout test.

``make_tiny_checkpoint(out_dir)`` writes a Diffusers-layout checkpoint with the exact structure of
``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`` (same classes, module / parameter
names, attention layout, VAE / text-encoder wiring, ``model_index.json`` + component folders) but
shrunk dims and few layers, so loading, weight-name mapping, sharding, compilation and a device
forward can be exercised in minutes. It says nothing about quality.

It needs no download: the scheduler / guider configs are the real ones (inlined), and the two
tokenizers are synthetic (a byte-level BPE with the real Qwen chat special tokens and an empty
merge table, and a stock ``ByT5Tokenizer``), so token counts differ from the real model but the
diffusers reference and the Neuron pipeline see identical inputs. Generated weights are never
committed; regenerate with::

    python test/unit/test_hunyuanvideo15_tiny_ckpt.py <out_dir>

``HV15_REAL_WEIGHTS=<real 480p_t2v dir>`` additionally checks the tiny parameter names against it.
"""

from __future__ import annotations

import argparse
import json
import os

import pytest
import torch

REAL_DIR = os.environ.get("HV15_REAL_WEIGHTS", "")

# Shrunk transformer: 8 heads x 32 (TP up to 8 divides it), 3 dual-stream blocks, 2 refiner blocks.
TINY_TRANSFORMER = dict(
    in_channels=65,  # 2 x latent_channels + 1 (latents | cond latents | cond mask), as the real model
    out_channels=32,
    num_attention_heads=8,
    attention_head_dim=32,
    num_layers=3,
    num_refiner_layers=2,
    mlp_ratio=4.0,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    text_embed_dim=64,  # = tiny Qwen2.5-VL hidden
    text_embed_2_dim=32,  # = tiny byT5 d_model
    image_embed_dim=1152,  # SigLIP width; the pipeline hard-codes 729 x 1152 image tokens
    rope_theta=256.0,
    rope_axes_dim=(8, 12, 12),
    target_size=640,
    task_type="t2v",
    use_meanflow=False,
)
TINY_VAE = dict(block_out_channels=(32, 32, 64, 64, 64), layers_per_block=1, latent_channels=32)

# Real component configs of the 480p_t2v checkpoint (non-weight files, inlined).
MODEL_INDEX = {
    "_class_name": "HunyuanVideo15Pipeline",
    "_diffusers_version": "0.36.0.dev0",
    "guider": ["diffusers", "ClassifierFreeGuidance"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "tokenizer_2": ["transformers", "ByT5Tokenizer"],
    "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
}
SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
GUIDER = {
    "_class_name": "ClassifierFreeGuidance",
    "_diffusers_version": "0.36.0.dev0",
    "enabled": True,
    "guidance_rescale": 0.0,
    "guidance_scale": 6.0,
    "start": 0.0,
    "stop": 1.0,
    "use_original_formulation": False,
}
QWEN_SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _byte_unicode() -> list[str]:
    """GPT-2 / Qwen byte-level alphabet: one printable unicode char per byte value."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for _, c in sorted(zip(bs, cs))]


def _write_tokenizers(out_dir: str) -> int:
    """Synthetic Qwen2 (byte-level BPE, no merges) + ByT5 tokenizers; returns the Qwen vocab size."""
    from transformers import ByT5Tokenizer, Qwen2Tokenizer

    tdir = os.path.join(out_dir, "tokenizer")
    os.makedirs(tdir, exist_ok=True)
    vocab = {ch: i for i, ch in enumerate(_byte_unicode())}
    with open(os.path.join(tdir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(tdir, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
    tok = Qwen2Tokenizer(
        os.path.join(tdir, "vocab.json"),
        os.path.join(tdir, "merges.txt"),
        unk_token=None,
        bos_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=QWEN_SPECIALS[1:],
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(tdir)
    ByT5Tokenizer().save_pretrained(os.path.join(out_dir, "tokenizer_2"))
    return len(tok)


def _perturb_vectors(model: torch.nn.Module, seed: int) -> None:
    """Randomize 1-D params (norm scales, biases) so a mis-mapped norm/bias cannot pass a parity test."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.add_(0.1 * torch.randn(p.shape, generator=g).to(p.dtype))


def make_tiny_checkpoint(out_dir: str, seed: int = 0) -> str:
    from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
    from transformers import Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel, T5Config, T5EncoderModel

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    n_vocab = _write_tokenizers(out_dir)

    tf = HunyuanVideo15Transformer3DModel(**TINY_TRANSFORMER)
    _perturb_vectors(tf, seed + 1)
    tf.save_pretrained(os.path.join(out_dir, "transformer"), safe_serialization=True)

    vae = AutoencoderKLHunyuanVideo15(**TINY_VAE)
    _perturb_vectors(vae, seed + 2)
    vae.save_pretrained(os.path.join(out_dir, "vae"), safe_serialization=True)

    te_cfg = Qwen2_5_VLTextConfig(
        vocab_size=n_vocab,
        hidden_size=TINY_TRANSFORMER["text_embed_dim"],
        intermediate_size=128,
        num_hidden_layers=3,  # the pipeline reads hidden_states[-3]
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        rope_scaling={"type": "default", "rope_type": "default", "mrope_section": [2, 3, 3]},
    )
    te = Qwen2_5_VLTextModel(te_cfg).to(torch.bfloat16)
    _perturb_vectors(te, seed + 3)
    te.save_pretrained(os.path.join(out_dir, "text_encoder"), safe_serialization=True)

    t5_cfg = T5Config(
        vocab_size=1510,  # the real byT5 glyph encoder's vocabulary size
        d_model=TINY_TRANSFORMER["text_embed_2_dim"],
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=False,
        use_cache=False,
        dropout_rate=0.0,
    )
    t5 = T5EncoderModel(t5_cfg)
    _perturb_vectors(t5, seed + 4)
    t5.save_pretrained(os.path.join(out_dir, "text_encoder_2"), safe_serialization=True)

    for sub, cfg, name in (
        ("scheduler", SCHEDULER, "scheduler_config.json"),
        ("guider", GUIDER, "guider_config.json"),
    ):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        with open(os.path.join(out_dir, sub, name), "w") as f:
            json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2)
    return out_dir


# Tiny SigLIP: width 1152 (the DiT's image_embed_dim) and 27 x 27 = 729 patches, as the real image tokens.
TINY_SIGLIP = dict(
    hidden_size=1152,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    image_size=54,
    patch_size=2,
    num_channels=3,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
)


def make_tiny_i2v_checkpoint(out_dir: str, seed: int = 0, guidance_scale: float = 6.0) -> str:
    """The I2V variant (``HunyuanVideo15ImageToVideoPipeline`` layout of ``*_i2v`` / ``*_i2v_distilled``):
    the tiny T2V components plus a SigLIP image encoder + processor; ``guidance_scale`` 1.0 = CFG-distilled."""
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    make_tiny_checkpoint(out_dir, seed)
    enc = SiglipVisionModel(SiglipVisionConfig(**TINY_SIGLIP))
    _perturb_vectors(enc, seed + 5)
    enc.save_pretrained(os.path.join(out_dir, "image_encoder"), safe_serialization=True)
    SiglipImageProcessor(
        size={"height": TINY_SIGLIP["image_size"], "width": TINY_SIGLIP["image_size"]},
        resample=3,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ).save_pretrained(os.path.join(out_dir, "feature_extractor"))
    with open(os.path.join(out_dir, "transformer", "config.json")) as f:
        tcfg = json.load(f)
    tcfg["task_type"] = "i2v"
    with open(os.path.join(out_dir, "transformer", "config.json"), "w") as f:
        json.dump(tcfg, f, indent=2)
    idx = dict(
        MODEL_INDEX,
        _class_name="HunyuanVideo15ImageToVideoPipeline",
        image_encoder=["transformers", "SiglipVisionModel"],
        feature_extractor=["transformers", "SiglipImageProcessor"],
    )
    with open(os.path.join(out_dir, "model_index.json"), "w") as f:
        json.dump(idx, f, indent=2)
    with open(os.path.join(out_dir, "guider", "guider_config.json"), "w") as f:
        json.dump(dict(GUIDER, guidance_scale=guidance_scale), f, indent=2)
    with open(os.path.join(out_dir, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(dict(SCHEDULER, shift=7.0), f, indent=2)
    return out_dir


# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory) -> str:
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("hv15_tiny")))


@pytest.fixture(scope="session")
def tiny_i2v_ckpt(tmp_path_factory) -> str:
    return make_tiny_i2v_checkpoint(str(tmp_path_factory.mktemp("hv15_tiny_i2v")))


@pytest.fixture(scope="session")
def tiny_i2v_distilled_ckpt(tmp_path_factory) -> str:
    return make_tiny_i2v_checkpoint(
        str(tmp_path_factory.mktemp("hv15_tiny_i2v_distilled")), guidance_scale=1.0
    )


def test_tiny_layout(tiny_ckpt):
    with open(os.path.join(tiny_ckpt, "model_index.json")) as f:
        assert json.load(f) == MODEL_INDEX
    for comp in (
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    ):
        assert os.path.isdir(os.path.join(tiny_ckpt, comp)), comp


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(REAL_DIR, "model_index.json")),
    reason="set HV15_REAL_WEIGHTS to a real 480p_t2v checkout to compare parameter names",
)
def test_tiny_layout_mirrors_real(tiny_ckpt):
    from safetensors import safe_open

    with open(os.path.join(REAL_DIR, "model_index.json")) as f:
        assert json.load(f) == MODEL_INDEX

    def keys(d):
        idx = os.path.join(d, "transformer", "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx) as f:
                return set(json.load(f)["weight_map"])
        with safe_open(
            os.path.join(d, "transformer", "diffusion_pytorch_model.safetensors"), "pt"
        ) as f:
            return set(f.keys())

    def canon(ks):
        return {
            k
            for k in ks
            if not k.startswith("transformer_blocks.") or k.startswith("transformer_blocks.0.")
        }

    assert canon(keys(tiny_ckpt)) == canon(keys(REAL_DIR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_tiny_checkpoint(a.out_dir, a.seed))
