# SPDX-License-Identifier: Apache-2.0
"""Generate a random-weight FLUX 3 Action structure model ("tiny") with the released layout.

    python -m test.unit.test_flux3_action_utils.make_tiny OUT_DIR [--tokenizer-from DIR]

Writes ``OUT_DIR/base`` (shared encoders, like ``black-forest-labs/flux-3-action-base``) and
``OUT_DIR/droid`` (a DROID policy package, like ``flux-3-action-droid``):

    base/config.json  base/flux-3-action-base.safetensors  base/video_vae.safetensors
    base/video_vae.json (dims sidecar; absent on the real VAE)  base/text_encoder/{config.json,...}
    droid/config.native.json  droid/config.json  droid/model.safetensors  droid/manifest.json

Every module, parameter name, layer type, attention layout and the VAE / text-encoder wiring are
the real model's; only widths and depths shrink (DiT 256 wide / 4 heads / 2+3 blocks, VAE 64 wide /
one block per stage, Qwen3-VL text tower 64 wide with 34 layers so the 4..32 tap layers exist).
The tokenizer is copied from a real ``text_encoder`` when one is given (``HF_HUB_OFFLINE``-safe).
Random weights say nothing about quality; the model exists to prove loading, name mapping,
sharding, compile and a device forward in minutes. Never commit the generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil

import torch


def default_tokenizer_dir() -> str | None:
    """The Qwen3-VL tokenizer of a local ``flux-3-action-base`` copy (``$FLUX3_ACTION_BASE/text_encoder``)."""
    base = os.environ.get("FLUX3_ACTION_BASE")
    return os.path.join(base, "text_encoder") if base else None


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)

TINY_DIT = dict(
    hidden_size=256,
    num_heads=4,
    depth=2,
    depth_single_blocks=3,
    axes_dim=[16, 16, 16, 16],
    context_in_dim=8 * 64,
    vec_in_dim=768,
    mlp_ratio=3.0,
    theta=10000,
)
TINY_VAE = dict(
    embed_dim=64, enc_depths=[1, 1, 1, 1], dec_depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8]
)
TINY_TEXT = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=128,
    num_hidden_layers=34,
    mrope_section=[8, 4, 4],
)

DROID_NATIVE = {
    "action_dim": 8,
    "action_modality": "action_prediction_droid",
    "camera_layout": "droid",
    "camera_keys": ["images.wrist", "images.left", "images.right"],
    "canvas_hw": [544, 736],
    "chunk_size": 32,
    "n_action_steps": 32,
    "fps": 15.0,
    "action_scale": 2.0,
    "gripper_flip_dims": [-1],
    "action_parameterization": "absolute",
    "absolute_action_dims": [],
    "action_normalization": None,
    "state_normalization": None,
    "normalization_clip": 6.0,
    "camera_dropout": {},
    "trunk_weights": None,
    "content_streams": ["video", "video_cond"],
    "head_init_seed": 0,
    "torch_dtype": "bfloat16",
    "quantization": None,
    "attn_mode": "torch",
    "compile_model": False,
    "single_frame_encode": False,
    "inference_profile": "default",
    "sampler": "cosmos_unipc",
    "num_inference_steps": 4,
    "guidance_scale": 4.0,
    "guidance_scale_action": 1.0,
    "sampler_shift": 5.0,
    "inference_seed": 0,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lin(g: torch.Generator, out_f: int, in_f: int, gain: float = 1.0) -> torch.Tensor:
    return (torch.randn(out_f, in_f, generator=g) * (gain / math.sqrt(in_f))).to(torch.bfloat16)


def _norm_scale(g: torch.Generator, n: int) -> torch.Tensor:
    return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)


def dit_state_dict(
    dims: dict,
    modality: str,
    action_dim: int,
    seed: int,
    *,
    prefix: str = "dit.",
    heads: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Exactly the key set of upstream ``JointSingleSeq`` (video/video_cond + one action modality)."""
    g = torch.Generator().manual_seed(seed)
    hid, mlp = dims["hidden_size"], int(dims["hidden_size"] * dims["mlp_ratio"])
    hd = hid // dims["num_heads"]
    streams = {"video": 96, "video_cond": 96, modality: action_dim, f"{modality}_cond": action_dim}
    sd: dict[str, torch.Tensor] = {}

    def block(p: str) -> None:
        for n in ("q_proj", "k_proj", "v_proj", "attn_out"):
            sd[f"{p}.{n}.weight"] = _lin(g, hid, hid)
        sd[f"{p}.mlp_in.weight"] = _lin(g, 2 * mlp, hid)
        sd[f"{p}.mlp_out.weight"] = _lin(g, hid, mlp, 0.5)
        sd[f"{p}.norm.query_norm.scale"] = _norm_scale(g, hd)
        sd[f"{p}.norm.key_norm.scale"] = _norm_scale(g, hd)

    for s in sorted(streams):
        for i in range(dims["depth"]):
            block(f"content_mode_blocks.{s}.{i}")
    for i in range(dims["depth"]):
        block(f"txt_mode_blocks.{i}")
    for i in range(dims["depth_single_blocks"]):
        block(f"single_blocks.{i}")
    for kind in ("early_stream_modulations", "single_stream_modulations"):
        for s in (*sorted(streams), "txt"):
            sd[f"{kind}.{s}.lin.weight"] = _lin(g, 3 * hid, hid, 0.3)
    for s, c in streams.items():
        sd[f"emb_in.{s}.weight"] = _lin(g, hid, c)
    for s, c in (heads and {h: streams[h] for h in heads} or streams).items():
        sd[f"final_layer.{s}.linear.weight"] = _lin(g, c, hid)
        sd[f"final_layer.{s}.adaLN_modulation.1.weight"] = _lin(g, 2 * hid, hid, 0.3)
    sd["txt_in.weight"] = _lin(g, hid, dims["context_in_dim"])
    sd["time_in.in_layer.weight"] = _lin(g, hid, 256)
    sd["time_in.out_layer.weight"] = _lin(g, hid, hid)
    sd["vector_in.in_layer.weight"] = _lin(g, hid, dims["vec_in_dim"])
    sd["vector_in.out_layer.weight"] = _lin(g, hid, hid)
    return {prefix + k: v.contiguous() for k, v in sd.items()}


def make_vae(path: str, seed: int) -> None:
    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.flux3_action.video_vae import VideoVAE, VideoVAEParams

    torch.manual_seed(seed)
    params = VideoVAEParams(**TINY_VAE)
    vae = VideoVAE(params)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("norm.weight") or ".norm1.weight" in name or ".norm2.weight" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.ndim >= 2:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))
            else:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
        zn = vae.model.z_normalizer
        zn.running_mean.copy_(0.1 * torch.randn(zn.running_mean.shape, generator=g))
        zn.running_var.copy_(0.5 + torch.rand(zn.running_var.shape, generator=g))
        zn.initialized.fill_(True)
    sd = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
        for k, v in vae.state_dict().items()
    }
    save_file(sd, path)
    with open(os.path.join(os.path.dirname(path), "video_vae.json"), "w") as f:
        f.write(params.to_json() + "\n")


def make_text_encoder(out: str, seed: int, tokenizer_from: str | None) -> None:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    t = TINY_TEXT
    cfg = Qwen3VLConfig(
        text_config=dict(
            hidden_size=t["hidden_size"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_hidden_layers"],
            vocab_size=151936,
            max_position_embeddings=262144,
            rope_theta=5000000,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            bos_token_id=151643,
            eos_token_id=151645,
            rope_scaling={
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": t["mrope_section"],
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            num_heads=2,
            intermediate_size=64,
            out_hidden_size=t["hidden_size"],
            deepstack_visual_indexes=[0],
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=2304,
            in_channels=3,
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    src = tokenizer_from or default_tokenizer_dir()
    if not src or not os.path.isfile(os.path.join(src, "tokenizer.json")):
        raise FileNotFoundError(
            "a Qwen3-VL tokenizer is needed: pass tokenizer_from or set FLUX3_ACTION_BASE"
        )
    for name in TOKENIZER_FILES:
        if os.path.isfile(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(out, name))


def make_tiny(root: str, seed: int = 0, tokenizer_from: str | None = None) -> dict[str, str]:
    from safetensors.torch import save_file

    base, droid = os.path.join(root, "base"), os.path.join(root, "droid")
    os.makedirs(base, exist_ok=True)
    os.makedirs(droid, exist_ok=True)
    # shared encoders
    make_vae(os.path.join(base, "video_vae.safetensors"), seed)
    make_text_encoder(os.path.join(base, "text_encoder"), seed, tokenizer_from)
    trunk = dit_state_dict(TINY_DIT, "action_prediction", 32, seed + 7, prefix="")
    save_file(trunk, os.path.join(base, "flux-3-action-base.safetensors"))
    with open(os.path.join(base, "config.json"), "w") as f:
        json.dump(
            {
                "type": "flux3_action_base",
                "format_version": 1,
                "tiny": True,
                "weights": {
                    "action_base": "flux-3-action-base.safetensors",
                    "video_vae": "video_vae.safetensors",
                    "text_encoder": "text_encoder",
                },
                "num_tensors": len(trunk),
                "dtype": "bfloat16",
                "streams": ["video", "text", "action"],
            },
            f,
            indent=2,
        )
    # DROID policy package
    sd = dit_state_dict(TINY_DIT, DROID_NATIVE["action_modality"], 8, seed + 11)
    save_file(sd, os.path.join(droid, "model.safetensors"))
    native = dict(
        DROID_NATIVE,
        dit_config=dict(TINY_DIT),
        video_vae_id=os.path.join(base, "video_vae.safetensors"),
        text_encoder_id=os.path.join(base, "text_encoder"),
    )
    for name in ("config.native.json", "config.json"):
        with open(os.path.join(droid, name), "w") as f:
            json.dump(native, f, indent=2)
            f.write("\n")
    manifest = {
        "format_version": 1,
        "kind": "policy_export",
        "weight_profile": "model",
        "sha256": {
            n: _sha256(os.path.join(droid, n))
            for n in ("config.native.json", "model.safetensors", "config.json")
        },
    }
    with open(os.path.join(droid, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return {"base": base, "droid": droid}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer-from", default=None)
    a = ap.parse_args()
    print(json.dumps(make_tiny(a.out, a.seed, a.tokenizer_from)))


if __name__ == "__main__":
    main()
