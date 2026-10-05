# SPDX-License-Identifier: Apache-2.0
"""Generate a random-weight Z-Image checkpoint with the real model's exact structure.

Same ``model_index.json`` + component folders, same module / parameter names, same layer types
(Qwen3 text encoder, ``ZImageTransformer2DModel``, Flux ``AutoencoderKL``, flow-match scheduler,
the real Qwen2 tokenizer), shrunk dims and few layers. It proves loading, weight-name mapping,
compile and the device forward in minutes; it says nothing about image quality.

    python test/unit/test_z_image_tiny_ckpt.py --src <real Z-Image dir (tokenizer/scheduler)> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

TINY_DIT = dict(
    all_patch_size=[2],
    all_f_patch_size=[1],
    axes_dims=[8, 12, 12],
    axes_lens=[1536, 512, 512],
    cap_feat_dim=64,
    dim=128,
    in_channels=16,
    n_heads=4,
    n_kv_heads=4,
    n_layers=2,
    n_refiner_layers=1,
    norm_eps=1e-5,
    qk_norm=True,
    rope_theta=256.0,
    t_scale=1000.0,
)
TINY_TE = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
)
TINY_VAE = dict(
    block_out_channels=[32, 32, 32, 32], layers_per_block=1, latent_channels=16, norm_num_groups=32
)


def _randomize(module: torch.nn.Module, gen: torch.Generator) -> None:
    for name, p in module.named_parameters():
        with torch.no_grad():
            if p.ndim == 1 and (
                "norm" in name
                or name.endswith("weight")
                and "conv" not in name
                and "embed" not in name
            ):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))


def make_tiny(src: str, out: str, seed: int = 0) -> str:
    from diffusers import AutoencoderKL, ZImageTransformer2DModel
    from transformers import AutoConfig, Qwen3ForCausalLM

    gen = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    for sub in ("tokenizer", "scheduler"):
        dst = os.path.join(out, sub)
        if not os.path.isdir(dst):
            shutil.copytree(os.path.join(src, sub), dst)
    shutil.copy(os.path.join(src, "model_index.json"), os.path.join(out, "model_index.json"))

    dit = ZImageTransformer2DModel(**TINY_DIT)
    _randomize(dit, gen)
    dit.save_pretrained(os.path.join(out, "transformer"))

    te_cfg = AutoConfig.from_pretrained(os.path.join(src, "text_encoder"))
    for k, v in TINY_TE.items():
        setattr(te_cfg, k, v)
    te_cfg.torch_dtype = "float32"
    if hasattr(te_cfg, "layer_types"):
        te_cfg.layer_types = ["full_attention"] * TINY_TE["num_hidden_layers"]
    te = Qwen3ForCausalLM(te_cfg)
    _randomize(te, gen)
    te.save_pretrained(os.path.join(out, "text_encoder"))

    with open(os.path.join(src, "vae", "config.json")) as f:
        vcfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    vcfg.update(TINY_VAE)
    vae = AutoencoderKL(**vcfg)
    _randomize(vae, gen)
    vae.save_pretrained(os.path.join(out, "vae"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.environ.get("WEIGHTS", ""), "z-image-turbo"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_tiny(a.src, a.out, a.seed))
