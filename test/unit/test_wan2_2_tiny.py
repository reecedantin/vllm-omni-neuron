# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight Wan2.2 checkpoints with the real on-disk layout.

Each variant mirrors one real Diffusers checkpoint: the same ``model_index.json``, component
folders, class names, parameter names, channel counts, patch sizes and VAE topology (Wan2.1 VAE
for the A14B models, the residual/patchified Wan2.2 VAE for TI2V-5B), with every width shrunk and
two DiT layers. Loading, weight-name mapping, TP sharding, compile and one device forward can then
be exercised in minutes. The weights are random: these checkpoints say nothing about quality.

Generate (CPU, a few seconds per variant)::

    python test/unit/test_wan2_2_tiny.py --out <dir> [--variant t2v-a14b ...]

The tests here check that a generated checkpoint loads in Diffusers and that its parameter names
match the real checkpoint's (when ``WAN22_WEIGHTS_ROOT`` points at the real weights).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass

import pytest
import torch

# Tiny widths. The DiT keeps head_dim 128 (the attention kernels' native tile) and a head count
# divisible by TP=4; the text encoder keeps the UMT5 vocabulary so the real tokenizer works.
TINY_DIT = {"num_attention_heads": 4, "attention_head_dim": 128, "ffn_dim": 1024, "num_layers": 2}
TINY_TEXT = {"d_model": 256, "num_heads": 4, "d_kv": 64, "d_ff": 512, "num_layers": 2}
TINY_VAE21 = {"base_dim": 32}
TINY_VAE22 = {"base_dim": 32, "decoder_base_dim": 32}


@dataclass(frozen=True)
class Variant:
    name: str
    real_dir: str  # directory name under the weights root
    pipeline_class: str
    in_channels: int
    out_channels: int
    two_experts: bool
    wan22_vae: bool
    boundary_ratio: float | None
    flow_shift: float
    dit_ffn_dim: int = TINY_DIT["ffn_dim"]


VARIANTS = {
    "t2v-a14b": Variant(
        "t2v-a14b", "wan22-t2v-a14b", "WanPipeline", 16, 16, True, False, 0.875, 3.0
    ),
    "i2v-a14b": Variant(
        "i2v-a14b", "wan22-i2v-a14b", "WanImageToVideoPipeline", 36, 16, True, False, 0.9, 3.0
    ),
    "ti2v-5b": Variant("ti2v-5b", "wan22-ti2v-5b", "WanPipeline", 48, 48, False, True, None, 5.0),
    "dmd2-ti2v-5b": Variant(
        "dmd2-ti2v-5b", "wan22-dmd2-ti2v-5b", "WanDMDPipeline", 48, 48, False, True, None, 5.0
    ),
}

_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _dit_config(variant: Variant) -> dict:
    return {
        "added_kv_proj_dim": None,
        "attention_head_dim": TINY_DIT["attention_head_dim"],
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": variant.dit_ffn_dim,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": variant.in_channels,
        "num_attention_heads": TINY_DIT["num_attention_heads"],
        "num_layers": TINY_DIT["num_layers"],
        "out_channels": variant.out_channels,
        "patch_size": [1, 2, 2],
        "pos_embed_seq_len": None,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": TINY_TEXT["d_model"],
    }


def _vae_kwargs(variant: Variant) -> dict:
    if not variant.wan22_vae:
        return {
            "base_dim": TINY_VAE21["base_dim"],
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
        }
    g = torch.Generator().manual_seed(7)
    return {
        "base_dim": TINY_VAE22["base_dim"],
        "decoder_base_dim": TINY_VAE22["decoder_base_dim"],
        "z_dim": 48,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 4,
        "latents_mean": (torch.randn(48, generator=g) * 0.2).tolist(),
        "latents_std": (torch.rand(48, generator=g) * 0.8 + 0.4).tolist(),
    }


def _scheduler_config(variant: Variant) -> dict:
    return {
        "_class_name": "UniPCMultistepScheduler",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "disable_corrector": [],
        "dynamic_thresholding_ratio": 0.995,
        "final_sigmas_type": "zero",
        "flow_shift": variant.flow_shift,
        "lower_order_final": True,
        "num_train_timesteps": 1000,
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "rescale_betas_zero_snr": False,
        "sample_max_value": 1.0,
        "solver_order": 2,
        "solver_p": None,
        "solver_type": "bh2",
        "steps_offset": 0,
        "thresholding": False,
        "time_shift_type": "exponential",
        "timestep_spacing": "linspace",
        "trained_betas": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_flow_sigmas": True,
        "use_karras_sigmas": False,
    }


def _model_index(variant: Variant) -> dict:
    index = {
        "_class_name": variant.pipeline_class,
        "_diffusers_version": "0.35.0.dev0",
        "boundary_ratio": variant.boundary_ratio,
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": (
            ["diffusers", "WanTransformer3DModel"] if variant.two_experts else [None, None]
        ),
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
    if variant.pipeline_class == "WanImageToVideoPipeline":
        index["image_encoder"] = [None, None]
        index["image_processor"] = [None, None]
    if variant.wan22_vae:
        index["expand_timesteps"] = True
    return index


def _find_tokenizer(tokenizer_src: str | None) -> str:
    candidates = [tokenizer_src] if tokenizer_src else []
    root = os.environ.get("WAN22_WEIGHTS_ROOT", "")
    if root:
        candidates += [os.path.join(root, v.real_dir, "tokenizer") for v in VARIANTS.values()]
    for path in candidates:
        if path and all(os.path.isfile(os.path.join(path, f)) for f in _TOKENIZER_FILES):
            return path
    raise FileNotFoundError(
        "no UMT5 tokenizer found; pass --tokenizer <dir with spiece.model/tokenizer.json> or set "
        "WAN22_WEIGHTS_ROOT to the directory holding the Wan2.2 checkpoints"
    )


def require_tokenizer() -> None:
    """Skip the calling test when no UMT5 tokenizer is available to build tiny checkpoints."""
    try:
        _find_tokenizer(None)
    except FileNotFoundError:
        pytest.skip("no UMT5 tokenizer; set WAN22_WEIGHTS_ROOT")


def make_tiny_checkpoint(
    out_dir: str,
    variant_name: str,
    *,
    seed: int = 0,
    tokenizer_src: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Write one tiny checkpoint to ``out_dir`` and return it."""
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    variant = VARIANTS[variant_name]
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    experts = ["transformer", "transformer_2"] if variant.two_experts else ["transformer"]
    for i, sub in enumerate(experts):
        torch.manual_seed(seed + 101 * (i + 1))
        dit = WanTransformer3DModel(**_dit_config(variant))
        # Diffusers zero-inits nothing important here, but scale_shift_table and the
        # projections are tiny random; widen the init so the forward is not near-degenerate.
        with torch.no_grad():
            for p in dit.parameters():
                if p.ndim >= 2:
                    p.normal_(0.0, 0.02)
                else:
                    p.normal_(0.0, 0.01)
        dit.to(dtype).save_pretrained(os.path.join(out_dir, sub), safe_serialization=True)

    torch.manual_seed(seed + 7)
    vae = AutoencoderKLWan(**_vae_kwargs(variant))
    vae.to(torch.float32).save_pretrained(os.path.join(out_dir, "vae"), safe_serialization=True)
    if variant.wan22_vae:
        # The real config carries ``clip_output`` (unknown to this diffusers' constructor, which
        # from_pretrained ignores); keep it so the config keys match the real checkpoint.
        vae_cfg_path = os.path.join(out_dir, "vae", "config.json")
        with open(vae_cfg_path) as f:
            vae_cfg = json.load(f)
        vae_cfg["clip_output"] = False
        with open(vae_cfg_path, "w") as f:
            json.dump(vae_cfg, f, indent=2)

    torch.manual_seed(seed + 11)
    text_cfg = UMT5Config(
        vocab_size=256384,
        d_model=TINY_TEXT["d_model"],
        d_kv=TINY_TEXT["d_kv"],
        d_ff=TINY_TEXT["d_ff"],
        num_layers=TINY_TEXT["num_layers"],
        num_decoder_layers=TINY_TEXT["num_layers"],
        num_heads=TINY_TEXT["num_heads"],
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        scalable_attention=True,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tokenizer_class="T5Tokenizer",
    )
    text_encoder = UMT5EncoderModel(text_cfg)
    text_encoder.to(dtype).save_pretrained(
        os.path.join(out_dir, "text_encoder"), safe_serialization=True
    )

    tok_dst = os.path.join(out_dir, "tokenizer")
    os.makedirs(tok_dst, exist_ok=True)
    tok_src = _find_tokenizer(tokenizer_src)
    for f in _TOKENIZER_FILES:
        shutil.copyfile(os.path.join(tok_src, f), os.path.join(tok_dst, f))

    os.makedirs(os.path.join(out_dir, "scheduler"), exist_ok=True)
    with open(os.path.join(out_dir, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(_scheduler_config(variant), f, indent=2)
    with open(os.path.join(out_dir, "model_index.json"), "w") as f:
        json.dump(_model_index(variant), f, indent=2)
    with open(os.path.join(out_dir, "TINY.md"), "w") as f:
        f.write(
            f"Tiny random-weight structure model for {variant.real_dir} (seed {seed}). "
            "Not a real model; says nothing about quality.\n"
        )
    return out_dir


# ---------------------------------------------------------------------------------------------
# fixtures shared by the Wan2.2 CPU tests
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wan_single_rank():
    """World size 1: vLLM TP group plus vllm-omni's diffusion groups (CP/CFG/...), on gloo."""
    import socket

    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as vllm_ps
    from vllm_omni.diffusion.distributed import parallel_state as omni_ps

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    if not omni_ps.model_parallel_is_initialized():
        # vllm-omni builds vLLM's TP group itself; its same-node probe needs no device here.
        vllm_ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
        omni_ps.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        omni_ps.initialize_model_parallel(backend="gloo")
    yield
    ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------


def _safetensors_keys(component_dir: str) -> set[str]:
    index = os.path.join(component_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return set(json.load(f)["weight_map"])
    from safetensors import safe_open

    keys: set[str] = set()
    for name in os.listdir(component_dir):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(component_dir, name), "pt") as f:
                keys |= set(f.keys())
    return keys


def _strip_layers(keys: set[str], num_layers: int) -> set[str]:
    """Keep only keys of the first ``num_layers`` blocks (and all non-block keys)."""
    kept = set()
    for k in keys:
        if k.startswith("blocks."):
            if int(k.split(".")[1]) >= num_layers:
                continue
        kept.add(k)
    return kept


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def tiny_ckpt(request, tmp_path_factory):
    require_tokenizer()
    out = tmp_path_factory.mktemp(f"tiny-{request.param}")
    return request.param, make_tiny_checkpoint(str(out), request.param)


def test_tiny_layout_matches_real(tiny_ckpt):
    name, path = tiny_ckpt
    variant = VARIANTS[name]
    with open(os.path.join(path, "model_index.json")) as f:
        index = json.load(f)
    assert index["_class_name"] == variant.pipeline_class
    assert os.path.isdir(os.path.join(path, "transformer_2")) == variant.two_experts

    root = os.environ.get("WAN22_WEIGHTS_ROOT", "")
    real = os.path.join(root, variant.real_dir)
    if not root or not os.path.isdir(real):
        pytest.skip(f"real checkpoint {real} not present")
    with open(os.path.join(real, "model_index.json")) as f:
        real_index = json.load(f)
    for key in ("_class_name", "transformer", "transformer_2", "vae", "text_encoder", "tokenizer"):
        assert index[key] == real_index[key], key
    assert index.get("expand_timesteps", False) == real_index.get("expand_timesteps", False)

    for sub in ("transformer", "transformer_2") if variant.two_experts else ("transformer",):
        tiny_keys = _safetensors_keys(os.path.join(path, sub))
        real_keys = _strip_layers(
            _safetensors_keys(os.path.join(real, sub)), TINY_DIT["num_layers"]
        )
        assert tiny_keys == real_keys, sorted(tiny_keys ^ real_keys)[:10]
    tiny_vae = _safetensors_keys(os.path.join(path, "vae"))
    real_vae = _safetensors_keys(os.path.join(real, "vae"))
    assert tiny_vae == real_vae, sorted(tiny_vae ^ real_vae)[:10]

    with open(os.path.join(real, "vae", "config.json")) as f:
        real_vae_cfg = json.load(f)
    with open(os.path.join(path, "vae", "config.json")) as f:
        tiny_vae_cfg = json.load(f)
    assert set(real_vae_cfg) <= set(tiny_vae_cfg) | {"_diffusers_version"}
    for key in ("z_dim", "is_residual", "patch_size", "in_channels", "temperal_downsample"):
        if key in real_vae_cfg:
            assert tiny_vae_cfg.get(key) == real_vae_cfg[key], key


def test_tiny_runs_in_diffusers(tiny_ckpt):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    name, path = tiny_ckpt
    variant = VARIANTS[name]
    dit = WanTransformer3DModel.from_pretrained(path, subfolder="transformer").float()
    x = torch.randn(1, variant.in_channels, 2, 8, 8)
    ctx = torch.randn(1, 16, TINY_TEXT["d_model"])
    out = dit(x, timestep=torch.tensor([500.0]), encoder_hidden_states=ctx, return_dict=False)[0]
    assert out.shape == (1, variant.out_channels, 2, 8, 8)
    assert torch.isfinite(out).all()

    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae")
    spatial = 16 if variant.wan22_vae else 8
    z = torch.randn(1, vae.config.z_dim, 2, 2, 2)
    video = vae.decode(z, return_dict=False)[0]
    assert video.shape == (1, 3, 5, 2 * spatial, 2 * spatial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="root dir; one sub-dir per variant")
    parser.add_argument("--variant", nargs="*", default=sorted(VARIANTS), choices=sorted(VARIANTS))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in args.variant:
        path = make_tiny_checkpoint(
            os.path.join(args.out, f"tiny-wan22-{name}"),
            name,
            seed=args.seed,
            tokenizer_src=args.tokenizer,
        )
        print(path)


if __name__ == "__main__":
    main()
