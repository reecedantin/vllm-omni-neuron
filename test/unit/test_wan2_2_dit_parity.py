# SPDX-License-Identifier: Apache-2.0
"""CPU parity: the Neuron Wan2.2 DiT vs Diffusers' WanTransformer3DModel on tiny weights.

Covers every Wan2.2 layout this package serves: A14B T2V (16 ch), A14B I2V (36 ch in), TI2V-5B
(48 ch, Wan2.2 VAE latents), and the TI2V per-token timestep (``expand_timesteps``) used when an
image conditions the first latent frame.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("PJRT_DEVICE", "CPU")

from test.unit.test_wan2_2_tiny import (  # noqa: E402,F401
    TINY_TEXT,
    VARIANTS,
    make_tiny_checkpoint,
    require_tokenizer,
    wan_single_rank,
)


def _neuron_dit(path: str, sub: str = "transformer"):
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config

    from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2 import (
        _create_transformer_from_config,
    )

    cfg = load_transformer_config(path, sub, True)
    model = _create_transformer_from_config(cfg)
    model.load_weights(os.path.join(path, sub))
    return model.float().eval()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@pytest.fixture(scope="module", params=["t2v-a14b", "i2v-a14b", "ti2v-5b"])
def tiny(request, tmp_path_factory):
    require_tokenizer()
    out = tmp_path_factory.mktemp(f"tiny-dit-{request.param}")
    return request.param, make_tiny_checkpoint(str(out), request.param)


@pytest.mark.usefixtures("wan_single_rank")
def test_dit_matches_diffusers(tiny):
    from diffusers import WanTransformer3DModel

    name, path = tiny
    variant = VARIANTS[name]
    ref = WanTransformer3DModel.from_pretrained(path, subfolder="transformer").float().eval()
    ours = _neuron_dit(path)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, variant.in_channels, 3, 8, 12, generator=g)
    ctx = torch.randn(1, 32, TINY_TEXT["d_model"], generator=g)
    t = torch.tensor([637.0])
    with torch.no_grad():
        want = ref(x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0]
        got = ours(x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0]
    assert got.shape == want.shape
    err = _rel_l2(got, want)
    assert err < 1e-4, f"{name}: rel-L2 {err:.2e}"


@pytest.mark.usefixtures("wan_single_rank")
def test_dit_expand_timesteps_matches_diffusers(tmp_path):
    """TI2V image conditioning: frame 0 tokens at t=0, the rest at t (per-token AdaLN)."""
    from diffusers import WanTransformer3DModel

    require_tokenizer()
    path = make_tiny_checkpoint(str(tmp_path / "ti2v"), "ti2v-5b")
    ref = WanTransformer3DModel.from_pretrained(path, subfolder="transformer").float().eval()
    ours = _neuron_dit(path)

    g = torch.Generator().manual_seed(1)
    frames, h, w = 3, 8, 12
    x = torch.randn(1, 48, frames, h, w, generator=g)
    ctx = torch.randn(1, 32, TINY_TEXT["d_model"], generator=g)
    mask = torch.ones(frames, h // 2, w // 2)
    mask[0] = 0
    ts = (mask * 811.0).flatten().unsqueeze(0)  # [B, S]
    with torch.no_grad():
        want = ref(x, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
        got = ours(x, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
    err = _rel_l2(got, want)
    assert err < 1e-4, f"rel-L2 {err:.2e}"
