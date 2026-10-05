# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 accuracy tier 1 on a NeuronCore: components vs their CPU fp32 / bf16 twins.

Three-way comparison (``vllm_neuron.accuracy.testing.assert_close_three_way``): fp32 CPU baseline,
bf16 CPU expected (dtype error alone), bf16 Neuron actual (adds the Neuron-specific error).

* DiT and text encoder: the tiny random-weight structure model (``test_flux2_tiny.build``), TP=1, on one
  core. It has the real block types, attention layout, RoPE and fused projections, so it exercises the
  same compiled graphs at small widths. The real-weight DiT does not fit one core unsharded, so it is
  covered at TP=8 by the single-step pipeline test (tier 2).
* VAE: real weights (``FLUX2_WEIGHTS``). The tiled decode with whole-image GroupNorm statistics
  runs on one core and is compared with the same algorithm on CPU. Its agreement with the untiled fp32
  CPU decode is asserted separately (>= 35 dB PSNR).

Skipped without a Neuron device. Run on ONE visible core (e.g. ``NEURON_RT_VISIBLE_CORES=0``).
"""

from __future__ import annotations

import math
import os

import pytest
import torch

FLUX2_WEIGHTS = os.environ.get("FLUX2_WEIGHTS", "")


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


def _psnr(a, b):
    mse = float(((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from ..unit.test_flux2_tiny import build

    tok = FLUX2_WEIGHTS if os.path.isdir(os.path.join(FLUX2_WEIGHTS, "tokenizer")) else None
    return build(str(tmp_path_factory.mktemp("tiny-flux2")), tok)


def _backend():
    from vllm_neuron.envs import get_compile_backend_name

    return get_compile_backend_name()


def test_dit_three_way(vllm_single_rank, tiny):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    from vllm_omni_neuron.diffusion.models.flux2.transformer_flux2 import NeuronFlux2Transformer

    from ..unit.test_flux2_components import dit_inputs

    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = NeuronFlux2Transformer(tiny, dtype=dt)
        m.load_weights(tiny, "cpu")
        x, ctx, ii, ti, t, g = dit_inputs(m.cfg, s_txt=64, hw=(16, 16))
        with torch.no_grad():
            outs[dt] = m(
                hidden_states=x.to(dt),
                encoder_hidden_states=ctx.to(dt),
                timestep=t,
                img_ids=ii,
                txt_ids=ti,
                guidance=g,
                return_dict=False,
            )[0].float()
    m = NeuronFlux2Transformer(tiny, dtype=torch.bfloat16)
    m.load_weights(tiny, torch.device("privateuseone", 0))
    m.compile(_backend())
    with torch.no_grad():
        dev = m(
            hidden_states=x.bfloat16(),
            encoder_hidden_states=ctx.bfloat16(),
            timestep=t,
            img_ids=ii,
            txt_ids=ti,
            guidance=g,
            return_dict=False,
        )[0].float()
    assert_close_three_way(outs[torch.float32], outs[torch.bfloat16], dev, name="flux2_dit_tiny")


def test_text_encoder_three_way(vllm_single_rank, tiny):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    from vllm_omni_neuron.diffusion.models.flux2.text_encoder import NeuronFlux2TextEncoder

    g = torch.Generator().manual_seed(1)
    s = 128
    ids = torch.randint(100, 120000, (1, s), generator=g)
    mask = torch.ones(1, s, dtype=torch.long)
    mask[0, 100:] = 0
    ref = {}
    for dt in (torch.float32, torch.bfloat16):
        te = NeuronFlux2TextEncoder(tiny, dtype=dt)
        te.load_weights(tiny, "cpu")
        with torch.no_grad():
            hs = te(input_ids=ids, attention_mask=mask).hidden_states
        ref[dt] = [hs[k].float() for k in (10, 20, 30)]
    te = NeuronFlux2TextEncoder(tiny, dtype=torch.bfloat16)
    te.load_weights(tiny, torch.device("privateuseone", 0))
    te.compile(_backend())
    with torch.no_grad():
        hs = te(input_ids=ids, attention_mask=mask).hidden_states
    assert_close_three_way(
        ref[torch.float32],
        ref[torch.bfloat16],
        [hs[k].float() for k in (10, 20, 30)],
        name="flux2_text_encoder_tiny",
    )


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(FLUX2_WEIGHTS, "vae")), reason="set FLUX2_WEIGHTS"
)
def test_vae_tiled_global_gn_three_way():
    from diffusers import AutoencoderKLFlux2
    from vllm_neuron.accuracy.testing import assert_close_three_way

    os.environ.update(
        FLUX2_VAE_TILE="64",
        FLUX2_VAE_OVERLAP="16",
        FLUX2_VAE_GN_PASSES="2",
        FLUX2_VAE_UNTILED_MAX="0",
    )
    from vllm_omni_neuron.diffusion.models.flux2.vae_flux2 import NeuronFlux2Vae

    # 768 px: 2x2 tiles of the same 64-latent shape the 1024 px default uses (one NEFF per pass)
    z = torch.randn(1, 32, 96, 96, generator=torch.Generator().manual_seed(0))

    def load(dt):
        return AutoencoderKLFlux2.from_pretrained(
            FLUX2_WEIGHTS, subfolder="vae", torch_dtype=dt
        ).eval()

    with torch.no_grad():
        untiled32 = load(torch.float32).decode(z, return_dict=False)[0].float()
    cpu = {}
    for dt in (torch.float32, torch.bfloat16):
        v = NeuronFlux2Vae(load(dt))
        v._use_device, v.tile_parallel = True, False  # device algorithm, eager on CPU
        with torch.no_grad():
            cpu[dt] = v.decode(z, return_dict=False)[0].float()
    v = NeuronFlux2Vae(load(torch.bfloat16))
    v.tile_parallel = False
    v.to(torch.device("privateuseone", 0))
    v.compile(_backend())
    with torch.no_grad():
        dev = v.decode(z, return_dict=False)[0].float()
    assert_close_three_way(cpu[torch.float32], cpu[torch.bfloat16], dev, name="flux2_vae_tiled_gn2")
    assert _psnr(dev, untiled32) >= 35.0, _psnr(dev, untiled32)
