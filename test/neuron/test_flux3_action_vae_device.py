# SPDX-License-Identifier: Apache-2.0
"""Accuracy tier 1 (Neuron): the video VAE encoder, three-way.

fp32 CPU (baseline) / bf16 CPU (the host encoder, dtype error alone) / bf16 Neuron (the encoder
compiled piecewise, neighborhood attention through the shared halo-tiled op) on one 544x736 DROID
canvas encode. Uses the tiny structure model's VAE by default, or the released VAE when
``FLUX3_ACTION_BASE`` points at a local flux-3-action-base copy. Skips without a Neuron device.
"""

from __future__ import annotations

import os

import pytest
import torch

from ..unit.test_flux3_action_utils.fixtures import tiny  # noqa: F401
from .test_flux3_action_dit_device import _neuron_available

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]


def test_vae_encoder_three_way(tiny):  # noqa: F811
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.flux3_action.video_vae import VideoVAE

    base = os.environ.get("FLUX3_ACTION_BASE") or tiny["base"]
    path = os.path.join(base, "video_vae.safetensors")
    x = torch.rand(1, 3, 544, 736, generator=torch.Generator().manual_seed(0)) * 2 - 1
    with torch.no_grad():
        v32 = VideoVAE.from_file(path, dtype=torch.float32)
        fp32 = v32.encode_frame_mu(x)
        vb = VideoVAE.from_file(path, dtype=torch.bfloat16)
        bf16 = vb.encode_frame_mu(x.to(torch.bfloat16))
        dev = torch.device("neuron", 0)
        vb.model.encoder.to(dev)
        vb.prepare_device_attention((544, 736), dev, torch.bfloat16)
        vb.compile_encoder(get_compile_backend_name(), None, COMPILER_ARGS)
        out = vb.encode_frame_mu(x.to(torch.bfloat16).to(dev).contiguous())
    assert len(vb._device_graphs) < len(vb._device_pieces)  # repeated blocks replay one graph
    assert_close_three_way(fp32, bf16, out, name="vae_encoder_mu", plot_on_failure=False)
