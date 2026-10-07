# SPDX-License-Identifier: Apache-2.0
"""pi0 (base) component accuracy (onboarding guide tier 1): same three-way methodology as
test_pi05_components_accuracy.py, against the pi0 M-tiny checkpoint."""

from __future__ import annotations

import os
import sys

import pytest
import torch


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


require_neuron_device = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "unit"))
import test_pi0_base_tiny as _tiny  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    ref = _tiny.REF_CHECKPOINT if os.path.isdir(_tiny.REF_CHECKPOINT) else None
    return _tiny.make_tiny_checkpoint(str(tmp_path_factory.mktemp("pi0_tiny_accuracy")), ref)


@require_neuron_device
def test_prefix_graph_three_way(tiny_dir):
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi0.config import Pi0Config

    cfg = Pi0Config.from_pretrained(tiny_dir)
    images, masks, tokens, tmask, _state, _noise = _tiny._inputs(cfg)
    pix = torch.stack(images, dim=1).reshape(cfg.max_cameras, *images[0].shape[1:]).float()
    iv = torch.stack([m.float() for m in masks], dim=1)

    m32 = NeuronPi0ActionModel(cfg, dtype=torch.float32)
    m32.load_checkpoint(tiny_dir)
    with torch.no_grad():
        k32, v32, _ = m32.prefix(pix, iv, tokens, tmask.float())
    baseline = torch.cat([k32.flatten(), v32.flatten()])

    m16 = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
    m16.load_checkpoint(tiny_dir)
    with torch.no_grad():
        k16, v16, _ = m16.prefix(pix, iv, tokens, tmask.float())
    expected = torch.cat([k16.float().flatten(), v16.float().flatten()])

    dev = torch.device("neuron", 0)
    mdev = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
    mdev.load_checkpoint(tiny_dir)
    mdev.to(dev)
    mdev.compile(get_compile_backend_name())
    with torch.no_grad():
        kd, vd, _ = mdev._prefix_fn(pix.to(dev), iv.to(dev), tokens.to(dev), tmask.float().to(dev))
    actual = torch.cat([kd.float().cpu().flatten(), vd.float().cpu().flatten()])

    assert_close_three_way(baseline, expected, actual, name="pi0_prefix_kv")


@require_neuron_device
def test_denoise_graph_three_way(tiny_dir):
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0ActionModel
    from vllm_omni_neuron.diffusion.models.pi0._vendor.pi0.config import Pi0Config
    from vllm_omni_neuron.diffusion.models.pi0.model_pi0 import _sinusoidal_time_embedding

    cfg = Pi0Config.from_pretrained(tiny_dir)
    images, masks, tokens, tmask, state, noise = _tiny._inputs(cfg)
    pix = torch.stack(images, dim=1).reshape(cfg.max_cameras, *images[0].shape[1:]).float()
    iv = torch.stack([m.float() for m in masks], dim=1)

    def one_step(model, device=None):
        dim_ = model.ref.action_in_proj.out_features
        tsc = _sinusoidal_time_embedding(torch.tensor([0.5], dtype=torch.float64), dim_).float()
        with torch.no_grad():
            if device is None:
                k, v, valid = model.prefix(pix, iv, tokens, tmask.float())
                out = model.denoise(state, noise, tsc, k, v, valid)
            else:
                k, v, valid = model._prefix_fn(
                    pix.to(device), iv.to(device), tokens.to(device), tmask.float().to(device)
                )
                out = model._denoise_fn(
                    state.to(device), noise.to(device), tsc.to(device), k, v, valid
                )
        return out.float().cpu()

    m32 = NeuronPi0ActionModel(cfg, dtype=torch.float32)
    m32.load_checkpoint(tiny_dir)
    baseline = one_step(m32)

    m16 = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
    m16.load_checkpoint(tiny_dir)
    expected = one_step(m16)

    dev = torch.device("neuron", 0)
    mdev = NeuronPi0ActionModel(cfg, dtype=torch.bfloat16)
    mdev.load_checkpoint(tiny_dir)
    mdev.to(dev)
    mdev.compile(get_compile_backend_name())
    actual = one_step(mdev, device=dev)

    assert_close_three_way(
        baseline.flatten(), expected.flatten(), actual.flatten(), name="pi0_denoise_v_t"
    )
