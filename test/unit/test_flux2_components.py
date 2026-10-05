# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Neuron FLUX.2 components against diffusers / transformers (tiny-flux2).

The tiny checkpoint is generated on the fly (``test_flux2_tiny.py``) unless ``FLUX2_TINY`` points at
one. Runs in fp32 so any difference is a math / weight-mapping bug, not dtype noise.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

FLUX2_WEIGHTS = os.environ.get("FLUX2_WEIGHTS", "")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def tiny_flux2(tmp_path_factory) -> str:
    path = os.environ.get("FLUX2_TINY")
    if path and os.path.isdir(os.path.join(path, "transformer")):
        return path
    from .test_flux2_tiny import build

    tok = FLUX2_WEIGHTS if os.path.isdir(os.path.join(FLUX2_WEIGHTS, "tokenizer")) else None
    return build(str(tmp_path_factory.mktemp("tiny-flux2")), tok)


@pytest.fixture(scope="session")
def vllm_tp1():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    yield
    ctx.__exit__(None, None, None)


def rel(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def dit_inputs(cfg, s_txt=24, hw=(4, 6), seed=0):
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    g = torch.Generator().manual_seed(seed)
    h, w = hw
    x = torch.randn(1, cfg.in_channels, h, w, generator=g)
    img_ids = Flux2Pipeline._prepare_latent_ids(x)
    x = x.flatten(2).transpose(1, 2)
    ctx = torch.randn(1, s_txt, cfg.joint_attention_dim, generator=g)
    txt_ids = Flux2Pipeline._prepare_text_ids(ctx)
    return x, ctx, img_ids, txt_ids, torch.tensor([0.63]), torch.tensor([4.0])


def test_dit_matches_diffusers(tiny_flux2, vllm_tp1):
    from diffusers import Flux2Transformer2DModel

    from vllm_omni_neuron.diffusion.models.flux2.transformer_flux2 import NeuronFlux2Transformer

    ref = Flux2Transformer2DModel.from_pretrained(
        tiny_flux2, subfolder="transformer", torch_dtype=torch.float32
    )
    ours = NeuronFlux2Transformer(tiny_flux2, dtype=torch.float32)
    ours.load_weights(tiny_flux2, "cpu")
    x, ctx, img_ids, txt_ids, t, gd = dit_inputs(ours.cfg)
    with torch.no_grad():
        want = ref(
            hidden_states=x,
            encoder_hidden_states=ctx,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=gd,
            return_dict=False,
        )[0]
        got = ours(
            hidden_states=x,
            encoder_hidden_states=ctx,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=gd,
            return_dict=False,
        )[0]
    assert got.shape == want.shape
    assert rel(got, want) < 1e-5, rel(got, want)


def test_dit_step_dump(tiny_flux2, vllm_tp1, tmp_path, monkeypatch):
    """FLUX2_STEP_DUMP: the selected DiT calls save their inputs and output for CPU replay."""
    from vllm_omni_neuron.diffusion.models.flux2.transformer_flux2 import NeuronFlux2Transformer

    monkeypatch.setenv("FLUX2_STEP_DUMP", str(tmp_path))
    monkeypatch.setenv("FLUX2_STEP_DUMP_CALLS", "1")
    ours = NeuronFlux2Transformer(tiny_flux2, dtype=torch.float32)
    ours.load_weights(tiny_flux2, "cpu")
    x, ctx, img_ids, txt_ids, t, gd = dit_inputs(ours.cfg)
    kw = dict(encoder_hidden_states=ctx, timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=gd)
    with torch.no_grad():
        ours(hidden_states=x, return_dict=False, **kw)
        got = ours(hidden_states=x * 0.5, return_dict=False, **kw)[0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_001.pt"]
    d = torch.load(tmp_path / "step_001.pt")
    assert d["call"] == 1
    assert torch.equal(d["hidden_states"], x * 0.5)
    assert torch.equal(d["encoder_hidden_states"], ctx) and torch.equal(d["guidance"], gd)
    assert torch.equal(d["output"], got.float())


def test_text_encoder_matches_transformers(tiny_flux2, vllm_tp1):
    from transformers import Mistral3ForConditionalGeneration

    from vllm_omni_neuron.diffusion.models.flux2.text_encoder import NeuronFlux2TextEncoder

    ref = Mistral3ForConditionalGeneration.from_pretrained(
        os.path.join(tiny_flux2, "text_encoder"), torch_dtype=torch.float32
    )
    ours = NeuronFlux2TextEncoder(tiny_flux2, dtype=torch.float32)
    ours.load_weights(tiny_flux2, "cpu")
    assert len(ours.layers) == 30
    g = torch.Generator().manual_seed(1)
    s, real = 48, 29
    ids = torch.randint(100, 120000, (1, s), generator=g)
    mask = torch.zeros(1, s, dtype=torch.long)
    mask[0, :real] = 1
    ids[0, real:] = 11  # <pad>
    with torch.no_grad():
        want = ref(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        ).hidden_states
        got = ours(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    for k in (10, 20, 30):
        assert rel(got[k], want[k]) < 1e-5, (k, rel(got[k], want[k]))
