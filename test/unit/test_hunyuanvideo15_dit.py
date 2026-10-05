# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Neuron HunyuanVideo-1.5 DiT vs diffusers' ``HunyuanVideo15Transformer3DModel``
on the tiny random-weight checkpoint (fp32, TP=1, and TP=2/4 over gloo ranks).

Covers the weight-name mapping (strict load, no remapping), the sharded loaders, the fixed-shape
encoder assembly (gather index + key bias instead of upstream's boolean compaction), the token
refiner, RoPE and the split ``prologue -> run_blocks -> epilogue`` path.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_hunyuanvideo15_tiny_ckpt import tiny_ckpt  # noqa: E402,F401  (session fixture)


def spawn_with_free_port(fn, make_args, nprocs, attempts=4):
    """``mp.spawn`` with a freshly picked rendezvous port, retried when another process grabbed the
    port between picking and binding it (EADDRINUSE on a busy shared host)."""
    import torch.multiprocessing as mp

    for i in range(attempts):
        try:
            return mp.spawn(fn, args=make_args(_port()), nprocs=nprocs, join=True)
        except mp.ProcessRaisedException as e:
            if "EADDRINUSE" not in str(e) or i == attempts - 1:
                raise


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(cfg, n_text=23, n_byt5=5, glyph=True, seed=3):
    g = torch.Generator().manual_seed(seed)
    lt, l2 = 40, 16
    x = torch.randn(1, cfg.in_channels, 3, 4, 6, generator=g)
    text = torch.randn(1, lt, cfg.text_embed_dim, generator=g)
    tmask = torch.zeros(1, lt)
    tmask[0, :n_text] = 1
    text2 = torch.randn(1, l2, cfg.text_embed_2_dim, generator=g)
    t2mask = torch.zeros(1, l2)
    if glyph:
        t2mask[0, :n_byt5] = 1
    else:
        text2.zero_()
    return x, torch.tensor([637.0]), text, tmask, text2, t2mask


def _reference(ckpt, inputs):
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = HunyuanVideo15Transformer3DModel.from_pretrained(
        os.path.join(ckpt, "transformer"), torch_dtype=torch.float32
    )
    x, ts, text, tmask, text2, t2mask = inputs
    cfg = ref.config
    image = torch.zeros(1, 729, cfg.image_embed_dim)
    with torch.no_grad():
        return ref(
            hidden_states=x,
            timestep=ts,
            encoder_hidden_states=text,
            encoder_attention_mask=tmask,
            encoder_hidden_states_2=text2,
            encoder_attention_mask_2=t2mask,
            image_embeds=image,
            return_dict=False,
        )[0]


def _ours(ckpt, inputs, split: bool = False):
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import (
        HV15Config,
        NeuronHunyuanVideo15DiT,
        prepare_encoder_inputs,
        rope_tables,
    )

    cfg = HV15Config.from_model_dir(ckpt)
    dit = NeuronHunyuanVideo15DiT(cfg, dtype=torch.float32)
    dit.load_weights(ckpt, "cpu")
    x, ts, text, tmask, text2, t2mask = inputs
    t, h, w = x.shape[2:]
    cos, sin = rope_tables(cfg, t, h, w)
    enc_index, text_bias, key_bias = prepare_encoder_inputs(tmask, t2mask, None, t * h * w)
    with torch.no_grad():
        if not split:
            return dit(
                x, ts, text, tmask, text2, None, enc_index, cos, sin, text_bias, key_bias
            ), dit
        hid, enc, temb = dit.prologue(x, ts, text, tmask, text2, None, enc_index, text_bias)
        for s in range(cfg.num_layers):  # one block per call
            hid, enc = dit.run_blocks(hid, enc, temb, cos, sin, key_bias, s, s + 1)
        return dit.epilogue(hid, temb, (t, h, w)), dit


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(scope="module")
def single_rank():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_port()}",
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    yield
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("glyph", [True, False])
def test_dit_matches_diffusers(tiny_ckpt, single_rank, glyph):  # noqa: F811
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import HV15Config

    inputs = _inputs(HV15Config.from_model_dir(tiny_ckpt), glyph=glyph)
    ref = _reference(tiny_ckpt, inputs)
    out, _ = _ours(tiny_ckpt, inputs)
    out_split, _ = _ours(tiny_ckpt, inputs, split=True)
    assert out.shape == ref.shape
    assert torch.isfinite(ref).all()
    assert _rel(out, ref) < 1e-5, _rel(out, ref)
    assert torch.equal(out, out_split)


def _tp_worker(rank, world, port, ckpt, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(world, 1)
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import HV15Config

    out, dit = _ours(ckpt, _inputs(HV15Config.from_model_dir(ckpt)))
    assert dit.tp_size == world
    if rank == 0:
        torch.save(out, out_path)


@pytest.mark.parametrize("world", [2, 4])
def test_dit_tp_matches_tp1(tiny_ckpt, single_rank, world, tmp_path):  # noqa: F811
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import HV15Config

    ref = _reference(tiny_ckpt, _inputs(HV15Config.from_model_dir(tiny_ckpt)))
    path = tmp_path / f"tp{world}.pt"
    spawn_with_free_port(_tp_worker, lambda port: (world, port, tiny_ckpt, str(path)), world)
    assert _rel(torch.load(path), ref) < 1e-5
