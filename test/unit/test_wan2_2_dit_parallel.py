# SPDX-License-Identifier: Apache-2.0
"""CPU, 2 gloo ranks: the Neuron Wan2.2 DiT under TP=2 (+sequence parallel) and CP=2 must
reproduce the single-rank Diffusers reference, including TI2V per-token timesteps (whose
modulation is sliced with the rank-local tokens)."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("PJRT_DEVICE", "CPU")

from test.unit.test_wan2_2_tiny import (  # noqa: E402
    TINY_TEXT,
    make_tiny_checkpoint,
    require_tokenizer,
)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(in_channels: int, per_token: bool, odd: bool = False):
    g = torch.Generator().manual_seed(5)
    # 3 * 4 * 6 = 72 tokens (divisible by 2); odd: 3 * 3 * 5 = 45 tokens (CP2 pads 1 token)
    frames, h, w = (3, 6, 10) if odd else (3, 8, 12)
    x = torch.randn(1, in_channels, frames, h, w, generator=g)
    ctx = torch.randn(1, 24, TINY_TEXT["d_model"], generator=g)
    if per_token:
        mask = torch.ones(frames, h // 2, w // 2)
        mask[0] = 0
        t = (mask * 702.0).flatten().unsqueeze(0)
    else:
        t = torch.tensor([702.0])
    return x, ctx, t


def _worker(rank, world, port, path, tp, cp, sp, per_token, out_path, odd=False):
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as vllm_ps
    from vllm_omni.diffusion.distributed import parallel_state as omni_ps
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config

    from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2 import (
        _create_transformer_from_config,
    )

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    vllm_ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    omni_ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    omni_ps.initialize_model_parallel(
        tensor_parallel_size=tp, ring_degree=cp, sequence_parallel_size=cp, backend="gloo"
    )
    cfg = load_transformer_config(path, "transformer", True)
    cfg["tp_sequence_parallel"] = sp
    model = _create_transformer_from_config(cfg)
    model.load_weights(os.path.join(path, "transformer"))
    model = model.float().eval()
    x, enc, t = _inputs(cfg["in_channels"], per_token, odd)
    with torch.no_grad():
        out = model(x, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]
    if rank == 0:
        torch.save(out, out_path)


@pytest.mark.parametrize(
    "tp,cp,sp",
    [(2, 1, False), (2, 1, True), (1, 2, False)],
    ids=["tp2", "tp2-sp", "cp2"],
)
@pytest.mark.parametrize("per_token", [False, True], ids=["scalar-t", "per-token-t"])
def test_parallel_dit_matches_reference(tmp_path, tp, cp, sp, per_token):
    from diffusers import WanTransformer3DModel

    require_tokenizer()
    path = make_tiny_checkpoint(str(tmp_path / "ti2v"), "ti2v-5b")
    ref = WanTransformer3DModel.from_pretrained(path, subfolder="transformer").float().eval()
    x, enc, t = _inputs(48, per_token)
    with torch.no_grad():
        want = ref(x, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]

    out_path = str(tmp_path / "out.pt")
    mp.spawn(
        _worker,
        args=(2, _port(), path, tp, cp, sp, per_token, out_path),
        nprocs=2,
        join=True,
    )
    got = torch.load(out_path)
    err = float((got - want).norm() / want.norm())
    assert err < 1e-4, f"tp={tp} cp={cp} sp={sp} per_token={per_token}: rel-L2 {err:.2e}"


@pytest.mark.parametrize("per_token", [False, True], ids=["scalar-t", "per-token-t"])
def test_cp_pads_non_divisible_sequence(tmp_path, per_token):
    """45 tokens over CP2: padded to 46, pad keys masked out, output exact vs single rank."""
    from diffusers import WanTransformer3DModel

    require_tokenizer()
    path = make_tiny_checkpoint(str(tmp_path / "ti2v"), "ti2v-5b")
    ref = WanTransformer3DModel.from_pretrained(path, subfolder="transformer").float().eval()
    x, enc, t = _inputs(48, per_token, odd=True)
    with torch.no_grad():
        want = ref(x, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]
    out_path = str(tmp_path / "out.pt")
    mp.spawn(
        _worker,
        args=(2, _port(), path, 1, 2, False, per_token, out_path, True),
        nprocs=2,
        join=True,
    )
    got = torch.load(out_path)
    err = float((got - want).norm() / want.norm())
    assert got.shape == want.shape
    assert err < 1e-4, f"CP2 padded (45 tokens) per_token={per_token}: rel-L2 {err:.2e}"
