# SPDX-License-Identifier: Apache-2.0
"""The TP text encoder (the device path's module, run on CPU) vs transformers' Qwen3-VL on the structure checkpoint.

* TP=1 fp32 reproduces ``hidden_states[text_encoder_layer]`` of the reference encode, padded to a length bucket;
* TP=2 (gloo ranks, query heads split, one KV head per rank) reproduces TP=1.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

PROMPT = "A golden retriever runs through the surf at sunset."


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _device_encode(weights, tp=1, rank=0, group=None):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import text_encoder_layer
    from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder_device import DeviceTextEncoder

    enc = DeviceTextEncoder(weights, text_encoder_layer(weights), tp, rank, group, torch.float32)
    return enc.encode(PROMPT).float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_bucket():
    from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder_device import bucket

    assert (bucket(1), bucket(64), bucket(65), bucket(512), bucket(600)) == (64, 64, 128, 512, 600)


def test_device_encoder_matches_reference(h3_tiny):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import text_encoder_layer
    from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder import H3TextEncoder

    ref = H3TextEncoder(h3_tiny, text_encoder_layer(h3_tiny), torch.float32).encode(PROMPT).float()
    out = _device_encode(h3_tiny)
    assert out.shape == ref.shape
    assert _rel(out, ref) < 1e-5, _rel(out, ref)


def _tp_worker(rank, world, port, weights, out_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
    )
    out = _device_encode(weights, world, rank, dist.group.WORLD)
    if rank == 0:
        torch.save(out, out_path)


def test_tp2_matches_tp1(h3_tiny, tmp_path):
    path = tmp_path / "tp2.pt"
    mp.spawn(_tp_worker, args=(2, _port(), h3_tiny, str(path)), nprocs=2, join=True)
    assert _rel(torch.load(path), _device_encode(h3_tiny)) < 1e-5
