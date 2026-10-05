# SPDX-License-Identifier: Apache-2.0
"""TP on CPU (gloo ranks): the Qwen-Image 2.1 text encoder and DiT sharded over 2, 4 and 8 ranks
must reproduce TP=1 (sharded loaders, GQA head grouping, KV-head replication once TP exceeds the
KV-head count -- the tiny encoder has 4 -- and row-parallel all-reduces), and the VAE's
tile-parallel decode (tiles dealt across the ranks, gathered to rank 0) must reproduce the
single-process decode. The device TP run then only adds the Neuron collective lowering."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_qwen_image_21_pipeline import REAL  # noqa: E402


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(rank, world, port, tiny, out_path):
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
    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    os.environ["QWEN_IMAGE_VAE_TILE"] = "4,1"  # 8x8 latents -> 3x3 tiles dealt across the ranks
    p = NeuronQwenImage21Pipeline(model_path=tiny, dtype=torch.float32)
    assert p.transformer.tp == world and p.text_encoder.tp == world
    p.load_weights()
    emb, _ = p.encode_prompt("a quiet harbour at night")
    lat = p.generate(
        "a quiet harbour at night",
        height=128,
        width=128,
        num_inference_steps=3,
        seed=1,
        output_type="latent",
    )
    img = p.generate(
        "a quiet harbour at night", height=128, width=128, num_inference_steps=3, seed=1
    )
    if rank == 0:
        torch.save({"emb": emb, "lat": lat, "img": img}, out_path)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    if not os.path.isdir(os.path.join(REAL, "processor")):
        pytest.skip(
            "set QWEN_IMAGE21_WEIGHTS to a Qwen-Image-2.1 checkout (for its processor files)"
        )
    from .test_qwen_image_tiny import build

    return build(str(tmp_path_factory.mktemp("qwen21tp")), processor_from=REAL)


def test_tp_matches(tiny_dir, tmp_path):
    outs = {}
    for world in (1, 2, 4, 8):
        path = tmp_path / f"tp{world}.pt"
        mp.spawn(_run, args=(world, _port(), tiny_dir, str(path)), nprocs=world, join=True)
        outs[world] = torch.load(path)
    for world in (2, 4, 8):
        for key in ("emb", "lat", "img"):
            a, b = outs[world][key].float(), outs[1][key].float()
            rel = ((a - b).norm() / b.norm()).item()
            assert rel < 1e-5, (world, key, rel)
