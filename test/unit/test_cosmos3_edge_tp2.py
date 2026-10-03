# SPDX-License-Identifier: Apache-2.0
"""TP=2 on CPU (2 gloo ranks): UND + GEN with heads sharded across ranks must reproduce TP=1.

Validates the sharded weight loaders, the GQA head grouping per shard and the row-parallel
all-reduces; the device TP=2 run then only adds the Neuron collective lowering.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs():
    torch.manual_seed(3)
    real, bucket = 19, 32
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    return ids, mask, torch.randn(1, 48, 3, 16, 16), torch.tensor([421.0])


def _run(rank, world, port, weights, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    # vLLM's same-node probe does a device barrier; the CPU-only test has no accelerator hooks.
    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                 distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
    initialize_model_parallel(world, 1)
    cfg = EdgeGenConfig.from_model_dir(weights)
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.float32)
    gen.load_weights(weights, "cpu")
    assert und.tp_size == world and gen.tp_size == world
    ids, mask, lat, ts = _inputs()
    t, h, w = lat.shape[2:]
    with torch.no_grad():
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu, su)
        cg, sg = gen.rope_tables(mask, t, h, w, 24.0)
        s_gen = t * (h // 2) * (w // 2)
        out = gen(lat, ts, cg, sg, gen.key_bias(mask, s_gen), torch.ones(1, s_gen, 1), *kv)
    if rank == 0:
        torch.save(out, out_path)


@pytest.mark.parametrize("world", [1, 2])
def test_tp_matches(world, edge_weights, tmp_path):
    out = tmp_path / f"tp{world}.pt"
    mp.spawn(_run, args=(world, _port(), edge_weights, str(out)), nprocs=world, join=True)
    ref_path = tmp_path.parent / "tp_ref.pt"
    if world == 1:
        torch.save(torch.load(out), ref_path)
        return
    if not ref_path.exists():
        pytest.skip("run with the world=1 case first")
    ref, got = torch.load(ref_path), torch.load(out)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-5, rel
