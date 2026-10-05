# SPDX-License-Identifier: Apache-2.0
"""CPU check of the tile-parallel conditioning-frame encode (``device_tiled_encode(group=...)``) and
the TP-group broadcast of its result (``NeuronEdgeVae._rank0_broadcast(all_ranks=True)``).

Four gloo ranks split into two groups of two, as under TP=2 x CFG=2: each group deals the tiles of
the same frame across its ranks, gathers them to the group's own first rank (global rank 0 or 2),
merges, and broadcasts. Every rank must end with exactly the single-process result. A stand-in VAE
(16x average pool per tile, plus a per-tile offset that makes a wrong merge visible) keeps it fast.
"""

from __future__ import annotations

import socket

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F


class _FakeVae:
    spatial_compression_ratio = 16
    dtype = torch.float32

    def _tile_encode_one(self, tile: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = tile.shape
        z = F.avg_pool2d(tile[:, :, 0], 16).unsqueeze(2)
        return torch.cat([z, z * 2 + tile.mean()], dim=1)  # [B, 2C, 1, h/16, w/16] "moments"


def _frame(px: int = 320) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, 1, px, px, generator=g)


def _worker(rank: int, world: int, port: int, out) -> None:
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.cosmos3_edge import pipeline_cosmos3_edge as pc

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    pc._tp_cpu_group = lambda: (2, group)  # what vLLM's TP group would hand back under TP=2 x CFG=2

    edge = pc.NeuronEdgeVae.__new__(pc.NeuronEdgeVae)
    torch.nn.Module.__init__(edge)
    edge.vae = _FakeVae()
    x = _frame()
    h = edge._rank0_broadcast(
        lambda v, g: pc.device_tiled_encode(edge.vae, v, 128, 64, group=g), x, all_ranks=True
    )
    out[rank] = h.clone()
    dist.destroy_process_group()


def test_tile_parallel_encode_matches_single_process():
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import (
        device_tiled_encode,
    )

    ref = device_tiled_encode(_FakeVae(), _frame(), 128, 64)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(4, port, out), nprocs=4, join=True)
    assert sorted(out.keys()) == [0, 1, 2, 3]
    for r in range(4):
        assert torch.equal(out[r], ref.to(out[r].dtype)), (
            f"rank {r} differs from the single-process encode"
        )
