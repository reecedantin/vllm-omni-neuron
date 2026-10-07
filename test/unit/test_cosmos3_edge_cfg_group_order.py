# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 CFG-parallel branch exchange over unsorted (Trn2 physical-mesh) CFG groups.

At TP8 x CP x CFG2 half the CFG groups are descending (e.g. ``[12, 8]``): CFG group rank 0 (the
positive branch) is the HIGHER global rank, while the c10d group built from the list is sorted. A raw
gather over the c10d group swaps positive and negative on those groups. Four gloo ranks, CFG groups
``[0, 2]`` (ascending) and ``[3, 1]`` (descending), both exchange paths: the transformer facade's
gather (``_host_out``, the device path's code on gloo) and the host fallback (``host_all_gather``).
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

_GROUPS = [[0, 2], [3, 1]]
_POS, _NEG, _SCALE = 3.0, 1.0, 2.0
_SHAPE = (1, 4, 2, 2, 2)


def _worker(rank, world, init_file, out_dir, path):
    import torch.distributed as dist
    import vllm_omni.diffusion.distributed.parallel_state as ops

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import (
        NeuronCosmos3EdgePipeline,
        NeuronCosmos3EdgeTransformer,
    )

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    pgs = [dist.new_group(g) for g in _GROUPS]
    mine = next(i for i, g in enumerate(_GROUPS) if rank in g)
    ranks = _GROUPS[mine]
    coord = SimpleNamespace(
        ranks=ranks,
        world_size=2,
        rank_in_group=ranks.index(rank),
        cpu_group=pgs[mine],
        device_group=pgs[mine],
    )
    ops.get_cfg_group = lambda: coord
    ops.get_classifier_free_guidance_rank = lambda: coord.rank_in_group

    tf = NeuronCosmos3EdgeTransformer.__new__(NeuronCosmos3EdgeTransformer)
    tf._init_cfg_gather()
    if path == "host":
        tf._cfg_gather_fn = None  # no device gather: the pipeline falls back to host_all_gather

    def predict_noise(**kw):
        video = torch.full(_SHAPE, kw["value"])
        action = torch.full((1, 3, 2), kw["value"] * 10)
        if path == "device":  # what the facade does with the device GEN outputs
            return tf._host_out(video), tf._host_out(action)
        return video, action

    stub = SimpleNamespace(
        transformer=tf,
        _cfg_parallel_active=lambda: True,
        predict_noise=predict_noise,
        combine_cfg_noise=lambda pos, neg, scale, norm, **kw: tuple(
            n + scale * (p - n) for p, n in zip(pos, neg)
        ),
    )
    video, action = NeuronCosmos3EdgePipeline.predict_noise_maybe_with_cfg(
        stub, True, _SCALE, {"value": _POS}, {"value": _NEG}, cfg_normalize=False
    )
    torch.save(
        {"video": video, "action": action, "armed_after": tf._cfg_parts},
        os.path.join(out_dir, f"{rank}.pt"),
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("path", ["device", "host"])
def test_cfg_parallel_descending_group(path):
    want = _NEG + _SCALE * (_POS - _NEG)  # 5.0; swapped branches would give -1.0
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        mp.spawn(_worker, args=(4, init, d, path), nprocs=4, join=True)
        for rank in range(4):
            r = torch.load(os.path.join(d, f"{rank}.pt"))
            assert torch.equal(r["video"], torch.full(_SHAPE, want)), (
                path,
                rank,
                r["video"].flatten()[0],
            )
            assert torch.equal(r["action"], torch.full((1, 3, 2), want * 10)), (path, rank)
            assert r["armed_after"] is None
