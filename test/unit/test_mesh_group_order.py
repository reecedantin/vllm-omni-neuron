# SPDX-License-Identifier: Apache-2.0
"""Host collectives over unsorted (physical-mesh) CP / CFG groups return parts in group-rank order."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_neuron.diffusion.distributed.parallel_state import (
    _c10d_positions,
    _tp_contiguous_mesh_groups,
)


def _descending(groups):
    return [g for g in groups if list(g) != sorted(g)]


@pytest.mark.parametrize(
    "tp,cp,cfg,cp_desc,cfg_desc",
    [
        (8, 2, 1, 4, 0),  # 16 cores: CP [12, 8] ...
        (8, 4, 1, 4, 0),  # 32 cores
        (8, 8, 1, 4, 0),  # 64 cores, Layout A
        (4, 8, 2, 4, 0),  # 64 cores, Layout A with CFG
        (8, 4, 2, 0, 16),  # 64 cores, Layout B with CFG: CFG [12, 8] ...
    ],
)
def test_mesh_layouts_have_descending_cp_or_cfg_groups(tp, cp, cfg, cp_desc, cfg_desc):
    """Documents the hazard: the validated mesh mapping emits descending CP / CFG groups (TP never)."""
    tp_g, cp_g, cfg_g = _tp_contiguous_mesh_groups(tp, cp, cfg)
    assert not _descending(tp_g)
    assert len(_descending(cp_g)) == cp_desc
    assert len(_descending(cfg_g)) == cfg_desc


def test_c10d_positions():
    assert _c10d_positions(SimpleNamespace(ranks=[0, 4])) == [0, 1]
    assert _c10d_positions(SimpleNamespace(ranks=[12, 8])) == [1, 0]
    assert _c10d_positions(SimpleNamespace(ranks=[12, 8, 28, 24])) == [1, 0, 3, 2]


# Two CP groups over 4 gloo ranks, one ascending and one descending (the mesh shape of TP x CP2).
_GROUPS = [[0, 2], [3, 1]]


def _worker(rank, world, init_file, out_dir):
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.distributed.parallel_state import (
        host_all_gather,
        host_all_gather_object,
    )

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    coord = None
    for ranks in _GROUPS:
        pg = dist.new_group(ranks)  # every rank creates every group, as GroupCoordinator does
        if rank in ranks:
            coord = SimpleNamespace(
                ranks=ranks, world_size=len(ranks), rank_in_group=ranks.index(rank), cpu_group=pg
            )
    # each member contributes its CP slice index: the gather must come back as [slice 0, slice 1]
    x = torch.full((3,), float(coord.rank_in_group))
    got = [int(p[0]) for p in host_all_gather(coord, x)]
    raw = [torch.empty_like(x) for _ in range(coord.world_size)]
    dist.all_gather(raw, x, group=coord.cpu_group)
    objs = host_all_gather_object(coord, ("slice", coord.rank_in_group))
    torch.save(
        {"ranks": coord.ranks, "got": got, "raw": [int(p[0]) for p in raw], "obj": objs},
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_host_all_gather_follows_group_rank_order_on_descending_groups():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(r, 4, os.path.join(d, "init"), d)) for r in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(100)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        res = [torch.load(os.path.join(d, f"r{r}.pt")) for r in range(4)]
    for r in res:
        assert r["got"] == [0, 1], r  # helper: group-rank order on both groups
        assert r["obj"] == [("slice", 0), ("slice", 1)], r
        # the raw c10d gather is sorted order: right on [0, 2], swapped on [3, 1] (the bug)
        assert r["raw"] == ([0, 1] if r["ranks"] == [0, 2] else [1, 0]), r
