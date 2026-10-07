# SPDX-License-Identifier: Apache-2.0
"""All-rank agreement helper (vllm_omni_neuron.testing.rank_agreement) on CPU."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_neuron.testing import (
    check_rank_agreement,
    compare_digests,
    compare_rank_digest_files,
    outputs_digest,
    tensor_digest,
    write_rank_digest,
)


def _latents(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 16, 5, 8, 8, generator=g).to(torch.bfloat16)


def test_digest_is_layout_independent_and_dtype_aware():
    x = torch.randn(4, 6)
    assert tensor_digest(x)["sha256"] == tensor_digest(x.t().contiguous().t())["sha256"]
    assert tensor_digest(x)["sha256"] != tensor_digest(x.to(torch.bfloat16))["sha256"]
    d = tensor_digest(torch.tensor([1.0, float("nan"), float("inf")]))
    assert d["nonfinite"] == 2 and d["sum"] == 1.0
    assert all(v == v for v in d["sample"])  # no NaN in the JSON sample
    assert tensor_digest(torch.zeros(0))["sample"] == []
    assert tensor_digest(torch.arange(10), sample=3)["sample"] == [0.0, 4.0, 9.0]


def test_compare_exact_close_and_mismatch():
    x = _latents()
    same = [outputs_digest({"latents": x}) for _ in range(4)]
    rep = compare_digests(same)
    assert rep.ok and rep.max_rel == {"latents": 0.0}
    assert "bit-exact" in rep.summary()

    y = x.clone()
    y.view(-1)[7] += 0.25  # one element off on rank 2
    digs = [outputs_digest({"latents": t}) for t in (x, x, y, x)]
    rep = compare_digests(digs)
    assert not rep.ok and rep.disagreeing_ranks == [2]
    assert "rank 2" in rep.summary() and "latents" in rep.disagreeing[2]
    with pytest.raises(AssertionError, match="rank 2"):
        rep.raise_if_failed()
    # a tolerance accepts a tiny deviation, rejects a large one
    z = (x.float() * (1 + 1e-4)).to(torch.float32)
    xf = x.float()
    assert compare_digests([outputs_digest(xf), outputs_digest(z)], rtol=1e-3).ok
    assert not compare_digests([outputs_digest(xf), outputs_digest(xf * 1.1)], rtol=1e-3).ok


def test_compare_shape_names_and_rank0_nonfinite():
    a, b = torch.zeros(2, 3), torch.zeros(3, 2)
    rep = compare_digests([outputs_digest(a), outputs_digest(b)], ranks=[8, 12])
    assert rep.disagreeing_ranks == [12] and "shape" in rep.disagreeing[12]["output"]
    rep = compare_digests([outputs_digest({"v": a}), outputs_digest({"w": a})])
    assert rep.disagreeing_ranks == [1] and "outputs" in rep.disagreeing[1]
    nan = torch.tensor([float("nan")])
    rep = compare_digests([outputs_digest(nan), outputs_digest(nan)])  # all agree, but on NaN
    assert not rep.ok and not rep.disagreeing and "non-finite" in rep.problems[0]


def test_single_process_without_distributed_is_trivially_ok():
    rep = check_rank_agreement(torch.ones(3))
    assert rep.ok and rep.world_size == 1


def test_digest_files_roundtrip_and_missing_rank():
    x = _latents()
    with tempfile.TemporaryDirectory() as d:
        for r in range(4):
            write_rank_digest(
                d, r, {"latents": x if r != 3 else x * 2, "step": torch.tensor(r * 0)}
            )
        rep = compare_rank_digest_files(d, world_size=4)
        assert rep.disagreeing_ranks == [3] and not rep.problems
        rep = compare_rank_digest_files(d, world_size=6)
        assert "missing digests for ranks [4, 5]" in rep.problems
        assert rep.to_json()["disagreeing_ranks"] == [3]


_GROUPS = [[0, 2], [3, 1]]  # one ascending, one descending group (mesh CP shape)


def _worker(rank, world, init_file, out_dir):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    coord = None
    for ranks in _GROUPS:
        pg = dist.new_group(ranks)
        if rank in ranks:
            coord = SimpleNamespace(
                ranks=ranks, world_size=len(ranks), rank_in_group=ranks.index(rank), cpu_group=pg
            )
    x = _latents()
    if rank == 1:  # rank 1 drifts (a swapped slice in the real bug)
        x = x.flip(-1)
    world_rep = check_rank_agreement({"latents": x})
    coord_rep = check_rank_agreement({"latents": x}, coord=coord)
    torch.save(
        {
            "world": world_rep.to_json(),
            "coord": coord_rep.to_json(),
            "coord_ranks": coord_rep.ranks,
        },
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_collective_reports_the_disagreeing_rank_on_every_rank():
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
        res = {r: torch.load(os.path.join(d, f"r{r}.pt")) for r in range(4)}
    for r, out in res.items():
        assert out["world"]["disagreeing_ranks"] == [1], (r, out)
        if r in (0, 2):
            assert out["coord"]["ok"], (r, out)
        else:
            # group [3, 1]: reference is its FIRST member (3), so the drifting rank 1 is reported
            assert out["coord_ranks"] == [3, 1]
            assert out["coord"]["disagreeing_ranks"] == [1], (r, out)
