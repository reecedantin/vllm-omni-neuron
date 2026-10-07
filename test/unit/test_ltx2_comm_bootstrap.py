# SPDX-License-Identifier: Apache-2.0
"""CPU check of the LTX-2.5 device-collective bootstrap (pipeline_ltx25.bootstrap_collective) over a
4-rank gloo world split into two TP groups of 2, the shape of a TP x CP stage: every rank runs it,
the host barriers do not touch an accelerator device, and the all-reduce result is checked."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, tp, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    def no_barrier(*a, **k):  # under Neuron Lite dist.barrier raises (PrivateUse1 accelerator)
        raise NotImplementedError("dist.barrier must not be used")

    dist.barrier = no_barrier
    from vllm_omni_neuron.diffusion.models.ltx2.pipeline_ltx25 import bootstrap_collective

    groups = [dist.new_group(list(range(i, i + tp)), backend="gloo") for i in range(0, world, tp)]
    stage = dist.new_group(list(range(world)), backend="gloo")
    try:
        bootstrap_collective(
            stage, groups[rank // tp], tp, torch.device("cpu"), torch.float32, lambda f: f
        )
        results[rank] = "ok"
    except Exception as exc:  # noqa: BLE001
        results[rank] = repr(exc)
    finally:
        dist.destroy_process_group()


def test_bootstrap_collective_all_ranks():
    world, tp = 4, 2
    with mp.Manager() as m:
        results = m.dict()
        mp.spawn(_worker, args=(world, _free_port(), tp, results), nprocs=world, join=True)
        assert dict(results) == {r: "ok" for r in range(world)}


def test_bootstrap_collective_checks_the_sum(monkeypatch):
    from vllm_omni_neuron.diffusion.models.ltx2 import pipeline_ltx25 as p

    monkeypatch.setattr(p, "host_sync", lambda g: None)
    monkeypatch.setattr(dist, "all_reduce", lambda x, group=None: x)  # a no-op "reduce"
    with pytest.raises(RuntimeError, match="bootstrap"):
        p.bootstrap_collective(None, None, 2, torch.device("cpu"), torch.float32, lambda f: f)
