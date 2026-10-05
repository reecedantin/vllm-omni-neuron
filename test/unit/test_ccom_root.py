# SPDX-License-Identifier: Apache-2.0
"""Deterministic CCOM root endpoint for the native Lite diffusion worker (worker/ccom_root.py).

A 32-rank job on cores 32-63 died at its first collective graph with
``Failed to bind(127.0.0.1<59573>) Address already in use`` -> ``ncclInitGlobalComm failed`` ->
``Failed to schedule neff execution`` on rank 0, and the other 31 ranks retried the bootstrap for
three hours: the worker popped ``NEURON_RT_ROOT_COMM_ID`` and the runtime probed a random port that
another engine took first. These tests pin the replacement rule and its fail-fast.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from vllm_omni_neuron.diffusion.worker import ccom_root
from vllm_omni_neuron.diffusion.worker.ccom_root import (
    CCOM_BASE_PORT,
    ccom_root_comm_id,
    ccom_root_port,
    check_ccom_root_port_free,
    probe_port_free,
    set_ccom_root_comm_id,
)

# Engine shapes on a 64-core trn2.48xlarge: the whole box, two 32-core halves, 16-core chip rows,
# 8-core pairs, 4-core single chips, and TP=1 single-core stages.
_ENGINES = {
    "lease64": list(range(64)),
    "half-lo": list(range(32)),
    "half-hi": list(range(32, 64)),
    "row1": list(range(16, 32)),
    "row3": list(range(48, 64)),
    "pair-20": list(range(20, 28)),
    "chip1": [4, 5, 6, 7],
    "chip15": [60, 61, 62, 63],
    "core9": [9],
}


def test_port_is_base_plus_lowest_core():
    assert ccom_root_port([0]) == CCOM_BASE_PORT
    assert ccom_root_port(range(64)) == CCOM_BASE_PORT
    assert ccom_root_port([4, 5, 6, 7]) == CCOM_BASE_PORT + 4
    assert (
        ccom_root_port([7, 6, 5, 4]) == CCOM_BASE_PORT + 4
    )  # order of the device list is irrelevant
    assert ccom_root_port(range(32, 64)) == CCOM_BASE_PORT + 32
    assert CCOM_BASE_PORT == 61234  # vllm-neuron's _CCOM_BASE_PORT: one rule for text + diffusion


def test_engines_on_disjoint_cores_never_share_a_port():
    names = list(_ENGINES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = set(_ENGINES[a]), set(_ENGINES[b])
            if ca.isdisjoint(cb):
                assert ccom_root_port(ca) != ccom_root_port(cb), (a, b)


def test_sixty_four_core_lease_and_small_jobs_differ_unless_they_overlap():
    """A 64-core lease vs a 4-core job: same port ONLY when they share cores (which the scheduler
    forbids while both are alive); every disjoint small job gets its own port."""
    lease = ccom_root_port(_ENGINES["lease64"])
    for start in range(4, 64, 4):
        assert ccom_root_port(range(start, start + 4)) != lease
    assert ccom_root_port(range(0, 4)) == lease  # overlapping cores: same engine slot


def test_all_single_chip_jobs_are_pairwise_distinct_and_in_range():
    ports = {ccom_root_port(range(c, c + 4)) for c in range(0, 64, 4)}
    assert len(ports) == 16
    assert min(ports) == CCOM_BASE_PORT and max(ports) == CCOM_BASE_PORT + 60 <= 65535


def test_port_rejects_empty_or_negative_cores():
    with pytest.raises(ValueError):
        ccom_root_port([])
    with pytest.raises(ValueError):
        ccom_root_port([-1, 0])


def test_comm_id_uses_master_addr_then_loopback(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    assert ccom_root_comm_id([32, 33]) == f"127.0.0.1:{CCOM_BASE_PORT + 32}"
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.7")
    assert ccom_root_comm_id([32, 33]) == f"10.0.0.7:{CCOM_BASE_PORT + 32}"
    assert ccom_root_comm_id([32, 33], host="localhost") == f"localhost:{CCOM_BASE_PORT + 32}"


def test_set_comm_id_pins_core_derived_value_over_anything_inherited(monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.setenv("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:61600")  # an explicit environment pin
    assert set_ccom_root_comm_id([48, 49]) == f"127.0.0.1:{CCOM_BASE_PORT + 48}"
    assert os.environ["NEURON_RT_ROOT_COMM_ID"] == f"127.0.0.1:{CCOM_BASE_PORT + 48}"


def test_set_comm_id_without_cores_keeps_an_explicit_pin_but_drops_lites_preset(monkeypatch):
    monkeypatch.setenv("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:61600")
    assert set_ccom_root_comm_id([]) == "127.0.0.1:61600"
    assert os.environ["NEURON_RT_ROOT_COMM_ID"] == "127.0.0.1:61600"
    monkeypatch.setenv("NEURON_RT_ROOT_COMM_ID", "localhost:62182")  # Lite's import-time preset
    assert set_ccom_root_comm_id(None) is None
    assert "NEURON_RT_ROOT_COMM_ID" not in os.environ
    assert set_ccom_root_comm_id([]) is None  # nothing pinned: left to the runtime


def _held_port() -> tuple[socket.socket, int]:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    return holder, holder.getsockname()[1]


def test_probe_reports_a_listening_holder_and_frees_the_port_after_release():
    holder, port = _held_port()
    try:
        reason = probe_port_free("127.0.0.1", port)
        assert reason is not None and "in use" in reason.lower()
    finally:
        holder.close()
    assert probe_port_free("127.0.0.1", port) is None
    # The probe never keeps the port: a second bind right after it succeeds.
    with socket.socket() as again:
        again.bind(("127.0.0.1", port))


def test_check_raises_with_the_reason_when_rank0_finds_the_port_taken_single_process():
    holder, port = _held_port()
    try:
        with pytest.raises(RuntimeError, match=f"127.0.0.1:{port}.*cores 8-11.*in use.*61234"):
            check_ccom_root_port_free(f"127.0.0.1:{port}", rank=0, cores=[8, 9, 10, 11])
    finally:
        holder.close()
    check_ccom_root_port_free(f"127.0.0.1:{port}", rank=0, cores=[8, 9, 10, 11])  # free now


def test_check_rejects_malformed_comm_id():
    with pytest.raises(ValueError):
        check_ccom_root_port_free("nonsense", rank=0, cores=[0])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gloo_barrier():
    """dist.barrier() asks the accelerator hooks (unregistered 'neuron' here); all_reduce is gloo-only."""
    import torch.distributed as dist

    dist.all_reduce(torch.zeros(1, dtype=torch.int32))


def _collective_check(rank, world, init_port, ccom_port, hold_on_rank, out_dir):
    """Every rank runs the check; rank ``hold_on_rank`` (!= 0 so rank 0 is NOT the holder of the
    port it probes) holds the CCOM port open. Writes 'raised'/'ok' per rank."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{init_port}", world_size=world, rank=rank
    )
    holder = None
    if rank == hold_on_rank:
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", ccom_port))
        holder.listen(1)
    _gloo_barrier()  # the holder is listening before rank 0 probes
    try:
        check_ccom_root_port_free(f"127.0.0.1:{ccom_port}", rank=rank, cores=[32, 33, 34])
        verdict = "ok"
    except RuntimeError as error:
        verdict = "raised" if "already in use" in str(error) else f"wrong: {error}"
    finally:
        if holder is not None:
            holder.close()
    _gloo_barrier()
    with open(os.path.join(out_dir, f"rank{rank}"), "w") as f:
        f.write(verdict)
    dist.destroy_process_group()


@pytest.mark.parametrize("hold", [True, False])
def test_every_rank_fails_together_when_rank0_sees_the_port_taken(tmp_path, hold):
    """The fail-fast is collective: ranks 1..N would otherwise sit in the runtime's bootstrap
    retry loop (the 3-hour hang) after rank 0 died at its first collective graph."""
    world = 3
    ccom_port = _free_port()
    mp.spawn(
        _collective_check,
        args=(world, _free_port(), ccom_port, 1 if hold else -1, str(tmp_path)),
        nprocs=world,
        join=True,
    )
    verdicts = {r: (tmp_path / f"rank{r}").read_text() for r in range(world)}
    assert verdicts == {r: ("raised" if hold else "ok") for r in range(world)}


def test_worker_wiring_pins_before_lite_and_checks_after_gloo():
    """Structural guard on diffusion_worker.py: the pop that caused the race is gone, the pin
    happens before initialize_lite(), and the collective check after init_distributed_environment."""
    import inspect

    from vllm_omni_neuron.diffusion.worker import diffusion_worker

    src = inspect.getsource(diffusion_worker.NeuronDiffusionWorker.init_device)
    assert 'os.environ.pop("NEURON_RT_ROOT_COMM_ID"' not in src
    pin, lite = src.index("set_ccom_root_comm_id("), src.index("initialize_lite()")
    gloo, check = (
        src.index("init_distributed_environment("),
        src.index("check_ccom_root_port_free("),
    )
    assert pin < lite < gloo < check
    assert ccom_root.LITE_IMPORT_PRESETS == frozenset({"localhost:62182"})
