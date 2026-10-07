# SPDX-License-Identifier: Apache-2.0
"""Deterministic CCOM bootstrap root endpoint for the native Lite diffusion worker.

The Neuron runtime bootstraps every collective communicator through one TCP root socket,
``NEURON_RT_ROOT_COMM_ID`` (``host:port``), that rank 0 binds at the FIRST collective graph
launch -- long after process start. When the variable is unset, libtorch-neuronx-lite's process
group setup has rank 0 pick an ephemeral port by probing a free one and releasing it; on a host
where many engines start at once another process can take that port in between, and the
bootstrap then dies with ``Failed to bind(127.0.0.1<port>) Address already in use`` ->
``ncclInitGlobalComm failed`` -> ``Failed to schedule neff execution`` on rank 0, while every
other rank retries ``bootstrap: rank N sends its info to root`` forever (one 32-rank job hung
for three hours).

The endpoint is derived here from the engine's visible cores instead, the same rule
vllm-neuron's text worker uses (``_CCOM_BASE_PORT + first visible core``): two engines alive at
the same time on one host hold disjoint cores (a NeuronCore is opened exclusively), so their
first cores -- and therefore their ports -- differ, with no inter-process coordination. The port
is also probed up front so a stale holder fails the engine with the reason instead of a hang.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterable

# Same base as vllm-neuron's worker and test harness, so a text engine and a diffusion engine
# on the same host follow one rule (and never pick the same port for different cores).
CCOM_BASE_PORT = 61234
_DEFAULT_HOST = "127.0.0.1"


def ccom_root_port(visible_devices: Iterable[int], base_port: int = CCOM_BASE_PORT) -> int:
    """``base_port + min(visible_devices)``: distinct for engines on disjoint core sets.

    Keyed on the SMALLEST visible core, not the first listed, so a permuted device list
    (``"4,5,6,7"`` vs ``"7,6,5,4"``) names the same engine slot.
    """
    cores = sorted({int(c) for c in visible_devices})
    if not cores:
        raise ValueError("ccom_root_port needs at least one visible core")
    if cores[0] < 0:
        raise ValueError(f"negative core index in visible devices: {cores}")
    port = base_port + cores[0]
    if not 1024 <= port <= 65535:
        raise ValueError(f"CCOM root port {port} out of range (base {base_port}, core {cores[0]})")
    return port


def ccom_root_comm_id(visible_devices: Iterable[int], host: str | None = None) -> str:
    """The ``host:port`` value for ``NEURON_RT_ROOT_COMM_ID`` on this engine."""
    host = host or os.environ.get("MASTER_ADDR") or _DEFAULT_HOST
    return f"{host}:{ccom_root_port(visible_devices)}"


def probe_port_free(host: str, port: int) -> str | None:
    """Try to bind ``host:port`` once (then release it). ``None`` if free, else the OS reason.

    A listening CCOM root from another engine (or a stale one from a run that did not exit) shows
    up as ``EADDRINUSE`` here. The bind does not hold the port: the runtime binds it itself at
    the first collective graph, and holding it open until then would block the runtime instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            return f"{error.strerror or error}" + (f" (errno {error.errno})" if error.errno else "")
    return None


def split_comm_id(addr: str) -> tuple[str, int]:
    host, sep, port = addr.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"NEURON_RT_ROOT_COMM_ID {addr!r} is not host:port")
    return host, int(port)


def set_ccom_root_comm_id(visible_devices: Iterable[int], host: str | None = None) -> str | None:
    """Pin ``NEURON_RT_ROOT_COMM_ID`` for this engine BEFORE libtorch-neuronx-lite initializes.

    Precedence mirrors vllm-neuron's ``rendezvous_ccom_bootstrap``:
      1. a core-derived endpoint whenever this engine knows its cores (the normal case);
      2. otherwise an explicitly pinned ``NEURON_RT_ROOT_COMM_ID`` from the environment is kept
         -- EXCEPT Lite's own import-time preset (``localhost:62182``), which is identical in every
         co-located engine and therefore the collision it is meant to avoid; that one is dropped
         so Lite falls back to its ephemeral port, the pre-existing behaviour.
    Returns the value now in the environment (``None`` when left to the runtime).
    """
    cores = list(visible_devices or ())
    if cores:
        addr = ccom_root_comm_id(cores, host)
        os.environ["NEURON_RT_ROOT_COMM_ID"] = addr
        return addr
    pinned = os.environ.get("NEURON_RT_ROOT_COMM_ID")
    if pinned and pinned not in LITE_IMPORT_PRESETS:
        return pinned
    os.environ.pop("NEURON_RT_ROOT_COMM_ID", None)
    return None


# libtorch_neuronx_lite/__init__.py sets this when the variable is unset at import time.
LITE_IMPORT_PRESETS = frozenset({"localhost:62182"})


def check_ccom_root_port_free(addr: str, *, rank: int, cores: Iterable[int], group=None) -> None:
    """Fail EVERY rank fast if the engine's CCOM root port is already taken.

    Rank 0 probes the port (it is the one that binds it at the first collective); the verdict is
    all-reduced over ``group`` (CPU/gloo) so the other ranks raise too instead of waiting on a
    root that will never come up. Call after the process group exists and before the first
    device graph runs.
    """
    import torch
    import torch.distributed as dist

    host, port = split_comm_id(addr)
    reason = probe_port_free(host, port) if rank == 0 else None
    flag = torch.tensor([int(reason is not None)], dtype=torch.int32, device="cpu")
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if not flag.item():
        return
    cores = sorted(int(c) for c in cores)
    detail = f": {reason}" if reason else " (reported by rank 0)"
    raise RuntimeError(
        f"CCOM bootstrap root {addr} for the engine on cores {cores[0]}-{cores[-1]} is already "
        f"in use{detail}. The port is {CCOM_BASE_PORT} + the engine's lowest core, so another "
        "process on these cores is still alive (a previous run that did not exit, or an "
        "overlapping core assignment). Stop it or move this engine to free cores; the runtime "
        "would otherwise fail rank 0 at the first collective and hang the other ranks."
    )
