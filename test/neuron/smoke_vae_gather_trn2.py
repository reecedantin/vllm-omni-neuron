# SPDX-License-Identifier: Apache-2.0
"""Device smoke for the Wan VAE patch-parallel plane gather at 16 / 32 (/ 64) ranks.

Drives ``_NeuronDistributedVaeExecutor._stream_gather_planes`` -- the compiled per-chunk cut, the
Lite precompile barrier, the compiled device all-gather over the registered replica group and the
slot-major restore -- on one NeuronCore per rank, and checks every yielded ``[slots, size, H, W]``
view bit-exactly against a host (gloo) all-gather of the same bf16 planes. Also runs the full
``gather_and_blend_tiles`` entry point on a synthetic tile grid against the CPU executor path.

Each rank is set up the way ``NeuronDiffusionWorker.init_device`` does it: one visible core, Lite
imported before the process group, ``cpu:gloo,neuron:neuron`` composite backend, the VAE group as a
``new_group`` over ranks ``0..world-1``.

Run with the cores to use in ``SMOKE_CORES`` (comma list or ``lo-hi``; rank r -> cores[r])::

    python test/neuron/smoke_vae_gather_trn2.py --world 16 --out $SMOKE_OUT
    python test/neuron/smoke_vae_gather_trn2.py --world 32 --out $SMOKE_OUT
    python test/neuron/smoke_vae_gather_trn2.py --probe-offset-copy --out $SMOKE_OUT   # 1 core

``--probe-offset-copy`` isolates the runtime behaviour behind the original failure: an eager
device-to-device ``copy_`` whose SOURCE is a view with a non-zero storage offset (and the
device-to-host variant), independent of any VAE code. It runs alone, last, so a runtime error there
cannot disturb the gather results.

Exit code 0 iff every gate passed; the JSON summary is ``<out>/vae_gather_w<world>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

# Bits per element never matter here: everything is bf16 (the VAE output dtype on device).
DTYPE = torch.bfloat16


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cores(arg: str | None) -> list[int]:
    raw = arg or os.environ.get("SMOKE_CORES") or os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    cores: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return cores


# ----------------------------------------------------------------------------------------------
# per-rank process
# ----------------------------------------------------------------------------------------------


def _rank_env(rank: int, world: int, core: int, port: int) -> None:
    """Mirror NeuronDiffusionWorker.init_device's environment, one core per process."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ.pop("NEURON_RT_NUM_CORES", None)
    # Keep an inherited NEURON_RT_ROOT_COMM_ID (a port no other job on the host uses);
    # without one Lite defaults every world on the box to the same localhost port.
    os.environ.pop("NEURON_LIBRARY_PATH", None)
    os.environ.pop("SMOKE_CORES", None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_WORLD_SIZE"] = str(world)


def _init_rank(rank: int, world: int, core: int, port: int):
    if core < 0:  # --cpu dry run: gloo only, CPU tensors, no Lite, no NeuronCore
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
        )
        return torch.device("cpu"), dist.new_group(list(range(world)))
    _rank_env(rank, world, core, port)
    from vllm_neuron.vllm.platform import NeuronPlatform

    import vllm_omni_neuron  # noqa: F401  (bootstrap: registers the 'neuron' device + plugin)
    from vllm_omni_neuron.lite_compat import initialize as initialize_lite

    NeuronPlatform.set_device_count(world)
    initialize_lite()
    import torch.distributed as dist

    device = torch.device("neuron", rank)
    dist.init_process_group(
        backend="cpu:gloo,neuron:neuron",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
    )
    group = dist.new_group(list(range(world)))
    return device, group


def _barrier(group) -> None:
    """CPU (gloo) barrier: dist.barrier() would probe the 'neuron' device hooks."""
    import torch.distributed as dist

    dist.all_reduce(torch.zeros(1, dtype=torch.int32), group=group)


def _executor(group, world: int, rank: int, cls=None):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor,
    )

    cls = cls or _NeuronDistributedVaeExecutor
    ex = cls.__new__(cls)
    ex.group, ex.world_size, ex.rank = group, world, rank
    ex.parallel_size, ex.parallel_mode = world, "tile"
    return ex


def _cpu_reference_executor(group, world: int, rank: int):
    """Same class, compiled graphs run EAGERLY on CPU tensors, gather over gloo (``--cpu`` dry run)."""
    ex = _executor(group, world, rank)
    ex._compile_device_graph = lambda name, key, fn: fn
    return ex


def _rank_planes(rank, slots, planes, h, w):
    """Deterministic per-rank payload; rank 0 regenerates every rank's copy as the oracle, so
    the reference needs no host collective (the production gather path is the ONLY collective
    traffic in the check)."""
    gen = torch.Generator().manual_seed(1000 + rank)
    return torch.randn(slots, planes, h, w, generator=gen).to(DTYPE)


def _check_gather(
    ex, group, device, rank, world, *, slots, planes, h, w, chunk_planes, tag, graphs
):
    """One _stream_gather_planes run vs a host all-gather; bit-exact gate on rank 0."""
    host = _rank_planes(rank, slots, planes, h, w)
    local = host.to(device)
    ex._await_device_tensor(local)

    starts = list(range(0, planes, chunk_planes))
    t0 = time.time()
    out = list(ex._stream_gather_planes(local, chunk_planes, planes))
    first_s = time.time() - t0
    t0 = time.time()
    out2 = list(ex._stream_gather_planes(local, chunk_planes, planes))
    warm_s = time.time() - t0
    del local

    result = {
        "tag": tag,
        "world": world,
        "slots": slots,
        "planes": planes,
        "tile": [h, w],
        "chunk_planes": chunk_planes,
        "chunks": len(starts),
        "chunk_sizes": [min(chunk_planes, planes - s) for s in starts],
        "local_chunk_mib": round(slots * min(chunk_planes, planes) * h * w * 2 / 2**20, 2),
        "gathered_chunk_mib": round(
            world * slots * min(chunk_planes, planes) * h * w * 2 / 2**20, 1
        ),
        "first_s": round(first_s, 3),
        "warm_s": round(warm_s, 3),
    }
    if rank != 0:
        assert all(o is None for o in out) and all(o is None for o in out2)
        return result
    ref = [_rank_planes(r, slots, planes, h, w) for r in range(world)]
    exact, max_abs, mismatched = True, 0.0, []
    for run_name, run in (("first", out), ("warm", out2)):
        for start, gathered in zip(starts, run):
            size = min(chunk_planes, planes - start)
            # planes-major [world, size, slots, H, W], a zero-offset view of the collective's output
            assert tuple(gathered.shape) == (world, size, slots, h, w), tuple(gathered.shape)
            assert gathered.storage_offset() == 0
            got = gathered.cpu().transpose(1, 2)  # -> [world, slots, size, H, W] for the comparison
            want = torch.stack([r.narrow(1, start, size) for r in ref], dim=0)
            if not torch.equal(got, want):
                exact = False
                diff = (got.float() - want.float()).abs()
                max_abs = max(max_abs, diff.max().item())
                mismatched.append(
                    {
                        "run": run_name,
                        "start": start,
                        "bad_ranks": (diff.flatten(1).amax(1) > 0).nonzero().flatten().tolist(),
                    }
                )
    result.update(exact=exact, max_abs=max_abs, mismatched=mismatched[:8], passed=exact)
    return result


# ----------------------------------------------------------------------------------------------
# HBM sampling (neuron-monitor), parent process
# ----------------------------------------------------------------------------------------------

# On PATH in the Neuron SDK environment (it ships next to neuron-top / neuron-ls).
NEURON_MONITOR = shutil.which("neuron-monitor") or "neuron-monitor"
# neuron-monitor's floor is a 1 s period (a sub-second `period` silently becomes 2 s); several
# instances started a fraction of a second apart give a finer merged timeline.
HBM_MONITOR_INSTANCES = 4
HBM_SETTLE_S = 2.5  # >= 2 monitor periods: the sample after a request is of the idle state


class HbmMonitor:
    """Per-logical-core device memory over time, from neuron-monitor's runtime reports.

    The sysfs counters (`/sys/devices/virtual/neuron_device/*/neuron_core*/stats/memory_usage/
    device_mem/present`) do NOT track the Lite runtime's allocations (a core holding 2.1 GiB per
    neuron-monitor showed 8 MiB there, and its `peak` is a never-reset lifetime maximum), so this
    is the measurement to use. `neuroncore_memory_usage` keys are logical core ids
    (0..63 at LNC=2); totals are summed over every runtime process on that core.
    """

    def __init__(self, cores: list[int], out_dir: str, instances: int = HBM_MONITOR_INSTANCES):
        self.cores = set(cores)
        self.samples: list[tuple[float, dict[int, int]]] = []  # (time.time(), {core: bytes})
        self._procs = []
        self._threads = []
        self._lock = __import__("threading").Lock()
        self.error = None
        self._cfg = os.path.join(out_dir, "neuron_monitor_cfg.json")
        with open(self._cfg, "w") as f:
            json.dump(
                {
                    "period": "1s",
                    "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "memory_used"}]}],
                    "system_metrics": [],
                },
                f,
            )
        self.instances = instances

    def start(self) -> "HbmMonitor":
        import subprocess
        import threading

        if shutil.which(NEURON_MONITOR) is None:
            self.error = f"{NEURON_MONITOR} not found on PATH"
            return self
        for i in range(self.instances):
            try:
                p = subprocess.Popen(
                    [NEURON_MONITOR, "-c", self._cfg],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except Exception as error:
                self.error = repr(error)
                break
            self._procs.append(p)
            t = threading.Thread(target=self._reader, args=(p,), daemon=True)
            t.start()
            self._threads.append(t)
            time.sleep(1.0 / self.instances)
        return self

    def _reader(self, proc) -> None:
        for line in proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            now = time.time()
            snap: dict[int, int] = {}
            for rt in d.get("neuron_runtime_data") or []:
                try:
                    per = rt["report"]["memory_used"]["neuron_runtime_used_bytes"][
                        "usage_breakdown"
                    ]["neuroncore_memory_usage"]
                except (KeyError, TypeError):
                    continue
                for k, v in per.items():
                    c = int(k)
                    if c in self.cores:
                        snap[c] = snap.get(c, 0) + sum(
                            x for x in v.values() if isinstance(x, (int, float))
                        )
            for c in self.cores:
                snap.setdefault(c, 0)
            with self._lock:
                self.samples.append((now, snap))

    def stop(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=5)
        with self._lock:
            self.samples.sort(key=lambda s: s[0])

    # -- queries -------------------------------------------------------------------------------

    def present_at(self, ts: float) -> dict[int, float]:
        """Latest sample at or before `ts` (GiB per core; the newest overall if none precedes)."""
        before = [s for s in self.samples if s[0] <= ts]
        if not before:
            return {}
        _, snap = before[-1]
        return {c: round(b / 2**30, 3) for c, b in sorted(snap.items())}

    def peak_between(self, t0: float, t1: float) -> dict[int, float]:
        peak: dict[int, int] = {}
        for ts, snap in self.samples:
            if t0 <= ts <= t1:
                for c, b in snap.items():
                    peak[c] = max(peak.get(c, 0), b)
        return {c: round(b / 2**30, 3) for c, b in sorted(peak.items())}

    def rate_hz(self) -> float | None:
        if len(self.samples) < 2:
            return None
        span = self.samples[-1][0] - self.samples[0][0]
        return round((len(self.samples) - 1) / span, 2) if span > 0 else None

    def resolve(self, windows: dict) -> dict:
        """Per-request present / peak for a check's recorded `hbm_windows`:
        {"before": ts, "requests": [{"t_start", "t_end", "t_settled"}]} -> GiB per core."""
        if self.error:
            return {"error": self.error}
        slack = 0.3  # a sample taken just before the call still describes its starting state
        out = {
            "sample_hz": self.rate_hz(),
            "n_samples": len(self.samples),
            "before_gib": self.present_at(windows["before"]),
            "requests": [],
        }
        for w in windows["requests"]:
            out["requests"].append(
                {
                    "present_gib": self.present_at(w["t_settled"]),
                    "peak_gib": self.peak_between(w["t_start"] - slack, w["t_settled"]),
                    "wall_s": round(w["t_end"] - w["t_start"], 3),
                }
            )
        if out["requests"]:
            first, last = out["requests"][0]["present_gib"], out["requests"][-1]["present_gib"]
            out["growth_gib"] = {
                c: round(last.get(c, 0) - first.get(c, 0), 3) for c in sorted(first or last)
            }
            out["growth_max_gib"] = max(out["growth_gib"].values()) if out["growth_gib"] else None
        return out


def _grid_setup(world, rank, *, grid, tile_hw, stride_hw, frame_shape, device, full_hw=None):
    """Synthetic tile grid: tile (row, col) has id row*gw+col, dealt round-robin to ranks.

    ``full_hw=None``: every tile is full size (a canvas one tile overhang larger, output =
    grid * stride). ``full_hw=(H, W)``: the real ``tile_split`` geometry -- tiles start every
    stride over an H x W canvas, so the last row / column are SMALLER than the tile and arrive
    zero-padded to the slot shape (as ``_pack_local_tiles`` pads them), with their real size in
    the metadata. ``grid`` is then derived and ignored."""
    th, tw = tile_hw
    sh, sw = stride_hw
    if full_hw is not None:
        full_h, full_w = full_hw
        gh, gw = len(range(0, full_h, sh)), len(range(0, full_w, sw))
    else:
        gh, gw = grid
        full_h, full_w = gh * sh, gw * sw
    tiles = gh * gw
    slots = max(1, math.ceil(tiles / world))
    planes = math.prod(frame_shape)
    tid_coord = {r * gw + c: (r, c) for r in range(gh) for c in range(gw)}
    torch.manual_seed(7)
    # The full "image" everyone agrees on, so the oracle is simply the blend of consistent tiles.
    if full_hw is not None:
        canvas = torch.randn(*frame_shape, full_h, full_w).to(DTYPE)
    else:
        canvas = torch.randn(*frame_shape, (gh - 1) * sh + th, (gw - 1) * sw + tw).to(DTYPE)
    local_tiles = torch.zeros(slots, *frame_shape, th, tw, dtype=DTYPE)
    meta = torch.full((slots, 1 + 2), -1, dtype=torch.int64)
    for slot, tid in enumerate(range(rank, tiles, world)):
        row, col = tid_coord[tid]
        tile = canvas[..., row * sh : row * sh + th, col * sw : col * sw + tw]
        local_tiles[slot, ..., : tile.shape[-2], : tile.shape[-1]] = tile
        meta[slot, 0] = tid
        meta[slot, 1], meta[slot, 2] = tile.shape[-2], tile.shape[-1]
    grid_spec = type("GridSpec", (), {"grid_shape": (gh, gw)})()
    return dict(
        local_tiles=local_tiles,
        meta=meta,
        grid_spec=grid_spec,
        tid_coord=tid_coord,
        full_hw=(full_h, full_w),
        grid=(gh, gw),
        planes=planes,
        slots=slots,
        canvas=canvas,
    )


def _per_frame_errors(got, canvas, edge=24):
    """Max |got - canvas| per frame (the last frame_shape axis), split into the interior and the
    right / bottom ``edge`` px strips (where padded edge tiles land), plus the first bad frame."""
    d = (got - canvas).abs()
    d = d.flatten(0, -4) if d.ndim > 3 else d.unsqueeze(0)  # [lead, T, H, W]
    d = d.amax(0)  # [T, H, W]
    inner = d[:, :-edge, :-edge].flatten(1).amax(1)
    right = d[:, :, -edge:].flatten(1).amax(1)
    bottom = d[:, -edge:, :].flatten(1).amax(1)
    worst = d.flatten(1).amax(1)
    bad = (worst > 0.0625).nonzero().flatten().tolist()
    return {
        "frames": int(d.shape[0]),
        "bad_frames": len(bad),
        "first_bad_frames": bad[:8],
        "max_abs_inner": round(inner.max().item(), 5),
        "max_abs_right_strip": round(right.max().item(), 5),
        "max_abs_bottom_strip": round(bottom.max().item(), 5),
        "max_abs_per_frame_sample": [
            round(v, 4) for v in worst[:: max(1, d.shape[0] // 12)].tolist()
        ],
    }


def _check_blend(
    ex_dev,
    group,
    device,
    rank,
    world,
    *,
    grid,
    tile_hw,
    stride_hw,
    frame_shape,
    gather_budget_bytes,
    tag,
    to_host=False,
    repeats=2,
    core=None,
    full_hw=None,
):
    """gather_and_blend_tiles on device, `repeats` requests back to back (first + warm), each
    checked against the shared canvas; per-request HBM (present / peak) of this rank's core."""
    setup = _grid_setup(
        world,
        rank,
        grid=grid,
        tile_hw=tile_hw,
        stride_hw=stride_hw,
        frame_shape=frame_shape,
        device=device,
        full_hw=full_hw,
    )
    grid = setup["grid"]
    th, tw = tile_hw
    sh, sw = stride_hw
    full_h, full_w = setup["full_hw"]
    blend_h, blend_w = th - sh, tw - sw
    common = dict(
        full_height=full_h,
        full_width=full_w,
        stride_height=sh,
        stride_width=sw,
        blend_height=blend_h,
        blend_width=blend_w,
        clamp=True,
    )
    ex_dev.MAX_DEVICE_GATHER_BYTES = gather_budget_bytes
    plane_bytes = world * setup["slots"] * th * tw * 2
    chunk_planes = max(1, min(setup["planes"], max(1, gather_budget_bytes // plane_bytes)))

    local_dev = setup["local_tiles"].to(device)
    ex_dev._await_device_tensor(local_dev)
    meta_dev = ex_dev.gather_tensors(setup["meta"])
    # HBM: the parent process samples neuron-monitor; this rank only records WHEN each request
    # ran and when the ranks had settled afterwards (barrier + HBM_SETTLE_S), and the parent
    # resolves present / peak per core from the timeline (HbmMonitor.resolve).
    settle = device.type != "cpu"
    if settle:
        _barrier(group)
        time.sleep(HBM_SETTLE_S)
    windows = {"before": time.time(), "requests": []}
    outs, times = [], []
    for _ in range(repeats):
        t0 = time.time()
        got = ex_dev.gather_and_blend_tiles(
            local_dev, meta_dev, setup["grid_spec"], setup["tid_coord"], to_host=to_host, **common
        )
        if rank == 0 and got is not None and got.device.type != "cpu":
            ex_dev._await_device_tensor(got)
        t1 = time.time()
        times.append(t1 - t0)
        if rank == 0:
            outs.append(got.cpu().float() if got.device.type != "cpu" else got.float())
            del got  # the pipeline would hand the frames on; here free the device copy
        if settle:
            _barrier(group)
            time.sleep(HBM_SETTLE_S)
        windows["requests"].append({"t_start": t0, "t_end": t1, "t_settled": time.time()})
    first_s, warm_s = times[0], (min(times[1:]) if len(times) > 1 else times[0])

    result = {
        "tag": tag,
        "world": world,
        "grid": list(grid),
        "tiles": grid[0] * grid[1],
        "slots": setup["slots"],
        "planes": setup["planes"],
        "tile": [th, tw],
        "chunk_planes": chunk_planes,
        "chunks": math.ceil(setup["planes"] / chunk_planes),
        "full_hw": [full_h, full_w],
        "to_host": to_host,
        "repeats": repeats,
        "times_s": [round(t, 3) for t in times],
        "first_s": round(first_s, 3),
        "warm_s": round(warm_s, 3),
        "hbm_windows": windows,
    }
    if rank != 0:
        return result
    got_h = outs[0]
    assert all(o.is_contiguous() for o in outs)
    # Oracle: every tile is a window of ONE canvas all ranks derive from the same seed, so the
    # blend of consistent tiles reproduces the canvas up to bf16 blend rounding at the seams
    # (the CPU executor path measures 4.8e-4 on these grids; a misplaced tile or a mis-ordered
    # gather is O(1)). Rank 0 needs no collective to know the answer.
    canvas = setup["canvas"][..., :full_h, :full_w].float().clamp(-1, 1)
    rel_canvas = ((got_h - canvas).norm() / canvas.norm()).item()
    max_abs = (got_h - canvas).abs().max().item()
    # Every frame of every request, not just the aggregate: a corrupted edge strip in late frames
    # is a small fraction of the norm.
    per_request = [_per_frame_errors(o, canvas) for o in outs]
    result.update(
        shape=list(got_h.shape),
        padded_edges=full_hw is not None,
        rel_vs_canvas=rel_canvas,
        max_abs_vs_canvas=max_abs,
        per_frame=per_request[0],
        bad_frames_per_request=[r["bad_frames"] for r in per_request],
        warm_bit_equal_first=all(torch.equal(got_h, o) for o in outs[1:]),
        passed=bool(
            tuple(got_h.shape) == tuple(canvas.shape)
            and rel_canvas <= 2e-3
            and max_abs <= 0.0625
            and all(r["bad_frames"] == 0 for r in per_request)
            and all(torch.equal(got_h, o) for o in outs[1:])
        ),
    )
    return result


def _rank_main(
    rank: int, world: int, cores: list[int], port: int, out_dir: str, cases: str, repeats: int = 3
) -> None:
    import faulthandler
    import signal

    faulthandler.enable()
    # A hang leaves a Python stack of every rank in the job log every 5 minutes (and on SIGUSR1)
    # instead of needing a kill to see where the ranks sit.
    faulthandler.dump_traceback_later(300, repeat=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    core = cores[rank]
    report = {"rank": rank, "core": core, "world": world, "checks": [], "ok": True}
    try:
        device, group = _init_rank(rank, world, core, port)
        import torch.distributed as dist

        ex = _executor(group, world, rank)
        if device.type == "cpu":
            ex = _cpu_reference_executor(group, world, rank)  # dry run: eager graphs, gloo gather
        graphs: dict = {}

        gather_cases = []
        if world >= 32:
            # Wan2.2-TI2V 704x1280 x 121 frames at 32 ranks: 28 tiles -> 1 slot, 363 planes, 3 chunks
            # of 128/128/107 at the 512 MiB budget (the failing layout), real 256x256 tiles.
            gather_cases.append(
                dict(
                    slots=1, planes=363, h=256, w=256, chunk_planes=128, tag="a4-363p-3chunks-256px"
                )
            )
            gather_cases.append(
                dict(slots=1, planes=99, h=256, w=256, chunk_planes=256, tag="single-chunk-256px")
            )
            gather_cases.append(
                dict(slots=2, planes=48, h=64, w=64, chunk_planes=16, tag="two-slots-3chunks")
            )
            gather_cases.append(
                dict(slots=1, planes=7, h=16, w=16, chunk_planes=3, tag="ragged-3chunks-tiny")
            )
        else:
            # Cosmos3-Super 832x480 x 189 frames at 16 ranks: 8 tiles -> 1 slot, 567 planes,
            # 3 chunks of 256/256/55 (the failing layout), real 256x256 tiles.
            gather_cases.append(
                dict(
                    slots=1, planes=567, h=256, w=256, chunk_planes=256, tag="a1-567p-3chunks-256px"
                )
            )
            gather_cases.append(
                dict(slots=1, planes=99, h=256, w=256, chunk_planes=256, tag="single-chunk-256px")
            )
            gather_cases.append(
                dict(slots=2, planes=48, h=64, w=64, chunk_planes=16, tag="two-slots-3chunks")
            )
        if cases == "quick":
            gather_cases = gather_cases[-1:]
        elif cases in ("hbm189", "edge"):
            gather_cases = []  # only the blend requests below
        for case in gather_cases:
            try:
                res = _check_gather(ex, group, device, rank, world, graphs=graphs, **case)
            except Exception as error:  # keep going: the next case is still informative
                res = {
                    "tag": case["tag"],
                    "world": world,
                    "passed": False,
                    "error": repr(error),
                    "trace": traceback.format_exc(),
                }
            res["check"] = "stream_gather_planes"
            report["checks"].append(res)
            if rank == 0:
                print(
                    f"[smoke] gather {case['tag']}: passed={res.get('passed')} first={res.get('first_s')}s warm={res.get('warm_s')}s {res.get('error', '')}",
                    flush=True,
                )
                if res.get("trace"):
                    print(res["trace"], flush=True)
            _barrier(group)

        # Full entry point, multi-chunk by budget: 28 tiles (the TI2V 704x1280 grid class), 3 planes x 16
        # frames = 48 planes, budget sized for 16-plane chunks -> 3 chunks; 64x64 tiles, stride 48.
        blend_cases = [
            dict(
                grid=(4, 7),
                tile_hw=(64, 64),
                stride_hw=(48, 48),
                frame_shape=(3, 16),
                tag="blend-28tiles-3chunks",
            )
        ]
        if cases == "hbm189":
            blend_cases = []
        if cases not in ("quick", "hbm189"):
            blend_cases.append(
                dict(
                    grid=(2, 4),
                    tile_hw=(64, 64),
                    stride_hw=(48, 48),
                    frame_shape=(3, 16),
                    tag="blend-8tiles-1slot-3chunks",
                )
            )
        if cases in ("full", "hbm189") and device.type != "cpu":
            # The Cosmos3-Super 832x480 x 189-frame layout (2x4 tiles of 256 px, stride 224/192,
            # 3 x 189 = 567 planes; at 8 ranks one slot each) at the production 512 MiB budget:
            # `repeats` back-to-back requests, device-resident output (and host-streamed output
            # in `full`), per-rank HBM (present / peak) after every request.
            for to_host in (False,) if cases == "hbm189" else (False, True):
                blend_cases.append(
                    dict(
                        grid=(2, 4),
                        tile_hw=(256, 256),
                        stride_hw=(224, 192),
                        frame_shape=(3, 189),
                        tag=f"blend-189f-832x480-{'host' if to_host else 'device'}-out",
                        to_host=to_host,
                        budget=512 * 1024 * 1024,
                        repeats=repeats,
                    )
                )
        if cases == "edge":
            # Real tile_split geometry with PADDED edge tiles (requests/A1-vae-devgather-edge-
            # corruption.md): every frame of every request compared, right / bottom strips apart.
            blend_cases = [
                # Cosmos3 / TI2V patchified 832x480: 240x416, tile 128, strides 112/96 -> 3x5
                # tiles, last row 16 px, last column 32 px; 12 x 189 = 2268 planes.
                dict(
                    full_hw=(240, 416),
                    grid=None,
                    tile_hw=(128, 128),
                    stride_hw=(112, 96),
                    frame_shape=(12, 189),
                    tag="edge-patchified-240x416-host-out",
                    to_host=True,
                    budget=512 * 1024 * 1024,
                    repeats=repeats,
                ),
                dict(
                    full_hw=(240, 416),
                    grid=None,
                    tile_hw=(128, 128),
                    stride_hw=(112, 96),
                    frame_shape=(12, 189),
                    tag="edge-patchified-240x416-device-out",
                    to_host=False,
                    budget=512 * 1024 * 1024,
                    repeats=repeats,
                ),
                # same grid, small budget: many chunks and a ragged tail
                dict(
                    full_hw=(240, 416),
                    grid=None,
                    tile_hw=(128, 128),
                    stride_hw=(112, 96),
                    frame_shape=(12, 189),
                    tag="edge-patchified-240x416-9chunks-host-out",
                    to_host=True,
                    budget=128 * 1024 * 1024,
                    repeats=repeats,
                ),
                # pixel-space 480x832, tile 256, strides 224/192: 567 planes in 256/256/55 chunks
                dict(
                    full_hw=(480, 832),
                    grid=None,
                    tile_hw=(256, 256),
                    stride_hw=(224, 192),
                    frame_shape=(3, 189),
                    tag="edge-480x832-3chunks-device-out",
                    to_host=False,
                    budget=512 * 1024 * 1024,
                    repeats=repeats,
                ),
            ]
        if cases == "hbm189" and device.type != "cpu":
            # Approximation of the 32-rank share on 8 ranks: at 32 ranks the 8 decode tiles sit in
            # a 32-slot gather (24 empty slots), so plane_bytes is 4x the 8-rank figure -> 128-plane
            # chunks, FIVE chunks for 567 planes, the gathered chunk still at the 512 MiB budget.
            # 512 px tiles on 8 ranks give the same plane_bytes / chunking / gathered-chunk size, and
            # a 4x larger merged chunk and output (1664x960 canvas) on rank 0 -- a superset of the
            # rank-0 transient at 32 ranks.
            blend_cases.append(
                dict(
                    grid=(2, 4),
                    tile_hw=(512, 512),
                    stride_hw=(448, 384),
                    frame_shape=(3, 189),
                    tag="blend-189f-32rank-share-512px-device-out",
                    to_host=False,
                    budget=512 * 1024 * 1024,
                    repeats=repeats,
                )
            )
        for case in blend_cases:
            case = dict(case)
            if case.get("full_hw"):
                fh, fw = case["full_hw"]
                n_tiles = len(range(0, fh, case["stride_hw"][0])) * len(
                    range(0, fw, case["stride_hw"][1])
                )
            else:
                n_tiles = case["grid"][0] * case["grid"][1]
            slots = max(1, math.ceil(n_tiles / world))
            budget = case.pop("budget", None) or (
                world * slots * 64 * 64 * 2 * 16
            )  # 16 planes/chunk
            try:
                res = _check_blend(
                    ex,
                    group,
                    device,
                    rank,
                    world,
                    gather_budget_bytes=budget,
                    core=core if device.type != "cpu" else None,
                    **case,
                )
            except Exception as error:
                res = {
                    "tag": case["tag"],
                    "world": world,
                    "passed": False,
                    "error": repr(error),
                    "trace": traceback.format_exc(),
                }
            res["check"] = "gather_and_blend_tiles"
            report["checks"].append(res)
            if rank == 0:
                print(
                    f"[smoke] blend {case['tag']}: passed={res.get('passed')} rel={res.get('rel_vs_canvas')} times={res.get('times_s')} bad_frames={res.get('bad_frames_per_request')} per_frame={res.get('per_frame')} {res.get('error', '')}",
                    flush=True,
                )
                if res.get("trace"):
                    print(res["trace"], flush=True)
            _barrier(group)

        if rank == 0 and device.type != "cpu":
            # Diagnostic only (not a gate): does the device->HOST copy of an offset view behave?
            # Production never does this (results are read from base tensors), but the answer
            # tells us whether the runtime bug is specific to the device->device path.
            try:
                base = (
                    torch.arange(2 * 4096, dtype=torch.int32).reshape(2, 4096).to(DTYPE).to(device)
                )
                ex._await_device_tensor(base)
                view = base[1]
                host = view.cpu()
                ok = torch.equal(host, torch.arange(4096, 2 * 4096, dtype=torch.int32).to(DTYPE))
                report["offset_view_to_host_copy"] = {
                    "ok": bool(ok),
                    "note": "" if ok else "WRONG DATA",
                }
            except Exception as error:
                report["offset_view_to_host_copy"] = {"ok": False, "error": repr(error)}
        report["ok"] = all(c.get("passed", "error" not in c) for c in report["checks"])
        _barrier(group)
        dist.destroy_process_group()
    except Exception as error:
        report["ok"] = False
        report["error"] = repr(error)
        report["trace"] = traceback.format_exc()
    with open(os.path.join(out_dir, f"vae_gather_w{world}_rank{rank}.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


# ----------------------------------------------------------------------------------------------
# runtime probe (single core): eager copies whose source is an offset view
# ----------------------------------------------------------------------------------------------


def _probe_main(
    rank: int, world: int, cores: list[int], port: int, out_dir: str, _cases: str
) -> None:
    import faulthandler

    faulthandler.enable()
    report = {"core": cores[0], "probes": {}}
    try:
        device, group = _init_rank(0, 1, cores[0], port)
        n = 1 << 20  # 1 Mi bf16 = 2 MiB per row; row 1 starts at byte offset 2^21
        base = torch.zeros(2, n, dtype=DTYPE, device=device)
        base_wait = base.view(-1)[:1]
        torch.empty_like(base_wait).copy_(base_wait)  # base tensor: must work
        report["probes"]["base_offset0_d2d"] = {"ok": True}
        # Expect: nrt_tensor_copy status=2 with TDRV 'Cannot copy 18446744073707454466 bytes'
        # (= 2 - 2^21 as uint64) if the runtime sizes the copy as dst_bytes - src_offset_bytes.
        src = base[1].view(-1)[:1]
        try:
            torch.empty_like(src).copy_(src)
            report["probes"]["offset_view_d2d"] = {"ok": True, "offset_elems": n}
        except Exception as error:
            report["probes"]["offset_view_d2d"] = {
                "ok": False,
                "offset_elems": n,
                "error": repr(error),
                "expected_tdrv_bytes": str((2 - 2 * n) % 2**64),
            }
        try:
            host = base[1].view(-1)[:4].cpu()
            report["probes"]["offset_view_d2h"] = {
                "ok": bool(torch.equal(host, torch.zeros(4, dtype=DTYPE)))
            }
        except Exception as error:
            report["probes"]["offset_view_d2h"] = {"ok": False, "error": repr(error)}
        import torch.distributed as dist

        dist.destroy_process_group()
    except Exception as error:
        report["error"] = repr(error)
        report["trace"] = traceback.format_exc()
    with open(os.path.join(out_dir, "offset_copy_probe.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


# ----------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--cores", default=None, help="comma list / lo-hi; default SMOKE_CORES")
    ap.add_argument("--out", default=os.environ.get("SMOKE_OUT", "."))
    ap.add_argument(
        "--cases",
        choices=["full", "quick", "hbm189", "edge"],
        default="full",
        help="hbm189: only the 189-frame 832x480 device-out blend (+ the 32-rank-share case), "
        "`--repeats` back-to-back requests with per-rank HBM",
    )
    ap.add_argument("--repeats", type=int, default=3, help="requests per 189f blend case")
    ap.add_argument(
        "--no-hbm-monitor", action="store_true", help="skip the neuron-monitor HBM sampler"
    )
    ap.add_argument("--probe-offset-copy", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="dry-run the harness on CPU/gloo (no cores)")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cores = [-1] * args.world if args.cpu else _cores(args.cores)
    port = _free_port()

    if args.probe_offset_copy:
        if not cores:
            sys.exit("no cores (SMOKE_CORES / --cores)")
        mp.spawn(_probe_main, args=(1, cores[:1], port, args.out, args.cases), nprocs=1, join=True)
        with open(os.path.join(args.out, "offset_copy_probe.json")) as f:
            report = json.load(f)
        print("PROBE_SUMMARY " + json.dumps(report))
        return 0 if "error" not in report else 1

    world = args.world
    if len(cores) < world:
        sys.exit(f"need {world} cores, have {len(cores)}: {cores}")
    cores = cores[:world]
    print(f"[smoke] world={world} cores={cores[0]}-{cores[-1]} port={port}", flush=True)
    monitor = None
    if not args.cpu and not args.no_hbm_monitor:
        monitor = HbmMonitor(cores, args.out).start()
        time.sleep(1.5)  # first samples before any rank opens its core
    t0 = time.time()
    try:
        mp.spawn(
            _rank_main,
            args=(world, cores, port, args.out, args.cases, args.repeats),
            nprocs=world,
            join=True,
        )
    finally:
        wall = time.time() - t0
        if monitor is not None:
            monitor.stop()
            with open(os.path.join(args.out, f"hbm_timeline_w{world}.json"), "w") as f:
                json.dump(
                    {
                        "cores": sorted(monitor.cores),
                        "sample_hz": monitor.rate_hz(),
                        "error": monitor.error,
                        "samples": [
                            [round(ts, 3), {str(c): b for c, b in sorted(snap.items())}]
                            for ts, snap in monitor.samples
                        ],
                    },
                    f,
                )

    ranks = []
    for r in range(world):
        path = os.path.join(args.out, f"vae_gather_w{world}_rank{r}.json")
        if os.path.exists(path):
            with open(path) as f:
                ranks.append(json.load(f))
        else:
            ranks.append({"rank": r, "ok": False, "error": "no report written"})
    rank0 = ranks[0]
    bad_ranks = [r["rank"] for r in ranks if not r.get("ok", False)]
    checks = rank0.get("checks", [])
    core_of_rank = {str(c): r for r, c in enumerate(cores)}
    for c in checks:
        if monitor is not None and c.get("hbm_windows"):
            hbm = monitor.resolve(c["hbm_windows"])
            hbm["core_of_rank"] = core_of_rank
            c["hbm"] = hbm
            print(
                f"[smoke] hbm {c.get('tag')}: before={hbm.get('before_gib')} "
                + " ".join(
                    f"req{i}: present={r['present_gib']} peak={r['peak_gib']}"
                    for i, r in enumerate(hbm.get("requests", []))
                )
                + f" growth_max_gib={hbm.get('growth_max_gib')} sample_hz={hbm.get('sample_hz')}",
                flush=True,
            )
    summary = {
        "world": world,
        "cores": [cores[0], cores[-1]],
        "wall_s": round(wall, 1),
        "passed": not bad_ranks and bool(checks) and all(c.get("passed") for c in checks),
        "bad_ranks": bad_ranks,
        "rank0_error": rank0.get("error"),
        "offset_view_to_host_copy": rank0.get("offset_view_to_host_copy"),
        "checks": [
            {k: v for k, v in c.items() if k not in ("trace", "mismatched", "hbm_windows")}
            | ({"mismatched": c["mismatched"]} if c.get("mismatched") else {})
            for c in checks
        ],
        "other_rank_errors": {str(r["rank"]): r.get("error") for r in ranks[1:] if r.get("error")},
    }
    with open(os.path.join(args.out, f"vae_gather_w{world}.json"), "w") as f:
        json.dump({"summary": summary, "ranks": ranks}, f, indent=1, default=str)
    print("SMOKE_SUMMARY " + json.dumps(summary, default=str))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
