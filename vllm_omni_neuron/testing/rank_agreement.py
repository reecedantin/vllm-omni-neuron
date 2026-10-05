# SPDX-License-Identifier: Apache-2.0
"""All-rank agreement check for device gates.

A multi-rank layout (TP / CP / CFG) is only verified when EVERY rank ends with the same output, not
just rank 0: a swapped CP slice or CFG branch on some ranks leaves rank 0 correct at the first call
and drifts later, or contaminates rank 0 only through a TP group it shares with a bad rank. Each rank
digests its replicated output (final latents, decoded video, actions), the digests are gathered over
a host group, and every rank that disagrees with rank 0 is reported.

A digest is small (shape, dtype, an exact SHA-256 of the bytes, non-finite count, fp64 sum / L2 /
max-abs and a fixed strided sample of up to ``sample`` values), so it is cheap to gather with
``all_gather_object`` or to write to a JSON file per rank.

Two ways to run it:

* in-process, collective (all ranks call it at the same point)::

    from vllm_omni_neuron.testing import check_rank_agreement

    report = check_rank_agreement({"latents": latents, "video": video})   # world group
    if report.rank == 0:
        print(report.summary())            # one line, ok / which ranks disagree and why
    report.raise_if_failed()               # optional: fail the gate on every rank

* file-based, when ranks are separate worker processes the gate cannot call into::

    write_rank_digest(out_dir, rank, {"latents": latents})          # in each worker
    report = compare_rank_digest_files(out_dir, world_size=16)      # in the gate, after the run

Pass only tensors that SHOULD be identical on every rank (replicated, or gathered first). A CP
shard differs by design: gather it (device all-gather, or ``host_all_gather``) before digesting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

DIGEST_VERSION = 1
_DEFAULT_SAMPLE = 4096


def tensor_digest(tensor: torch.Tensor, *, sample: int = _DEFAULT_SAMPLE) -> dict[str, Any]:
    """Return a JSON-serialisable digest of ``tensor``.

    The tensor is moved to the host FIRST (``.detach().cpu()``) and only then made contiguous /
    cast: on the Neuron Lite runtime an eager device cast or copy of a non-contiguous tensor can
    raise, while the host copy is always safe.
    """
    t = tensor.detach().cpu().contiguous()
    flat = t.reshape(-1)
    sha = hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()
    d: dict[str, Any] = {
        "v": DIGEST_VERSION,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(flat.numel()),
        "sha256": sha,
    }
    if t.is_floating_point() or t.is_complex():
        f = flat.to(torch.complex128 if t.is_complex() else torch.float64)
        if t.is_complex():
            f = torch.view_as_real(f).reshape(-1)
        finite = torch.isfinite(f)
        d["nonfinite"] = int((~finite).sum())
        ff = f[finite]
        d["sum"] = float(ff.sum()) if ff.numel() else 0.0
        d["l2"] = float(ff.norm()) if ff.numel() else 0.0
        d["maxabs"] = float(ff.abs().max()) if ff.numel() else 0.0
    else:
        f = flat.to(torch.float64)
        d["nonfinite"] = 0
        d["sum"] = float(f.sum()) if f.numel() else 0.0
        d["l2"] = float(f.norm()) if f.numel() else 0.0
        d["maxabs"] = float(f.abs().max()) if f.numel() else 0.0
    n = f.numel()
    if n and sample > 0:
        k = min(sample, n)
        idx = torch.linspace(0, n - 1, k, dtype=torch.float64).round().long()
        # non-finite values are counted above; zero them here so the sample stays strict JSON
        d["sample"] = [float(x) for x in f[idx].nan_to_num(0.0, 0.0, 0.0)]
    else:
        d["sample"] = []
    return d


def outputs_digest(outputs: Any, *, sample: int = _DEFAULT_SAMPLE) -> dict[str, dict[str, Any]]:
    """Digest a tensor, a ``{name: tensor}`` mapping, or a list/tuple of tensors.

    Returns ``{name: digest}`` (a bare tensor is named ``"output"``, list items ``"0"``, ``"1"``,
    ...). Already-computed digests (dicts carrying ``"sha256"``) pass through unchanged."""
    if isinstance(outputs, torch.Tensor):
        items = {"output": outputs}
    elif isinstance(outputs, Mapping):
        if "sha256" in outputs:
            return {"output": dict(outputs)}
        items = dict(outputs)
    elif isinstance(outputs, (list, tuple)):
        items = {str(i): o for i, o in enumerate(outputs)}
    else:
        raise TypeError(f"unsupported outputs type {type(outputs).__name__}")
    out = {}
    for name, v in items.items():
        if isinstance(v, Mapping) and "sha256" in v:
            out[str(name)] = dict(v)
        elif isinstance(v, torch.Tensor):
            out[str(name)] = tensor_digest(v, sample=sample)
        else:
            raise TypeError(f"output {name!r}: expected a tensor, got {type(v).__name__}")
    return out


def _compare(ref: dict[str, Any], got: dict[str, Any], rtol: float) -> tuple[bool, str, float]:
    """``(agrees, reason, rel)`` for one output; ``rel`` = max relative deviation seen."""
    if got["shape"] != ref["shape"] or got["dtype"] != ref["dtype"]:
        return (
            False,
            f"shape/dtype {got['shape']}/{got['dtype']} != {ref['shape']}/{ref['dtype']}",
            math.inf,
        )
    if got.get("nonfinite", 0) != ref.get("nonfinite", 0):
        return False, f"nonfinite {got.get('nonfinite')} != {ref.get('nonfinite')}", math.inf
    if got["sha256"] == ref["sha256"]:
        return True, "exact", 0.0
    scale = max(ref.get("maxabs", 0.0), 1e-30)
    a, b = ref.get("sample", []), got.get("sample", [])
    if len(a) != len(b):
        return False, "sample length differs (different sample= settings?)", math.inf
    rel_sample = max((abs(x - y) for x, y in zip(a, b)), default=0.0) / scale
    rel_l2 = abs(got.get("l2", 0.0) - ref.get("l2", 0.0)) / max(ref.get("l2", 0.0), 1e-30)
    rel_max = abs(got.get("maxabs", 0.0) - ref.get("maxabs", 0.0)) / scale
    rel = max(rel_sample, rel_l2, rel_max)
    if rtol > 0 and rel <= rtol:
        return True, f"close rel={rel:.3g}", rel
    return (
        False,
        f"bytes differ, rel={rel:.3g}" + (f" > rtol {rtol:g}" if rtol > 0 else " (exact required)"),
        rel,
    )


@dataclass
class RankAgreementReport:
    """Result of an all-rank agreement check. ``ranks[i]`` is the global rank (or the file's rank)
    of member ``i``; member 0 is the reference."""

    rank: int
    world_size: int
    ranks: list[int]
    rtol: float
    digests: list[dict[str, dict[str, Any]]]
    disagreeing: dict[int, dict[str, str]] = field(default_factory=dict)
    max_rel: dict[str, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.disagreeing and not self.problems

    @property
    def disagreeing_ranks(self) -> list[int]:
        return sorted(self.disagreeing)

    def summary(self) -> str:
        names = ",".join(sorted(self.digests[0])) if self.digests else ""
        if self.ok:
            exact = all(r == 0.0 for r in self.max_rel.values())
            how = (
                "bit-exact"
                if exact
                else f"max rel {max(self.max_rel.values(), default=0.0):.3g} <= {self.rtol:g}"
            )
            return f"rank agreement OK: {self.world_size} ranks, [{names}] {how}"
        parts = [
            f"rank {r}: " + "; ".join(f"{k} {v}" for k, v in why.items())
            for r, why in sorted(self.disagreeing.items())
        ]
        return (
            f"rank agreement FAILED: {len(self.disagreeing)}/{self.world_size} ranks disagree with "
            f"rank {self.ranks[0] if self.ranks else 0}"
            + (" | " if parts or self.problems else "")
            + " | ".join(parts + self.problems)
        )

    def to_json(self) -> dict[str, Any]:
        """Compact record for a gate's summary JSON (no samples)."""
        return {
            "ok": self.ok,
            "world_size": self.world_size,
            "rtol": self.rtol,
            "disagreeing_ranks": self.disagreeing_ranks,
            "disagreeing": {str(r): v for r, v in self.disagreeing.items()},
            "max_rel": self.max_rel,
            "problems": self.problems,
            "sha256_rank0": {
                k: v["sha256"] for k, v in (self.digests[0] if self.digests else {}).items()
            },
        }

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError(self.summary())


def compare_digests(
    digests: list[dict[str, dict[str, Any]]],
    *,
    ranks: list[int] | None = None,
    rtol: float = 0.0,
    rank: int = 0,
) -> RankAgreementReport:
    """Compare per-rank digests (``outputs_digest`` results, member order) against member 0.

    ``rtol=0`` (default) requires bit-identical outputs, which is what replicated TP/CP/CFG outputs
    give when every rank runs the same graphs. Use a small ``rtol`` (e.g. 1e-3) only for outputs
    that are legitimately reduced in a rank-dependent order, and state it in the gate."""
    ranks = list(range(len(digests))) if ranks is None else list(ranks)
    rep = RankAgreementReport(
        rank=rank, world_size=len(digests), ranks=ranks, rtol=rtol, digests=digests
    )
    if not digests:
        rep.problems.append("no digests")
        return rep
    ref = digests[0]
    for name, d in ref.items():
        if d.get("nonfinite", 0):
            rep.problems.append(f"rank {ranks[0]} {name}: {d['nonfinite']} non-finite values")
    for i, got in enumerate(digests):
        if i == 0:
            continue
        why: dict[str, str] = {}
        if set(got) != set(ref):
            why["outputs"] = f"names {sorted(got)} != {sorted(ref)}"
        for name in ref:
            if name not in got:
                continue
            agrees, reason, rel = _compare(ref[name], got[name], rtol)
            rep.max_rel[name] = max(rep.max_rel.get(name, 0.0), rel)
            if not agrees:
                why[name] = reason
        if why:
            rep.disagreeing[ranks[i]] = why
    for name in ref:
        rep.max_rel.setdefault(name, 0.0)
    return rep


def _default_group():
    """The world host (gloo) group: vLLM's world ``cpu_group`` when it exists, else the default."""
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().cpu_group
    except Exception:
        return None


def check_rank_agreement(
    outputs: Any,
    *,
    coord=None,
    group=None,
    rtol: float = 0.0,
    sample: int = _DEFAULT_SAMPLE,
) -> RankAgreementReport:
    """Collective: digest ``outputs`` on this rank, gather every rank's digest, compare to rank 0.

    Every rank of the group must call it (it is an ``all_gather_object``); every rank gets the same
    report. Scope, in order:

    * ``coord``: a GroupCoordinator (e.g. ``get_tp_group()``, the CP / CFG coordinator); members
      come back in ``coord.ranks`` order via :func:`host_all_gather_object`, so "rank 0" is the
      coordinator's first member even for the unsorted physical-mesh groups.
    * ``group``: a host (gloo) ``ProcessGroup``; members in c10d (sorted global rank) order.
    * neither: the whole world (vLLM's world ``cpu_group`` if initialised, else the default group).

    Without an initialised ``torch.distributed`` it degrades to a 1-rank report (always ok)."""
    import torch.distributed as dist

    mine = outputs_digest(outputs, sample=sample)
    if not (dist.is_available() and dist.is_initialized()):
        return compare_digests([mine], rtol=rtol)
    if coord is not None:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import host_all_gather_object

        digests = host_all_gather_object(coord, mine)
        ranks = list(coord.ranks)
        me = dist.get_rank()
        return compare_digests(digests, ranks=ranks, rtol=rtol, rank=me)
    if group is None:
        group = _default_group()
    world = dist.get_world_size(group)
    digests: list[Any] = [None] * world
    dist.all_gather_object(digests, mine, group=group)
    try:
        ranks = (
            sorted(dist.get_process_group_ranks(group)) if group is not None else list(range(world))
        )
    except Exception:
        ranks = list(range(world))
    return compare_digests(digests, ranks=ranks, rtol=rtol, rank=dist.get_rank())


def write_rank_digest(
    out_dir: str, rank: int, outputs: Any, *, sample: int = _DEFAULT_SAMPLE
) -> str:
    """Write this rank's digest to ``<out_dir>/rank_digest_<rank>.json``; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rank_digest_{int(rank):04d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"rank": int(rank), "outputs": outputs_digest(outputs, sample=sample)}, f)
    os.replace(tmp, path)
    return path


def compare_rank_digest_files(
    out_dir: str, *, world_size: int | None = None, rtol: float = 0.0
) -> RankAgreementReport:
    """Load every ``rank_digest_*.json`` in ``out_dir`` and compare to the lowest rank.

    With ``world_size`` set, a missing rank file is reported as a problem (a rank that never
    reached the check is a failed gate, not a pass)."""
    found: dict[int, dict] = {}
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if name.startswith("rank_digest_") and name.endswith(".json"):
            with open(os.path.join(out_dir, name)) as f:
                rec = json.load(f)
            found[int(rec["rank"])] = rec["outputs"]
    ranks = sorted(found)
    rep = compare_digests([found[r] for r in ranks], ranks=ranks, rtol=rtol)
    if world_size is not None:
        missing = sorted(set(range(world_size)) - set(ranks))
        if missing:
            rep.problems.append(f"missing digests for ranks {missing}")
        rep.world_size = world_size
    return rep


__all__ = [
    "RankAgreementReport",
    "check_rank_agreement",
    "compare_digests",
    "compare_rank_digest_files",
    "outputs_digest",
    "tensor_digest",
    "write_rank_digest",
]
