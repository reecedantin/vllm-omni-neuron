# SPDX-License-Identifier: Apache-2.0
"""Debug dumps for full-size accuracy gates (inactive unless ``WAN22_GATE_DUMP`` is set).

``WAN22_GATE_DUMP=<dir>`` turns on two records for the FIRST real request of a process:

* Teacher-forced step records (output rank only). At the denoise steps listed in
  ``WAN22_GATE_STEPS`` (comma list, default ``0,24,49``; DMD2 has steps 0-2) the DiT input
  (``s<i>_hidden_states``), its timestep (``s<i>_timestep``), the first DiT prediction of the
  step (``s<i>_pred``: the positive branch, or the DMD2 flow) and, with CFG, the combined
  prediction the scheduler consumed (``s<i>_noise_pred``) are saved as float32 host tensors.
  A CPU reference can then recompute exactly those steps from the device's own inputs.
* Per-rank agreement record (every rank). After the denoise loop each rank writes
  ``rank<r>.json`` with the SHA-256 of its final latent (float32 host bytes), its shape and two
  sums, and saves the latent itself as ``rank<r>.final.pt``. All ranks must hold the same
  latent: the VAE patch-parallel ranks each decode their own tiles from their own copy.

Device tensors are copied to the host before any cast (an eager cast of a device tensor is
rejected under the Lite runtime).
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.distributed as dist


def _root() -> str | None:
    return os.environ.get("WAN22_GATE_DUMP") or None


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _steps() -> set[int]:
    raw = os.environ.get("WAN22_GATE_STEPS", "0,24,49")
    return {int(v) for v in raw.split(",") if v.strip()}


def _host(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().float()


def begin_request(pipe, req) -> None:
    """Count real requests (the warmup dummy run is not one)."""
    if not _root() or getattr(req, "request_ids", None) == ["dummy_req_id"]:
        return
    pipe._gate_request = getattr(pipe, "_gate_request", -1) + 1


def _active(pipe) -> bool:
    return bool(_root()) and getattr(pipe, "_gate_request", -1) == 0


def step_tensor(pipe, step: int | None, name: str, tensor) -> None:
    """Save ``tensor`` as ``s<step>_<name>.pt`` (output rank, first request, selected steps).
    The first write of a name in a step wins, so a sequential-CFG step keeps its positive call."""
    if step is None or not isinstance(tensor, torch.Tensor) or not _active(pipe):
        return
    if _rank() != 0 or step not in _steps():
        return
    root = _root()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"s{step}_{name}.pt")
    if not os.path.exists(path):
        torch.save(_host(tensor), path)


def rank_digest(pipe, latents) -> None:
    """Every rank: SHA-256 + sums of its final latent, and the latent itself."""
    if not isinstance(latents, torch.Tensor) or not _active(pipe):
        return
    root = _root()
    os.makedirs(root, exist_ok=True)
    rank = _rank()
    host = _host(latents)
    record = {
        "rank": rank,
        "shape": list(host.shape),
        "sha256": hashlib.sha256(host.numpy().tobytes()).hexdigest(),
        "sum": float(host.double().sum()),
        "abs_sum": float(host.double().abs().sum()),
        "finite": bool(torch.isfinite(host).all()),
    }
    torch.save(host, os.path.join(root, f"rank{rank}.final.pt"))
    with open(os.path.join(root, f"rank{rank}.json"), "w") as f:
        json.dump(record, f)


def current_step() -> int | None:
    """Denoise step index the loop set in the forward context (``None`` outside the loop)."""
    try:
        from vllm_omni.diffusion.forward_context import get_forward_context

        return get_forward_context().denoise_step_idx
    except Exception:  # pragma: no cover - no forward context (unit tests)
        return None
