# SPDX-License-Identifier: Apache-2.0
"""Request-side helpers shared by the pi0 family pipelines: the compact camera wire form, the
host thread budget for preprocessing inside the worker, and the per-request stage timing.

Cameras may travel as raw bytes, ``{"data": bytes, "shape": [H, W, 3], "dtype": "uint8"}``
(:func:`encode_camera`), instead of ndarrays or nested lists: the request is serialized on every
hop between the client and the diffusion worker, and a raw byte string is the cheapest form to
copy. Decoded frames are bit-identical to the ndarray form.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Intra-op torch threads for host preprocessing inside the worker (image normalization, camera
# stack, state). 0 (default) leaves the worker's setting alone: at 224 px preprocessing is ~1 ms
# and 8 threads measured no gain.
DEFAULT_HOST_THREADS = int(os.environ.get("PI0_HOST_THREADS", "0"))
# When set, each request appends one JSON line with its stage breakdown here (benchmarks).
STATS_FILE = os.environ.get("PI0_STATS_FILE", "")


def encode_camera(frame) -> dict:
    """HWC uint8 frame -> the raw-bytes request form ``{"data", "shape", "dtype"}``."""
    frame = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
    return {"data": frame.tobytes(), "shape": list(frame.shape), "dtype": "uint8"}


def _decode_value(v):
    if isinstance(v, dict) and "data" in v and "shape" in v:
        dtype = np.dtype(v.get("dtype", "uint8"))
        return np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"]).copy()  # writable
    return v


def decode_robot_obs(robot_obs: dict) -> dict:
    """Decode raw-bytes entries (cameras, and optionally the state) in a request's
    ``robot_obs``; other values pass through unchanged."""
    out = {k: _decode_value(v) for k, v in robot_obs.items()}
    images = out.get("images")
    if isinstance(images, dict):
        out["images"] = {k: _decode_value(v) for k, v in images.items()}
    return out


@contextlib.contextmanager
def torch_threads(n: int):
    """Run the block with ``n`` intra-op torch threads (restored afterwards); ``n <= 0``: no-op."""
    if n <= 0:
        yield
        return
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


class RequestTimer:
    """Wall-clock stage breakdown of one request (ms). ``client_send_ts`` (``time.time()`` on
    the client, in ``extra_args``) adds the client -> worker transport time."""

    def __init__(self, extra: dict):
        self.t0 = time.perf_counter()
        self.ms: dict[str, float] = {}
        sent = extra.get("client_send_ts")
        if sent is not None:
            self.ms["client_to_worker"] = round(1e3 * (time.time() - float(sent)), 3)
        self._last = self.t0

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.ms[name] = round(self.ms.get(name, 0.0) + 1e3 * (now - self._last), 3)
        self._last = now

    def finish(self, actions=None, **extra) -> dict:
        """``actions``: when given and ``PI0_STATS_FILE`` is set, the record carries a digest of
        the action chunk, so every tensor-parallel rank's output can be compared."""
        self.ms["forward"] = round(1e3 * (time.perf_counter() - self.t0), 3)
        rec = {**self.ms, **extra}
        if STATS_FILE and actions is not None:
            arr = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
            rec["actions_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        logger.debug("pi0 request timing (ms): %s", rec)
        if STATS_FILE:
            with open(STATS_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec


def tp_state() -> tuple[int, int, object]:
    """``(tp_size, tp_rank, tp_group)`` of the stage's vLLM tensor-parallel group, ``(1, 0, None)``
    outside one. Under TP>1 the group's partition is registered with the Neuron compiler's mesh
    registry so the in-graph all-reduces legalize (``register_replica_groups``)."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        size = get_tensor_model_parallel_world_size()
        rank, group = get_tensor_model_parallel_rank(), get_tp_group().device_group
    except (AssertionError, ImportError, AttributeError):
        return 1, 0, None
    if size > 1:
        from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups

        register_replica_groups(tp_size=size, cp_size=1)
    return size, rank, group
