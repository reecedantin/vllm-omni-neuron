# SPDX-License-Identifier: Apache-2.0
"""DROID observation I/O for FLUX 3 Action.

An observation file is an NPZ with three synchronized HWC uint8 RGB camera frames of 360x640
(``images.wrist``, ``images.left``, ``images.right``) and ``state`` (8 float32: seven joint positions in
radians and the gripper closed fraction). An optional JSON sidecar with the same stem carries the
``task`` instruction. This is the input format of the upstream ``flux-action infer`` command.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

CAMERA_KEYS = ("images.wrist", "images.left", "images.right")
DEFAULT_TASK = "put the screwdriver in the box"


def load_observation(path: str, task: str | None = None) -> dict:
    """NPZ -> policy batch: cameras ``(1, 3, 360, 640)`` uint8, ``state (1, 8)``, ``task [str]``."""
    z = np.load(path)
    batch = {k: torch.from_numpy(z[k]).permute(2, 0, 1)[None].contiguous() for k in CAMERA_KEYS}
    batch["state"] = torch.from_numpy(z["state"].astype(np.float32))[None]
    if task is None:
        side = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(side):
            with open(side) as f:
                task = json.load(f).get("task")
    batch["task"] = [task or DEFAULT_TASK]
    return batch


def encode_camera(frame: np.ndarray) -> dict:
    """HWC uint8 camera frame -> the compact request form ``{"data": bytes, "shape", "dtype"}``.

    One raw byte string per camera instead of a nested Python list (~0.7 M ints each): the request
    is serialized between processes on every hop, and a list costs ~0.1-0.4 s to build and copy."""
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    return {"data": frame.tobytes(), "shape": list(frame.shape), "dtype": "uint8"}


def observation_extra_args(path: str, task: str | None = None) -> dict:
    """NPZ -> the ``extra_args`` of an Omni request served by ``Flux3ActionPipeline``.

    Cameras travel as raw bytes (:func:`encode_camera`); nested lists of uint8 are also accepted.
    """
    batch = load_observation(path, task)
    z = np.load(path)
    extra = {"task": batch["task"][0], "state": z["state"].astype(np.float32).tolist()}
    for key in CAMERA_KEYS:
        extra[key] = encode_camera(z[key])
    return extra
