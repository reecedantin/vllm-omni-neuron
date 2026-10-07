# SPDX-License-Identifier: Apache-2.0
"""Host fast path for the upstream ``Gr00tPolicy`` processor (same outputs, less host time).

The serving worker runs the upstream processor on the host for every request: per-frame image
transforms, PIL conversion + chat template, the Qwen3-VL processor (image processor + tokenizer),
then action decoding (per-pose ``scipy`` objects). Measured at one host thread (what the
diffusion worker gives a stage) that is ~18 ms per DROID request. :func:`install` patches the
policy instance -- not the upstream classes -- with equivalent fast paths:

* the eval image transform runs once on the batch of all frames (one resize per stage instead of
  one per frame; the transforms are per-image, so the result is identical);
* the chat-template text is cached per (image count, instruction): it depends on nothing else;
* the tokenizer output is cached per (text, image grids); pixels still go through the upstream
  image processor every request. The first request of every key runs the full upstream
  processor and is compared field by field with the fast path; any mismatch turns the fast path
  off for good;
* relative->absolute action decoding is vectorized over the horizon for the EEF ``XYZ_ROT6D``
  and joint groups (same ``scipy`` calls, batched); other formats use upstream's code.

Each fast path falls back to the upstream method on any input it does not cover.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import OrderedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CACHE_MAX = 64


def host_threads() -> int:
    """Torch threads for the host stages inside the worker (``GR00T_HOST_THREADS``, default 8).

    The diffusion worker pins torch to one thread; the processor's image ops scale to ~8."""
    return int(os.environ.get("GR00T_HOST_THREADS", "8"))


@contextlib.contextmanager
def torch_threads(n: int):
    if n <= 0:
        yield
        return
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


class _LRU(OrderedDict):
    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > _CACHE_MAX:
            self.popitem(last=False)


# ----------------------------------------------------------------------------------------
# images + chat template
# ----------------------------------------------------------------------------------------


def _patch_vlm_inputs(proc) -> None:
    orig_get = proc._get_vlm_inputs
    orig_apply = proc._apply_vlm_processing
    steps = list(getattr(proc.eval_image_transform, "transforms", []))[1:]  # after ToImage
    text_cache = _LRU()

    def apply_vlm_processing(images, language: str):
        """``images``: the transformed frames, ``[N, C, H, W]`` uint8 (tensor or array)."""
        key = (int(images.shape[0]), language)
        text = text_cache.get_or_none(key)
        if text is None:
            out = orig_apply(np.asarray(images), language)
            text_cache.put(key, out["vlm_content"]["text"])
            return out
        # the CHW uint8 frames go to the image processor as tensors: no PIL round trip
        return {"vlm_content": {"text": text, "images": list(torch.as_tensor(images).unbind(0))}}

    def get_vlm_inputs(image_keys, images, masks, image_transform, language):
        frames = [images.get(view) if hasattr(images, "get") else None for view in image_keys]
        ok = (
            masks is None
            and image_transform is proc.eval_image_transform
            and steps
            and all(
                isinstance(f, np.ndarray)
                and f.dtype == np.uint8
                and f.ndim == 4
                and f.shape[-1] == 3
                for f in frames
            )
            and len({f.shape for f in frames}) == 1
        )
        if not ok:
            return orig_get(image_keys, images, masks, image_transform, language)
        # [V, T, H, W, C] -> (T*V) frames in the upstream order (stack over views at dim 1, flatten)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames, axis=1)))  # [T, V, H, W, C]
        t, v = x.shape[:2]
        x = x.reshape(t * v, *x.shape[2:]).permute(0, 3, 1, 2).contiguous()  # ToImage: HWC -> CHW
        for step in steps:
            x = step(x)
        return apply_vlm_processing(x, language)

    proc._apply_vlm_processing = apply_vlm_processing
    proc._get_vlm_inputs = get_vlm_inputs


# ----------------------------------------------------------------------------------------
# collator: tokenizer cache
# ----------------------------------------------------------------------------------------


class FastCollator:
    """Wraps ``Gr00tN1d7DataCollator``: caches the text side of the Qwen3-VL processor."""

    TEXT_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")

    def __init__(self, inner):
        self.inner = inner
        self.processor = inner.processor
        self._text = _LRU()
        self.enabled = True
        qp = self.processor
        merged = qp._merge_kwargs(
            qp.valid_processor_kwargs,
            tokenizer_init_kwargs=qp.tokenizer.init_kwargs,
            return_tensors="pt",
            padding=True,
        )
        self._images_kwargs = dict(merged["images_kwargs"])

    def __getattr__(self, name):  # anything else the policy reads off the collator
        return getattr(self.inner, name)

    def _images(self, images):
        return self.processor.image_processor(images, **self._images_kwargs)

    def __call__(self, features):
        from transformers import BatchFeature

        if not self.enabled or len(features) != 1 or "vlm_content" not in features[0]:
            return self.inner(features)
        vc = features[0]["vlm_content"]
        text, imgs = vc["text"], vc["images"]
        image_inputs = self._images(imgs)
        grids = tuple(image_inputs["image_grid_thw"].reshape(-1).tolist())
        key = (text, grids)
        cached = self._text.get_or_none(key)
        if cached is None:
            full = self.inner(features)
            ref = full["inputs"]
            same = all(
                torch.equal(ref[k], image_inputs[k]) for k in image_inputs.keys() if k in ref
            )
            if not same:
                logger.warning(
                    "GR00T host fast path: image inputs differ from the upstream processor; disabled"
                )
                self.enabled = False
                return full
            self._text.put(key, {k: ref[k] for k in self.TEXT_KEYS if k in ref})
            return full
        batch = {**cached, **dict(image_inputs)}
        for k, v in features[0].items():
            if k != "vlm_content":
                batch[k] = torch.from_numpy(np.stack([v]))
        return BatchFeature(data={"inputs": batch})


# ----------------------------------------------------------------------------------------
# action decode: vectorized relative -> absolute
# ----------------------------------------------------------------------------------------


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Batched ``EndEffectorPose._rot6d_to_matrix``: [N, 6] -> [N, 3, 3]."""
    r = r6.reshape(-1, 2, 3)
    row1 = r[:, 0] / np.linalg.norm(r[:, 0], axis=-1, keepdims=True)
    row2 = r[:, 1] - np.sum(row1 * r[:, 1], axis=-1, keepdims=True) * row1
    row2 = row2 / np.linalg.norm(row2, axis=-1, keepdims=True)
    row3 = np.cross(row1, row2)
    return np.stack([row1, row2, row3], axis=1)


def _homogeneous(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``EndEffectorPose(translation, rotation=matrix).homogeneous`` batched: the rotation goes
    through ``scipy`` ``Rotation.from_matrix(...).as_matrix()`` exactly as upstream stores it."""
    from scipy.spatial.transform import Rotation

    h = np.zeros((xyz.shape[0], 4, 4))
    h[:, :3, :3] = Rotation.from_matrix(rot).as_matrix()
    h[:, :3, 3] = xyz
    h[:, 3, 3] = 1.0
    return h


def eef_rot6d_to_absolute(action: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    """Relative ``XYZ_ROT6D`` chunk [T, 9] + reference pose [9] -> absolute ``XYZ_ROT6D`` [T, 9]."""
    from scipy.spatial.transform import Rotation

    a = np.asarray(action)  # upstream keeps the input dtype through the rot6d Gram-Schmidt
    s = np.asarray(reference_state)[None]
    t_ref = _homogeneous(s[:, :3], _rot6d_to_matrix(s[:, 3:]))[0]
    t_rel = _homogeneous(a[:, :3], _rot6d_to_matrix(a[:, 3:]))
    t_abs = t_ref[None] @ t_rel
    rot = Rotation.from_matrix(t_abs[:, :3, :3]).as_matrix()
    return np.concatenate([t_abs[:, :3, 3], rot[:, :2, :].reshape(-1, 6)], axis=1)


def _patch_decode(sap) -> None:
    from vllm_omni.diffusion.models.gr00t.dataio.types import ActionFormat, ActionType

    orig = sap._convert_to_absolute_action

    def convert(action, reference_state, action_type, action_format):
        if (
            action.ndim == 2
            and reference_state.ndim == 1
            and reference_state.shape[0] == action.shape[1]
        ):
            if action_type == ActionType.NON_EEF and action_format == ActionFormat.DEFAULT:
                return np.asarray(reference_state, dtype=np.float64)[None] + np.asarray(
                    action, dtype=np.float64
                )
            if (
                action_type == ActionType.EEF
                and action_format == ActionFormat.XYZ_ROT6D
                and action.shape[1] == 9
            ):
                return eef_rot6d_to_absolute(action, reference_state)
        return orig(action, reference_state, action_type, action_format)

    sap._convert_to_absolute_action = convert


def install(policy) -> None:
    """Patch one ``Gr00tPolicy`` instance with the fast paths (``GR00T_HOST_FASTPATH=0`` skips)."""
    if os.environ.get("GR00T_HOST_FASTPATH", "1") != "1":
        return
    proc = policy.processor
    _patch_vlm_inputs(proc)
    _patch_decode(proc.state_action_processor)
    policy.collate_fn = FastCollator(policy.collate_fn)
