# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the FLUX 3 Action tests (tiny structure model, synthetic observation, upstream).

Environment:

* ``FLUX3_ACTION_BASE`` -- local ``black-forest-labs/flux-3-action-base`` copy; only its Qwen3-VL
  tokenizer is used, to build the tiny model's text encoder. Tests that need it skip without it.
* ``FLUX_ACTION_SRC`` -- ``<checkout of black-forest-labs/flux-action>/src``, the upstream reference.
  Parity tests skip without it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def upstream_available() -> bool:
    src = os.environ.get("FLUX_ACTION_SRC", "")
    return bool(src) and os.path.isdir(os.path.join(src, "flux_action"))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 2.0) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * float(np.log10(peak**2 / mse))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from .make_tiny import default_tokenizer_dir, make_tiny

    tok = default_tokenizer_dir()
    if not tok or not os.path.isfile(os.path.join(tok, "tokenizer.json")):
        pytest.skip(
            "set FLUX3_ACTION_BASE to a flux-3-action-base copy (for the Qwen3-VL tokenizer)"
        )
    return make_tiny(str(tmp_path_factory.mktemp("flux3_tiny")))


@pytest.fixture(scope="module")
def observation(tmp_path_factory):
    g = np.random.default_rng(0)
    path = str(tmp_path_factory.mktemp("obs") / "obs.npz")
    cams = {
        k: g.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        for k in ("images.wrist", "images.left", "images.right")
    }
    np.savez(
        path, **cams, state=np.asarray([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.3], dtype=np.float32)
    )
    return path


@pytest.fixture(scope="module")
def upstream():
    if not upstream_available():
        pytest.skip("set FLUX_ACTION_SRC to <flux-action checkout>/src for the upstream reference")
    from .reference import _import_upstream

    return _import_upstream()
