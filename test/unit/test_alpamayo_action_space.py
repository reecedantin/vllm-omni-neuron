# SPDX-License-Identifier: Apache-2.0
"""Verify the vendored (_vendor/, dependency-trimmed) unicycle action-space math is bit-exact
against upstream alpamayo1_5 -- the einops->matmul rewrite (see _vendor/NOTICE) must not change
any number. Needs the upstream ``alpamayo1_5`` package for the upstream comparison; the vendored module
itself is tested standalone too, which runs in the shared venv (no hydra/einops needed there)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "vllm_omni_neuron",
        "diffusion",
        "models",
        "alpamayo",
    ),
)
_vendor = importlib.import_module("_vendor.unicycle_accel_curvature")

KW = dict(
    dt=0.1,
    accel_mean=0.029,
    accel_std=0.681,
    curvature_mean=0.00027,
    curvature_std=0.0261,
    accel_bounds=(-9.8, 9.8),
    curvature_bounds=(-0.33, 0.33),
    a_lambda=1e-4,
    a_ridge=1e-4,
    kappa_lambda=1e-4,
    kappa_ridge=1e-4,
    theta_lambda=1e-6,
    theta_ridge=1e-8,
    v_lambda=1e-6,
    v_ridge=1e-4,
)


def _synthetic(seed=0, B=2, T=16, N=64):
    g = torch.Generator().manual_seed(seed)
    hist_xyz = torch.randn(B, T, 3, generator=g) * 2
    hist_rot = torch.eye(3).expand(B, T, 3, 3).clone()
    action = torch.randn(B, N, 2, generator=g) * 0.3
    fut_xyz = torch.randn(B, N, 3, generator=g) * 2
    fut_rot = torch.eye(3).expand(B, N, 3, 3).clone()
    return hist_xyz, hist_rot, action, fut_xyz, fut_rot


def test_vendored_module_runs_standalone():
    """No einops/hydra/scipy needed to import and use the vendored module."""
    sp = _vendor.UnicycleAccelCurvatureActionSpace(**KW)
    hist_xyz, hist_rot, action, _, _ = _synthetic()
    xyz, rot = sp.action_to_traj(action, hist_xyz, hist_rot)
    assert xyz.shape == (2, action.shape[1], 3)
    assert rot.shape == (2, action.shape[1], 3, 3)
    assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()


@pytest.mark.skipif(
    importlib.util.find_spec("alpamayo1_5") is None,
    reason="needs the upstream alpamayo1_5 package",
)
def test_vendored_matches_upstream_exactly():
    from alpamayo1_5.action_space.unicycle_accel_curvature import (
        UnicycleAccelCurvatureActionSpace as Upstream,
    )

    up = Upstream(**KW)
    vd = _vendor.UnicycleAccelCurvatureActionSpace(**KW)
    hist_xyz, hist_rot, action, fut_xyz, fut_rot = _synthetic()

    xyz_u, rot_u = up.action_to_traj(action, hist_xyz, hist_rot)
    xyz_v, rot_v = vd.action_to_traj(action, hist_xyz, hist_rot)
    assert torch.equal(xyz_u, xyz_v) and torch.equal(rot_u, rot_v)

    act_u = up.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
    act_v = vd.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
    assert torch.equal(act_u, act_v)
