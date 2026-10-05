# SPDX-License-Identifier: Apache-2.0
"""Neighborhood attention (NATTEN ``na2d`` / ``na3d`` semantics) without NATTEN.

The FLUX Action video VAE attends over a ``kernel``-sized window around every token. NATTEN's
definition (stride 1, dilation 1, odd kernel), per axis of length ``L``:

* non-causal: query ``i`` sees keys ``[s(i), s(i) + k)`` with ``s(i) = clamp(i - k // 2, 0, L - k)``
  (the window is shifted inward at the borders, so every query sees exactly ``k`` keys);
* causal: query ``i`` sees keys ``[max(0, i - k + 1), i]``.

The multi-axis neighborhood is the product of the per-axis ones; the softmax scale is
``head_dim ** -0.5``. Layout (NATTEN's "heads last"): ``[B, *spatial, heads, head_dim]``.

Two implementations:

* :func:`neighborhood_attention` gathers each query's ``prod(kernel)`` keys with static index
  tables (``index_select``; no scatter, no ``[N, N]`` mask) and runs a small softmax over the
  window. This is the device path; its memory is ``prod(kernel)`` x K/V, not ``N^2``.
* :func:`neighborhood_attention_reference` is a literal dense masked attention built from the
  definition above, used to check the gather path and as a CPU stand-in for NATTEN when running
  the upstream reference. It is O(N^2) and only meant for small inputs or CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _axis_window(length: int, kernel: int, causal: bool) -> tuple[Tensor, Tensor | None]:
    """``[L, k]`` key indices per query and, for causal axes, a ``[L, k]`` validity mask."""
    i = torch.arange(length)[:, None]
    j = torch.arange(kernel)[None, :]
    if causal:
        idx = i - (kernel - 1) + j
        valid = idx >= 0
        return idx.clamp(min=0), valid
    if length < kernel:
        raise ValueError(f"neighborhood kernel {kernel} exceeds the axis length {length}")
    start = (i - kernel // 2).clamp(0, length - kernel)
    return start + j, None


def neighborhood_attention(
    q: Tensor, k: Tensor, v: Tensor, kernel: Sequence[int], causal: Sequence[bool] | None = None
) -> Tensor:
    """``q/k/v [B, *axes, heads, D]`` -> same shape; NATTEN semantics (see module docstring)."""
    n_ax = q.ndim - 3
    kernel = list(kernel)
    causal = list(causal) if causal is not None else [False] * n_ax
    assert len(kernel) == n_ax == len(causal), "kernel/causal rank mismatch"
    axes = q.shape[1 : 1 + n_ax]
    kk, vv = k, v
    mask = None  # broadcastable to [*axes, *kernel]
    # Gather the window along each axis in turn: after axis a, K has shape
    # [B, ax_0, k_0, ..., ax_a, k_a, ax_{a+1}, ..., heads, D].
    for a in range(n_ax):
        idx, valid = _axis_window(axes[a], kernel[a], causal[a])
        dim = 1 + 2 * a  # position of this axis in the partially gathered tensor
        idx = idx.to(q.device)
        shp = list(kk.shape)
        kk = kk.index_select(dim, idx.reshape(-1)).reshape(
            *shp[:dim], axes[a], kernel[a], *shp[dim + 1 :]
        )
        vv = vv.index_select(dim, idx.reshape(-1)).reshape(
            *shp[:dim], axes[a], kernel[a], *shp[dim + 1 :]
        )
        if valid is not None:
            view = [1] * (2 * n_ax)
            view[2 * a], view[2 * a + 1] = axes[a], kernel[a]
            m = valid.reshape(view).to(q.device)
            mask = m if mask is None else mask & m
    # kk: [B, ax0, k0, ax1, k1, ..., heads, D] -> [B, *axes, heads, prod(k), D]
    b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
    perm = (
        [0]
        + [1 + 2 * a for a in range(n_ax)]
        + [1 + 2 * n_ax]
        + [2 + 2 * a for a in range(n_ax)]
        + [2 + 2 * n_ax]
    )
    win = 1
    for x in kernel:
        win *= x
    kk = kk.permute(perm).reshape(b, *axes, heads, win, d)
    vv = vv.permute(perm).reshape(b, *axes, heads, win, d)
    scores = torch.einsum("b...hd,b...hwd->b...hw", q.float(), kk.float()) * (d**-0.5)
    if mask is not None:
        mperm = [2 * a for a in range(n_ax)] + [2 * a + 1 for a in range(n_ax)]
        m = mask.expand(*[s for a in range(n_ax) for s in (axes[a], kernel[a])]).permute(mperm)
        m = m.reshape(*axes, 1, win)
        scores = scores.masked_fill(~m, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("b...hw,b...hwd->b...hd", probs, vv.float())
    return out.to(q.dtype)


def neighborhood_mask(axes: Sequence[int], kernel: Sequence[int], causal: Sequence[bool]) -> Tensor:
    """Dense ``[N, N]`` bool mask (query x key) of the neighborhood, from NATTEN's mask definition."""
    coords = torch.stack(
        torch.meshgrid(*[torch.arange(n) for n in axes], indexing="ij"), dim=-1
    ).reshape(-1, len(axes))
    qc, kc = coords[:, None, :], coords[None, :, :]
    allowed = torch.ones(qc.shape[0], kc.shape[1], dtype=torch.bool)
    for a, (n, kn, c) in enumerate(zip(axes, kernel, causal, strict=True)):
        qa, ka = qc[..., a], kc[..., a]
        if c:
            ok = (qa - ka >= 0) & (qa - ka < kn)
        else:
            left, right = kn // 2, kn // 2 + (kn % 2 - 1)
            center = qa.clamp(left, n - 1 - right)
            ok = ((center - ka >= 0) & (center - ka <= left)) | (
                (ka - center >= 0) & (ka - center <= right)
            )
        allowed &= ok
    return allowed


def neighborhood_attention_reference(
    q: Tensor, k: Tensor, v: Tensor, kernel: Sequence[int], causal: Sequence[bool] | None = None
) -> Tensor:
    """Dense masked attention, literal from the definition (O(N^2); CPU / small inputs)."""
    n_ax = q.ndim - 3
    causal = list(causal) if causal is not None else [False] * n_ax
    axes = q.shape[1 : 1 + n_ax]
    b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
    mask = neighborhood_mask(axes, kernel, causal).to(q.device)
    qf, kf, vf = (t.reshape(b, -1, heads, d).transpose(1, 2).float() for t in (q, k, v))
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * (d**-0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), vf)
    return out.transpose(1, 2).reshape(q.shape).to(q.dtype)
