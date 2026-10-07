# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the Wan2.2 context-parallel self-attention dispatch and of why it avoids the
const-max ring kernel by default.

The ring kernel (``ring_attention_const_max_fwd``) shifts each row's scores by a Cauchy-Schwarz
bound ``scale * ||q_i|| * max_j ||k_j||`` instead of the row's true maximum, stores the shift in
fp16 and the probabilities in bf16, and floors the row sum at 1e-37 before the reciprocal. Once
the bound overshoots the true maximum by more than ~85-87 nats every probability of the row
underflows and the row comes back as zeros. ``_constmax_reference`` models those numerics on CPU;
the tests pin the hazard on a synthetic row and check that the default CP path (all-gather + flash
attention with the true maximum) is exact on that same row.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("PJRT_DEVICE", "CPU")

import vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer as W  # noqa: E402

_SUM_CLAMP = 1e-37  # the ring kernel's floor on the row sum
_BF16_TINY = 2.0**-126


def _bf16(x: torch.Tensor) -> torch.Tensor:
    y = x.to(torch.bfloat16).float()
    return torch.where(y.abs() < _BF16_TINY, torch.zeros_like(y), y)


def _constmax_reference(q, k, v, scale, *, shift_dtype=torch.float16):
    """[B, N, S, D] attention with the ring kernel's softmax numerics (trn2)."""
    q, k, v = q.float(), k.float(), _bf16(v.float())
    s = torch.matmul(q * scale, k.transpose(-2, -1))
    bound = _bf16(scale * q.norm(dim=-1, keepdim=True) * k.norm(dim=-1).amax(-1)[..., None, None])
    p = _bf16(torch.exp((s - bound).to(shift_dtype).float()))
    return torch.matmul(p, v) / p.sum(-1, keepdim=True).clamp_min(_SUM_CLAMP)


def _exact(q, k, v, scale):
    return torch.softmax(torch.matmul(q * scale, k.transpose(-2, -1)), -1) @ v


def _gap_inputs(gap: float, seq: int = 64, d: int = 16, seed: int = 0):
    """q/k/v ``[1, 1, seq, D]`` whose CS bound overshoots every row's true max by ~``gap`` nats:
    one large-norm key orthogonal to every query sets max ||k||."""
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(1, 1, seq, d)
    q[..., 0] = 1.0 + 0.1 * torch.rand(seq, generator=g)
    k = torch.zeros(1, 1, seq, d)
    k[..., 0] = torch.randn(seq, generator=g) * 2.0
    k[..., 2:] = torch.randn(seq, d - 2, generator=g) * 0.01
    k[0, 0, -1] = 0.0
    k[0, 0, -1, 1] = gap + float(k[..., 0].abs().max()) * 1.1
    v = torch.randn(1, 1, seq, d, generator=g)
    return q, k, v


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_constmax_numerics_are_accurate_for_a_small_gap():
    q, k, v = _gap_inputs(1.0)
    assert _rel(_constmax_reference(q, k, v, 1.0), _exact(q, k, v, 1.0)) < 0.01


def test_constmax_numerics_zero_rows_past_the_underflow_gap():
    q, k, v = _gap_inputs(110.0)
    out = _constmax_reference(q, k, v, 1.0)
    assert torch.isfinite(out).all()
    assert float(out.abs().amax()) == 0.0  # every row silently zeroed


class _FakeCPGroup:
    """Stands in for the CP group: each all_gather returns the next full tensor (K, then V)."""

    def __init__(self, *full):
        self.full = list(full)

    def all_gather(self, tensor, dim):
        assert dim == 2
        return self.full.pop(0)


def _shard_call(q, k, v, scale, cp, monkeypatch):
    """wan_cp_self_attention on the first of ``cp`` sequence shards, token-major in, d-major out."""
    tok = lambda x: x.transpose(1, 2).contiguous()  # noqa: E731  [B, N, S, D] -> [B, S, N, D]
    local = q.shape[2] // cp
    group = _FakeCPGroup(k.contiguous(), v.contiguous())
    monkeypatch.setattr(W, "_can_use_wan_attention_kernel", lambda *a: False)  # torch flash path
    out = W.wan_cp_self_attention(
        tok(q[:, :, :local]),
        tok(k[:, :, :local]),
        tok(v[:, :, :local]),
        scale,
        cp,
        group,
        ((0, 1),),
    )
    return out.transpose(2, 3)  # -> [B, N, local, D]


def test_default_cp_path_uses_the_true_max_and_skips_the_ring_kernel(monkeypatch):
    monkeypatch.delenv("WAN22_CP_RING_ATTENTION", raising=False)
    monkeypatch.setattr(W, "can_run_kernel", lambda t: True)

    def ring(*a, **kw):
        raise AssertionError("ring kernel used without WAN22_CP_RING_ATTENTION")

    monkeypatch.setattr(W, "_nf_ring_attend", ring)
    q, k, v = _gap_inputs(110.0)
    got = _shard_call(q, k, v, 1.0, 2, monkeypatch)
    want = _exact(q, k, v, 1.0)[:, :, : q.shape[2] // 2]
    assert _rel(got, want) < 1e-5
    assert float(got.abs().amax()) > 0.1  # the rows the ring kernel would zero


def test_ring_kernel_is_opt_in(monkeypatch):
    monkeypatch.setenv("WAN22_CP_RING_ATTENTION", "1")
    monkeypatch.setattr(W, "can_run_kernel", lambda t: True)
    sentinel = torch.zeros(1, 32, 1, 16)

    def ring(query, key, value, **kw):
        assert kw["num_workers"] == 2
        return sentinel

    monkeypatch.setattr(W, "_nf_ring_attend", ring)
    q, k, v = _gap_inputs(1.0)
    out = _shard_call(q, k, v, 1.0, 2, monkeypatch)
    assert out.shape == (1, 32, 1, 16) and float(out.abs().max()) == 0.0


@pytest.mark.parametrize("value", ["", "0"])
def test_ring_kernel_flag_off_values(monkeypatch, value):
    monkeypatch.setenv("WAN22_CP_RING_ATTENTION", value)
    assert not W.cp_ring_attention_enabled()
