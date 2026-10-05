# SPDX-License-Identifier: Apache-2.0
"""The plugin's Neuron-safe ``patchify``/``unpatchify`` (adjacent-transpose chains, no 7D permute)
vs diffusers' originals -- bit-exact on CPU, and the reason to prefer the chain on-device: a single
non-adjacent ``permute`` compiles to a ``DramToDramTranspose`` the Trn2 compiler rejects
(``NCC_IDDT901``) for every Wan2.2-5B-VAE encoder (patchify runs inside the chunked encoder graph)."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify as _patchify_torch
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify as _unpatchify_torch

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    patchify,
    unpatchify,
)

_SHAPES = [
    (1, 3, 1, 8, 8),
    (2, 3, 2, 8, 6),
    (1, 4, 1, 16, 16),
    (1, 12, 3, 32, 24),
]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_patchify_matches_diffusers(shape, patch_size):
    b, c, f, h, w = shape
    if h % patch_size or w % patch_size:
        pytest.skip("shape not divisible by this patch_size")
    torch.manual_seed(0)
    x = torch.randn(*shape)
    want = _patchify_torch(x, patch_size)
    got = patchify(x, patch_size)
    torch.testing.assert_close(got, want, rtol=0, atol=0)  # bit-exact: pure reshape/transpose


@pytest.mark.parametrize("c,patch_size", [(3, 1), (3, 2), (4, 2), (12, 4)])
def test_unpatchify_matches_diffusers(c, patch_size):
    torch.manual_seed(1)
    x = torch.randn(2, c * patch_size * patch_size, 2, 5, 7)
    want = _unpatchify_torch(x, patch_size)
    got = unpatchify(x, patch_size)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_patchify_unpatchify_roundtrip(patch_size):
    torch.manual_seed(2)
    x = torch.randn(1, 3, 2, 16, 16)
    y = patchify(x, patch_size)
    z = unpatchify(y, patch_size)
    torch.testing.assert_close(z, x, rtol=0, atol=0)


def test_patchify_rejects_bad_shape():
    with pytest.raises(ValueError, match="Invalid input shape"):
        patchify(torch.randn(3, 8, 8), 2)
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.randn(1, 3, 1, 7, 8), 2)


def test_unpatchify_rejects_bad_shape():
    with pytest.raises(ValueError, match="Invalid input shape"):
        unpatchify(torch.randn(3, 8, 8), 2)


def test_patchify_identity_at_patch_size_one_is_the_same_object():
    x = torch.randn(1, 3, 1, 4, 4)
    assert patchify(x, 1) is x
    assert unpatchify(x, 1) is x


def test_patchify_builds_only_adjacent_transposes():
    """Guard against a future edit silently reintroducing a wide permute: every transpose call in
    patchify/unpatchify must swap two ADJACENT axes (|i - j| == 1), the property that avoids the
    DramToDramTranspose compile failure. Patches Tensor.transpose for the duration of one call."""
    seen = []
    orig = torch.Tensor.transpose

    def spy(self, i, j):
        seen.append((i % self.dim(), j % self.dim()))
        return orig(self, i, j)

    torch.Tensor.transpose = spy
    try:
        patchify(torch.randn(1, 3, 1, 8, 8), 2)
        unpatchify(torch.randn(1, 12, 1, 4, 4), 2)
    finally:
        torch.Tensor.transpose = orig
    assert seen, "no transpose calls observed"
    assert all(abs(i - j) == 1 for i, j in seen), seen


@pytest.mark.parametrize("patch_size", [2, 4])
def test_patchify_compiles_fullgraph(patch_size):
    """A graph-break-free trace is the closest CPU proxy for 'the compiler can lower this as one
    graph' available without a device; the real pass/fail is the device smoke."""
    torch._dynamo.reset()
    fn = torch.compile(patchify, backend="eager", fullgraph=True, dynamic=False)
    x = torch.randn(1, 3, 1, 8, 8)
    torch.testing.assert_close(fn(x, patch_size), patchify(x, patch_size))


@pytest.mark.parametrize("patch_size", [2, 4])
def test_unpatchify_compiles_fullgraph(patch_size):
    torch._dynamo.reset()
    fn = torch.compile(unpatchify, backend="eager", fullgraph=True, dynamic=False)
    x = torch.randn(1, 48, 1, 4, 4)
    torch.testing.assert_close(fn(x, patch_size), unpatchify(x, patch_size))


# ------------------------------------------------------------------------------------------------
# AvgDown3D / DupUp3D: the second DramToDramTranspose (NCC_IDDT901) found inside the Wan VAE
# encoder on Trn2, after the patchify fix above. Same class of bug (a non-adjacent-axis permute), same fix (an
# adjacent-transpose chain), applied via a module-level monkeypatch of AvgDown3D.forward /
# DupUp3D.forward (both classes are instantiated directly from diffusers by this module, as
# attributes, not subclassed) so every existing instance gets the fixed forward.


def _avgdown_reference_forward(mod, x):
    """diffusers' original AvgDown3D.forward (the permute version), captured before this module's
    import-time monkeypatch replaces it, so the comparison is against the un-patched behaviour."""
    pad_t = (mod.factor_t - x.shape[2] % mod.factor_t) % mod.factor_t
    x = torch.nn.functional.pad(x, (0, 0, 0, 0, pad_t, 0))
    b, c, t, h, w = x.shape
    x = x.view(
        b,
        c,
        t // mod.factor_t,
        mod.factor_t,
        h // mod.factor_s,
        mod.factor_s,
        w // mod.factor_s,
        mod.factor_s,
    )
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(b, c * mod.factor, t // mod.factor_t, h // mod.factor_s, w // mod.factor_s)
    x = x.view(
        b, mod.out_channels, mod.group_size, t // mod.factor_t, h // mod.factor_s, w // mod.factor_s
    )
    return x.mean(dim=2)


def _dupup_reference_forward(mod, x, first_chunk=False):
    x = x.repeat_interleave(mod.repeats, dim=1)
    x = x.view(
        x.size(0),
        mod.out_channels,
        mod.factor_t,
        mod.factor_s,
        mod.factor_s,
        x.size(2),
        x.size(3),
        x.size(4),
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(
        x.size(0),
        mod.out_channels,
        x.size(2) * mod.factor_t,
        x.size(4) * mod.factor_s,
        x.size(6) * mod.factor_s,
    )
    if first_chunk:
        x = x[:, :, mod.factor_t - 1 :, :, :]
    return x


@pytest.mark.parametrize(
    "in_c,out_c,factor_t,factor_s", [(16, 32, 2, 2), (96, 96, 1, 2), (16, 16, 1, 1)]
)
def test_avg_down_3d_matches_reference(in_c, out_c, factor_t, factor_s):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    torch.manual_seed(0)
    mod = AvgDown3D(in_c, out_c, factor_t=factor_t, factor_s=factor_s)
    x = torch.randn(2, in_c, 5, 8, 8)
    want = _avgdown_reference_forward(mod, x)
    got = mod(x)  # the module-level monkeypatch is already applied
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("in_c,out_c,factor_t,factor_s", [(32, 16, 2, 2), (96, 96, 1, 2)])
def test_dup_up_3d_matches_reference(in_c, out_c, factor_t, factor_s):
    from diffusers.models.autoencoders.autoencoder_kl_wan import DupUp3D

    torch.manual_seed(1)
    mod = DupUp3D(in_c, out_c, factor_t=factor_t, factor_s=factor_s)
    x = torch.randn(2, in_c, 3, 4, 4)
    for first_chunk in (False, True):
        want = _dupup_reference_forward(mod, x, first_chunk=first_chunk)
        got = mod(x, first_chunk=first_chunk)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_avg_down_3d_builds_only_adjacent_transposes():
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    seen = []
    orig = torch.Tensor.transpose

    def spy(self, i, j):
        seen.append((i % self.dim(), j % self.dim()))
        return orig(self, i, j)

    torch.Tensor.transpose = spy
    try:
        AvgDown3D(16, 32, factor_t=2, factor_s=2)(torch.randn(1, 16, 5, 8, 8))
    finally:
        torch.Tensor.transpose = orig
    assert seen, "no transpose calls observed"
    assert all(abs(i - j) == 1 for i, j in seen), seen


def test_dup_up_3d_builds_only_adjacent_transposes():
    from diffusers.models.autoencoders.autoencoder_kl_wan import DupUp3D

    seen = []
    orig = torch.Tensor.transpose

    def spy(self, i, j):
        seen.append((i % self.dim(), j % self.dim()))
        return orig(self, i, j)

    torch.Tensor.transpose = spy
    try:
        DupUp3D(32, 16, factor_t=2, factor_s=2)(torch.randn(1, 32, 3, 4, 4))
    finally:
        torch.Tensor.transpose = orig
    assert seen, "no transpose calls observed"
    assert all(abs(i - j) == 1 for i, j in seen), seen


@pytest.mark.parametrize("factor_t,factor_s", [(2, 2), (1, 2)])
def test_avg_down_3d_compiles_fullgraph(factor_t, factor_s):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    torch._dynamo.reset()
    mod = AvgDown3D(16, 32, factor_t=factor_t, factor_s=factor_s)
    fn = torch.compile(mod, backend="eager", fullgraph=True, dynamic=False)
    x = torch.randn(1, 16, 5, 8, 8)
    torch.testing.assert_close(fn(x), mod(x))
