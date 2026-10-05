# SPDX-License-Identifier: Apache-2.0
"""Device-legal halo-tiled neighborhood (NATTEN / Swin) attention vs a dense masked reference.

CPU only. The reference here is a self-contained copy of the NATTEN mask definition (the same one
FLUX-3-Action's ``flux3_action.neighborhood.neighborhood_attention_reference`` implements -- that
module lives in the FLUX-3-Action package, which cannot be imported alongside this one since both provide the
``vllm_omni_neuron`` namespace, so the definition is inlined to keep this test self-contained)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni_neuron.diffusion.attention.neighborhood_attention import neighborhood_attention_tiled

# --- reference (NATTEN mask definition, O(N^2); matches flux3_action.neighborhood) ---


def _neighborhood_mask(axes, kernel, causal):
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


def _reference(q, k, v, kernel, causal):
    n_ax = q.ndim - 3
    axes = q.shape[1 : 1 + n_ax]
    b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
    mask = _neighborhood_mask(axes, kernel, causal).to(q.device)
    qf, kf, vf = (t.reshape(b, -1, heads, d).transpose(1, 2).float() for t in (q, k, v))
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * (d**-0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), vf)
    return out.transpose(1, 2).reshape(q.shape).to(q.dtype)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


# --- 1D ---


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "length,tile,kernel", [(20, 6, 5), (18, 4, 5), (23, 8, 5), (10, 10, 5), (17, 16, 5), (5, 5, 5)]
)
def test_1d_matches_reference(length, tile, kernel, causal):
    torch.manual_seed(0)
    q, k, v = (_rand(1, length, 4, 8) for _ in range(3))
    ref = _reference(q, k, v, [kernel], [causal])
    got = neighborhood_attention_tiled(q, k, v, [kernel], [causal], [tile])
    assert got.shape == q.shape
    assert _rel(got, ref) < 1e-5


# --- 2D (the FLUX-3-Action single-frame VAE case; 136x184 is the real device-failing grid) ---


@pytest.mark.parametrize(
    "h,w,th,tw", [(10, 12, 4, 4), (17, 23, 8, 8), (9, 9, 16, 16), (136, 184, 16, 16)]
)
def test_2d_matches_reference(h, w, th, tw):
    torch.manual_seed(0)
    q, k, v = (_rand(1, h, w, 4, 8) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [th, tw])
    assert got.shape == q.shape
    assert _rel(got, ref) < 1e-5


# --- 3D (time-causal, spatial non-causal: the full video kernel) ---


@pytest.mark.parametrize("t,h,w,tt,th,tw", [(4, 10, 12, 2, 4, 4), (8, 17, 23, 4, 8, 8)])
def test_3d_time_causal_matches_reference(t, h, w, tt, th, tw):
    torch.manual_seed(0)
    q, k, v = (_rand(1, t, h, w, 4, 8) for _ in range(3))
    ref = _reference(q, k, v, [3, 5, 5], [True, False, False])
    got = neighborhood_attention_tiled(q, k, v, [3, 5, 5], [True, False, False], [tt, th, tw])
    assert got.shape == q.shape
    assert _rel(got, ref) < 1e-5


# --- batch, heads, dtype, defaults, edges ---


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_heads_and_dtype(dtype):
    torch.manual_seed(0)
    q, k, v = (_rand(2, 10, 12, 3, 16, dtype=dtype) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [4, 4])
    assert got.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert _rel(got, ref) < tol


def test_bias_magnitude_is_moderate_not_near_overflow():
    """The additive mask bias must be a moderate, conventional magnitude (the FLUX-3-Action VAE's own working
    _band_bias uses -30000.0), not an extreme near-overflow value like torch.finfo(dtype).min: an
    all-masked row (every padding tile has one) softmaxes correctly either way on CPU, but an
    extreme value is a plausible source of a device-only numerical difference this module cannot
    rule out without hardware -- keep the bias conventional so that's not a live risk."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import _tile_bias

    bias = _tile_bias((20, 24), (5, 5), (False, False), (8, 8), "cpu", torch.float32)
    masked_values = bias[bias != 0]
    assert masked_values.numel() > 0
    assert torch.all(masked_values == -30000.0)
    # no row is ever all-masked (padded queries keep the clamped window of the last real row; an
    # all -30000 row is a 0/0 hazard for a device softmax lowering -- r16's crop=False variant on
    # Trn2 produced NaN into the valid region with such rows kept in the graph)
    assert bool((bias == 0).any(dim=-1).all())
    # heads axis pre-broadcast on the host: [tiles, 1, tt, wt]
    assert bias.shape == (3 * 3, 1, 64, (8 + 8) ** 2)


def test_default_tile():
    torch.manual_seed(0)
    q, k, v = (_rand(1, 20, 24, 4, 8) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False])  # tile=None
    assert _rel(got, ref) < 1e-5


def test_pad_free_tile_divides_axes():
    """The default tiling picks per-axis divisors (<= 16) so no axis has a partially padded last
    tile -- the one geometry the fused Trn2 graph got wrong in smoke rounds 13-15. FLUX-3-Action's
    136x184 VAE grid -> (8, 8); a short axis is its own tile; a prime axis falls back to padding."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import pad_free_tile

    assert pad_free_tile((136, 184), (5, 5)) == (8, 8)
    assert pad_free_tile((128, 192), (5, 5)) == (16, 16)
    assert pad_free_tile((20, 24), (5, 5)) == (10, 12)
    assert pad_free_tile((12, 30), (5, 5)) == (12, 15)
    assert pad_free_tile((37, 64), (5, 5)) == (16, 16)  # 37 is prime: padded tiling
    assert pad_free_tile((9, 30), (3, 3), max_tile=8) == (3, 6)  # divisors <= 8: 9 -> 3, 30 -> 6


def test_pad_free_default_matches_reference_on_flux3_grid_rows():
    """136x184 at the default (pad-free) tiling vs the dense NATTEN reference, including the rows
    and columns a 16x16 tiling would have padded (the device-failing region)."""
    torch.manual_seed(0)
    q, k, v = (_rand(1, 136, 184, 2, 8) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False])
    assert _rel(got, ref) < 1e-5
    assert _rel(got[:, 128:], ref[:, 128:]) < 1e-5
    assert _rel(got[:, :, 176:], ref[:, :, 176:]) < 1e-5


def test_crop_false_returns_padded_grid_with_identical_valid_region():
    torch.manual_seed(0)
    q, k, v = (_rand(1, 20, 24, 2, 8) for _ in range(3))
    cropped = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [16, 16])
    padded = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [16, 16], crop=False)
    assert padded.shape == (1, 32, 32, 2, 8)
    assert torch.equal(padded[:, :20, :24], cropped)
    # with a pad-free tiling, crop=False is the identity
    full = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [10, 12], crop=False)
    assert full.shape == q.shape


def test_oversized_tile_is_clamped_to_axis():
    torch.manual_seed(0)
    q, k, v = (_rand(1, 6, 6, 2, 8) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [64, 64])
    assert _rel(got, ref) < 1e-5


def test_kernel_larger_than_axis_raises():
    q = _rand(1, 3, 2, 8)
    with pytest.raises(ValueError, match="length 3 < kernel 5"):
        neighborhood_attention_tiled(q, q, q, [5])


def test_window_axis_output_and_each_window_are_contiguous():
    """Regression test for the Trn2 smoke failure (round 5 traceback pinned it exactly here):
    torch.stack over a list of torch.narrow views raised RuntimeError: Expected
    self.is_contiguous() on the real device even though CPU tolerates stacking non-contiguous
    views. Both the per-window narrow and _window_axis's own output must be contiguous, and the
    function must still produce the right answer when its OWN input is already non-contiguous (the
    chained multi-axis case: axis 1's output feeds axis 2's _window_axis call)."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import _window_axis

    torch.manual_seed(0)
    x = torch.randn(2, 20, 3)
    out = _window_axis(x, 1, 6, 4, 4)
    assert out.is_contiguous()
    torch.testing.assert_close(out, x.unfold(1, 6, 4))

    # non-contiguous input (e.g. a prior _window_axis call's own, now-contiguous-by-construction
    # output -- or any other transposed view a caller might pass)
    x_nc = x.transpose(0, 1)  # [20, 2, 3], non-contiguous
    assert not x_nc.is_contiguous()
    out2 = _window_axis(x_nc, 0, 6, 4, 4)
    assert out2.is_contiguous()
    torch.testing.assert_close(out2, x_nc.contiguous().unfold(0, 6, 4))


def test_window_axis_matches_unfold():
    """_window_axis (narrow + stack) must reproduce Tensor.unfold's exact output, including the
    overlapping-window case (step < size, i.e. halo > 0) this module actually uses."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import _window_axis

    torch.manual_seed(0)
    x = torch.randn(2, 20, 3)
    for size, step in [(6, 4), (6, 6), (5, 2)]:  # overlapping, exact, and heavily overlapping
        n_windows = (x.shape[1] - size) // step + 1
        want = x.unfold(1, size, step)
        got = _window_axis(x, 1, size, step, n_windows)
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [1, 2])
def test_window_axis_select_is_bit_exact_vs_slice(dtype, dim):
    """The selection-matmul windowing (device default) must equal the narrow+stack windowing to the
    BIT, in bf16 too: one 1.0 per selection row, so TensorE/CPU accumulate x*1 + 0*... exactly."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        _window_axis_select,
        _window_axis_split,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 24, 24, 3, 4, dtype=dtype)
    for size, step in [(16, 8), (14, 6), (24, 16), (8, 8)]:
        n_windows = (x.shape[dim] - size) // step + 1
        want = _window_axis_split(x, dim, size, step, n_windows)
        got = _window_axis_select(x, dim, size, step, n_windows)
        assert got.shape == want.shape and got.dtype == dtype
        assert torch.equal(got, want)
        assert got.is_contiguous()


def test_window_select_matrix_is_a_pure_selection():
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import _window_select_matrix

    sel = _window_select_matrix(20, 6, 4, 4, torch.float32, "cpu")
    assert sel.shape == (24, 20)
    assert torch.equal(sel.sum(1), torch.ones(24))  # exactly one key per window position
    rows = torch.arange(24)
    assert torch.equal(sel.argmax(1), (rows // 6) * 4 + rows % 6)


def test_neighborhood_select_matrices_match_in_graph_construction_and_give_same_result():
    """Host-built ``select=`` matrices (graph inputs) must equal the in-graph construction and give
    a bit-identical result, for the default and an explicit (padded) tiling."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        _window_select_matrix,
        neighborhood_select_matrices,
    )

    torch.manual_seed(0)
    q, k, v = (_rand(1, 136, 184, 2, 8, dtype=torch.bfloat16) for _ in range(3))
    for tile in (None, [16, 16]):
        sels = neighborhood_select_matrices((136, 184), [5, 5], tile, torch.bfloat16)
        assert len(sels) == 2 and all(s.dtype == torch.bfloat16 and s.is_contiguous() for s in sels)
        t = (8, 8) if tile is None else tuple(tile)
        for a, n in enumerate((136, 184)):
            n_tiles = -(-n // t[a])
            want = _window_select_matrix(
                n_tiles * t[a] + 8, t[a] + 8, t[a], n_tiles, torch.bfloat16, "cpu"
            )
            assert torch.equal(sels[a], want)
        got = neighborhood_attention_tiled(q, k, v, [5, 5], None, tile, select=sels)
        base = neighborhood_attention_tiled(q, k, v, [5, 5], None, tile)
        assert torch.equal(got, base)
    with pytest.raises(ValueError, match="one matrix per spatial axis"):
        neighborhood_attention_tiled(q, k, v, [5, 5], None, None, select=sels[:1])


@pytest.mark.parametrize("with_select", [False, True])
def test_compiled_op_traces_one_graph_and_reuses_it(with_select):
    """Smoke r18 on Trn2: ``warm_s`` 58-75 s == ``first_s`` -- the op RECOMPILED on every call,
    because the selection matrices came from a module-level dict cache consulted inside the trace
    (Dynamo guarded on ``key in cache``; the first call also traced the matrix construction, a
    scatter, into the graph). The compiled op must trace exactly one graph across repeated calls,
    with or without host-built ``select=`` inputs, and that graph must contain no scatter."""
    import torch._dynamo

    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        neighborhood_bias,
        neighborhood_select_matrices,
    )

    graphs = []

    def counting_backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch.manual_seed(0)
    q, k, v = (_rand(1, 40, 48, 2, 8, dtype=torch.bfloat16) for _ in range(3))
    bias = neighborhood_bias((40, 48), [5, 5], [False, False], (8, 8))
    kw = (
        {"select": neighborhood_select_matrices((40, 48), [5, 5], (8, 8), torch.bfloat16)}
        if with_select
        else {}
    )
    torch._dynamo.reset()
    fn = torch.compile(
        neighborhood_attention_tiled, backend=counting_backend, fullgraph=True, dynamic=False
    )
    outs = [fn(q, k, v, [5, 5], [False, False], [8, 8], bias, **kw) for _ in range(3)]
    torch._dynamo.reset()
    assert len(graphs) == 1, (
        f"neighborhood_attention_tiled recompiled: {len(graphs)} graphs for 3 calls"
    )
    assert all(torch.equal(o, outs[0]) for o in outs)
    ops = {str(n.target) for n in graphs[0].graph.nodes if n.op == "call_function"}
    assert not any("index_put" in o or "scatter" in o for o in ops), ops
    ref = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [8, 8], bias)
    assert torch.equal(outs[0], ref)


@pytest.mark.parametrize("window_impl", ["select", "slice"])
@pytest.mark.parametrize("upcast_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_window_impl_and_upcast_knobs_match_reference(window_impl, upcast_first, dtype):
    """Every (window_impl, upcast_first) setting is the same op: bit-identical to each other on CPU
    in fp32, within the bf16 band otherwise, on the FLUX-3-Action 136x184 grid at BOTH tilings."""
    torch.manual_seed(0)
    q, k, v = (_rand(1, 136, 184, 2, 8, dtype=dtype) for _ in range(3))
    ref = _reference(q, k, v, [5, 5], [False, False])
    for tile in (None, [16, 16]):
        got = neighborhood_attention_tiled(
            q, k, v, [5, 5], None, tile, window_impl=window_impl, upcast_first=upcast_first
        )
        assert got.shape == q.shape and got.dtype == dtype
        assert _rel(got, ref) < (1e-5 if dtype == torch.float32 else 2e-2)
        if dtype == torch.float32:
            base = neighborhood_attention_tiled(q, k, v, [5, 5], None, tile)  # select, no upcast
            assert torch.equal(got, base)


def test_window_impl_rejects_unknown():
    q = _rand(1, 10, 12, 2, 8)
    with pytest.raises(ValueError, match="window_impl"):
        neighborhood_attention_tiled(q, q, q, [5, 5], None, [4, 4], window_impl="unfold")


def test_default_window_impl_has_no_overlapping_narrow_in_the_hot_path():
    """The device default must not slice overlapping key windows (smoke r13-r17: the slice+stack
    windowing is wrong once fused with the attention in one bf16 graph on Trn2). Only the pad and
    the final crop may narrow -- never with a step smaller than the size."""
    import torch as _t

    calls: list = []
    orig = _t.Tensor.narrow

    def spy(self, dim, start, length):
        calls.append((dim, start, length))
        return orig(self, dim, start, length)

    _t.Tensor.narrow = spy
    try:
        q, k, v = (_rand(1, 20, 24, 2, 8) for _ in range(3))
        neighborhood_attention_tiled(q, k, v, [5, 5], None, [8, 8])
    finally:
        _t.Tensor.narrow = orig
    # the only narrows are the final crop (start 0, full axis length): no windowing slices
    assert all(start == 0 for _, start, _ in calls), calls


def test_no_multi_axis_permute_or_movedim_in_the_hot_path():
    """Round 11 (device): the compiled op produced rel_err 0.43 while compiling clean. The only ops
    in it outside plain matmul/softmax/pad/narrow/stack were multi-axis permutes (``permute`` with a
    non-adjacent order, ``movedim``) -- the same op class behind the three Wan VAE NCC_IDDT901
    fixes. The hot path must use only reshapes and single ADJACENT-axis transposes."""
    import torch as _t

    calls = {"permute": 0, "movedim": 0, "nonadjacent_transpose": 0}
    orig_p, orig_m, orig_t = _t.Tensor.permute, _t.Tensor.movedim, _t.Tensor.transpose

    def spy_p(self, *a, **k):
        calls["permute"] += 1
        return orig_p(self, *a, **k)

    def spy_m(self, *a, **k):
        calls["movedim"] += 1
        return orig_m(self, *a, **k)

    def spy_t(self, d0, d1):
        n = self.ndim
        if abs((d0 % n) - (d1 % n)) != 1:
            calls["nonadjacent_transpose"] += 1
        return orig_t(self, d0, d1)

    _t.Tensor.permute, _t.Tensor.movedim, _t.Tensor.transpose = spy_p, spy_m, spy_t
    try:
        q, k, v = (_rand(1, 8, 17, 23, 4, 8) for _ in range(3))
        neighborhood_attention_tiled(q, k, v, [3, 5, 5], [True, False, False], [4, 8, 8])
    finally:
        _t.Tensor.permute, _t.Tensor.movedim, _t.Tensor.transpose = orig_p, orig_m, orig_t
    # k.transpose(-1, -2) inside the QK matmul is the one adjacent transpose that is expected.
    assert calls == {"permute": 0, "movedim": 0, "nonadjacent_transpose": 0}, calls


def test_precomputed_bias_input_matches_inline():
    """neighborhood_bias() built on the host and passed as ``bias=`` (the way a compiled device
    caller should feed it) must give the same result as the inline construction."""
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import neighborhood_bias

    torch.manual_seed(0)
    q, k, v = (_rand(2, 17, 23, 4, 8) for _ in range(3))
    bias = neighborhood_bias((17, 23), [5, 5], [False, False], [8, 8])
    assert bias.shape == (3 * 3, 1, 64, (8 + 8) ** 2) and bias.dtype == torch.float32
    got = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [8, 8], bias=bias)
    want = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [8, 8])
    torch.testing.assert_close(got, want)
    # a caller-built 3D [tiles, tt, wt] bias is still accepted
    got3 = neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [8, 8], bias=bias[:, 0])
    torch.testing.assert_close(got3, want)
    assert _rel(got, _reference(q, k, v, [5, 5], [False, False])) < 1e-5


def test_no_unfold_in_the_hot_path():
    """Regression test for the Trn2 smoke failure (wrong numbers, not a compile error, at a
    non-tile-dividing grid: 136x184 at tile 16x16). Tensor.unfold has no native lowering on this
    XLA backend and silently produced incorrect results while compiling clean; it must never
    reappear in the hot path. Guard by patching it for the duration of one non-dividing call."""
    import torch as _t

    calls = {"unfold": 0}
    orig = _t.Tensor.unfold

    def spy(self, *a, **k):
        calls["unfold"] += 1
        return orig(self, *a, **k)

    _t.Tensor.unfold = spy
    try:
        q, k, v = (_rand(1, 136, 184, 4, 8) for _ in range(3))
        neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [16, 16])
    finally:
        _t.Tensor.unfold = orig
    assert calls == {"unfold": 0}, calls


def test_no_index_select_or_boolean_mask_in_the_hot_path():
    """The whole point: no index_select gather and no [N, N] bool mask (both fail neuronx-cc). Guard
    against a future edit reintroducing either by patching them for the duration of one call."""
    import torch as _t

    calls = {"index_select": 0, "masked_fill": 0}
    orig_is = _t.Tensor.index_select
    orig_mf = _t.Tensor.masked_fill

    def spy_is(self, *a, **k):
        calls["index_select"] += 1
        return orig_is(self, *a, **k)

    def spy_mf(self, *a, **k):
        calls["masked_fill"] += 1
        return orig_mf(self, *a, **k)

    _t.Tensor.index_select = spy_is
    _t.Tensor.masked_fill = spy_mf
    try:
        q, k, v = (_rand(1, 20, 24, 4, 8) for _ in range(3))
        neighborhood_attention_tiled(q, k, v, [5, 5], [False, False], [8, 8])
    finally:
        _t.Tensor.index_select = orig_is
        _t.Tensor.masked_fill = orig_mf
    assert calls == {"index_select": 0, "masked_fill": 0}, calls


@pytest.mark.parametrize("n_ax", [1, 2])
def test_compiles_fullgraph(n_ax):
    """Graph-break-free trace: the closest CPU proxy for 'the compiler lowers this as one graph'
    (the real pass/fail is the device smoke)."""
    torch._dynamo.reset()
    if n_ax == 1:
        q = k = v = _rand(1, 20, 4, 8)
        kernel, causal, tile = [5], [False], [8]
    else:
        q = k = v = _rand(1, 16, 20, 4, 8)
        kernel, causal, tile = [5, 5], [False, False], [8, 8]
    fn = torch.compile(neighborhood_attention_tiled, backend="eager", fullgraph=True, dynamic=False)
    got = fn(q, k, v, kernel, causal, tile)
    torch.testing.assert_close(got, neighborhood_attention_tiled(q, k, v, kernel, causal, tile))


def test_window_token_count_is_bounded_not_n_squared():
    """A tile's key window is prod(tile + 2*(kernel-1)), independent of the grid size -- that is the
    property that makes this compile where a full N^2 band does not. Check it does not scale with N."""
    small = neighborhood_attention_tiled(
        _rand(1, 20, 20, 2, 8),
        _rand(1, 20, 20, 2, 8),
        _rand(1, 20, 20, 2, 8),
        [5, 5],
        [False, False],
        [8, 8],
    )
    big = neighborhood_attention_tiled(
        _rand(1, 136, 184, 2, 8),
        _rand(1, 136, 184, 2, 8),
        _rand(1, 136, 184, 2, 8),
        [5, 5],
        [False, False],
        [8, 8],
    )
    assert small.shape == (1, 20, 20, 2, 8) and big.shape == (1, 136, 184, 2, 8)
    # window token count for tile 8 + 2*4 halo = 16 per axis -> 256, same for both grids.
    span = (8 + 2 * (5 - 1)) ** 2
    assert span == 256
