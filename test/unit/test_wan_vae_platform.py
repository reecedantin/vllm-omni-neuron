# SPDX-License-Identifier: Apache-2.0
"""Shared Wan VAE (platform code): config preservation and encode parity vs diffusers, on CPU.

Covers both layouts the plugin serves: Wan2.1-style (no patchify; Wan2.2-A14B) and Wan2.2-TI2V-5B
(``patch_size=2``, patchify inside the encoder graph; Cosmos3 family).
"""

from __future__ import annotations

import inspect

import pytest
import torch
from diffusers import AutoencoderKLWan

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    NeuronAutoencoderKLWan,
)

# Shrunk copies of the real configs (same block structure, tiny widths).
_TI2V_5B = dict(  # Wan-AI/Wan2.2-TI2V-5B-Diffusers vae/config.json, shrunk
    base_dim=16,
    decoder_base_dim=32,
    z_dim=8,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    latents_mean=[0.1 * i for i in range(8)],
    latents_std=[1.0 + 0.1 * i for i in range(8)],
    is_residual=True,
    in_channels=12,
    out_channels=12,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
)
_WAN21 = dict(  # Wan2.1 / Wan2.2-A14B VAE layout, shrunk
    base_dim=16,
    z_dim=4,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    latents_mean=[0.0] * 4,
    latents_std=[1.0] * 4,
)


# Cosmos3-Nano / Wan2.2-TI2V-5B VAE's REAL (not shrunk) config: base_dim=160 means the encoder
# mid-block attention is channel-width 640 (base_dim * dim_mult[-1] = 160*4), the exact scale at
# which a NCC_IDDT901 persisted after the patchify/AvgDown3D/DupUp3D fixes -- the shrunk _TI2V_5B config above (base_dim=16,
# mid-block width 64) never exercises that width. Real widths, tiny spatial dims for test speed.
_TI2V_5B_REAL_WIDTH = dict(
    base_dim=160,
    decoder_base_dim=256,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    latents_mean=[0.1 * i for i in range(48)],
    latents_std=[1.0 + 0.1 * i for i in range(48)],
    is_residual=True,
    in_channels=12,
    out_channels=12,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
)


def _kwargs(cfg):
    sig = inspect.signature(NeuronAutoencoderKLWan.__init__).parameters
    return {k: v for k, v in cfg.items() if k in sig}


@pytest.mark.parametrize("cfg", [_TI2V_5B, _WAN21], ids=["ti2v5b", "wan21"])
def test_config_not_clobbered_by_parent_defaults(cfg):
    vae = NeuronAutoencoderKLWan(**_kwargs(cfg))
    for key in ("z_dim", "patch_size", "latents_mean", "latents_std", "base_dim"):
        if key in cfg:
            assert vae.config[key] == cfg[key], key
    assert vae.spatial_compression_ratio == cfg.get("scale_factor_spatial", 8)
    for attr in ("use_slicing", "use_tiling", "tile_sample_min_height", "tile_sample_stride_width"):
        assert hasattr(vae, attr), attr


@pytest.mark.parametrize("cfg,hw", [(_TI2V_5B, 64), (_WAN21, 32)], ids=["ti2v5b", "wan21"])
def test_encode_matches_diffusers(cfg, hw):
    kw = _kwargs(cfg)
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**kw).eval()
    neu = NeuronAutoencoderKLWan(**kw).eval()
    neu.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(1, 3, 5, hw, hw)
    with torch.no_grad():
        want = ref.encode(x).latent_dist.mean
        got = neu.encode(x).latent_dist.mean
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_encode_matches_diffusers_at_real_cosmos3_nano_width():
    """Regression test: a NCC_IDDT901 persisted on
    device at the real Cosmos3-Nano VAE's channel width (base_dim=160 -> encoder mid-block
    attention at 640 channels) after the patchify/AvgDown3D/DupUp3D fixes, which were only
    exercised at shrunk widths. Real widths here (tiny spatial dims for test speed) catch any
    remaining shape-sensitive correctness issue the shrunk configs cannot."""
    kw = _kwargs(_TI2V_5B_REAL_WIDTH)
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**kw).eval()
    neu = NeuronAutoencoderKLWan(**kw).eval()
    neu.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(1, 3, 1, 64, 64)  # single pixel frame, as an I2V conditioning encode uses
    with torch.no_grad():
        want = ref.encode(x).latent_dist.mean
        got = neu.encode(x).latent_dist.mean
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


# --- fixed-shape spatial tiling (shared helper vae_tiling.py) ---


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().clamp_min(1e-20)
    return (10 * torch.log10(torch.tensor(4.0) / mse)).item()  # pixel range [-1, 1]


def _tiled_vae(cfg, tile_px, stride_px):
    kw = _kwargs(cfg)
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**kw).eval()
    neu = NeuronAutoencoderKLWan(**kw).eval()
    neu.load_state_dict(ref.state_dict(), strict=True)
    neu.use_tiling = True
    neu.tile_sample_min_height = neu.tile_sample_min_width = tile_px
    neu.tile_sample_stride_height = neu.tile_sample_stride_width = stride_px
    neu.compile(backend="eager", compile_encoder=True)  # eager graphs: CPU, no device needed
    return ref, neu


def _count_tile_shapes(neu, method_name):
    """Spy on the per-tile callable to record every tile input shape it was handed."""
    shapes = []
    orig = getattr(neu, method_name)

    def spy(t):
        shapes.append(tuple(t.shape))
        return orig(t)

    setattr(neu, method_name, spy)
    return shapes


@pytest.mark.parametrize("hw", [(160, 144), (176, 176)])
def test_tiled_decode_single_tile_shape_and_no_worse_than_diffusers_tiling(hw):
    """The grid is not a multiple of the stride on either axis, so diffusers' scheme produces
    narrower edge tiles (= extra NEFFs on device). Every tile here must have ONE shape, and the
    blended result must be at least as close to the untiled decode as diffusers' own tiled_decode
    with the same tile/stride (the seams are the only difference in both schemes)."""
    ref, neu = _tiled_vae(_TI2V_5B, tile_px=128, stride_px=96)
    h, w = hw
    torch.manual_seed(1)
    x = torch.randn(1, 3, 5, h, w).clamp(-1, 1)
    with torch.no_grad():
        z = ref.encode(x).latent_dist.mean
        want = ref.decode(z).sample
        ref.enable_tiling(
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_stride_height=96,
            tile_sample_stride_width=96,
        )
        diffusers_tiled = ref.decode(z).sample
        shapes = _count_tile_shapes(neu, "_tile_decode_one")
        got = neu.decode(z).sample
    assert got.shape == want.shape == (1, 3, 5, h, w)
    assert len(set(shapes)) == 1, shapes  # one compiled tile shape
    assert len(shapes) == 4  # 2 x 2: 160 = 128 + 32 (pulled back), 144 = 128 + 16, 176 = 128 + 48
    assert _psnr(got, want) > _psnr(diffusers_tiled, want) - 1.0, (
        _psnr(got, want),
        _psnr(diffusers_tiled, want),
    )


def test_tiled_encode_single_tile_shape_and_no_worse_than_diffusers_tiling():
    """Raw-pixel tiles of ONE shape (patchify runs in-graph per tile), no worse than diffusers'
    tiled_encode at the same geometry. diffusers' Wan tiled_encode runs AFTER host patchify and
    reads tile_sample_min in patchified units, so its equivalent of a 128/96 px raw tiling is
    tiled_encode(patchify(x)) with 64/48 -- called directly, since its _encode threshold check
    compares pre-patchify dims against post-patchify tile sizes and would not tile here at all."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify as hf_patchify

    ref, neu = _tiled_vae(_TI2V_5B, tile_px=128, stride_px=96)
    torch.manual_seed(1)
    x = torch.randn(1, 3, 5, 176, 160).clamp(-1, 1)
    with torch.no_grad():
        want = ref.encode(x).latent_dist.mean
        ref.tile_sample_min_height = ref.tile_sample_min_width = 64
        ref.tile_sample_stride_height = ref.tile_sample_stride_width = 48
        ref.clear_cache()
        h_tiled = ref.tiled_encode(hf_patchify(x, patch_size=2))
        diffusers_tiled = h_tiled[:, : h_tiled.shape[1] // 2]
        shapes = _count_tile_shapes(neu, "_tile_encode_one")
        got = neu.encode(x).latent_dist.mean
    assert got.shape == want.shape == diffusers_tiled.shape
    assert len(set(shapes)) == 1 and shapes[0][-2:] == (128, 128), shapes  # raw-pixel tiles
    rel = lambda a: ((a - want).norm() / want.norm()).item()  # noqa: E731
    assert rel(diffusers_tiled) > 1e-3  # the reference really tiled (seams present)
    assert rel(got) < rel(diffusers_tiled) * 1.25 + 1e-3, (rel(got), rel(diffusers_tiled))


def test_tiled_paths_never_pass_strided_views_to_the_graph():
    """Every tile handed to the compiled decoder/encoder must be a contiguous base tensor (the
    device executor refuses strided views as graph inputs -- smoke round 11)."""
    ref, neu = _tiled_vae(_TI2V_5B, tile_px=128, stride_px=96)
    seen = []
    orig_dec = neu._tile_decode_one
    orig_enc = neu._tile_encode_one
    neu._tile_decode_one = lambda t: (seen.append(t.is_contiguous()), orig_dec(t))[1]
    neu._tile_encode_one = lambda t: (seen.append(t.is_contiguous()), orig_enc(t))[1]
    x = torch.randn(1, 3, 1, 160, 144).clamp(-1, 1)
    with torch.no_grad():
        z = neu.encode(x).latent_dist.mean
        neu.decode(z)
    assert seen and all(seen)


# --- patch-parallel plane-chunk gather: no eager slicing of a device tensor, no aliased chunks ---


def _aot_compile(fn):
    """Real Dynamo + AOT autograd on CPU (``aot_eager``): the output-aliasing classification that
    turned ``.contiguous()`` of a split into ``as_strided`` views of the graph input only happens
    under AOT autograd, so an eager stub cannot reproduce the bug this file guards."""
    return torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)


def _fake_executor(monkeypatch, world=3, backend="aot_eager"):
    """A bare _NeuronDistributedVaeExecutor with the collective stubbed and the PRODUCTION compile
    path (``_compile_device_graph``: cache per (name, key), own code object per graph) pointed at
    ``backend``. The gather stacks `world` rank-shifted copies of the local chunk."""
    import vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan as mod

    monkeypatch.setattr(mod, "get_compile_backend_name", lambda: backend)
    ex = mod._NeuronDistributedVaeExecutor.__new__(mod._NeuronDistributedVaeExecutor)
    ex.world_size, ex.rank = world, 0
    ex.compiled_names = []
    real_compile = ex._compile_device_graph

    def compile_device_graph(name, key, fn):
        if (name, key) not in getattr(ex, "_device_graphs", {}):
            ex.compiled_names.append(name)
        return real_compile(name, key, fn)

    ex._compile_device_graph = compile_device_graph
    ex._get_compiled_device_gather = lambda: "gather"
    ex._raise_if_device_gather_failed = lambda err, phase: None
    ex._precompile_lite_device_gather = lambda g, chunks, t: None
    ex.gather_calls = []

    def fake_gather_chunk(compiled_gather, flat, chunk_index):
        ex.gather_calls.append(
            (chunk_index, tuple(flat.shape), flat.is_contiguous(), flat.storage_offset())
        )
        return torch.cat([flat + 1000.0 * r for r in range(world)])  # rank r adds 1000*r

    ex._gather_chunk = fake_gather_chunk
    return ex


@pytest.mark.parametrize("slots", [1, 4], ids=["one-slot", "four-slots"])
@pytest.mark.parametrize(
    "planes,chunk_planes", [(7, 3), (6, 3), (5, 5), (4, 8)], ids=["ragged", "even", "one", "short"]
)
def test_lite_plane_gather_matches_slot_major_narrow_without_eager_device_slicing(
    planes, chunk_planes, slots, monkeypatch
):
    """The patch-parallel chunk payload used to be
    local_planes.narrow(1, start, size).contiguous() on a DEVICE tensor, refused by the Lite executor
    (TI2V-5B 704p patch-parallel decode). Now every chunk is a contiguous base tensor cut by its
    own compiled graph, and the gathered chunk is yielded planes-major, [world, size, slots, H, W],
    a view of the collective's output (no slot-major restore copy: the blend graph indexes
    gathered[rank, :, slot] inside itself). Check the yielded chunks equal what the old narrow path
    produced, for even and ragged (tail) chunking and a single whole chunk -- with ONE slot (fewer
    tiles than ranks: a Cosmos3-Super clip with 8 tiles on 16/32 ranks, a TI2V clip with 28 tiles on
    32 ranks), where a
    transpose is a no-op layout-wise and a split+contiguous cut aliased the input."""
    torch.manual_seed(0)
    h, w = 5, 6
    local_planes = torch.randn(slots, planes, h, w)
    ex = _fake_executor(monkeypatch, world=3)
    starts = list(range(0, planes, chunk_planes))
    got = list(ex._stream_gather_planes_lite(local_planes, chunk_planes, planes, starts))
    assert len(got) == len(starts) == len(ex.gather_calls)
    for (start, gathered), (ci, flat_shape, contig, offset) in zip(
        zip(starts, got), ex.gather_calls
    ):
        size = min(chunk_planes, planes - start)
        assert contig and offset == 0 and flat_shape == (size * slots * h * w,)
        old_chunk = local_planes.narrow(1, start, size)  # [slots, size, H, W] (old layout)
        assert gathered.shape == (3, size, slots, h, w) and gathered.storage_offset() == 0
        for r in range(3):
            torch.testing.assert_close(gathered[r].transpose(0, 1), old_chunk + 1000.0 * r)
    # One cut graph per chunk (start, size); nothing else is compiled per chunk any more.
    assert ex.compiled_names.count("plane_chunk") == len(starts)
    assert "slot_major" not in ex.compiled_names


def test_many_plane_chunks_never_trip_dynamo_recompile_limit(monkeypatch):
    """Each per-chunk cut is a closure over (start, size) made from ONE ``def``; Dynamo caches and
    counts recompiles per CODE object, so past ``recompile_limit`` (8) the 9th chunk raises
    ``FailOnRecompileLimitHit`` (fullgraph=True; reproduced on CPU with the isolation disabled).
    ``_compile_device_graph`` gives every graph its own code object: 14 chunks here (a 64-rank
    decode cuts up to 7, a second resolution doubles it) must each reach the backend once."""
    import torch._dynamo

    compiled = []

    def counting_backend(gm, example_inputs, **options):
        compiled.append(gm)
        return gm.forward

    torch._dynamo.reset()
    planes, chunk_planes, slots, h, w = 40, 3, 1, 2, 2
    assert -(-planes // chunk_planes) > torch._dynamo.config.recompile_limit
    local_planes = torch.randn(slots, planes, h, w)
    ex = _fake_executor(monkeypatch, world=2, backend=counting_backend)
    starts = list(range(0, planes, chunk_planes))
    got = list(ex._stream_gather_planes_lite(local_planes, chunk_planes, planes, starts))
    assert len(got) == len(starts) == 14
    assert ex.compiled_names.count("plane_chunk") == 14
    assert len(compiled) == 14  # 14 cuts, no per-size restore graphs
    for start, gathered in zip(starts, got):
        size = min(chunk_planes, planes - start)
        torch.testing.assert_close(
            gathered[1].transpose(0, 1), local_planes.narrow(1, start, size) + 1000.0
        )


def test_device_gather_is_compiled_once_per_chunk_shape_with_no_recompile_limit(monkeypatch):
    """8-rank smoke r3: a dozen distinct chunk shapes (several gather checks x full/tail sizes) plus
    the no_grad precompile vs grad-enabled call drove the ONE compiled ``device_gather`` past
    Dynamo's recompile limit ('Lite VAE device gather failed during compiled gather precompile').
    The gather is now dispatched per shape through ``_compile_device_graph`` (own code object each)
    and always runs under no_grad: 12 shapes x (precompile-style no_grad call + plain call) must
    compile exactly 12 graphs and stay exact."""
    import torch._dynamo

    import vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan as mod

    world = 3
    compiled = []

    def counting_backend(gm, example_inputs, **options):
        compiled.append(gm)
        return gm.forward

    def fake_all_gather(tensor, gather_dim, group):  # the collective, stood in for on CPU
        return torch.cat([tensor + 1000.0 * r for r in range(world)], dim=gather_dim)

    monkeypatch.setattr(mod.funcol, "all_gather_tensor", fake_all_gather)
    monkeypatch.setattr(mod, "get_compile_backend_name", lambda: counting_backend)
    monkeypatch.setattr(mod, "is_lite_runtime", lambda: False)  # skip the process-group registry
    torch._dynamo.reset()
    ex = mod._NeuronDistributedVaeExecutor.__new__(mod._NeuronDistributedVaeExecutor)
    ex.group, ex.world_size, ex.rank = None, world, 0
    gather = ex._get_compiled_device_gather()
    assert ex._get_compiled_device_gather() is gather  # cached dispatcher, stable id for signatures
    shapes = [(n,) for n in range(64, 64 + 12 * 16, 16)]
    assert len(shapes) > torch._dynamo.config.recompile_limit
    for shape in shapes:
        x = torch.randn(*shape)
        with torch.no_grad():
            g1 = gather(x)
        g2 = gather(x)
        want = torch.stack([x + 1000.0 * r for r in range(world)])
        torch.testing.assert_close(g1, want)
        torch.testing.assert_close(g2, want)
    assert len(compiled) == len(shapes)


@pytest.mark.parametrize(
    "world,planes,chunk_planes,offset_elems",
    [(16, 567, 256, 2 * 256), (32, 363, 128, 2 * 128)],
    ids=["super-189f-16r", "ti2v-121f-32r"],
)
def test_old_split_cut_aliased_the_input_at_one_slot_and_the_payload_check_catches_it(
    world, planes, chunk_planes, offset_elems
):
    """Root cause of the multi-chunk (189-frame Cosmos3-Super) and 32-rank (Wan2.2-TI2V) patch-parallel decode
    ``nrt_tensor_copy status=2``: with one slot, ``transpose(0, 1)`` -> ``torch.split`` ->
    ``.contiguous()`` inside one compiled graph returns ``as_strided`` VIEWS of the input (AOT
    autograd's alias-of-input output path), chunk k at element offset k * chunk_numel. The Lite
    device copy_ sized the wait probe of the LAST chunk as 2 bytes - offset bytes: the 16-rank run logged
    2 - 2^26 (two 256-plane chunks at 256x256 bf16), the 32-rank run 2 - 2^25 (two 128-plane
    chunks at 256x256, 32 ranks). Same arithmetic here at a tiny tile, and the payload guard refuses
    exactly those chunks while accepting the clone-based cut."""
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor as Ex,
    )

    h = w = 2  # 256 -> 2: same plane arithmetic, 2^14 times less memory
    local_planes = torch.randn(1, planes, h, w)
    # The chunk width gather_and_blend_tiles picks at a 256x256 bf16 tile for this world.
    assert Ex.MAX_DEVICE_GATHER_BYTES // (world * 1 * 256 * 256 * 2) == chunk_planes

    def old_cut(t):
        pm = t.transpose(0, 1)
        return tuple(c.contiguous() for c in torch.split(pm, chunk_planes, dim=0))

    chunks = _aot_compile(old_cut)(local_planes)
    assert len(chunks) == 3
    assert chunks[-1]._is_view() and chunks[-1].storage_offset() == offset_elems * h * w
    with pytest.raises(RuntimeError, match="offset view"):
        Ex._check_gather_payload(chunks[-1], tuple(chunks[-1].shape), local_planes)
    with pytest.raises(RuntimeError, match="aliases"):  # chunk 0: offset 0 but still a view
        Ex._check_gather_payload(chunks[0], tuple(chunks[0].shape), local_planes)

    def new_cut(t, start=2 * chunk_planes, size=planes - 2 * chunk_planes):
        return t.transpose(0, 1).narrow(0, start, size).clone(memory_format=torch.contiguous_format)

    tail = _aot_compile(new_cut)(local_planes)
    assert not tail._is_view() and tail.storage_offset() == 0 and tail.is_contiguous()
    Ex._check_gather_payload(tail, (planes - 2 * chunk_planes, 1, h, w), local_planes)
    torch.testing.assert_close(tail, local_planes[0, 2 * chunk_planes :].unsqueeze(1))


# --- VAE gather replica groups on the trn2 chip torus (32- and 64-rank layouts) ---


@pytest.mark.parametrize(
    "world,global_world,expected",
    [
        (16, 16, [list(range(16))]),
        (16, 32, [list(range(16)), list(range(16, 32))]),
        (32, 32, [list(range(32))]),
        (16, 64, [list(range(s, s + 16)) for s in range(0, 64, 16)]),
        (32, 64, [list(range(32)), list(range(32, 64))]),
        (64, 64, [list(range(64))]),
    ],
)
def test_gather_replica_groups_partition_the_world_into_rank0_blocks(world, global_world, expected):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor as Ex,
    )

    groups = Ex.gather_replica_groups(world, global_world)
    assert groups == expected
    assert sorted(r for g in groups for r in g) == list(range(global_world))
    with pytest.raises(ValueError):
        Ex.gather_replica_groups(24, 32)


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 32, 64])
def test_rank0_gather_groups_of_power_of_two_size_are_torus_routable(size):
    """1-4 ranks = inside one chip; 8 = a torus-adjacent chip pair; 16 = a full torus row (the models'
    working 16-rank layout); 32 = two full rows (TP4xCP4xCFG2, TP8xCP4); 64 = the whole torus."""
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        gather_group_torus_routable,
    )

    assert gather_group_torus_routable(range(size))


@pytest.mark.parametrize(
    "ranks,why",
    [
        (range(12), "three chips in a row: a path, not a ring"),
        (range(24), "six chips: not a rectangle of full rows/columns"),
        (range(48), "three full rows: a path of rows"),
        (range(6), "chip 0 gives 4 cores, chip 1 gives 2: unequal core offsets"),
        ([0, 32], "chips 0 and 8 share a torus column two hops apart (parallel_state.py)"),
        ([0, 8], "chips 0 and 2: same row, two hops apart"),
    ],
)
def test_off_ring_gather_groups_are_rejected(ranks, why):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        gather_group_torus_routable,
    )

    assert not gather_group_torus_routable(ranks), why


@pytest.mark.parametrize(
    "ranks",
    [
        list(range(4)) + list(range(12, 16)),  # chips 0 and 3: adjacent through the row wrap
        [0, 1, 4, 5],  # cores 0-1 of chips 0 and 1: same offsets on an adjacent pair
        list(range(0, 16)) + list(range(48, 64)),  # rows 0 and 3: adjacent through the column wrap
        [r for r in range(64) if (r // 4) % 4 in (0, 1)],  # columns 0-1 of every row: 2x4 block
    ],
)
def test_wrapped_and_partial_chip_groups_are_torus_routable(ranks):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        gather_group_torus_routable,
    )

    assert gather_group_torus_routable(ranks)


def test_gather_group_check_raises_on_trn2_only():
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor as Ex,
    )

    for world, global_world in ((16, 32), (32, 32), (32, 64), (64, 64)):
        Ex._check_gather_groups_routable(Ex.gather_replica_groups(world, global_world), "trn2")
    bad = Ex.gather_replica_groups(48, 48)
    with pytest.raises(RuntimeError, match="not routable"):
        Ex._check_gather_groups_routable(bad, "trn2")
    Ex._check_gather_groups_routable(bad, "trn3")  # switch fabric: no torus rule
    Ex._check_gather_groups_routable(bad, "inf2")


# ---------------------------------------------------------------- VAE mid-block attention dispatch


@pytest.mark.parametrize(
    "gen,dim,nki",
    [
        (3, 384, True),
        (3, 512, True),
        (3, 640, False),
        (3, 1024, False),
        (2, 384, False),
        (4, 1024, False),
    ],
)
def test_vae_attention_nki_gate(monkeypatch, gen, dim, nki):
    """attention_cte only for NC-v3+ AND head_dim <= 512 (Wan2.2-TI2V-5B mid blocks are 640 / 1024)."""
    from vllm_omni_neuron import nc_generation
    from vllm_omni_neuron.diffusion.distributed.autoencoders import autoencoder_kl_wan as vae

    monkeypatch.setenv("VLLM_OMNI_NEURON_CORE_GEN", str(gen))
    nc_generation.neuron_core_generation.cache_clear()
    assert vae._vae_attn_can_use_nki(dim) is nki


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("dim,tokens", [(384, 256), (1024, 400)])
def test_vae_attention_fallback_matches_sdpa(dtype, tol, dim, tokens):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _vae_attention_explicit,
    )

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, tokens, dim) for _ in range(3))
    scale = dim**-0.5
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[:, None], k[:, None], v[:, None]
    ).squeeze(1)
    out = _vae_attention_explicit(q.to(dtype), k.to(dtype), v.to(dtype), scale)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)


# --- gather_and_blend_tiles end to end on CPU (gather stubbed): layout, to_host streaming, budget ---


class _GridSpec:
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape


def _blend_setup(world, grid, tile_hw, stride_hw, frame_shape, seed=7):
    """Synthetic tile grid dealt round-robin to `world` ranks; every rank's local pack + meta."""
    gh, gw = grid
    th, tw = tile_hw
    sh, sw = stride_hw
    tiles = gh * gw
    slots = max(1, -(-tiles // world))
    torch.manual_seed(seed)
    canvas = torch.randn(*frame_shape, (gh - 1) * sh + th, (gw - 1) * sw + tw).to(torch.bfloat16)
    tid_coord = {r * gw + c: (r, c) for r in range(gh) for c in range(gw)}
    packs, metas = [], []
    for rank in range(world):
        local = torch.zeros(slots, *frame_shape, th, tw, dtype=torch.bfloat16)
        meta = torch.full((slots, 3), -1, dtype=torch.int64)
        for slot, tid in enumerate(range(rank, tiles, world)):
            row, col = tid_coord[tid]
            local[slot] = canvas[..., row * sh : row * sh + th, col * sw : col * sw + tw]
            meta[slot] = torch.tensor([tid, th, tw])
        packs.append(local)
        metas.append(meta)
    full_h, full_w = gh * sh, gw * sw
    common = dict(
        full_height=full_h,
        full_width=full_w,
        stride_height=sh,
        stride_width=sw,
        blend_height=th - sh,
        blend_width=tw - sw,
        clamp=True,
    )
    return (
        canvas[..., :full_h, :full_w].float().clamp(-1, 1),
        packs,
        metas,
        tid_coord,
        _GridSpec(grid),
        common,
    )


def _rank0_executor_cpu(monkeypatch, world, packs):
    """Rank-0 executor whose gather returns every rank's real pack (the CPU path of
    _stream_gather_planes), compiled graphs eager."""
    import vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan as mod

    monkeypatch.setattr(mod, "is_lite_runtime", lambda: False)
    ex = mod._NeuronDistributedVaeExecutor.__new__(mod._NeuronDistributedVaeExecutor)
    ex.world_size, ex.rank, ex.group = world, 0, None
    ex._compile_device_graph = lambda name, key, fn: fn
    planes_by_rank = None

    def gather_tensors(t):  # t = this rank's chunk [slots, size, H, W]; emulate the all-gather
        nonlocal planes_by_rank
        return [p for p in planes_by_rank(t)]

    ex._chunk_from = None

    def set_source(local_planes_all):  # [world, slots, planes, H, W]
        nonlocal planes_by_rank

        def by_rank(t):
            # find which (start, size) this chunk is by matching against rank 0's planes
            size = t.shape[1]
            for start in range(0, local_planes_all.shape[2] - size + 1):
                if torch.equal(local_planes_all[0, :, start : start + size], t):
                    return [
                        local_planes_all[r, :, start : start + size].contiguous()
                        for r in range(world)
                    ]
            raise AssertionError("chunk not found")

        planes_by_rank = by_rank

    ex.gather_tensors = gather_tensors
    ex.set_source = set_source
    return ex


@pytest.mark.parametrize("world,grid", [(3, (2, 4)), (8, (2, 4)), (4, (4, 7))])
@pytest.mark.parametrize("to_host", [False, True])
def test_gather_and_blend_tiles_reproduces_the_canvas_across_chunks(
    monkeypatch, world, grid, to_host
):
    """The blend graph now takes the planes-major gathered chunk and picks (rank, slot) views inside;
    across several chunks (budget forced small) and with/without host streaming the merged output
    must reproduce the canvas every tile was cut from (bf16 blend rounding only), be contiguous, and
    land on the host when to_host=True."""
    canvas, packs, metas, tid_coord, grid_spec, common = _blend_setup(
        world, grid, (16, 16), (12, 12), (3, 5)
    )
    ex = _rank0_executor_cpu(monkeypatch, world, packs)
    slots = packs[0].shape[0]
    local_planes_all = torch.stack([p.reshape(slots, 15, 16, 16) for p in packs])
    ex.set_source(local_planes_all)
    monkeypatch.delenv(ex.GATHER_BUDGET_ENV, raising=False)
    plane_bytes = world * slots * 16 * 16 * 2
    ex.MAX_DEVICE_GATHER_BYTES = (
        plane_bytes * 4
    )  # 4 planes per chunk: 15 planes -> 4 chunks (ragged tail)
    assert ex.gather_budget_bytes() // plane_bytes == 4
    merged = ex.gather_and_blend_tiles(
        packs[0], metas, grid_spec, tid_coord, to_host=to_host, **common
    )
    assert merged.shape == canvas.shape and merged.is_contiguous() and merged.device.type == "cpu"
    rel = ((merged.float() - canvas).norm() / canvas.norm()).item()
    assert rel < 3e-3, rel


def test_gather_budget_env_overrides_the_default(monkeypatch):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor as Ex,
    )

    ex = Ex.__new__(Ex)
    monkeypatch.delenv(Ex.GATHER_BUDGET_ENV, raising=False)
    assert ex.gather_budget_bytes() == 512 * 1024 * 1024
    monkeypatch.setenv(Ex.GATHER_BUDGET_ENV, "128")
    assert ex.gather_budget_bytes() == 128 * 1024 * 1024
    monkeypatch.setenv(Ex.GATHER_BUDGET_ENV, "lots")
    assert ex.gather_budget_bytes() == 512 * 1024 * 1024  # ignored with a warning


def test_blend_graph_is_keyed_on_tile_sources_and_gathered_shape(monkeypatch):
    """Two decodes whose tiles land on different (rank, slot) sources must not share a compiled
    blend graph (the source map is baked into the graph), while identical ones must."""
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _NeuronDistributedVaeExecutor as Ex,
    )

    ex = Ex.__new__(Ex)
    seen = []
    ex._compile_device_graph = lambda name, key, fn: seen.append((name, key)) or fn
    geo = dict(
        grid_height=1,
        grid_width=2,
        heights=(4,),
        widths=(4, 4),
        full_height=4,
        full_width=8,
        stride_height=4,
        stride_width=4,
        blend_height=0,
        blend_width=0,
        clamp=False,
    )
    ex._compiled_blend_graph((3, 2, 1, 4, 4), ((0, 0), (1, 0)), **geo)
    ex._compiled_blend_graph((3, 2, 1, 4, 4), ((0, 0), (2, 0)), **geo)
    ex._compiled_blend_graph((3, 2, 1, 4, 4), ((0, 0), (1, 0)), **geo)
    assert len(seen) == 3 and seen[0] == seen[2] and seen[0] != seen[1]
    gathered = torch.arange(3 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(3, 2, 1, 4, 4)
    merge = ex._compiled_blend_graph((3, 2, 1, 4, 4), ((0, 0), (2, 0)), **geo)
    out = merge(gathered)
    torch.testing.assert_close(out[:, :, :4], gathered[0, :, 0])
    torch.testing.assert_close(out[:, :, 4:], gathered[2, :, 0])


def test_vae_group_smaller_than_world_warns_or_refuses(monkeypatch):
    from vllm_omni_neuron.diffusion.distributed.autoencoders import autoencoder_kl_wan as mod

    monkeypatch.delenv(mod.VAE_GROUP_STRICT_ENV, raising=False)
    mod.check_vae_group_covers_world(32, 32)  # full group: silent
    mod.check_vae_group_covers_world(8, None)  # no process group: nothing to compare
    warned = []
    monkeypatch.setattr(mod.logger, "warning", lambda msg, *a: warned.append(msg))
    mod.check_vae_group_covers_world(8, 32)
    assert warned and "8 ranks" in warned[0] and "32" in warned[0]
    monkeypatch.setenv(mod.VAE_GROUP_STRICT_ENV, "1")
    with pytest.raises(RuntimeError, match="outside the group"):
        mod.check_vae_group_covers_world(8, 32)


# --- full-size decode tiles: last tile pulled back to the frame edge, start-aware device blend ---


def _diffusers_inplace_merge(tiles, row_starts, col_starts, stride, blend, total):
    """diffusers ``tiled_decode`` merge semantics (``blend_v``/``blend_h`` write IN PLACE, so each
    tile blends against the already-blended neighbour), generalised to arbitrary tile starts with
    the kept-range rule: tile k owns ``[prev_end, start_k + stride)`` (the last: to the edge) and
    ramps over the first ``blend`` samples of it. Loop form, independent of the graph code."""
    (sh, sw), (bh, bw), (fh, fw) = stride, blend, total

    def kept(starts, s, t):
        out, prev = [], 0
        for k, st in enumerate(starts):
            end = t if k == len(starts) - 1 else min(st + s, t)
            out.append((prev, end))
            prev = end
        return out

    kr, kc = kept(row_starts, sh, fh), kept(col_starts, sw, fw)
    work = {idx: t.clone().float() for idx, t in tiles.items()}
    out = torch.zeros(*next(iter(tiles.values())).shape[:-2], fh, fw)
    for r, rs in enumerate(row_starts):
        for c, cs in enumerate(col_starts):
            b = work[(r, c)]
            if r > 0:
                a, ps, (beg, end) = work[(r - 1, c)], row_starts[r - 1], kr[r]
                n = min(bh, end - beg)
                for y in range(n):
                    b[..., beg - rs + y, :] = a[..., beg - ps + y, :] * (1 - y / n) + b[
                        ..., beg - rs + y, :
                    ] * (y / n)
            if c > 0:
                a, ps, (beg, end) = work[(r, c - 1)], col_starts[c - 1], kc[c]
                n = min(bw, end - beg)
                for x in range(n):
                    b[..., :, beg - cs + x] = a[..., :, beg - ps + x] * (1 - x / n) + b[
                        ..., :, beg - cs + x
                    ] * (x / n)
            (r0, r1), (c0, c1) = kr[r], kc[c]
            out[..., r0:r1, c0:c1] = b[..., r0 - rs : r1 - rs, c0 - cs : c1 - cs]
    return out


def _pack_tiles(tiles, world, frame_shape, th, tw):
    gh = 1 + max(r for r, _ in tiles)
    gw = 1 + max(c for _, c in tiles)
    n = gh * gw
    slots = max(1, -(-n // world))
    tid_coord = {r * gw + c: (r, c) for r in range(gh) for c in range(gw)}
    packs, metas = [], []
    for rank in range(world):
        local = torch.zeros(slots, *frame_shape, th, tw, dtype=torch.bfloat16)
        meta = torch.full((slots, 3), -1, dtype=torch.int64)
        for slot, tid in enumerate(range(rank, n, world)):
            local[slot] = tiles[tid_coord[tid]]
            meta[slot] = torch.tensor([tid, th, tw])
        packs.append(local)
        metas.append(meta)
    return packs, metas, tid_coord, _GridSpec((gh, gw))


@pytest.mark.parametrize(
    "total,tile,stride,world",
    [
        ((30, 52), (16, 16), (14, 12), 8),  # A1's 256/224/192 geometry, latent units
        ((60, 104), (32, 32), (28, 24), 3),  # same in patchified output units (x2)
        ((32, 40), (16, 16), (8, 12), 4),  # even rows, pulled-back columns
        ((10, 12), (16, 16), (14, 12), 2),  # smaller than one tile
    ],
)
def test_start_aware_blend_matches_diffusers_inplace_reference(
    monkeypatch, total, tile, stride, world
):
    """Random (inconsistent) tile contents, full-size tiles at vae_tiling.tile_starts: the device
    blend graph equals the diffusers in-place merge generalised to the pulled-back last tile, and
    tiles cut from one canvas reproduce it exactly at every pixel incl. the right/bottom strips."""
    from vllm_omni_neuron.diffusion.layers.vae_tiling import tile_starts

    fh, fw = total
    th, tw = min(tile[0], fh), min(tile[1], fw)
    rs, cs = tile_starts(fh, tile[0], stride[0]), tile_starts(fw, tile[1], stride[1])
    frame_shape = (2, 3)
    torch.manual_seed(3)
    common = dict(
        full_height=fh,
        full_width=fw,
        stride_height=stride[0],
        stride_width=stride[1],
        blend_height=tile[0] - stride[0],
        blend_width=tile[1] - stride[1],
        clamp=False,
        row_starts=tuple(rs),
        col_starts=tuple(cs),
    )
    for mode in ("random", "canvas"):
        canvas = torch.randn(*frame_shape, fh, fw).to(torch.bfloat16)
        tiles = {}
        for r, r0 in enumerate(rs):
            for c, c0 in enumerate(cs):
                tiles[(r, c)] = (
                    torch.randn(*frame_shape, th, tw).to(torch.bfloat16)
                    if mode == "random"
                    else canvas[..., r0 : r0 + th, c0 : c0 + tw].clone()
                )
        packs, metas, tid_coord, grid_spec = _pack_tiles(tiles, world, frame_shape, th, tw)
        ex = _rank0_executor_cpu(monkeypatch, world, packs)
        slots = packs[0].shape[0]
        ex.set_source(torch.stack([p.reshape(slots, 6, th, tw) for p in packs]))
        merged = ex.gather_and_blend_tiles(packs[0], metas, grid_spec, tid_coord, **common)
        assert merged.shape == (*frame_shape, fh, fw)
        if mode == "canvas":
            torch.testing.assert_close(merged.float(), canvas.float(), atol=1e-2, rtol=0)
        else:
            want = _diffusers_inplace_merge(
                {k: v.float() for k, v in tiles.items()},
                rs,
                cs,
                stride,
                (tile[0] - stride[0], tile[1] - stride[1]),
                (fh, fw),
            )
            torch.testing.assert_close(merged.float(), want, atol=3e-2, rtol=0)


def test_start_aware_blend_is_unchanged_on_evenly_spaced_grid(monkeypatch):
    """No starts passed (encode path, old callers) == explicit evenly spaced starts, bit for bit."""
    canvas, packs, metas, tid_coord, grid_spec, common = _blend_setup(
        3, (2, 4), (16, 16), (12, 12), (3, 5)
    )
    torch.manual_seed(1)
    packs = [torch.randn_like(p.float()).to(torch.bfloat16) for p in packs]
    ex = _rank0_executor_cpu(monkeypatch, 3, packs)
    ex.set_source(torch.stack([p.reshape(p.shape[0], 15, 16, 16) for p in packs]))
    a = ex.gather_and_blend_tiles(packs[0], metas, grid_spec, tid_coord, **common)
    b = ex.gather_and_blend_tiles(
        packs[0],
        metas,
        grid_spec,
        tid_coord,
        row_starts=(0, 12),
        col_starts=(0, 12, 24, 36),
        **common,
    )
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    "latent_hw,tiles_px,patch,want_starts",
    [
        # A1: Cosmos3 30x52 latent, 256/224/192 -> 2x4 full tiles (diffusers: 3x5 with thin edges)
        ((30, 52), (256, 224, 192), 2, ((0, 14), (0, 12, 24, 36))),
        ((30, 52), (256, 224, 192), None, ((0, 14), (0, 12, 24, 36))),
        ((60, 104), (480, 416, 416), 2, ((0, 26, 30), (0, 26, 52, 74))),  # Wan2.2 default-ish
        ((8, 10), (256, 224, 192), 2, ((0,), (0,))),  # smaller than a tile: one tile
    ],
)
def test_tile_split_cuts_full_size_tiles_ending_at_the_edge(
    latent_hw, tiles_px, patch, want_starts
):
    from types import SimpleNamespace

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )

    tile, stride_h, stride_w = tiles_px
    ratio = 16 if patch else 8
    tile_l = tile // 16  # latent units; the pixel sizes below are given at Wan2.2's ratio 16
    fake = SimpleNamespace(
        spatial_compression_ratio=ratio,
        tile_sample_min_height=tile_l * ratio,
        tile_sample_min_width=tile_l * ratio,
        tile_sample_stride_height=stride_h // 16 * ratio,
        tile_sample_stride_width=stride_w // 16 * ratio,
        config=SimpleNamespace(patch_size=patch),
        dtype=torch.float32,
    )
    h, w = latent_hw
    z = torch.randn(1, 4, 3, h, w)
    tasks, spec = DistributedAutoencoderKLWan.tile_split(fake, z)
    rows, cols = want_starts
    assert spec.grid_shape == (len(rows), len(cols))
    out_ratio = ratio // patch if patch else ratio
    assert spec.tile_spec["row_starts"] == tuple(r * out_ratio for r in rows)
    assert spec.tile_spec["col_starts"] == tuple(c * out_ratio for c in cols)
    th, tw = min(h, tile_l), min(w, tile_l)
    for t in tasks:
        r, c = t.grid_coord
        assert len(t.tensor) == 3
        assert tuple(t.tensor[0].shape[-2:]) == (th, tw)  # every tile one shape
        torch.testing.assert_close(
            t.tensor[1][0, :, 0], z[0, :, 1, rows[r] : rows[r] + th, cols[c] : cols[c] + tw]
        )
    assert {t.grid_coord for t in tasks} == {
        (r, c) for r in range(len(rows)) for c in range(len(cols))
    }
