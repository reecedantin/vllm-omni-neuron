# SPDX-License-Identifier: Apache-2.0
"""Accuracy tier 1 (CPU): components in isolation.

* neighborhood attention (gather, banded and the shared tiled forms) against NATTEN's dense masked
  definition, and the device encoder path (tiled attention, host-prepared constants) against the
  host path;
* this package's video VAE against the upstream VAE on the same weights (fp32);
* tensor-parallel head sharding of the DiT (two gloo ranks) against TP1.

The device half of tier 1 (fp32 CPU / bf16 CPU / bf16 Neuron three-way on one DiT forward) is
``test/neuron/test_flux3_action_dit_device.py``.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni_neuron.diffusion.models.flux3_action import video_vae as video_vae_mod
from vllm_omni_neuron.diffusion.models.flux3_action.neighborhood import (
    neighborhood_attention,
    neighborhood_attention_reference,
)
from vllm_omni_neuron.diffusion.models.flux3_action.video_vae import (
    VideoVAE,
    _na_tiled,
    na_device_consts,
    neighborhood_attention_banded,
)

from .test_flux3_action_utils.fixtures import observation, rel, tiny, upstream  # noqa: F401
from .test_flux3_action_utils.reference import load_upstream_vae

CASES = [
    ((2, 9, 11, 3, 16), [5, 5], [False, False]),
    ((1, 7, 6, 8, 2, 16), [5, 5, 5], [True, False, False]),
    ((1, 3, 6, 7, 2, 8), [5, 5, 5], [True, False, False]),
    ((1, 17, 23, 4, 16), [5, 5], [False, False]),
]


@pytest.mark.parametrize("shape,kernel,causal", CASES)
def test_neighborhood_attention_matches_definition(shape, kernel, causal):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(*shape, generator=g) for _ in range(3))
    ref = neighborhood_attention_reference(q, k, v, kernel, causal)
    assert (neighborhood_attention(q, k, v, kernel, causal) - ref).abs().max().item() < 1e-5
    assert (neighborhood_attention_banded(q, k, v, kernel, causal) - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("shape", [(1, 17, 23, 4, 16), (1, 34, 46, 2, 16), (1, 9, 11, 3, 16)])
def test_tiled_neighborhood_attention_with_prepared_consts(shape):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(*shape, generator=g) for _ in range(3))
    kernel, causal = [5, 5], [False, False]
    ref = neighborhood_attention_reference(q, k, v, kernel, causal)
    consts = na_device_consts(shape[1:3], kernel, causal, q.dtype)
    assert (_na_tiled(q, k, v, kernel, causal, consts) - ref).abs().max().item() < 1e-5
    assert (_na_tiled(q, k, v, kernel, causal, None) - ref).abs().max().item() < 1e-5


def test_video_vae_device_path_matches_host_path(tiny, monkeypatch):  # noqa: F811
    """The encoder as the NeuronCore runs it (tiled attention, prepared per-layer constants,
    host-side normalization) equals the host encoder on CPU."""
    vae_path = os.path.join(tiny["base"], "video_vae.safetensors")
    vae = VideoVAE.from_file(vae_path).float()
    x = torch.rand(1, 3, 544, 736, generator=torch.Generator().manual_seed(0)) * 2 - 1
    host = vae.encode_frame(x)
    grids = vae.prepare_device_attention((544, 736), "cpu", torch.float32)
    assert grids and grids[0] == (136, 184) and grids[-1] == (17, 23)
    monkeypatch.setattr(video_vae_mod, "NA_IMPL", "tiled")
    dev = vae.normalize(vae.encode_frame_mu(x))
    assert rel(dev, host) < 1e-5
    # piecewise compile (one graph per distinct piece, weights as inputs): same result, and every
    # repeated block replays its piece's graph instead of tracing a new one
    graphs = []

    def counting_backend(gm, example_inputs, **_options):
        graphs.append(gm)
        return gm.forward

    vae.compile_encoder(counting_backend)
    for _ in range(2):
        assert rel(vae.normalize(vae.encode_frame_mu(x)), host) < 1e-5
    n_pieces = len(vae._device_pieces)
    assert len(graphs) == len(vae._device_graphs) < n_pieces, (len(graphs), n_pieces)


def test_video_vae_matches_upstream_fp32(tiny, upstream):  # noqa: F811
    vae_path = os.path.join(tiny["base"], "video_vae.safetensors")
    theirs = load_upstream_vae(tiny["base"])
    ours = VideoVAE.from_file(vae_path)
    theirs.module.float()
    ours.float()
    x = torch.rand(1, 3, 544, 736, generator=torch.Generator().manual_seed(0)) * 2 - 1
    lat_theirs = theirs.encode_frame(x)
    lat_ours = ours.encode_frame(x)
    assert rel(lat_ours, lat_theirs) < 1e-4
    z = lat_ours[:, :, :, :17, :20].repeat(1, 1, 2, 1, 1)
    assert rel(ours.decode(z), theirs.module.decode(z)) < 1e-4


def _tp_forward(policy_dir, tp, rank, group):
    from vllm_omni_neuron.diffusion.models.flux3_action.dit import (
        DiTDims,
        Flux3ActionDiT,
        load_policy_config,
    )

    dims = DiTDims.from_policy_config(load_policy_config(policy_dir))
    dit = Flux3ActionDiT(dims, tp_size=tp, tp_rank=rank, tp_group=group)
    dit.load(os.path.join(policy_dir, "model.safetensors"))
    g = torch.Generator().manual_seed(0)
    lt, lv, la = 80, 12, 4

    def ids(n, axis):
        x = torch.zeros(1, n, 4, dtype=torch.long)
        x[0, :, axis] = torch.arange(n)
        return x

    req = dit.prepare(
        torch.randn(1, lt, dims.context_in_dim, generator=g),
        ids(lt, 3),
        video_ids=ids(lv, 1),
        video_cond=torch.randn(1, 6, 96, generator=g),
        video_cond_ids=ids(6, 2),
        action_ids=ids(la, 0),
        action_cond=torch.randn(1, 1, 8, generator=g),
        action_cond_ids=ids(1, 0),
    )
    step = dit.prepare_step(req, 0.7, 0.7)
    return dit.forward(
        req, step, torch.randn(1, lv, 96, generator=g), torch.randn(1, la, 8, generator=g)
    )


def _tp_worker(rank, world, port, policy_dir, q):
    import torch.distributed as dist

    os.environ["VLLM_NEURON_CPU_MODE"] = "1"  # gloo on CPU: no Neuron replica-group registration
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    q.put(_tp_forward(policy_dir, world, rank, dist.group.WORLD))
    dist.destroy_process_group()


def test_dit_tp2_matches_tp1(tiny):  # noqa: F811
    """Head sharding: two gloo ranks with sharded weights and one all-reduce per block == TP1."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29000 + os.getpid() % 1000
    procs = [ctx.Process(target=_tp_worker, args=(r, 2, port, tiny["droid"], q)) for r in range(2)]
    for p in procs:
        p.start()
    res = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join(60)
    ref = _tp_forward(tiny["droid"], 1, 0, None)
    for r in res:
        for a, b in zip(r, ref, strict=True):
            assert rel(a, b) < 2e-2


CAPTIONS = ["put the screwdriver in the box", "", "pick up the red block and place it on the plate"]


def _text_encoder(tiny):  # noqa: F811
    from vllm_omni_neuron.diffusion.models.flux3_action.text_encoder import Qwen3VLTextEncoder

    return Qwen3VLTextEncoder(os.path.join(tiny["base"], "text_encoder"), dtype=torch.float32)


def test_text_encoder_layer_stack_matches_transformers(tiny):  # noqa: F811
    """The device path (layer inputs captured from transformers, our decoder-layer math, stacked
    hidden states 4..32) equals transformers' own forward, run here on the CPU."""
    enc = _text_encoder(tiny)
    ref = enc.encode(CAPTIONS)
    enc.to_device("cpu")
    got = enc.encode(CAPTIONS)
    assert len(enc.lm.layers) == 1  # host keeps only layer 0's module (the input hook)
    for a, b in zip(got, ref, strict=True):
        assert a.shape == b.shape
        assert rel(a, b) < 1e-5


def _text_tp_worker(rank, world, port, tiny, q):  # noqa: F811
    import torch.distributed as dist

    os.environ["VLLM_NEURON_CPU_MODE"] = "1"
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    enc = _text_encoder(tiny)
    enc.to_device("cpu", tp_size=world, tp_rank=rank, tp_group=dist.group.WORLD)
    # numpy pickles by value: a torch tensor would travel as shared memory that vanishes with the
    # worker process
    q.put((rank, [t.numpy() for t in enc.encode(CAPTIONS)]))
    dist.destroy_process_group()


def test_text_encoder_tp2_matches_tp1(tiny):  # noqa: F811
    """Head sharding of the caption encoder (2 KV heads over two gloo ranks, two all-reduces per
    layer) matches the unsharded encoder on every rank."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 28000 + os.getpid() % 1000
    procs = [ctx.Process(target=_text_tp_worker, args=(r, 2, port, tiny, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join(60)
    ref = _text_encoder(tiny).encode(CAPTIONS)
    for _rank, out in res:
        for a, b in zip(out, ref, strict=True):
            assert rel(torch.from_numpy(a), b) < 1e-4
