# SPDX-License-Identifier: Apache-2.0
"""CPU check of the host-gathered tile-parallel decode used for long clips
(``NeuronEdgeVae._host_pp_decode``, opt-in via ``COSMOS3_EDGE_VAE_HOST_GATHER_T``, point-to-point tile gather): 2 and 3 gloo ranks deal the decode tiles of a random latent and
rank 0 merges them; the result must equal the single-process tiled decode, and the other rank must
get a zero tensor of the output shape. Needs a Cosmos3 checkout (``COSMOS3_TINY_SRC``) for the VAE
config; the weights are random (shape-only synthesis), so this checks the plumbing, not quality."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

SRC = os.environ.get("COSMOS3_TINY_SRC", "")


def _vae(path):
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import NeuronEdgeVae

    edge = NeuronEdgeVae.from_pretrained(path, torch_dtype=torch.float32)
    edge._ensure_decoder()  # eager decoder graphs on CPU
    v = edge.vae
    v.use_tiling = True
    v.tile_sample_min_height = v.tile_sample_min_width = 128
    v.tile_sample_stride_height = v.tile_sample_stride_width = 96
    return edge


def _z():
    return torch.randn(1, 48, 3, 12, 16, generator=torch.Generator().manual_seed(0))


def _worker(rank, world, port, path, out):
    import torch.distributed as dist

    torch.set_num_threads(
        1
    )  # same intra-op split as the reference: CPU conv sums are thread-count dependent
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    edge = _vae(path)
    edge._host_pp_group = dist.new_group(list(range(world)), backend="gloo")
    with torch.no_grad():
        out[rank] = edge._host_pp_decode(_z()).clone()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def vae_dir(tmp_path_factory):
    if not SRC or not os.path.isdir(os.path.join(SRC, "vae")):
        pytest.skip("set COSMOS3_TINY_SRC to a local Cosmos3 checkout")
    from vllm_omni_neuron import tiny_models as tm

    out = str(tmp_path_factory.mktemp("vae"))
    sdir = os.path.join(SRC, "vae")
    cfg = tm._load_json(os.path.join(sdir, "config.json"))
    tm.synthesize_component(
        sdir, os.path.join(out, "vae"), 2, tm.shrink_layer_counts(cfg, 2)[1], name="vae"
    )
    import shutil

    shutil.copy2(os.path.join(sdir, "config.json"), os.path.join(out, "vae", "config.json"))
    return out


def test_host_gather_decode_matches_single_process(vae_dir):
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    edge = _vae(vae_dir)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # earlier tests may change the thread count; match the workers
    try:
        with torch.no_grad():
            ref = NeuronAutoencoderKLWan.tiled_decode(edge.vae, _z(), return_dict=False)[0]
    finally:
        torch.set_num_threads(threads)
    for world in (2, 3):  # 4 tiles: even and uneven dealing
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        out = mp.Manager().dict()
        mp.spawn(_worker, args=(world, port, vae_dir, out), nprocs=world, join=True)
        assert torch.equal(out[0], ref.to(out[0].dtype)), world
        for r in range(1, world):
            assert out[r].shape == ref.shape and not out[r].any()
