# SPDX-License-Identifier: Apache-2.0
"""TP on CPU (gloo ranks): the sharded DiT and text encoder must reproduce TP=1 (tiny-flux2).

Validates the segment loaders (fused gate/up, the single block's ``[q|k|v|gate|up]`` input and
``[attn|mlp]`` output projections), head grouping per shard and the row-parallel all-reduces;
the device run then only adds the Neuron collective lowering.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from .test_flux2_components import _free_port, dit_inputs, rel, tiny_flux2  # noqa: F401

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _run(rank, world, port, weights, out_path):
    torch.set_num_threads(2)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_omni_neuron.diffusion.models.flux2.text_encoder import NeuronFlux2TextEncoder
    from vllm_omni_neuron.diffusion.models.flux2.transformer_flux2 import NeuronFlux2Transformer

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(world, 1)

    dit = NeuronFlux2Transformer(weights, dtype=torch.float32)
    dit.load_weights(weights, "cpu")
    assert dit.tp_size == world
    x, c, img_ids, txt_ids, t, gd = dit_inputs(dit.cfg)
    te = NeuronFlux2TextEncoder(weights, dtype=torch.float32)
    te.load_weights(weights, "cpu")
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(100, 120000, (1, 32), generator=g)
    mask = torch.ones(1, 32, dtype=torch.long)
    mask[0, 21:] = 0
    with torch.no_grad():
        out = dit(
            hidden_states=x,
            encoder_hidden_states=c,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=gd,
            return_dict=False,
        )[0]
        hs = te(input_ids=ids, attention_mask=mask).hidden_states
    if rank == 0:
        torch.save({"dit": out, "te": torch.stack([hs[k] for k in (10, 20, 30)])}, out_path)


@pytest.mark.parametrize("world", [2, 8])
def test_tp_matches_tp1(world, tiny_flux2, tmp_path):  # noqa: F811
    outs = {}
    for w in (1, world):
        path = tmp_path / f"tp{w}.pt"
        mp.spawn(_run, args=(w, _free_port(), tiny_flux2, str(path)), nprocs=w, join=True)
        outs[w] = torch.load(path)
    for key in ("dit", "te"):
        assert rel(outs[world][key], outs[1][key]) < 1e-5, (
            key,
            rel(outs[world][key], outs[1][key]),
        )
