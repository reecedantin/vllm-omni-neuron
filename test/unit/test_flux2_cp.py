# SPDX-License-Identifier: Apache-2.0
"""Context parallelism and KV-head replication on CPU (gloo ranks), tiny-flux2.

* DiT under TP x CP (``ring_degree``): each rank holds a slice of the text and image tokens and
  all-gathers K/V; the gathered output must reproduce TP=1 / CP=1.
* Text encoder at a TP above its KV-head count: KV heads are replicated over ``tp // num_kv_heads``
  ranks; hidden states must reproduce TP=1.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from .test_flux2_components import _free_port, dit_inputs, rel, tiny_flux2  # noqa: F401

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _run(rank, world, tp, cp, port, weights, out_path, with_te):
    torch.set_num_threads(1)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni_neuron.diffusion.models.flux2.text_encoder import NeuronFlux2TextEncoder
    from vllm_omni_neuron.diffusion.models.flux2.transformer_flux2 import NeuronFlux2Transformer

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
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
    initialize_model_parallel(tensor_parallel_size=tp, ring_degree=cp, backend="gloo")

    out = {}
    if not with_te:
        dit = NeuronFlux2Transformer(weights, dtype=torch.float32)
        dit.load_weights(weights, "cpu")
        assert (dit.tp_size, dit.cp_size) == (tp, cp)
        x, c, img_ids, txt_ids, t, gd = dit_inputs(dit.cfg)
        with torch.no_grad():
            out["dit"] = dit(
                hidden_states=x,
                encoder_hidden_states=c,
                timestep=t,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=gd,
                return_dict=False,
            )[0]
    else:
        te = NeuronFlux2TextEncoder(weights, dtype=torch.float32)
        te.load_weights(weights, "cpu")
        g = torch.Generator().manual_seed(2)
        ids = torch.randint(100, 120000, (1, 32), generator=g)
        mask = torch.ones(1, 32, dtype=torch.long)
        mask[0, 21:] = 0
        with torch.no_grad():
            hs = te(input_ids=ids, attention_mask=mask).hidden_states
        out["te"] = torch.stack([hs[k] for k in (10, 20, 30)])
    if rank == 0:
        torch.save(out, out_path)


def _spawn(tp, cp, weights, path, with_te=False):
    world = tp * cp
    mp.spawn(
        _run,
        args=(world, tp, cp, _free_port(), weights, str(path), with_te),
        nprocs=world,
        join=True,
    )
    return torch.load(path)


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (1, 4)])
def test_dit_cp_matches_tp1(tp, cp, tiny_flux2, tmp_path):  # noqa: F811
    want = _spawn(1, 1, tiny_flux2, tmp_path / "ref.pt")["dit"]
    got = _spawn(tp, cp, tiny_flux2, tmp_path / f"tp{tp}cp{cp}.pt")["dit"]
    assert got.shape == want.shape
    assert rel(got, want) < 1e-5, rel(got, want)


def test_text_encoder_kv_replication(tiny_flux2, tmp_path):  # noqa: F811
    # tiny text encoder: 16 query heads, 8 KV heads -> TP=16 replicates every KV head on 2 ranks
    want = _spawn(1, 1, tiny_flux2, tmp_path / "ref.pt", with_te=True)["te"]
    got = _spawn(16, 1, tiny_flux2, tmp_path / "tp16.pt", with_te=True)["te"]
    assert rel(got, want) < 1e-5, rel(got, want)
