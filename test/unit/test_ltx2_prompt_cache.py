# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the LTX-2.5 served pipeline's text-conditioning sharing across TP ranks.

``NeuronLTX25Pipeline._shared_text_conditioning`` must run the (expensive) text encoder and
connectors on TP rank 0 only, give every rank identical connector outputs, and skip the encoder,
the connectors and the broadcast once every rank has the prompt cached. Two gloo ranks, the
encoder and the connectors replaced by counting stubs.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

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

    from vllm_omni_neuron.diffusion.layers.embedding_cache import PromptEmbeddingCache
    from vllm_omni_neuron.diffusion.models.ltx2.pipeline_ltx25 import NeuronLTX25Pipeline

    calls, conn_calls = [], []

    def encode_prompt(prompt, **_):
        calls.append(prompt)
        g = torch.Generator().manual_seed(len(prompt))
        return (
            torch.randn(1, 8, 16, generator=g).bfloat16(),
            torch.ones(1, 8, dtype=torch.long),
            None,
            None,
        )

    def connectors(embeds, mask, padding_side="left"):
        conn_calls.append(padding_side)
        return embeds * 2, embeds[..., :4] + 1, mask

    p = SimpleNamespace(
        _emb_cache=PromptEmbeddingCache(),
        _encoder_id="stub",
        _pipe=SimpleNamespace(encode_prompt=encode_prompt, tokenizer=None),
        _connectors_forward=connectors,
        host_threads=4,
    )
    for name in (
        "_cached_text_conditioning",
        "_shared_text_conditioning",
        "_run_connectors",
        "_encode_prompt_embeds",
    ):
        setattr(p, name, getattr(NeuronLTX25Pipeline, name).__get__(p))

    c1, s1 = p._shared_text_conditioning("a red fox")
    c2, s2 = p._shared_text_conditioning("a red fox")
    c3, s3 = p._shared_text_conditioning("a blue whale")
    e1, a1, m1 = c1
    e2, e3 = c2[0], c3[0]
    torch.save(
        {
            "rank": rank,
            "calls": calls,
            "conn_calls": conn_calls,
            "a1": a1,
            "status": [s1, s2, s3],
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "m1": m1,
        },
        f"{out_path}.{rank}",
    )


def test_text_encoder_and_connectors_run_on_rank0_only_and_ranks_agree(tmp_path):
    out = str(tmp_path / "res")
    mp.start_processes(_worker, args=(2, _port(), out), nprocs=2, join=True, start_method="spawn")
    r0, r1 = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    assert r0["calls"] == ["a red fox", "a blue whale"]
    assert r1["calls"] == []
    assert r0["conn_calls"] == ["left", "left"] and r1["conn_calls"] == []
    assert r0["status"] == ["miss-broadcast", "hit", "miss-broadcast"]
    assert r1["status"] == ["miss-broadcast", "hit", "miss-broadcast"]
    for k in ("e1", "e2", "e3", "m1", "a1"):
        assert torch.equal(r0[k], r1[k]), k
    assert r0["e1"].dtype == torch.bfloat16 and torch.equal(r0["e1"], r0["e2"])
