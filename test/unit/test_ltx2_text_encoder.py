# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the device Gemma text tower (``models/ltx2/text_encoder.py``) vs transformers'
``Gemma4UnifiedForConditionalGeneration`` (random tiny weights, LTX-2.5's layer layout: sliding
layers with GQA, global layers with one shared K=V head and proportional partial RoPE).

* fp32, TP=1: the 49-state hidden stack of the valid (non-padded) tokens matches diffusers'
  ``_get_gemma_prompt_embeds`` for every prompt bucket; padded rows are zero.
* fp32, TP=2 on two gloo ranks: matches TP=1 (sharded heads/MLP, replicated global KV head).
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

MAX_LEN = 64
TEXT = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    head_dim=16,
    global_head_dim=32,
    intermediate_size=96,
    num_hidden_layers=6,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    sliding_window=1024,
    attention_k_eq_v=True,
    vocab_size=300,
    rms_norm_eps=1e-6,
    hidden_activation="gelu_pytorch_tanh",
    rope_parameters={
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
)


@pytest.fixture(scope="module")
def tiny_te(tmp_path_factory):
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

    out = str(tmp_path_factory.mktemp("tiny-gemma"))
    torch.manual_seed(0)
    m = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=dict(TEXT)))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "layernorm" in n or "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "layer_scalar" not in n:
                p.copy_(0.05 * torch.randn_like(p))
        for layer in m.model.language_model.layers:
            layer.layer_scalar.fill_(0.9)
    m.save_pretrained(out, safe_serialization=True)
    return out


def _inputs(n_valid: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(1, MAX_LEN, dtype=torch.long)
    mask = torch.zeros(1, MAX_LEN, dtype=torch.long)
    ids[:, MAX_LEN - n_valid :] = torch.randint(3, TEXT["vocab_size"], (n_valid,), generator=g)
    mask[:, MAX_LEN - n_valid :] = 1
    return ids, mask


def _reference(path, ids, mask):
    """diffusers ``_get_gemma_prompt_embeds``' stacking of transformers' hidden states."""
    from transformers import Gemma4UnifiedForConditionalGeneration

    m = Gemma4UnifiedForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
    with torch.no_grad():
        hs = m.eval()(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    return torch.stack(hs, dim=-1).flatten(2, 3)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("n_valid,buckets", [(5, (8, 16, 64)), (20, (8, 16, 64)), (40, (64,))])
def test_text_tower_matches_transformers(tiny_te, n_valid, buckets):
    from vllm_omni_neuron.diffusion.models.ltx2.text_encoder import NeuronGemmaTextEncoder

    ids, mask = _inputs(n_valid)
    ref = _reference(tiny_te, ids, mask)
    enc = NeuronGemmaTextEncoder(tiny_te, "cpu", dtype=torch.float32, buckets=buckets)
    enc.max_len = MAX_LEN
    got = enc.hidden_states(ids, mask)
    assert got.shape == ref.shape
    v = mask[0].bool()
    assert _rel(got[0, v], ref[0, v]) < 1e-5, _rel(got[0, v], ref[0, v])
    assert float(got[0, ~v].abs().max()) == 0.0


def _tp_worker(rank, world, port, path, out):
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
    from vllm_omni_neuron.diffusion.models.ltx2.text_encoder import NeuronGemmaTextEncoder

    enc = NeuronGemmaTextEncoder(path, "cpu", dtype=torch.float32, buckets=(16, 64))
    enc.max_len = MAX_LEN
    torch.save(enc.hidden_states(*_inputs(12)), f"{out}.{rank}")


def test_text_tower_tp2_matches_tp1(tiny_te, tmp_path):
    from vllm_omni_neuron.diffusion.models.ltx2.text_encoder import NeuronGemmaTextEncoder

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "tp2")
    mp.start_processes(
        _tp_worker, args=(2, port, tiny_te, out), nprocs=2, join=True, start_method="spawn"
    )
    enc = NeuronGemmaTextEncoder(tiny_te, "cpu", dtype=torch.float32, buckets=(16, 64))
    enc.max_len = MAX_LEN
    ref = enc.hidden_states(*_inputs(12))
    for r in range(2):
        assert _rel(torch.load(f"{out}.{r}"), ref) < 1e-5
