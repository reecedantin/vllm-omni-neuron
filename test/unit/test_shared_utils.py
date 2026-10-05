# SPDX-License-Identifier: Apache-2.0
"""Shared FastH3-derived utilities: int8 weight-only linear, modulation tables, prompt-embedding
cache, N-block graph splitting. CPU only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.layers.block_graphs import BlockGraphRunner, plan_groups
from vllm_omni_neuron.diffusion.layers.embedding_cache import (
    PromptEmbeddingCache,
    TextEncoderPhase,
    embedding_key,
    stack_embeddings,
)
from vllm_omni_neuron.diffusion.layers.modulation_tables import (
    HostModulation,
    ModulationTables,
    bake_linear_modulation,
)
from vllm_omni_neuron.diffusion.quantization.int8_weight_only import (
    Int8WeightOnlyLinear,
    quantize_linears_,
    quantize_weight_int8,
)

# ---------------------------------------------------------------- int8 weight-only


def test_quantize_weight_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(64, 48)
    q, s = quantize_weight_int8(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and q.abs().max() <= 127
    err = (q.float() * s[:, None] - w).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()  # within half a quantization step per channel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_linear_matches_dequantized(dtype):
    torch.manual_seed(0)
    lin = nn.Linear(96, 80).to(dtype)
    q = Int8WeightOnlyLinear.from_linear(lin)
    x = torch.randn(3, 7, 96, dtype=dtype)
    ref = F.linear(x.float(), q.dequantized_weight(), lin.bias.float())
    out = q(x)
    assert out.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    # and close to the unquantized layer (W8 error ~ 1% relative for gaussian weights)
    rel = (out.float() - lin(x).float()).norm() / lin(x).float().norm()
    assert rel < 0.02


def test_quantize_linears_filter_and_savings():
    model = nn.Sequential(
        nn.Linear(32, 64), nn.GELU(), nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 16))
    )
    model[2][0].adaln = True
    n, saved = quantize_linears_(model, include=lambda name, m: name != "2.0")
    assert n == 2
    assert isinstance(model[0], Int8WeightOnlyLinear) and isinstance(
        model[2][1], Int8WeightOnlyLinear
    )
    assert isinstance(model[2][0], nn.Linear)
    assert saved == (32 * 64 + 64 * 16) * 3 - (64 + 16) * 4  # fp32 -> int8 minus the fp32 scales
    assert model(torch.randn(2, 32)).shape == (2, 16)


def test_int8_linear_compiles_fullgraph():
    lin = Int8WeightOnlyLinear.from_linear(nn.Linear(16, 8))
    x = torch.randn(4, 16)
    out = torch.compile(lin, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(out, lin(x))


def test_int8_state_dict_roundtrip():
    lin = Int8WeightOnlyLinear.from_linear(nn.Linear(16, 8))
    other = Int8WeightOnlyLinear(16, 8, bias_dtype=torch.float32)
    other.load_state_dict(lin.state_dict())
    x = torch.randn(2, 16)
    torch.testing.assert_close(other(x), lin(x))


# ---------------------------------------------------------------- modulation tables


class _AdaBlock(nn.Module):
    """Minimal DiT block with an AdaLN projection: x * (1 + scale) + shift, gated MLP."""

    def __init__(self, dim=16, temb_dim=12):
        super().__init__()
        self.adaln = nn.Linear(temb_dim, 3 * dim)
        self.mlp = nn.Linear(dim, dim)
        self.host_mod = HostModulation(3)

    def modulation(self, temb):
        return self.adaln(F.silu(temb.float()).to(self.adaln.weight.dtype))

    def forward(self, x, temb=None, table=None):
        shift, scale, gate = (
            self.host_mod.split(table) if table is not None else self.modulation(temb).chunk(3, -1)
        )
        return x + gate * self.mlp(F.layer_norm(x, x.shape[-1:]) * (1 + scale) + shift)


def test_bake_matches_device_rounding_bf16():
    torch.manual_seed(0)
    blk = _AdaBlock().to(torch.bfloat16)
    temb = torch.randn(2, 12)
    baked = bake_linear_modulation(temb, blk.adaln.weight, blk.adaln.bias)
    # bf16 module path, as the device graph would compute it (fp32 accumulate, one rounding)
    ref = blk.modulation(temb)
    assert baked.dtype == torch.bfloat16
    torch.testing.assert_close(baked.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_tables_reproduce_block_outputs_exactly_fp32():
    torch.manual_seed(0)
    blocks = [_AdaBlock() for _ in range(4)]
    tabs = ModulationTables.from_linears([b.adaln for b in blocks], compute_dtype=torch.float32)
    temb = torch.randn(1, 12)
    x = torch.randn(1, 5, 16)
    t = tabs.tables(temb)
    assert t.shape == (4, 1, 48)
    ref = out = x
    for i, b in enumerate(blocks):
        ref = b(ref, temb=temb)
        out = b(out, table=t[i])
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_tables_cache_memory_and_disk(tmp_path):
    calls = []

    def layer_fn(i, temb):
        calls.append(i)
        return temb * (i + 1)

    tabs = ModulationTables(layer_fn, 3, fingerprint="m1", cache_dir=str(tmp_path))
    temb = torch.randn(2, 4)
    a = tabs.tables(temb)
    b = tabs.tables(temb.clone())
    assert a is b and calls == [0, 1, 2]
    fresh = ModulationTables(layer_fn, 3, fingerprint="m1", cache_dir=str(tmp_path))
    torch.testing.assert_close(fresh.tables(temb), a)
    assert calls == [0, 1, 2]  # served from disk
    other = ModulationTables(layer_fn, 3, fingerprint="m2", cache_dir=str(tmp_path))
    other.tables(temb)
    assert calls == [0, 1, 2] * 2  # different weights fingerprint: miss
    tabs.tables(temb + 1)  # different timestep: miss
    assert len(calls) == 9


def test_tables_from_checkpoint_streams_layers():
    torch.manual_seed(0)
    lins = [nn.Linear(6, 8) for _ in range(3)]
    sd = {
        f"blocks.{i}.ada.{k}": v for i, lin in enumerate(lins) for k, v in lin.state_dict().items()
    }
    loaded = []

    def get_tensor(name):
        loaded.append(name)
        return sd[name]

    tabs = ModulationTables.from_checkpoint(
        get_tensor, "blocks.{}.ada", 3, compute_dtype=torch.float32
    )
    temb = torch.randn(1, 6)
    ref = torch.stack(
        [
            bake_linear_modulation(temb, lin.weight, lin.bias, compute_dtype=torch.float32)
            for lin in lins
        ]
    )
    torch.testing.assert_close(tabs.tables(temb), ref)
    assert loaded == [f"blocks.{i}.ada.{k}" for i in range(3) for k in ("weight", "bias")]


def test_device_tables_are_per_layer_and_cached():
    tabs = ModulationTables(lambda i, t: t + i, 4)
    temb = torch.zeros(1, 3)
    d = tabs.device_tables(temb, "cpu")
    assert isinstance(d, list) and len(d) == 4 and d[2].shape == (1, 3)
    assert tabs.device_tables(temb, "cpu") is d


def test_host_modulation_requires_table():
    m = HostModulation(2)
    with pytest.raises(RuntimeError):
        m()
    m.table = torch.arange(8.0).view(1, 8)
    a, b = m()
    assert a.shape == (1, 4) and b[0, 0] == 4


# ---------------------------------------------------------------- prompt-embedding cache


def test_embedding_key_distinguishes_settings():
    k = embedding_key("enc", "a cat")
    assert k == embedding_key("enc", "a cat")
    assert k != embedding_key("enc", "a dog")
    assert k != embedding_key("enc2", "a cat")
    assert k != embedding_key("enc", "a cat", {"max_len": 512})
    assert embedding_key("enc", [1, 2]) != embedding_key("enc", [2, 1])


def test_cache_lru_bounds():
    c = PromptEmbeddingCache(max_entries=2)
    for i in range(3):
        c.put(str(i), torch.full((1, 4), float(i)))
    assert len(c) == 2 and c.get("0") is None and c.get("2") is not None
    c = PromptEmbeddingCache(max_entries=10, max_bytes=40)
    c.put("a", torch.zeros(1, 8))  # 32 bytes
    c.put("b", torch.zeros(1, 8))
    assert len(c) == 1 and c.get("b") is not None


def test_cache_disk_layer_and_tuples(tmp_path):
    c = PromptEmbeddingCache(cache_dir=str(tmp_path))
    c.put("k", (torch.ones(1, 3), torch.tensor([[1, 0]])))
    c2 = PromptEmbeddingCache(cache_dir=str(tmp_path))
    got = c2.get("k")
    assert isinstance(got, tuple) and torch.equal(got[1], torch.tensor([[1, 0]]))


def test_text_encoder_phase_batches_misses_and_releases():
    seen, state = [], {"loaded": True, "loads": 0}

    def encode(prompts):
        assert state["loaded"]
        seen.append(list(prompts))
        emb = torch.stack([torch.full((2, 4), float(len(p))) for p in prompts])
        return emb, torch.ones(len(prompts), 2, dtype=torch.long)

    def load():
        state["loaded"] = True
        state["loads"] += 1

    def release():
        state["loaded"] = False

    phase = TextEncoderPhase(
        encode, "enc@bf16", settings={"max_len": 2}, load_fn=load, release_fn=release
    )
    phase.precompute(["a", "bb", "a"], release_after=True)
    assert seen == [["a", "bb"]] and not state["loaded"]
    out = phase.encode(["bb", "a"])  # all hits: encoder stays released
    assert seen == [["a", "bb"]] and not state["loaded"]
    emb, mask = stack_embeddings(out)
    assert (
        emb.shape == (2, 2, 4) and emb[0, 0, 0] == 2 and emb[1, 0, 0] == 1 and mask.shape == (2, 2)
    )
    phase.encode(["ccc"])  # miss: lazily reloaded
    assert state["loads"] == 1 and seen[-1] == ["ccc"]


def test_cached_embeddings_live_on_host():
    phase = TextEncoderPhase(lambda ps: torch.zeros(len(ps), 3, requires_grad=True) * 2, "e")
    (e,) = phase.encode(["x"])
    assert e.device.type == "cpu" and not e.requires_grad


# ---------------------------------------------------------------- N-block graph splitting


def test_plan_groups():
    assert plan_groups(12, 5) == [(0, 5), (5, 10), (10, 12)]
    assert plan_groups(10, 5) == [(0, 5), (5, 10)]
    assert plan_groups(4, 0) == [(0, 4)]
    assert plan_groups(4, 8) == [(0, 4)]


def _counting_compile():
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    return graphs, (lambda f: torch.compile(f, backend=backend, fullgraph=True, dynamic=False))


@pytest.mark.parametrize("n,group,expect_graphs", [(12, 5, 2), (10, 5, 1), (6, 0, 1)])
def test_block_runner_shares_one_graph(n, group, expect_graphs):
    torch.manual_seed(0)
    torch._dynamo.reset()
    blocks = nn.ModuleList(_AdaBlock() for _ in range(n))
    tabs = ModulationTables.from_linears([b.adaln for b in blocks], compute_dtype=torch.float32)
    temb = torch.randn(1, 12)
    x = torch.randn(1, 5, 16)
    ref = x
    for b in blocks:
        ref = b(ref, temb=temb)

    graphs, compile_fn = _counting_compile()
    runner = BlockGraphRunner(
        blocks,
        group,
        compile_fn=compile_fn,
        block_call=lambda blk, c, sa, la, kw: blk(c, table=la[0]),
    )
    assert runner.num_graphs == expect_graphs
    per_layer = [(t,) for t in tabs.device_tables(temb, "cpu")]
    with (
        torch.no_grad()
    ):  # inference: a chunk's output must not differ in requires_grad from its input
        out = runner(x, per_layer=per_layer)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        assert len(graphs) == expect_graphs
        out2 = runner(x + 1, per_layer=per_layer)  # warm: no new graphs
    assert len(graphs) == expect_graphs and out2.shape == out.shape


def test_block_runner_tracks_weight_changes():
    torch._dynamo.reset()
    blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
    runner = BlockGraphRunner(blocks, 2, compile_fn=_counting_compile()[1])
    x = torch.randn(2, 4)
    with torch.no_grad():
        for b in blocks:
            b.weight.mul_(0.5)  # in place: same tensors, new values
        out = runner(x)
        ref = x
        for b in blocks:
            ref = b(ref)
    torch.testing.assert_close(out, ref)


def test_block_runner_rejects_mixed_blocks_and_bad_per_layer():
    with pytest.raises(ValueError):
        BlockGraphRunner([nn.Linear(4, 4), nn.Linear(4, 8)], 1)
    runner = BlockGraphRunner([nn.Linear(4, 4)] * 2, 1)
    with pytest.raises(ValueError):
        runner(torch.zeros(1, 4), per_layer=[()])


def test_block_runner_does_not_own_blocks():
    blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
    runner = BlockGraphRunner(blocks, 1)
    assert list(runner.state_dict()) == []
