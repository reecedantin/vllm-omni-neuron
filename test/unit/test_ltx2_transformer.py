# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Neuron LTX-2.5 DiT vs the vendored upstream transformer (tiny weights).

* TP=1 fp32: :class:`NeuronLTX2Transformer` (host fp32 conditioning + head/chunk/tail split,
  chunk graphs fed block weights as inputs) must reproduce upstream's forward to fp32 noise.
* TP=4 fp32 on 4 gloo ranks: head-sharded attention, the across-heads QK-norm all-reduce and
  the row-parallel projections must reproduce TP=1.
* bf16: the port stays inside the CPU bf16-vs-fp32 noise band of upstream.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

GEOM = dict(num_frames=3, height=4, width=6, audio_frames=20, text=16)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from .test_ltx2_tiny import build_transformer

    out = tmp_path_factory.mktemp("tiny-ltx25")
    m = build_transformer(os.environ.get("LTX25_WEIGHTS"))
    m.to(torch.float32).save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)
    return str(out)


def make_inputs(cfg_dict: dict, batch: int = 2, seed: int = 0, sigma: float = 0.6):
    g = torch.Generator().manual_seed(seed)
    f, h, w = GEOM["num_frames"], GEOM["height"], GEOM["width"]
    n, na, st = f * h * w, GEOM["audio_frames"], GEOM["text"]
    d = cfg_dict["num_attention_heads"] * cfg_dict["attention_head_dim"]
    da = cfg_dict["audio_num_attention_heads"] * cfg_dict["audio_attention_head_dim"]
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import first_frame_keyframes_mask

    ts = torch.full((batch,), sigma * 1000.0)
    return dict(
        hidden_states=torch.randn(batch, n, cfg_dict["in_channels"], generator=g),
        audio_hidden_states=torch.randn(batch, na, cfg_dict["audio_in_channels"], generator=g),
        encoder_hidden_states=torch.randn(batch, st, d, generator=g),
        audio_encoder_hidden_states=torch.randn(batch, st, da, generator=g),
        timestep=ts,
        sigma=ts,
        num_frames=f,
        height=h,
        width=w,
        fps=24.0,
        audio_num_frames=na,
        use_cross_timestep=True,
        video_keyframes_mask=first_frame_keyframes_mask(batch, n, f),
    )


def reference(tiny: str, inputs: dict, dtype=torch.float32):
    from vllm_omni_neuron.diffusion.models.ltx2._vendor.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )

    ref = LTX2VideoTransformer3DModel.from_pretrained(
        os.path.join(tiny, "transformer"), torch_dtype=dtype
    ).eval()
    kw = {
        k: (
            v.to(dtype)
            if torch.is_tensor(v) and v.is_floating_point() and k != "timestep" and k != "sigma"
            else v
        )
        for k, v in inputs.items()
    }
    with torch.no_grad():
        return ref(**kw, return_dict=False)


def port(tiny: str, inputs: dict, dtype=torch.float32, blocks_per_graph: int = 2):
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import NeuronLTX2Transformer

    m = NeuronLTX2Transformer.from_dir(
        os.path.join(tiny, "transformer"), dtype=dtype, blocks_per_graph=blocks_per_graph
    )
    with torch.no_grad():
        return m(**inputs)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _cfg(tiny):
    import json

    with open(os.path.join(tiny, "transformer", "config.json")) as f:
        return json.load(f)


def test_port_matches_upstream_fp32(tiny_dir):
    inputs = make_inputs(_cfg(tiny_dir))
    rv, ra = reference(tiny_dir, inputs)
    for k in (1, 2, 4):
        pv, pa = port(tiny_dir, inputs, blocks_per_graph=k)
        assert _rel(pv, rv) < 1e-5, (k, _rel(pv, rv))
        assert _rel(pa, ra) < 1e-5, (k, _rel(pa, ra))


def test_keyframe_marker_matters(tiny_dir):
    """The LTX-2.5 keyframe embedding is live (guards against silently dropping the mask)."""
    inputs = make_inputs(_cfg(tiny_dir))
    pv, _ = port(tiny_dir, inputs)
    inputs["video_keyframes_mask"] = torch.zeros_like(inputs["video_keyframes_mask"])
    pv0, _ = port(tiny_dir, inputs)
    assert _rel(pv0, pv) > 1e-4


def test_port_bf16_within_noise(tiny_dir):
    inputs = make_inputs(_cfg(tiny_dir))
    rv, ra = reference(tiny_dir, inputs)
    bv, ba = reference(tiny_dir, inputs, torch.bfloat16)
    pv, pa = port(tiny_dir, inputs, torch.bfloat16)
    noise_v, noise_a = _rel(bv, rv), _rel(ba, ra)
    assert _rel(pv, rv) <= max(2.0 * noise_v, 0.02), (_rel(pv, rv), noise_v)
    assert _rel(pa, ra) <= max(2.0 * noise_a, 0.02), (_rel(pa, ra), noise_a)


def _tp_worker(rank, world, port_, tiny, out_path):
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
        distributed_init_method=f"tcp://127.0.0.1:{port_}",
        backend="gloo",
    )
    initialize_model_parallel(world, 1)
    pv, pa = port(tiny, make_inputs(_cfg(tiny)))
    if rank == 0:
        torch.save({"v": pv, "a": pa}, out_path)


def test_port_tp4_matches_tp1(tiny_dir, tmp_path):
    out = str(tmp_path / "tp4.pt")
    mp.start_processes(
        _tp_worker, args=(4, _port(), tiny_dir, out), nprocs=4, join=True, start_method="spawn"
    )
    got = torch.load(out)
    pv, pa = port(tiny_dir, make_inputs(_cfg(tiny_dir)))
    assert _rel(got["v"], pv) < 1e-5, _rel(got["v"], pv)
    assert _rel(got["a"], pa) < 1e-5, _rel(got["a"], pa)


def test_step_cache_reuses_rope_and_text_only_while_inputs_match(tiny_dir, monkeypatch):
    """Per-request device copies (RoPE tables, text embeddings) are reused across steps and
    rebuilt when the geometry or the text changes; results equal the uncached forward."""
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import NeuronLTX2Transformer

    m = NeuronLTX2Transformer.from_dir(os.path.join(tiny_dir, "transformer"), blocks_per_graph=2)
    built = []
    orig = m.host.rope_tables

    def spy(*a, **k):
        built.append(1)
        return orig(*a, **k)

    m.host.rope_tables = spy
    cfg = _cfg(tiny_dir)
    a, b = make_inputs(cfg, sigma=0.6), make_inputs(cfg, sigma=0.3)
    b["encoder_hidden_states"] = a["encoder_hidden_states"].clone()  # same request, next step
    b["audio_encoder_hidden_states"] = a["audio_encoder_hidden_states"].clone()
    with torch.no_grad():
        outs = [m(**a), m(**b)]
    assert len(built) == 1
    c = make_inputs(cfg, seed=1)  # new text: new text embeddings, same geometry
    with torch.no_grad():
        outs.append(m(**c))
    assert len(built) == 1
    monkeypatch.setenv("LTX2_STEP_CACHE", "0")
    with torch.no_grad():
        ref = [m(**a), m(**b), m(**c)]
    for (pv, pa), (rv, ra) in zip(outs, ref, strict=True):
        assert torch.equal(pv, rv) and torch.equal(pa, ra)


def _tp_cp_worker(rank, world, port_, tiny, tp, cp, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port_}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_parallel_size=tp, ring_degree=cp, sequence_parallel_size=cp)
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import cp_state

    pv, pa = port(tiny, make_inputs(_cfg(tiny)))
    torch.save({"v": pv, "a": pa, "cp": cp_state()[0]}, f"{out_path}.{rank}")


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (1, 4)])
def test_port_context_parallel_matches_tp1(tiny_dir, tmp_path, tp, cp):
    """CP over the video tokens (gathered K/V for the video self-attention and video-to-audio)
    reproduces the single-rank forward on every rank, composed with TP."""
    out = str(tmp_path / "cp")
    world = tp * cp
    mp.start_processes(
        _tp_cp_worker,
        args=(world, _port(), tiny_dir, tp, cp, out),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    pv, pa = port(tiny_dir, make_inputs(_cfg(tiny_dir)))
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        assert got["cp"] == cp
        assert got["v"].shape == pv.shape
        assert _rel(got["v"], pv) < 1e-5, (r, _rel(got["v"], pv))
        assert _rel(got["a"], pa) < 1e-5, (r, _rel(got["a"], pa))
