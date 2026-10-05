# SPDX-License-Identifier: Apache-2.0
"""TP teacher-forced parity for a large Qwen3-VL Cosmos3 checkpoint (Super) that does not fit one core.

Single-rank parity (``test_cosmos3_edge_qwen3_device.py``) cannot load Super: 128 GB bf16 on one 24 GB
core. This spawns a ``world`` TP group on ``world`` NeuronCores, runs the UND + one GEN call sharded,
all-gathers rank outputs, and compares on rank 0 against a CPU fp32 oracle (full model, host RAM).

Env: ``COSMOS3_QWEN3_WEIGHTS`` (real Super), ``COSMOS3_TP`` (default 16), ``COSMOS3_GEN_TEST_GEOM``
(default ``1,40,40`` for t2i / ``3,16,16`` for i2v/action / ``5,16,16`` for the action modes). Cases:
t2i, i2v (clean conditioning first frame via noisy_mask), action (extra action stream through
forward_action), and the action modes policy / forward_dynamics / inverse_dynamics (video AND action
outputs, each gated against a pure-bf16 CPU run as well as the fp32 oracle); action cases are skipped
if the checkpoint has no action head. Each spawned rank is pinned to one core of the launch core set
(``FLEET_CORE_LIST``, else ``NEURON_RT_VISIBLE_CORES``) before the Neuron runtime starts.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.multiprocessing as mp

from .test_cosmos3_edge_pipeline_accuracy import launch_core_list, pin_rank_core
from .test_cosmos3_edge_qwen3_device import ACTION_MODES, action_case
from .test_cosmos3_edge_und_device import _cos, _neuron_available, _rel

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

WEIGHTS = os.environ.get("COSMOS3_QWEN3_WEIGHTS", "")
TP = int(os.environ.get("COSMOS3_TP", "16"))
COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


def _prompt(cfg_vocab: int):
    torch.manual_seed(0)
    real, bucket = 37, 64
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, min(100000, cfg_vocab), (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    return ids, mask


def _oracle_entry(rank, weights, geom, case, out_path, dtype=torch.float32):
    _cpu_oracle(weights, geom, case, out_path, dtype)


def _outputs(case, out):
    """Compared outputs of one GEN call: the video for t2i / i2v / the generic action case; video AND
    action for the action modes. Host copy first, then the cast: an eager dtype cast of a device
    tensor fails on the Lite runtime."""
    if case in ACTION_MODES:
        return {"video": out[0].cpu().float(), "action": out[1].cpu().float()}
    return {"video": (out[0] if case == "action" else out).cpu().float()}


def _cpu_oracle(weights, geom, case, out_path, dtype=torch.float32):
    """Full UND+GEN on the host (TP=1) at ``dtype`` (fp32 oracle; bf16 for the action modes' error
    reference), saved for the device ranks to compare against."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    cfg = EdgeGenConfig.from_model_dir(weights)
    t, h, w = geom
    ids, mask = _prompt(cfg.vocab_size)
    und = NeuronCosmos3EdgeUND(cfg, dtype=dtype)
    und.load_weights(weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=dtype)
    gen.load_weights(weights, "cpu")
    lat, ts, spec = _inputs(cfg, geom, case)
    s_video = t * (h // 2) * (w // 2)
    s_action = spec["s_action"]
    with torch.no_grad():
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu.to(dtype), su.to(dtype))
        cg, sg = gen.rope_tables(
            mask, t, h, w, spec["fps"], t_action=s_action, action_fps=spec["action_fps"]
        )
        kb = gen.key_bias(mask, s_video + s_action)
        a = [lat.to(dtype), ts, cg.to(dtype), sg.to(dtype), kb, spec["nm"].to(dtype)]
        if s_action:
            w_in, b_in, w_out, b_out = gen.domain_weights(spec["domain"], "cpu")
            out = gen.forward_action(
                *a,
                spec["act"].to(dtype),
                spec["act_mask"].to(dtype),
                w_in.to(dtype),
                b_in.to(dtype),
                w_out.to(dtype),
                b_out.to(dtype),
                *kv,
            )
        else:
            out = gen(*a, *kv)
    torch.save(_outputs(case, out), out_path)


def _inputs(cfg, geom, case):
    """Deterministic latents / timestep / conditioning of one case, identical on the oracle and every
    device rank (seeded here, independent of what the caller drew before)."""
    t, h, w = geom
    torch.manual_seed(1234)
    lat, ts = torch.randn(1, 48, t, h, w), torch.tensor([500.0])
    return lat, ts, action_case(case, t, h, w, cfg.action_dim)


def _device_rank(rank, world, port, cores, weights, geom, case, oracle_path, result_path):
    # One logical core per rank, set before anything imports the Neuron runtime (else every rank races
    # for the same cores and nrt_init fails). Cores come from the launch core set, not absolute 0..N-1.
    pin_rank_core(rank, cores)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

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
    dev = torch.device("neuron", 0)  # one visible core per process (pinned above)
    cfg = EdgeGenConfig.from_model_dir(weights)
    has_action = case == "action" or case in ACTION_MODES
    if has_action and not cfg.action_gen:
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"skipped": "no action head"}, f)
        return
    t, h, w = geom
    ids, mask = _prompt(cfg.vocab_size)
    lat, ts, spec = _inputs(cfg, geom, case)
    s_video = t * (h // 2) * (w // 2)
    s_action = spec["s_action"]
    s_gen = s_video + s_action
    backend = get_compile_backend_name()
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.bfloat16)
    und.load_weights(weights, dev)
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.bfloat16)
    gen.load_weights(weights, dev)
    und_fn = torch.compile(
        und,
        backend=backend,
        fullgraph=True,
        dynamic=False,
        options={"model_name": f"super_und_{world}", "compiler_args": COMPILER_ARGS},
    )
    fwd = gen.forward_action if has_action else gen.forward
    gen_fn = torch.compile(
        fwd,
        backend=backend,
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"super_gen_{case}_{world}_{t}x{h}x{w}",
            "compiler_args": COMPILER_ARGS,
        },
    )
    cu, su = und.rope_tables(mask)
    cg, sg = gen.rope_tables(
        mask, t, h, w, spec["fps"], t_action=s_action, action_fps=spec["action_fps"]
    )
    kb = gen.key_bias(mask, s_gen)
    bf = torch.bfloat16

    def call():
        kv = und_fn(ids.to(dev), cu.to(bf).to(dev), su.to(bf).to(dev))
        common = [
            lat.to(bf).to(dev),
            ts.to(dev),
            cg.to(bf).to(dev),
            sg.to(bf).to(dev),
            kb.to(dev),
            spec["nm"].to(bf).to(dev),
        ]
        if has_action:
            w_in, b_in, w_out, b_out = gen.domain_weights(spec["domain"], dev)
            common += [
                spec["act"].to(bf).to(dev),
                spec["act_mask"].to(bf).to(dev),
                w_in,
                b_in,
                w_out,
                b_out,
            ]
        return _outputs(case, gen_fn(*common, *kv))

    with torch.no_grad():
        out = call()
        out2 = call()
    if rank == 0:
        oracle = torch.load(oracle_path)
        rep = {
            "geom": [t, h, w],
            "tokens": s_gen,
            "tp": world,
            "case": case,
            "rel_dev": _rel(out["video"], oracle["video"]),
            "cos_dev": _cos(out["video"], oracle["video"]),
            "deterministic": all(torch.equal(out[k], out2[k]) for k in out),
        }
        bf_path = oracle_path.replace(".pt", "_bf16.pt")
        ref16 = torch.load(bf_path) if os.path.exists(bf_path) else None
        for k in out:
            rep[f"{k}_rel_dev"], rep[f"{k}_cos_dev"] = (
                _rel(out[k], oracle[k]),
                _cos(out[k], oracle[k]),
            )
            if ref16 is not None:
                rep[f"{k}_rel_cpu_bf16"] = _rel(ref16[k], oracle[k])
                rep[f"{k}_cos_cpu_bf16"] = _cos(ref16[k], oracle[k])
        with open(result_path, "w") as f:
            json.dump(rep, f)


@pytest.mark.parametrize("case", ["t2i", "i2v", "action", *ACTION_MODES])
def test_super_tp_parity(tmp_path, case):
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "transformer")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a real Cosmos3-Super checkout")
    default_geom = {"t2i": "1,40,40"}.get(case, "5,16,16" if case in ACTION_MODES else "3,16,16")
    geom = tuple(int(x) for x in os.environ.get("COSMOS3_GEN_TEST_GEOM", default_geom).split(","))
    oracle = str(tmp_path / "oracle.pt")
    result = os.path.join(
        os.environ.get("COSMOS3_TEST_OUT", str(tmp_path)), f"super_tp{TP}_{case}_parity.json"
    )
    # CPU oracles in their own processes (own torch.distributed world=1), then the TP device ranks.
    mp.spawn(_oracle_entry, args=(WEIGHTS, geom, case, oracle), nprocs=1, join=True)
    if case in ACTION_MODES:  # bf16 CPU reference: the action output's bar is relative to pure bf16
        mp.spawn(
            _oracle_entry,
            args=(WEIGHTS, geom, case, oracle.replace(".pt", "_bf16.pt"), torch.bfloat16),
            nprocs=1,
            join=True,
        )
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(
        _device_rank,
        args=(TP, port, launch_core_list()[:TP], WEIGHTS, geom, case, oracle, result),
        nprocs=TP,
        join=True,
    )
    rep = json.load(open(result))
    print("[super-tp-parity]", json.dumps(rep))
    if rep.get("skipped"):
        pytest.skip(rep["skipped"])
    assert rep["deterministic"], rep
    if case not in ACTION_MODES:
        assert rep["cos_dev"] >= 0.999, rep
        assert rep["rel_dev"] <= 0.03, rep
        return
    for k in (
        "video",
        "action",
    ):  # same bars as the single-rank gate: <= 2x pure bf16, cos >= 0.999
        assert rep[f"{k}_rel_dev"] <= 2 * rep[f"{k}_rel_cpu_bf16"] + 1e-3, (k, rep)
        assert rep[f"{k}_cos_dev"] >= 0.999, (k, rep)
