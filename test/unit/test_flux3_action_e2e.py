# SPDX-License-Identifier: Apache-2.0
"""Accuracy tier 3 (CPU): the full 4-step DROID policy end to end, golden against the upstream policy.

Per-step error compounds, so the end-to-end comparison is the gate: actions, predicted video latents
and the decoded predicted frames of this port against the upstream reference run on the same tiny
structure model, observation and seed. Also checks the served contract: the Omni pipeline's
``forward`` returns exactly what ``NeuronFlux3ActionPolicy.predict`` does. The device half is
``test/neuron/test_flux3_action_policy_device.py``.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni_neuron.diffusion.models.flux3_action.observation import (
    load_observation,
    observation_extra_args,
)
from vllm_omni_neuron.diffusion.models.flux3_action.pipeline_flux3_action import (
    NeuronFlux3ActionPipeline,
)
from vllm_omni_neuron.diffusion.models.flux3_action.policy import NeuronFlux3ActionPolicy

from .test_flux3_action_utils.fixtures import cos, observation, psnr, tiny, upstream  # noqa: F401
from .test_flux3_action_utils.reference import load_upstream_vae, run_reference


def test_e2e_matches_upstream(tiny, observation, upstream):  # noqa: F811
    batch = load_observation(observation)
    ref = run_reference(tiny["droid"], tiny["base"], batch)
    pol = NeuronFlux3ActionPolicy(tiny["droid"], base_dir=tiny["base"])
    out = pol.predict(batch)
    assert out.actions.shape == (1, 32, 8)
    assert ((out.actions - ref["actions"]) ** 2).mean().item() < 1e-4
    assert cos(out.video_latents, ref["video_latents"]) > 0.999

    # decoded predicted frames: this port's VAE on its latents vs the upstream VAE on upstream latents
    theirs = load_upstream_vae(tiny["base"])
    frames = pol.decode_frames(out, max_latent_frames=2)
    z_ref = torch.cat([ref["cond_latent"][0], ref["video_latents"][0]], dim=1)[:, :2][None].to(
        torch.bfloat16
    )
    frames_ref = theirs.module.decode(z_ref)[0].float()
    assert frames.shape == frames_ref.shape
    assert psnr(frames, frames_ref) > 30.0


class _SamplingParams:
    def __init__(self, seed, extra_args):
        self.seed = seed
        self.extra_args = extra_args
        self.height = self.width = None


class _Request:
    def __init__(self, prompt, seed, extra_args):
        self.prompt = prompt
        self.sampling_params = _SamplingParams(seed, extra_args)
        self.multi_modal_data = {}


class _Config:
    def __init__(self, model, base):
        self.model = model
        self.model_config = {"flux3_action_base": base, "decode_frames": False}
        self.parallel_config = type("P", (), {"tensor_parallel_size": 1})()


def test_pipeline_forward_matches_policy(tiny, observation):  # noqa: F811
    extra = observation_extra_args(observation, task="test")
    pipe = NeuronFlux3ActionPipeline(od_config=_Config(tiny["droid"], tiny["base"]))
    served = pipe.forward(_Request({"prompt": "test"}, 0, extra)).output["actions"]
    assert list(served.shape) == [1, 32, 8]
    direct = (
        NeuronFlux3ActionPolicy(tiny["droid"], base_dir=tiny["base"])
        .predict(load_observation(observation, task="test"), seed=0)
        .actions
    )
    assert (served.float() - direct.float()).abs().max().item() < 1e-4


def test_pipeline_dummy_request_compiles_without_inputs(tiny):  # noqa: F811
    """The engine's startup dummy run sends a request with no observation; it must not raise."""
    pipe = NeuronFlux3ActionPipeline(od_config=_Config(tiny["droid"], tiny["base"]))
    out = pipe.forward(_Request({"prompt": ""}, 0, {})).output
    assert list(out["actions"].shape) == [1, 32, 8]
    assert "video" not in out


def test_raw_byte_and_list_cameras_give_identical_requests(tiny, observation):  # noqa: F811
    """Cameras as raw bytes (the request form) and as nested lists (still accepted) decode to the
    same observation, so the served actions are identical."""
    import numpy as np

    extra = observation_extra_args(observation, task="test")
    assert isinstance(extra["images.wrist"]["data"], bytes)
    as_list = dict(extra)
    for key in ("images.wrist", "images.left", "images.right"):
        c = extra[key]
        as_list[key] = np.frombuffer(c["data"], dtype=np.uint8).reshape(c["shape"]).tolist()
    pipe = NeuronFlux3ActionPipeline(od_config=_Config(tiny["droid"], tiny["base"]))
    a = pipe.forward(_Request({"prompt": "test"}, 0, extra)).output
    b = pipe.forward(_Request({"prompt": "test"}, 0, as_list)).output
    assert torch.equal(a["actions"], b["actions"])
    for key in ("parse_s", "forward_s", "worker_enter_time", "worker_exit_time"):
        assert key in a["timing"]


def _cfg_worker(rank, world, tp, port, policy_dir, base, obs, q, descending=False):
    from types import SimpleNamespace

    import torch.distributed as dist

    os.environ["VLLM_NEURON_CPU_MODE"] = "1"  # gloo on CPU: no Neuron replica-group registration
    os.environ["FLUX3_ACTION_RANK_CHECK"] = "1"  # the served all-rank agreement gather
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    # TP replica = rank // tp (upstream RankGenerator order tp-sp-pp-cfg-dp). A descending CFG
    # group ([t + tp, t], as the Trn2 physical-mesh layouts emit) makes the HIGHER rank the
    # conditional branch (rank_in_group follows the list), while c10d orders the group sorted.
    tp_rank = rank % tp
    tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(world // tp)]
    cfg_lists = [[t + tp, t] if descending else [t, t + tp] for t in range(tp)]
    cfg_pgs = [dist.new_group(ranks) for ranks in cfg_lists]  # every rank creates every group
    ranks = cfg_lists[tp_rank]
    cfg = SimpleNamespace(
        ranks=ranks, world_size=2, rank_in_group=ranks.index(rank), cpu_group=cfg_pgs[tp_rank]
    )
    pol = NeuronFlux3ActionPolicy(
        policy_dir,
        base_dir=base,
        tp_size=tp,
        tp_rank=tp_rank,
        tp_group=tp_groups[rank // tp] if tp > 1 else None,
        cfg_size=2,
        cfg_rank=cfg.rank_in_group,
        cfg_group=cfg,
        world_rank=rank,
        world_group=dist.group.WORLD,
    )
    out = pol.predict(load_observation(obs))
    # numpy pickles by value (a tensor would travel as shared memory that dies with this process)
    ok = out.timing["cfg_parallel"] and out.timing["ranks_agree"]
    q.put((rank, out.actions.numpy(), out.video_latents.numpy(), ok))
    dist.destroy_process_group()


def _run_cfg_parallel(tiny, observation, tp, descending=False):  # noqa: F811
    import torch.multiprocessing as mp

    world = 2 * tp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (os.getpid() + tp + 7 * descending) % 400
    procs = [
        ctx.Process(
            target=_cfg_worker,
            args=(r, world, tp, port, tiny["droid"], tiny["base"], observation, q, descending),
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    res = sorted((q.get(timeout=600) for _ in procs), key=lambda r: r[0])
    for p in procs:
        p.join(60)
    return [(r, torch.from_numpy(a), torch.from_numpy(v), used) for r, a, v, used in res]


@pytest.mark.parametrize("descending", [False, True])
def test_cfg_parallel_matches_sequential_exactly(tiny, observation, descending):  # noqa: F811
    """CFG-parallel (two ranks, one guidance branch each, host all-gather) == sequential CFG,
    bit for bit, on every rank; only rank 0 runs the VAE encode. ``descending`` uses the CFG group
    ``[1, 0]`` (physical-mesh order): a positional c10d gather would swap the two branches."""
    res = _run_cfg_parallel(tiny, observation, tp=1, descending=descending)
    ref = NeuronFlux3ActionPolicy(tiny["droid"], base_dir=tiny["base"]).predict(
        load_observation(observation)
    )
    for _rank, actions, video, used in res:
        assert used
        assert torch.equal(actions, ref.actions)
        assert torch.equal(video, ref.video_latents)


@pytest.mark.parametrize("descending", [False, True])
def test_tp2_cfg_parallel_matches_tp2_sequential(tiny, observation, descending):  # noqa: F811
    """TP2 x CFG2 (four ranks): each TP group's result equals TP2 with sequential CFG exactly, so the
    CFG split adds no error on top of the TP sharding; with descending CFG groups too."""
    res = _run_cfg_parallel(tiny, observation, tp=2, descending=descending)
    for _rank, actions, video, ok in res[1:]:
        assert ok  # CFG-parallel ran and the all-rank check agreed
        assert torch.equal(actions, res[0][1])
        assert torch.equal(video, res[0][2])
    seq = _run_tp_sequential(tiny, observation, tp=2)
    assert torch.equal(res[0][1], seq[0])
    assert torch.equal(res[0][2], seq[1])


def _tp_seq_worker(rank, world, port, policy_dir, base, obs, q):
    import torch.distributed as dist

    os.environ["VLLM_NEURON_CPU_MODE"] = "1"
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    pol = NeuronFlux3ActionPolicy(
        policy_dir,
        base_dir=base,
        tp_size=world,
        tp_rank=rank,
        tp_group=dist.group.WORLD,
        world_rank=rank,
        world_group=dist.group.WORLD,
    )
    out = pol.predict(load_observation(obs))
    if rank == 0:
        q.put((out.actions.numpy(), out.video_latents.numpy()))
    dist.destroy_process_group()


def _run_tp_sequential(tiny, observation, tp):  # noqa: F811
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29950 + os.getpid() % 40
    procs = [
        ctx.Process(
            target=_tp_seq_worker,
            args=(r, tp, port, tiny["droid"], tiny["base"], observation, q),
        )
        for r in range(tp)
    ]
    for p in procs:
        p.start()
    out = q.get(timeout=600)
    for p in procs:
        p.join(60)
    return torch.from_numpy(out[0]), torch.from_numpy(out[1])
