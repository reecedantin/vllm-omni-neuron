# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 DiT context parallelism on CPU (gloo ranks), with the random-weight structure checkpoint.

* the host CP plan: every packed row lands on exactly one rank, pads at the end (dense) or in empty tail tiles (VSA);
* TP x CP = 1x2 and 2x2 reproduce the TP=1 forward, dense and VSA, on a geometry whose sequence (dense) and tile
  count (VSA) do not divide the CP degree, so the pad paths run.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_minimax_h3_vsa import _vsa_tiny  # noqa: E402

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

GEOM = dict(
    height=256, width=256, num_frames=22
)  # grid (2, 8, 8): 4 video tiles; 13 + 88 + 128 = 229 rows
NUM_TEXT = 13
SPARSITY = 0.5


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _forward(weights: str, vsa: bool):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise
    from vllm_omni_neuron.diffusion.models.minimax_h3.transformer import NeuronMiniMaxH3Transformer

    tdir = os.path.join(weights, "transformer")
    cfg = MiniMaxH3DiTConfig.from_dir(tdir)
    dit = NeuronMiniMaxH3Transformer(
        cfg, dtype=torch.float32, vsa_sparsity=SPARSITY if vsa else None
    )
    dit.load_weights(tdir, "cpu")
    layout = build_layout(NUM_TEXT, **GEOM)
    g = torch.Generator().manual_seed(0)
    video_rows, audio_rows = draw_noise(layout, g)
    text = torch.randn(1, NUM_TEXT, cfg.text_dim, generator=g)
    grid = (layout.num_latent_frames, layout.latent_height // 2, layout.latent_width // 2)
    dit.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows, grid)
    cos, sin, shard = dit.shard_inputs(*layout.rotary(cfg.rope_freq_dim, cfg.rope_theta))
    with torch.no_grad():
        v, a = dit(
            text,
            audio_rows[None],
            video_rows[None],
            torch.tensor([0.4375, 0.8125]),
            cos,
            sin,
            None,
            shard,
        )
    return v.float(), a.float()


def _worker(rank, tp, cp, port, weights, vsa, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps
    from vllm_omni.diffusion.distributed import parallel_state as ops

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    ops.init_distributed_environment(
        world_size=tp * cp,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    ops.initialize_model_parallel(tensor_parallel_size=tp, ring_degree=cp, backend="gloo")
    v, a = _forward(weights, vsa)
    if rank == 0:
        torch.save((v, a), out_path)


def _run(tp, cp, weights, vsa, path):
    mp.spawn(_worker, args=(tp, cp, _port(), weights, vsa, str(path)), nprocs=tp * cp, join=True)
    return torch.load(path)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_cp_plan_covers_every_row():
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout
    from vllm_omni_neuron.diffusion.models.minimax_h3.transformer import CPPlan
    from vllm_omni_neuron.diffusion.models.minimax_h3.vsa import VSAGeometry

    layout = build_layout(NUM_TEXT, **GEOM)
    nt, na, nv = layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows
    seq = nt + na + nv
    grid = (layout.num_latent_frames, layout.latent_height // 2, layout.latent_width // 2)
    for cp in (2, 4, 8):
        for geom in (None, VSAGeometry.build((nt, na), grid, SPARSITY)):
            plan = CPPlan.build(cp, nt, na, nv, geom)
            assert plan.order.numel() == cp * plan.local_len
            real = plan.order[plan.order < seq]
            assert torch.equal(real.sort().values, torch.arange(seq))  # every row exactly once
            assert torch.equal(plan.order[plan.video_pos], torch.arange(nt + na, seq))
            assert torch.equal(plan.order[plan.audio_pos], torch.arange(nt, nt + na))
            rows, onehot = plan.local(cp - 1)
            assert onehot.shape == (plan.local_len, 3) and bool(
                (onehot.sum(-1) == (rows < seq).float()).all()
            )
            if geom is None:
                assert plan.kv_len == seq and bool((plan.order[:seq] == torch.arange(seq)).all())
            else:
                assert plan.local_len % 64 == 0 and plan.vsa_geom.n_tiles % cp == 0
                assert plan.vsa_geom.k_vid == geom.k_vid and plan.vsa_geom.n_video == geom.n_video


def test_cp_dense_matches_tp1(h3_tiny, tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_TP_SP", "0")  # plain TP all-reduce inside the CP slices
    v1, a1 = _run(1, 1, h3_tiny, False, tmp_path / "ref.pt")
    for tp, cp in ((1, 2), (2, 2)):
        v, a = _run(tp, cp, h3_tiny, False, tmp_path / f"tp{tp}cp{cp}.pt")
        assert v.shape == v1.shape and a.shape == a1.shape
        assert _rel(v, v1) < 1e-5 and _rel(a, a1) < 1e-5, (tp, cp, _rel(v, v1), _rel(a, a1))


def test_cp_vsa_matches_tp1(h3_tiny, tmp_path):
    w = _vsa_tiny(h3_tiny, str(tmp_path / "tiny-vsa"))
    v1, a1 = _run(1, 1, w, True, tmp_path / "ref.pt")
    for tp, cp in ((1, 2), (2, 2)):
        v, a = _run(tp, cp, w, True, tmp_path / f"tp{tp}cp{cp}.pt")
        assert _rel(v, v1) < 1e-5 and _rel(a, a1) < 1e-5, (tp, cp, _rel(v, v1), _rel(a, a1))


def test_cp_sequence_parallel_tp_matches_tp1(h3_tiny, tmp_path, monkeypatch):
    """Sequence-parallel TP inside the CP slices (the default) reproduces the TP=1 forward."""
    v1, a1 = _run(1, 1, h3_tiny, False, tmp_path / "ref.pt")
    monkeypatch.setenv("MINIMAX_H3_TP_SP", "1")
    v, a = _run(2, 2, h3_tiny, False, tmp_path / "sp.pt")
    assert v.shape == v1.shape and a.shape == a1.shape
    assert _rel(v, v1) < 1e-5 and _rel(a, a1) < 1e-5, (_rel(v, v1), _rel(a, a1))


def test_cp_sharded_adaln_matches_tp1(h3_tiny, tmp_path, monkeypatch):
    """The AdaLN projection sharded over CP too (the default) and the whole TP shard per CP rank
    (MINIMAX_H3_ADALN_CP=0) both reproduce the TP=1 forward."""
    v1, a1 = _run(1, 1, h3_tiny, False, tmp_path / "ref.pt")
    monkeypatch.setenv("MINIMAX_H3_ADALN_CP", "0")
    v, a = _run(2, 2, h3_tiny, False, tmp_path / "adaln_cp.pt")
    assert _rel(v, v1) < 1e-5 and _rel(a, a1) < 1e-5, (_rel(v, v1), _rel(a, a1))
