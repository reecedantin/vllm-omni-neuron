# SPDX-License-Identifier: Apache-2.0
"""Neuron MiniMax-H3 DiT vs the diffusers reference, on CPU, with the random-weight structure checkpoint.

* TP=1 fp32: the plugin's re-implementation (segment AdaLN, contiguous layout, fused-SwiGLU split, fp32 heads)
  reproduces ``MiniMaxH3Transformer3DModel`` driven exactly like diffusers' modular pipeline drives it.
* ``adaln="host"`` (host-computed modulation tables) is bit-identical to the on-device AdaLN path.
* TP=2 (two gloo ranks) reproduces TP=1: sharded loaders, the SwiGLU half-split, the all-reduces and the AdaLN
  all-gather.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

GEOM = dict(
    height=64, width=96, num_frames=22
)  # 2 latent frames x 4 x 6, 44 audio latents per channel
NUM_TEXT = 13


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(cfg):
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise

    layout = build_layout(NUM_TEXT, **GEOM)
    g = torch.Generator().manual_seed(0)
    video_rows, audio_rows = draw_noise(layout, g)
    text = torch.randn(1, NUM_TEXT, cfg.text_dim, generator=g)
    return (
        layout,
        text,
        video_rows,
        audio_rows,
        0.4375,
        0.8125,
    )  # t_video != t_audio exercises both AdaLN rows


def _reference(weights, dtype=torch.float32):
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig

    cfg = MiniMaxH3DiTConfig.from_dir(os.path.join(weights, "transformer"))
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        weights, subfolder="transformer", torch_dtype=dtype
    ).eval()
    model = model.float() if dtype == torch.float32 else model
    layout, text, video_rows, audio_rows, t_v, t_a = _inputs(cfg)
    ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        layout.video_indices, layout.audio_indices, 0, 0, NUM_TEXT, t_v, t_a, t_v, 1.0
    )
    with torch.no_grad():
        v, a = model(
            video_rows[None],
            audio_rows[None],
            text.to(dtype),
            ts,
            ts_idx,
            layout.token_tags,
            layout.position_ids,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            return_dict=False,
        )
    return v.float(), a.float()


def _neuron(weights, adaln="device", dtype=torch.float32):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig
    from vllm_omni_neuron.diffusion.models.minimax_h3.transformer import (
        NeuronMiniMaxH3Transformer,
        host_adaln_tables,
    )

    tdir = os.path.join(weights, "transformer")
    cfg = MiniMaxH3DiTConfig.from_dir(tdir)
    dit = NeuronMiniMaxH3Transformer(cfg, dtype=dtype, adaln=adaln)
    dit.load_weights(tdir, "cpu")
    layout, text, video_rows, audio_rows, t_v, t_a = _inputs(cfg)
    dit.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows)
    cos, sin = layout.rotary(cfg.rope_freq_dim, cfg.rope_theta)
    ts = torch.tensor([t_v, t_a], dtype=torch.float32)
    tables = host_adaln_tables(tdir, cfg, dtype).tables(dit.temb(ts)) if adaln == "host" else None
    with torch.no_grad():
        v, a = dit(text, audio_rows[None], video_rows[None], ts, cos, sin, tables)
    return v.float(), a.float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_dit_matches_reference_fp32(vllm_single_rank, h3_tiny):
    v_ref, a_ref = _reference(h3_tiny)
    v, a = _neuron(h3_tiny)
    assert v.shape == v_ref.shape and a.shape == a_ref.shape
    assert _rel(v, v_ref) < 1e-5, _rel(v, v_ref)
    assert _rel(a, a_ref) < 1e-5, _rel(a, a_ref)


def test_host_adaln_bit_identical(vllm_single_rank, h3_tiny):
    v0, a0 = _neuron(h3_tiny, "device", torch.bfloat16)
    v1, a1 = _neuron(h3_tiny, "host", torch.bfloat16)
    assert torch.equal(v0, v1) and torch.equal(a0, a1)


def test_dit_bf16_close_to_reference(vllm_single_rank, h3_tiny):
    """bf16 port vs fp32 reference vs bf16 reference: the port sits inside the bf16 band."""
    v_ref, a_ref = _reference(h3_tiny)
    v16_ref, _ = _reference(h3_tiny, torch.bfloat16)
    v16, a16 = _neuron(h3_tiny, "device", torch.bfloat16)
    band = _rel(v16_ref, v_ref)
    assert _rel(v16, v_ref) < max(2.5 * band, 2e-2), (_rel(v16, v_ref), band)


def _tp_worker(rank, world, port, weights, out_path):
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
    v, a = _neuron(weights)
    if rank == 0:
        torch.save((v, a), out_path)


def test_tp2_matches_tp1(h3_tiny, tmp_path):
    outs = {}
    for world in (1, 2):
        path = tmp_path / f"tp{world}.pt"
        mp.spawn(_tp_worker, args=(world, _port(), h3_tiny, str(path)), nprocs=world, join=True)
        outs[world] = torch.load(path)
    (v1, a1), (v2, a2) = outs[1], outs[2]
    assert _rel(v2, v1) < 1e-5 and _rel(a2, a1) < 1e-5, (_rel(v2, v1), _rel(a2, a1))
