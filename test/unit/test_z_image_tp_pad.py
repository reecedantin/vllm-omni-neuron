# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the head-padded tensor-parallel DiT (``ZImageDiTConfig.tp_heads``).

Z-Image has 30 attention heads, so TP 4 or 8 needs the head axis zero-padded to 32. Here a tiny
DiT with 3 heads runs at TP 2 (padded to 4 heads, 2 per rank) in four gloo CPU processes, as two TP
groups with different inputs (the CFG-parallel layout: two guidance branches at once). Each group
must match the diffusers fp32 reference for its own inputs, which also checks that the block
all-reduces stay inside their TP group instead of the world group.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import torch

# 3 heads of 32 (axes_dims sum to the head dim); ffn_dim = 96 * 8 / 3 = 256 divides by 2.
TINY_DIT_3H = dict(
    all_patch_size=[2],
    all_f_patch_size=[1],
    axes_dims=[8, 12, 12],
    axes_lens=[1536, 512, 512],
    cap_feat_dim=64,
    dim=96,
    in_channels=16,
    n_heads=3,
    n_kv_heads=3,
    n_layers=2,
    n_refiner_layers=1,
    norm_eps=1e-5,
    qk_norm=True,
    rope_theta=256.0,
    t_scale=1000.0,
)


def _inputs(cfg, seed: int = 0):
    torch.manual_seed(seed)
    imgs = [torch.randn(16, 1, 16, 24) for _ in range(2)]
    caps = [torch.randn(45, cfg.cap_feat_dim), torch.randn(7, cfg.cap_feat_dim)]
    return imgs, caps, torch.tensor([0.3, 0.3])


def _rank_main(rank: int, port: int, ckpt: str, out: str) -> None:
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.z_image.transformer import (
        NeuronZImageDiT,
        RopeTables,
        ZImageDiTConfig,
        prepare_dit_inputs,
        unpatchify,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=4, rank=rank)
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]  # every rank creates every group
    pair, tp_rank = divmod(rank, 2)
    cfg = ZImageDiTConfig.from_model_dir(ckpt)
    dit = NeuronZImageDiT(cfg, dtype=torch.float32, tp=(2, tp_rank, groups[pair]))
    dit.load_weights(ckpt)
    assert dit.layers[0].n_heads == 2  # 3 heads padded to 4
    imgs, caps, t = _inputs(cfg, seed=pair)
    prep = prepare_dit_inputs(cfg, RopeTables(cfg), imgs, caps, torch.float32)
    args = list(prep["args"])
    args[9] = t
    with torch.no_grad():
        got = unpatchify(cfg, dit(*args), prep["meta"])
        dit.setup_runner(1)  # the served block-split path (deep-copies a block as its template)
        split = unpatchify(cfg, dit(*args), prep["meta"])
    for a, b in zip(got, split):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    if tp_rank == 0:
        torch.save(got, out.format(pair))
    dist.destroy_process_group()


def test_padded_tp2_two_groups_match_diffusers(tmp_path):
    from diffusers import ZImageTransformer2DModel

    from vllm_omni_neuron.diffusion.models.z_image.transformer import ZImageDiTConfig

    torch.manual_seed(0)
    ref = ZImageTransformer2DModel(**TINY_DIT_3H).eval()
    for p in ref.parameters():  # random weights at a scale that keeps activations O(1)
        p.data = torch.randn_like(p) * 0.05
    ckpt = str(tmp_path / "tiny-3h")
    ref.save_pretrained(os.path.join(ckpt, "transformer"))
    cfg = ZImageDiTConfig.from_model_dir(ckpt)
    assert cfg.tp_heads(2) == 4 and cfg.tp_heads(1) == 3
    wants = []
    for pair in range(2):
        imgs, caps, t = _inputs(cfg, seed=pair)
        with torch.no_grad():
            wants.append(
                ref([i.clone() for i in imgs], t, [c.clone() for c in caps], return_dict=False)[0]
            )

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "tp2_group{}.pt")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    )
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(r), str(port), ckpt, out], env=env
        )
        for r in range(4)
    ]
    assert [p.wait(timeout=600) for p in procs] == [0] * 4
    for pair, want in enumerate(wants):
        got = torch.load(out.format(pair))
        for g, w in zip(got, want):
            assert g.shape == w.shape
            rel = ((g - w).norm() / w.norm()).item()
            assert rel < 1e-5, (pair, rel)


if __name__ == "__main__":  # a rank of test_padded_tp2_two_groups_match_diffusers
    _rank_main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
