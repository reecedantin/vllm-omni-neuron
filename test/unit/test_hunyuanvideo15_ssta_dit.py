# SPDX-License-Identifier: Apache-2.0
"""The sparse (SSTA) DiT path on a tiny distilled-sparse I2V checkpoint (CPU, fp32):

* the tile-major slot layout, the RoPE / encoder-padding handling and the epilogue's inverse permutation
  against a reference built from the DENSE path with its joint attention replaced by
  :func:`ssta.ssta_attention` (the upstream-checked reference) and the encoder padding zeroed as upstream's
  ``zero_feat`` reorder;
* context parallelism (CP = 2, 3) of the sparse path equal to one rank (gloo).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_hunyuanvideo15_pipeline import (  # noqa: E402,F401
    _port,
    single_rank,
    spawn_with_free_port,
)
from .test_hunyuanvideo15_tiny_ckpt import make_tiny_i2v_checkpoint  # noqa: E402

TILE = [
    1,
    4,
    4,
]  # 16-token tiles: the tiny grid (3, 8, 20) -> 3 x 2 x 5 = 30 tiles; window 3x2x3 + top-k 4
TOPK = 8  # halved to 4 (3 latent frames)
LATENT = (3, 8, 20)


@pytest.fixture(scope="module")
def tiny_sparse_ckpt(tmp_path_factory) -> str:
    out = make_tiny_i2v_checkpoint(
        str(tmp_path_factory.mktemp("hv15_tiny_sparse")), guidance_scale=1.0
    )
    path = os.path.join(out, "transformer", "config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg["attn_mode"] = "flex-block-attn"
    cfg["attn_param"] = {
        "attn_mask_share_within_head": 0,
        "attn_pad_type": "zero",
        "attn_sparse_type": "ssta",
        "attn_use_text_mask": 1,
        "ssta_adaptive_pool": None,
        "ssta_lambda": 0.7,
        "ssta_sampling_type": "importance",
        "ssta_threshold": 0.0,
        "ssta_topk": TOPK,
        "tile_size": TILE,
        "win_ratio": 10,
        "win_size": [[3, 3, 3]],
        "win_type": "fixed",
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    t, h, w = LATENT
    text_mask = torch.zeros(1, 40)
    text_mask[0, :9] = 1
    text2_mask = torch.zeros(1, 16)
    text2_mask[0, :5] = 1
    return dict(
        hidden_states=torch.randn(1, 65, t, h, w, generator=g),
        timestep=torch.tensor([700.0]),
        encoder_hidden_states=torch.randn(1, 40, 64, generator=g),
        encoder_attention_mask=text_mask,
        encoder_hidden_states_2=torch.randn(1, 16, 32, generator=g),
        encoder_attention_mask_2=text2_mask,
        image_embeds=torch.randn(1, 729, 1152, generator=g),
        return_dict=False,
    )


def _facade(ckpt):
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import (
        NeuronHunyuanVideo15Transformer,
    )

    m = NeuronHunyuanVideo15Transformer(
        SimpleNamespace(model=ckpt, dtype=torch.float32, model_config={}, flow_shift=None)
    )
    m.load()
    return m


def test_sparse_dit_matches_reference(tiny_sparse_ckpt, single_rank, monkeypatch):  # noqa: F811
    import vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 as P
    import vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer as T
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15 import ssta

    monkeypatch.setattr(P, "BLOCKS_PER_GRAPH", 2)
    m = _facade(tiny_sparse_ckpt)
    assert m.cfg.ssta is not None and m.cfg.ssta.tile == tuple(TILE)
    inp = _inputs()
    with torch.no_grad():
        (got,) = m(**inp)

    # reference: the dense path, joint block attention -> ssta.ssta_attention, encoder padding zeroed
    params = m.cfg.ssta
    m.cfg.ssta = None
    sv = LATENT[0] * LATENT[1] * LATENT[2]
    ne = 729 + 16 + 1000  # SigLIP | byT5 | MLLM bucket
    n_valid = 729 + 5 + 9
    dense_attn = T.hv15_attention

    def attn(q, k, v, scale, key_bias=None):
        if q.shape[2] != sv + ne:  # the token refiner (text only)
            return dense_attn(q, k, v, scale, key_bias=key_bias)
        return ssta.ssta_attention(
            q,
            k,
            v,
            LATENT,
            ne,
            topk=params.topk,
            lambda_=params.lambda_,
            n_valid_text=n_valid,
            tile=params.tile,
        )

    prologue = T.NeuronHunyuanVideo15DiT.prologue

    def zero_pad_prologue(self, *args):
        hidden, enc, temb = prologue(self, *args)
        keep = (torch.arange(enc.shape[1]) < n_valid).to(enc.dtype)[None, :, None]
        return hidden, enc * keep, temb

    monkeypatch.setattr(T, "hv15_attention", attn)
    monkeypatch.setattr(T.NeuronHunyuanVideo15DiT, "prologue", zero_pad_prologue)
    m._fns.clear()
    with torch.no_grad():
        (want,) = m(**inp)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"[ssta-dit] sparse vs reference rel-L2 {rel:.3e}")
    assert rel < 1e-5, rel


def _cp_worker(rank, world, cp, port, ckpt, out_path, inp):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as vps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        VLLM_NEURON_CPU_MODE="1",
        HV15_BLOCKS_PER_GRAPH="2",
    )
    vps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, backend="gloo")
    tp = world // cp
    initialize_model_parallel(tensor_parallel_size=tp, sequence_parallel_size=cp, ring_degree=cp)
    m = _facade(ckpt)
    assert m.dit.cp_size == cp and m.dit.tp_size == tp
    with torch.no_grad():
        (out,) = m(**inp)
    torch.save(out, f"{out_path}.{rank}")


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (1, 3)])
def test_sparse_context_parallel_matches_single_rank(tiny_sparse_ckpt, tmp_path, tp, cp):
    """30 video tiles split over CP ranks (whole tiles), text query tiles split and all-gathered."""
    inp = _inputs(seed=1)
    world = tp * cp
    out = str(tmp_path / f"cp{cp}_tp{tp}")
    spawn_with_free_port(
        _cp_worker, lambda port: (world, cp, port, tiny_sparse_ckpt, out, inp), world
    )
    ref = str(tmp_path / "ref")
    spawn_with_free_port(_cp_worker, lambda port: (1, 1, port, tiny_sparse_ckpt, ref, inp), 1)
    want = torch.load(f"{ref}.0")
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        rel = ((got - want).norm() / want.norm()).item()
        assert rel < 1e-5, (tp, cp, r, rel)
