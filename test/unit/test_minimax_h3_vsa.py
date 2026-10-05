# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 (FastH3 8-Step-V2) on CPU with the random-weight structure checkpoint.

* the geometry matches FastVideo's tiling rules (segment-pure prefix chunks, (4,4,4) video tiles, exact sizes);
* the compiled-graph formulation (query-blocked, argmax top-k) equals the token-mask reference;
* the plugin DiT with VSA (gate = ``to_gate_compress`` of the attention input) reproduces diffusers' transformer
  driven with the reference VSA processor, at TP=1 fp32;
* the 8-Step-V2 contract: 9 grid points from the trained ladder with the checkpoint's shifts.
"""

from __future__ import annotations

import json
import math
import os
import shutil

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

GEOM = dict(
    height=128, width=192, num_frames=22
)  # 2 latent frames x 8 x 12 -> grid (2, 4, 6): 2 video tiles
NUM_TEXT = 13
SPARSITY = 0.5


def test_geometry_matches_fastvideo_rules():
    from vllm_omni_neuron.diffusion.models.minimax_h3.vsa import VSAGeometry, compute_topk

    g = VSAGeometry.build((70, 130), (5, 6, 9), 0.8)
    assert g.n_prefix == 2 + 3  # text 64+6, audio 64+64+2: segment-pure chunks
    assert g.n_video == math.ceil(5 / 4) * math.ceil(6 / 4) * math.ceil(9 / 4)
    assert g.k_vid == compute_topk(0.8, g.n_video)
    assert g.tile_sizes[:5].tolist() == [64, 6, 64, 64, 2]
    assert int(g.tile_sizes.sum()) == g.seq_len == 200 + 5 * 6 * 9
    rows = g.slot_to_row[g.valid.bool()]
    assert sorted(rows.tolist()) == list(range(g.seq_len))  # every row exactly once
    assert torch.equal(g.slot_to_row.index_select(0, g.row_to_slot), torch.arange(g.seq_len))


def test_compiled_formulation_equals_reference():
    import vllm_omni_neuron.diffusion.models.minimax_h3.vsa as V

    g = V.VSAGeometry.build((19, 90), (3, 6, 10), 0.75)
    torch.manual_seed(0)
    q, k, v, gate = (torch.randn(g.seq_len, 3, 32) for _ in range(4))
    want = V.vsa_attention_reference(q, k, v, gate, g)
    saved = V.QBLOCK
    try:
        V.QBLOCK = 256  # several query blocks
        torch._dynamo.reset()
        got = torch.compile(V.vsa_attention, backend="eager")(
            q, k, v, gate, g
        )  # traced: argmax top-k path
    finally:
        V.QBLOCK = saved
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-5, rel


def _vsa_tiny(src: str, dst: str) -> str:
    """The structure checkpoint + random to_gate_compress weights + an 8-Step-V2-style contract."""
    from safetensors.torch import load_file, save_file

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    fn = os.path.join(dst, "transformer", "diffusion_pytorch_model.safetensors")
    sd = load_file(fn)
    cfg = json.load(open(os.path.join(dst, "transformer", "config.json")))
    inner = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    g = torch.Generator().manual_seed(1)
    for i in range(cfg["num_layers"]):
        sd[f"transformer_blocks.{i}.attn.to_gate_compress.weight"] = (
            0.05 * torch.randn(inner, cfg["hidden_size"], generator=g)
        ).to(torch.bfloat16)
    save_file(sd, fn)
    with open(os.path.join(dst, "fastvideo_inference.json"), "w") as f:
        json.dump(
            {
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "vsa_sparsity": SPARSITY,
                "vsa_tile_size": 64,
                "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
                "num_inference_steps": 9,
            },
            f,
        )
    return dst


@pytest.fixture(scope="module")
def h3_vsa_tiny(h3_tiny, tmp_path_factory):
    return _vsa_tiny(h3_tiny, str(tmp_path_factory.mktemp("vsa") / "tiny-vsa"))


def test_dit_vsa_matches_reference(vllm_single_rank, h3_vsa_tiny):
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig, vsa_sparsity
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise
    from vllm_omni_neuron.diffusion.models.minimax_h3.transformer import NeuronMiniMaxH3Transformer
    from vllm_omni_neuron.diffusion.models.minimax_h3.vsa import (
        VSAGeometry,
        install_reference_processors,
    )

    tdir = os.path.join(h3_vsa_tiny, "transformer")
    cfg = MiniMaxH3DiTConfig.from_dir(tdir)
    assert vsa_sparsity(h3_vsa_tiny) == SPARSITY
    layout = build_layout(NUM_TEXT, **GEOM)
    gen = torch.Generator().manual_seed(0)
    video_rows, audio_rows = draw_noise(layout, gen)
    text = torch.randn(1, NUM_TEXT, cfg.text_dim, generator=gen)
    t_v, t_a = 0.4375, 0.8125
    grid = (layout.num_latent_frames, layout.latent_height // 2, layout.latent_width // 2)

    ref = (
        MiniMaxH3Transformer3DModel.from_pretrained(h3_vsa_tiny, subfolder="transformer")
        .float()
        .eval()
    )
    with safe_open(os.path.join(tdir, "diffusion_pytorch_model.safetensors"), "pt") as f:
        gates = [
            f.get_tensor(f"transformer_blocks.{i}.attn.to_gate_compress.weight").float()
            for i in range(cfg.num_layers)
        ]
    holder = install_reference_processors(ref, gates)
    geom = VSAGeometry.build((NUM_TEXT, layout.num_audio_rows), grid, SPARSITY)
    assert geom.k_vid < geom.n_video  # the fixture is genuinely sparse
    holder["geom"] = geom
    ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        layout.video_indices, layout.audio_indices, 0, 0, NUM_TEXT, t_v, t_a, t_v, 1.0
    )
    with torch.no_grad():
        v_ref, a_ref = ref(
            video_rows[None],
            audio_rows[None],
            text,
            ts,
            ts_idx,
            layout.token_tags,
            layout.position_ids,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            return_dict=False,
        )

    dit = NeuronMiniMaxH3Transformer(cfg, dtype=torch.float32, vsa_sparsity=SPARSITY)
    dit.load_weights(tdir, "cpu")
    dit.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows, grid)
    cos, sin = layout.rotary(cfg.rope_freq_dim, cfg.rope_theta)
    with torch.no_grad():
        v, a = dit(text, audio_rows[None], video_rows[None], torch.tensor([t_v, t_a]), cos, sin)
    rel_v = ((v - v_ref).norm() / v_ref.norm()).item()
    rel_a = ((a - a_ref).norm() / a_ref.norm()).item()
    assert rel_v < 1e-4 and rel_a < 1e-4, (rel_v, rel_a)

    dense = NeuronMiniMaxH3Transformer(
        cfg, dtype=torch.float32
    )  # same weights, dense attention: must differ
    dense.load_weights(tdir, "cpu")
    dense.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows)
    with torch.no_grad():
        v_d, _ = dense(text, audio_rows[None], video_rows[None], torch.tensor([t_v, t_a]), cos, sin)
    assert ((v_d - v_ref).norm() / v_ref.norm()).item() > 10 * rel_v


def test_contract_ladder(h3_vsa_tiny):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import step_positions
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import load_schedulers

    pos = step_positions(h3_vsa_tiny, 9)
    assert pos == (0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0)
    assert (
        step_positions(h3_vsa_tiny, 5) is None
    )  # a mismatched request falls back to the linspace grid
    sv, sa = load_schedulers(h3_vsa_tiny, 9, pos)
    assert len(sv.timesteps) == len(sa.timesteps) == 8
    s = sv.shift
    assert (
        abs(float(sv.sigmas[0]) - s * 0.999 / (1 + (s - 1) * 0.999)) < 1e-6
        and float(sv.sigmas[-1]) == 0.0
    )
