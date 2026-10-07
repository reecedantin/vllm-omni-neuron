# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 pipeline plumbing on CPU with the random-weight structure checkpoint.

* the plugin registry resolves both MiniMax-H3 architecture keys to the Neuron pipeline;
* a full t2va request runs: text encoder -> layout -> 4 DiT forwards -> video + audio decode, with the shapes the
  released model produces (17n+5 frames, 32 kHz stereo matching the clip length);
* the 4-step denoise matches the same loop driven by the diffusers reference transformer (fp32).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

# the session fixture h3_tiny, registered once as a plugin (one checkpoint per session for every module)
pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

GEOM = dict(height=64, width=96, num_frames=22)
PROMPT = "A golden retriever runs through the surf at sunset, waves crashing."


def _pipe(weights, dtype=torch.float32, **mc):
    from vllm_omni_neuron.diffusion.models.minimax_h3 import NeuronMiniMaxH3Pipeline

    od = SimpleNamespace(model=weights, dtype=dtype, model_config=mc)
    p = NeuronMiniMaxH3Pipeline(od_config=od)
    p.load_weights()
    return p


def test_registry_entries():
    from vllm_omni_neuron.diffusion.models import minimax_h3

    archs = {e["model_arch"]: e["class_name"] for e in minimax_h3.PIPELINE_REGISTRY}
    assert archs == {
        "MiniMaxH3ModularPipeline": "NeuronMiniMaxH3Pipeline",
        "MiniMaxH3Pipeline": "NeuronMiniMaxH3Pipeline",
    }
    for e in minimax_h3.PIPELINE_REGISTRY:
        assert hasattr(minimax_h3, e["class_name"]) and hasattr(
            minimax_h3, e["post_process_func_name"]
        )


def test_t2va_request_shapes(vllm_single_rank, h3_tiny):
    p = _pipe(h3_tiny)
    assert p.default_steps == 5  # FastH3 release contract: 4 forwards
    out = p.generate(PROMPT, num_inference_steps=p.default_steps, seed=7, **GEOM)
    video, audio = out["video"], out["audio"]
    assert video.shape == (1, 3, 22, 64, 96), video.shape
    assert video.dtype == torch.uint8  # the default video_output: the clip's 8-bit pixels
    assert audio.shape[:2] == (1, 2)
    assert abs(audio.shape[-1] - 22 / 24 * 32000) <= 800, (
        audio.shape
    )  # one 800-sample hop of rounding
    assert len(p.stats["forward_s"]) == 4
    assert torch.isfinite(audio).all()


def test_rank_check_reports_agreement(vllm_single_rank, h3_tiny, monkeypatch):
    """MINIMAX_H3_RANK_CHECK=1 records the shared all-rank agreement report of the final latents (one rank here)."""
    monkeypatch.setenv("MINIMAX_H3_RANK_CHECK", "1")
    p = _pipe(h3_tiny)
    out = p.generate(PROMPT, num_inference_steps=3, seed=1, return_latents=True, **GEOM)
    ra = p.stats["rank_agreement"]
    assert ra["ok"] and ra["world_size"] == 1 and ra["disagreeing_ranks"] == []
    sha = ra["sha256_rank0"]["video_rows"]
    assert (
        p._rank_agreement(out["video_rows"].clone(), out["audio_rows"])["sha256_rank0"][
            "video_rows"
        ]
        == sha
    )
    bumped = out["video_rows"].clone()
    bumped.view(-1)[0] += 1e-3
    assert p._rank_agreement(bumped, out["audio_rows"])["sha256_rank0"]["video_rows"] != sha


def test_schedule_auto_pins_4step_to_linspace(vllm_single_rank, h3_tiny, tmp_path):
    """auto: the Preview-v1 4-step contract (no scheduler shifts) keeps the linspace grid its accepted gate ran on;
    a contract that declares shifts (8-Step-V2 and later) runs its trained ladder."""
    import json
    import shutil

    p = _pipe(h3_tiny)
    assert p.schedule == "linspace"
    dst = tmp_path / "with-shifts"
    shutil.copytree(h3_tiny, dst, symlinks=True)
    c = json.load(open(dst / "fastvideo_inference.json"))
    c.update(video_scheduler_shift=12.0, audio_scheduler_shift=3.0)
    json.dump(c, open(dst / "fastvideo_inference.json", "w"))
    assert _pipe(str(dst)).schedule == "contract"
    assert _pipe(h3_tiny, schedule="contract").schedule == "contract"


def test_denoise_matches_reference_loop(vllm_single_rank, h3_tiny):
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import (
        build_layout,
        draw_noise,
        load_schedulers,
    )

    p = _pipe(h3_tiny)
    out = p.generate(PROMPT, num_inference_steps=5, seed=3, return_latents=True, **GEOM)

    ref = (
        MiniMaxH3Transformer3DModel.from_pretrained(h3_tiny, subfolder="transformer").float().eval()
    )
    embeds = p._encode(PROMPT).float()
    layout = build_layout(embeds.shape[1], **GEOM)
    v, a = draw_noise(layout, torch.Generator().manual_seed(3))
    sv, sa = load_schedulers(
        h3_tiny, 5
    )  # tiny copies the 4-step contract (no shifts) -> "auto" = linspace
    with torch.no_grad():
        for i in range(len(sv.timesteps)):
            tv, ta = float(sv.timesteps[i]), float(sa.timesteps[i])
            ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                layout.video_indices,
                layout.audio_indices,
                0,
                0,
                layout.num_text_tokens,
                tv,
                ta,
                tv,
                1.0,
            )
            vel_v, vel_a = ref(
                v[None],
                a[None],
                embeds,
                ts,
                ts_idx,
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                return_dict=False,
            )
            v = sv.step(vel_v[0].float(), sv.timesteps[i], v, return_dict=False)[0]
            a = sa.step(vel_a[0].float(), sa.timesteps[i], a, return_dict=False)[0]
    rel_v = ((out["video_rows"] - v).norm() / v.norm()).item()
    rel_a = ((out["audio_rows"] - a).norm() / a.norm()).item()
    assert rel_v < 1e-5 and rel_a < 1e-5, (rel_v, rel_a)
