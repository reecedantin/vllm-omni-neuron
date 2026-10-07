# SPDX-License-Identifier: Apache-2.0
"""Base MiniMax-H3 (50-step, T2VA) request contract on CPU with the random-weight structure checkpoint.

The base release carries no FastVideo ``fastvideo_inference.json``; the structure checkpoint without that file is a
base checkpoint. The contract mirrored is vLLM-Omni v0.30.0's MiniMax-H3 pipeline:

* ``num_inference_steps`` counts denoiser evaluations (default 50); the sigma boundaries are
  ``linspace(1, 0, N + 1)`` through the exponential shift, video ``flow_shift`` 12 and ``audio_flow_shift`` 3 by
  default (``time_request._time_shift_sigmas``);
* one conditional forward per step: the checkpoint is guidance-distilled (no negative branch, ``do_true_cfg`` is
  always False), so ``guidance_scale`` and ``negative_prompt`` do not change the output;
* the update is ``x0 = x_t + sigma(t) * v`` then the Euler eta=0 blend ``r * x_t + (1 - r) * x0``,
  ``r = sigma_next / sigma`` (``scheduling_minimax_h3_euler_ancestral``);
* tasks other than t2va are rejected (FL2VA / Ref2VA not supported yet).
"""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

pytest_plugins = [f"{__package__}.test_minimax_h3_tiny_ckpt"]

GEOM = dict(height=64, width=96, num_frames=22)
PROMPT = "A golden retriever runs through the surf at sunset, waves crashing."


def _upstream_sigmas(num_steps: int, shift: float) -> list[float]:
    """vLLM-Omni v0.30.0 ``minimax_h3/time_request.py::_time_shift_sigmas`` (no base_schedule), verbatim math."""
    base = torch.linspace(1.0, 0.0, int(num_steps) + 1, device="cpu", dtype=torch.float32)
    shifted = float(shift) * base / (1 + (float(shift) - 1) * base)
    return [float(v) for v in shifted.tolist()]


@pytest.fixture(scope="session")
def h3_base_tiny(h3_tiny, tmp_path_factory) -> str:
    """The structure checkpoint as a base release: everything but FastVideo's sampling contract."""
    dst = tmp_path_factory.mktemp("h3base") / "tiny-h3-base"
    shutil.copytree(
        h3_tiny, dst, symlinks=True, ignore=shutil.ignore_patterns("fastvideo_inference.json")
    )
    return str(dst)


def _pipe(weights, dtype=torch.float32, **mc):
    from vllm_omni_neuron.diffusion.models.minimax_h3 import NeuronMiniMaxH3Pipeline

    od = SimpleNamespace(model=weights, dtype=dtype, model_config=mc)
    p = NeuronMiniMaxH3Pipeline(od_config=od)
    p.load_weights()
    return p


def _req(steps=None, seed=5, **extra_sp):
    extra = extra_sp.pop("extra_args", {})
    sp = SimpleNamespace(
        num_inference_steps=steps, seed=seed, generator=None, extra_args=extra, **GEOM, **extra_sp
    )
    return SimpleNamespace(prompts=[PROMPT], sampling_params=sp)


def _counting(p):
    """Wrap the pipeline's DiT callable and record the video timestep of every call."""
    calls = []

    class Counting(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            calls.append(float(args[3][0]))
            return self.inner(*args)

    p._dit_fn = Counting(p._dit_fn)
    return calls


@pytest.mark.parametrize("steps,shift", [(50, 12.0), (50, 3.0), (8, 12.0), (1, 6.0)])
def test_base_sigmas_match_upstream(steps, shift):
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import base_sigmas

    got = base_sigmas(steps, shift)
    assert got == _upstream_sigmas(steps, shift)
    assert len(got) == steps + 1 and got[0] == 1.0 and got[-1] == 0.0


def test_contract_detection(vllm_single_rank, h3_tiny, h3_base_tiny):
    """FastH3 keeps its contract (5 grid points = 4 forwards); the base defaults to 50 evaluations, shifts 12 / 3."""
    fast = _pipe(h3_tiny)
    assert not fast.is_base and fast.default_steps == 5
    base = _pipe(h3_base_tiny)
    assert base.is_base and base.default_steps == 50
    sv, sa = base._base_request_sigmas(50, {})
    assert sv == _upstream_sigmas(50, 12.0) and sa == _upstream_sigmas(50, 3.0)
    sv, sa = base._base_request_sigmas(50, {"flow_shift": 6.0, "audio_flow_shift": 2.0})
    assert sv == _upstream_sigmas(50, 6.0) and sa == _upstream_sigmas(50, 2.0)
    for task in ("fl2va", "ref2va"):
        with pytest.raises(ValueError, match="not supported"):
            base._base_request_sigmas(50, {"task": task})


def test_default_request_runs_50_conditional_forwards(vllm_single_rank, h3_base_tiny):
    """The default base request: 50 DiT calls, one per sigma interval, at t = 1 - sigma (upstream's timesteps)."""
    p = _pipe(h3_base_tiny)
    calls = _counting(p)
    out = p.forward(_req())
    video, audio = out.output
    assert len(calls) == 50 and len(p.stats["forward_s"]) == 50
    want = [1.0 - s for s in _upstream_sigmas(50, 12.0)[:-1]]
    assert torch.allclose(torch.tensor(calls), torch.tensor(want), atol=0, rtol=0)
    assert video.shape == (1, 3, 22, 64, 96) and audio.shape[:2] == (1, 2)


def test_guidance_has_no_negative_branch(vllm_single_rank, h3_base_tiny):
    """Guidance-distilled: guidance_scale / negative_prompt add no forward and do not change a bit of the output."""
    p = _pipe(h3_base_tiny)
    calls = _counting(p)
    a = p.forward(_req(steps=3)).output
    n_plain = len(calls)
    b = p.forward(_req(steps=3, guidance_scale=5.0, negative_prompt="blurry, low quality")).output
    assert n_plain == 3 and len(calls) == 6
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_base_denoise_matches_upstream_loop(vllm_single_rank, h3_base_tiny):
    """generate() with the base sigmas == the upstream update rule (x0 + Euler eta=0) driven by diffusers'
    reference transformer in fp32, with the same noise and layout."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise

    steps = 6
    p = _pipe(h3_base_tiny)
    sig = p._base_request_sigmas(steps, {})
    out = p.generate(
        PROMPT, num_inference_steps=steps + 1, seed=3, return_latents=True, sigmas=sig, **GEOM
    )
    ref = (
        MiniMaxH3Transformer3DModel.from_pretrained(h3_base_tiny, subfolder="transformer")
        .float()
        .eval()
    )
    embeds = p._encode(PROMPT).float()
    layout = build_layout(embeds.shape[1], **GEOM)
    v, a = draw_noise(layout, torch.Generator().manual_seed(3))
    sv, sa = _upstream_sigmas(steps, 12.0), _upstream_sigmas(steps, 3.0)

    def euler(x, vel, s, s_next):
        x0 = x + s * vel
        r = torch.tensor(s_next, dtype=torch.float32) / torch.tensor(s, dtype=torch.float32)
        return r * x + (1.0 - r) * x0

    with torch.no_grad():
        for i in range(steps):
            tv, ta = 1.0 - sv[i], 1.0 - sa[i]
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
            v = euler(v, vel_v[0].float(), 1.0 - tv, sv[i + 1])
            a = euler(a, vel_a[0].float(), 1.0 - ta, sa[i + 1])
    rel_v = ((out["video_rows"] - v).norm() / v.norm()).item()
    rel_a = ((out["audio_rows"] - a).norm() / a.norm()).item()
    assert rel_v < 1e-5 and rel_a < 1e-5, (rel_v, rel_a)


def test_step_capture(vllm_single_rank, h3_base_tiny, tmp_path, monkeypatch):
    """MINIMAX_H3_CAPTURE records the chosen steps' exact inputs and velocities; a replay reproduces them."""
    path = tmp_path / "cap.pt"
    monkeypatch.setenv("MINIMAX_H3_CAPTURE", str(path))
    monkeypatch.setenv("MINIMAX_H3_CAPTURE_STEPS", "0,2")
    p = _pipe(h3_base_tiny)
    p.forward(_req(steps=3))
    cap = torch.load(path, weights_only=False)
    assert [s["step"] for s in cap["steps"]] == [0, 2]
    assert cap["sigmas_video"] == pytest.approx(_upstream_sigmas(3, 12.0), abs=0)
    s2 = cap["steps"][1]
    assert set(s2) >= {"video_rows", "audio_rows", "vel_video", "vel_audio", "t_video", "t_audio"}
    assert "MINIMAX_H3_CAPTURE" not in os.environ  # one request per capture
