# SPDX-License-Identifier: Apache-2.0
"""Tier-2 accuracy: one full denoising step through the served pipeline vs diffusers.

docs/model-dev/onboarding-models.md, Step 4 tier 2: the whole pipeline for ``num_steps = 1``
(text encoding, DiT with CFG, scheduler update; VAE decode excluded) against diffusers' CPU
``WanPipeline`` denoised latent, three-way: FP32 CPU, BF16 CPU, BF16 Neuron. All legs start from
the same injected FP32 initial latent (a fixed seed does not give the same noise across devices).

The Neuron leg is the served path: ``examples/wan2_2/run_ti2v.py --steps 1 --latents
--latents-in <noise.pt>`` in a subprocess, so the multi-rank worker does the Lite and collective
setup. With ``VLLM_NEURON_CPU_MODE=1`` the same subprocess runs eagerly on CPU, which checks the
harness itself on a tiny checkpoint.

    WAN22_MODEL=<checkpoint dir> pytest test/neuron/test_wan2_2_ti2v_pipeline_accuracy.py

Optional: ``WAN22_SERVED_ARGS`` (extra runner flags, e.g. the parallel layout, default TP4+SP on
``0,1,2,3``), ``WAN22_SHAPE`` (``HxWxF``, default ``128x128x9``), ``WAN22_GUIDANCE`` (default 5).
"""

from __future__ import annotations

import gc
import os
import shlex
import subprocess
import sys

import pytest
import torch

MODEL = os.environ.get("WAN22_MODEL", "")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "examples", "wan2_2", "run_ti2v.py")
PROMPT = "A fluffy orange cat walking gracefully across a sunny garden path, high quality"


def _shape() -> tuple[int, int, int]:
    h, w, f = (int(v) for v in os.environ.get("WAN22_SHAPE", "128x128x9").split("x"))
    return h, w, f


def _cpu_mode() -> bool:
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") not in ("", "0")


def initial_latents(model_path: str, out_path: str, seed: int = 42) -> torch.Tensor:
    """One FP32 CPU latent for the checkpoint's VAE geometry, saved for every leg."""
    import json

    with open(os.path.join(model_path, "vae", "config.json")) as f:
        vae = json.load(f)
    h, w, nf = _shape()
    spatial = int(vae.get("scale_factor_spatial", 8))
    temporal = int(vae.get("scale_factor_temporal", 4))
    shape = (1, int(vae["z_dim"]), (nf - 1) // temporal + 1, h // spatial, w // spatial)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=g, dtype=torch.float32)
    torch.save(noise, out_path)
    return noise


def diffusers_latent(model_path: str, noise: torch.Tensor, dtype: torch.dtype, guidance: float):
    """Diffusers' WanPipeline on CPU, one step, latent output (loaded fresh per dtype)."""
    from diffusers import WanPipeline

    h, w, nf = _shape()
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    with torch.inference_mode():
        out = pipe(
            prompt=PROMPT,
            negative_prompt="",
            height=h,
            width=w,
            num_frames=nf,
            num_inference_steps=1,
            guidance_scale=guidance,
            latents=noise.to(dtype),
            output_type="latent",
        ).frames
    out = out.float().cpu()
    del pipe
    gc.collect()
    return out


def served_latent(
    model_path: str,
    noise_path: str,
    out_stem: str,
    guidance: float,
    extra_args: tuple[str, ...] = (),
) -> torch.Tensor:
    h, w, nf = _shape()
    if _cpu_mode():
        layout = ["--tp", "1", "--sp", "off", "--devices", "0", "--eager"]
    else:
        layout = shlex.split(
            os.environ.get("WAN22_SERVED_ARGS", "--tp 4 --sp on --cp 1 --devices 0,1,2,3")
        )
    cmd = [
        sys.executable, RUNNER, "--model-path", model_path, "--prompt", PROMPT,
        "--height", str(h), "--width", str(w), "--num-frames", str(nf),
        "--steps", "1", "--guidance-scale", str(guidance), "--cfg-parallel", "1",
        "--vae-pp", "1", *layout,
        "--latents-in", noise_path, "--latents", "--output", out_stem, *extra_args,
    ]  # fmt: skip
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5400)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return torch.load(out_stem + ".latents.pt").float()


def _three_way_result(fp32, bf16, neuron, name):
    """``assert_close_three_way``'s report, returned (pass or fail) instead of raised on a fail."""
    from vllm_neuron.accuracy.testing import assert_close_three_way

    try:
        return assert_close_three_way(fp32, bf16, neuron, name=name)
    except AssertionError as e:
        print(e)
        return type("FailedThreeWay", (), {"passed": False, "name": name, "report": str(e)})()


def _rel_cos(a, b):
    return (
        float((a - b).norm() / b.norm()),
        float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)),
    )


def run_three_way(model_path: str, work_dir: str, guidance: float):
    os.makedirs(work_dir, exist_ok=True)
    noise_path = os.path.join(work_dir, "noise.pt")
    noise = initial_latents(model_path, noise_path)
    neuron = served_latent(model_path, noise_path, os.path.join(work_dir, "served"), guidance)
    fp32 = diffusers_latent(model_path, noise, torch.float32, guidance)
    bf16 = diffusers_latent(model_path, noise, torch.bfloat16, guidance)
    for name, leg in (("bf16_cpu", bf16), ("bf16_neuron", neuron)):
        rel, cos = _rel_cos(leg, fp32)
        print(f"[g={guidance:g}] {name} vs fp32_cpu: rel_l2={rel:.4f} cos={cos:.5f}")
    return _three_way_result(fp32, bf16, neuron, f"wan22_ti2v_step1_g{guidance:g}")


@pytest.mark.skipif(not os.path.isdir(os.path.join(MODEL, "transformer")), reason="set WAN22_MODEL")
def test_single_step_matches_diffusers(tmp_path):
    guidance = float(os.environ.get("WAN22_GUIDANCE", "5"))
    result = run_three_way(MODEL, str(tmp_path), guidance)
    print(result)
    assert result.passed, result


if __name__ == "__main__":
    for g in [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,5").split(",")]:
        print(run_three_way(sys.argv[1], sys.argv[2], g))
