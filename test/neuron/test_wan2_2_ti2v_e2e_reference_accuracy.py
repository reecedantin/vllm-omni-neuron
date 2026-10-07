# SPDX-License-Identifier: Apache-2.0
"""Tier-3 accuracy against an independent reference: FastWan DMD2 end to end vs diffusers on CPU.

The served Neuron generation (text encode, the three DMD2 steps, VAE decode) is compared with the
same generation computed on CPU from diffusers' own components -- ``WanPipeline``'s UMT5 text
encoder, ``WanTransformer3DModel`` and ``AutoencoderKLWan`` -- driven by FastVideo's DMD2 rule
(timesteps 1000/757/522, the shift-8 flow-matching training table, clean-latent prediction
``x0 = x_t - sigma_t * v`` and re-noising ``(1 - sigma_next) * x0 + sigma_next * eps``). None of the
Neuron model code is used by the reference.

All legs share the injected FP32 initial latent and the per-step re-noising draws
(``dmd_renoise_draw``, the same convention the served pipeline uses). Two references are run,
FP32 and BF16; the BF16 one sets the floor of ordinary low-precision error. Gate:

    latent:  rel_l2(neuron, fp32)      <= 2 * rel_l2(bf16, fp32)      + 0.005
    frames:  1 - SSIM(neuron, fp32)    <= 2 * (1 - SSIM(bf16, fp32))  + 0.005   (mean per frame)

Shape: the native 704x1280, 17 frames, so the CPU references take ~10 minutes. The DMD2 student is
very sensitive to its inputs (at t=757 a 1e-3 input perturbation moves the prediction ~1% at
704x1280 but ~6% at 448x256, and a one-unit timestep shift moves it ~45%), so the BF16 floor is
wide: measured at 704x1280x17, rel-L2 0.32 and SSIM 0.83 vs FP32; at 448x256x17 it is 0.77 / 0.28,
i.e. two unrelated samples, which makes the gate meaningless there. At 704x1280x17 the DiT
sequence is 5 x 22 x 40 = 4400 tokens (1100 per CP rank, no CP padding) and the VAE decode is the
same 4 x 7 tile grid as the 704x1280x121 clip (28 tiles, every patch-parallel rank decodes one or
two). ``WAN22_REF_SHAPE=HxWxF`` overrides the shape (e.g. a quick CPU-mode check on a tiny
checkpoint).

The work is split into steps that cache their outputs in the work directory, so a driver can run
each one under its own timeout and the comparison reuses whatever already exists:

    served   one served generation: decoded frames + the denoised latent (dumped before decode)
    ref32    diffusers on CPU, FP32
    ref16    diffusers on CPU, BF16 (the floor)
    compare  the gate (computes any missing step first)

    WAN22_MODEL=<FastWan2.2-TI2V-5B checkpoint> WAN22_REF_WORKDIR=<dir> \\
        pytest -s test/neuron/test_wan2_2_ti2v_e2e_reference_accuracy.py
    WAN22_MODEL=<checkpoint> \\
        python -u test/neuron/test_wan2_2_ti2v_e2e_reference_accuracy.py <work dir> [step ...]

``WAN22_SERVED_ARGS`` sets the served parallel layout (default TP4 x CP4 x VAE-pp 16 on 16 cores;
with ``VLLM_NEURON_CPU_MODE=1`` the default is one eager CPU rank, and any layout runs eagerly).
The served subprocess streams its output and is bounded by ``WAN22_STEP_TIMEOUT`` (default
1800 s); its workers print every thread's stack each ``WAN_STACK_DUMP_S`` seconds (default 600),
so a stalled collective shows where it stalled.
"""

from __future__ import annotations

import gc
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from test.neuron.test_wan2_2_ti2v_e2e_accuracy import DEFAULT_LAYOUT, ssim_per_frame

MODEL = os.environ.get("WAN22_MODEL", "")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(ROOT, "examples", "wan2_2", "run_ti2v.py")
SETTINGS = {
    "prompt": "A fluffy orange cat walking gracefully across a sunny garden path, high quality",
    "height": 704,
    "width": 1280,
    "num_frames": 17,
    "seed": 42,
}
if os.environ.get("WAN22_REF_SHAPE"):
    _h, _w, _f = (int(v) for v in os.environ["WAN22_REF_SHAPE"].lower().split("x"))
    SETTINGS.update(height=_h, width=_w, num_frames=_f)
# FastWan2.2-TI2V-5B contract (FastVideo FastWan2_2_TI2V_5B_Config, DMD_TRAINING_NOISE_SHIFT).
DMD_TIMESTEPS = (1000, 757, 522)
DMD_SHIFT = 8.0


def _cpu_mode() -> bool:
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") not in ("", "0")


def _step_timeout() -> int:
    return int(os.environ.get("WAN22_STEP_TIMEOUT", "1800"))


def dmd_renoise_draw(seed: int, step: int, shape) -> torch.Tensor:
    """Re-noising draw between DMD2 steps; ``NeuronWanDMDPipeline._dmd_noise`` uses the same
    convention, so both trajectories see identical noise."""
    g = torch.Generator(device="cpu").manual_seed(seed * 1000 + 17 + step)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def initial_noise(model_path: str, path: str) -> torch.Tensor:
    with open(os.path.join(model_path, "vae", "config.json")) as f:
        vae = json.load(f)
    sp = int(vae.get("scale_factor_spatial", 16))
    tp = int(vae.get("scale_factor_temporal", 4))
    shape = (
        1,
        int(vae["z_dim"]),
        (SETTINGS["num_frames"] - 1) // tp + 1,
        SETTINGS["height"] // sp,
        SETTINGS["width"] // sp,
    )
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(1234), dtype=torch.float32)
    torch.save(noise, path)
    return noise


def served(model_path: str, noise_path: str, out_stem: str):
    """One served generation from the injected noise: (denoised latent, frames [T, H, W, 3]).

    The latent is dumped by the pipeline on its output rank right before VAE decode
    (``WAN_DUMP_FINAL_LATENTS``), so both come from the same request. Output is streamed, not
    captured, so a stall is visible while it happens."""
    layout = shlex.split(os.environ.get("WAN22_SERVED_ARGS", ""))
    if _cpu_mode():
        default = ["--tp", "1", "--sp", "off", "--cp", "1", "--vae-pp", "1", "--devices", "0"]
        layout = (layout or default) + ["--eager"]
    elif not layout:
        layout = shlex.split(DEFAULT_LAYOUT)
    latent_path = out_stem + ".final_latents.pt"
    cmd = [
        sys.executable, "-u", RUNNER, "--model-path", model_path, "--prompt", SETTINGS["prompt"],
        "--height", str(SETTINGS["height"]), "--width", str(SETTINGS["width"]),
        "--num-frames", str(SETTINGS["num_frames"]), "--seed", str(SETTINGS["seed"]),
        "--cfg-parallel", "1", "--latents-in", noise_path, *layout,
        "--output", out_stem + ".mp4", "--save-frames",
    ]  # fmt: skip
    env = dict(
        os.environ,
        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
        WAN_DUMP_FINAL_LATENTS=latent_path,
    )
    env.setdefault("WAN_STACK_DUMP_S", "600")
    print("SERVED " + shlex.join(cmd), flush=True)
    # Own process group, so a timeout takes the stage workers down with the runner instead of
    # leaving them holding cores.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=_step_timeout())
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise AssertionError(f"served run exceeded {_step_timeout()} s (stacks above)") from None
    assert rc == 0, f"served run failed rc={rc} (output above)"
    return torch.load(latent_path).float(), np.load(out_stem + ".mp4.npy").astype(np.float32)


def diffusers_dmd2(model_path: str, noise: torch.Tensor, dtype: torch.dtype):
    """FastWan DMD2 with diffusers components on CPU: (final latent, frames [T, H, W, 3])."""
    from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline

    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    table = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=DMD_SHIFT)
    t_table = table.timesteps.double()
    s_table = table.sigmas[: len(t_table)].double()

    def sigma(t) -> float:  # FastVideo's nearest-timestep lookup into the training table
        return float(s_table[int(torch.argmin((t_table - float(t)).abs()))])

    with torch.inference_mode():
        embeds, _ = pipe.encode_prompt(
            prompt=SETTINGS["prompt"],
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device="cpu",
            dtype=dtype,
        )
        latents = noise.clone()
        for i, t in enumerate(DMD_TIMESTEPS):
            flow = pipe.transformer(
                hidden_states=latents.to(dtype),
                timestep=torch.tensor([float(t)]),
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0].float()
            x0 = latents.float() - sigma(t) * flow
            if i + 1 < len(DMD_TIMESTEPS):
                s_next = sigma(DMD_TIMESTEPS[i + 1])
                eps = dmd_renoise_draw(SETTINGS["seed"], i, tuple(latents.shape))
                latents = (1.0 - s_next) * x0 + s_next * eps
            else:
                latents = x0
        cfg = pipe.vae.config
        z = latents.to(pipe.vae.dtype)
        mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(z.dtype)
        std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(z.dtype)
        video = pipe.vae.decode(z * std + mean, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video.float(), output_type="np")[0]
    del pipe
    gc.collect()
    return latents.float(), np.asarray(frames, dtype=np.float32)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _cached(work_dir: str, name: str, compute):
    """(latent, frames) for one leg, computed once and kept in ``work_dir``."""
    lat_path = os.path.join(work_dir, name + ".latents.pt")
    frames_path = os.path.join(work_dir, name + ".frames.npy")
    if not (os.path.isfile(lat_path) and os.path.isfile(frames_path)):
        t0 = time.time()
        lat, frames = compute()
        torch.save(lat, lat_path)
        np.save(frames_path, frames)
        print(f"STEP {name} done in {time.time() - t0:.0f} s", flush=True)
    return torch.load(lat_path).float(), np.load(frames_path).astype(np.float32)


def run_step(model_path: str, work_dir: str, step: str):
    os.makedirs(work_dir, exist_ok=True)
    noise_path = os.path.join(work_dir, "noise.pt")
    if not os.path.isfile(noise_path):
        initial_noise(model_path, noise_path)
    noise = torch.load(noise_path)
    if step == "served":
        stem = os.path.join(work_dir, "served")
        return _cached(work_dir, "neuron", lambda: served(model_path, noise_path, stem))
    if step == "ref32":
        return _cached(work_dir, "fp32", lambda: diffusers_dmd2(model_path, noise, torch.float32))
    if step == "ref16":
        return _cached(work_dir, "bf16", lambda: diffusers_dmd2(model_path, noise, torch.bfloat16))
    raise ValueError(f"unknown step {step!r}")


def reference_check(model_path: str, work_dir: str) -> dict:
    dev_lat, dev_frames = run_step(model_path, work_dir, "served")
    lat32, frames32 = run_step(model_path, work_dir, "ref32")
    lat16, frames16 = run_step(model_path, work_dir, "ref16")
    ssim_dev = ssim_per_frame(dev_frames, frames32)
    ssim_bf16 = ssim_per_frame(frames16, frames32)
    res = {
        "latent_rel_l2": {"bf16_cpu": _rel_l2(lat16, lat32), "neuron": _rel_l2(dev_lat, lat32)},
        "ssim_vs_fp32": {
            "bf16_cpu_mean": float(ssim_bf16.mean()),
            "neuron_mean": float(ssim_dev.mean()),
            "neuron_min": float(ssim_dev.min()),
        },
    }
    res["bars"] = {
        "latent_rel_l2_max": 2 * res["latent_rel_l2"]["bf16_cpu"] + 0.005,
        "one_minus_ssim_max": 2 * (1.0 - res["ssim_vs_fp32"]["bf16_cpu_mean"]) + 0.005,
    }
    res["latent_ok"] = res["latent_rel_l2"]["neuron"] <= res["bars"]["latent_rel_l2_max"]
    res["frames_ok"] = 1.0 - res["ssim_vs_fp32"]["neuron_mean"] <= res["bars"]["one_minus_ssim_max"]
    res["passed"] = res["latent_ok"] and res["frames_ok"]
    print("E2E_REFERENCE " + json.dumps(res), flush=True)
    return res


def _is_dmd2(path: str) -> bool:
    index = os.path.join(path, "model_index.json")
    if not os.path.isfile(index):
        return False
    with open(index) as f:
        return json.load(f).get("_class_name") == "WanDMDPipeline"


@pytest.mark.skipif(not _is_dmd2(MODEL), reason="set WAN22_MODEL to a FastWan DMD2 checkpoint")
def test_e2e_matches_reference(tmp_path):
    res = reference_check(MODEL, os.environ.get("WAN22_REF_WORKDIR") or str(tmp_path))
    assert res["passed"], res


if __name__ == "__main__":
    # python -u <this file> <work dir> [served|ref32|ref16|compare ...]; WAN22_MODEL = checkpoint
    work, steps = sys.argv[1], sys.argv[2:] or ["compare"]
    for name in steps:
        if name == "compare":
            sys.exit(0 if reference_check(MODEL, work)["passed"] else 1)
        run_step(MODEL, work, name)
