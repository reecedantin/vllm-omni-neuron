# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 full-size gate on Neuron: the served pipeline class at the headline shape (512x768x121,
8 distilled steps) vs the diffusers CPU pipeline, every decoded frame and the waveform.

Compared against the CPU fp32 reference (diffusers ``LTX2Pipeline`` + the vendored transformer, fp32
VAE / audio VAE / vocoder), with CPU bf16 (the same pipeline in bf16) as the dtype-only band:

* denoised latents (video, audio), captured where the pipeline unpacks them, before the decode;
* every decoded video frame: rel-L2, PSNR and SSIM (luma, Gaussian window) per frame;
* the 48 kHz waveform.

Pass (k = 2): device error <= 2 x (CPU bf16 error) + 0.005 for the latents, the waveform, the whole
clip and EVERY frame (per-frame band), and every rank agrees: each rank's denoised latents (TP x CP,
captured on every rank) are digested and compared with rank 0's by the shared all-rank check
(``vllm_omni_neuron.testing.check_rank_agreement``, rtol 1e-3; the report says whether they are
bit-exact). The device run is the served class with the served
defaults: device-tiled VAE over all ranks, vocoder spans on the host, device Gemma text encoder.

Three steps (the reference needs ~150 GB of host RAM and no cores; the device run needs the cores):

    python -m test.neuron.test_ltx2_fullsize_gate_device --model <dir> --mode reference --out <dir>
    torchrun --nproc_per_node 16 -m test.neuron.test_ltx2_fullsize_gate_device --model <dir> \\
        --mode device --out <dir> --cp 4          # TP = nproc / cp
    python -m test.neuron.test_ltx2_fullsize_gate_device --model <dir> --mode compare --out <dir>

``--mode device`` compares too when the reference is already in ``--out``. As a pytest it skips
unless ``LTX2_GATE_OUT`` points at a directory with both results.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pytest
import torch

from test.neuron.test_ltx2_pipeline_parity_device import (
    PROMPT,
    _np,
    _reference_pipeline,
    _rel_cos,
    device_request,
    setup_device_pipeline,
)

SHAPE = dict(height=512, width=768, num_frames=121, frame_rate=24.0, seed=42, steps=8)
K = 2.0  # bar = K x CPU-bf16 error + EPS
EPS = 0.005
RANK_TOL = 1e-3


def _pipeline_kwargs(shape: dict) -> dict:
    return dict(
        prompt=PROMPT,
        height=shape["height"],
        width=shape["width"],
        num_frames=shape["num_frames"],
        frame_rate=shape["frame_rate"],
        num_inference_steps=shape["steps"],
        guidance_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        audio_guidance_scale=1.0,
        audio_stg_scale=0.0,
        audio_modality_scale=1.0,
        generator=torch.Generator().manual_seed(shape["seed"]),
        output_type="np",
        use_cross_timestep=True,
        return_dict=True,
    )


def capture_latents(pipe) -> dict:
    """Record the denoised latents where the diffusers pipeline unpacks them (the last call wins):
    video before the decode-noise / denormalize step, audio after its denormalize. Works on every
    rank: the non-output ranks stop at ``output_type="latent"`` but unpack the same way."""
    seen: dict = {}
    for name, key in (("_unpack_latents", "video"), ("_unpack_audio_latents", "audio")):
        fn = getattr(pipe, name)

        def wrapped(*args, _fn=fn, _key=key, **kwargs):
            out = _fn(*args, **kwargs)
            seen[_key] = out.detach().float().cpu().clone()
            return out

        setattr(pipe, name, wrapped)
    return seen


def _u8(frames) -> np.ndarray:
    v = _np(frames)
    if v.dtype != np.uint8:
        v = (np.clip(v, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return v


def _save(out: str, tag: str, frames, audio, lat: dict) -> None:
    np.save(os.path.join(out, f"{tag}_frames_u8.npy"), _u8(frames))
    np.save(os.path.join(out, f"{tag}_wav.npy"), _np(audio).astype(np.float32))
    for k, v in lat.items():
        np.save(os.path.join(out, f"{tag}_latent_{k}.npy"), _np(v).astype(np.float32))


def run_reference(model: str, out: str, shape: dict, dtypes: list[str], threads: int) -> None:
    os.makedirs(out, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    meta = {"shape": shape, "threads": torch.get_num_threads()}
    for name in dtypes:
        pipe = _reference_pipeline(model, {"fp32": torch.float32, "bf16": torch.bfloat16}[name])
        lat = capture_latents(pipe)
        t0 = time.time()
        with torch.no_grad():
            o = pipe(**_pipeline_kwargs(shape))
        meta[f"{name}_s"] = round(time.time() - t0, 1)
        _save(out, f"ref_{name}", o.frames[0], o.audio[0], lat)
        del pipe, o
        print(f"REFERENCE {name} {meta[f'{name}_s']} s", flush=True)
    with open(os.path.join(out, "reference.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("REFERENCE_DONE " + json.dumps(meta), flush=True)


def dump_transformer_steps(transformer, out: str, steps) -> None:
    """Teacher-forced step check, device side (long shapes with no affordable CPU pipeline run):
    at the listed denoise steps, save the transformer's inputs (as the diffusers pipeline passes
    them) and the device's velocities to ``<out>/step_<i>.pt``; ``--mode replay`` reruns those
    exact inputs through the CPU reference transformer in fp32 and bf16."""
    want, state = set(steps), {"i": 0}
    fwd = transformer.forward

    def forward(*args, **kwargs):
        i = state["i"]
        state["i"] += 1
        o = fwd(*args, **kwargs)
        if i in want:
            keep = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in kwargs.items()
                if (torch.is_tensor(v) or isinstance(v, (int, float, bool)) or v is None)
                and k != "return_dict"
            }
            torch.save(
                {"step": i, "kwargs": keep, "out": [t.detach().float().cpu() for t in o[:2]]},
                os.path.join(out, f"step_{i}.pt"),
            )
        return o

    transformer.forward = forward


def replay_kwargs(kwargs: dict, dtype) -> dict:
    """The dumped transformer inputs as the CPU reference pipeline in ``dtype`` would pass them:
    the latents and text states take ``dtype``; the timestep / sigma and the RoPE coordinates stay
    fp32 as upstream keeps them (coordinates up to ~1500 rounded to bf16 corrupt the RoPE)."""
    keep = {"timestep", "sigma", "audio_timestep", "audio_sigma", "video_coords", "audio_coords"}
    return {
        k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() and k not in keep else v)
        for k, v in kwargs.items()
    }


def run_replay(model: str, out: str, threads: int) -> dict:
    """Teacher-forced step check, CPU side: every ``step_<i>.pt`` through the vendored reference
    transformer in fp32 and in bf16; per step and modality, device rel-L2 vs fp32 must be within
    ``K x (bf16 rel-L2) + EPS``. Writes ``replay.json``."""
    import glob

    from vllm_omni_neuron.diffusion.models.ltx2._vendor.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )

    if threads:
        torch.set_num_threads(threads)
    files = sorted(glob.glob(os.path.join(out, "step_*.pt")))
    dumps = [torch.load(f, weights_only=False) for f in files]
    ref: dict = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        tr = LTX2VideoTransformer3DModel.from_pretrained(
            os.path.join(model, "transformer"), torch_dtype=dtype
        ).eval()
        for d in dumps:
            kw = replay_kwargs(d["kwargs"], dtype)
            t0 = time.time()
            with torch.no_grad():
                o = tr(**kw, return_dict=False)
            ref[(name, d["step"])] = [t.float() for t in o[:2]]
            print(f"REPLAY {name} step {d['step']} {time.time() - t0:.0f} s", flush=True)
        del tr
    report: dict = {"k": K, "eps": EPS, "steps": {}}
    ok = True
    for d in dumps:
        row = {}
        for j, mod in enumerate(("video", "audio")):
            r32 = ref[("fp32", d["step"])][j].numpy()
            row[mod] = _tier(r32, ref[("bf16", d["step"])][j].numpy(), d["out"][j].numpy())
            ok &= row[mod]["pass"]
        report["steps"][d["step"]] = row
    report["pass"] = bool(ok and dumps)
    with open(os.path.join(out, "replay.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def _rank_agreement(lat: dict) -> dict:
    """Every rank's denoised latents (replicated after the CP gather) vs rank 0's, with the shared
    all-rank check (digests gathered over the world host group). Collective: every rank calls it."""
    from vllm_omni_neuron.testing import check_rank_agreement

    rep = check_rank_agreement(dict(lat), rtol=RANK_TOL)
    exact = all(v == 0.0 for v in rep.max_rel.values())
    return {
        "world": rep.world_size,
        "bit_identical": rep.world_size if (rep.ok and exact) else None,
        "max_rel": max(rep.max_rel.values(), default=0.0),
        "pass": rep.ok,
        "summary": rep.summary(),
        "report": rep.to_json(),
    }


def run_device(model: str, out: str, shape: dict, cp: int, dump_steps=()) -> None:
    os.makedirs(out, exist_ok=True)
    pipe, rank, world = setup_device_pipeline(model, cp)
    lat = capture_latents(pipe._pipe)
    batch = device_request(
        PROMPT,
        height=shape["height"],
        width=shape["width"],
        num_frames=shape["num_frames"],
        frame_rate=shape["frame_rate"],
        num_inference_steps=shape["steps"],
        seed=shape["seed"],
        output_type="np",
    )
    t0 = time.time()
    if dump_steps and rank == 0:
        dump_transformer_steps(pipe.transformer, out, dump_steps)
    res = pipe.forward(batch)
    dt = time.time() - t0
    agree = _rank_agreement(lat)
    if rank != 0:
        return
    _save(out, "dev", res.output["video"], res.output["audio"], lat)
    meta = {"world": world, "cp": cp, "tp": world // cp, "request_s": round(dt, 1), "ranks": agree}
    with open(os.path.join(out, "device.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(
        "GATE_DEVICE "
        + json.dumps({k: v for k, v in meta.items() if k != "ranks"} | _rank_summary(agree)),
        flush=True,
    )
    if os.path.exists(os.path.join(out, "ref_bf16_frames_u8.npy")):
        report = compare(out)
        print("GATE " + json.dumps(_summary(report)), flush=True)


def _rank_summary(agree: dict) -> dict:
    return {
        "ranks_bit_identical": agree["bit_identical"],
        "ranks_max_rel": agree["max_rel"],
        "ranks_pass": agree["pass"],
        "ranks_summary": agree.get("summary"),
    }


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on the luma of two uint8 RGB frames (Gaussian window, sigma 1.5)."""
    from scipy.ndimage import gaussian_filter

    w = np.array([0.299, 0.587, 0.114])
    x, y = a.astype(np.float64) @ w, b.astype(np.float64) @ w
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    sxx = gaussian_filter(x * x, 1.5) - mx * mx
    syy = gaussian_filter(y * y, 1.5) - my * my
    sxy = gaussian_filter(x * y, 1.5) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def _tier(r32, r16, dev) -> dict:
    band, _ = _rel_cos(r32, r16)
    rel, cos = _rel_cos(r32, dev.reshape(r32.shape))
    bar = K * band + EPS
    return {
        "device_rel_l2": round(rel, 5),
        "device_cos": round(cos, 6),
        "cpu_bf16_band": round(band, 5),
        "bar": round(bar, 5),
        "pass": bool(rel <= bar),
    }


def compare(out: str) -> dict:
    load = lambda name: np.load(os.path.join(out, name))  # noqa: E731
    report: dict = {"k": K, "eps": EPS}
    for mod in ("video", "audio"):
        report[f"latent_{mod}"] = _tier(
            load(f"ref_fp32_latent_{mod}.npy"),
            load(f"ref_bf16_latent_{mod}.npy"),
            load(f"dev_latent_{mod}.npy"),
        )
    report["wav"] = _tier(load("ref_fp32_wav.npy"), load("ref_bf16_wav.npy"), load("dev_wav.npy"))
    f32, f16, dev = (load(f"{t}_frames_u8.npy") for t in ("ref_fp32", "ref_bf16", "dev"))
    assert f32.shape == dev.shape == f16.shape, (f32.shape, f16.shape, dev.shape)
    report["clip"] = _tier(f32 / 255.0, f16 / 255.0, dev / 255.0)
    frames, failed = [], []
    for i in range(f32.shape[0]):
        t = _tier(f32[i] / 255.0, f16[i] / 255.0, dev[i] / 255.0)
        row = {
            "frame": i,
            "rel": t["device_rel_l2"],
            "band": t["cpu_bf16_band"],
            "psnr": round(_psnr(f32[i], dev[i]), 2),
            "psnr_bf16": round(_psnr(f32[i], f16[i]), 2),
            "ssim": round(_ssim(f32[i], dev[i]), 4),
            "ssim_bf16": round(_ssim(f32[i], f16[i]), 4),
            "pass": t["pass"],
        }
        frames.append(row)
        if not t["pass"]:
            failed.append(i)
    report["frames"] = frames
    report["frames_failed"] = failed
    for key in ("psnr", "psnr_bf16", "ssim", "ssim_bf16"):
        vals = np.array([r[key] for r in frames], dtype=np.float64)
        report[f"{key}_min"] = round(float(vals.min()), 4)
        report[f"{key}_mean"] = round(float(vals[np.isfinite(vals)].mean()), 4)
    ranks = {}
    if os.path.exists(os.path.join(out, "device.json")):
        with open(os.path.join(out, "device.json")) as f:
            ranks = json.load(f).get("ranks") or {}
    report["ranks"] = _rank_summary(ranks) if ranks else None
    report["pass"] = bool(
        all(report[k]["pass"] for k in ("latent_video", "latent_audio", "wav", "clip"))
        and not failed
        and bool(ranks)
        and ranks["pass"]
    )
    with open(os.path.join(out, "gate.json"), "w") as f:
        json.dump(report, f, indent=1)
    _contact_sheet(out, f32, f16, dev)
    return report


def _summary(report: dict) -> dict:
    s = {k: v for k, v in report.items() if k != "frames"}
    s["frames_failed"] = len(report["frames_failed"])
    return s


def _contact_sheet(out: str, *clips) -> None:
    """Rows: CPU fp32, CPU bf16, device; columns: 5 frames across the clip, half size."""
    from PIL import Image

    n = clips[0].shape[0]
    idx = np.linspace(0, n - 1, 5).round().astype(int)
    rows = [np.concatenate([c[i][::2, ::2] for i in idx], axis=1) for c in clips]
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(out, "contact.png"))


@pytest.mark.skipif(
    not os.environ.get("LTX2_GATE_OUT"), reason="set LTX2_GATE_OUT to a reference+device run"
)
def test_fullsize_gate_device():
    out = os.environ["LTX2_GATE_OUT"]
    if not os.path.exists(os.path.join(out, "dev_frames_u8.npy")):
        pytest.skip("no device result")
    report = compare(out)
    assert report["pass"], _summary(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("reference", "device", "compare", "replay"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=SHAPE["height"])
    ap.add_argument("--width", type=int, default=SHAPE["width"])
    ap.add_argument("--frames", type=int, default=SHAPE["num_frames"])
    ap.add_argument("--steps", type=int, default=SHAPE["steps"])
    ap.add_argument("--cp", type=int, default=1, help="context-parallel degree (device mode)")
    ap.add_argument("--dtypes", default="fp32,bf16", help="reference dtypes, in run order")
    ap.add_argument("--threads", type=int, default=0, help="reference torch threads (0: default)")
    ap.add_argument(
        "--dump-steps", default="", help="device mode: comma list of denoise steps to dump"
    )
    a = ap.parse_args()
    shape = dict(SHAPE, height=a.height, width=a.width, num_frames=a.frames, steps=a.steps)
    if a.mode == "reference":
        run_reference(a.model, a.out, shape, a.dtypes.split(","), a.threads)
    elif a.mode == "device":
        steps = [int(s) for s in a.dump_steps.split(",") if s]
        run_device(a.model, a.out, shape, a.cp, steps)
    elif a.mode == "replay":
        print("REPLAY_DONE " + json.dumps(run_replay(a.model, a.out, a.threads)), flush=True)
    else:
        print("GATE " + json.dumps(_summary(compare(a.out))), flush=True)


if __name__ == "__main__":
    main()
