"""Decode the same final latents with the CPU fp32 diffusers VAE and the Neuron device-wrapper VAE path, save
contact sheets for both, and report per-frame PSNR + a striping metric (vertical-vs-horizontal gradient energy
ratio -- a comb/streak artifact concentrates energy in one gradient direction).

Usage:
    python examples/minimax_h3/eval/vae_compare.py --latents dump.pt --weights <FastH3 dir> --out-dir DIR \
        [--device-wrapper-dtype bfloat16] [--vae-mode device|cpu]

With --vae-mode cpu, the "device" column runs the SAME wrapper code path (NeuronMiniMaxH3VideoVAE) but on the CPU
backend (tiling off, mixed precision) -- this isolates "did my wrapper change the math" from "does Neuron rounding
look different", without needing a device job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def _load_latents(path: str) -> dict:
    import types

    for name in ("fasth3_neuron", "fasth3_neuron.layout"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class H3Layout:
        def __setstate__(self, s):
            self.__dict__.update(s)

    sys.modules["fasth3_neuron.layout"].H3Layout = H3Layout
    return torch.load(path, map_location="cpu", weights_only=False)


def _video_frames(video: torch.Tensor) -> np.ndarray:
    """(1, 3, F, H, W) in [0, 1] -> (F, H, W, 3) uint8."""
    return (video[0].permute(1, 2, 3, 0).float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def striping_ratio(frame: np.ndarray) -> float:
    """Mean |horizontal gradient| / mean |vertical gradient| of a grayscale frame. A vertical-comb artifact
    (energy concentrated in columns) raises the HORIZONTAL gradient (adjacent columns differ) much more than the
    vertical one -- so this ratio rises well above 1 when striping is present."""
    g = frame.astype(np.float64).mean(axis=-1)
    dx = np.abs(np.diff(g, axis=1)).mean()  # column-to-column (catches vertical stripes)
    dy = np.abs(np.diff(g, axis=0)).mean()  # row-to-row
    return float(dx / max(dy, 1e-6))


def contact_sheet(frames: np.ndarray, n: int = 6):
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return Image.fromarray(np.concatenate([frames[i] for i in idx], axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device-wrapper-dtype", default="bfloat16")
    ap.add_argument("--vae-mode", default="device", choices=("device", "cpu"))
    ap.add_argument(
        "--device-mp4",
        default=None,
        help="with --vae-mode device: the clip already decoded on the "
        "NeuronCore; its frames are compared against the fp32 CPU decode of --latents (PSNR carries "
        "h264 loss, but the striping-ratio metric is per-frame structural and still valid)",
    )
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = _load_latents(a.latents)
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, unpatchify_video
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import NeuronMiniMaxH3VideoVAE

    if "layout" in d:
        layout = d["layout"]
    else:
        # MINIMAX_H3_DUMP_LATENTS dumps (prompt, geom, steps, seed, video_rows, audio_rows) without the layout
        # object; audio_rows' row count fixes num_text_tokens given the known audio-row formula, but simplest is
        # to re-run build_layout with the same geometry and read num_text_tokens off the row count directly:
        # video_rows has no text rows mixed in (it is video-only), so just rebuild from geom with the embeds'
        # token count, which we don't have here -- instead, derive it from the total row counts via build_layout's
        # own audio-latent formula at a few candidate text lengths. Simpler: audio_latent_num_frames depends only
        # on num_frames, so num_audio_rows is independent of num_text_tokens; try num_text_tokens=0 and compute
        # geometry-only fields (video/audio shapes don't depend on num_text_tokens).
        h, w, nf = d["geom"]
        layout = build_layout(0, h, w, nf)

    dtype = getattr(torch, a.device_wrapper_dtype)
    latents = unpatchify_video(d["video_rows"], layout)
    mean = torch.tensor(
        AutoencoderKLMiniMaxH3.from_pretrained(a.weights, subfolder="vae").config.latents_mean
    ).view(1, -1, 1, 1, 1)
    ref32 = AutoencoderKLMiniMaxH3.from_pretrained(
        a.weights, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    std = torch.tensor(ref32.config.latents_std).view(1, -1, 1, 1, 1)
    z = (latents * std + mean).float()

    pm_g = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1, 1)
    ps_g = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video_cpu = ref32.decode(z, return_dict=False)[0]  # fp32, tiling ON (released default)
    f_cpu = _video_frames((video_cpu.float() * ps_g + pm_g).clamp(0, 1))

    os.environ["MINIMAX_H3_VAE"] = a.vae_mode
    if a.vae_mode == "device":
        # Compare the ALREADY-DECODED device clip (its mp4 frames) against the fp32 CPU decode of the same latents --
        # no second device decode here, so this half runs on the host (cpumode). Pass the device clip via --device-mp4.
        import av

        c = av.open(a.device_mp4)
        f_dev = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    else:
        ref_mixed = AutoencoderKLMiniMaxH3.from_pretrained(
            a.weights, subfolder="vae", torch_dtype=dtype
        ).eval()
        wrap = NeuronMiniMaxH3VideoVAE(ref_mixed, torch.device("cpu"), dtype)
        with torch.no_grad():
            video_dev = wrap.decode(z, return_dict=False)[0]
        f_dev = _video_frames((video_dev.float() * ps_g + pm_g).clamp(0, 1))

    n = min(len(f_cpu), len(f_dev))
    per_frame_psnr = [psnr(f_cpu[i], f_dev[i]) for i in range(n)]
    stripe_cpu = [striping_ratio(f_cpu[i]) for i in range(n)]
    stripe_dev = [striping_ratio(f_dev[i]) for i in range(n)]

    contact_sheet(f_cpu).save(os.path.join(a.out_dir, "sheet_cpu_fp32.png"))
    contact_sheet(f_dev).save(os.path.join(a.out_dir, "sheet_device_wrapper.png"))

    report = {
        "weights": a.weights,
        "vae_mode": a.vae_mode,
        "device_wrapper_dtype": a.device_wrapper_dtype,
        "num_frames": n,
        "psnr_mean": float(np.mean(per_frame_psnr)),
        "psnr_min": float(np.min(per_frame_psnr)),
        "psnr_per_frame": per_frame_psnr,
        "striping_ratio_cpu_mean": float(np.mean(stripe_cpu)),
        "striping_ratio_device_mean": float(np.mean(stripe_dev)),
        "striping_ratio_cpu": stripe_cpu,
        "striping_ratio_device": stripe_dev,
    }
    with open(os.path.join(a.out_dir, "vae_compare.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if not k.endswith("per_frame")
                and k not in ("striping_ratio_cpu", "striping_ratio_device")
            }
        )
    )


if __name__ == "__main__":
    main()
