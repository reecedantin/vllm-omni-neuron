#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""End-to-end (tier 3) accuracy of a device run against an independent CPU reference.

    python examples/minimax_h3/eval/tier3_compare.py --model-path <ckpt> --ref ref_fp32.pt --floor ref_bf16.pt \
        --run device.pt [--run2 device_again.pt] [--video device_video.pt] [--audio device_audio.pt] [--out t3.json]

``--ref`` / ``--floor`` come from ``reference_cpu.py`` (diffusers' transformer on CPU, fp32 / bf16); ``--run`` /
``--run2`` are ``MINIMAX_H3_DUMP_LATENTS`` files of two separate device runs of the same request. Checks:

1. **Latents**: final video rel-L2 (and the step-0 velocity, when both files carry it) of the device run vs the fp32
   reference, against the fleet bar ``2 x (CPU bf16 vs fp32) + 0.5 %``.
2. **Pixels**: the reference, the floor and the device latents are all decoded by the same CPU fp32 video VAE; the
   device video's per-frame SSIM to the reference video must satisfy ``1 - SSIM <= 2 x (1 - SSIM_floor) + 0.005``
   (frame mean). Decoding all three with one decoder isolates the DiT from the VAE. Per-frame SSIM and PSNR are
   reported. With ``--video`` (``MINIMAX_H3_SAVE_VIDEO``) the device's own decoded frames get the same check.
3. **Audio** (with ``--audio``, ``MINIMAX_H3_SAVE_AUDIO``): the device waveform against a CPU fp32 decode of the
   reference's audio latents, bar ``2 x floor + 0.5 %`` on the waveform rel-L2.
4. **Repeatability** (with ``--run2``): the two device runs are bit-equal.

Run with ``fleet/bin/cpumode.sh`` sourced. Exit 0 = every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_compare import _load, compare  # noqa: E402


def bar(x: float) -> float:
    """The fleet parity bar: device error <= 2 x the CPU-bf16 error + 0.5 %."""
    return 2.0 * x + 0.005


def ssim_per_frame(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Per-frame mean SSIM of two ``(3, F, H, W)`` videos in ``[0, 1]`` (11x11 Gaussian window, sigma 1.5)."""
    c1, c2 = 0.01**2, 0.03**2
    x = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    a = a.float().permute(1, 0, 2, 3)  # (F, 3, H, W)
    b = b.float().permute(1, 0, 2, 3)
    mu_a, mu_b = F.conv2d(a, win, groups=3), F.conv2d(b, win, groups=3)
    s_aa = F.conv2d(a * a, win, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, win, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, win, groups=3) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean(dim=(1, 2, 3)).tolist()


def psnr_per_frame(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Per-frame PSNR (dB) of two ``(3, F, H, W)`` videos in ``[0, 1]``."""
    mse = (a.float() - b.float()).pow(2).mean(dim=(0, 2, 3)).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).tolist()


def _frames(x: list[float]) -> dict:
    return {
        "mean": sum(x) / len(x),
        "min": min(x),
        "argmin": int(min(range(len(x)), key=x.__getitem__)),
        "per_frame": [round(v, 4) for v in x],
    }


class CPUVideoDecoder:
    """The checkpoint's video VAE in fp32 on the host, with the pipeline's latent/pixel normalization and tiling."""

    def __init__(self, model_path: str):
        from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
            AutoencoderKLMiniMaxH3,
        )

        self.vae = (
            AutoencoderKLMiniMaxH3.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.float32
            )
            .float()
            .eval()
        )

    @torch.no_grad()
    def __call__(self, video_rows: torch.Tensor, n_text: int, geom) -> torch.Tensor:
        from vllm_omni_neuron.diffusion.models.minimax_h3.layout import (
            build_layout,
            unpatchify_video,
        )
        from vllm_omni_neuron.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            PIXEL_MEAN,
            PIXEL_STD,
        )

        layout = build_layout(n_text, *geom)
        lat = unpatchify_video(video_rows.float(), layout)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        video = self.vae.decode((lat * std + mean).float(), return_dict=False)[0]
        pm = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
        ps = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
        return (video.float() * ps + pm).clamp(0, 1)[0]  # (3, F, H, W)


class CachedDecoder:
    """``CPUVideoDecoder`` with an on-disk cache keyed by a digest of the latent rows, so the (slow, large-canvas)
    CPU decodes can run as separate processes in parallel (``--decode-only``) before the comparison."""

    def __init__(self, model_path: str, cache_dir: str):
        self.model_path, self.cache_dir, self.dec = model_path, cache_dir, None
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, video_rows: torch.Tensor) -> str:
        import hashlib

        h = hashlib.sha256(video_rows.float().contiguous().numpy().tobytes()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"video_{h}.pt")

    def __call__(self, video_rows: torch.Tensor, n_text: int, geom) -> torch.Tensor:
        p = self.path(video_rows)
        if os.path.exists(p):
            return torch.load(p).float()
        self.dec = self.dec or CPUVideoDecoder(self.model_path)
        v = self.dec(video_rows, n_text, geom)
        torch.save(v, p + ".tmp")
        os.replace(p + ".tmp", p)
        return v


def _wave_err(x: torch.Tensor, r: torch.Tensor) -> dict:
    assert x.shape == r.shape, (x.shape, r.shape)
    e = (x.float() - r.float()).pow(2).mean().clamp_min(1e-20)
    return {
        "rel_l2": ((x.float() - r.float()).norm() / r.float().norm()).item(),
        "snr_db": (10 * torch.log10(r.float().pow(2).mean() / e)).item(),
    }


@torch.no_grad()
def audio_check(
    model_path: str, ref: dict, floor: dict, run: dict, device_audio: torch.Tensor
) -> dict:
    """The device waveform (``MINIMAX_H3_SAVE_AUDIO``) against a CPU fp32 audio-VAE decode of the fp32 reference's
    final audio latents, with the CPU bf16 reference's latents through the same decoder as the floor. Also reports
    the device waveform against the CPU fp32 decode of the device run's own latents (the audio VAE alone)."""
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, unpack_audio

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        model_path, subfolder="audio_vae", torch_dtype=torch.float32
    ).eval()
    c = vae.config
    geom = tuple(ref.get("geom") or run["geom"])
    n_text = (
        run["layout"].num_text_tokens if "layout" in run else int(ref["prompt_embeds"].shape[1])
    )
    layout = build_layout(n_text, *geom)

    def dec(rows: torch.Tensor) -> torch.Tensor:
        z = unpack_audio(rows.float(), layout)
        z = z * torch.tensor(c.latents_std).view(1, -1, 1) + torch.tensor(c.latents_mean).view(
            1, -1, 1
        )
        return (
            vae.decode(z, return_dict=False)[0].float().permute(1, 0, 2)
        )  # the pipeline's output layout

    w_ref, w_flo, w_run = dec(ref["audio_rows"]), dec(floor["audio_rows"]), dec(run["audio_rows"])
    dev = device_audio.float().reshape(w_ref.shape)
    d, f = _wave_err(dev, w_ref), _wave_err(w_flo, w_ref)
    return {
        "device_vs_ref": d,
        "floor_vs_ref": f,
        "device_vs_cpu_decode_of_same_latents": _wave_err(dev, w_run),
        "rule": "waveform rel-L2 <= 2 x floor + 0.005",
        "bar": bar(f["rel_l2"]),
        "pass": d["rel_l2"] <= bar(f["rel_l2"]),
    }


def tier3(
    model_path: str,
    ref: dict,
    floor: dict,
    run: dict,
    run2: dict | None = None,
    decoder: CPUVideoDecoder | None = None,
    device_video: torch.Tensor | None = None,
    device_audio: torch.Tensor | None = None,
) -> dict:
    res = compare(ref, run, floor)
    lat_ok = bool(res["gate"]["pass"])
    out: dict = {"latents": res, "latents_pass": lat_ok}
    geom = tuple(ref.get("geom") or run["geom"])
    n_text = (
        run["layout"].num_text_tokens if "layout" in run else int(ref["prompt_embeds"].shape[1])
    )
    dec = decoder or CPUVideoDecoder(model_path)
    v_ref, v_flo, v_dev = (dec(d["video_rows"], n_text, geom) for d in (ref, floor, run))
    s_dev, s_flo = ssim_per_frame(v_dev, v_ref), ssim_per_frame(v_flo, v_ref)
    m_dev, m_flo = sum(s_dev) / len(s_dev), sum(s_flo) / len(s_flo)
    ssim_ok = (1.0 - m_dev) <= bar(1.0 - m_flo)
    out["ssim"] = {
        "device_vs_ref_mean": m_dev,
        "device_vs_ref_min": min(s_dev),
        "floor_vs_ref_mean": m_flo,
        "floor_vs_ref_min": min(s_flo),
        "rule": "1 - ssim_dev <= 2 x (1 - ssim_floor) + 0.005",
        "pass": ssim_ok,
    }
    out["frames"] = {
        "ssim_device": _frames(s_dev),
        "ssim_floor": _frames(s_flo),
        "psnr_device": _frames(psnr_per_frame(v_dev, v_ref)),
        "psnr_floor": _frames(psnr_per_frame(v_flo, v_ref)),
    }
    if (
        device_video is not None
    ):  # the device's own decode (device VAE, uint8) against the CPU fp32 reference video
        dv = device_video[0] if device_video.dim() == 5 else device_video
        dv = dv.float() / 255.0 if dv.dtype == torch.uint8 else dv.float()
        s_e2e = ssim_per_frame(dv, v_ref)
        out["device_pixels"] = {
            "ssim": _frames(s_e2e),
            "psnr": _frames(psnr_per_frame(dv, v_ref)),
            "psnr_vs_cpu_decode_of_same_latents": _frames(psnr_per_frame(dv, v_dev)),
            "rule": "1 - ssim <= 2 x (1 - ssim_floor) + 0.005",
            "pass": (1.0 - sum(s_e2e) / len(s_e2e)) <= bar(1.0 - m_flo),
        }
    ok = lat_ok and ssim_ok and out.get("device_pixels", {}).get("pass", True)
    if device_audio is not None:
        out["audio"] = audio_check(model_path, ref, floor, run, device_audio)
        ok = ok and out["audio"]["pass"]
    if run2 is not None:
        rep = bool(
            torch.equal(run["video_rows"], run2["video_rows"])
            and torch.equal(run["audio_rows"], run2["audio_rows"])
        )
        out["repeatable"] = rep
        ok = ok and rep
    out["pass"] = ok
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--floor")
    ap.add_argument("--run")
    ap.add_argument("--run2", default=None)
    ap.add_argument(
        "--video", default=None, help="MINIMAX_H3_SAVE_VIDEO file of --run (device-decoded pixels)"
    )
    ap.add_argument(
        "--audio", default=None, help="MINIMAX_H3_SAVE_AUDIO file of --run (device waveform)"
    )
    ap.add_argument("--decode-cache", default=None, help="directory caching the CPU video decodes")
    ap.add_argument(
        "--decode-only", default=None, help="decode this latent file into --decode-cache and exit"
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    dec = CachedDecoder(a.model_path, a.decode_cache) if a.decode_cache else None
    if a.decode_only:
        d = _load(a.decode_only)
        n = d["layout"].num_text_tokens if "layout" in d else int(d["prompt_embeds"].shape[1])
        dec(d["video_rows"], n, tuple(d["geom"]))
        print(json.dumps({"decoded": a.decode_only, "cache": dec.path(d["video_rows"])}))
        sys.exit(0)
    dv = torch.load(a.video, weights_only=False) if a.video else None
    da = torch.load(a.audio, weights_only=False) if a.audio else None
    r = tier3(
        a.model_path,
        _load(a.ref),
        _load(a.floor),
        _load(a.run),
        _load(a.run2) if a.run2 else None,
        decoder=dec,
        device_video=dv,
        device_audio=da,
    )
    print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    sys.exit(0 if r["pass"] else 1)
