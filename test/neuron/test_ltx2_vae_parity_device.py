# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 device VAE parity: the device-tiled decode (``models/ltx2/vae_tiling.py``) against the untiled
CPU decode of the same latent.

The served pipeline saves the latent it decoded and the device output when ``LTX2_VAE_DUMP_DIR`` is set.
This script decodes that latent with the untiled CPU VAE and reports per-frame PSNR in the [0, 1] pixel
range (frame 0 is the causal first frame), PSNR on the tile-overlap columns vs the rest (a seam shows as a
low seam-column PSNR), and writes a 6-frame contact sheet of the device output. Pass: mean > 35 dB.

    LTX2_VAE_DUMP_DIR=<dir> python examples/ltx2/run.py ...
    python -m test.neuron.test_ltx2_vae_parity_device --dump <dir> --model <LTX-2.5 dir>

As a pytest it skips unless ``LTX2_VAE_DUMP_DIR`` and ``LTX25_WEIGHTS`` point at a dump and the weights.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytest
import torch


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean()) / 4.0
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def vae_parity(dump: str, model: str) -> dict:
    from diffusers import AutoencoderKLLTX2Video

    from vllm_omni_neuron.diffusion.models.ltx2.vae_tiling import OVERLAP, TILE_W, tile_starts

    d = torch.load(os.path.join(dump, "vae_decode_dump.pt"))
    z, tiled = d["latent"], d["tiled"]
    vae = AutoencoderKLLTX2Video.from_pretrained(model, subfolder="vae", torch_dtype=z.dtype).eval()
    with torch.no_grad():
        ref = vae.decoder(z, None)
    per_frame = [_psnr(tiled[:, :, i], ref[:, :, i]) for i in range(ref.shape[2])]
    scale = int(getattr(vae, "spatial_compression_ratio", 32))
    starts = tile_starts(z.shape[-1], TILE_W, TILE_W - OVERLAP)
    seam = torch.zeros(ref.shape[-1], dtype=torch.bool)
    for s in starts[1:]:
        seam[s * scale : (s + OVERLAP) * scale] = True
    report = {
        "latent_shape": list(z.shape),
        "dtype": str(z.dtype),
        "tile_w": TILE_W,
        "overlap": OVERLAP,
        "tile_starts": starts,
        "psnr_mean_db": round(float(np.mean(per_frame)), 2),
        "psnr_min_db": round(float(np.min(per_frame)), 2),
        "psnr_frame0_db": round(per_frame[0], 2),
        "psnr_seam_cols_db": round(_psnr(tiled[..., seam], ref[..., seam]), 2)
        if seam.any()
        else None,
        "psnr_nonseam_cols_db": round(_psnr(tiled[..., ~seam], ref[..., ~seam]), 2),
    }
    from PIL import Image

    vid = ((tiled[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()  # F,H,W,C
    idx = np.linspace(0, vid.shape[0] - 1, 6).astype(int)
    h, ww = vid.shape[1], vid.shape[2]
    sheet = Image.new("RGB", (ww * 3, h * 2))
    for n, i in enumerate(idx):
        sheet.paste(
            Image.fromarray((vid[i] * 255).round().astype("uint8")), ((n % 3) * ww, (n // 3) * h)
        )
    sheet.save(os.path.join(dump, "contact_sheet_device_vae.png"))
    with open(os.path.join(dump, "vae_parity.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


@pytest.mark.skipif(
    not (os.environ.get("LTX2_VAE_DUMP_DIR") and os.environ.get("LTX25_WEIGHTS")),
    reason="set LTX2_VAE_DUMP_DIR (served-run dump) and LTX25_WEIGHTS",
)
def test_vae_parity_device():
    report = vae_parity(os.environ["LTX2_VAE_DUMP_DIR"], os.environ["LTX25_WEIGHTS"])
    assert report["psnr_mean_db"] > 35.0, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    print("VAE_PARITY " + json.dumps(vae_parity(a.dump, a.model)), flush=True)


if __name__ == "__main__":
    main()
