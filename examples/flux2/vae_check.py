# SPDX-License-Identifier: Apache-2.0
"""Device VAE decode check for FLUX.2: tiled device decode vs an untiled fp32 CPU decode of the SAME
latents. Reports cold + warm decode seconds and PSNR, one JSON line at the end.

Single process on one NeuronCore (the VAE is ~160 MB). Latents: a ``run.py --parity-latents`` dump
(``{"latents": [1, 32, H/8, W/8]}``), so the comparison isolates the VAE.

    python examples/flux2/vae_check.py --model-path <dir> --latents dev_1024.pt --out vae.json [--png out.png]
"""

import argparse
import json
import math
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch


def _psnr(a, b):
    a, b = a.float().clamp(-1, 1), b.float().clamp(-1, 1)
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(4.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    from diffusers import AutoencoderKLFlux2
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.flux2.vae_flux2 import NeuronFlux2Vae

    z = torch.load(a.latents)["latents"].float()
    rep = {
        "latent_shape": list(z.shape),
        "tile": int(os.environ.get("FLUX2_VAE_TILE", "64")),
        "overlap": int(os.environ.get("FLUX2_VAE_OVERLAP", "16")),
    }

    cpu = AutoencoderKLFlux2.from_pretrained(
        a.model_path, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    t0 = time.time()
    with torch.no_grad():
        ref = cpu.decode(z, return_dict=False)[0]
    rep["cpu_fp32_untiled_s"] = round(time.time() - t0, 2)
    del cpu

    vae = NeuronFlux2Vae(
        AutoencoderKLFlux2.from_pretrained(
            a.model_path, subfolder="vae", torch_dtype=torch.bfloat16
        ).eval()
    )
    vae.tile_parallel = False
    vae.to(torch.device("privateuseone", 0))
    vae.compile(get_compile_backend_name())
    t0 = time.time()
    out = vae.decode(z, return_dict=False)[0]
    rep["device_cold_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    out = vae.decode(z, return_dict=False)[0]
    rep["device_warm_s"] = round(time.time() - t0, 2)
    rep["psnr_db_vs_cpu_fp32"] = round(_psnr(out, ref), 2)

    if a.png:
        from PIL import Image

        img = ((out[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(a.png)
        rep["png"] = os.path.abspath(a.png)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)
    print("[vae] " + json.dumps(rep))


if __name__ == "__main__":
    main()
