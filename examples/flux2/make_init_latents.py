# SPDX-License-Identifier: Apache-2.0
"""Generate the fixed fp32 initial-noise tensor shared by the M1 parity runs.

Writes the pre-pack latents ``(1, in_channels, H//2, W//2)`` that FLUX.2's prepare_latents expects
(``in_channels = transformer.config.in_channels``, spatial = ``height // (vae_scale_factor*2)``),
drawn once in fp32 so the device run and both CPU references start from identical noise. Point
``FLUX2_INIT_LATENTS`` at the output for all three runs (run.py --parity-latents and parity_ref.py).

    python examples/flux2/make_init_latents.py --model-path <dir> --height 256 --width 256 --out init_256.pt
"""

import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(os.path.join(a.model_path, "transformer", "config.json")) as f:
        in_channels = int(json.load(f)["in_channels"])
    with open(os.path.join(a.model_path, "vae", "config.json")) as f:
        vsf = 2 ** (len(json.load(f)["block_out_channels"]) - 1)

    h = 2 * (a.height // (vsf * 2)) // 2
    w = 2 * (a.width // (vsf * 2)) // 2
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn(1, in_channels, h, w, generator=g, dtype=torch.float32)
    torch.save({"latents": lat, "height": a.height, "width": a.width, "seed": a.seed}, a.out)
    print(f"[init] {tuple(lat.shape)} fp32 -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
