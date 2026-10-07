"""CPU-only: is the Wan VAE tiled-ENCODE cost acceptable? Decode the tiled latent and the untiled
latent with the same (untiled, fp32) decoder and report the PSNR between the two decodes, plus the
latent rel_err, per overlap. Real weights via SMOKE_VAE_WEIGHTS (default: the TI2V-5B VAE on nvme).

    python test/neuron/tools/vae_tiling_roundtrip_cpu.py 480 704
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "SMOKE_VAE_WEIGHTS" not in os.environ:
    raise SystemExit("set SMOKE_VAE_WEIGHTS to a local Wan2.2 VAE directory (diffusers layout)")
os.environ["SMOKE_DEVICE"] = "cpu"
import smoke_platform_trn2 as S  # noqa: E402


def main() -> None:
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    ref, neu = S._real_width_vae_pair()
    frames = int(os.environ.get("FRAMES", "5"))
    overlaps = [int(o) for o in os.environ.get("OVERLAPS", "96,128").split(",")]
    res = {}
    for px in [int(a) for a in sys.argv[1:]] or [480]:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, px), torch.linspace(-1, 1, px), indexing="ij")
        base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
        x = torch.stack([base * (1 - 0.05 * f) for f in range(frames)], dim=1)[None].clamp(-1, 1)
        with torch.no_grad():
            t0 = time.time()
            z_un = ref.encode(x).latent_dist.mean
            t_enc = time.time() - t0
            t0 = time.time()
            dec_un = ref.decode(z_un).sample
            t_dec = time.time() - t0
            r = {"cpu_enc_s": round(t_enc, 1), "cpu_dec_s": round(t_dec, 1), "overlaps": {}}
            for ov in overlaps:
                S._set_tiling(neu, 192, ov)
                z_t = neu._encode(x)
                z_t = z_t[:, : z_t.shape[1] // 2]
                dec_t = ref.decode(z_t).sample
                r["overlaps"][ov] = {
                    "latent_rel": round(S._rel(z_t, z_un), 4),
                    "roundtrip_psnr_db": [
                        round(S._psnr(dec_t[:, :, f], dec_un[:, :, f]), 2) for f in range(frames)
                    ],
                    "psnr_vs_input_untiled": round(S._psnr(dec_un[:, :, 0], x[:, :, 0]), 2),
                    "psnr_vs_input_tiled": round(S._psnr(dec_t[:, :, 0], x[:, :, 0]), 2),
                }
                print(px, ov, r["overlaps"][ov], flush=True)
        res[px] = r
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
