"""Z-Image device check: CPU oracle vs NeuronCore run of the same generation, one job.

Runs the full T2I loop (text encoder -> N DiT steps -> VAE decode) through diffusers' own
``ZImagePipeline`` with the Neuron components on ``neuron:0``, and (unless ``--no-oracle``)
the pure-diffusers fp32 CPU pipeline from the same seed. Reports final-latent rel-L2 / cosine,
image PSNR, per-component timings (cold + warm), and prints one JSON summary line.

    python examples/z_image/device_check.py --model $WEIGHTS/z-image-turbo --steps 9 --guidance 0 \
        --height 1024 --width 1024 --out ./z_image_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument(
    "--prompt",
    default="A red fox standing in fresh snow at golden hour, photorealistic, detailed fur",
)
ap.add_argument("--negative-prompt", default=None)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--steps", type=int, default=9)
ap.add_argument("--guidance", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--repeat", type=int, default=2, help="device runs (first is cold)")
ap.add_argument("--no-oracle", action="store_true")
ap.add_argument("--oracle-dtype", default="float32")
ap.add_argument("--cpu", action="store_true", help="run 'device' side on CPU (bf16) for debugging")
ap.add_argument("--te-on-host", action="store_true", help="keep the text encoder on the CPU (bf16)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

from vllm_omni_neuron.diffusion.models.z_image.standalone import (  # noqa: E402
    build_components,
    build_diffusers_pipeline,
)

os.makedirs(args.out, exist_ok=True)
summary: dict = {
    "model": args.model,
    "hw": [args.height, args.width],
    "steps": args.steps,
    "cfg": args.guidance,
}
kw = dict(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    output_type="latent",
)


def to_img(lat, vae_like):
    sf, sh = vae_like.config.scaling_factor, vae_like.config.shift_factor
    z = lat / sf + sh
    im = vae_like.decode(z.to(vae_like.dtype), return_dict=False)[0]
    return (
        ((im.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).numpy()
    )


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def save(arr, name):
    from PIL import Image

    p = os.path.join(args.out, name)
    Image.fromarray(arr).save(p)
    return p


try:
    t0 = time.time()
    comps = build_components(args.model, torch.bfloat16)
    summary["load_s"] = round(time.time() - t0, 1)
    gb = lambda m: round(sum(p.numel() * p.element_size() for p in m.parameters()) / 2**30, 2)  # noqa: E731
    summary["weights_gb"] = {
        "text_encoder": gb(comps[0].enc),
        "dit": gb(comps[1].dit),
        "vae": gb(comps[2].vae),
    }
    if args.cpu:
        dev, backend = "cpu", None
    else:
        from vllm_neuron.envs import get_compile_backend_name

        dev, backend = torch.device("neuron", 0), get_compile_backend_name()
    t0 = time.time()
    pipe = build_diffusers_pipeline(
        args.model,
        torch.bfloat16,
        device=dev,
        compile_backend=backend,
        components=comps,
        te_on_host=args.te_on_host,
    )
    summary["to_device_s"] = round(time.time() - t0, 1)
    te, dit, vae = comps
    runs = []
    lat = None
    for r in range(args.repeat):
        dit.stats = {"calls": 0, "s": 0.0}
        t0 = time.time()
        with torch.no_grad():
            lat = pipe(generator=torch.Generator().manual_seed(args.seed), **kw).images
        gen_s = time.time() - t0
        t1 = time.time()
        img = to_img(lat.float(), vae)
        dec_s = time.time() - t1
        runs.append(
            {
                "gen_s": round(gen_s, 3),
                "te_s": round(te.last_s, 3),
                "dit_calls": dit.stats["calls"],
                "dit_s": round(dit.stats["s"], 3),
                "vae_s": round(dec_s, 3),
                "total_s": round(gen_s + dec_s, 3),
            }
        )
        print(f"[run {r}] {runs[-1]}", flush=True)
    summary["runs"] = runs
    summary["device_png"] = save(img, "device.png")
    torch.save(lat.float().cpu(), os.path.join(args.out, "device_latent.pt"))
    if not args.no_oracle:
        from diffusers import ZImagePipeline

        odt = getattr(torch, args.oracle_dtype)
        t0 = time.time()
        ref = ZImagePipeline.from_pretrained(args.model, torch_dtype=odt)
        with torch.no_grad():
            rlat = ref(generator=torch.Generator().manual_seed(args.seed), **kw).images.float()
        summary["oracle_s"] = round(time.time() - t0, 1)
        torch.save(rlat, os.path.join(args.out, "oracle_latent.pt"))
        a, b = lat.float().flatten(), rlat.flatten()
        summary["latent_rel_l2"] = round(((a - b).norm() / b.norm()).item(), 5)
        summary["latent_cos"] = round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 5)
        rimg = to_img(rlat, ref.vae)
        summary["oracle_png"] = save(rimg, "oracle.png")
        summary["psnr_vs_oracle_db"] = round(psnr(img, rimg), 2)
    summary["ok"] = True
    print("SUMMARY " + json.dumps(summary), flush=True)
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    summary["ok"] = False
    summary["error"] = f"{type(e).__name__}: {e}"[:500]
    print("SUMMARY " + json.dumps(summary), flush=True)
    sys.exit(1)
