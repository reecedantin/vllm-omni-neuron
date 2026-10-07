"""Accuracy noise floor: bf16-on-CPU vs fp32-on-CPU, same generation, no NeuronCore involved.

Isolates the error purely from bf16 quantization (what any device run inherits as a floor)
from the error the device adds on top. Run alongside device_check.py's device-vs-fp32-oracle
number: device error should be within roughly 2x this + 0.5% (rel-L2), else the device path is
adding real numerical error beyond expected dtype rounding.
"""

from __future__ import annotations

import argparse
import json

import torch
from diffusers import ZImagePipeline

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument(
    "--prompt",
    default="A red fox standing in fresh snow at golden hour, photorealistic, detailed fur",
)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--steps", type=int, default=9)
ap.add_argument("--guidance", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument(
    "--ref-latent",
    default=None,
    help="fp32 oracle latent already computed (e.g. device_check.py's oracle_latent.pt); skips the fp32 run",
)
ap.add_argument("--out", required=True)
args = ap.parse_args()

kw = dict(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    output_type="latent",
)

if args.ref_latent:
    want = torch.load(args.ref_latent).float()
else:
    fp32 = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    with torch.no_grad():
        want = fp32(generator=torch.Generator().manual_seed(args.seed), **kw).images.float()
    del fp32

bf16 = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
with torch.no_grad():
    got = bf16(generator=torch.Generator().manual_seed(args.seed), **kw).images.float()
del bf16

a, b = got.flatten(), want.flatten()
summary = {
    "rel_l2": round(((a - b).norm() / b.norm()).item(), 5),
    "cos": round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 5),
}
print("SUMMARY " + json.dumps(summary), flush=True)
import os  # noqa: E402

os.makedirs(args.out, exist_ok=True)
with open(f"{args.out}/summary.json", "w") as f:
    json.dump(summary, f)
