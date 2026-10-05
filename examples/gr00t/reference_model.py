# SPDX-License-Identifier: Apache-2.0
"""Model-level upstream reference (no processor) for checkpoints whose processor upstream
vLLM-Omni cannot load -- GR00T-H-N1.7's processor config uses fork-only fields
(``min_max_embedding_keys``, ``REL_XYZ_ROT6D``, quaternion inputs) and surgical embodiment
tags absent from upstream's ``EmbodimentTag`` enum. The *model* is upstream ``Gr00tN1d7``
unchanged, so it is fed synthetic collator-layout inputs with a chosen image grid.

    python examples/gr00t/reference_model.py --model /path/to/gr00t-h-n17 --hf-root /path/to/hfroot \
        --n-images 3 --grid 14,24 --embodiment 3 --out ref_h.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--hf-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-images", type=int, default=3)
ap.add_argument("--grid", default="14,24", help="patch grid H,W per image (even numbers)")
ap.add_argument("--n-text", type=int, default=24)
ap.add_argument("--embodiment", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
model_dir, out = os.path.abspath(args.model), os.path.abspath(args.out)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "unit"))
os.chdir(args.hf_root)

from test_gr00t_tiny import synthetic_inputs, upstream_get_action  # noqa: E402
from transformers import AutoModel  # noqa: E402

import vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 as up  # noqa: E402,F401  (registers Gr00tN1d7)

t0 = time.time()
up.Gr00tN1d7DataCollator, _orig = (lambda **kw: None), up.Gr00tN1d7DataCollator  # processor not needed
try:
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
finally:
    up.Gr00tN1d7DataCollator = _orig
print(f"upstream load {time.time() - t0:.1f}s")
gh, gw = (int(x) for x in args.grid.split(","))
inputs = synthetic_inputs(n_images=args.n_images, grid=(gh, gw), n_text=args.n_text, seed=args.seed,
                          embodiment=args.embodiment)
cfg = model.config
noise = torch.randn((1, cfg.action_horizon, cfg.max_action_dim), generator=torch.Generator().manual_seed(args.seed + 1))
res = {}
for dt in (torch.float32, torch.bfloat16):
    model.to(dt)
    t = time.time()
    res[dt], feat = upstream_get_action(model, inputs, noise)
    print(f"upstream {dt} {time.time() - t:.2f}s")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
torch.save({"inputs": inputs, "noise": noise, "action_pred": res[torch.float32].float(),
            "action_pred_bf16": res[torch.bfloat16].float()}, out)
a, b = res[torch.float32].float(), res[torch.bfloat16].float()
print(f"saved {out}: {tuple(a.shape)} seq {inputs['input_ids'].shape[1]} upstream bf16-vs-fp32 rel "
      f"{((b - a).norm() / a.norm()).item():.5f}")
