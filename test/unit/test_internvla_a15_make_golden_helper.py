# SPDX-License-Identifier: Apache-2.0
"""Save upstream InternVLA-A1.5 golden actions (CPU, fp32 + bf16) for a synthetic request.

The request and noise are built exactly as ``examples/internvla/run.py`` builds them for the
same arguments, so its ``--golden`` check compares like with like.

    INTERNVLA_REF_SRC=<InternVLA-A-series>/src python -m test.unit.test_internvla_a15_make_golden_helper \
        --model CKPT --vlm-config QWEN35_CONFIG --out golden.pt [--n-images 3 --grid 16 16 ...]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from vllm_omni_neuron.diffusion.models.internvla import InternVLAConfig
from vllm_omni_neuron.diffusion.models.internvla import preprocess as pp

from .test_internvla_a15_upstream_helper import build_upstream, ref_src, upstream_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vlm-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--grid", type=int, nargs=2, default=(16, 16))
    ap.add_argument("--text-before", type=int, default=14)
    ap.add_argument("--text-after", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument(
        "--batch",
        default=None,
        help="torch.save'd request dict (e.g. a served observation) "
        "instead of the synthetic request",
    )
    a = ap.parse_args()
    src = ref_src()
    if src is None:
        raise SystemExit("set INTERNVLA_REF_SRC")
    cfg = InternVLAConfig.from_model_dir(a.model, a.vlm_config)
    if a.batch:
        batch = torch.load(a.batch)
    else:
        batch = pp.synthetic_request(
            cfg,
            n_images=a.n_images,
            grid=tuple(a.grid),
            text_before=a.text_before,
            text_after=a.text_after,
            seed=a.seed,
        )
    noise = pp.initial_noise(cfg, seed=a.seed)
    out = {"args": vars(a), "noise": noise}
    for name in a.dtypes.split(","):
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[name]
        t0 = time.time()
        model = build_upstream(a.model, a.vlm_config, dt, src)
        out[f"actions_{name}"] = upstream_sample(model, batch, noise, dt).float()[
            :, :, : cfg.policy.action_dim
        ]
        out[f"{name}_s"] = time.time() - t0
        del model
    torch.save(out, a.out)
    print(
        json.dumps(
            {
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in out.items()
                if k.endswith("_s")
            }
            | {"out": a.out}
        )
    )


if __name__ == "__main__":
    main()
