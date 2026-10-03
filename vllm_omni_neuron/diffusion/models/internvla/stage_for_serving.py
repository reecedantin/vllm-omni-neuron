# SPDX-License-Identifier: Apache-2.0
"""Stage a served-ready InternVLA-A1.5 checkpoint directory without modifying the shared,
read-only source weights.

vLLM-Omni's ``OmniDiffusionConfig.enrich_config`` has no built-in model_type mapping for a
third-party policy; its generic fallback reads ``config.json["architectures"]`` (a single-element
list) as the pipeline class name. The upstream InternVLA-A1.5 checkpoint doesn't carry that key, so
this writes a small directory with everything symlinked except a patched ``config.json``.

    python -m vllm_omni_neuron.diffusion.models.internvla.stage_for_serving SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import json
import os


def stage(src: str, dst: str) -> str:
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["InternVLAA15Pipeline"]
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name in os.listdir(src):
        if name == "config.json":
            continue
        link = os.path.join(dst, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, name), link)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    print(stage(a.src, a.dst))
