# SPDX-License-Identifier: Apache-2.0
"""Build a thin SERVING directory for a FLUX Action policy package so vLLM-Omni can discover it.

Omni's ``enrich_config`` requires a ``model_index.json`` (with ``_class_name``) and a
``transformer/config.json`` in the model directory; the released FLUX Action packages ship neither
(they have ``config.json`` / ``config.native.json``). Rather than mutate the read-only weights, this
writes a small directory that SYMLINKS every file of the real package and adds the two discovery
files, so ``--policy <serving-dir>`` resolves while :class:`NeuronFlux3ActionPipeline` still loads
the real ``model.safetensors`` from it.

    python -m examples.flux3_action.make_serving_dir --policy $WEIGHTS/flux3-action-droid --out serving/droid
"""

from __future__ import annotations

import argparse
import json
import os


def make_serving_dir(policy_dir: str, out_dir: str) -> str:
    policy_dir = os.path.abspath(policy_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(policy_dir):
        src = os.path.join(policy_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)
    # discovery files Omni's enrich_config reads
    model_index = {
        "_class_name": "Flux3ActionPipeline",
        "_diffusers_version": "0.0.0",
        "transformer": ["vllm_omni_neuron", "NeuronFlux3ActionPipeline"],
    }
    with open(os.path.join(out_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)
    # a transformer/ config so the non-diffusers path finds a TransformerConfig; mirror the policy config
    tdir = os.path.join(out_dir, "transformer")
    os.makedirs(tdir, exist_ok=True)
    cfg_name = (
        "config.native.json"
        if os.path.isfile(os.path.join(policy_dir, "config.native.json"))
        else "config.json"
    )
    with open(os.path.join(policy_dir, cfg_name)) as f:
        cfg = json.load(f)
    with open(os.path.join(tdir, "config.json"), "w") as f:
        json.dump({"model_type": "flux3_action", **cfg}, f, indent=2)
    return os.path.abspath(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(make_serving_dir(a.policy, a.out))


if __name__ == "__main__":
    main()
