# SPDX-License-Identifier: Apache-2.0
"""Run the UPSTREAM FLUX Action policy (black-forest-labs/flux-action) on CPU as the parity oracle.

    python -m test.unit.test_flux3_action_utils.reference --policy DIR --base DIR --obs OBS.npz --out OUT.pt

Upstream needs CUDA only for NATTEN and its prepared/compiled path; the reference (unprepared)
path runs on CPU with :mod:`natten_shim` providing ``na2d``/``na3d``. Two settings differ from a
plain ``predict_action_chunk`` call, both to match the released serving path the Neuron port
implements:

* ``single_frame_encode = True`` (what ``prepare_inference()`` forces for every package);
* the solver output is captured so the predicted VIDEO latents can be compared too.

Saves ``{actions, targets, video_latents, cond_latent, config}``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# a checkout of https://github.com/black-forest-labs/flux-action: <checkout>/src
UPSTREAM_SRC = os.environ.get("FLUX_ACTION_SRC", "")
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natten_shim")


def _import_upstream():
    if not UPSTREAM_SRC:
        raise RuntimeError("set FLUX_ACTION_SRC to <flux-action checkout>/src")
    for p in (SHIM, UPSTREAM_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("F3_NATTEN_BACKEND", "flex-fna")
    import flux_action  # noqa: F401

    return flux_action


def load_upstream_vae(base_dir: str):
    """The upstream video VAE for ``base_dir`` (a tiny model's ``video_vae.json`` dims are honoured)."""
    import json
    from dataclasses import replace

    _import_upstream()
    from flux_action.models import video_vae as up_vae

    side = os.path.join(base_dir, "video_vae.json")
    orig = up_vae.ViTNormInferenceParams
    if os.path.isfile(side):  # tiny structure model: same classes, shrunk dims
        with open(side) as f:
            dims = json.load(f)
        keep = {k: v for k, v in dims.items() if k in orig.__dataclass_fields__}
        up_vae.ViTNormInferenceParams = lambda **kw: replace(orig(**kw), **keep)  # type: ignore[assignment]
    try:
        return up_vae.load_video_vae(
            os.path.join(base_dir, "video_vae.safetensors"), compile_model=False
        )
    finally:
        up_vae.ViTNormInferenceParams = orig


def run_reference(
    policy_dir: str,
    base_dir: str,
    batch: dict,
    *,
    seed: int = 0,
    threads: int | None = None,
    dtype: str = "keep",
    num_steps: int | None = None,
) -> dict:
    _import_upstream()

    from flux_action import policy as up_policy
    from flux_action.inference import sampling as up_sampling
    from flux_action.models.text_encoder import load_text_encoder
    from flux_action.processing import packing as up_packing

    if threads:
        torch.set_num_threads(threads)
    vae = load_upstream_vae(base_dir)
    text = load_text_encoder(os.path.join(base_dir, "text_encoder"))
    pol = up_policy.FluxActionPolicy.from_pretrained(
        policy_dir, video_vae=vae, text_encoder=text, device="cpu"
    )
    if dtype == "float32":
        # fp32 oracle for the DiT/solver (what the action parity band measures). The frozen VAE and
        # text encoder keep their package dtype -- upstream forces a bf16 VAE input internally, and the
        # conditioning latent is a tiny, shared input to all three runs -- so only the DiT is cast.
        pol.dit.float()
        pol.set_compute_dtype(torch.float32)
    pol.config.single_frame_encode = True
    if num_steps is not None:
        pol.config.num_inference_steps = num_steps
    captured: dict = {}
    orig_sampler = up_sampling.cosmos_unipc_order2
    orig_enc = up_packing.encode_single_frame

    def sampler(*a, **k):
        out = orig_sampler(*a, **k)
        captured["flow"] = {kk: v.detach().clone() for kk, v in out.items()}
        return out

    def enc(*a, **k):
        lat = orig_enc(*a, **k)
        captured["cond_latent"] = lat.detach().clone()
        return lat

    up_sampling.cosmos_unipc_order2 = sampler
    up_policy.packing.encode_single_frame = enc
    try:
        pol.config.inference_seed = seed
        t0 = time.time()
        targets = pol.predict_normalized_targets(batch)
        actions = pol.actions_from_targets(targets, pol._state(batch))
        dt = time.time() - t0
    finally:
        up_sampling.cosmos_unipc_order2 = orig_sampler
        up_policy.packing.encode_single_frame = orig_enc
    cfg = pol.config
    n_pred = up_packing.latent_frames(cfg.window_frames) - 1
    h, w = cfg.latent_hw
    vid = captured["flow"]["x_video"][0].reshape(n_pred, h, w, -1).permute(3, 0, 1, 2)[None]
    return {
        "actions": actions.float(),
        "targets": targets.float(),
        "video_latents": vid.float(),
        "cond_latent": captured["cond_latent"].float(),
        "seconds": dt,
        "dtype": dtype,
        "config": cfg.to_dict(),
    }


def main() -> None:
    from vllm_omni_neuron.diffusion.models.flux3_action.observation import (
        load_observation as load_batch,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--dtype",
        choices=("keep", "float32"),
        default="keep",
        help="keep = package dtype (bf16 for released DROID); float32 = fp32 oracle",
    )
    ap.add_argument(
        "--num-steps", type=int, default=None, help="sampler steps (default: the package's)"
    )
    a = ap.parse_args()
    ref = run_reference(
        a.policy,
        a.base,
        load_batch(a.obs),
        seed=a.seed,
        threads=a.threads,
        dtype=a.dtype,
        num_steps=a.num_steps,
    )
    torch.save(ref, a.out)
    print(
        {
            "out": a.out,
            "dtype": a.dtype,
            "seconds": round(ref["seconds"], 1),
            "actions_mean": ref["actions"].mean().item(),
        }
    )


if __name__ == "__main__":
    main()
