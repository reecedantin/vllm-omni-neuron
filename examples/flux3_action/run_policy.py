# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action policy on Neuron (offline): one observation -> action chunk, with parity vs a reference.

Usage (single core):
    python examples/flux3_action/run_policy.py --policy $WEIGHTS/flux3-action-droid \
        --base $WEIGHTS/flux3-action-base --obs observation.npz --out-dir out/ [--reference ref.pt]

Tensor parallel (TP ranks on adjacent NeuronCores of one chip):
    torchrun --nproc-per-node 4 examples/flux3_action/run_policy.py --tp 4 ...

``--device cpu`` runs the same port eagerly on the host. ``--reference`` is a file written by
``test/unit/test_flux3_action_utils/reference.py`` (upstream policy on CPU); the script then reports
action MSE / rel-L2 and video-latent rel-L2 / cosine against it. A one-line JSON summary is the
last line printed; the exit code is nonzero when a ``--gate-*`` threshold fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--policy", required=True)
ap.add_argument("--base", default=None)
ap.add_argument("--obs", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--reference", default=None)
ap.add_argument("--device", default="neuron", choices=("neuron", "cpu"))
ap.add_argument("--tp", type=int, default=1)
ap.add_argument(
    "--repeat", type=int, default=2, help="timed predictions after the first (warm latency)"
)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gate-action-mse", type=float, default=None)
ap.add_argument("--gate-video-cos", type=float, default=None)
ap.add_argument(
    "--decode", action="store_true", help="decode observation + predicted frames on the host"
)
ap.add_argument(
    "--decode-max-frames",
    type=int,
    default=None,
    help="cap latent frames decoded (quick frame sample; CPU decode is heavy)",
)
ap.add_argument(
    "--no-compile-vae",
    action="store_true",
    help="leave the VAE encoder eager (isolate DiT compiles)",
)
ap.add_argument(
    "--vae-host",
    dest="vae_host",
    action="store_true",
    default=False,
    help="run the VAE encoder on the host CPU instead of the Neuron device",
)
ap.add_argument(
    "--vae-device",
    dest="vae_host",
    action="store_false",
    help="run the VAE encoder on the Neuron device (default)",
)
ap.add_argument(
    "--guidance-override",
    type=float,
    default=None,
    help="force guidance_scale and _action (1.0 drops the CFG negative branch; diagnostic)",
)
args = ap.parse_args()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _hbm_per_core() -> dict | None:
    """HBM used, via one neuron-monitor sample: total device bytes + per-NeuronCore breakdown. None/err dict on failure."""
    import json as _json
    import subprocess

    try:
        proc = subprocess.run(
            ["neuron-monitor"],
            input='{"period":"0.5s"}',
            capture_output=True,
            text=True,
            timeout=60,
        )
        last = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
        rep = _json.loads(last)
        out: dict = {}
        for dev in rep.get("neuron_runtime_data", []):
            used = dev.get("report", {}).get("memory_used", {}).get("neuron_runtime_used_bytes", {})
            out["device_total_gb"] = round(used.get("neuron_device", 0) / 1e9, 3)
            cores = used.get("usage_breakdown", {}).get("neuroncore_memory_usage", {})
            for core, c in cores.items():
                tot = sum(v for v in c.values() if isinstance(v, (int, float)))
                if tot:
                    out[f"nc{core}_gb"] = round(tot / 1e9, 3)
            if out.get("device_total_gb"):
                break  # the first runtime with memory is ours
        return out or {"note": "no memory sample"}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _init_tp():
    if args.tp == 1:
        return 1, 0, None
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    rank = int(os.environ["RANK"])
    set_current_vllm_config(VllmConfig()).__enter__()
    init_distributed_environment(
        world_size=args.tp,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT', '29500')}",
        backend="gloo" if args.device == "cpu" else "neuron",
    )
    initialize_model_parallel(args.tp, 1)
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import _tp_state

    return _tp_state()


def main() -> None:
    if args.device == "neuron":
        import vllm_omni_neuron.bootstrap  # noqa: F401
    from vllm_omni_neuron.diffusion.models.flux3_action.observation import (
        load_observation as load_batch,
    )
    from vllm_omni_neuron.diffusion.models.flux3_action.policy import NeuronFlux3ActionPolicy

    tp, rank, group = _init_tp()
    dev = torch.device("neuron", 0) if args.device == "neuron" else torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    vae_dev = torch.device("cpu") if (args.device == "neuron" and args.vae_host) else dev
    pol = NeuronFlux3ActionPolicy(
        args.policy,
        base_dir=args.base,
        device=dev,
        tp_size=tp,
        tp_rank=rank,
        tp_group=group,
        vae_device=vae_dev,
    )
    load_s = time.time() - t0
    if args.device == "neuron":
        from vllm_neuron.envs import get_compile_backend_name

        pol.compile(get_compile_backend_name(), compile_vae=not args.no_compile_vae)
    if args.guidance_override is not None:
        pol.config.guidance_scale = args.guidance_override
        pol.config.guidance_scale_action = args.guidance_override
    batch = load_batch(args.obs)
    t0 = time.time()
    out = pol.predict(batch, seed=args.seed)
    first_s = time.time() - t0
    warm = []
    for _ in range(args.repeat):
        t0 = time.time()
        out2 = pol.predict(batch, seed=args.seed)
        warm.append(time.time() - t0)
    rep = {
        "device": args.device,
        "tp": tp,
        "vae_device": str(vae_dev),
        "load_s": round(load_s, 1),
        "load_parts_s": pol.load_s,
        "first_s": round(first_s, 2),
        "warm_s": [round(x, 3) for x in warm],
        "timing": out.timing,
        "dit_param_gb_per_rank": round(pol.dit.param_bytes() / 1e9, 3),
        "deterministic": bool(args.repeat == 0 or torch.equal(out.actions, out2.actions)),
        "actions_shape": list(out.actions.shape),
    }
    if args.device == "neuron":
        rep["hbm_per_core"] = _hbm_per_core()
    ok = True
    if args.reference:
        ref = torch.load(args.reference)
        rep["parity"] = {
            "action_mse": ((out.actions - ref["actions"]) ** 2).mean().item(),
            "action_rel": _rel(out.actions, ref["actions"]),
            "action_max_abs": (out.actions - ref["actions"]).abs().max().item(),
            "video_rel": _rel(out.video_latents, ref["video_latents"]),
            "video_cos": _cos(out.video_latents, ref["video_latents"]),
            "cond_rel": _rel(out.cond_latents, ref["cond_latent"]),
        }
        if args.gate_action_mse is not None and rep["parity"]["action_mse"] > args.gate_action_mse:
            ok = False
        if args.gate_video_cos is not None and rep["parity"]["video_cos"] < args.gate_video_cos:
            ok = False
    if rank == 0:
        torch.save(
            {
                "actions": out.actions,
                "targets": out.targets,
                "video_latents": out.video_latents,
                "cond_latent": out.cond_latents,
            },
            os.path.join(args.out_dir, "outputs.pt"),
        )
        if args.decode:
            frames = pol.decode_frames(out, max_latent_frames=args.decode_max_frames)
            torch.save(frames, os.path.join(args.out_dir, "frames.pt"))
            rep["frames_shape"] = list(frames.shape)
        with open(os.path.join(args.out_dir, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        rep["ok"] = ok
        print(json.dumps(rep))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
