# SPDX-License-Identifier: Apache-2.0
"""Warm-latency breakdown and layout sweep for Qwen-Image 2.1: per-stage time (text encoder, DiT
prefix, DiT target mean/sum over all denoising steps, VAE decode), each timed to device completion
(every compiled call returns its output to the host), median of ``--repeats`` warm requests after
one cold request (which compiles every graph and is reported as ``first_request_s``). Before each
timed request rank 0 logs ``/proc/loadavg`` and waits for a quiet host (``--load-limit``).

At each layout it also runs the accuracy check (``--parity``: the M1 request vs the cached CPU fp32
oracle from ``test_qwen_image_m1_parity --mode oracle``, same bar: rel-L2 <= 2 x CPU-bf16 band +
0.5%) and, for the VAE, compares the tile-parallel decode against the output-rank-only decode of
the same latent (``QWEN_IMAGE_VAE_PARALLEL=1`` vs ``0``, both in this process).

    torchrun --nproc_per_node 4 -m test.neuron.test_qwen_image_m3_perf --model $WEIGHTS/qwen-image-21 \\
        --height 1024 --width 1024 --steps 40 --repeats 3 --parity <m1par dir> [--cfg 4.0]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

import vllm_omni_neuron  # noqa: F401  isort: skip  bootstrap before vllm

# reuse the proven torchrun TP-init + core pinning
from .test_qwen_image_m1_parity import _init_tp  # noqa: E402

PROMPT = 'A cozy bookshop window on a rainy evening, warm light, a hand-lettered sign that reads "Open Late"'
STAGES = (
    "text_encoder_s",
    "dit_prefix_s",
    "dit_target_s_sum",
    "dit_target_s_mean",
    "vae_s",
    "total_s",
)


def _hbm_per_core() -> dict | None:
    """Best-effort HBM in use per logical core of this job (``FLEET_CORE_LIST``) from neuron-monitor."""
    import subprocess

    cores = {c.strip() for c in os.environ.get("FLEET_CORE_LIST", "").split(",") if c.strip()}
    cfg = {
        "period": "1s",
        "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "memory_used"}]}],
    }
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"qwen_nmon_{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    p = None
    try:
        p = subprocess.Popen(
            ["neuron-monitor", "-c", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        t0, gb = time.time(), {}
        while time.time() - t0 < 15:
            line = p.stdout.readline()
            if not line.strip().startswith("{"):
                continue
            for rt in json.loads(line).get("neuron_runtime_data", []):
                mu = (
                    rt.get("report", {}).get("memory_used", {}).get("neuron_runtime_used_bytes", {})
                )
                for c, v in (
                    mu.get("usage_breakdown", {}).get("neuroncore_memory_usage", {}).items()
                ):
                    if c in cores and sum(v.values()) > 0:
                        gb[c] = round(sum(v.values()) / 2**30, 2)
            if gb:
                return {"per_core_gb": gb, "max_gb": max(gb.values())}
        return {"error": "no runtime on our cores"}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        if p is not None:
            p.kill()


def _quiet_gate(rank: int, world: int, limit: float, tries: int = 5, wait_s: int = 60) -> dict:
    """Before a timed request: rank 0 logs ``/proc/loadavg`` and, while the 1-minute load is at or
    above ``limit``, waits ``wait_s`` and re-reads, up to ``tries`` times (then times anyway). Every
    rank leaves together with rank 0's reading, so the request starts in lockstep."""
    rec = None
    if rank == 0:
        for attempt in range(tries + 1):
            with open("/proc/loadavg") as f:
                line = f.read().strip()
            load1 = float(line.split()[0])
            print(f"LOADAVG {line}", flush=True)
            if load1 < limit or attempt == tries:
                break
            time.sleep(wait_s)
        rec = {"load1": load1, "loadavg": line, "retries": attempt}
    if world > 1:
        from vllm.distributed.parallel_state import get_tp_group

        out = [None] * world
        torch.distributed.all_gather_object(out, rec, group=get_tp_group().cpu_group)
        rec = out[0]
    return rec


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()  # images in [-1, 1]: peak-to-peak 2
    return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)


def _parity(p, oracle_dir: str) -> dict:
    from .test_qwen_image_m1_parity import PROMPT as M1_PROMPT
    from .test_qwen_image_m1_parity import REQUEST

    lat = (
        p.generate(
            M1_PROMPT,
            height=REQUEST["height"],
            width=REQUEST["width"],
            num_inference_steps=REQUEST["num_inference_steps"],
            seed=REQUEST["seed"],
            output_type="latent",
        )
        .float()
        .numpy()
        .ravel()
    )
    fp32 = np.load(os.path.join(oracle_dir, "oracle_latent.npy")).ravel()
    bf16 = np.load(os.path.join(oracle_dir, "oracle_bf16_latent.npy")).ravel()
    rn = max(np.linalg.norm(fp32), 1e-12)
    rel, band = float(np.linalg.norm(fp32 - lat) / rn), float(np.linalg.norm(fp32 - bf16) / rn)
    bar = 2 * band + 0.005
    return {
        "rel_dev": round(rel, 5),
        "cpu_bf16_band": round(band, 5),
        "bar": round(bar, 5),
        "cos": round(float(np.vdot(fp32, lat) / (rn * max(np.linalg.norm(lat), 1e-12))), 5),
        "pass": bool(np.isfinite(lat).all() and rel <= bar),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--cfg",
        type=float,
        default=0.0,
        help="also time true CFG at this scale (negative prompt ' ')",
    )
    ap.add_argument("--parity", default=None, help="m1par dir holding the CPU oracle latents")
    ap.add_argument(
        "--vae-compare", action="store_true", help="tile-parallel vs output-rank-only VAE decode"
    )
    ap.add_argument(
        "--load-limit",
        type=float,
        default=20.0,
        help="wait (60 s, up to 5 times) before a timed request while the 1-minute load is >= this",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    world, rank = _init_tp()
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    dev = torch.device("neuron", 0) if os.path.exists("/dev/neuron0") else torch.device("cpu")
    t_load = time.time()
    p = NeuronQwenImage21Pipeline(model_path=a.model, dtype=torch.bfloat16)
    p.load_weights()
    p.to(dev)
    if dev.type != "cpu":
        p.compile(backend=get_compile_backend_name())
    load_s = time.time() - t_load
    req = dict(height=a.height, width=a.width, num_inference_steps=a.steps, seed=a.seed)

    def timed(counted=True, **kw):
        load = _quiet_gate(rank, world, a.load_limit) if counted else {}
        t0 = time.time()
        img = p.generate(PROMPT, **req, **kw)
        return img, {"wall_s": time.time() - t0, **p.stats, **load}

    def summary(runs):
        keys = ["wall_s", *STAGES]
        return {
            k: round(statistics.median(r[k] for r in runs), 3)
            for k in keys
            if all(k in r for r in runs)
        }

    # One uncounted full-image request first: compiles every graph (incl. the VAE on its first call).
    _, first = timed(counted=False)
    runs, img = [], None
    for _ in range(max(a.repeats, 1)):
        img, st = timed()
        runs.append(st)
    rep = {
        "tag": a.tag,
        "world": world,
        "hw": [a.height, a.width],
        "steps": a.steps,
        "load_s": round(load_s, 1),
        "first_request_s": round(first["wall_s"], 1),
        "repeats": len(runs),
        "warm": summary(runs),
        "warm_wall_all": [round(r["wall_s"], 3) for r in runs],
        "warm_load1_all": [r.get("load1") for r in runs],
    }
    if a.vae_compare:
        os.environ["QWEN_IMAGE_VAE_PARALLEL"] = "0"
        ref_runs = []
        for _ in range(max(a.repeats, 1)):
            ref_img, st = timed()
            ref_runs.append(st)
        os.environ["QWEN_IMAGE_VAE_PARALLEL"] = "1"
        rep["warm_vae_rank0_only"] = {
            **summary(ref_runs),
            "wall_all": [round(r["wall_s"], 3) for r in ref_runs],
            "load1_all": [r.get("load1") for r in ref_runs],
        }
        if rank == 0:
            rep["vae_parallel_vs_rank0"] = {
                "max_abs": round((img.float() - ref_img.float()).abs().max().item(), 5),
                "psnr_db": round(_psnr(img, ref_img), 2),
            }
    if a.cfg > 1:
        # uncounted: warms the negative prompt's prefix bucket
        timed(counted=False, negative_prompt=" ", true_cfg_scale=a.cfg)
        cfg_runs = [
            timed(negative_prompt=" ", true_cfg_scale=a.cfg)[1] for _ in range(max(a.repeats, 1))
        ]
        rep["warm_cfg"] = {
            "scale": a.cfg,
            **summary(cfg_runs),
            "wall_all": [round(r["wall_s"], 3) for r in cfg_runs],
            "load1_all": [r.get("load1") for r in cfg_runs],
        }
    if a.parity:
        rep["parity"] = _parity(p, a.parity)
    if rank == 0:
        print("M3_PART " + json.dumps(rep), flush=True)
    hbm = _hbm_per_core() if (rank == 0 and dev.type != "cpu") else None
    if world > 1:  # keep every rank's runtime alive until rank 0 has read the HBM in use (a host
        # object collective: barrier() on this group reaches for the Neuron device and fails)
        from vllm.distributed.parallel_state import get_tp_group

        torch.distributed.all_gather_object([None] * world, rank, group=get_tp_group().cpu_group)
    if rank != 0:
        return
    rep["hbm"] = hbm
    print("M3_PERF " + json.dumps(rep), flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"perf{('-' + a.tag) if a.tag else ''}.json"), "w") as f:
            json.dump(rep, f, indent=2)
        if img is not None:
            from PIL import Image

            arr = (
                ((img[0, :3].float().clamp(-1, 1) + 1) * 127.5)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            Image.fromarray(arr).save(
                os.path.join(a.out, f"sample{('-' + a.tag) if a.tag else ''}.png")
            )


if __name__ == "__main__":
    main()
