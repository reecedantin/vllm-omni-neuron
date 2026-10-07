# SPDX-License-Identifier: Apache-2.0
"""M1 parity for Qwen-Image 2.1: device (TP=4, bf16) final latents vs a CPU fp32 oracle, judged
against the CPU-bf16 band (device rel-L2 <= 2 x CPU-bf16-band + 0.5%, LESSONS section 9 / A0).

Two phases, like A7's LTX-2.5 M1 (the 7B DiT + 8B Qwen3-VL encoder fit on CPU fp32 in host RAM but
not on one NeuronCore):

    # 1. oracle (CPU, once; big RAM, no cores): fp32 + a bf16 band run, latents saved to disk
    python -m test.neuron.test_qwen_image_m1_parity --model $WEIGHTS/qwen-image-21 --mode oracle --out $FLEET_RUNS/m1par
    # 2. device (TP=4) + compare against the cached oracle
    torchrun --nproc_per_node 4 -m test.neuron.test_qwen_image_m1_parity --model $WEIGHTS/qwen-image-21 --mode device --out $FLEET_RUNS/m1par

Latents are the gate: pixels below the VAE cannot settle parity (LESSONS section 6). A small but
non-degenerate request keeps the CPU oracle tractable.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import vllm_omni_neuron  # noqa: F401  isort: skip  bootstrap before vllm

PROMPT = "A cozy bookshop window on a rainy evening, warm light"
REQUEST = dict(height=256, width=256, num_inference_steps=8, seed=42)


def _gen_latent(
    model: str, dtype: torch.dtype, device: torch.device, compile_it: bool
) -> np.ndarray:
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    p = NeuronQwenImage21Pipeline(model_path=model, dtype=dtype)
    p.load_weights()
    if device.type != "cpu":
        p.to(device)
        if compile_it:
            p.compile(backend=get_compile_backend_name())
    lat = p.generate(
        PROMPT,
        height=REQUEST["height"],
        width=REQUEST["width"],
        num_inference_steps=REQUEST["num_inference_steps"],
        seed=REQUEST["seed"],
        output_type="latent",
    )
    del p
    return lat.float().cpu().numpy()


def run_oracle(model: str, out_dir: str) -> None:
    _init_tp()
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    fp32 = _gen_latent(model, torch.float32, torch.device("cpu"), False)
    bf16 = _gen_latent(model, torch.bfloat16, torch.device("cpu"), False)
    np.save(os.path.join(out_dir, "oracle_latent.npy"), fp32)
    np.save(os.path.join(out_dir, "oracle_bf16_latent.npy"), bf16)
    band = float(
        np.linalg.norm(fp32.ravel() - bf16.ravel()) / max(np.linalg.norm(fp32.ravel()), 1e-12)
    )
    meta = {
        "gen_s": round(time.time() - t0, 1),
        "latent_shape": list(fp32.shape),
        "cpu_bf16_band": round(band, 5),
    }
    with open(os.path.join(out_dir, "oracle_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("ORACLE_DONE " + json.dumps(meta), flush=True)


def _pin_rank_core(local_rank: int) -> None:
    """torchrun workers all inherit the job's NEURON_RT_VISIBLE_CORES range; narrow each to ONE core
    by LOCAL_RANK before the Lite runtime opens a device, or ranks 1-3 race rank 0 for the whole
    range (LESSONS section 7, A7)."""
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if "-" in cores and "," not in cores:
        lo, hi = (int(x) for x in cores.split("-"))
        ids = list(range(lo, hi + 1))
    elif cores:
        ids = [int(x) for x in cores.split(",")]
    else:
        ids = []
    if local_rank < len(ids):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(ids[local_rank])


_VLLM_CTX = None


def _init_tp():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        _pin_rank_core(local_rank)
    global _VLLM_CTX
    if world <= 1:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel

        _VLLM_CTX = set_current_vllm_config(VllmConfig())
        _VLLM_CTX.__enter__()
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        return 1, 0
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_neuron.envs import get_dist_backend

    _VLLM_CTX = set_current_vllm_config(VllmConfig())
    _VLLM_CTX.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend=get_dist_backend(),
    )
    initialize_model_parallel(world, 1)
    return world, rank


def run_device(model: str, out_dir: str) -> None:
    world, rank = _init_tp()
    dev = torch.device("neuron", 0) if os.path.exists("/dev/neuron0") else torch.device("cpu")
    lat = _gen_latent(model, torch.bfloat16, dev, dev.type != "cpu")
    if rank != 0:
        return
    np.save(os.path.join(out_dir, "device_latent.npy"), lat)
    fp32 = np.load(os.path.join(out_dir, "oracle_latent.npy"))
    bf16 = np.load(os.path.join(out_dir, "oracle_bf16_latent.npy"))
    r, p, b = fp32.ravel(), lat.ravel(), bf16.ravel()
    rn = max(np.linalg.norm(r), 1e-12)
    rep = {
        "world": world,
        "latent_shape": list(lat.shape),
        "finite": bool(np.isfinite(lat).all()),
        "rel_dev": round(float(np.linalg.norm(r - p) / rn), 5),
        "cpu_bf16_band": round(float(np.linalg.norm(r - b) / rn), 5),
        "cos_dev": round(float(np.vdot(r, p) / (rn * max(np.linalg.norm(p), 1e-12))), 5),
    }
    rep["bar"] = round(2.0 * rep["cpu_bf16_band"] + 0.005, 5)
    rep["pass"] = bool(rep["finite"] and rep["rel_dev"] <= rep["bar"])
    with open(os.path.join(out_dir, "parity.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("M1_PARITY " + json.dumps(rep), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("oracle", "device"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (run_oracle if a.mode == "oracle" else run_device)(a.model, a.out)


if __name__ == "__main__":
    main()
