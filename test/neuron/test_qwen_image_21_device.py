# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 on one NeuronCore: every compiled graph and the whole pipeline vs CPU.

Three runs of the same pipeline and seed: CPU fp32 (oracle), CPU bf16 (precision band), Neuron
bf16 (compiled text encoder, DiT prefix, DiT target, VAE decode). Reports per-stage rel-L2 /
cosine against the oracle, the CPU-bf16 band, first-call (compile) and warm times, and
determinism, as one JSON line (``QWEN21_DEVICE {...}``) plus ``<out>/device_<tag>.json``.

Usage (inside a fleet job)::

    python test/neuron/test_qwen_image_21_device.py --model <dir> --height 256 --width 256 --steps 4 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import torch

import vllm_omni_neuron  # noqa: F401  isort: skip  (bootstrap before vllm)


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _psnr(a, b):  # images in [-1, 1]
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


_VLLM_CTX = None


def _init_single_rank():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    global _VLLM_CTX
    _VLLM_CTX = set_current_vllm_config(
        VllmConfig()
    )  # keep a reference: the context must stay entered
    _VLLM_CTX.__enter__()
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


def run(
    model: str,
    height: int,
    width: int,
    steps: int,
    prompt: str,
    seed: int,
    out: str,
    tag: str,
    refs: tuple = ("cpu32", "cpu16"),
) -> dict:
    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    rep: dict = {
        "model": os.path.basename(model.rstrip("/")),
        "hw": [height, width],
        "steps": steps,
    }
    res = {}
    dtypes = {"cpu32": torch.float32, "cpu16": torch.bfloat16}
    for name in refs:  # a 15B model has no tractable CPU fp32 oracle; pass refs=("cpu16",) then
        dtype = dtypes[name]
        t0 = time.time()
        p = NeuronQwenImage21Pipeline(model_path=model, dtype=dtype)
        p.load_weights()
        emb, _ = p.encode_prompt(prompt)
        lat = p.generate(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed,
            output_type="latent",
        )
        img = p.generate(prompt, height=height, width=width, num_inference_steps=steps, seed=seed)
        res[name] = {"emb": emb, "lat": lat, "img": img}
        rep[f"{name}_s"] = round(time.time() - t0, 1)
        del p

    from vllm_neuron.envs import get_compile_backend_name

    dev = torch.device(os.environ.get("QWEN21_DEVICE", "neuron:0"))  # "cpu" = dry run of the script
    t0 = time.time()
    p = NeuronQwenImage21Pipeline(model_path=model, dtype=torch.bfloat16)
    p.load_weights()
    p.to(dev)
    if dev.type != "cpu":
        p.compile(backend=get_compile_backend_name())
    rep["load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    emb, _ = p.encode_prompt(prompt)
    lat = p.generate(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=seed,
        output_type="latent",
    )
    img = p.generate(prompt, height=height, width=width, num_inference_steps=steps, seed=seed)
    rep["first_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    img2 = p.generate(prompt, height=height, width=width, num_inference_steps=steps, seed=seed)
    rep["warm_s"] = round(time.time() - t0, 2)
    rep["warm_stats"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in p.stats.items()}
    rep["deterministic"] = bool(torch.equal(img, img2))

    ref = res.get("cpu32") or res["cpu16"]  # fp32 oracle when present, else the bf16 CPU run
    band = res["cpu16"]
    ref_name = "cpu32" if "cpu32" in res else "cpu16"
    rep["ref"] = ref_name
    for key in ("emb", "lat", "img"):
        got = {"emb": emb, "lat": lat, "img": img}[key]
        rep[f"{key}_rel_dev"] = round(_rel(got, ref[key]), 5)
        rep[f"{key}_rel_cpu16"] = round(_rel(band[key], ref[key]), 5)
    rep["lat_cos_dev"] = round(_cos(lat, ref["lat"]), 5)
    rep["img_psnr_dev"] = round(_psnr(img, ref["img"]), 2)
    rep["img_psnr_cpu16"] = round(_psnr(band["img"], ref["img"]), 2)
    os.makedirs(out, exist_ok=True)
    torch.save(
        {"dev": img, "cpu32": ref["img"], "cpu16": band["img"]},
        os.path.join(out, f"images_{tag}.pt"),
    )
    if ref_name == "cpu16":
        # No fp32 oracle (model too big for CPU fp32): the reference IS the bf16 CPU run, so
        # rel_cpu16 is 0 and the bar is the absolute device-vs-CPU-bf16 floor — the device must add
        # no more than this on top of the identical-dtype CPU result.
        rep["pass"] = bool(
            rep["deterministic"] and all(rep[f"{k}_rel_dev"] <= 0.005 for k in ("emb", "lat"))
        )
    else:
        tol = lambda k: max(2.0 * rep[f"{k}_rel_cpu16"], 0.005)  # noqa: E731  device <= 2x CPU-bf16 err + 0.5%
        rep["pass"] = bool(
            rep["deterministic"] and all(rep[f"{k}_rel_dev"] <= tol(k) for k in ("emb", "lat"))
        )
    with open(os.path.join(out, f"device_{tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def test_device_tiny(tmp_path):
    import pytest

    if not _neuron_available():
        pytest.skip("needs a Neuron device")
    model = os.environ.get("QWEN_IMAGE21_TINY", "")
    if not os.path.isdir(os.path.join(model, "transformer")):
        pytest.skip("set QWEN_IMAGE21_TINY to a checkpoint from test/unit/test_qwen_image_tiny.py")
    _init_single_rank()
    rep = run(model, 256, 256, 3, "a cat", 0, str(tmp_path), "pytest")
    assert rep["pass"], rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="A red fox reading a newspaper on a park bench, watercolor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="tp1")
    ap.add_argument(
        "--refs",
        default="cpu32,cpu16",
        help="CPU reference runs: 'cpu32,cpu16' (small models) or 'cpu16' (no fp32 oracle)",
    )
    a = ap.parse_args()
    _init_single_rank()
    rep = run(
        a.model,
        a.height,
        a.width,
        a.steps,
        a.prompt,
        a.seed,
        a.out,
        a.tag,
        refs=tuple(a.refs.split(",")),
    )
    print("QWEN21_DEVICE " + json.dumps(rep), flush=True)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
