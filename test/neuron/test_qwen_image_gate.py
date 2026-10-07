# SPDX-License-Identifier: Apache-2.0
"""Full-size accuracy gate for Qwen-Image 2.1 at a published TP layout, with all-rank agreement.

Three phases (the CPU phases need no cores; run them after ``fleet/bin/cpumode.sh``):

    # 1. CPU oracle for one request: fp32 + bf16 (the band), final latents + the step-0 velocity
    python -m test.neuron.test_qwen_image_gate --mode oracle --model $W --height 1024 --width 1024 \\
        --steps 40 --out $RUNS/gate/oracle-1024
    #    (--probe-only: the step-0 velocity alone, a single DiT forward; for sizes where a CPU
    #    trajectory is too slow)
    # 2. device run at a TP layout: EVERY rank saves its step-0 velocity and final latents
    torchrun --nproc_per_node 16 -m test.neuron.test_qwen_image_gate --mode device --model $W \\
        --height 1024 --width 1024 --steps 40 --out $RUNS/gate/tp16-1024
    # 3. compare on the host: all ranks agree, device vs fp32 within k x the CPU-bf16 band, and
    #    the device VAE vs a host fp32 decode of the same device latents (DiT vs VAE bisection)
    python -m test.neuron.test_qwen_image_gate --mode compare --model $W --oracle <oracle dir> \\
        --device-dir $RUNS/gate/tp16-1024

Bar (both the step-0 velocity and the final latents): rel-L2 vs fp32 <= K x CPU-bf16 rel-L2 +
0.5% (K = 2, the M1 bar). Latents are the gate; pixels are reported, not gated (LESSONS 6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import vllm_omni_neuron  # noqa: F401  isort: skip  bootstrap before vllm

from .test_qwen_image_m1_parity import _init_tp  # noqa: E402

PROMPT = 'A cozy bookshop window on a rainy evening, warm light, a hand-lettered sign that reads "Open Late"'
K_BAND = 2.0
FLOOR = 0.005


class _ProbeDone(Exception):
    pass


def _capture_v0(p, stop: bool) -> dict:
    """Wrap the pipeline's velocity call: keep the first call's input latents and output."""
    box: dict = {}
    inner = p._velocity

    def wrapped(latents, t, branch):
        out = inner(latents, t, branch)
        if "v0" not in box:
            box.update(x0=latents.float().clone(), t0=float(t), v0=out.float().clone())
            if stop:
                raise _ProbeDone
        return out

    p._velocity = wrapped
    return box


def _build(model: str, dtype, device):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    p = NeuronQwenImage21Pipeline(model_path=model, dtype=dtype)
    p.load_weights()
    if device.type != "cpu":
        p.to(device)
        p.compile(backend=get_compile_backend_name())
    return p


def _req(a):
    return dict(height=a.height, width=a.width, num_inference_steps=a.steps, seed=a.seed)


def _save_png(img: torch.Tensor, path: str) -> None:
    from PIL import Image

    arr = ((img[0, :3].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def run_oracle(a) -> None:
    _init_tp()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    meta_path = os.path.join(a.out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"req": _req(a), "prompt": PROMPT, "probe_only": a.probe_only})
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        if name not in a.dtypes:
            continue
        t0 = time.time()
        p = _build(a.model, dtype, torch.device("cpu"))
        box = _capture_v0(p, a.probe_only)
        try:
            lat = p.generate(PROMPT, **_req(a), output_type="latent").float().numpy()
            np.save(os.path.join(a.out, f"latent_{name}.npy"), lat)
        except _ProbeDone:
            pass
        np.save(os.path.join(a.out, f"v0_{name}.npy"), box["v0"].numpy())
        if name == "fp32":
            np.save(os.path.join(a.out, "x0.npy"), box["x0"].numpy())
        meta[f"{name}_s"] = round(time.time() - t0, 1)
        print(f"ORACLE {name} {meta[f'{name}_s']} s", flush=True)
        del p
        with open(meta_path, "w") as f:  # after each dtype, so a lost later phase keeps this one
            json.dump(meta, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("ORACLE_DONE " + json.dumps(meta), flush=True)


def run_device(a) -> None:
    world, rank = _init_tp()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("neuron", 0) if os.path.exists("/dev/neuron0") else torch.device("cpu")
    t0 = time.time()
    p = _build(a.model, torch.bfloat16, dev)
    load_s = time.time() - t0
    box = _capture_v0(p, False)
    t0 = time.time()
    lat = p.generate(PROMPT, **_req(a), output_type="latent")
    gen_s = time.time() - t0
    h, w = a.height // 32 * 32, a.width // 32 * 32
    t0 = time.time()
    img = p.decode_latents(lat, h, w)
    vae_s = time.time() - t0
    lat_np, v0_np = lat.float().numpy(), box["v0"].numpy()
    np.save(os.path.join(a.out, f"latent_rank{rank}.npy"), lat_np)
    np.save(os.path.join(a.out, f"v0_rank{rank}.npy"), v0_np)
    from vllm_omni_neuron.testing import check_rank_agreement

    coord = None
    if world > 1:
        from vllm.distributed.parallel_state import get_tp_group

        coord = get_tp_group()
    agree = check_rank_agreement({"latent": lat, "v0": box["v0"]}, coord=coord)
    if rank != 0:
        return
    np.save(os.path.join(a.out, "x0.npy"), box["x0"].numpy())
    np.save(os.path.join(a.out, "image_device.npy"), img.float().numpy())
    _save_png(img, os.path.join(a.out, "image_device.png"))
    rep = {
        "world": world,
        "req": _req(a),
        "load_s": round(load_s, 1),
        "first_gen_s": round(gen_s, 1),
        "vae_s": round(vae_s, 2),
        "ranks_agree": agree.ok,
        "rank_agreement": agree.summary(),
        "latent_digest": _digest(lat_np),
        "v0_digest": _digest(v0_np),
    }
    with open(os.path.join(a.out, "device.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("GATE_DEVICE " + json.dumps(rep), flush=True)


def _rel(ref: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm((ref - x).ravel()) / max(np.linalg.norm(ref.ravel()), 1e-12))


def _cos(ref: np.ndarray, x: np.ndarray) -> float:
    r, y = ref.ravel().astype(np.float64), x.ravel().astype(np.float64)
    return float(r @ y / max(np.linalg.norm(r) * np.linalg.norm(y), 1e-12))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(
        np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    )  # [-1, 1]: peak-to-peak 2
    return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)


def _host_decode(model: str, lat: np.ndarray, h: int, w: int) -> np.ndarray:
    """fp32 CPU decode of packed latents, untiled (the ground truth the device VAE tiles approximate)."""
    from vllm_omni_neuron.diffusion.models.qwen_image import NeuronQwenImage21Pipeline

    os.environ["QWEN_IMAGE_VAE_TILE"] = "0"
    p = NeuronQwenImage21Pipeline.__new__(NeuronQwenImage21Pipeline)
    torch.nn.Module.__init__(p)
    from vllm_omni_neuron.diffusion.models.qwen_image.vae_qwenimage21 import NeuronQwenImage21VAE

    p.vae = NeuronQwenImage21VAE.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
    p.latent_channels = p.vae.config.z_dim
    p.stats = {}
    return p.decode_latents(torch.from_numpy(lat), h, w).float().numpy()


def run_compare(a) -> None:
    _init_tp()
    torch.set_num_threads(a.threads)
    d, o = a.device_dir, a.oracle
    dev = json.load(open(os.path.join(d, "device.json")))
    world = dev["world"]
    rep = {
        "device_dir": d,
        "oracle": o,
        "world": world,
        "req": dev["req"],
        "k": K_BAND,
        "floor": FLOOR,
    }
    # all ranks agree (bitwise, compared directly, not only by digest)
    lat0, v00 = (
        np.load(os.path.join(d, "latent_rank0.npy")),
        np.load(os.path.join(d, "v0_rank0.npy")),
    )
    worst = 0.0
    for r in range(1, world):
        worst = max(
            worst,
            float(np.abs(np.load(os.path.join(d, f"latent_rank{r}.npy")) - lat0).max()),
            float(np.abs(np.load(os.path.join(d, f"v0_rank{r}.npy")) - v00).max()),
        )
    rep["ranks_max_abs_diff"] = worst
    rep["ranks_agree"] = worst == 0.0
    x0_ref = np.load(os.path.join(o, "x0.npy"))
    # the fp32 oracle starts from the fp32 noise, the bf16 device from the same noise cast to bf16
    rep["x0_rel"] = round(_rel(x0_ref, np.load(os.path.join(d, "x0.npy"))), 5)
    rep["same_x0"] = rep["x0_rel"] < 1e-2
    ok = rep["ranks_agree"] and rep["same_x0"] and bool(np.isfinite(lat0).all())
    meta_path = os.path.join(o, "meta.json")
    probe_only = os.path.exists(meta_path) and json.load(open(meta_path)).get("probe_only", False)
    rep["probe_only_oracle"] = probe_only  # no CPU trajectory: the step-0 velocity is the gate
    for key, dev_arr in (("v0", v00), ("latent", lat0)):
        if probe_only and key == "latent":
            continue
        f32 = os.path.join(o, f"{key}_fp32.npy")
        b16_path = os.path.join(o, f"{key}_bf16.npy")
        if not (os.path.exists(f32) and os.path.exists(b16_path)):
            rep[f"{key}_missing_oracle"] = True
            ok = False
            continue
        ref, b16 = np.load(f32), np.load(b16_path)
        band = _rel(ref, b16)
        m = {
            "rel_dev": round(_rel(ref, dev_arr), 5),
            "cpu_bf16_band": round(band, 5),
            "bar": round(K_BAND * band + FLOOR, 5),
            "cos_dev": round(_cos(ref, dev_arr), 5),
        }
        m["pass"] = m["rel_dev"] <= m["bar"]
        ok = ok and m["pass"]
        rep[key] = m
    h, w = dev["req"]["height"] // 32 * 32, dev["req"]["width"] // 32 * 32
    img_dev = np.load(os.path.join(d, "image_device.npy"))
    img_host = _host_decode(a.model, lat0, h, w)  # same device latents, host fp32 VAE, untiled
    np.save(os.path.join(d, "image_hostvae.npy"), img_host)
    _save_png(torch.from_numpy(img_host), os.path.join(d, "image_hostvae.png"))
    rep["device_vae_vs_host_vae_psnr_db"] = round(_psnr(img_host, img_dev), 2)
    lat_ref = os.path.join(o, "latent_fp32.npy")
    if os.path.exists(lat_ref):
        golden = _host_decode(a.model, np.load(lat_ref), h, w)
        _save_png(torch.from_numpy(golden), os.path.join(d, "image_golden_fp32.png"))
        rep["device_image_vs_golden_psnr_db"] = round(_psnr(golden, img_dev), 2)
        rep["hostvae_image_vs_golden_psnr_db"] = round(_psnr(golden, img_host), 2)
    rep["pass"] = bool(ok)
    with open(os.path.join(d, "gate.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("GATE " + json.dumps(rep), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oracle", "device", "compare"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"], help="oracle phases to run")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--device-dir", default=None)
    a = ap.parse_args()
    {"oracle": run_oracle, "device": run_device, "compare": run_compare}[a.mode](a)


if __name__ == "__main__":
    main()
