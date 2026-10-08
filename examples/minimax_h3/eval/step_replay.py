"""Step replay: CPU single forwards on the exact inputs a device run captured, at full size.

A device request run with ``MINIMAX_H3_CAPTURE=<cap.pt>`` (and ``MINIMAX_H3_CAPTURE_STEPS=0,25,49``) records, for
each chosen step, the host latents fed to the DiT, its timesteps and the device's velocities. This script runs
diffusers' ``MiniMaxH3Transformer3DModel`` (vendored) on the same inputs on the host CPU, in fp32 (the reference)
and in bf16 (the floor), then compares:

    python examples/minimax_h3/eval/step_replay.py run --model-path <ckpt> --capture cap.pt --dtype fp32 --out f32.pt
    python examples/minimax_h3/eval/step_replay.py run --model-path <ckpt> --capture cap.pt --dtype bf16 --out b16.pt
    python examples/minimax_h3/eval/step_replay.py compare --capture cap.pt --fp32 f32.pt --bf16 b16.pt [--out r.json]

Bar per step and modality: device rel-L2 vs fp32 <= 2 x the CPU-bf16 rel-L2 + 0.5%. Run on the host CPU
(``PJRT_DEVICE=CPU VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_LIBTORCH_NEURONX_LITE=0 NEURON_RT_VISIBLE_CORES=``).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def _rc(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    return {
        "rel_l2": float((a - b).norm() / b.norm()),
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def run(a) -> None:
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    cap = torch.load(a.capture, map_location="cpu", weights_only=False)
    embeds = cap["prompt_embeds"]
    h, w, f = cap["geom"]
    layout = build_layout(int(embeds.shape[1]), h, w, f)
    t0 = time.time()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        a.model_path, subfolder="transformer", torch_dtype=dtype
    ).eval()
    if dtype == torch.float32:
        model = model.float()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))  # the loader may reset it
    t_load = time.time() - t0
    steps = [s for s in cap["steps"] if a.steps is None or s["step"] in a.steps]
    out = {"dtype": a.dtype, "load_s": t_load, "steps": []}
    with torch.no_grad():
        for s in steps:
            t1 = time.time()
            ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                layout.video_indices,
                layout.audio_indices,
                0,
                0,
                layout.num_text_tokens,
                s["t_video"],
                s["t_audio"],
                s["t_video"],
                1.0,
            )
            v, au = model(
                s["video_rows"][None].to(dtype),
                s["audio_rows"][None].to(dtype),
                embeds.to(dtype),
                ts,
                ts_idx,
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                return_dict=False,
            )
            out["steps"].append(
                {
                    "step": s["step"],
                    "vel_video": v[0].float(),
                    "vel_audio": au[0].float(),
                    "seconds": time.time() - t1,
                }
            )
            print(f"step {s['step']}: {time.time() - t1:.0f}s", flush=True)
            torch.save(out, a.out)  # partial results survive a kill
    print(json.dumps({"out": a.out, "load_s": t_load, "steps": [s["step"] for s in out["steps"]]}))


def compare(a) -> dict:
    cap = torch.load(a.capture, map_location="cpu", weights_only=False)
    f32 = {
        s["step"]: s for s in torch.load(a.fp32, map_location="cpu", weights_only=False)["steps"]
    }
    b16 = {
        s["step"]: s for s in torch.load(a.bf16, map_location="cpu", weights_only=False)["steps"]
    }
    rows, ok = [], True
    for s in cap["steps"]:
        i = s["step"]
        if i not in f32 or i not in b16:
            continue
        for mod in ("video", "audio"):
            k = f"vel_{mod}"
            dev, floor = _rc(s[k], f32[i][k]), _rc(b16[i][k], f32[i][k])
            bar = 2 * floor["rel_l2"] + 0.005
            passed = dev["rel_l2"] <= bar
            ok &= passed
            rows.append(
                {
                    "step": i,
                    "modality": mod,
                    "device": dev,
                    "cpu_bf16": floor,
                    "bar": bar,
                    "pass": passed,
                }
            )
    rep = {
        "rule": "device rel-L2 vs CPU fp32 <= 2 x CPU-bf16 rel-L2 + 0.5% (per step, per modality)",
        "rows": rows,
        "pass": ok and bool(rows),
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-path", required=True)
    r.add_argument("--capture", required=True)
    r.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    r.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")], default=None)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--capture", required=True)
    c.add_argument("--fp32", required=True)
    c.add_argument("--bf16", required=True)
    c.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        rep = compare(a)
        raise SystemExit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
