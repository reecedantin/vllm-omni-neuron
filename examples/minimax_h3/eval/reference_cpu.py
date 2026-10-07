"""CPU fp32 reference for a FastH3 / MiniMax-H3 t2va denoise (the parity oracle), dense or VSA-H3.

    python examples/minimax_h3/eval/reference_cpu.py --model-path <ckpt> --prompt-embeds embeds.pt \
        --height 256 --width 256 --num-frames 124 --seed 0 --out ref.pt [--dtype fp32|bf16] [--max-steps N]

Drives diffusers' ``MiniMaxH3Transformer3DModel`` (vendored) exactly like diffusers' modular pipeline: same layout,
noise draw, row timesteps and schedulers (``--schedule``, the same rule as the pipeline). For a VSA checkpoint
(``fastvideo_inference.json`` says ``VIDEO_SPARSE_ATTN_H3``) every block's attention is swapped for the token-mask
VSA reference with the checkpoint's ``to_gate_compress`` weights. Saves the final ``video_rows`` / ``audio_rows``
(and every step's velocities) for ``gate_compare.py``. Run it with ``fleet/bin/cpumode.sh`` sourced.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt-embeds", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument(
        "--steps", type=int, default=None, help="grid points (default: the checkpoint's)"
    )
    ap.add_argument(
        "--max-steps", type=int, default=None, help="stop after N forwards (single-step tier: 1)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument(
        "--schedule",
        choices=("auto", "contract", "linspace"),
        default="auto",
        help="step grid, same rule as the pipeline's model_config.schedule (auto: linspace for a contract "
        "without scheduler shifts, i.e. the Preview-v1 4-step export; the trained ladder otherwise)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))

    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.mp_before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import (
        inference_contract,
        step_positions,
        vsa_sparsity,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import (
        build_layout,
        draw_noise,
        load_schedulers,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vsa import (
        VSAGeometry,
        install_reference_processors,
    )

    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    steps = a.steps or int(inference_contract(a.model_path).get("num_inference_steps", 5))
    embeds = torch.load(a.prompt_embeds, map_location="cpu", weights_only=False)["prompt_embeds"]
    t0 = time.time()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        a.model_path, subfolder="transformer", torch_dtype=dtype
    ).eval()
    if dtype == torch.float32:
        model = model.float()
    sparsity = vsa_sparsity(a.model_path)
    layout = build_layout(int(embeds.shape[1]), a.height, a.width, a.num_frames)
    holder = None
    if sparsity is not None:
        tdir = os.path.join(a.model_path, "transformer")
        wm = {}  # key -> file, from the tensors actually on disk (an index file may be stale)
        for fn in sorted(glob.glob(os.path.join(tdir, "*.safetensors"))):
            with safe_open(fn, "pt") as f:
                wm.update({k: os.path.basename(fn) for k in f.keys()})
        gates = []
        for i in range(len(model.transformer_blocks)):
            key = f"transformer_blocks.{i}.attn.to_gate_compress.weight"
            with safe_open(os.path.join(tdir, wm[key]), "pt") as f:
                gates.append(f.get_tensor(key).to(dtype))
        holder = install_reference_processors(model, gates)
        grid = (layout.num_latent_frames, layout.latent_height // 2, layout.latent_width // 2)
        holder["geom"] = VSAGeometry.build(
            (layout.num_text_tokens, layout.num_audio_rows), grid, sparsity
        )
    t_load = time.time() - t0

    video_rows, audio_rows = draw_noise(layout, torch.Generator().manual_seed(a.seed))
    schedule = a.schedule
    if schedule == "auto":
        schedule = (
            "contract"
            if "video_scheduler_shift" in inference_contract(a.model_path)
            else "linspace"
        )
    positions = step_positions(a.model_path, steps) if schedule == "contract" else None
    sv, sa = load_schedulers(a.model_path, steps, positions)
    text = embeds.to(dtype)
    vel = []
    t1 = time.time()
    with torch.no_grad():
        for i in range(len(sv.timesteps)):
            if a.max_steps is not None and i >= a.max_steps:
                break
            tv, ta = float(sv.timesteps[i]), float(sa.timesteps[i])
            ts, ts_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                layout.video_indices,
                layout.audio_indices,
                0,
                0,
                layout.num_text_tokens,
                tv,
                ta,
                tv,
                1.0,
            )
            v, au = model(
                video_rows[None].to(dtype),
                audio_rows[None].to(dtype),
                text,
                ts,
                ts_idx,
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                return_dict=False,
            )
            vel.append(
                {
                    "t_video": tv,
                    "t_audio": ta,
                    "vel_video": v[0].float(),
                    "vel_audio": au[0].float(),
                }
            )
            video_rows = sv.step(v[0].float(), sv.timesteps[i], video_rows, return_dict=False)[
                0
            ].float()
            audio_rows = sa.step(au[0].float(), sa.timesteps[i], audio_rows, return_dict=False)[
                0
            ].float()
            print(f"step {i + 1}: t_video={tv:.4f} ({time.time() - t1:.0f}s)", flush=True)
    torch.save(
        {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "steps": vel,
            "seed": a.seed,
            "dtype": a.dtype,
            "num_inference_steps": steps,
            "schedule": schedule,
            "vsa_sparsity": sparsity,
            "geom": (a.height, a.width, a.num_frames),
            "prompt_embeds": embeds,
            "load_s": t_load,
            "denoise_s": time.time() - t1,
        },
        a.out,
    )
    print(
        json.dumps(
            {"out": a.out, "load_s": t_load, "denoise_s": time.time() - t1, "forwards": len(vel)}
        )
    )


if __name__ == "__main__":
    main()
