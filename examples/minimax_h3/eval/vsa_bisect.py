"""CPU bisection of the VSA-H3 device path vs the token-mask reference on the real FastH3 8-Step-V2 weights.

    python examples/minimax_h3/eval/vsa_bisect.py --model-path <8-Step-V2> --prompt-embeds e.pt --ref ref_fp32.pt \
        --dtype bf16|fp32 [--fine-fp32] --out bisect.json

One step-0 forward of the plugin DiT (TP=1, the same code the device compiles, on CPU) from the reference's input
noise. Every block's VSA call is wrapped: it also runs ``vsa_attention_reference`` on the SAME q/k/v/gate (fp32),
and records per layer (a) the local output error of the device formulation, (b) the fraction of selected video
tiles that differ between the device-precision selection and an fp32 selection. The step-0 velocity is compared
with the fp32 reference run's. Separates "formulation bug" (large local error / flips at fp32) from
"precision chaos" (small local error, flips that compound across layers).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt-embeds", required=True)
    ap.add_argument(
        "--ref", required=True, help="reference_cpu.py fp32 output (step-0 velocity + seed + geom)"
    )
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument(
        "--fine-fp32",
        action="store_true",
        help="fine-stage scores in fp32 (MINIMAX_H3_VSA_FINE_FP32=1)",
    )
    ap.add_argument(
        "--save-masks",
        default=None,
        help="save per-layer selection masks (run this with --dtype fp32)",
    )
    ap.add_argument(
        "--cmp-masks", default=None, help="compare per-layer selections against saved fp32 masks"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fine_fp32:
        os.environ["MINIMAX_H3_VSA_FINE_FP32"] = "1"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "48")))

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
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

    import vllm_omni_neuron.diffusion.models.minimax_h3.transformer as T
    import vllm_omni_neuron.diffusion.models.minimax_h3.vsa as V
    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig, vsa_sparsity
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise

    ref = torch.load(a.ref, map_location="cpu", weights_only=False)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    tdir = os.path.join(a.model_path, "transformer")
    cfg = MiniMaxH3DiTConfig.from_dir(tdir)
    t0 = time.time()
    dit = T.NeuronMiniMaxH3Transformer(cfg, dtype=dtype, vsa_sparsity=vsa_sparsity(a.model_path))
    dit.load_weights(tdir, "cpu")
    t_load = time.time() - t0

    embeds = torch.load(a.prompt_embeds, map_location="cpu", weights_only=False)["prompt_embeds"]
    h, w, f = ref["geom"]
    layout = build_layout(int(embeds.shape[1]), h, w, f)
    video_rows, audio_rows = draw_noise(layout, torch.Generator().manual_seed(int(ref["seed"])))
    grid = (layout.num_latent_frames, layout.latent_height // 2, layout.latent_width // 2)
    dit.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows, grid)
    cos, sin = layout.rotary(cfg.rope_freq_dim, cfg.rope_theta)
    st = ref["steps"][0]
    ts = torch.tensor([st["t_video"], st["t_audio"]], dtype=torch.float32)

    per_layer = []
    saved_masks = []
    cmp_masks = torch.load(a.cmp_masks) if a.cmp_masks else None
    orig = T.vsa_attention

    def wrapped(q, k, v, gate, g):
        out = orig(q, k, v, gate, g)
        with torch.no_grad():
            want = V.vsa_attention_reference(q, k, v, gate, g)  # same inputs, fp32 math
            local = ((out.float() - want).norm() / want.norm()).item()
            vid = slice(g.n_prefix, None)
            m = V._tile_mask(
                V._tile_scores(V._to_slots(q, g), V._to_slots(k, g), g), g, use_topk=True
            )[:, vid, vid]
            if a.save_masks:
                saved_masks.append(m.clone())
            rec = {"local_rel": local}
            if cmp_masks is not None:
                m0 = cmp_masks[len(per_layer)]
                rec["tile_flip_frac_vs_fp32"] = (m != m0).float().sum().item() / max(
                    1.0, 2 * m0.float().sum().item()
                )
        per_layer.append(rec)
        return out

    T.vsa_attention = wrapped
    t1 = time.time()
    with torch.no_grad():
        vel_v, vel_a = dit(embeds.to(dtype), audio_rows[None], video_rows[None], ts, cos, sin)
    T.vsa_attention = orig

    def rc(x, y):
        x, y = x.float().flatten(), y.float().flatten()
        return {
            "rel_l2": ((x - y).norm() / y.norm()).item(),
            "cos": torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
        }

    res = {
        "dtype": a.dtype,
        "fine_fp32": a.fine_fp32,
        "load_s": t_load,
        "forward_s": time.time() - t1,
        "step0_video": rc(vel_v[0], st["vel_video"]),
        "step0_audio": rc(vel_a[0], st["vel_audio"]),
        "local_rel_max": max(p["local_rel"] for p in per_layer),
        "local_rel_mean": sum(p["local_rel"] for p in per_layer) / len(per_layer),
        "flip_frac_vs_fp32": [round(p.get("tile_flip_frac_vs_fp32", -1), 4) for p in per_layer],
        "per_layer": per_layer,
    }
    if a.save_masks:
        torch.save(saved_masks, a.save_masks)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_layer"}))


if __name__ == "__main__":
    main()
