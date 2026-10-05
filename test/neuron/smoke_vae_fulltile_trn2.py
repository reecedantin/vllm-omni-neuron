# SPDX-License-Identifier: Apache-2.0
"""Real-VAE patch-parallel decode on trn2, frame by frame against an untiled host fp32 decode.

The Wan2.2 TI2V VAE (Cosmos3 / Wan2.2-5B, ``patch_size`` 2) is decoded through
``DistributedAutoencoderKLWan.tiled_decode`` -- the device gather path of a ``vae_patch_parallel_size``
stage -- on ``--world`` ranks, one NeuronCore each, behind the same ``NeuronEdgeVae`` facade the
Cosmos3 pipeline uses. Rank 0 compares every output frame with diffusers' untiled fp32 CPU decode
of the same (bf16-rounded) latent: PSNR and max |d| of the whole frame, the interior and the
right / bottom strips where the last tiles land.

Cases (``--cases``, comma list):

* ``thin``    -- 256/256/224/192 px tiles on a 30x52 latent (A1's setting): full-size tiles, 2x4.
* ``thin_old``-- the same setting through the previous split (diffusers grid: 3x5 with tiles of 2
  latent rows / 4 columns at the edges), to show the before state on the same cores.
* ``default`` -- the Cosmos3 default 480/480/480/416 (1x2).
* ``padded``  -- 256/256/192/160 (3x5 = 15 tiles over 8 ranks: padded slots) with a small gather
  budget so the blend runs over several plane chunks.

``--repeats`` back-to-back requests per case; the parent samples neuron-monitor (the HBM sampler of
``smoke_vae_gather_trn2``) and reports the settled HBM after every request. ``--cpu``: dry-run on
gloo with eager fp32 decoders (no NeuronCore).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_vae_gather_trn2 as g  # noqa: E402  (rank init, cores, HBM sampler, CPU barrier)

CASES = {
    "thin": dict(tiles=(256, 256, 224, 192), split="full"),
    "thin_old": dict(tiles=(256, 256, 224, 192), split="old"),
    "default": dict(tiles=(480, 480, 480, 416), split="full"),
    "padded": dict(tiles=(256, 256, 192, 160), split="full", gather_mb=16),
}
STRIP_PX = 64  # right / bottom strip width: the last tile column (4 latent cols) / row (2 rows) +


def _old_split(vae, z):
    """The split as it was before full-size tiles: diffusers' range(0, H, stride) grid with thin
    last tiles; no starts in the spec, so the blend spaces tiles evenly (the old behaviour)."""
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
        GridSpec,
        TileTask,
    )

    _, _, num_frames, height, width = z.shape
    r, p = vae.spatial_compression_ratio, vae.config.patch_size
    th, tw = vae.tile_sample_min_height // r, vae.tile_sample_min_width // r
    sh, sw = vae.tile_sample_stride_height // r, vae.tile_sample_stride_width // r
    tasks = []
    for i in range(0, height, sh):
        for j in range(0, width, sw):
            hi, wj = min(height, i + th), min(width, j + tw)
            tile = torch.index_select(z, 3, torch.arange(i, hi, device=z.device))
            tile = torch.index_select(tile, 4, torch.arange(j, wj, device=z.device))
            frames = [
                torch.index_select(tile, 2, torch.tensor([k], device=z.device))
                for k in range(num_frames)
            ]
            tasks.append(
                TileTask(len(tasks), (i // sh, j // sw), frames, workload=(hi - i) * (wj - j))
            )
    ssh, ssw = vae.tile_sample_stride_height // p, vae.tile_sample_stride_width // p
    spec = {
        "sample_height": height * r // p,
        "sample_width": width * r // p,
        "blend_height": vae.tile_sample_min_height // p - ssh,
        "blend_width": vae.tile_sample_min_width // p - ssw,
        "tile_sample_stride_height": ssh,
        "tile_sample_stride_width": ssw,
    }
    grid = (tasks[-1].grid_coord[0] + 1, tasks[-1].grid_coord[1] + 1)
    return tasks, GridSpec(
        split_dims=(3, 4), grid_shape=grid, tile_spec=spec, output_dtype=vae.dtype
    )


def _frame_metrics(got, ref, strip=STRIP_PX):
    """Per output frame: PSNR (range [-1, 1]) and max |d| of the frame, interior, right/bottom
    strips. ``got``/``ref``: [3, F, H, W] float."""
    d = got.float().clamp(-1, 1) - ref.float().clamp(-1, 1)
    frames = []
    for f in range(d.shape[1]):
        df = d[:, f]

        def psnr(x):
            mse = x.pow(2).mean().item()
            return round(10 * math.log10(4.0 / max(mse, 1e-12)), 2)

        frames.append(
            {
                "psnr": psnr(df),
                "psnr_inner": psnr(df[:, :-strip, :-strip]),
                "psnr_right": psnr(df[:, :, -strip:]),
                "psnr_bottom": psnr(df[:, -strip:, :]),
                "max_abs": round(df.abs().max().item(), 4),
                "max_abs_right": round(df[:, :, -strip:].abs().max().item(), 4),
                "max_abs_bottom": round(df[:, -strip:, :].abs().max().item(), 4),
            }
        )
    keys = ("psnr", "psnr_inner", "psnr_right", "psnr_bottom")
    worst = {k: min(fr[k] for fr in frames) for k in keys}
    worst_frame = {k: min(range(len(frames)), key=lambda i, k=k: frames[i][k]) for k in keys}
    return {
        "frames": len(frames),
        "min": worst,
        "min_at_frame": worst_frame,
        "mean_psnr": round(sum(fr["psnr"] for fr in frames) / len(frames), 2),
        "max_abs": max(fr["max_abs"] for fr in frames),
        "max_abs_right": max(fr["max_abs_right"] for fr in frames),
        "max_abs_bottom": max(fr["max_abs_bottom"] for fr in frames),
        "per_frame": frames,
    }


def _build_vae(weights, group, world, device, tiles, split, cpu):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import (
        NeuronEdgeVae,
    )

    dtype = torch.float32 if cpu else torch.bfloat16
    vae = DistributedAutoencoderKLWan.from_pretrained(weights, subfolder="vae", torch_dtype=dtype)
    vae = vae.eval()
    # init_distributed() needs the engine's DiT group; bind the executor to this smoke's group the
    # way it would (group, world, rank), then the same set_parallel_size the pipelines call.
    import torch.distributed as dist

    vae.distributed_executor = g._executor(group, world, dist.get_rank(group))
    vae.set_parallel_size(world)
    vae.use_tiling = True
    (
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    ) = tiles
    if split == "old":
        vae.tile_split = lambda z, _v=vae: _old_split(_v, z)
    edge = NeuronEdgeVae(vae)
    if cpu:  # dry run: payload graphs eager, gather over gloo
        vae.distributed_executor._compile_device_graph = lambda name, key, fn: fn
    else:
        edge.to(device)
        edge.compile(get_compile_backend_name(), None, compile_encoder=False)
    return edge


def _rank_main(rank, world, cores, port, args):
    out = {"rank": rank, "ok": False, "checks": []}
    path = os.path.join(args.out, f"fulltile_w{world}_rank{rank}.json")
    try:
        device, group = g._init_rank(rank, world, cores[rank], port)
        ref = torch.load(args.ref, weights_only=True) if rank == 0 else None
        z = torch.load(args.latent, weights_only=True)
        for name in args.cases.split(","):
            case = CASES[name]
            if case.get("gather_mb"):
                os.environ["VLLM_OMNI_NEURON_VAE_GATHER_MB"] = str(case["gather_mb"])
            else:
                os.environ.pop("VLLM_OMNI_NEURON_VAE_GATHER_MB", None)
            t0 = time.time()
            vae = _build_vae(
                args.weights, group, world, device, case["tiles"], case["split"], args.cpu
            )
            build_s = time.time() - t0
            g._barrier(group)
            if not args.cpu:
                time.sleep(g.HBM_SETTLE_S)
            windows = {"before": time.time(), "requests": []}
            times, metrics, grid = [], [], None
            for _ in range(args.repeats if name != "thin_old" else min(args.repeats, 2)):
                g._barrier(group)
                t1 = time.time()
                with torch.no_grad():
                    res = vae.decode(z, return_dict=False)[0]
                t2 = time.time()
                times.append(round(t2 - t1, 3))
                if rank == 0:
                    m = _frame_metrics(res[0], ref[0])
                    metrics.append(m)
                    if len(metrics) == 1:
                        torch.save(
                            res[:, :, :: max(1, res.shape[2] // 8)].to(torch.bfloat16),
                            os.path.join(args.out, f"frames_{name}.pt"),
                        )
                del res
                g._barrier(group)
                if not args.cpu:
                    time.sleep(g.HBM_SETTLE_S)
                windows["requests"].append({"t_start": t1, "t_end": t2, "t_settled": time.time()})
            if grid is None:
                _, spec = vae.vae.tile_split(z[:, :, :1])
                grid = list(spec.grid_shape)
            check = {
                "tag": name,
                "tiles_px": list(case["tiles"]),
                "split": case["split"],
                "grid": grid,
                "gather_mb": case.get("gather_mb"),
                "build_compile_s": round(build_s, 1),
                "times_s": times,
                "hbm_windows": windows,
            }
            if rank == 0:
                m0 = metrics[0]
                check |= {
                    "frames": m0["frames"],
                    "min_psnr": m0["min"],
                    "min_at_frame": m0["min_at_frame"],
                    "mean_psnr": m0["mean_psnr"],
                    "max_abs": m0["max_abs"],
                    "max_abs_right": m0["max_abs_right"],
                    "max_abs_bottom": m0["max_abs_bottom"],
                    "per_frame": m0["per_frame"],
                    "requests_min_psnr": [m["min"]["psnr"] for m in metrics],
                }
                if case["split"] == "full":
                    # strips no worse than the interior (2 dB slack) and no catastrophic frame
                    edge_ok = min(m0["min"]["psnr_right"], m0["min"]["psnr_bottom"]) >= min(
                        m0["min"]["psnr_inner"] - 2.0, args.min_psnr
                    )
                    check["passed"] = bool(edge_ok and m0["min"]["psnr"] >= args.min_psnr)
            out["checks"].append(check)
            print(
                f"[rank{rank}] {name}: grid={grid} times={times} "
                + (f"min_psnr={check.get('min_psnr')}" if rank == 0 else ""),
                flush=True,
            )
            del vae
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()
        print(out["trace"], flush=True)
    with open(path, "w") as f:
        json.dump(out, f, default=str)


def _make_reference(args):
    """Latent (bf16-rounded, seeded) + diffusers untiled fp32 CPU decode, cached on disk."""
    if os.path.exists(args.latent) and os.path.exists(args.ref):
        return
    from diffusers import AutoencoderKLWan

    torch.manual_seed(args.seed)
    h, w = args.latent_hw
    # Smooth-ish latent: random at 1/2 resolution, upsampled, so the decode looks like an image
    # (pure noise latents decode to saturated texture where clamp hides errors).
    z = torch.randn(1, 48, args.latent_t, (h + 1) // 2, (w + 1) // 2)
    z = torch.nn.functional.interpolate(z, size=(args.latent_t, h, w), mode="trilinear")
    z = (z * 1.5).to(torch.bfloat16)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "48")))
    ref_vae = AutoencoderKLWan.from_pretrained(
        args.weights, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    t0 = time.time()
    with torch.no_grad():
        ref = ref_vae.decode(z.float()).sample  # use_tiling off: untiled
    print(f"[ref] untiled fp32 decode {tuple(ref.shape)} in {time.time() - t0:.1f}s", flush=True)
    torch.save(z, args.latent)
    torch.save(ref.to(torch.float32), args.ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--cores", default=None)
    ap.add_argument("--out", default=os.environ.get("SMOKE_OUT", "."))
    ap.add_argument(
        "--weights",
        default=os.environ.get("WAN_VAE_WEIGHTS"),
        required="WAN_VAE_WEIGHTS" not in os.environ,
        help="model dir with a Wan2.2 TI2V vae/ subfolder (Cosmos3 / Wan2.2-5B)",
    )
    ap.add_argument("--cases", default="thin,default,padded,thin_old")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latent-t", type=int, default=9, help="latent frames (9 -> 33 output frames)")
    ap.add_argument("--latent-hw", type=int, nargs=2, default=(30, 52))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-dir", default=os.environ.get("SMOKE_REF_DIR"))
    ap.add_argument("--min-psnr", type=float, default=30.0)
    ap.add_argument("--ref-only", action="store_true", help="only build the CPU reference")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-hbm-monitor", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref_dir = args.ref_dir or args.out
    os.makedirs(ref_dir, exist_ok=True)
    tag = f"t{args.latent_t}_{args.latent_hw[0]}x{args.latent_hw[1]}_s{args.seed}"
    args.latent = os.path.join(ref_dir, f"latent_{tag}.pt")
    args.ref = os.path.join(ref_dir, f"ref_untiled_fp32_{tag}.pt")
    _make_reference(args)
    if args.ref_only:
        return 0

    world = args.world
    cores = [-1] * world if args.cpu else g._cores(args.cores)[:world]
    if len(cores) < world:
        sys.exit(f"need {world} cores, have {cores}")
    port = g._free_port()
    monitor = None
    if not args.cpu and not args.no_hbm_monitor:
        monitor = g.HbmMonitor(cores, args.out).start()
        time.sleep(1.5)
    t0 = time.time()
    try:
        mp.spawn(_rank_main, args=(world, cores, port, args), nprocs=world, join=True)
    finally:
        wall = time.time() - t0
        if monitor is not None:
            monitor.stop()
    ranks = []
    for r in range(world):
        p = os.path.join(args.out, f"fulltile_w{world}_rank{r}.json")
        ranks.append(json.load(open(p)) if os.path.exists(p) else {"rank": r, "ok": False})
    checks = ranks[0].get("checks", [])
    for c in checks:
        if monitor is not None and c.get("hbm_windows"):
            hbm = monitor.resolve(c["hbm_windows"])
            c["hbm"] = {k: hbm.get(k) for k in ("before_gib", "growth_max_gib", "sample_hz")} | {
                "requests_present_gib": [r["present_gib"] for r in hbm.get("requests", [])],
                "requests_peak_gib": [r["peak_gib"] for r in hbm.get("requests", [])],
            }
    gated = [c for c in checks if "passed" in c]
    summary = {
        "world": world,
        "wall_s": round(wall, 1),
        "passed": all(r.get("ok") for r in ranks)
        and bool(gated)
        and all(c["passed"] for c in gated),
        "bad_ranks": [r["rank"] for r in ranks if not r.get("ok")],
        "rank0_error": ranks[0].get("error"),
        "checks": [
            {k: v for k, v in c.items() if k not in ("per_frame", "hbm_windows")} for c in checks
        ],
    }
    with open(os.path.join(args.out, f"fulltile_w{world}.json"), "w") as f:
        json.dump({"summary": summary, "checks": checks, "ranks": ranks}, f, indent=1, default=str)
    print("SMOKE_SUMMARY " + json.dumps(summary, default=str))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
