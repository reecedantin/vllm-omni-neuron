# SPDX-License-Identifier: Apache-2.0
"""Trn2 (NeuronCore-v3) smoke test for the shared platform code.

Self-contained device check, one process, one core, small graphs (minutes, a few GB host RAM):

1. NeuronCore-generation detection: platform target, driver sysfs, and the NKI gates on a device tensor.
2. Wan VAE attention block (the shared ``WanAttentionBlock``) compiled on ``neuron:0`` at the real
   Wan2.1 (dim 384) and Wan2.2-TI2V-5B decoder (dim 1024) mid-block widths, vs CPU fp32 SDPA.
   Default run takes the NKI ``attention_cte`` path; ``--force-gen 2`` takes the NC-v2 torch path.
3. A shrunk Wan2.2-TI2V-5B VAE (patchified; same block structure) compiled on ``neuron:0``:
   encode (patchify inside the encoder graph) and decode, vs diffusers ``AutoencoderKLWan`` CPU fp32.
   Preceded by standalone compiles of ``patchify``/``unpatchify`` and ``AvgDown3D``/``DupUp3D``
   (isolates each op).
4. ``BlockGraphRunner`` (shared N-block graph splitting) with per-layer modulation tables: 7 blocks
   in chunks of 3 must compile exactly 2 graphs; parity vs CPU fp32.
5. Shared decode-attention layer (Qwen3-shaped): prefill + 3 decode steps on device vs CPU fp32
   on the default torch path (one compiled graph for every decode position), the opt-in fused
   ``attention_block_tkg`` kernel path reported alongside; ``decode_attention_kernel_diag`` bisects
   the kernel path (MHA / GQA with identical KV heads / GQA).
6. Halo-tiled neighborhood (NATTEN/Swin) attention at FLUX-3-Action's single-frame VAE grid
   (136x184, window 5x5) on device vs CPU fp32 -- the grid whose gather/mask formulations all failed
   neuronx-cc -- with per-tile error maps, timing and graph-count gates.
7. Opt-in real-weight VAE checks (``SMOKE_VAE_WEIGHTS``): encode/decode size sweeps against the
   single-graph compile threshold, tiled encode/decode timing and parity at 480/704 px.

Every check also dry-runs on CPU (``SMOKE_DEVICE=cpu``), which exercises the harness, not the
compiler. ``--only <check>`` runs one check per process; ``--out`` (or ``SMOKE_OUT``) is where the
JSON results and any saved tensors go.

Prints one JSON line (``SMOKE_SUMMARY {...}``) and exits nonzero if any check fails.

    python test/neuron/smoke_platform_trn2.py --out "$OUT"                # NC-v3 / NKI paths
    python test/neuron/smoke_platform_trn2.py --out "$OUT" --force-gen 2  # NC-v2 torch paths on trn2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]

TINY_TI2V_5B_VAE = dict(
    base_dim=16,
    decoder_base_dim=32,
    z_dim=8,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    latents_mean=[0.1 * i for i in range(8)],
    latents_std=[1.0 + 0.1 * i for i in range(8)],
    is_residual=True,
    in_channels=12,
    out_channels=12,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
)

# Cosmos3-Nano / Wan2.2-TI2V-5B's REAL (not shrunk) channel width: encoder mid-block attention at
# 640 channels (base_dim 160 * dim_mult[-1] 4). The Cosmos3 port reported
# a NCC_IDDT901 persisting at this width, one pixel frame, after the patchify/AvgDown3D/DupUp3D
# fixes (which were only exercised at TINY_TI2V_5B_VAE's shrunk base_dim=16 / width-64 mid-block).
REAL_WIDTH_TI2V_5B_VAE = dict(
    base_dim=160,
    decoder_base_dim=256,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    latents_mean=[0.1 * i for i in range(48)],
    latents_std=[1.0 + 0.1 * i for i in range(48)],
    is_residual=True,
    in_channels=12,
    out_channels=12,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
)


def _device():
    """``SMOKE_DEVICE=cpu`` dry-runs every check's harness on CPU (eager backend) before booking a core.

    ``torch.device("neuron:0")`` is only parseable once the plugin has registered the ``neuron``
    backend (round 16: ``decode_attention_kernel_diag`` called this before its first plugin import
    and died with ``device type at start of device string: neuron``), so import it here first."""
    import torch

    name = os.environ.get("SMOKE_DEVICE", "neuron:0")
    if not name.startswith("cpu"):
        import vllm_omni_neuron  # noqa: F401  (its bootstrap / vllm_neuron import registers 'neuron')
    return torch.device(name)


def _backend():
    from vllm_neuron.envs import get_compile_backend_name

    if _device().type == "cpu":
        return _EagerWithOptions()
    return get_compile_backend_name()


class _EagerWithOptions:
    """Eager backend that accepts (and ignores) torch.compile ``options``, for the CPU dry run."""

    def __call__(self, gm, example_inputs, **_):
        return gm.forward


def _rel(a, b) -> float:
    # .cpu() BEFORE .float(): an eager dtype cast on a device tensor is unsupported on this backend
    # (`Expected self.dtype() == dst.dtype()`), while the same cast on the host copy is free.
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _compile_on(device, fn, name: str, compiler_args=None):
    """Compile ``fn`` for the real device, or return it UNCOMPILED for CPU inputs.

    Round 11's segfault (rc=139, faulthandler frame at the CPU reference call of the neighborhood
    stage diagnostics) was this harness compiling a CPU-input function with the Neuron backend: the
    ``neuron_native_lite`` executor received CPU tensors and died in ``nrt_tensor_get_size`` via
    ``NeuronTensorImpl::CreateSlice``. ``_backend()`` picks the backend from SMOKE_DEVICE, not from
    the tensors, so every CPU-side reference computation must go through this helper (or stay eager).

    Two more rules for anything compiled here, both from round 11: every graph INPUT must be a
    contiguous base device tensor (a ``.transpose()``/``[..., :n]`` view of a device tensor is
    refused with ``Detected non-contiguous slicing for requested Device Tensor``), and every graph
    OUTPUT should be a fresh tensor, not a view of an intermediate (end stage fns with a real op or
    ``.clone()``), so AOT autograd never has to regenerate a strided view eagerly on the device.
    """
    import torch

    if torch.device(device).type == "cpu":
        return fn
    return torch.compile(
        fn,
        backend=_backend(),
        fullgraph=True,
        dynamic=False,
        options={"model_name": name, "compiler_args": compiler_args or COMPILER_ARGS},
    )


def _psnr(a, b) -> float:
    import torch

    mse = (a.float().clamp(-1, 1) - b.float().clamp(-1, 1)).pow(2).mean().item()
    return 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def check_detection(expect_gen: int) -> dict:
    import torch

    from vllm_omni_neuron import nc_generation as ncg
    from vllm_omni_neuron.lite_compat import get_platform_target

    dev = _device()
    r = {
        "platform_target": str(get_platform_target()),
        "generation": ncg.neuron_core_generation(),
        "sysfs_generation": ncg.generation_from_sysfs(),
        "supports_nki": ncg.supports_nki(),
        "use_nki_kernels_device": ncg.use_nki_kernels(torch.zeros(1, device=dev)),
        "use_nki_kernels_cpu": ncg.use_nki_kernels(torch.zeros(1)),
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
    }
    want_nki = expect_gen >= 3
    r["ok"] = (
        r["generation"] == expect_gen
        and r["supports_nki"] is want_nki
        and r["use_nki_kernels_device"] is want_nki
        and r["use_nki_kernels_cpu"] is False
        and (expect_gen != 3 or r["sysfs_generation"] == 3)
    )
    return r


def _nki_path(dim: int) -> bool:
    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        _vae_attn_can_use_nki,
    )

    return _vae_attn_can_use_nki(dim)


def check_vae_attention(dim: int, hw, frames: int, tag: str) -> dict:
    import torch

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        WanAttentionBlock,
    )

    torch.manual_seed(0)
    blk = WanAttentionBlock(dim).eval()
    h, w = (hw, hw) if isinstance(hw, int) else hw
    x = torch.randn(1, dim, frames, h, w)
    with torch.no_grad():
        ref = blk(x)  # CPU fp32: the SDPA branch
    dev = _device()
    blk_d = WanAttentionBlock(dim).eval()
    blk_d.load_state_dict(blk.state_dict())
    blk_d = blk_d.to(torch.bfloat16).to(dev)
    fn = torch.compile(
        blk_d,
        backend=_backend(),
        fullgraph=True,
        dynamic=False,
        options={"model_name": f"smoke_vae_attn_{tag}", "compiler_args": COMPILER_ARGS},
    )
    xd = x.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        out = fn(xd).cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = fn(xd).cpu()
    warm = time.time() - t0
    # Three-way, on the attention contribution (the residual dominates the full output): device bf16
    # vs CPU fp32, judged against the pure-dtype error of a CPU bf16 run of the same block.
    x16 = x.to(torch.bfloat16)
    blk16 = WanAttentionBlock(dim).eval()
    blk16.load_state_dict(blk.state_dict())
    with torch.no_grad():
        ref16 = blk16.to(torch.bfloat16)(x16)
    rel = _rel(out.float() - x16.float(), ref - x)
    rel16 = _rel(ref16.float() - x16.float(), ref - x)
    r = {
        "dim": dim,
        "tokens": h * w,
        "path": "nki" if _nki_path(dim) else "torch",
        "frames": frames,
        "rel_err": rel,
        "cpu_bf16_rel_err": rel16,
        "full_out_rel_err": _rel(out, ref),
        "first_s": round(first, 1),
        "warm_s": round(warm, 4),
        "deterministic": bool(torch.equal(out, out2)),
    }
    r["ok"] = rel <= 2 * rel16 + 0.005 and r["deterministic"]
    return r


def check_patchify_compile() -> dict:
    """patchify/unpatchify alone, compiled on neuron:0: the direct regression check for the
    DramToDramTranspose (NCC_IDDT901) encoder-compile failure of the Wan2.2-5B VAE on Trn2 -- isolates the op from the rest of the VAE graph so a
    failure here points straight at the transpose chain, not at check_tiny_vae's much larger graph.
    """
    import torch

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        patchify,
        unpatchify,
    )

    torch.manual_seed(0)
    dev = _device()
    x = torch.randn(1, 3, 2, 64, 64)
    ref_y = patchify(x, 2)
    ref_z = unpatchify(ref_y, 2)
    opts_p = {
        "model_name": "smoke_patchify",
        "compiler_args": ["--model-type=unet-inference", "-O1"],
    }
    opts_u = {
        "model_name": "smoke_unpatchify",
        "compiler_args": ["--model-type=unet-inference", "-O1"],
    }
    patchify_c = torch.compile(
        patchify, backend=_backend(), fullgraph=True, dynamic=False, options=opts_p
    )
    unpatchify_c = torch.compile(
        unpatchify, backend=_backend(), fullgraph=True, dynamic=False, options=opts_u
    )
    xd = x.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        yd = patchify_c(xd, 2)
    patchify_s = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        zd = unpatchify_c(yd, 2)
    unpatchify_s = time.time() - t0
    r = {
        "y_shape": list(yd.shape),
        "z_shape": list(zd.shape),
        "patchify_rel_err": _rel(yd.cpu().float(), ref_y),
        "unpatchify_rel_err": _rel(zd.cpu().float(), ref_z),
        "patchify_first_s": round(patchify_s, 1),
        "unpatchify_first_s": round(unpatchify_s, 1),
    }
    r["ok"] = (
        tuple(yd.shape) == tuple(ref_y.shape)
        and tuple(zd.shape) == tuple(x.shape)
        and r["patchify_rel_err"] < 0.01  # bf16 roundtrip only; the op itself is exact
        and r["unpatchify_rel_err"] < 0.01
    )
    return r


def check_avg_down_up_3d_compile() -> dict:
    """AvgDown3D/DupUp3D alone, compiled on neuron:0: the second DramToDramTranspose (NCC_IDDT901)
    found inside the Wan VAE encoder after the patchify fix -- isolates these two ops, which also
    use a module-level forward monkeypatch rather than a module-level function override."""
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D, DupUp3D

    torch.manual_seed(0)
    dev = _device()
    down = AvgDown3D(16, 32, factor_t=2, factor_s=2)
    up = DupUp3D(32, 16, factor_t=2, factor_s=2)
    x = torch.randn(1, 16, 5, 8, 8)
    ref_y = down(x)
    y = torch.randn(1, 32, 3, 4, 4)
    ref_z = up(y)

    opts_d = {
        "model_name": "smoke_avg_down_3d",
        "compiler_args": ["--model-type=unet-inference", "-O1"],
    }
    opts_u = {
        "model_name": "smoke_dup_up_3d",
        "compiler_args": ["--model-type=unet-inference", "-O1"],
    }
    down_c = torch.compile(down, backend=_backend(), fullgraph=True, dynamic=False, options=opts_d)
    up_c = torch.compile(up, backend=_backend(), fullgraph=True, dynamic=False, options=opts_u)
    xd = x.to(torch.bfloat16).to(dev)
    yd = y.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        down_out = down_c(xd)
    down_s = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        up_out = up_c(yd)
    up_s = time.time() - t0
    r = {
        "down_shape": list(down_out.shape),
        "up_shape": list(up_out.shape),
        "down_rel_err": _rel(down_out.cpu().float(), ref_y),
        "up_rel_err": _rel(up_out.cpu().float(), ref_z),
        "down_first_s": round(down_s, 1),
        "up_first_s": round(up_s, 1),
    }
    r["ok"] = (
        tuple(down_out.shape) == tuple(ref_y.shape)
        and tuple(up_out.shape) == tuple(ref_z.shape)
        and r["down_rel_err"] < 0.01
        and r["up_rel_err"] < 0.01
    )
    return r


def _vae_weights_source() -> str:
    """``SMOKE_VAE_WEIGHTS``: a diffusers VAE directory (e.g. the ``vae`` folder of a
    Wan2.2-TI2V-5B checkout) for every real-width VAE check, else ``random`` -- random weights at
    the real width give the production graph sizes but NOT representative tiling numbers (a conv
    encoder with random weights has no learned locality: tiled-vs-untiled latent rel was 0.20-0.36
    with random weights vs 0.03-0.13 with the real weights on CPU)."""
    return os.environ.get("SMOKE_VAE_WEIGHTS", "random")


def _real_width_vae_pair():
    """(diffusers reference VAE, NeuronAutoencoderKLWan with the same weights), both eval, on CPU."""
    import torch
    from diffusers import AutoencoderKLWan

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    src = _vae_weights_source()
    torch.manual_seed(0)
    if src == "random":
        ref = AutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
        neu = NeuronAutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
        neu.load_state_dict(ref.state_dict(), strict=True)
    else:
        ref = AutoencoderKLWan.from_pretrained(src).eval()
        neu = NeuronAutoencoderKLWan.from_pretrained(src).eval()
    return ref, neu


def _set_tiling(vae, tile_px: int, overlap_px: int) -> None:
    vae.use_tiling = True
    vae.tile_sample_min_height = vae.tile_sample_min_width = tile_px
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = tile_px - overlap_px


def check_real_width_vae_encode() -> dict:
    """A full-channel-width (base_dim=160, encoder mid-block attention at 640 channels) Wan2.2-
    TI2V-5B / Cosmos3-Nano VAE encoder, one pixel frame -- the exact geometry
    at which a NCC_IDDT901 persisted after the
    patchify/AvgDown3D/DupUp3D fixes (those were only exercised at TINY_TI2V_5B_VAE's shrunk
    base_dim=16 / width-64 mid-block). Spatial dims kept small for compile time; channel widths are
    the real ones."""
    import torch
    from diffusers import AutoencoderKLWan

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    torch.manual_seed(0)
    ref = AutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
    h = w = 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    x = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])[
        None, :, None
    ]
    with torch.no_grad():
        ref_mean = ref.encode(x).latent_dist.mean

    dev = _device()
    vae = NeuronAutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
    vae.load_state_dict(ref.state_dict(), strict=True)
    vae = vae.to(torch.bfloat16).to(dev)
    vae.compile(
        backend=_backend(),
        fullgraph=True,
        dynamic=False,
        compile_encoder=True,
        options={"model_name": "smoke_real_width_vae_encode", "compiler_args": COMPILER_ARGS},
    )
    t0 = time.time()
    with torch.no_grad():
        hd = vae._encode(x.to(torch.bfloat16).to(dev)).cpu()
    enc_s = time.time() - t0
    mean = hd[:, : hd.shape[1] // 2].float()
    r = {
        "latent_shape": list(mean.shape),
        "enc_rel_err": _rel(mean, ref_mean),
        "enc_first_s": round(enc_s, 1),
    }
    r["ok"] = tuple(mean.shape) == tuple(ref_mean.shape) and r["enc_rel_err"] < 0.02
    return r


# The Cosmos3 port's bisection: real-width Nano VAE device
# encode, one pixel frame, OK through 192px, FAILS NCC_IDDT901 at 208px and above (same compile
# site as _encode's patchify/AvgDown3D/DupUp3D fixes) -- a spatial-resolution compile threshold,
# strictly below any real conditioning-frame size (256px action min, 640px T2I/I2V). This sweep
# reproduces that threshold on our own harness and extends it to a STEADY (4-frame) chunk, since
# only the first_chunk path (check_real_width_vae_encode, one frame) was exercised before.
VAE_ENCODE_SWEEP_SIZES = [64, 80, 96, 112, 128, 160, 192, 208, 224, 240, 256, 480, 640, 704]


def check_vae_encode_size_sweep() -> dict:
    """Bisect the device encoder compile threshold (OK <=192px, NCC_IDDT901 >=208px) at
    the real Cosmos3-Nano/TI2V-5B channel width, for BOTH the first_chunk (1 pixel frame) and a
    steady (4 pixel frame) encoder chunk -- ``NeuronWanEncoder3d.forward`` compiles two
    specializations (``first_chunk=True/False``) and the original repro only exercised the first.

    Opt-in (``--only vae_encode_size_sweep`` or ``SMOKE_VAE_ENCODE_SWEEP=1``): each size is its own
    ``torch.compile`` call (new NEFF), so the full sweep is slow -- override the size list with
    ``SMOKE_VAE_ENCODE_SIZES`` (comma-separated px) for a faster bisection once a rough threshold is
    known. Not a pass/fail gate on the overall smoke: the point is to locate and report the
    threshold (and whether it moves between first_chunk and a steady chunk), not to block on it --
    device encode above threshold stays on the host-encode workaround either way.
    """
    import torch
    from diffusers import AutoencoderKLWan

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    sizes_env = os.environ.get("SMOKE_VAE_ENCODE_SIZES")
    sizes = [int(s) for s in sizes_env.split(",")] if sizes_env else VAE_ENCODE_SWEEP_SIZES

    torch.manual_seed(0)
    ref = AutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
    dev = _device()

    def encode_chunk_at(size: int, n_frames: int):
        """Build a fresh VAE + fresh compile for one (size, n_frames) point -- each size is a
        different NEFF, and reusing one `vae.compile()` call across sizes would just report the
        first size's compile outcome for every later size (`dynamic=False` pins the first shape)."""
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
        x = torch.stack([base * (1 - 0.05 * f) for f in range(n_frames)], dim=1)[None]
        with torch.no_grad():
            ref_mean = ref.encode(x).latent_dist.mean

        vae = NeuronAutoencoderKLWan(**REAL_WIDTH_TI2V_5B_VAE).eval()
        vae.load_state_dict(ref.state_dict(), strict=True)
        vae = vae.to(torch.bfloat16).to(dev)
        vae.compile(
            backend=_backend(),
            fullgraph=True,
            dynamic=False,
            compile_encoder=True,
            options={
                "model_name": f"smoke_vae_encode_sweep_{size}_{n_frames}f",
                "compiler_args": COMPILER_ARGS,
            },
        )
        xd = x.to(torch.bfloat16).to(dev).contiguous()
        t0 = time.time()
        with torch.no_grad():
            hd = vae._encode(xd).cpu()
        enc_s = time.time() - t0
        mean = hd[:, : hd.shape[1] // 2].float()
        return {
            "ok": tuple(mean.shape) == tuple(ref_mean.shape),
            "rel_err": _rel(mean, ref_mean),
            "enc_s": round(enc_s, 2),
        }

    r: dict = {"sizes": {}, "ok": True}
    for size in sizes:
        point: dict = {}
        for label, n_frames in (("first_chunk", 1), ("steady_chunk", 4)):
            try:
                point[label] = encode_chunk_at(size, n_frames)
            except Exception as exc:
                import traceback

                point[label] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-1200:],
                }
        r["sizes"][str(size)] = point
    # ok reports whether the sweep RAN (every size attempted, no harness crash), not whether every
    # size compiled -- the compile threshold itself is the information this check exists to report.
    return r


def check_vae_tiled_encode_sweep() -> dict:
    """The shared fix for Cosmos3 / Wan2.2-TI2V-5B device ENCODE above the 192/208 px single-graph
    threshold: fixed-shape spatial tiles BELOW it. Per (tile px, overlap px) ONE VAE + ONE compile
    (``use_tiling``, no fullgraph, ``compile_encoder=True``), then a sweep over frame sizes -- every
    size reuses the same two encoder graphs (first / steady chunk), which is the point of the
    fixed-shape tiling. Reports per size: rel vs the untiled fp32 CPU encode (= seams + device
    numerics), rel vs an fp32 CPU copy with the SAME tiling (device numerics only: the pass metric),
    the seams alone (CPU tiled vs CPU untiled), cold (first size) / warm timings, tile count.

    Env: ``SMOKE_VAE_TILE_SIZES`` (default ``192``; 160 FAILS NCC_IDDT901 on device, round 14, while
    192 passes -- the threshold is not monotonic in size), ``SMOKE_VAE_TILE_OVERLAPS`` (px, default
    ``96``), ``SMOKE_VAE_TILED_SIZES`` (frame px, default ``256,480,640,704``),
    ``SMOKE_VAE_TILED_FRAMES`` (default ``5``: first + one steady chunk), ``SMOKE_VAE_WEIGHTS``
    (see :func:`_vae_weights_source`). Opt-in: ``--only vae_tiled_encode_sweep``.
    """
    import torch

    tile_sizes = [int(s) for s in os.environ.get("SMOKE_VAE_TILE_SIZES", "192").split(",")]
    sizes = [int(s) for s in os.environ.get("SMOKE_VAE_TILED_SIZES", "256,480,640,704").split(",")]
    frames = int(os.environ.get("SMOKE_VAE_TILED_FRAMES", "5"))
    # Overlap in px. CPU, REAL Wan2.2-5B weights, tile 192, 1 frame: latent rel tiled-vs-untiled at
    # 256 px is 0.080 / 0.045 / 0.033 / 0.049 for overlap 32 / 64 / 96 / 128 (decode of the tiled
    # latent vs decode of the untiled one: 36.9 / 42.1 / 45.0 / 41.0 dB); at 480 px 0.126 / 0.092 /
    # 0.085 / 0.096 (34.1 / 36.8 / 37.2 / 35.9 dB). 96 is the knee: beyond it the linear ramp blends
    # across increasingly edge-corrupted tile content. The Wan VAE has NO GroupNorm (WanRMS_norm is a
    # per-pixel RMS over channels), so the cross-tile-statistics fix that repaired FLUX.2's tiled
    # decoder does not apply here; the residual is the conv receptive field + the mid-block
    # attention (ablating the attention changes 0.080 -> 0.070 at 256 px).
    overlaps = [int(s) for s in os.environ.get("SMOKE_VAE_TILE_OVERLAPS", "96").split(",")]
    # SMOKE_VAE_ROUNDTRIP=1: also decode each tiled latent on CPU and report the PSNR vs the decode
    # of the untiled latent (the end-to-end seam cost; costs one CPU fp32 decode per size).
    roundtrip = os.environ.get("SMOKE_VAE_ROUNDTRIP") == "1"

    ref, _ = _real_width_vae_pair()
    dev = _device()
    r: dict = {"frames": frames, "weights": _vae_weights_source(), "tiles": {}, "ok": True}

    def frame_stack(size):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
        return torch.stack([base * (1 - 0.05 * f) for f in range(frames)], dim=1)[None]

    for tile in tile_sizes:
        for overlap in overlaps:
            stride = tile - overlap
            _, vae = _real_width_vae_pair()
            # fp32 CPU copy with the SAME tiling: isolates device numerics from the tiling seams
            # (the seams are a property of the tile geometry and show up identically on CPU).
            _, cpu_tiled = _real_width_vae_pair()
            for v in (vae, cpu_tiled):
                _set_tiling(v, tile, overlap)
            cpu_tiled.compile(backend="eager", compile_encoder=True)
            vae = vae.to(torch.bfloat16).to(dev)
            vae.compile(
                backend=_backend(),
                fullgraph=False,  # the tiling loop is host code; only the per-tile graphs compile
                dynamic=False,
                compile_encoder=True,
                options={
                    "model_name": f"smoke_vae_tiled_enc_t{tile}",
                    "compiler_args": COMPILER_ARGS,
                },
            )
            per_tile: dict = {"stride": stride, "overlap": overlap, "sizes": {}}
            for size in sizes:
                x = frame_stack(size)
                with torch.no_grad():
                    ref_mean = ref.encode(x).latent_dist.mean
                    h_ct = cpu_tiled._encode(x)
                    cpu_tiled_mean = h_ct[:, : h_ct.shape[1] // 2]
                xd = x.to(torch.bfloat16).to(dev).contiguous()
                try:
                    t0 = time.time()
                    with torch.no_grad():
                        hd = vae._encode(xd)
                    cold = time.time() - t0
                    t0 = time.time()
                    with torch.no_grad():
                        hd = vae._encode(xd)
                    warm = time.time() - t0
                except Exception as exc:
                    import traceback

                    per_tile["sizes"][str(size)] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc()[-1200:],
                    }
                    r["ok"] = False
                    continue
                mean = hd[:, : hd.shape[1] // 2].cpu().float()
                n_per_axis = len(range(0, max(size - tile, 0), stride)) + 1 if size > tile else 1
                entry = {
                    "ok": tuple(mean.shape) == tuple(ref_mean.shape),
                    "rel_err_vs_cpu_untiled": _rel(mean, ref_mean),  # seams + device numerics
                    "rel_err_vs_cpu_tiled": _rel(mean, cpu_tiled_mean),  # device numerics only
                    "cpu_tiled_vs_untiled": _rel(cpu_tiled_mean, ref_mean),  # the seams alone
                    "first_s": round(cold, 2),
                    "warm_s": round(warm, 3),
                    "n_tiles": n_per_axis**2,
                }
                if roundtrip:
                    # Is the seam cost acceptable? Decode the device's tiled latent and the untiled
                    # CPU latent with the same untiled fp32 CPU decoder; PSNR between the two decodes
                    # is the end-to-end effect of the encode tiling (CPU, ~30 s at 480 px / frame5).
                    with torch.no_grad():
                        dec_tiled = ref.decode(mean).sample
                        dec_untiled = ref.decode(ref_mean).sample
                    pf = [
                        round(_psnr(dec_tiled[:, :, f], dec_untiled[:, :, f]), 2)
                        for f in range(dec_tiled.shape[2])
                    ]
                    entry["roundtrip_psnr_per_frame_db"] = pf
                    entry["roundtrip_psnr_min_db"] = min(pf)
                    entry["psnr_vs_input_untiled_db"] = round(
                        _psnr(dec_untiled[:, :, 0], x[:, :, 0]), 2
                    )
                    entry["psnr_vs_input_tiled_db"] = round(
                        _psnr(dec_tiled[:, :, 0], x[:, :, 0]), 2
                    )
                per_tile["sizes"][str(size)] = entry
            r["tiles"][f"{tile}/{overlap}"] = per_tile
    return r


def check_vae_decode_size_sweep() -> dict:
    """Decode analogue of :func:`check_vae_encode_size_sweep`: the real-width decoder has never run
    on device (round 14: tiled decode with the default 256 px / latent-16 tile failed NCC_IDDT901, as
    did the untiled 256 px decode), so bisect the decoder's single-graph threshold directly. Per
    pixel-equivalent size (default 128/192/256 px -> latent 8/12/16), ONE fresh VAE + compile, decode
    two latent frames (the first-frame AND the rest-frame decoder graphs), PSNR vs fp32 CPU.
    ``SMOKE_VAE_DECODE_SIZES`` overrides. Opt-in: ``--only vae_decode_size_sweep``."""
    import torch

    sizes = [int(s) for s in os.environ.get("SMOKE_VAE_DECODE_SIZES", "128,192,256").split(",")]
    ref, _ = _real_width_vae_pair()
    dev = _device()
    ratio = 16
    r: dict = {"weights": _vae_weights_source(), "sizes": {}, "ok": True}
    for size in sizes:
        torch.manual_seed(0)
        z = torch.randn(1, 48, 2, size // ratio, size // ratio) * 0.5
        with torch.no_grad():
            ref_dec = ref.decode(z).sample
        _, vae = _real_width_vae_pair()
        vae.use_tiling = False
        vae = vae.to(torch.bfloat16).to(dev)
        try:
            vae.compile(
                backend=_backend(),
                fullgraph=True,
                dynamic=False,
                options={
                    "model_name": f"smoke_vae_dec_sweep_{size}",
                    "compiler_args": COMPILER_ARGS,
                },
            )
            zd = z.to(torch.bfloat16).to(dev).contiguous()
            t0 = time.time()
            with torch.no_grad():
                dec = vae.decode(zd).sample.cpu()
            cold = time.time() - t0
            t0 = time.time()
            with torch.no_grad():
                vae.decode(zd).sample.cpu()
            warm = time.time() - t0
        except Exception as exc:
            import traceback

            r["sizes"][str(size)] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1200:],
            }
            continue
        per_frame = [round(_psnr(dec[:, :, f], ref_dec[:, :, f]), 2) for f in range(dec.shape[2])]
        r["sizes"][str(size)] = {
            "ok": tuple(dec.shape) == tuple(ref_dec.shape),
            "dec_psnr_per_frame_db": per_frame,
            "first_s": round(cold, 2),
            "warm_s": round(warm, 3),
        }
    return r


def check_tiny_vae() -> dict:
    import torch
    from diffusers import AutoencoderKLWan

    from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NeuronAutoencoderKLWan,
    )

    torch.manual_seed(0)
    ref = AutoencoderKLWan(**TINY_TI2V_5B_VAE).eval()
    h = w = 128
    frames = 5
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
    x = torch.stack([base * (1 - 0.05 * f) for f in range(frames)], dim=1)[None].clamp(-1, 1)
    with torch.no_grad():
        ref_mean = ref.encode(x).latent_dist.mean
        ref_dec = ref.decode(ref_mean).sample

    dev = _device()
    vae = NeuronAutoencoderKLWan(**TINY_TI2V_5B_VAE).eval()
    vae.load_state_dict(ref.state_dict(), strict=True)
    vae = vae.to(torch.bfloat16).to(dev)
    vae.compile(
        backend=_backend(),
        fullgraph=True,
        dynamic=False,
        compile_encoder=True,
        options={"model_name": "smoke_tiny_ti2v_vae", "compiler_args": COMPILER_ARGS},
    )
    t0 = time.time()
    with torch.no_grad():
        hd = vae._encode(x.to(torch.bfloat16).to(dev)).cpu()
    enc_s = time.time() - t0
    mean = hd[:, : hd.shape[1] // 2].float()
    t0 = time.time()
    with torch.no_grad():
        dec = vae.decode(ref_mean.to(torch.bfloat16).to(dev)).sample.cpu()
    dec_s = time.time() - t0
    per_frame = [round(_psnr(dec[:, :, f], ref_dec[:, :, f]), 2) for f in range(dec.shape[2])]
    r = {
        "latent_shape": list(mean.shape),
        "enc_rel_err": _rel(mean, ref_mean),
        "enc_first_s": round(enc_s, 1),
        "dec_shape": list(dec.shape),
        "dec_psnr_per_frame_db": per_frame,
        "dec_first_s": round(dec_s, 1),
    }
    r["ok"] = (
        tuple(mean.shape) == tuple(ref_mean.shape)
        and tuple(dec.shape) == tuple(ref_dec.shape)
        and r["enc_rel_err"] < 0.02
        and min(per_frame) > 35.0
    )
    return r


def check_vae_timing() -> dict:
    """Priority-3 baseline: the shared Wan VAE's existing device path (chunked encode + decode with
    the on-device feat cache, encoder compiled via ``compile_encoder=True``) timed per shape, cold and
    warm, encode AND decode, vs an eager fp32 CPU reference (PSNR for decode, rel_err for encode).
    Random weights at the REAL TI2V-5B/Cosmos3 channel width (``REAL_WIDTH_TI2V_5B_VAE``), so the
    graph sizes and instruction counts are the production ones even though the pixels are not.

    Shapes come from ``SMOKE_VAE_SHAPES`` as ``TxHxW[,TxHxW..]`` (default ``5x256x256``); set
    ``SMOKE_VAE_TILING=1`` to also time the tiled path (``use_tiling``, no fullgraph). Opt-in: only
    registered via ``--only vae_timing`` or ``SMOKE_VAE_TIMING=1`` -- it is a measurement, not a gate.
    """
    import torch

    shapes = []
    for spec in os.environ.get("SMOKE_VAE_SHAPES", "5x256x256").split(","):
        t_, h_, w_ = (int(s) for s in spec.lower().split("x"))
        shapes.append((t_, h_, w_))
    tiling = os.environ.get("SMOKE_VAE_TILING") == "1"

    ref, vae = _real_width_vae_pair()
    dev = _device()
    cpu_tiled = None
    if tiling:  # SMOKE_VAE_TILE_PX / SMOKE_VAE_TILE_OVERLAP: the same tile for encode and decode
        tile_px = int(os.environ.get("SMOKE_VAE_TILE_PX", "192"))
        overlap_px = int(os.environ.get("SMOKE_VAE_TILE_OVERLAP", "96"))
        _set_tiling(vae, tile_px, overlap_px)
        # fp32 CPU copy with the SAME tiling: the encode pass metric is device-vs-CPU-tiled (device
        # numerics only); the tiling's own cost is reported separately as the end-to-end PSNR of
        # decoding the device's tiled latent vs decoding the untiled CPU latent (round 15: the 480 px
        # run was flagged only because enc_rel_err 0.067 vs the UNTILED latent mixed the two).
        _, cpu_tiled = _real_width_vae_pair()
        _set_tiling(cpu_tiled, tile_px, overlap_px)
        cpu_tiled.compile(backend="eager", compile_encoder=True)
    else:
        vae.use_tiling = False
    vae = vae.to(torch.bfloat16).to(dev)
    t0 = time.time()
    vae.compile(
        backend=_backend(),
        fullgraph=not tiling,
        dynamic=False,
        compile_encoder=True,
        options={"model_name": "smoke_vae_timing", "compiler_args": COMPILER_ARGS},
    )
    r: dict = {
        "width_config": "ti2v5b_real_width",
        "weights": _vae_weights_source(),
        "tiling": tiling,
        "tile_px": vae.tile_sample_min_height if tiling else None,
        "tile_stride_px": vae.tile_sample_stride_height if tiling else None,
        "shapes": {},
        "ok": True,
    }
    r["compile_setup_s"] = round(time.time() - t0, 1)

    def timed(fn):
        t0 = time.time()
        with torch.no_grad():
            out = fn()
        return out, time.time() - t0

    for frames, h, w in shapes:
        key = f"{frames}x{h}x{w}"
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        base = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])
        x = torch.stack([base * (1 - 0.05 * f) for f in range(frames)], dim=1)[None].clamp(-1, 1)
        with torch.no_grad():
            ref_mean = ref.encode(x).latent_dist.mean
            t0 = time.time()
            ref_dec = ref.decode(ref_mean).sample
            cpu_dec_s = time.time() - t0
        xd = x.to(torch.bfloat16).to(dev).contiguous()
        zd = ref_mean.to(torch.bfloat16).to(dev).contiguous()
        try:
            hd, enc_cold = timed(lambda: vae._encode(xd).cpu())
            _, enc_warm = timed(lambda: vae._encode(xd).cpu())
            dec, dec_cold = timed(lambda: vae.decode(zd).sample.cpu())
            _, dec_warm = timed(lambda: vae.decode(zd).sample.cpu())
        except Exception as exc:
            import traceback

            r["shapes"][key] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1500:],
            }
            r["ok"] = False
            continue
        mean = hd[:, : hd.shape[1] // 2].float()
        per_frame = [round(_psnr(dec[:, :, f], ref_dec[:, :, f]), 2) for f in range(dec.shape[2])]
        s = {
            "latent_shape": list(mean.shape),
            "enc_rel_err": _rel(
                mean, ref_mean
            ),  # vs the UNTILED CPU latent: seams + device numerics
            "enc_cold_s": round(enc_cold, 2),
            "enc_warm_s": round(enc_warm, 3),
            "dec_psnr_min_db": min(per_frame),
            "dec_psnr_per_frame_db": per_frame,
            "dec_cold_s": round(dec_cold, 2),
            "dec_warm_s": round(dec_warm, 3),
            "cpu_fp32_dec_s": round(cpu_dec_s, 2),
        }
        enc_gate = s["enc_rel_err"]
        if cpu_tiled is not None:
            with torch.no_grad():
                h_ct = cpu_tiled._encode(x)
                cpu_tiled_mean = h_ct[:, : h_ct.shape[1] // 2]
                # end-to-end cost of the encode tiling: decode the device's tiled latent with the
                # untiled fp32 CPU decoder and compare with the decode of the untiled CPU latent
                rt = ref.decode(mean).sample
            s["enc_rel_err_vs_cpu_tiled"] = _rel(mean, cpu_tiled_mean)  # device numerics only
            s["enc_tiling_seams_rel"] = _rel(cpu_tiled_mean, ref_mean)  # CPU tiled vs untiled
            s["enc_tiled_roundtrip_psnr_db"] = [
                round(_psnr(rt[:, :, f], ref_dec[:, :, f]), 2) for f in range(rt.shape[2])
            ]
            s["enc_tiled_roundtrip_psnr_min_db"] = min(s["enc_tiled_roundtrip_psnr_db"])
            enc_gate = s["enc_rel_err_vs_cpu_tiled"]
        s["ok"] = (
            tuple(mean.shape) == tuple(ref_mean.shape)
            and tuple(dec.shape) == tuple(ref_dec.shape)
            and enc_gate < 0.05
            and s["dec_psnr_min_db"] > 30.0
            and s.get("enc_tiled_roundtrip_psnr_min_db", 99.0) > 30.0
        )
        r["shapes"][key] = s
        r["ok"] = r["ok"] and s["ok"]
    return r


def check_block_runner() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch._dynamo.utils import counters

    from vllm_omni_neuron.diffusion.layers.block_graphs import BlockGraphRunner
    from vllm_omni_neuron.diffusion.layers.modulation_tables import ModulationTables

    class Block(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.ada = nn.Linear(64, 3 * dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.out = nn.Linear(dim, dim)

        def forward(self, x, table):
            shift, scale, gate = table.chunk(3, -1)
            h = F.layer_norm(x, x.shape[-1:]) * (1 + scale) + shift
            q, k, v = self.qkv(h).view(*h.shape[:2], 3, 4, -1).permute(2, 0, 3, 1, 4).unbind(0)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
            return x + gate * self.out(a)

    torch.manual_seed(0)
    blocks = nn.ModuleList(Block() for _ in range(7))
    tabs = ModulationTables.from_linears([b.ada for b in blocks], compute_dtype=torch.float32)
    temb = torch.randn(1, 64)
    x = torch.randn(1, 512, 256)
    with torch.no_grad():
        ref = x
        for b, t in zip(blocks, tabs.tables(temb)):
            ref = b(ref, t)
        # pure-dtype baseline: the same blocks in bf16 on CPU
        blocks = blocks.to(torch.bfloat16)
        ref16 = x.to(torch.bfloat16)
        for b, t in zip(blocks, tabs.tables(temb)):
            ref16 = b(ref16, t.to(torch.bfloat16))
    dev = _device()
    blocks = blocks.to(dev)
    opts = {
        "model_name": "smoke_block_runner",
        "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
    }
    runner = BlockGraphRunner(
        blocks,
        3,
        block_call=lambda blk, c, sa, la, kw: blk(c, la[0]),
        compile_fn=lambda f: torch.compile(
            f, backend=_backend(), fullgraph=True, dynamic=False, options=opts
        ),
    )
    tables = [(t.to(torch.bfloat16).to(dev),) for t in tabs.tables(temb).unbind(0)]
    before = counters["stats"]["unique_graphs"]
    xd = x.to(torch.bfloat16).to(dev)
    t0 = time.time()
    with torch.no_grad():
        out = runner(xd, per_layer=tables).cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        runner(xd, per_layer=tables).cpu()
    warm = time.time() - t0
    graphs = counters["stats"]["unique_graphs"] - before
    r = {
        "dynamo_graphs": graphs,
        "expected_graphs": runner.num_graphs,
        "rel_err": _rel(out.float() - x, ref - x),
        "cpu_bf16_rel_err": _rel(ref16.float() - x, ref - x),
        "first_s": round(first, 1),
        "warm_s": round(warm, 4),
    }
    # three-way: device error within 2x the pure-bf16 error of the same blocks on CPU
    r["ok"] = graphs == runner.num_graphs and r["rel_err"] <= 2 * r["cpu_bf16_rel_err"] + 0.005
    return r


def _decode_attention_harness(
    q_heads: int,
    kv_heads: int,
    head_dim: int = 128,
    max_len: int = 128,
    prompt_len: int = 12,
    *,
    identical_kv: bool = False,
):
    """Shared setup for the decode-attention checks: a Qwen3-shaped layer with weights scaled like
    a trained model's (``1/sqrt(fan_in)``: with unscaled ``randn`` weights at hidden 2048 the scores
    are ~1e3, the softmax is a near-argmax and any bf16 rounding flips the winner -- round 15's
    'prefill 8.1%' was that artefact, not a device fault). ``identical_kv`` makes every KV head's
    projection the same, so a GQA head-grouping mismatch in a kernel cannot change the result.

    Returns ``(cfg, weights_cpu_fp32, make_dev_weights, run)`` where ``run(weights, device, dtype,
    diag)`` does prefill + 3 compiled decode steps and returns ``(outputs, prefill_s, step_times,
    cache_snapshots)``."""
    import torch

    from vllm_omni_neuron.diffusion.attention.decode_attention import (
        DecodeAttentionConfig,
        DecodeWeights,
        StaticKVCache,
        decode_prefill_from_hidden,
        decode_step,
    )

    torch.manual_seed(0)
    cfg = DecodeAttentionConfig(
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, max_len=max_len, dtype=torch.bfloat16
    )
    hidden_size = cfg.hidden_size
    s_in = hidden_size**-0.5
    q_w = torch.randn(q_heads * head_dim, hidden_size) * s_in
    k_w = torch.randn(kv_heads * head_dim, hidden_size) * s_in
    v_w = torch.randn(kv_heads * head_dim, hidden_size) * s_in
    o_w = torch.randn(hidden_size, hidden_size) * s_in
    if identical_kv:
        k_w = k_w[:head_dim].repeat(kv_heads, 1)
        v_w = v_w[:head_dim].repeat(kv_heads, 1)
    weights_cpu = DecodeWeights.from_separate(q_w, k_w, v_w, o_w).to(torch.float32)

    def make_dev_weights(dev):
        return DecodeWeights.from_separate(q_w, k_w, v_w, o_w).to(cfg.dtype).to(dev)

    def rope_tables(n, theta=10000.0):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(n).float(), inv_freq)
        return freqs.cos(), freqs.sin()

    cos_table, sin_table = rope_tables(max_len)
    full_hidden = torch.randn(1, prompt_len + 3, hidden_size)

    def _run_step(fn, label, diag):
        try:
            return fn()
        except Exception as exc:
            if diag is not None:
                diag.append({"stage": label, "error": f"{type(exc).__name__}: {exc}"})
            raise

    def run(weights, device, dtype, diag: list | None = None):
        cfg_d = DecodeAttentionConfig(
            q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, max_len=max_len, dtype=dtype
        )
        cache = StaticKVCache(cfg_d, device)
        # Every tensor handed to a compiled region below is a fresh, contiguous base tensor built
        # on the host and moved whole (never a view of a device tensor): round 11's device run was
        # refused at the executor with 'Detected non-contiguous slicing for requested Device Tensor'
        # because the natural q/k/v inputs to decode_prefill are .transpose(1, 2) views of the QKV
        # projection -- which is why the projection now lives INSIDE the compiled region
        # (decode_prefill_from_hidden), exactly as decode_step always did.
        hidden_prompt = full_hidden[:, :prompt_len].to(dtype).to(device).contiguous()
        cos_p = cos_table[:prompt_len].to(dtype).to(device).contiguous()
        sin_p = sin_table[:prompt_len].to(dtype).to(device).contiguous()
        if diag is not None:
            import vllm_omni_neuron

            diag.append({"stage": "env", "module_file": vllm_omni_neuron.__file__})
        # CPU reference runs eagerly; the device runs compiled (never the Neuron backend on CPU
        # tensors -- see _compile_on). Dynamo caches compiled code on decode_step.__code__ and does
        # NOT guard on env reads inside it (the kernel opt-in, the core generation), so without a
        # reset a second run with a different gate silently REUSES the first run's graph -- round 14
        # reported bit-identical rel_err for both paths. Reset before each compiled run.
        torch._dynamo.reset()
        from torch._dynamo.utils import counters as _dyn_counters

        graphs_before = _dyn_counters["stats"]["unique_graphs"]
        prefill_c = _compile_on(device, decode_prefill_from_hidden, "smoke_decode_prefill")
        step_c = _compile_on(device, decode_step, "smoke_decode_step")
        t0 = time.time()
        prefill_out = _run_step(
            lambda: prefill_c(cfg_d, weights, cache, hidden_prompt, cos=cos_p, sin=sin_p),
            "decode_prefill",
            diag,
        )
        prefill_out.cpu()  # eager .float() on a device tensor is an unsupported eager cast here
        prefill_s = time.time() - t0
        caches = {"after_prefill": (cache.fill, cache.k.cpu().float(), cache.v.cpu().float())}
        outs, times = [], []
        for i, t in enumerate(range(prompt_len, prompt_len + 3)):
            tok = full_hidden[:, t : t + 1].to(dtype).to(device).contiguous()
            cos_t = cos_table[t : t + 1].to(dtype).to(device).contiguous()
            sin_t = sin_table[t : t + 1].to(dtype).to(device).contiguous()
            t0 = time.time()
            step_out = _run_step(
                lambda: step_c(cfg_d, weights, cache, tok, cos=cos_t, sin=sin_t),
                f"decode_step[{i}]",
                diag,
            )
            step_out.cpu()  # force completion for a representative per-call timing
            times.append(time.time() - t0)
            outs.append(step_out)
        caches["after_steps"] = (cache.fill, cache.k.cpu().float(), cache.v.cpu().float())
        # Dynamo unique graphs traced for prefill + 3 steps: 2 means the steps shared ONE graph
        # (position-static decode); 4 means a graph per position (the pre-r19 behaviour).
        caches["unique_graphs"] = _dyn_counters["stats"]["unique_graphs"] - graphs_before
        return (
            torch.cat([prefill_out.cpu().float(), *[o.cpu().float() for o in outs]], 0),
            prefill_s,
            times,
            caches,
        )

    return cfg, weights_cpu, make_dev_weights, run


class _decode_step_kernel_gate:
    """Context manager: force decode_step's fused-kernel opt-in on or off for one compiled run."""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        from vllm_omni_neuron.diffusion.attention.decode_attention import DECODE_STEP_NKI_ENV

        self.key = DECODE_STEP_NKI_ENV
        self.prev = os.environ.get(self.key)
        os.environ[self.key] = "1" if self.on else "0"
        return self

    def __exit__(self, *exc):
        if self.prev is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.prev


def check_decode_attention() -> dict:
    """Qwen3-shaped decode-attention layer: prefill + 3 decode steps on device vs CPU fp32 (and a
    CPU bf16 run of the same code, the error band the device must land in). Runs BOTH decode paths:
    the default torch path (the gate: ``ok``) and the opt-in fused ``attention_decode`` kernel path
    (``VLLM_OMNI_NEURON_DECODE_STEP_NKI=1``; reported, NOT gating -- round 15 found it at rel 1.03
    with a verified cache, see :func:`decode_step_uses_nki_kernel`). Also reads the cache back after
    prefill and after the steps (fill, written-slot rel vs CPU, unfilled-slot absmax)."""
    import torch

    from vllm_omni_neuron.diffusion.attention.decode_attention import (
        can_use_decode_kernel,
        prefill_uses_nki_kernel,
    )

    q_heads, kv_heads, head_dim, prompt_len = 16, 2, 128, 12  # Qwen3-VL-ish GQA shape
    cfg, weights_cpu, make_dev_weights, run = _decode_attention_harness(
        q_heads, kv_heads, head_dim, prompt_len=prompt_len
    )
    ref, _, _, caches_ref = run(weights_cpu, torch.device("cpu"), torch.float32)
    ref_bf16, _, _, _ = run(weights_cpu.to(torch.bfloat16), torch.device("cpu"), torch.bfloat16)
    dev = _device()
    weights_dev = make_dev_weights(dev)
    diag_kernel: list = []
    diag_torch: list = []

    with _decode_step_kernel_gate(False):
        try:
            out_torch, prefill_s_torch, times_torch, caches_torch = run(
                weights_dev, dev, cfg.dtype, diag=diag_torch
            )
        except Exception as exc:
            import traceback

            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc} (torch path)",
                "trace": traceback.format_exc()[-2000:],
                "diag_torch": diag_torch,
            }
    kernel_err = None
    with _decode_step_kernel_gate(True):
        try:
            out_kernel, prefill_s_kernel, times_kernel, caches_kernel = run(
                weights_dev, dev, cfg.dtype, diag=diag_kernel
            )
        except Exception as exc:
            import traceback

            kernel_err = {
                "error": f"{type(exc).__name__}: {exc} (kernel path, opt-in)",
                "trace": traceback.format_exc()[-1500:],
                "diag_kernel": diag_kernel,
            }

    P = prompt_len  # rows [0:P] are the prefill output, [P:] the three decode steps

    def cache_diag(caches):
        d = {}
        for stage in ("after_prefill", "after_steps"):
            fill, k, v = caches[stage]
            fill_r, k_r, v_r = caches_ref[stage]
            n = fill_r
            d[f"fill_{stage}"] = fill
            d[f"k_rel_{stage}"] = _rel(k[:, :, :n], k_r[:, :, :n])  # written slots vs CPU
            d[f"v_rel_{stage}"] = _rel(v[:, :, :n], v_r[:, :, :n])
            d[f"k_unfilled_absmax_{stage}"] = k[:, :, n:].abs().max().item()  # must stay 0
        d["unique_graphs"] = caches.get("unique_graphs")
        return d

    rel_torch = _rel(out_torch, ref)
    cpu_bf16_prefill = _rel(ref_bf16[:P], ref[:P])
    cpu_bf16_decode = _rel(ref_bf16[P:], ref[P:])
    r = {
        "shape": {"q_heads": q_heads, "kv_heads": kv_heads, "head_dim": head_dim},
        "prefill_path": "nki" if prefill_uses_nki_kernel() else "torch",  # opt-in; torch by default
        "decode_step_default_path": "torch",  # the kernel is opt-in (DECODE_STEP_NKI_ENV)
        "kernel_eligible_shape": can_use_decode_kernel(cfg, 1) and dev.type != "cpu",
        # the error band: the SAME code on CPU in bf16 vs CPU fp32
        "cpu_bf16_prefill_rel_err": cpu_bf16_prefill,
        "cpu_bf16_decode_rel_err": cpu_bf16_decode,
        # default (torch) path on device vs CPU fp32 -- the gate
        "rel_err_torch_path": rel_torch,
        "prefill_rel_err_torch_path": _rel(out_torch[:P], ref[:P]),
        "decode_rel_err_torch_path": _rel(out_torch[P:], ref[P:]),
        "prefill_first_s_torch_path": round(prefill_s_torch, 2),
        "decode_step_s_torch_path": [round(t, 4) for t in times_torch],
        "cache_torch_path": cache_diag(caches_torch),
    }
    if kernel_err is None:
        r.update(
            {
                "rel_err_kernel_path": _rel(out_kernel, ref),
                "prefill_rel_err_kernel_path": _rel(out_kernel[:P], ref[:P]),
                "decode_rel_err_kernel_path": _rel(out_kernel[P:], ref[P:]),
                "prefill_first_s_kernel_path": round(prefill_s_kernel, 2),
                "decode_step_s_kernel_path": [round(t, 4) for t in times_kernel],
                # bit-identical kernel/torch outputs would mean the two runs shared one Dynamo graph
                "decode_rel_kernel_vs_torch": _rel(out_kernel[P:], out_torch[P:]),
                "cache_kernel_path": cache_diag(caches_kernel),
            }
        )
        r["kernel_path_ok"] = r["decode_rel_err_kernel_path"] <= 2 * cpu_bf16_decode + 0.01
    else:
        r["kernel_path"] = kernel_err
        r["kernel_path_ok"] = False
    # pass band for the default path: within 2x the CPU-bf16 error of the identical code, +1%
    r["ok_numerics"] = (
        r["prefill_rel_err_torch_path"] <= 2 * cpu_bf16_prefill + 0.01
        and r["decode_rel_err_torch_path"] <= 2 * cpu_bf16_decode + 0.01
        and r["cache_torch_path"]["k_unfilled_absmax_after_steps"] == 0.0
    )
    # Timing gate (r19): decode_step is position-static since r19 (cache.pos is a device tensor), so
    # steps 2 and 3 must REUSE step 1's graph -- no Dynamo retrace, no NEFF lookup: warm step well
    # under r17's 0.32 s (which was a per-step retrace + cache hit) and r18's 2.4 s (retrace + cache
    # MISS). `graphs_torch_path` is Dynamo's unique-graph count for prefill + 3 steps: expect 2.
    warm_torch = sorted(times_torch[1:])
    r["decode_warm_s_torch_path"] = round(warm_torch[0], 4) if warm_torch else None
    r["decode_warm_budget_s"] = 0.48  # 1.5x r17's 0.32 s
    r["ok_timing"] = dev.type == "cpu" or (
        bool(warm_torch) and warm_torch[0] <= r["decode_warm_budget_s"]
    )
    r["ok"] = r["ok_numerics"] and r["ok_timing"]
    return r


def check_decode_attention_kernel_diag() -> dict:
    """Bisect the opt-in fused ``attention_decode`` kernel path's round-15 failure (decode rel 1.03 at
    the Qwen3 GQA shape, identical verified cache, kernel's own CPU fallback == our torch path):

    * ``mha``: ``kv_heads == q_heads`` (16/16) -- no GQA grouping at all;
    * ``gqa_identical_kv``: 16 query heads over 2 KV heads whose projections are IDENTICAL, so
      whichever KV head the kernel pairs a query head with, the answer is the same;
    * ``gqa``: the production shape (16/2), the failing case, as the control.

    mha + gqa_identical_kv passing while gqa fails = the kernel's GQA head grouping differs from
    ``repeat_interleave`` (query head ``h`` <-> KV head ``h // groups``). All three failing = the
    kernel reads a different cache layout / convention than this layer writes. Each case: prefill
    (torch) + 3 compiled kernel decode steps, decode rows vs CPU fp32. Opt-in:
    ``--only decode_attention_kernel_diag``."""
    import torch

    dev = _device()
    r: dict = {"cases": {}, "ok": True}
    for name, q_heads, kv_heads, identical in (
        ("mha", 16, 16, False),
        ("gqa_identical_kv", 16, 2, True),
        ("gqa", 16, 2, False),
    ):
        cfg, weights_cpu, make_dev_weights, run = _decode_attention_harness(
            q_heads, kv_heads, identical_kv=identical
        )
        P = 12
        ref, _, _, _ = run(weights_cpu, torch.device("cpu"), torch.float32)
        ref_bf16, _, _, _ = run(weights_cpu.to(torch.bfloat16), torch.device("cpu"), torch.bfloat16)
        diag: list = []
        with _decode_step_kernel_gate(True):
            try:
                out, prefill_s, times, _ = run(make_dev_weights(dev), dev, cfg.dtype, diag=diag)
            except Exception as exc:
                import traceback

                r["cases"][name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-1500:],
                    "diag": diag,
                }
                r["ok"] = False
                continue
        band = _rel(ref_bf16[P:], ref[P:])
        c = {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "identical_kv": identical,
            "cpu_bf16_decode_rel_err": band,
            "decode_rel_err_kernel_path": _rel(out[P:], ref[P:]),
            "prefill_rel_err": _rel(out[:P], ref[:P]),
            "decode_step_s": [round(t, 4) for t in times],
        }
        c["ok"] = c["decode_rel_err_kernel_path"] <= 2 * band + 0.01
        r["cases"][name] = c
        r["ok"] = r["ok"] and c["ok"]
    return r


def check_decode_prefill_nki_opt_in() -> dict:
    """The NKI attention_cte prefill path, OPT-IN (VLLM_OMNI_NEURON_DECODE_PREFILL_NKI=1) and off
    by default (see decode_attention.py's module docstring: it is the only causal_mask=True use of
    this kernel in the plugin, and failed to compile under neuron_native_lite in rounds 7-8 even
    with the kernel at module scope -- the fix that made the VAE's own causal_mask=False use of the
    same kernel work). Differences from the working _vae_nki_attn, recorded here for whoever picks
    this up: causal_mask=True (vs False), single-head-per-call multi-head-batch layout from this
    module's own q[0]/k_full[0]/v_full[0] slicing (vs the VAE's per-frame single-head layout), same
    tp_q/tp_k/tp_out/softmax_dtype/mm_out_dtype args, same wrap_nki + module-scope-kernel pattern.
    Isolated here so a failure doesn't block check_decode_attention's (default, torch-path) result.
    """
    import torch

    from vllm_omni_neuron.diffusion.attention.decode_attention import (
        PREFILL_NKI_ENV,
        DecodeAttentionConfig,
        _prefill_attention_cte,
    )

    torch.manual_seed(0)
    q_heads, kv_heads, head_dim, prompt_len = 16, 2, 128, 12
    cfg = DecodeAttentionConfig(
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, max_len=128, dtype=torch.bfloat16
    )
    dev = _device()
    q = torch.randn(1, q_heads, prompt_len, head_dim)
    k = torch.randn(1, kv_heads, prompt_len, head_dim)
    v = torch.randn(1, kv_heads, prompt_len, head_dim)
    k_full = k.repeat_interleave(cfg.num_kv_groups, dim=1)
    v_full = v.repeat_interleave(cfg.num_kv_groups, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k_full, v_full, is_causal=True, scale=cfg.scale
    )

    kernel_c = torch.compile(
        _prefill_attention_cte,
        backend=_backend(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": "smoke_decode_prefill_nki",
            "compiler_args": ["--model-type=unet-inference", "--auto-cast=none", "-O1"],
        },
    )
    qd = (q[0] * cfg.scale).to(torch.bfloat16).to(dev).contiguous()
    kd = k_full[0].to(torch.bfloat16).to(dev).contiguous()
    vd = v_full[0].to(torch.bfloat16).to(dev).contiguous()
    t0 = time.time()
    with torch.no_grad():
        out = kernel_c(qd, kd, vd)[None].cpu()
    first = time.time() - t0
    r = {
        "env_flag": PREFILL_NKI_ENV,
        "rel_err": _rel(out.float(), ref),
        "first_s": round(first, 1),
        "shape": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "prompt_len": prompt_len,
        },
    }
    r["ok"] = tuple(out.shape) == tuple(ref.shape) and r["rel_err"] < 0.05
    return r


def check_neighborhood_attention() -> dict:
    """Halo-tiled neighborhood (NATTEN/Swin) attention on device vs CPU fp32, at FLUX-3-Action's
    single-frame VAE grid (136x184, window 5x5) -- the grid whose gather/mask formulations all failed
    neuronx-cc. Also times it vs the
    dense masked reference (the formulation that does not compile, so CPU-only, for scale)."""
    import torch

    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        neighborhood_attention_tiled,
        neighborhood_bias,
    )

    def dense_ref(q, k, v, kernel):
        # NATTEN non-causal 2D mask, O(N^2) -- the formulation that fails to compile; CPU reference.
        import torch as t

        n_ax = 2
        axes = q.shape[1 : 1 + n_ax]
        coords = t.stack(t.meshgrid(*[t.arange(n) for n in axes], indexing="ij"), -1).reshape(
            -1, n_ax
        )
        qc, kc = coords[:, None, :], coords[None, :, :]
        allowed = t.ones(qc.shape[0], kc.shape[1], dtype=t.bool)
        for a in range(n_ax):
            n, kn = axes[a], kernel[a]
            qa, ka = qc[..., a], kc[..., a]
            left, right = kn // 2, kn // 2 + (kn % 2 - 1)
            center = qa.clamp(left, n - 1 - right)
            allowed &= ((center - ka >= 0) & (center - ka <= left)) | (
                (ka - center >= 0) & (ka - center <= right)
            )
        b, heads, d = q.shape[0], q.shape[-2], q.shape[-1]
        qf, kf, vf = (x.reshape(b, -1, heads, d).transpose(1, 2).float() for x in (q, k, v))
        sc = torch.matmul(qf, kf.transpose(-2, -1)) * (d**-0.5)
        sc = sc.masked_fill(~allowed, float("-inf"))
        o = torch.matmul(torch.softmax(sc, -1), vf)
        return o.transpose(1, 2).reshape(q.shape)

    torch.manual_seed(0)
    h, w, nh, d, kernel, tile = 136, 184, 4, 64, [5, 5], [16, 16]
    q = torch.randn(1, h, w, nh, d)
    k = torch.randn(1, h, w, nh, d)
    v = torch.randn(1, h, w, nh, d)
    ref = dense_ref(q, k, v, kernel)
    dev = _device()
    qd, kd, vd = (x.to(torch.bfloat16).to(dev).contiguous() for x in (q, k, v))
    # The additive band bias is a host constant for this layout; hand it to the compiled region as
    # a plain contiguous graph input rather than letting the trace build it (see neighborhood_bias).
    bias_dev = neighborhood_bias((h, w), kernel, [False, False], tile).to(dev).contiguous()
    # CPU bf16 baseline at the same shape: the error band the device result must land in.
    cpu_bf16 = neighborhood_attention_tiled(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        kernel,
        [False, False],
        tile,
    )
    cpu_bf16_rel = _rel(cpu_bf16.float(), ref)

    torch._dynamo.reset()
    from torch._dynamo.utils import counters as _dyn_counters

    fn = _compile_on(
        dev,
        neighborhood_attention_tiled,
        "smoke_neighborhood_attn",
        ["--model-type=unet-inference", "--auto-cast=none", "-O1"],
    )
    graphs0 = _dyn_counters["stats"]["unique_graphs"]
    t0 = time.time()
    with torch.no_grad():
        out = fn(qd, kd, vd, kernel, [False, False], tile, bias_dev).cpu()
    first = time.time() - t0
    warms = []
    for _ in range(3):  # best of 3: r18's warm_s 74.8 s was a RECOMPILE per call, not load
        t0 = time.time()
        with torch.no_grad():
            fn(qd, kd, vd, kernel, [False, False], tile, bias_dev).cpu()
        warms.append(time.time() - t0)
    warm = min(warms)
    main_graphs = _dyn_counters["stats"]["unique_graphs"] - graphs0
    r = {
        "grid": [h, w],
        "window": kernel,
        "tile": tile,
        "rel_err": _rel(out.float(), ref),
        "cpu_bf16_rel_err": cpu_bf16_rel,
        "first_s": round(first, 1),
        "warm_s": round(warm, 4),
        "warm_s_all": [round(t, 4) for t in warms],
        "unique_graphs": main_graphs,  # must be 1: r18's 74.8 s warm was a recompile per call
    }
    # Where is the error? Interior = the fully covered tiles; border = the last (padded/cropped) tile
    # row/column (136 = 8*16 + 8, 184 = 11*16 + 8). A border-concentrated error points at the
    # uncollapse/crop or the padding tiles; a uniform one at the attention arithmetic.
    hi, wi = (h // tile[0]) * tile[0], (w // tile[1]) * tile[1]
    r["rel_err_interior"] = _rel(out[:, :hi, :wi], ref[:, :hi, :wi])
    r["rel_err_border_rows"] = _rel(out[:, hi:], ref[:, hi:])
    r["rel_err_border_cols"] = _rel(out[:, :, wi:], ref[:, :, wi:])
    r["rel_err_first_tile"] = _rel(out[:, : tile[0], : tile[1]], ref[:, : tile[0], : tile[1]])

    # Rounds 15-17 (per-tile maps + shift detection in r17): the wrong region was a function of the
    # TILING -- 16x16: only the last tile ROW (0.20); 8x8: everything but the last tile COLUMN (0.19,
    # then NaN in r17) -- not the padding, not the edge semantics, and best_shift found no
    # displacement. r17's single-graph bisect localised the trigger: the SAME graph passed with
    # host-windowed q/k/v (both tilings, 0.0036), with fp32 inputs (8x8, 2e-6) and with the crop on
    # the host (16x16), while every stage alone had always matched. So the compiler mis-fuses the
    # overlapping strided bf16 window reads (narrow + stack) with their consumer. Fix (r18): the
    # overlapping K/V windows are formed by a 0/1 SELECTION MATMUL (window_impl="select", the new
    # default) -- no overlapping slice in the graph at all; r18 confirmed the numerics (0.0036 at
    # both tilings) and r19 fixes its recompile-per-call (see run_variant / the variant list).
    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        neighborhood_select_matrices,
        pad_free_tile,
    )

    flags = ["--model-type=unet-inference", "--auto-cast=none", "-O1"]
    band = 2 * cpu_bf16_rel + 0.005

    def tile_map(o, rf, t):
        """Per-tile rel error as a list of strings ('.' within 3x the CPU-bf16 band, 'x' outside),
        plus the bad-tile count, so the wrong REGION reads off directly."""
        rows, bad = [], 0
        for i in range(0, h, t[0]):
            line = ""
            for j in range(0, w, t[1]):
                e = _rel(o[:, i : i + t[0], j : j + t[1]], rf[:, i : i + t[0], j : j + t[1]])
                ok_ = e <= 3 * cpu_bf16_rel + 0.005
                bad += 0 if ok_ else 1
                line += "." if ok_ else "x"
            rows.append(line)
        return rows, bad

    def best_shift(o, rf, t):
        """Is the device result the reference DISPLACED? rel of out vs ref shifted by (dh, dw) over
        the overlap, for small shifts and whole-tile shifts; returns the best (dh, dw, rel)."""
        best = None
        dhs = sorted({*range(-2, 3), -t[0], t[0]})
        dws = sorted({*range(-2, 3), -t[1], t[1]})
        for dh in dhs:
            for dw in dws:
                oo = o[:, max(0, dh) : h + min(0, dh), max(0, dw) : w + min(0, dw)]
                rr = rf[:, max(0, -dh) : h + min(0, -dh), max(0, -dw) : w + min(0, -dw)]
                e = _rel(oo, rr)
                if best is None or e < best[2]:
                    best = (dh, dw, e)
        return {"dh": best[0], "dw": best[1], "rel_err": best[2]}

    def region_stats(o, rf, t):
        hi_, wi_ = (h // t[0]) * t[0], (w // t[1]) * t[1]
        m, nbad = tile_map(o, rf, t)
        return {
            "rel_err": _rel(o.float(), rf),
            "rel_err_interior": _rel(o[:, :hi_, :wi_], rf[:, :hi_, :wi_]),
            "rel_err_border_rows": _rel(o[:, hi_:], rf[:, hi_:]),
            "rel_err_border_cols": _rel(o[:, :, wi_:], rf[:, :, wi_:]),
            "bad_tiles": nbad,
            "tile_map": m,
            "best_shift": best_shift(o.float(), rf, t),
        }

    def run_variant(name, fn, args, t, rf, kwargs=None, warm_budget_s=None):
        # Each variant is its own Dynamo universe: all variants compile the SAME code object, and
        # r18 hit Dynamo's recompile limit (8) on the 7th variant (FailOnRecompileLimitHit) because
        # every distinct kwarg set is another cache entry on that frame.
        torch._dynamo.reset()
        g0 = _dyn_counters["stats"]["unique_graphs"]
        try:
            c_fn = _compile_on(dev, fn, f"smoke_na_{name}", flags)
            t0 = time.time()
            with torch.no_grad():
                o = c_fn(*args, **(kwargs or {})).cpu()
            v_first = time.time() - t0
            v_warms = []
            for _ in range(3):
                t0 = time.time()
                with torch.no_grad():
                    c_fn(*args, **(kwargs or {})).cpu()
                v_warms.append(time.time() - t0)
            v_warm = min(v_warms)
        except Exception as exc:
            return {"tile": list(t), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        res = {
            "tile": list(t),
            "first_s": round(v_first, 1),
            "warm_s": round(v_warm, 4),
            "warm_s_all": [round(x, 4) for x in v_warms],
            "unique_graphs": _dyn_counters["stats"]["unique_graphs"] - g0,
        }
        if not torch.isfinite(o).all():
            res["nonfinite"] = int((~torch.isfinite(o)).sum())
        res.update(region_stats(o, rf, t))
        res["ok_numerics"] = tuple(o.shape) == (1, h, w, nh, d) and res["rel_err"] <= band
        res["ok_timing"] = (
            True
            if warm_budget_s is None or dev.type == "cpu"  # the budget is a device number
            else (v_warm <= warm_budget_s and res["unique_graphs"] == 1)
        )
        res["ok"] = res["ok_numerics"] and res["ok_timing"]
        return res

    variants = {}
    pf_tile = tuple(pad_free_tile((h, w), kernel))
    WARM_BUDGET_S = (
        0.1  # r19 gate: warm neighborhood attention < 100 ms at 136x184 (r17 slice: 12.7 ms)
    )
    for tname, t in (("pf", pf_tile), ("t16", tuple(tile))):
        v_bias = neighborhood_bias((h, w), kernel, [False, False], t).to(dev).contiguous()
        v_sel = tuple(
            s.to(dev).contiguous()
            for s in neighborhood_select_matrices((h, w), kernel, t, torch.bfloat16)
        )
        args = (qd, kd, vd, kernel, [False, False], list(t), v_bias)
        # r19: `select` = the production default (selection matrices built IN-GRAPH from arange +
        # compare, no cache: r18's dict cache made Dynamo recompile on every call, warm_s 58-75 s);
        # `select_host` = the same op with the matrices as host-built graph inputs (r18's t16_select
        # graph #2, 14.8 ms); `slice` = the old windowing (control: numerically wrong on Trn2).
        # Dropped: *_upcast (r18: select_upcast failed to compile NCC_ILSA902; slice_upcast was
        # wrong, 0.0158 in the first tile column) -- upcast_first is documented as not a device path.
        for vname, kw in (
            ("select", {"window_impl": "select"}),
            ("select_host", {"window_impl": "select", "select": v_sel}),
            ("slice", {"window_impl": "slice"}),
        ):
            variants[f"{tname}_{vname}"] = run_variant(
                f"{tname}_{vname}",
                neighborhood_attention_tiled,
                args,
                t,
                ref,
                kw,
                warm_budget_s=WARM_BUDGET_S if vname != "slice" else None,
            )
    r["variants"] = variants
    r["warm_budget_s"] = WARM_BUDGET_S
    r["ok_tile16_padded"] = tuple(out.shape) == (1, h, w, nh, d) and r["rel_err"] <= band
    # the gate is the op's DEFAULTS (tile=None -> pad_free_tile, window_impl="select", in-graph
    # selection matrices), i.e. what an integrator gets with neighborhood_attention_tiled(q, k, v,
    # kernel): numerics within the band AND one graph AND warm under budget; plus the main 16x16 run
    # (default windowing, bias input) must be one graph and warm under budget too.
    r["ok_timing"] = dev.type == "cpu" or (r["unique_graphs"] == 1 and r["warm_s"] <= WARM_BUDGET_S)
    r["ok"] = bool(variants.get("pf_select", {}).get("ok", False)) and r["ok_timing"]
    # Per-stage bisect (windowing / QK / softmax / PV), each its own compiled graph on the device vs
    # eager fp32 on CPU. Never allowed to flip the real result above.
    try:
        r["stages"] = _neighborhood_attention_stage_diagnostics(q, k, v, kernel, tile, dev)
    except Exception as exc:
        r["stages_error"] = f"{type(exc).__name__}: {exc}"
    return r


def _neighborhood_attention_stage_diagnostics(q, k, v, kernel, tile, dev) -> dict:
    """Run neighborhood_attention_tiled's four stages (pad+window+collapse, QK matmul, bias+softmax,
    PV matmul) as SEPARATE compiled graphs on the device (bf16 in, fp32 scores) and diff each against
    the same stage eager in fp32 on CPU. Every device graph input is a fresh contiguous base tensor
    (host-built and moved whole, or the previous stage's own output -- the executor allocates those
    contiguous) and every stage output is a fresh tensor (``.clone()`` where the library would return
    a reshape view), per the round-11 rules in :func:`_compile_on`. The CPU side is NEVER compiled with
    the Neuron backend -- that was round 11's segfault.
    """
    import torch

    from vllm_omni_neuron.diffusion.attention.neighborhood_attention import (
        _uncollapse_tiles_adjacent,
        _window_qkv,
        neighborhood_bias,
    )

    n_ax = 2
    axes = tuple(q.shape[1:3])
    halo = tuple(ker - 1 for ker in kernel)
    span = tuple(tile[a] + 2 * halo[a] for a in range(n_ax))
    n_tiles = tuple(-(-axes[a] // tile[a]) for a in range(n_ax))
    heads, d = q.shape[-2], q.shape[-1]
    scale = d**-0.5
    b = q.shape[0]

    def window(q_, k_, v_):
        q_t, k_t, v_t = _window_qkv(q_, k_, v_, n_ax, axes, halo, span, tile, n_tiles, heads, d)
        return q_t.clone(), k_t.clone(), v_t.clone()

    def qk_matmul(q_t, k_t):
        return torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale

    def add_bias_softmax(scores, bias_b):
        return torch.softmax(scores + bias_b, -1)

    def pv_matmul(probs, v_t):
        return torch.matmul(probs, v_t.float())

    def fused_attn(q_, k_, v_, bias_b):
        # window + qk + softmax + pv in ONE graph (what the real op does before uncollapse)
        q_t, k_t, v_t = _window_qkv(q_, k_, v_, n_ax, axes, halo, span, tile, n_tiles, heads, d)
        scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale + bias_b
        return torch.matmul(torch.softmax(scores, dim=-1), v_t.float())

    def uncollapse(out_t):
        return _uncollapse_tiles_adjacent(out_t, n_ax, b, heads, d, n_tiles, tile, axes).clone()

    bias_cpu = neighborhood_bias(axes, kernel, [False, False], tile)  # [T, 1, tt, wt]
    bias_dev = bias_cpu.to(dev).contiguous()

    stage_fns = {}
    for name, f in (
        ("window", window),
        ("qk", qk_matmul),
        ("softmax", add_bias_softmax),
        ("pv", pv_matmul),
        ("fused", fused_attn),
        ("uncollapse", uncollapse),
    ):
        stage_fns[name] = _compile_on(dev, f, f"smoke_na_diag_{name}")

    out = {}
    with torch.no_grad():
        q_cpu, k_cpu, v_cpu = window(q, k, v)
        qd, kd, vd = (x.to(torch.bfloat16).to(dev).contiguous() for x in (q, k, v))
        q_dev, k_dev, v_dev = stage_fns["window"](qd, kd, vd)
        out["q_t_rel_err"] = _rel(q_dev.cpu().float(), q_cpu)
        out["k_t_rel_err"] = _rel(k_dev.cpu().float(), k_cpu)
        out["v_t_rel_err"] = _rel(v_dev.cpu().float(), v_cpu)

        scores_cpu = qk_matmul(q_cpu, k_cpu)
        scores_dev = stage_fns["qk"](q_dev, k_dev)
        out["scores_rel_err"] = _rel(scores_dev.cpu(), scores_cpu)

        probs_cpu = add_bias_softmax(scores_cpu, bias_cpu)
        probs_dev = stage_fns["softmax"](scores_dev, bias_dev)
        out["probs_rel_err"] = _rel(probs_dev.cpu(), probs_cpu)

        out_cpu = pv_matmul(probs_cpu, v_cpu)
        out_dev = stage_fns["pv"](probs_dev, v_dev)
        out["out_rel_err"] = _rel(out_dev.cpu(), out_cpu)
        # cross-check: feed the CPU-windowed operands to the device QK stage, to separate a wrong
        # windowing from a wrong matmul when both stages differ.
        scores_dev_from_cpu_window = stage_fns["qk"](
            q_cpu.to(torch.bfloat16).to(dev).contiguous(),
            k_cpu.to(torch.bfloat16).to(dev).contiguous(),
        )
        out["scores_rel_err_given_cpu_window"] = _rel(scores_dev_from_cpu_window.cpu(), scores_cpu)

        # Round 13: all four stages matched (~0.002-0.003) while the real op was at 0.050, so the
        # excess is either in what the compiler does when the four are FUSED into one graph, or in
        # the one stage the split did not cover -- the uncollapse (reshape + adjacent transposes +
        # crop) after the PV matmul. These two separate them.
        fused_dev = stage_fns["fused"](qd, kd, vd, bias_dev)
        out["fused_attn_rel_err"] = _rel(fused_dev.cpu(), out_cpu)  # window..pv in ONE graph
        unc_cpu = uncollapse(out_cpu)
        out["uncollapse_rel_err_given_cpu_out"] = _rel(
            stage_fns["uncollapse"](out_cpu.to(dev).contiguous()).cpu(), unc_cpu
        )  # uncollapse alone, fed the CPU attention output
        out["uncollapse_rel_err_given_dev_out"] = _rel(
            stage_fns["uncollapse"](fused_dev).cpu(), unc_cpu
        )  # fused + uncollapse = the whole op minus the final bf16 cast
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("SMOKE_OUT", "."))
    ap.add_argument(
        "--force-gen",
        type=int,
        default=None,
        help="force VLLM_OMNI_NEURON_CORE_GEN (2 = NC-v2 paths)",
    )
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument(
        "--only",
        default=None,
        help="run exactly one check by name (e.g. neighborhood_attention), for isolating a crash",
    )
    args = ap.parse_args()
    import faulthandler

    faulthandler.enable()  # a native crash (segfault) prints a Python-frame traceback to stderr
    if args.force_gen is not None:
        os.environ["VLLM_OMNI_NEURON_CORE_GEN"] = str(args.force_gen)
    expect_gen = args.force_gen or 3
    os.makedirs(args.out, exist_ok=True)

    checks = [
        ("detection", lambda: check_detection(expect_gen)),
        ("vae_attn_wan21_d384", lambda: check_vae_attention(384, 32, 1, f"d384_g{expect_gen}")),
        (
            "vae_attn_ti2v_dec_d1024",
            lambda: check_vae_attention(1024, 20, 1, f"d1024_g{expect_gen}"),
        ),
        (
            "vae_attn_ti2v_enc_d640",
            lambda: check_vae_attention(640, 24, 1, f"d640_g{expect_gen}"),
        ),
        # real Wan2.2-TI2V-5B decoder mid-block token count at 704x1280 (44x80 latent)
        (
            "vae_attn_ti2v_dec_d1024_704p",
            lambda: check_vae_attention(1024, (44, 80), 1, f"d1024_704p_g{expect_gen}"),
        ),
    ]
    if not args.skip_vae:
        checks.append(("patchify_compile", check_patchify_compile))
        checks.append(("avg_down_up_3d_compile", check_avg_down_up_3d_compile))
        checks.append(("real_width_vae_encode", check_real_width_vae_encode))
        checks.append(("tiny_ti2v_vae", check_tiny_vae))
        checks.append(("block_runner", check_block_runner))
    checks.append(("decode_attention", check_decode_attention))
    if args.only == "decode_attention_kernel_diag":
        checks.append(("decode_attention_kernel_diag", check_decode_attention_kernel_diag))
    if os.environ.get("VLLM_OMNI_NEURON_DECODE_PREFILL_NKI") == "1":
        # Opt-in, matching the layer's own default-off gate: a known-broken NKI path (round 7-8;
        # see decode_attention.py's module docstring) should never block the main smoke run.
        checks.append(("decode_prefill_nki_opt_in", check_decode_prefill_nki_opt_in))
    checks.append(("neighborhood_attention", check_neighborhood_attention))
    if args.only == "vae_timing" or os.environ.get("SMOKE_VAE_TIMING") == "1":
        checks.append(("vae_timing", check_vae_timing))  # measurement, opt-in (priority 3)
    if args.only == "vae_encode_size_sweep" or os.environ.get("SMOKE_VAE_ENCODE_SWEEP") == "1":
        checks.append(("vae_encode_size_sweep", check_vae_encode_size_sweep))  # opt-in bisection
    if args.only == "vae_tiled_encode_sweep" or os.environ.get("SMOKE_VAE_TILED_SWEEP") == "1":
        checks.append(("vae_tiled_encode_sweep", check_vae_tiled_encode_sweep))  # opt-in
    if args.only == "vae_decode_size_sweep":
        checks.append(("vae_decode_size_sweep", check_vae_decode_size_sweep))  # opt-in bisection

    if args.only is not None:
        checks = [(n, f) for n, f in checks if n == args.only]
        if not checks:
            print(f"ERROR: --only={args.only!r} matched no check", flush=True)
            return 2

    results = {}
    partial_path = os.path.join(args.out, f"smoke_platform_trn2_g{expect_gen}.partial.json")
    for name, fn in checks:
        t0 = time.time()
        try:
            results[name] = fn()
        except Exception as exc:  # report every check, do not stop at the first failure
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-2000:],
            }
        results[name]["wall_s"] = round(time.time() - t0, 1)
        print(
            name, json.dumps({k: v for k, v in results[name].items() if k != "trace"}), flush=True
        )
        # Written after EVERY check (not just at the end): a segfault (e.g. a hard crash inside the
        # Neuron runtime/compiler, which raises no Python exception for the except above to catch)
        # kills the process before the final JSON is ever written, so this partial file is the only
        # forensic record of which check was LAST TO START and never finished -- the next check in
        # `checks` order after the last key present here is the one that crashed the process.
        with open(partial_path, "w") as f:
            json.dump(
                {"completed": list(results.keys()), "results": results}, f, indent=1, default=str
            )

    ok = all(r["ok"] for r in results.values())
    import vllm_omni_neuron as _von

    env = {"module_file": _von.__file__}
    try:
        import subprocess

        env["git_sha"] = subprocess.run(
            ["git", "-C", os.path.dirname(_von.__file__), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:
        env["git_sha_error"] = str(exc)
    summary = {
        "ok": ok,
        "expect_gen": expect_gen,
        "env": env,
        **{k: v["ok"] for k, v in results.items()},
    }
    with open(os.path.join(args.out, f"smoke_platform_trn2_g{expect_gen}.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    if os.path.exists(partial_path):  # full JSON above superseded it; keep only on a crash
        os.remove(partial_path)
    print("SMOKE_SUMMARY " + json.dumps(summary), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
