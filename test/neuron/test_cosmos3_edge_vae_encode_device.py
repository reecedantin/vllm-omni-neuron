# SPDX-License-Identifier: Apache-2.0
"""Tier 1 accuracy for the Trn2 conditioning-frame encode: the fixed-shape TILED Wan VAE encode
(``device_tiled_encode``, 192 px tiles / 96 px overlap) on the NeuronCore.

Three-way, same algorithm in all three (so the comparison isolates the Neuron error):
fp32 CPU tiled (baseline) / bf16 CPU tiled (expected) / bf16 Neuron tiled with the compiled encoder
(actual), gated with ``assert_close_three_way`` on the latent mean. Separately, the algorithmic cost
of tiling against the untiled fp32 CPU encode (the host-encode path it replaces): latent rel-L2 and
the PSNR between the two latents after an fp32 CPU decode, gated at ``COSMOS3_ENCODE_MIN_PSNR_DB``
(35 dB) and reported next to the VAE's own reconstruction PSNR of the input for scale.

Env: ``COSMOS3_QWEN3_WEIGHTS`` (a Cosmos3 checkout; only ``vae/`` is read), ``COSMOS3_ENCODE_TEST_PX``
(frame size, default 640), ``COSMOS3_ENCODE_TEST_IMAGE`` (optional RGB image, else a smooth
synthetic pattern), ``COSMOS3_TEST_OUT``.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

from .test_cosmos3_edge_und_device import _cos, _neuron_available, _rel

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

WEIGHTS = os.environ.get("COSMOS3_QWEN3_WEIGHTS", "")
SIZE = int(os.environ.get("COSMOS3_ENCODE_TEST_PX", "640"))
TILE, OVERLAP = 192, 96
MIN_PSNR_DB = float(os.environ.get("COSMOS3_ENCODE_MIN_PSNR_DB", "35"))
VAE_COMPILER_ARGS = [
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
]


def _frame(size: int) -> torch.Tensor:
    """One conditioning frame ``[1, 3, 1, size, size]`` in [-1, 1]."""
    path = os.environ.get("COSMOS3_ENCODE_TEST_IMAGE")
    if path:
        import numpy as np
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return t[None, :, None]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    return torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy), torch.sin(xx * yy * 4)])[
        None, :, None
    ]


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR of two images in [-1, 1] (peak-to-peak 2)."""
    mse = ((a.float().clamp(-1, 1) - b.float().clamp(-1, 1)) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def test_tiled_encode_device_three_way(vllm_single_rank):
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "vae")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a Cosmos3 checkout (its vae/ is used)")
    from diffusers import AutoencoderKLWan
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import (
        NeuronEdgeVae,
        device_tiled_encode,
    )

    x = _frame(SIZE)
    zc = 48  # mean channels of the [B, 2*z, ...] moments
    report = {"size_px": SIZE, "tile_px": TILE, "overlap_px": OVERLAP}

    def cpu_vae(dtype):
        return NeuronEdgeVae.from_pretrained(WEIGHTS, torch_dtype=dtype).vae

    with torch.no_grad():
        v32 = cpu_vae(torch.float32)
        base = device_tiled_encode(v32, x, TILE, OVERLAP)[:, :zc].float()
        untiled = v32._encode(x)[:, :zc].float()  # the host-encode path being replaced
        del v32
        expected = device_tiled_encode(
            cpu_vae(torch.bfloat16), x.to(torch.bfloat16), TILE, OVERLAP
        )[:, :zc].float()

        dev = torch.device("neuron", 0)
        vae = NeuronEdgeVae.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
        vae.to(dev)
        vae.compile(get_compile_backend_name(), {}, compile_encoder=True)
        t0 = time.time()
        actual = device_tiled_encode(vae.vae, x.to(torch.bfloat16), TILE, OVERLAP)[:, :zc].float()
        report["device_first_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        actual2 = device_tiled_encode(vae.vae, x.to(torch.bfloat16), TILE, OVERLAP)[:, :zc].float()
        report["device_warm_s"] = round(time.time() - t0, 3)

        ref = AutoencoderKLWan.from_pretrained(
            WEIGHTS, subfolder="vae", torch_dtype=torch.float32
        ).eval()
        dec_untiled = ref.decode(untiled).sample
        dec_device = ref.decode(actual).sample

    report.update(
        latent_shape=list(base.shape),
        rel_cpu_bf16_vs_fp32_tiled=_rel(expected, base),
        rel_device_vs_fp32_tiled=_rel(actual, base),
        cos_device_vs_fp32_tiled=_cos(actual, base),
        rel_tiled_fp32_vs_untiled_fp32=_rel(base, untiled),  # tiling alone (seams)
        rel_device_vs_untiled_fp32=_rel(actual, untiled),  # what replaces the host encode
        decoded_psnr_device_vs_untiled_db=round(
            _psnr(dec_device[:, :, 0], dec_untiled[:, :, 0]), 2
        ),
        vae_recon_psnr_untiled_db=round(
            _psnr(dec_untiled[:, :, 0], x[:, :, 0]), 2
        ),  # the VAE's own error
        deterministic=bool(torch.equal(actual, actual2)),
    )
    try:
        r = assert_close_three_way(base, expected, actual, name=f"tiled_encode_{SIZE}")
        report["three_way"] = {
            "passed": True,
            "sigma_ratio": float(r.sigma_ratio),
            "bc": float(r.bc),
            "l2_ratio": float(r.l2_ratio),
            "linf_ratio": float(r.linf_ratio),
        }
    except AssertionError as exc:
        report["three_way"] = {"passed": False, "error": str(exc)[:2000]}
    with open(
        os.path.join(os.environ.get("COSMOS3_TEST_OUT", "."), f"tiled_encode_device_{SIZE}.json"),
        "w",
    ) as f:
        json.dump(report, f, indent=1, allow_nan=False)
    print("[tiled-encode]", json.dumps(report))
    assert report["three_way"]["passed"], report
    assert report["decoded_psnr_device_vs_untiled_db"] >= MIN_PSNR_DB, report
    assert report["deterministic"], report
