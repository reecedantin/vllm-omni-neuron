# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the LTX-2.5 full-size gate's comparison (test/neuron/test_ltx2_fullsize_gate_device.py)
on synthetic outputs: per-frame bars, the waveform / latent tiers, the rank-agreement requirement and
the latent capture hook."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
gate = pytest.importorskip("test.neuron.test_ltx2_fullsize_gate_device")


def _write(out, dev_noise: float, bad_frame: int | None = None, ranks_pass: bool = True):
    rng = np.random.default_rng(0)
    f32 = rng.integers(0, 256, size=(5, 32, 48, 3)).astype(np.float64)

    def perturb(x, s):
        return np.clip(x + rng.normal(0, s, x.shape), 0, 255).round().astype(np.uint8)

    f16, dev = perturb(f32, 4.0), perturb(f32, dev_noise)
    if bad_frame is not None:
        dev[bad_frame] = 255 - dev[bad_frame]
    np.save(out / "ref_fp32_frames_u8.npy", f32.astype(np.uint8))
    np.save(out / "ref_bf16_frames_u8.npy", f16)
    np.save(out / "dev_frames_u8.npy", dev)
    for name, shape in (
        ("wav", (2, 1000)),
        ("latent_video", (1, 8, 2, 4, 6)),
        ("latent_audio", (1, 8, 5, 4)),
    ):
        r = rng.normal(size=shape).astype(np.float32)
        np.save(out / f"ref_fp32_{name}.npy", r)
        np.save(out / f"ref_bf16_{name}.npy", r + 0.01 * rng.normal(size=shape).astype(np.float32))
        np.save(out / f"dev_{name}.npy", r + 0.012 * rng.normal(size=shape).astype(np.float32))
    ranks = {"bit_identical": 4, "max_rel": 0.0 if ranks_pass else 1.0, "pass": ranks_pass}
    (out / "device.json").write_text(json.dumps({"ranks": ranks}))


def test_gate_passes_within_band(tmp_path):
    _write(tmp_path, dev_noise=5.0)
    r = gate.compare(str(tmp_path))
    assert r["pass"], gate._summary(r)
    assert len(r["frames"]) == 5 and not r["frames_failed"]
    assert r["ssim_min"] > 0.9 and r["psnr_min"] > 30
    assert (tmp_path / "gate.json").exists() and (tmp_path / "contact.png").exists()


def test_gate_fails_one_bad_frame(tmp_path):
    _write(tmp_path, dev_noise=5.0, bad_frame=3)
    r = gate.compare(str(tmp_path))
    assert not r["pass"] and r["frames_failed"] == [3]


def test_gate_needs_ranks_to_agree(tmp_path):
    _write(tmp_path, dev_noise=5.0, ranks_pass=False)
    assert not gate.compare(str(tmp_path))["pass"]


def test_ssim_identity_and_psnr():
    a = np.random.default_rng(1).integers(0, 256, size=(16, 16, 3)).astype(np.uint8)
    assert gate._ssim(a, a) == pytest.approx(1.0)
    assert gate._psnr(a, a) == float("inf")


def test_capture_latents_records_last_call():
    class P:
        @staticmethod
        def _unpack_latents(x):
            return x * 2

        @staticmethod
        def _unpack_audio_latents(x):
            return x + 1

    p = P()
    seen = gate.capture_latents(p)
    p._unpack_latents(torch.ones(2))
    p._unpack_latents(torch.full((2,), 3.0))
    p._unpack_audio_latents(torch.zeros(3))
    assert torch.equal(seen["video"], torch.full((2,), 6.0))
    assert torch.equal(seen["audio"], torch.ones(3))


def test_rank_agreement_single_rank():
    r = gate._rank_agreement({"video": torch.zeros(2)})
    assert r["pass"] and r["bit_identical"] == 1 and r["max_rel"] == 0.0
