# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 / FastH3 device accuracy, in the three tiers of ``docs/model-dev/accuracy-evaluation-debugging.md``.

1. **Component three-way** (``assert_close_three_way``: FP32 CPU / BF16 CPU / BF16 Neuron) for one DiT forward,
   dense and VSA-H3, and for one video-VAE decoder clip.
2. **Single step**: the first denoising step's velocity of the full pipeline on the NeuronCore vs the same pipeline
   on CPU in fp32, against the CPU-bf16 floor.
3. **End to end** against an independent reference: final latents vs the bar, the decoded videos (one CPU fp32
   VAE for all) by per-frame SSIM, and repeatability (two device runs bit-equal). On the tiny checkpoints the
   reference is the plugin pipeline on CPU in fp32 (floor: bf16). On real weights (``test_tier3_real_weights``) it is
   diffusers' transformer on CPU (``examples/minimax_h3/eval/reference_cpu.py``, fp32 and the bf16 floor), compared
   with saved device runs listed in ``MINIMAX_H3_T3_CASES`` (a JSON list of ``{name, model_path, ref, floor, run,
   run2}``) -- a 33B TP=8 device run needs the multi-process Omni path, so the job makes it with ``run.py``.

Checkpoints: ``MINIMAX_H3_TINY`` (dense structure checkpoint, ``test/unit/test_minimax_h3_tiny_ckpt.py``) and
``MINIMAX_H3_TINY_VSA`` (the same plus ``to_gate_compress`` weights and a VSA contract,
``test/unit/test_minimax_h3_vsa.py::_vsa_tiny``). The parity bar for tiers 1-2 is the fleet bar: device-vs-fp32
error <= 2 x the CPU-bf16-vs-fp32 error + 0.5 %. Skipped without a Neuron device.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

GEOM = tuple(int(x) for x in os.environ.get("MINIMAX_H3_TEST_GEOM", "128,192,22,24").split(","))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _bar(floor: float) -> float:
    return 2.0 * floor + 0.005


def _weights(env: str) -> str:
    w = os.environ.get(env, "")
    if not os.path.isdir(os.path.join(w, "transformer")):
        pytest.skip(f"set {env}")
    return w


def _report(name: str, rep: dict) -> None:
    out = os.environ.get("MINIMAX_H3_TEST_OUT", os.environ.get("TMPDIR", "."))
    with open(os.path.join(out, f"{name}.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep, default=str))


def _compile(module_or_fn, name: str):
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(
        module_or_fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": name,
            "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
        },
    )


# ---------------------------------------------------------------------------------------------------- tier 1
@pytest.mark.parametrize("variant", ["dense", "vsa"])
def test_tier1_dit_three_way(vllm_single_rank, variant):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    from vllm_omni_neuron.diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig, vsa_sparsity
    from vllm_omni_neuron.diffusion.models.minimax_h3.layout import build_layout, draw_noise
    from vllm_omni_neuron.diffusion.models.minimax_h3.transformer import NeuronMiniMaxH3Transformer

    weights = _weights("MINIMAX_H3_TINY" if variant == "dense" else "MINIMAX_H3_TINY_VSA")
    sparsity = vsa_sparsity(weights) if variant == "vsa" else None
    h, w, f, n_text = GEOM
    tdir = os.path.join(weights, "transformer")
    cfg = MiniMaxH3DiTConfig.from_dir(tdir)
    layout = build_layout(n_text, h, w, f)
    g = torch.Generator().manual_seed(0)
    video_rows, audio_rows = draw_noise(layout, g)
    text = torch.randn(1, n_text, cfg.text_dim, generator=g)
    cos, sin = layout.rotary(cfg.rope_freq_dim, cfg.rope_theta)
    ts = torch.tensor([0.4375, 0.8125])
    p = cfg.patch_size
    grid = (
        layout.num_latent_frames // p[0],
        layout.latent_height // p[1],
        layout.latent_width // p[2],
    )

    def build(dtype, dev):
        m = NeuronMiniMaxH3Transformer(cfg, dtype=dtype, vsa_sparsity=sparsity)
        m.load_weights(tdir, dev)
        m.set_layout(layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows, grid)
        a = [
            x.contiguous().to(dev)
            for x in (text.to(dtype), audio_rows[None], video_rows[None], ts, cos, sin)
        ]
        return m, a

    outs = {}
    for dtype in (torch.float32, torch.bfloat16):
        m, a = build(dtype, "cpu")
        with torch.no_grad():
            outs[dtype] = [o.float() for o in m(*a)]
        del m
    m, a = build(torch.bfloat16, torch.device("neuron", 0))
    fn = _compile(m, f"minimax_h3_dit_{variant}_{h}x{w}x{f}_{n_text}")
    t0 = time.time()
    with torch.no_grad():
        dev_out = [o.cpu().float() for o in fn(*a)]
    first = time.time() - t0
    (v32, a32), (v16, a16), (vd, ad) = outs[torch.float32], outs[torch.bfloat16], dev_out
    rep = {
        "variant": variant,
        "first_s": first,
        "tokens": layout.sequence_length,
        "video": {"rel_dev": _rel(vd, v32), "rel_cpu_bf16": _rel(v16, v32)},
        "audio": {"rel_dev": _rel(ad, a32), "rel_cpu_bf16": _rel(a16, a32)},
    }
    _report(f"tier1_dit_{variant}", rep)
    for k in ("video", "audio"):
        assert rep[k]["rel_dev"] <= _bar(rep[k]["rel_cpu_bf16"]), rep
    if (
        variant == "dense"
    ):  # VSA top-k flips make the error distribution bimodal; the L2 bar above is the gate
        assert_close_three_way([v32, a32], [v16, a16], [vd, ad], name=f"minimax_h3_dit_{variant}")


def test_tier1_video_vae_three_way():
    from vllm_neuron.accuracy.testing import assert_close_three_way

    from vllm_omni_neuron.diffusion.models.minimax_h3._vendor.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )
    from vllm_omni_neuron.diffusion.models.minimax_h3.vae import NeuronMiniMaxH3VideoVAE

    weights = _weights("MINIMAX_H3_TINY")

    def load(dt):
        return AutoencoderKLMiniMaxH3.from_pretrained(
            weights, subfolder="vae", torch_dtype=dt
        ).eval()

    v32 = load(torch.float32)
    z = torch.randn(
        1, v32.config.latent_channels, 4, 8, 12, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        base = v32.float().decode(z, return_dict=False)[0].float()
        exp = (
            NeuronMiniMaxH3VideoVAE(load(torch.bfloat16), torch.device("cpu"), torch.bfloat16)
            .decode(z, return_dict=False)[0]
            .float()
        )
        vd = NeuronMiniMaxH3VideoVAE(
            load(torch.bfloat16), torch.device("neuron", 0), torch.bfloat16
        )
        act = vd.decode(z, return_dict=False)[0].float()
    rep = {
        "rel_dev": _rel(act, base),
        "rel_cpu_bf16": _rel(exp, base),
        "tile_calls": getattr(vd, "tile_calls", None),
    }
    _report("tier1_video_vae", rep)
    assert rep["rel_dev"] <= _bar(rep["rel_cpu_bf16"]), rep
    assert_close_three_way(base, exp, act, name="minimax_h3_video_vae")


# ---------------------------------------------------------------------------------------------------- tiers 2/3
def _pipe(weights, dtype, device):
    from vllm_omni_neuron.diffusion.models.minimax_h3 import NeuronMiniMaxH3Pipeline

    p = NeuronMiniMaxH3Pipeline(
        od_config=SimpleNamespace(model=weights, dtype=dtype, model_config={})
    )
    p.load_weights()
    if device != "cpu":
        p.to(torch.device(device))
        p.compile()
    return p


def _generate(p, tmp_path, tag, decode=False):
    h, w, f, _ = GEOM
    dump = str(tmp_path / f"{tag}.pt")
    os.environ["MINIMAX_H3_DUMP_LATENTS"] = dump
    try:
        out = p.generate(
            "A golden retriever runs through the surf.",
            h,
            w,
            f,
            p.default_steps,
            seed=0,
            return_latents=not decode,
        )
    finally:
        os.environ.pop("MINIMAX_H3_DUMP_LATENTS", None)
    return out, torch.load(dump, weights_only=False)


@pytest.mark.parametrize("env", ["MINIMAX_H3_TINY", "MINIMAX_H3_TINY_VSA"])
def test_tier2_single_step(vllm_single_rank, env, tmp_path):
    weights = _weights(env)
    _, ref = _generate(_pipe(weights, torch.float32, "cpu"), tmp_path, "fp32")
    _, flo = _generate(_pipe(weights, torch.bfloat16, "cpu"), tmp_path, "bf16")
    _, dev = _generate(_pipe(weights, torch.bfloat16, "neuron:0"), tmp_path, "dev")
    s0 = lambda d, k: d["steps"][0][k]  # noqa: E731
    rep = {"weights": env}
    for k in ("vel_video", "vel_audio"):
        rep[k] = {
            "rel_dev": _rel(s0(dev, k), s0(ref, k)),
            "rel_cpu_bf16": _rel(s0(flo, k), s0(ref, k)),
        }
    rep["final_video"] = {
        "rel_dev": _rel(dev["video_rows"], ref["video_rows"]),
        "rel_cpu_bf16": _rel(flo["video_rows"], ref["video_rows"]),
    }
    _report(f"tier2_{env.lower()}", rep)
    for k in ("vel_video", "vel_audio"):
        assert rep[k]["rel_dev"] <= _bar(rep[k]["rel_cpu_bf16"]), rep


def _tier3_mod():
    import importlib.util

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "minimax_h3", "eval", "tier3_compare.py"
    )
    spec = importlib.util.spec_from_file_location("minimax_h3_tier3_compare", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("env", ["MINIMAX_H3_TINY", "MINIMAX_H3_TINY_VSA"])
def test_tier3_e2e_tiny(vllm_single_rank, env, tmp_path):
    """Tiny checkpoints: device vs the plugin pipeline on CPU fp32 (floor: CPU bf16), plus repeatability."""
    t3 = _tier3_mod()
    weights = _weights(env)
    _, ref = _generate(_pipe(weights, torch.float32, "cpu"), tmp_path, "fp32")
    _, flo = _generate(_pipe(weights, torch.bfloat16, "cpu"), tmp_path, "bf16")
    dev_pipe = _pipe(weights, torch.bfloat16, "neuron:0")
    _, run = _generate(dev_pipe, tmp_path, "dev1")
    _, run2 = _generate(dev_pipe, tmp_path, "dev2")
    res = t3.tier3(weights, ref, flo, run, run2)
    _report(f"tier3_{env.lower()}", res)
    assert res["repeatable"], res
    assert res["latents_pass"], res
    assert res["ssim"]["pass"], res


def test_tier3_real_weights():
    """Real checkpoints at 256p: saved device runs vs diffusers' transformer on CPU (fp32 reference, bf16 floor)."""
    cases_file = os.environ.get("MINIMAX_H3_T3_CASES", "")
    if not os.path.isfile(cases_file):
        pytest.skip("set MINIMAX_H3_T3_CASES (see the module docstring)")
    t3 = _tier3_mod()
    results, failed = {}, []
    for c in json.load(open(cases_file)):
        load = t3._load
        r = t3.tier3(
            c["model_path"],
            load(c["ref"]),
            load(c["floor"]),
            load(c["run"]),
            load(c["run2"]) if c.get("run2") else None,
        )
        results[c["name"]] = r
        if not r["pass"]:
            failed.append(c["name"])
    _report("tier3_real_weights", results)
    assert not failed, {k: results[k] for k in failed}
