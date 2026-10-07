# SPDX-License-Identifier: Apache-2.0
"""Device parity + timing for one Neuron LTX-2.5 DiT call (three-way, per the onboarding guide).

CPU fp32 port (oracle; == upstream to ~1e-7, see test/unit/test_ltx2_transformer.py) vs CPU bf16
port (dtype noise alone) vs Neuron bf16 compiled. Run as a pytest (needs a Neuron device and
``LTX2_TEST_WEIGHTS``) or directly::

    python -m test.neuron.test_ltx2_transformer_device --model <dir> --geom 4,8,8,32,128 --out <dir>

``--geom`` = latent ``frames,height,width,audio_frames,text_tokens``. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import time

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


def _rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def _cos(a, b) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    )


def _inputs(cfg: dict, geom, batch=1, seed=0, sigma=0.6):
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import first_frame_keyframes_mask

    f, h, w, na, st = geom
    n = f * h * w
    d = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    da = cfg["audio_num_attention_heads"] * cfg["audio_attention_head_dim"]
    g = torch.Generator().manual_seed(seed)
    ts = torch.full((batch,), sigma * 1000.0)
    return dict(
        hidden_states=torch.randn(batch, n, cfg["in_channels"], generator=g),
        audio_hidden_states=torch.randn(batch, na, cfg["audio_in_channels"], generator=g),
        encoder_hidden_states=torch.randn(batch, st, d, generator=g),
        audio_encoder_hidden_states=torch.randn(batch, st, da, generator=g),
        timestep=ts,
        sigma=ts,
        num_frames=f,
        height=h,
        width=w,
        fps=24.0,
        audio_num_frames=na,
        use_cross_timestep=True,
        video_keyframes_mask=first_frame_keyframes_mask(batch, n, f),
    )


def _init_device_runtime() -> None:
    os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
    os.environ.setdefault("VLLM_NEURON_LIBTORCH_NEURONX_LITE", "1")
    os.environ.setdefault("VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND", "1")
    import vllm_omni_neuron  # noqa: F401  (bootstrap)
    from vllm_omni_neuron.lite_compat import initialize

    initialize()
    # Lite wraps F.gelu with a builtin Dynamo cannot trace under fullgraph (the worker does the same).
    torch.nn.functional.gelu = torch.ops.aten.gelu.default


def run(
    model: str, geom, blocks_per_graph: int = 2, out_dir: str | None = None, cpu_ref: bool = True
) -> dict:
    _init_device_runtime()
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import NeuronLTX2Transformer

    tdir = os.path.join(model, "transformer")
    with open(os.path.join(tdir, "config.json")) as f:
        cfg = json.load(f)
    inputs = _inputs(cfg, geom)
    report: dict = {
        "model": os.path.basename(os.path.normpath(model)),
        "geom": list(geom),
        "video_tokens": geom[0] * geom[1] * geom[2],
        "blocks_per_graph": blocks_per_graph,
    }
    refs = {}
    if cpu_ref:
        for dt in (torch.float32, torch.bfloat16):
            m = NeuronLTX2Transformer.from_dir(tdir, dtype=dt, blocks_per_graph=blocks_per_graph)
            with torch.no_grad():
                refs[dt] = m(**inputs)
            del m

    dev = torch.device("neuron", 0)
    t0 = time.time()
    m = NeuronLTX2Transformer.from_dir(
        tdir, dtype=torch.bfloat16, blocks_per_graph=blocks_per_graph, device=dev
    )
    report["load_s"] = round(time.time() - t0, 2)
    report["local_params_m"] = round(m.num_local_params() / 1e6, 2)
    m.compile(get_compile_backend_name())
    t0 = time.time()
    with torch.no_grad():
        v, a = m(**inputs)
        v, a = v.cpu(), a.cpu()
    report["first_call_s"] = round(time.time() - t0, 2)
    times = []
    for _ in range(3):
        t0 = time.time()
        with torch.no_grad():
            v2, a2 = m(**inputs)
            v2, a2 = v2.cpu(), a2.cpu()
        times.append(time.time() - t0)
    report["warm_s"] = round(min(times), 4)
    report["deterministic"] = bool(torch.equal(v, v2) and torch.equal(a, a2))
    report["finite"] = bool(torch.isfinite(v).all() and torch.isfinite(a).all())
    if cpu_ref:
        r32, r16 = refs[torch.float32], refs[torch.bfloat16]
        report.update(
            rel_dev_video=_rel(v, r32[0]),
            rel_dev_audio=_rel(a, r32[1]),
            cos_dev_video=_cos(v, r32[0]),
            cos_dev_audio=_cos(a, r32[1]),
            rel_cpu_bf16_video=_rel(r16[0], r32[0]),
            rel_cpu_bf16_audio=_rel(r16[1], r32[1]),
        )
        report["pass"] = bool(
            report["finite"]
            and report["deterministic"]
            and report["rel_dev_video"] <= max(2.0 * report["rel_cpu_bf16_video"], 0.02)
            and report["rel_dev_audio"] <= max(2.0 * report["rel_cpu_bf16_audio"], 0.02)
        )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = "x".join(str(x) for x in geom)
        with open(os.path.join(out_dir, f"dit_device_{report['model']}_{tag}.json"), "w") as f:
            json.dump(report, f, indent=1)
    return report


@pytest.mark.skipif(
    not _neuron_available() or not os.environ.get("LTX2_TEST_WEIGHTS"),
    reason="needs a Neuron device and LTX2_TEST_WEIGHTS (tiny or real LTX-2.5 dir)",
)
def test_dit_device_parity():
    geom = tuple(int(x) for x in os.environ.get("LTX2_TEST_GEOM", "4,8,8,32,128").split(","))
    report = run(os.environ["LTX2_TEST_WEIGHTS"], geom, out_dir=os.environ.get("LTX2_TEST_OUT"))
    assert report["pass"], report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--geom", action="append", default=None, help="f,h,w,audio,text (repeatable)")
    ap.add_argument("--blocks-per-graph", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cpu-ref", action="store_true")
    a = ap.parse_args()
    ok = True
    for g in a.geom or ["4,8,8,32,128"]:
        geom = tuple(int(x) for x in g.split(","))
        try:
            rep = run(a.model, geom, a.blocks_per_graph, a.out, cpu_ref=not a.no_cpu_ref)
        except Exception as exc:  # noqa: BLE001  report and keep the other geometries
            rep = {"geom": list(geom), "error": repr(exc)[:2000], "pass": False}
        ok &= bool(rep.get("pass", True))
        print("RESULT " + json.dumps(rep), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
