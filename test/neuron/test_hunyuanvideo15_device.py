# SPDX-License-Identifier: Apache-2.0
"""Device parity for the HunyuanVideo-1.5 DiT and VAE decoder (three-way: CPU fp32 / CPU bf16 /
Neuron bf16), single NeuronCore, TP=1.

Checkpoint: ``HV15_TEST_WEIGHTS`` (default: a tiny random-weight checkpoint generated on the fly by
``test/unit/test_hunyuanvideo15_tiny_ckpt.py``). Geometry ``HV15_TEST_GEOM=t,h,w`` latent units.
Reports go to ``HV15_TEST_OUT`` as JSON. Skipped without a Neuron device.
"""

from __future__ import annotations

import json
import os
import socket
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


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

OUT = os.environ.get("HV15_TEST_OUT", os.environ.get("FLEET_RUNS", "."))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = os.environ.get("HV15_TEST_WEIGHTS")
    if path:
        return path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hv15_tiny",
        os.path.join(os.path.dirname(__file__), "..", "unit", "test_hunyuanvideo15_tiny_ckpt.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_tiny_checkpoint(str(tmp_path_factory.mktemp("hv15_tiny")))


@pytest.fixture(scope="module")
def gloo():
    import torch.distributed as dist

    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    yield


def _report(name, rep):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[hv15-device] {name}: {json.dumps(rep)}", flush=True)


def test_dit_device_parity(weights, gloo):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import (
        HV15Config,
        NeuronHunyuanVideo15DiT,
        prepare_encoder_inputs,
        rope_tables,
    )

    cfg = HV15Config.from_model_dir(weights)
    t, h, w = (int(x) for x in os.environ.get("HV15_TEST_GEOM", "3,4,6").split(","))
    lt, l2 = int(os.environ.get("HV15_TEST_LT", "64")), 256
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, cfg.in_channels, t, h, w, generator=g)
    text = torch.randn(1, lt, cfg.text_embed_dim, generator=g)
    tmask = torch.zeros(1, lt)
    tmask[0, : lt // 2 + 3] = 1
    text2 = torch.randn(1, l2, cfg.text_embed_2_dim, generator=g)
    t2mask = torch.zeros(1, l2)
    t2mask[0, :7] = 1
    ts = torch.tensor([811.0])
    cos, sin = rope_tables(cfg, t, h, w)
    enc_index, text_bias, key_bias = prepare_encoder_inputs(tmask, t2mask, None, t * h * w)

    def args(dtype, dev):
        a = [
            x.to(dtype),
            ts,
            text.to(dtype),
            tmask.to(dtype),
            text2.to(dtype),
            None,
            enc_index,
            cos,
            sin,
            text_bias,
            key_bias,
        ]
        return [None if v is None else v.contiguous().to(dev) for v in a]

    dev = torch.device("neuron", 0)
    m = NeuronHunyuanVideo15DiT(cfg, dtype=torch.bfloat16, tp=(1, 0, None))
    t0 = time.time()
    m.load_weights(weights, "cpu")
    m.to(dev)
    load_s = time.time() - t0
    print("[hv15-device] weights on device", flush=True)
    fn = torch.compile(
        m,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"hv15_dit_test_{t}x{h}x{w}",
            "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"],
        },
    )
    a = args(torch.bfloat16, dev)
    print("[hv15-device] inputs on device", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = fn(*a).cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = fn(*a).cpu()
    warm = time.time() - t0
    outs = {}  # CPU references AFTER the device run (a CPU forward first crashed the device compile)
    for dtype in (torch.float32, torch.bfloat16):
        m = NeuronHunyuanVideo15DiT(cfg, dtype=dtype, tp=(1, 0, None))
        m.load_weights(weights, "cpu")
        with torch.no_grad():
            outs[dtype] = m(*args(dtype, "cpu"))
        del m

    ref32, ref16 = outs[torch.float32], outs[torch.bfloat16]
    rep = {
        "geom": [t, h, w],
        "video_tokens": t * h * w,
        "enc_tokens": int(enc_index.shape[1]),
        "load_s": load_s,
        "first_s": first,
        "warm_s": warm,
        "rel_dev": _rel(out, ref32),
        "rel_cpu_bf16": _rel(ref16, ref32),
        "cos_dev": _cos(out, ref32),
        "deterministic": bool(torch.equal(out, out2)),
        "finite": bool(torch.isfinite(out).all()),
    }
    _report(f"dit_device_{t}x{h}x{w}", rep)
    assert rep["finite"], rep
    assert rep["cos_dev"] >= 0.999, rep
    assert rep["rel_dev"] <= max(2.0 * rep["rel_cpu_bf16"], 0.02), rep


def test_vae_decoder_device_parity(weights):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.vae import NeuronHunyuanVideo15VAE

    t, h, w = (int(x) for x in os.environ.get("HV15_TEST_VAE_GEOM", "3,4,4").split(","))
    z = torch.randn(1, 32, t, h, w, generator=torch.Generator().manual_seed(1))
    outs = {}
    for dtype in (torch.float32, torch.bfloat16):
        v = NeuronHunyuanVideo15VAE.from_pretrained(weights, torch_dtype=dtype)
        with torch.no_grad():
            outs[dtype] = v.decode(z, return_dict=False)[0]
    v = NeuronHunyuanVideo15VAE.from_pretrained(weights, torch_dtype=torch.bfloat16)
    v.to(torch.device("neuron", 0))
    v.compile(get_compile_backend_name(), {})
    t0 = time.time()
    with torch.no_grad():
        out = v.decode(z, return_dict=False)[0]
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        v.decode(z, return_dict=False)
    warm = time.time() - t0
    ref32, ref16 = outs[torch.float32], outs[torch.bfloat16]
    mse = ((out - ref32) ** 2).mean().item()
    rep = {
        "latent": [t, h, w],
        "first_s": first,
        "warm_s": warm,
        "rel_dev": _rel(out, ref32),
        "rel_cpu_bf16": _rel(ref16, ref32),
        "psnr_db_vs_fp32": 10 * torch.log10(torch.tensor(4.0 / mse)).item(),
        "finite": bool(torch.isfinite(out).all()),
    }
    _report(f"vae_device_{t}x{h}x{w}", rep)
    assert rep["finite"], rep
    assert rep["rel_dev"] <= max(2.0 * rep["rel_cpu_bf16"], 0.02), rep


if __name__ == "__main__":
    # Under pytest the DiT device compile segfaults inside the Neuron runtime (a tensor repr ->
    # nrt_tensor_write; not reproducible outside pytest). Run the bodies directly instead:
    #   HV15_TEST_WEIGHTS=<ckpt> python test/neuron/test_hunyuanvideo15_device.py
    import sys

    import torch.distributed as dist

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    w = os.environ["HV15_TEST_WEIGHTS"]
    which = sys.argv[1:] or ["dit", "vae"]
    if "dit" in which:
        test_dit_device_parity(w, None)
    if "vae" in which:
        test_vae_decoder_device_parity(w)
    print("[hv15-device] ALL OK", flush=True)
