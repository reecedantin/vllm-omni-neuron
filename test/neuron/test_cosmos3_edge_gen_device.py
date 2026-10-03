# SPDX-License-Identifier: Apache-2.0
"""Device parity for one GEN-tower call (three-way: CPU fp32 / CPU bf16 / Neuron bf16).

Geometry from ``COSMOS3_GEN_TEST_GEOM`` = ``t,h,w`` in latent units (default ``1,40,40`` =
T2I 640x640, 400 tokens). The UND K/V feeding it come from the CPU fp32 tower for all three
runs, so the comparison isolates the GEN graph.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

from .test_cosmos3_edge_und_device import _cos, _neuron_available, _rel

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


def test_gen_device_parity(vllm_single_rank, edge_weights):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    t, h, w = (int(x) for x in os.environ.get("COSMOS3_GEN_TEST_GEOM", "1,40,40").split(","))
    cfg = EdgeGenConfig.from_model_dir(edge_weights)
    torch.manual_seed(0)
    real, bucket = 37, 64
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    lat = torch.randn(1, 48, t, h, w)
    ts = torch.tensor([500.0])
    fps = None if t == 1 else 24.0

    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(edge_weights, "cpu")
    cu, su = und.rope_tables(mask)
    with torch.no_grad():
        kv32 = und(ids, cu, su)
    del und

    s_gen = t * (h // 2) * (w // 2)
    nm = torch.ones(1, s_gen, 1)

    def run(gen, dtype, dev):
        cg, sg = gen.rope_tables(mask, t, h, w, fps)
        kb = gen.key_bias(mask, s_gen)
        args = [lat.to(dtype), ts, cg.to(dtype), sg.to(dtype), kb, nm.to(dtype), *[x.to(dtype) for x in kv32]]
        return [a.contiguous().to(dev) for a in args]

    outs = {}
    for dtype in (torch.float32, torch.bfloat16):
        gen = NeuronCosmos3EdgeGEN(cfg, dtype=dtype)
        gen.load_weights(edge_weights, "cpu")
        with torch.no_grad():
            outs[dtype] = gen(*run(gen, dtype, "cpu"))
        del gen

    dev = torch.device("neuron", 0)
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.bfloat16)
    gen.load_weights(edge_weights, dev)
    fn = torch.compile(gen, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                       options={"model_name": f"cosmos3_edge_gen_{t}x{h}x{w}",
                                "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"]})
    args = run(gen, torch.bfloat16, dev)
    t0 = time.time()
    with torch.no_grad():
        out = fn(*args).cpu()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = fn(*args).cpu()
    warm = time.time() - t0

    ref32, ref16 = outs[torch.float32], outs[torch.bfloat16]
    report = {"geom": [t, h, w], "tokens": s_gen, "first_s": first, "warm_s": warm,
              "rel_dev": _rel(out, ref32), "rel_cpu_bf16": _rel(ref16, ref32), "cos_dev": _cos(out, ref32),
              "deterministic": bool(torch.equal(out, out2))}
    with open(os.path.join(os.environ.get("COSMOS3_TEST_OUT", "/tmp"), f"gen_device_{t}x{h}x{w}.json"), "w") as f:
        json.dump(report, f, indent=1)
    assert report["cos_dev"] >= 0.999, report
    assert report["rel_dev"] <= max(2.0 * report["rel_cpu_bf16"], 0.02), report
    assert report["deterministic"], report
