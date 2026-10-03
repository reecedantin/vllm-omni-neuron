# SPDX-License-Identifier: Apache-2.0
"""Device parity for the Cosmos3-Edge UND tower (three-way, per the onboarding guide):

FP32 CPU baseline vs BF16 CPU (dtype error alone) vs BF16 Neuron compiled (adds the
Neuron-specific error). The device run must stay within ~2x of the pure-bf16 error.
Skipped automatically without a Neuron device.
"""

from __future__ import annotations

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


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

BUCKET = 64


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_und_device_parity(vllm_single_rank, edge_weights):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import (
        EdgeTextConfig,
        NeuronCosmos3EdgeUND,
    )

    cfg = EdgeTextConfig.from_model_dir(edge_weights)
    torch.manual_seed(0)
    real = 37
    ids = torch.zeros(1, BUCKET, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, BUCKET, dtype=torch.long)
    mask[0, :real] = 1

    def cpu_run(dtype):
        m = NeuronCosmos3EdgeUND(cfg, dtype=dtype)
        m.load_weights(edge_weights, "cpu")
        cos, sin = m.rope_tables(mask)
        with torch.no_grad():
            return m(ids, cos.to(dtype), sin.to(dtype))

    ref32 = cpu_run(torch.float32)
    ref16 = cpu_run(torch.bfloat16)

    dev = torch.device("neuron", 0)
    m = NeuronCosmos3EdgeUND(cfg, dtype=torch.bfloat16)
    m.load_weights(edge_weights, dev)
    cos, sin = m.rope_tables(mask)
    fn = torch.compile(m, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                       options={"model_name": "cosmos3_edge_und", "compiler_args": [
                           "--model-type=transformer", "--auto-cast=none", "-O1"]})
    args = (ids.to(dev), cos.to(torch.bfloat16).to(dev), sin.to(torch.bfloat16).to(dev))
    t0 = time.time()
    with torch.no_grad():
        out = [t.cpu() for t in fn(*args)]
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = [t.cpu() for t in fn(*args)]
    warm = time.time() - t0

    n = cfg.num_layers
    report = {"first_call_s": round(first, 1), "warm_call_s": round(warm, 4), "layers": []}
    worst = 0.0
    for i in range(2 * n):
        r_dev = _rel(out[i][:, :real], ref32[i][:, :real])
        r_bf = _rel(ref16[i][:, :real], ref32[i][:, :real])
        c = _cos(out[i][:, :real], ref32[i][:, :real])
        report["layers"].append({"t": ("k" if i < n else "v") + str(i % n), "rel_dev": r_dev, "rel_bf16": r_bf, "cos": c})
        worst = max(worst, r_dev)
        assert c >= 0.999, f"tensor {i}: cosine {c}"
        assert r_dev <= max(2.0 * r_bf, 0.01), f"tensor {i}: device rel {r_dev} vs cpu-bf16 rel {r_bf}"
        assert torch.equal(out[i], out2[i]), f"tensor {i}: non-deterministic across calls"
    report["worst_rel_dev"] = worst
    out_dir = os.environ.get("COSMOS3_TEST_OUT", "/tmp")
    with open(os.path.join(out_dir, "und_device_parity.json"), "w") as f:
        json.dump(report, f, indent=1)
