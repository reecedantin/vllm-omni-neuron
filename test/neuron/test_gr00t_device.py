# SPDX-License-Identifier: Apache-2.0
"""Device parity for GR00T N1.7 on one NeuronCore (three-way: CPU fp32 / CPU bf16 / Neuron bf16),
and the host-vs-device AdaLN-table bit-identity (see ``vllm_omni_neuron.diffusion.layers.modulation_tables``).

Needs ``GR00T_WEIGHTS`` (a local GR00T-N1.7-3B checkout) and a Neuron device; skips cleanly otherwise.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

GR00T_WEIGHTS = os.environ.get("GR00T_WEIGHTS", "")


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


pytestmark = pytest.mark.skipif(
    not (_neuron_available() and os.path.isdir(GR00T_WEIGHTS)),
    reason="needs a Neuron device and GR00T_WEIGHTS",
)


def _synthetic_inputs():
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unit"))
    from test_gr00t_tiny import synthetic_inputs

    return synthetic_inputs()


def test_device_parity_and_adaln_bit_identity():
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    inputs = _synthetic_inputs()
    noise = torch.randn((1, 40, 132), generator=torch.Generator().manual_seed(0))

    refs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = NeuronGr00tModel.from_pretrained(GR00T_WEIGHTS, dtype=dt)
        refs[dt] = m.get_action(inputs, noise=noise)["action_pred"]
        del m

    dev = torch.device("neuron", 0)
    m = NeuronGr00tModel.from_pretrained(GR00T_WEIGHTS, dtype=torch.bfloat16, device=dev)
    m.compile(backend=get_compile_backend_name())
    out = m.get_action(inputs, noise=noise)["action_pred"]
    out2 = m.get_action(inputs, noise=noise)["action_pred"]

    # AdaLN tables (host-baked) must reproduce the device's fp32-accumulate matmul exactly.
    dit = m.head.model
    mismatches = 0
    with torch.no_grad():
        for step, t in enumerate(m.head.timesteps()):
            temb = dit.temb(torch.full((1,), t, dtype=torch.long, device=dev)).to("cpu")
            for i, blk in enumerate(dit.transformer_blocks):
                baked = m.head.adaln_table[step, i].to("cpu")
                x = torch.nn.functional.silu(temb.float()).to(torch.bfloat16).float()
                ref = (
                    (x @ blk.norm1.linear.weight.to("cpu").to(torch.bfloat16).float().t()).to(
                        torch.bfloat16
                    )
                    + blk.norm1.linear.bias.to("cpu").to(torch.bfloat16)
                )[0]
                mismatches += int(not torch.equal(baked, ref))

    ref32, ref16 = refs[torch.float32], refs[torch.bfloat16]
    report = {
        "rel_dev": _rel(out, ref32),
        "rel_cpu_bf16": _rel(ref16, ref32),
        "cos_dev": _cos(out, ref32),
        "deterministic": bool(torch.equal(out, out2)),
        "adaln_table_mismatches": mismatches,
    }
    out_dir = os.environ.get("GR00T_TEST_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "device_parity.json"), "w") as f:
            json.dump(report, f, indent=1)
    assert report["cos_dev"] >= 0.999, report
    assert report["rel_dev"] <= max(2.0 * report["rel_cpu_bf16"], 0.05), report
    assert report["deterministic"], report
    assert report["adaln_table_mismatches"] == 0, report
