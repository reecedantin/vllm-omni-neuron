# SPDX-License-Identifier: Apache-2.0
"""Tier-1 component accuracy: Wan2.2 DiT, one teacher-forced forward, three-way vs diffusers.

Follows docs/model-dev/accuracy-evaluation-debugging.md (Step 3 / Step 5): FP32 CPU reference,
BF16 CPU reference (expected low-precision behavior) and BF16 Neuron (target), compared with
``vllm_neuron.accuracy.testing.assert_close_three_way``.

The Neuron leg comes from a served run (produced by the test itself unless ``WAN22_NEURON_DUMP``
points at an existing dump): ``examples/wan2_2/run_ti2v.py --steps 1 --guidance-scale 1
--dump-noise-pred <dump.pt>`` writes, from rank 0, the first DiT call's exact inputs
(``hidden_states``, ``timestep``, ``encoder_hidden_states``) and its noise prediction. Running the
two CPU legs on those same inputs makes the comparison teacher-forced: all three legs see the same
latent, timestep and prompt conditioning (the served worker does the Lite / collective setup that a
standalone device forward cannot).

    WAN22_NEURON_DUMP=<dump.pt> WAN22_MODEL=<checkpoint dir> \\
        pytest test/neuron/test_wan2_2_dit_accuracy.py

The checks also run as a script (``python test/neuron/test_wan2_2_dit_accuracy.py <dump> <model>``)
and print the three-way report plus rel-L2 / cosine of each leg against FP32.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest
import torch

DUMP = os.environ.get("WAN22_NEURON_DUMP", "")
MODEL = os.environ.get("WAN22_MODEL", "")


def cpu_reference(model_path: str, inputs: dict, dtype: torch.dtype) -> torch.Tensor:
    """Diffusers' own WanTransformer3DModel on the dumped inputs (loaded fresh per dtype)."""
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    ).eval()
    with torch.inference_mode():
        out = model(
            hidden_states=inputs["hidden_states"].to(dtype),
            timestep=inputs["timestep"].to(dtype),
            encoder_hidden_states=inputs["encoder_hidden_states"].to(dtype),
            return_dict=False,
        )[0]
    out = out.float().cpu()
    del model
    gc.collect()
    return out


def _three_way_result(fp32, bf16, neuron, name):
    """``assert_close_three_way``'s report, returned (pass or fail) instead of raised on a fail."""
    from vllm_neuron.accuracy.testing import assert_close_three_way

    try:
        return assert_close_three_way(fp32, bf16, neuron, name=name)
    except AssertionError as e:
        print(e)
        return type("FailedThreeWay", (), {"passed": False, "name": name, "report": str(e)})()


def _rel_cos(a: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    rel = float((a - ref).norm() / ref.norm())
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), ref.flatten(), 0))
    return rel, cos


def _legs_for_call(model_path: str, call: dict):
    fp32 = cpu_reference(model_path, call, torch.float32)
    bf16 = cpu_reference(model_path, call, torch.bfloat16)
    return fp32, bf16, call["noise_pred"]


def _report(name: str, fp32, bf16, neuron):
    for leg_name, leg in (("bf16_cpu", bf16), ("bf16_neuron", neuron)):
        rel, cos = _rel_cos(leg, fp32)
        print(f"[{name}] {leg_name} vs fp32_cpu: rel_l2={rel:.4f} cos={cos:.5f}")
    rel, cos = _rel_cos(neuron, bf16)
    print(f"[{name}] bf16_neuron vs bf16_cpu: rel_l2={rel:.4f} cos={cos:.5f}")
    return _three_way_result(fp32, bf16, neuron, name)


def three_way(dump_path: str, model_path: str):
    """Three-way per dumped DiT call (call 0 = positive, call 1 = negative under CFG) and, when
    the dump holds a CFG combine, for the combined prediction too (references combined in FP32
    with the diffusers formula ``neg + g * (pos - neg)``)."""
    dump = torch.load(dump_path)
    calls = dump.get("calls") or [dump]
    results, legs = [], []
    for i, call in enumerate(calls):
        fp32, bf16, neuron = _legs_for_call(model_path, call)
        legs.append((fp32, bf16, neuron))
        results.append(_report(f"wan22_transformer_call{i}", fp32, bf16, neuron))
    if "combined" in dump and len(legs) >= 2:
        g = float(dump["guidance_scale"])
        (p32, p16, _), (n32, n16, _) = legs[0], legs[1]
        ref32 = n32 + g * (p32 - n32)
        ref16 = (n16.bfloat16() + g * (p16.bfloat16() - n16.bfloat16())).float()
        results.append(_report(f"wan22_cfg_combine_g{g:g}", ref32, ref16, dump["combined"]))
        # Neuron combine re-done in fp32 from its own branch outputs: isolates combine arithmetic.
        pos_n, neg_n = calls[0]["noise_pred"], calls[1]["noise_pred"]
        rel, cos = _rel_cos(dump["combined"], neg_n + g * (pos_n - neg_n))
        print(
            "[combine] neuron combined vs fp32 combine of neuron branches: "
            f"rel_l2={rel:.4f} cos={cos:.5f}"
        )
    return results


def compare_dumps(a_path: str, b_path: str):
    """Same served code on two backends (e.g. device vs CPU eager): inputs, embeds and outputs
    per call, plus the combine. Points at the first boundary that differs."""
    a, b = torch.load(a_path), torch.load(b_path)
    for i, (ca, cb) in enumerate(zip(a.get("calls") or [a], b.get("calls") or [b])):
        for k in ("hidden_states", "timestep", "encoder_hidden_states", "noise_pred"):
            if k in ca and k in cb:
                rel, cos = _rel_cos(ca[k].float(), cb[k].float())
                print(f"call{i} {k:22s} rel_l2={rel:.4f} cos={cos:.5f} shape={list(ca[k].shape)}")
    if "combined" in a and "combined" in b:
        rel, cos = _rel_cos(a["combined"], b["combined"])
        print(f"combined               rel_l2={rel:.4f} cos={cos:.5f}")


@pytest.fixture(scope="module")
def served_dump(tmp_path_factory):
    """The served first-step dump: ``WAN22_NEURON_DUMP`` if given, else produced now by a served
    one-step CFG run (``run_ti2v.py --dump-noise-pred``) on ``WAN22_MODEL``."""
    if os.path.isfile(DUMP):
        return DUMP
    if not os.path.isdir(os.path.join(MODEL, "transformer")):
        pytest.skip("set WAN22_MODEL (and optionally WAN22_NEURON_DUMP)")
    from test.neuron.test_wan2_2_ti2v_pipeline_accuracy import initial_latents, served_latent

    work = tmp_path_factory.mktemp("dit_dump")
    noise_path = str(work / "noise.pt")
    initial_latents(MODEL, noise_path)
    dump = str(work / "dump.pt")
    guidance = float(os.environ.get("WAN22_GUIDANCE", "5"))
    served_latent(MODEL, noise_path, str(work / "served"), guidance, ("--dump-noise-pred", dump))
    return dump


def test_dit_three_way_from_served_dump(served_dump):
    """Per DiT call (positive, negative) and for the CFG combine."""
    results = three_way(served_dump, MODEL)
    assert results and all(r.passed for r in results), [r.name for r in results if not r.passed]


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare_dumps(sys.argv[2], sys.argv[3])
    else:
        for r in three_way(sys.argv[1], sys.argv[2]):
            print(r)
