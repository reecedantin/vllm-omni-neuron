# SPDX-License-Identifier: Apache-2.0
"""Device parity for the Qwen3-VL backbone (Cosmos3-Nano / Super): UND + one GEN call, three-way
(CPU fp32 / CPU bf16 / Neuron bf16), single rank. Three cases: t2i (plain), i2v (clean conditioning
first frame via noisy_mask), action (extra action stream through forward_action; skipped if the
checkpoint has no action head).

Weights: ``COSMOS3_QWEN3_WEIGHTS`` (a real Nano / Super checkout, or a tiny one built by
``vllm_omni_neuron.tiny_models``). Geometry ``COSMOS3_GEN_TEST_GEOM`` = ``t,h,w`` latent units
(default ``1,16,16`` for t2i, ``3,16,16`` for i2v/action). Report JSON goes to ``COSMOS3_TEST_OUT``.
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
COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]


@pytest.fixture(scope="module")
def weights():
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "transformer")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a Cosmos3-Nano / Super (or tiny) checkout")
    return WEIGHTS


def test_qwen3_und_gen_device_parity(vllm_single_rank, weights):
    _run_case(weights, case="t2i")


@pytest.mark.parametrize("case", ["i2v", "action"])
def test_qwen3_und_gen_device_parity_conditioned(vllm_single_rank, weights, case):
    """I2V (clean first frame via noisy_mask) and action (extra action stream) GEN-call parity.
    Skipped automatically if the checkpoint has no action head for the 'action' case."""
    _run_case(weights, case=case)


ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")


@pytest.mark.parametrize("mode", ACTION_MODES)
def test_qwen3_action_mode_device_parity(vllm_single_rank, weights, mode):
    """One GEN call per action mode: three-way on the video output and on the action output."""
    _run_case(weights, case=mode)


def action_case(case: str, t: int, h: int, w: int, action_dim: int) -> dict:
    """GEN-call conditioning of one case. The action modes take their masks from the vendored
    pipeline (``build_vision_condition_mask`` / ``build_action_condition_mask``): policy and
    forward_dynamics condition latent frame 0, inverse_dynamics every latent frame, and
    forward_dynamics every action row. Draws the action input from the global RNG."""
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.action import (
        build_action_condition_mask,
        build_vision_condition_mask,
    )

    hw = (h // 2) * (w // 2)
    nm = torch.ones(1, t * hw, 1)
    spec = {
        "fps": None if case == "t2i" else 24.0,
        "action_fps": None,
        "domain": 0,
        "s_action": 0,
        "act": None,
        "act_mask": None,
    }
    if case == "i2v":
        nm[:, :hw] = 0
    elif case == "action":
        spec.update(
            s_action=4,
            action_fps=12.0,
            act=torch.randn(1, 4, action_dim),
            act_mask=torch.ones(1, 4, 1),
        )
    elif case in ACTION_MODES:
        num_frames, chunk, cpu = (t - 1) * 4 + 1, (t - 1) * 4, torch.device("cpu")
        vis = 1.0 - build_vision_condition_mask(
            case, num_frames, 4, device=cpu, dtype=torch.float32
        )
        nm = vis[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, hw).reshape(1, -1, 1).contiguous()
        am = 1.0 - build_action_condition_mask(case, chunk, device=cpu, dtype=torch.float32)
        spec.update(
            fps=15.0,
            action_fps=15.0,
            domain=8,
            s_action=chunk,
            act=torch.randn(1, chunk, action_dim),
            act_mask=am,
        )
    spec["nm"] = nm
    return spec


def _run_case(weights, case: str):
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    cfg = EdgeGenConfig.from_model_dir(weights)
    assert cfg.backbone in ("qwen3", "nemotron")  # Nano / Super, or Cosmos3-Edge(-Policy) re-checks
    has_action = case == "action" or case in ACTION_MODES
    if has_action and not cfg.action_gen:
        pytest.skip(f"{os.path.basename(weights)} has no action head")
    default_geom = {"t2i": "1,16,16"}.get(case, "5,16,16" if case in ACTION_MODES else "3,16,16")
    t, h, w = (int(x) for x in os.environ.get("COSMOS3_GEN_TEST_GEOM", default_geom).split(","))
    torch.manual_seed(0)
    real, bucket = 37, 64
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    lat = torch.randn(1, 48, t, h, w)
    ts = torch.tensor([500.0])
    s_video = t * (h // 2) * (w // 2)
    spec = action_case(
        case, t, h, w, cfg.action_dim
    )  # i2v: first latent frame is the clean condition
    fps, nm, s_action, act, act_mask = (
        spec["fps"],
        spec["nm"],
        spec["s_action"],
        spec["act"],
        spec["act_mask"],
    )
    s_gen = s_video + s_action
    backend = get_compile_backend_name()
    dev = torch.device("neuron", 0)
    report = {
        "weights": os.path.basename(os.path.normpath(weights)),
        "case": case,
        "geom": [t, h, w],
        "tokens": s_gen,
    }

    # UND: CPU fp32 / bf16 / device bf16
    kv = {}
    for dtype in (torch.float32, torch.bfloat16):
        und = NeuronCosmos3EdgeUND(cfg, dtype=dtype)
        und.load_weights(weights, "cpu")
        cu, su = und.rope_tables(mask)
        with torch.no_grad():
            kv[dtype] = und(ids, cu.to(dtype), su.to(dtype))
        del und
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.bfloat16)
    t0 = time.time()
    und.load_weights(weights, dev)
    report["und_load_s"] = time.time() - t0
    fn = torch.compile(
        und,
        backend=backend,
        fullgraph=True,
        dynamic=False,
        options={"model_name": f"cosmos3_qwen3_und_{bucket}", "compiler_args": COMPILER_ARGS},
    )
    cu, su = und.rope_tables(mask)
    t0 = time.time()
    with torch.no_grad():
        kv_dev = [
            x.cpu()
            for x in fn(ids.to(dev), cu.to(torch.bfloat16).to(dev), su.to(torch.bfloat16).to(dev))
        ]
    report["und_first_s"] = time.time() - t0
    k32 = torch.cat([x[:, :real].flatten() for x in kv[torch.float32]])
    report["und_rel_dev"] = _rel(torch.cat([x[:, :real].flatten() for x in kv_dev]), k32)
    report["und_rel_cpu_bf16"] = _rel(
        torch.cat([x[:, :real].flatten() for x in kv[torch.bfloat16]]), k32
    )
    del und, fn

    # GEN: all three fed the CPU fp32 UND K/V, so the comparison isolates the GEN graph
    def args_for(gen, dtype, device):
        cg, sg = gen.rope_tables(
            mask, t, h, w, fps, t_action=s_action, action_fps=spec["action_fps"]
        )
        kb = gen.key_bias(mask, s_gen)
        common = [lat.to(dtype), ts, cg.to(dtype), sg.to(dtype), kb, nm.to(dtype)]
        if has_action:
            w_in, b_in, w_out, b_out = gen.domain_weights(spec["domain"], "cpu")
            common += [
                act.to(dtype),
                act_mask.to(dtype),
                w_in.to(dtype),
                b_in.to(dtype),
                w_out.to(dtype),
                b_out.to(dtype),
            ]
        a = common + [x.to(dtype) for x in kv[torch.float32]]
        return [x.contiguous().to(device) for x in a]

    def call(fn_, args):
        out = fn_(*args)
        if case == "action":
            return out[0]  # generic action case: video output only
        if has_action:  # action modes: video and action outputs, flattened into one tensor
            return torch.cat([out[0].reshape(-1), out[1].reshape(-1)])
        return out

    outs = {}
    for dtype in (torch.float32, torch.bfloat16):
        gen = NeuronCosmos3EdgeGEN(cfg, dtype=dtype)
        gen.load_weights(weights, "cpu")
        fwd = gen.forward_action if has_action else gen.forward
        with torch.no_grad():
            outs[dtype] = call(fwd, args_for(gen, dtype, "cpu"))
        del gen
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.bfloat16)
    t0 = time.time()
    gen.load_weights(weights, dev)
    report["gen_load_s"] = time.time() - t0
    fwd = gen.forward_action if has_action else gen.forward
    fn = torch.compile(
        fwd,
        backend=backend,
        fullgraph=True,
        dynamic=False,
        options={
            "model_name": f"cosmos3_qwen3_gen_{case}_{t}x{h}x{w}",
            "compiler_args": COMPILER_ARGS,
        },
    )
    a = args_for(gen, torch.bfloat16, dev)
    t0 = time.time()
    with torch.no_grad():
        out = call(fn, a)
        out = out.cpu() if not isinstance(out, tuple) else tuple(o.cpu() for o in out)
    report["gen_first_s"] = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out2 = call(fn, a)
        out2 = out2.cpu() if not isinstance(out2, tuple) else tuple(o.cpu() for o in out2)
    report["gen_warm_s"] = time.time() - t0
    ref32, ref16 = outs[torch.float32], outs[torch.bfloat16]
    report.update(
        rel_dev=_rel(out, ref32),
        rel_cpu_bf16=_rel(ref16, ref32),
        cos_dev=_cos(out, ref32),
        deterministic=bool(torch.equal(out, out2)),
    )
    parts = [("gen", ref32, ref16, out)]
    if case in ACTION_MODES:
        nv = 48 * t * h * w
        parts = [
            ("gen_video", ref32[:nv], ref16[:nv], out[:nv]),
            ("gen_action", ref32[nv:], ref16[nv:], out[nv:]),
        ]
        for name, r32, r16, dv in parts:
            report[f"{name}_rel_dev"], report[f"{name}_rel_cpu_bf16"] = (
                _rel(dv, r32),
                _rel(r16, r32),
            )
            report[f"{name}_cos_dev"] = _cos(dv, r32)
    # Tier 1 gate: the plugin-wide three-way comparison (fp32 CPU baseline, bf16 CPU expected,
    # bf16 Neuron actual) for the per-layer UND K/V and the GEN output.
    from vllm_neuron.accuracy.testing import assert_close_three_way

    und32 = [x[:, :real].float() for x in kv[torch.float32]]
    und16 = [x[:, :real].float() for x in kv[torch.bfloat16]]
    unddv = [x[:, :real].float() for x in kv_dev]
    failures = []
    checks = [("und_kv", (und32, und16, unddv))] + [
        (n, (a.float(), b.float(), c.float())) for n, a, b, c in parts
    ]
    for name, args in checks:
        try:
            r = assert_close_three_way(*args, name=f"{case}_{name}")
            # ThreeWayAssertResult fields are numpy float32 (not JSON serializable): cast explicitly
            report[f"{name}_three_way"] = {
                "passed": True,
                "sigma_ratio": float(r.sigma_ratio),
                "bc": float(r.bc),
                "l2_ratio": float(r.l2_ratio),
                "linf_ratio": float(r.linf_ratio),
            }
        except AssertionError as exc:
            report[f"{name}_three_way"] = {"passed": False, "error": str(exc)[:2000]}
            failures.append(name)
    with open(
        os.path.join(
            os.environ.get("COSMOS3_TEST_OUT", "."),
            f"qwen3_device_{report['weights']}_{case}_{t}x{h}x{w}.json",
        ),
        "w",
    ) as f:
        json.dump(
            report, f, indent=1, allow_nan=False
        )  # a NaN metric is a bug: fail, don't write NaN
    print("[qwen3-device]", json.dumps(report))
    assert not failures, report
    assert report["deterministic"], report
