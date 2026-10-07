# SPDX-License-Identifier: Apache-2.0
"""InternVLA-A1.5 device accuracy, in the three tiers of the onboarding guide (Step 4).

1. **Component** -- each compiled graph (vision tower + prefix embedding assembly, one fused run
   of 3 Gated-DeltaNet layers, one gated full-attention layer, one action-expert denoise step)
   against the same computation on the
   CPU, three-way via ``vllm_neuron.accuracy.testing.assert_close_three_way``: FP32 CPU baseline,
   BF16 CPU expected (dtype error alone), BF16 Neuron actual (adds the Neuron-specific error).
   Every graph gets the identical FP32-CPU-derived inputs, so each comparison isolates one graph.
2. **Single step** -- the whole policy (vision + prefix + the first Euler step of the action
   expert) three-way, same method, wider scope.
3. **Golden actions** -- the full 10-step action chunk on device against a cached golden: upstream
   ``InternVLAA15.sample_actions`` in FP32 on the CPU (``test/unit/test_internvla_a15_make_golden_helper.py``
   writes it), gated on cosine and relative L2 within 2x the CPU BF16-vs-FP32 error (floors: cos >= 0.999,
   rel <= 2%).

Weights: ``INTERNVLA_A15_WEIGHTS`` (checkpoint dir) + ``INTERNVLA_VLM_CONFIG`` (the Qwen3.5-2B
``config.json``); without them the tests run on a generated tiny random-weight checkpoint (plumbing
and graph parity only). Golden: ``INTERNVLA_A15_GOLDEN``; without it tier 3 compares against this
port's own FP32 CPU run, which ``test/unit/test_internvla_a15_cpu.py`` ties to upstream (1e-6).
The golden request is ``preprocess.synthetic_request(n_images=3, grid=(16, 16), text_before=14,
text_after=90, seed=0)``, the defaults of ``examples/internvla/run.py`` and the golden helper.
``INTERNVLA_A15_REQUEST`` replaces it with a ``torch.save``'d request dict -- e.g. a real
observation built by the served path (``pipeline_internvla._robot_obs_to_batch``, real
tokenizer); pass the golden made from the same file (helper ``--batch``).
Reports go to ``$INTERNVLA_TEST_OUT`` (default: the pytest tmp dir). Skipped without a Neuron device.
"""

from __future__ import annotations

import json
import os

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

COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]
REQUEST = dict(n_images=3, grid=(16, 16), text_before=14, text_after=90, seed=0)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    d = os.environ.get("INTERNVLA_TEST_OUT") or str(tmp_path_factory.mktemp("internvla_device"))
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = os.environ.get("INTERNVLA_A15_WEIGHTS")
    if path:
        return path, os.environ.get("INTERNVLA_VLM_CONFIG")
    from ..unit.test_internvla_a15_tiny_helper import make_tiny_checkpoint

    tiny = make_tiny_checkpoint(str(tmp_path_factory.mktemp("internvla_tiny")), seed=0)
    return tiny, os.path.join(tiny, "vlm", "config.json")


@pytest.fixture(scope="module")
def runners(ckpt):
    """(fp32 CPU, bf16 CPU, bf16 Neuron) runners. The Neuron graphs use the production
    ``model_name``s, so a warm NEFF cache from ``examples/internvla/run.py`` is reused."""
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.internvla import InternVLAA15, InternVLAA15Runner

    path, vlm = ckpt
    backend = get_compile_backend_name()

    def wrap(mod, name):
        return torch.compile(
            mod,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options={"model_name": name, "compiler_args": list(COMPILER_ARGS)},
        )

    dev = torch.device("neuron", 0)
    r32 = InternVLAA15Runner(
        InternVLAA15.from_pretrained(path, dtype=torch.float32, vlm_config=vlm)
    )
    r16 = InternVLAA15Runner(
        InternVLAA15.from_pretrained(path, dtype=torch.bfloat16, vlm_config=vlm)
    )
    rdev = InternVLAA15Runner(
        InternVLAA15.from_pretrained(path, dtype=torch.bfloat16, device=dev, vlm_config=vlm),
        dev,
        wrap,
    )
    return r32, r16, rdev


@pytest.fixture(scope="module")
def request_tables(runners):
    from vllm_omni_neuron.diffusion.models.internvla import preprocess as pp

    r32 = runners[0]
    path = os.environ.get("INTERNVLA_A15_REQUEST")
    batch = torch.load(path) if path else pp.synthetic_request(r32.cfg, **REQUEST)
    noise = pp.initial_noise(r32.cfg, seed=REQUEST["seed"])
    return batch, noise, r32.prepare(batch)


def _image_slots(img, h, dtype):
    slots = torch.zeros(h["batch"], h["length"], img.shape[-1], dtype=dtype)
    img = img.reshape(h["batch"], h["m_img"], -1).to(dtype)
    for i in range(h["batch"]):
        slots[i].index_copy_(0, h["img_pos"][i], img[i])
    return slots


def _fp32_reference_inputs(r32, h):
    """FP32-CPU intermediates every component test feeds to all three runs."""
    with torch.inference_mode():
        img = r32.vision(h["patches"].float(), h["pe_idx"], h["pe_w"], h["vcos"], h["vsin"])
        slots = _image_slots(img, h, torch.float32)
        lm = r32.model.vlm.language_model
        x = lm.embed_tokens(h["input_ids"])
        m = h["image_mask"].to(x.dtype)
        x = x * (1.0 - m) + slots * m
        ks, vs = r32.prefix(h["input_ids"], slots, h["image_mask"], h["cos"], h["sin"], h["bias"])
    return img, x, ks, vs


def _three_way(name, base, exp, act, report):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    report[name] = {
        "rel_cpu_bf16": _rel(exp, base),
        "rel_dev": _rel(act, base),
        "cos_dev": _cos(act, base),
        "linf_cpu_bf16": (exp.float() - base.float()).abs().max().item(),
        "linf_dev": (act.float() - base.float()).abs().max().item(),
    }
    try:
        res = assert_close_three_way(base.float(), exp.float(), act.float(), name=name)
    except AssertionError as exc:
        report[name]["failure"] = str(exc)[-2000:]
        raise
    report[name].update(sigma_ratio=float(res.sigma_ratio), bc=float(res.bc))


@torch.inference_mode()
def test_tier1_components(runners, request_tables, out_dir):
    report: dict = {}
    try:
        _tier1(runners, request_tables, report)
    finally:
        with open(os.path.join(out_dir, "tier1_components.json"), "w") as f:
            json.dump(report, f, indent=1)


def _tier1(runners, request_tables, report):
    from vllm_omni_neuron.diffusion.models.internvla.policy import SplitPrefix

    r32, r16, rdev = runners
    _, noise, h = request_tables
    d = rdev._dev
    img32, x32, ks32, vs32 = _fp32_reference_inputs(r32, h)

    # vision tower + prefix embedding assembly (one graph on device: VisionEmbedGraph); the CPU runs
    # use the reference path (vision graph, host scatter, embedding select)
    xs = []
    for r, dt in ((r32, torch.float32), (r16, torch.bfloat16)):
        img = r.vision(h["patches"].to(dt), h["pe_idx"], h["pe_w"], h["vcos"], h["vsin"])
        slots = _image_slots(img, h, dt)
        m = h["image_mask"].to(dt)
        xs.append(r.model.vlm.language_model.embed_tokens(h["input_ids"]) * (1.0 - m) + slots * m)
    st = rdev._device_tables(request_tables[0], None)
    xs.append(
        rdev.vision_embed(
            d(h["patches"].to(torch.bfloat16)),
            st["pe_idx"],
            st["pe_w"],
            st["vcos"],
            st["vsin"],
            st["text_emb"],
            st["place"],
        ).cpu()
    )
    _three_way("vision_embed", *xs, report)

    # first fused run of 3 Gated-DeltaNet layers (one BlockGraphRunner graph on device)
    sp: SplitPrefix = rdev.prefix
    start, run = sp.lin_runs[0]
    layers32 = r32.model.vlm.language_model.layers[start : start + run.group_size]
    layers16 = r16.model.vlm.language_model.layers[start : start + run.group_size]
    gdn = []
    for layers, dt in ((layers32, torch.float32), (layers16, torch.bfloat16)):
        x = x32.to(dt)
        for ly in layers:
            x = ly.forward_linear(x)
        gdn.append(x)
    gdn.append(run(d(x32.to(torch.bfloat16))).cpu())
    _three_way("gdn_run0", *gdn, report)

    # first gated full-attention layer, input = the FP32 output of the layers before it
    li = sp.full[0]
    x_in = x32
    for ly in r32.model.vlm.language_model.layers[:li]:
        x_in = ly.forward_linear(x_in)
    full_out, full_k = [], []
    for r, dt in ((r32, torch.float32), (r16, torch.bfloat16)):
        ly = r.model.vlm.language_model.layers[li]
        xo, (k, _v) = ly.forward_full(x_in.to(dt), h["cos"].to(dt), h["sin"].to(dt), h["bias"])
        full_out.append(xo)
        full_k.append(k)
    xo, k, _v = sp.fullg(
        d(x_in.to(torch.bfloat16)),
        d(h["cos"].to(torch.bfloat16)),
        d(h["sin"].to(torch.bfloat16)),
        d(h["bias"]),
        sp._full_weights(rdev.model.vlm.language_model.layers[li]),
    )
    full_out.append(xo.cpu())
    full_k.append(k.cpu())
    _three_way("full_layer_out", *full_out, report)
    _three_way("full_layer_k", *full_k, report)

    # one action-expert denoise step, prefix K/V from the FP32 CPU prefix; each run gets the time
    # embedding computed in its own model dtype, as upstream does (timestep cast before the sinusoid)
    temb16 = r16.prepare(request_tables[0])["temb"][0]
    den = []
    for r, dt, te in ((r32, torch.float32, h["temb"][0]), (r16, torch.bfloat16, temb16)):
        den.append(
            r.denoise(
                noise.float(),
                te,
                h["dt"],
                h["scos"],
                h["ssin"],
                h["sbias"],
                ks32.to(dt),
                vs32.to(dt),
            )
        )
    st = rdev._device_tables(request_tables[0], None)
    den.append(
        rdev.denoise(
            d(noise.float(), exact=True),
            d(temb16),
            st["dt"],
            st["scos"],
            st["ssin"],
            st["sbias"],
            d(ks32.to(torch.bfloat16)),
            d(vs32.to(torch.bfloat16)),
        ).cpu()
    )
    _three_way("denoise_step", *den, report)


@torch.inference_mode()
def test_tier2_single_step(runners, request_tables, out_dir):
    r32, r16, rdev = runners
    batch, noise, _ = request_tables
    first = [r.sample_actions(batch, noise, return_trajectory=True)[1][0] for r in (r32, r16, rdev)]
    report: dict = {}
    try:
        _three_way("policy_step1", *first, report)
    finally:
        with open(os.path.join(out_dir, "tier2_single_step.json"), "w") as f:
            json.dump(report, f, indent=1)


@torch.inference_mode()
def test_tier3_golden_actions(runners, request_tables, out_dir):
    r32, r16, rdev = runners
    batch, noise, _ = request_tables
    act32, act16 = r32.sample_actions(batch, noise), r16.sample_actions(batch, noise)
    out = rdev.sample_actions(batch, noise)
    again = rdev.sample_actions(batch, noise)
    golden_path = os.environ.get("INTERNVLA_A15_GOLDEN")
    if golden_path:
        g = torch.load(golden_path)
        golden, source = g["actions_fp32"], "upstream_fp32"
        band = _rel(g["actions_bf16"], golden) if "actions_bf16" in g else _rel(act16, golden)
    else:
        golden, source, band = act32, "port_fp32_cpu", _rel(act16, act32)
    report = {
        "golden": source,
        "rel_dev": _rel(out, golden),
        "cos_dev": _cos(out, golden),
        "rel_bf16_band": band,
        "cos_cpu_bf16": _cos(act16, golden),
        "deterministic": bool(torch.equal(out, again)),
        "finite": bool(torch.isfinite(out).all()),
    }
    with open(os.path.join(out_dir, "tier3_golden_actions.json"), "w") as f:
        json.dump(report, f, indent=1)
    torch.save(out, os.path.join(out_dir, "tier3_device_actions.pt"))
    assert report["finite"] and report["deterministic"], report
    # device error at most 2x the CPU BF16 error, floored at the absolute gates (cos >= 0.999,
    # rel <= 2%); the floors bind on real weights, the 2x band on the noisy tiny checkpoint
    assert 1.0 - report["cos_dev"] <= max(2.0 * (1.0 - report["cos_cpu_bf16"]), 1e-3), report
    assert report["rel_dev"] <= max(2.0 * band, 0.02), report
