# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo-1.5 accuracy on Neuron, three tiers (docs/model-dev/onboarding-models.md, Step 4).

All three tiers use ``vllm_neuron.accuracy.testing.assert_close_three_way``: FP32 CPU baseline
(diffusers ``HunyuanVideo15Transformer3DModel``), BF16 CPU expected (the same diffusers module in
bf16: isolates the dtype error), BF16 Neuron actual (this plugin's DiT on one NeuronCore, TP=1).
The prompt embeddings are computed once in fp32 (host text encoders, bit-equal to diffusers per
``test/unit/test_hunyuanvideo15_pipeline.py``) and cast per variant, so the tiers isolate the DiT.

1. **component** -- one DiT call (cond branch, t=811).
2. **single step** -- one CFG denoising step (cond + uncond DiT calls, CFG 6, Euler flow-match
   step) from the same noise, compared on the denoised latent.
3. **end to end** -- ``HV15_ACC_STEPS`` (default 4) CFG steps, final latents three-way, then every
   variant's latent decoded with the SAME fp32 host VAE: per-frame PSNR/SSIM of Neuron vs FP32 and of
   BF16-CPU vs FP32. The Neuron latent + video are saved as the regression golden.

Shape ``HV15_ACC_SIZE=H,W,F`` (default 256,256,5: latent 2x16x16 = 512 video + 1256 encoder
tokens). Weights ``HV15_ACC_WEIGHTS``. Results JSON in ``HV15_ACC_OUT``.

Run as a script (under pytest the device compile segfaults inside the Neuron runtime, see
``test_hunyuanvideo15_device.py``)::

    python test/neuron/test_hunyuanvideo15_accuracy.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

PROMPT = "A golden retriever runs across a sunlit meadow, slow motion, cinematic."
NEG = ""
GUIDANCE = 6.0


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _psnr(a, b):  # [-1, 1] video
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)


def _ssim(a, b, win=11):
    """Mean SSIM of two [3, H, W] frames in [-1, 1] (gaussian window, per channel)."""
    a, b = (a.float()[None] + 1) / 2, (b.float()[None] + 1) / 2
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2).float() / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    k = g.expand(a.shape[1], 1, win, win)
    f = lambda x: F.conv2d(x, k, groups=a.shape[1])  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    va, vb, cab = f(a * a) - mu_a**2, f(b * b) - mu_b**2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (
        (((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2)))
        .mean()
        .item()
    )


def _three_way(name, base, exp, act, out_dir):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    rec = {
        "rel_dev": _rel(act, base),
        "rel_cpu_bf16": _rel(exp, base),
        "cos_dev": _cos(act, base),
        "finite": bool(torch.isfinite(act).all()),
    }
    try:
        res = assert_close_three_way(
            base, exp, act, name=name, plot_on_failure=True, output_dir=out_dir
        )
        rec.update(passed=True, detail=str(res)[:2000])
    except AssertionError as e:
        rec.update(passed=False, detail=str(e)[:2000])
    print(
        f"[hv15-acc] {name}: {json.dumps({k: v for k, v in rec.items() if k != 'detail'})}",
        flush=True,
    )
    return rec


def main() -> int:
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import local_model_dir

    weights = local_model_dir(
        os.environ.get(
            "HV15_ACC_WEIGHTS", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
        )
    )
    out_dir = os.environ.get("HV15_ACC_OUT", os.path.join(os.environ.get("FLEET_RUNS", "."), "m1"))
    os.makedirs(out_dir, exist_ok=True)
    H, W, NF = (int(x) for x in os.environ.get("HV15_ACC_SIZE", "256,256,5").split(","))
    steps = int(os.environ.get("HV15_ACC_STEPS", "4"))
    os.environ.setdefault("HV15_BLOCKS_PER_GRAPH", "6")
    torch.set_num_threads(int(os.environ.get("HV15_ACC_THREADS", "16")))

    import torch.distributed as dist

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)

    from diffusers import HunyuanVideo15Transformer3DModel
    from vllm_neuron.envs import get_compile_backend_name

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import (
        NeuronHunyuanVideo15Pipeline,
        NeuronHunyuanVideo15Transformer,
    )

    report: dict = {
        "weights": weights,
        "size": [H, W, NF],
        "steps": steps,
        "guidance": GUIDANCE,
        "prompt": PROMPT,
    }
    t_all = time.time()

    # -- shared host inputs (fp32 text encoders, fp32 VAE, scheduler) ----------------------------------
    od32 = SimpleNamespace(
        model=weights,
        dtype=torch.float32,
        model_config={"vae_dtype": "float32"},
        flow_shift=None,
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = NeuronHunyuanVideo15Pipeline(
        od_config=od32
    )  # host encoders + VAE; its DiT stays unloaded
    pipe.transformer = None  # never loaded; the tiers build their own DiTs
    t0 = time.time()
    with torch.no_grad():
        emb = pipe.encode_prompt(
            PROMPT, torch.device("cpu"), torch.float32, NEG, do_classifier_free_guidance=True
        )
    report["encode_s"] = time.time() - t0
    pe, pm, pe2, pm2, ne, nm, ne2, nm2 = emb
    pipe.text_encoder = pipe.text_encoder_2 = None  # free ~30 GB before the DiTs load
    g = torch.Generator().manual_seed(42)
    lat0 = pipe.prepare_latents(1, H, W, NF, torch.float32, torch.device("cpu"), g)
    cond = torch.zeros_like(lat0)
    mask = torch.zeros_like(lat0[:, :1])
    image = torch.zeros(1, 729, 1152)
    print(f"[hv15-acc] latent {tuple(lat0.shape)} encoded in {report['encode_s']:.1f}s", flush=True)

    def kwargs(lat, t, positive, dtype):
        e, m, e2, m2 = (pe, pm, pe2, pm2) if positive else (ne, nm, ne2, nm2)
        x = torch.cat([lat, cond, mask], dim=1).to(dtype)
        return dict(
            hidden_states=x,
            timestep=t.expand(1).to(dtype),
            encoder_hidden_states=e.to(dtype),
            encoder_attention_mask=m.to(dtype),
            encoder_hidden_states_2=e2.to(dtype),
            encoder_attention_mask_2=m2.to(dtype),
            image_embeds=image.to(dtype),
            return_dict=False,
        )

    def sigmas():
        pipe.scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, steps + 1)[:-1], device="cpu")
        return pipe.scheduler.timesteps

    def denoise(model, dtype, n_steps, record=None):
        ts = sigmas()
        lat = lat0.clone().to(dtype)
        for i, t in enumerate(ts[:n_steps]):
            with torch.no_grad():
                pos = model(**kwargs(lat, t, True, dtype))[0].float()
                neg = model(**kwargs(lat, t, False, dtype))[0].float()
            if record is not None and i == 0:
                record["pos0"] = pos
            pred = neg + GUIDANCE * (pos - neg)
            lat = pipe.scheduler.step(pred, t, lat.float(), return_dict=False)[0].to(dtype)
        return lat.float()

    # -- device model first (a CPU forward before the first device compile has crashed it) --------------
    od16 = SimpleNamespace(model=weights, dtype=torch.bfloat16, model_config={}, flow_shift=None)
    dev_model = NeuronHunyuanVideo15Transformer(od16)
    t0 = time.time()
    dev_model.load()
    if (
        os.environ.get("HV15_ACC_DEVICE", "neuron") == "neuron"
    ):  # "cpu" = plumbing dry-run without a device
        dev_model.to(torch.device("neuron", 0))
        dev_model.compile(get_compile_backend_name(), {})
    report["device_load_s"] = time.time() - t0
    runs: dict = {}
    for name in ("neuron",):
        rec: dict = {}
        t0 = time.time()
        ts = sigmas()
        with torch.no_grad():
            comp = dev_model(**kwargs(lat0.to(torch.bfloat16), ts[0], True, torch.bfloat16))[
                0
            ].float()
        report["device_first_call_s"] = time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            dev_model(**kwargs(lat0.to(torch.bfloat16), ts[0], True, torch.bfloat16))
        report["device_warm_call_s"] = time.time() - t0
        rec["component"] = comp
        with torch.no_grad():
            rec["component_neg"] = dev_model(
                **kwargs(lat0.to(torch.bfloat16), ts[0], False, torch.bfloat16)
            )[0].float()
            dev_model._enc_cache.clear()
            neg_fresh = dev_model(**kwargs(lat0.to(torch.bfloat16), ts[0], False, torch.bfloat16))[
                0
            ].float()
            pos_again = dev_model(**kwargs(lat0.to(torch.bfloat16), ts[0], True, torch.bfloat16))[
                0
            ].float()
        report["diag"] = {
            "neg_cached_vs_fresh_rel": _rel(rec["component_neg"], neg_fresh),
            "pos_again_vs_first_rel": _rel(pos_again, comp),
            "neg_vs_pos_rel": _rel(rec["component_neg"], comp),
        }
        print(f"[hv15-acc] diag {json.dumps(report['diag'])}", flush=True)
        rec["step1"] = denoise(dev_model, torch.bfloat16, 1)
        rec["final"] = denoise(dev_model, torch.bfloat16, steps)
        runs[name] = rec
    report["dit_device_stats"] = dict(dev_model.stats)
    del dev_model
    print(
        f"[hv15-acc] device runs done ({report['device_first_call_s']:.1f}s first, "
        f"{report['device_warm_call_s']:.3f}s warm)",
        flush=True,
    )

    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        t0 = time.time()
        ref = HunyuanVideo15Transformer3DModel.from_pretrained(
            os.path.join(weights, "transformer"), torch_dtype=dtype
        ).eval()
        ts = sigmas()
        rec = {}
        with torch.no_grad():
            rec["component"] = ref(**kwargs(lat0.to(dtype), ts[0], True, dtype))[0].float()
            rec["component_neg"] = ref(**kwargs(lat0.to(dtype), ts[0], False, dtype))[0].float()
        rec["step1"] = denoise(ref, dtype, 1)
        rec["final"] = denoise(ref, dtype, steps)
        runs[name] = rec
        report[f"cpu_{name}_s"] = time.time() - t0
        del ref
        print(f"[hv15-acc] cpu {name} done in {report[f'cpu_{name}_s']:.0f}s", flush=True)

    b, e, a = runs["fp32"], runs["bf16"], runs["neuron"]
    report["tier1_component"] = _three_way(
        "hv15_dit_component", b["component"], e["component"], a["component"], out_dir
    )
    report["tier1_component_neg"] = _three_way(
        "hv15_dit_component_neg",
        b["component_neg"],
        e["component_neg"],
        a["component_neg"],
        out_dir,
    )
    report["tier2_single_step"] = _three_way(
        "hv15_single_step", b["step1"], e["step1"], a["step1"], out_dir
    )
    report["tier3_final_latent"] = _three_way(
        "hv15_e2e_latent", b["final"], e["final"], a["final"], out_dir
    )

    # tier 3 pixels: every latent through the SAME fp32 host VAE
    vids = {}
    for name in ("fp32", "bf16", "neuron"):
        with torch.no_grad():
            vids[name] = pipe.decode_latents(runs[name]["final"])[0]  # [3, F, H, W] in [-1, 1]
    nf = vids["fp32"].shape[1]
    for name in ("bf16", "neuron"):
        report[f"tier3_psnr_{name}_vs_fp32"] = [
            _psnr(vids[name][:, i], vids["fp32"][:, i]) for i in range(nf)
        ]
        report[f"tier3_ssim_{name}_vs_fp32"] = [
            _ssim(vids[name][:, i], vids["fp32"][:, i]) for i in range(nf)
        ]
    print(
        f"[hv15-acc] tier3 per-frame PSNR neuron/fp32 {[round(x, 1) for x in report['tier3_psnr_neuron_vs_fp32']]} "
        f"bf16/fp32 {[round(x, 1) for x in report['tier3_psnr_bf16_vs_fp32']]}",
        flush=True,
    )
    print(
        f"[hv15-acc] tier3 per-frame SSIM neuron/fp32 {[round(x, 4) for x in report['tier3_ssim_neuron_vs_fp32']]}",
        flush=True,
    )
    torch.save(
        {"latent": a["final"], "video": vids["neuron"], "size": [H, W, NF], "steps": steps},
        os.path.join(out_dir, "golden_neuron.pt"),
    )
    np.save(
        os.path.join(out_dir, "video_fp32.npy"),
        ((vids["fp32"].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy(),
    )
    np.save(
        os.path.join(out_dir, "video_neuron.npy"),
        ((vids["neuron"].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy(),
    )

    report["total_s"] = time.time() - t_all
    ok = all(
        report[k]["passed"] for k in ("tier1_component", "tier2_single_step", "tier3_final_latent")
    )
    report["all_passed"] = ok
    with open(os.path.join(out_dir, "m1_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(
        "[hv15-acc] SUMMARY "
        + json.dumps(
            {
                "tier1": report["tier1_component"]["passed"],
                "tier2": report["tier2_single_step"]["passed"],
                "tier3_latent": report["tier3_final_latent"]["passed"],
                "rel_dev": [
                    round(report[k]["rel_dev"], 5)
                    for k in ("tier1_component", "tier2_single_step", "tier3_final_latent")
                ],
                "rel_cpu_bf16": [
                    round(report[k]["rel_cpu_bf16"], 5)
                    for k in ("tier1_component", "tier2_single_step", "tier3_final_latent")
                ],
                "psnr_min_neuron": round(min(report["tier3_psnr_neuron_vs_fp32"]), 2),
                "psnr_min_bf16": round(min(report["tier3_psnr_bf16_vs_fp32"]), 2),
                "warm_call_s": round(report["device_warm_call_s"], 4),
                "total_s": round(report["total_s"]),
            }
        ),
        flush=True,
    )
    return 0 if ok else 1


def test_hv15_accuracy_three_tiers():  # collected by pytest only to document the entry point
    import pytest

    pytest.skip(
        "run as a script: python test/neuron/test_hunyuanvideo15_accuracy.py (device compile crashes under pytest)"
    )


if __name__ == "__main__":
    sys.exit(main())
