#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""pi0 family latency levers on one NeuronCore: stage timings (synced) and ``pipeline.forward``
warm medians, toggling one lever at a time on the same loaded pipeline.

pi0.52 levers: the Euler loop (``denoise_mode`` host / device / unrolled), the subtask greedy
pick (host argmax over read-back logits vs in-graph argmax with tokens fed back on the device,
EOS read every ``sync_every`` steps), and one SigLIP pass per request (the action prefix reuses
the subtask decode's image embedding). pi0: the Euler loop. The forward runs with torch pinned
to one host thread, as inside the vLLM-Omni diffusion worker (``host_threads`` lifts it).

    python examples/pi0/latency_levers.py --model <pi052 dir> --out <run dir>
    python examples/pi0/latency_levers.py --model <pi0 dir> --out <run dir>

Prints one final JSON line; the full report is ``<out>/latency_levers.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import numpy as np
import torch


def _bench(fn, repeat: int, warm: int = 2) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(1e3 * (time.perf_counter() - t0))
    return {"median_ms": round(statistics.median(ts), 2), "min_ms": round(min(ts), 2), "n": repeat}


def _rel(a, b) -> float:
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", choices=("neuron", "cpu"), default="neuron")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_neuron.envs import get_compile_backend_name
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi0Pipeline, NeuronPi05Pipeline

    with open(os.path.join(args.model, "config.json")) as f:
        kind = json.load(f).get("type", "pi05")
    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype = args.model, args.dtype
    od.model_config = {"tokenizer": args.tokenizer}
    t0 = time.time()
    pipe = (NeuronPi0Pipeline if kind == "pi0" else NeuronPi05Pipeline)(od_config=od)
    if args.device == "cpu":  # smoke test of this script (tiny checkpoint, eager graphs)
        dev = torch.device("cpu")
    else:
        dev = torch.device("neuron", 0)
        pipe.to(dev)
        pipe.compile(get_compile_backend_name())
    m = pipe.model
    report: dict = {"policy": kind, "load_s": round(time.time() - t0, 1), "stages": {}}
    torch.set_num_threads(1)  # the diffusion worker's setting

    rng = np.random.default_rng(0)
    cams = [k for k in pipe.config.input_features if "image" in k]
    with open(os.path.join(args.model, "config.json")) as f:
        feats = json.load(f).get("input_features", {})
    state_dim = int(feats.get("observation.state", {}).get("shape", [32])[0])
    obs = {"prompt": args.task}
    for k in cams:
        obs[k] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs["state"] = rng.uniform(-1, 1, state_dim).astype(np.float32)
    max_ad = pipe.config.max_action_dim
    noise = torch.from_numpy(rng.standard_normal((1, pipe.config.chunk_size, max_ad))).float()

    def forward():
        sp = types.SimpleNamespace(
            extra_args={"robot_obs": dict(obs), "noise": noise.clone()},
            num_inference_steps=args.steps,
            generator=None,
        )
        out = pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))
        assert out.error is None, out.error
        return out.output["actions"]

    configs = []
    if kind == "pi0":
        for mode in ("host", "device"):  # unrolled: NaN on device for pi0 (see model_pi0)
            configs.append((f"denoise_{mode}", {"denoise": mode}))
    else:
        base = {"denoise": "host", "greedy": "host", "reuse": False, "threads": 0, "tables": False}
        configs = [
            ("before", base),
            ("denoise_device", {**base, "denoise": "device"}),
            ("denoise_unrolled", {**base, "denoise": "unrolled"}),
            ("+greedy_in_graph", {**base, "denoise": "unrolled", "greedy": "device"}),
            ("+reuse_siglip", {**base, "denoise": "unrolled", "greedy": "device", "reuse": True}),
            (
                "+adarms_tables",
                {**base, "denoise": "unrolled", "greedy": "device", "reuse": True, "tables": True},
            ),
            (
                "adarms_tables_denoise_device",
                {**base, "denoise": "device", "greedy": "device", "reuse": True, "tables": True},
            ),
            (
                "+host_threads8",
                {
                    "denoise": "unrolled",
                    "greedy": "device",
                    "reuse": True,
                    "threads": 8,
                    "tables": True,
                },
            ),
        ]

    def apply(c):
        m.denoise_loop.mode = c["denoise"]
        pipe.host_threads = c.get("threads", 0)
        if kind != "pi0":
            m.subtask_host_greedy = c["greedy"] == "host"
            m.subtask_sync_every = c.get("sync", 1)
            pipe.reuse_image_embedding = c["reuse"]
            m.denoise.modulation_tables = c["tables"]

    fwd, actions = {}, {}
    for name, c in configs:
        apply(c)
        t0 = time.time()
        actions[name] = forward()  # first call compiles any new graph
        fwd[name] = {"first_s": round(time.time() - t0, 1), **_bench(forward, args.repeat)}
        print(name, json.dumps(fwd[name]), flush=True)
    ref = actions[configs[0][0]]
    for name in actions:
        fwd[name]["actions_rel_vs_" + configs[0][0]] = _rel(actions[name], ref)
    report["forward"] = fwd

    # Synced stage timings (each call waits for its graph). Mirrors forward()'s inputs.
    st = report["stages"]
    if kind == "pi0":
        from vllm_omni_neuron.diffusion.models.pi0._vendor.pi0.processor_pi0 import (
            build_model_inputs,
        )

        images, masks, lang, lmask, state = build_model_inputs(
            obs, pipe.config, pipe.tokenizer, torch.device("cpu")
        )
        state = m.ref._normalize_state(state)
        ctx_extra = (state.float().to(dev),)
        tc = m.time_sincos(args.steps, 1)
    else:
        images, masks, _, _ = pipe.processor.build_model_inputs(obs)
        lang, lmask = pipe._build_prompt({**obs, "prompt": "pick up the red cube"})
        ctx_extra = ()
        m.denoise.modulation_tables = True
        tc = m.time_conds(args.steps, 1)
    from vllm_omni_neuron.diffusion.models.pi0.subtask import _camera_stack

    pix = _camera_stack(images, m.vision_dtype, dev)
    iv = torch.stack([x.float() for x in masks], 1).to(dev)
    tok, tv = lang.long().to(dev), lmask.float().to(dev)

    def prefix():
        return m._prefix_fn(pix, iv, tok, tv)

    st["prefix_ms"] = _bench(lambda: prefix()[2].to("cpu"), args.repeat)
    k, v, valid = prefix()
    if kind != "pi0":
        st["embed_images_ms"] = _bench(lambda: m._embed_images_fn(pix).to("cpu"), args.repeat)
        emb = m._embed_images_fn(pix)
        st["prefix_from_emb_ms"] = _bench(
            lambda: m._prefix_from_emb_fn(emb, iv, tok, tv)[2].to("cpu"), args.repeat
        )
    ctx = (*ctx_extra, k, v, valid)
    for mode in ("host", "device", "unrolled"):
        m.denoise_loop.mode = mode
        st[f"denoise_{mode}_ms"] = _bench(
            lambda: m.denoise_loop.run(noise, tc, ctx, dev), args.repeat
        )
    x0, c0 = noise.to(dev), tc[0].contiguous().to(dev)
    st["denoise_one_step_graph_ms"] = _bench(
        lambda: m._denoise_fn(*ctx_extra[:1], x0, c0, k, v, valid).to("cpu"), args.repeat
    )
    m.denoise_loop.mode = "host"
    a_host = m.denoise_loop.run(noise, tc, ctx, dev)
    m.denoise_loop.mode = "unrolled"
    a_unr = m.denoise_loop.run(noise, tc, ctx, dev)
    st["denoise_unrolled_vs_host_rel"] = _rel(a_unr, a_host)
    m.denoise_loop.mode = "device"
    st["denoise_device_vs_host_rel"] = _rel(m.denoise_loop.run(noise, tc, ctx, dev), a_host)
    if kind != "pi0":  # the same schedule without the AdaRMS tables (dense(cond) per step)
        m.denoise.modulation_tables = False
        tc_raw = m.time_conds(args.steps, 1)
        for mode in ("host", "unrolled"):
            m.denoise_loop.mode = mode
            st[f"denoise_{mode}_no_tables_ms"] = _bench(
                lambda: m.denoise_loop.run(noise, tc_raw, ctx, dev), args.repeat
            )
        st["denoise_tables_vs_no_tables_rel"] = _rel(
            a_unr, m.denoise_loop.run(noise, tc_raw, ctx, dev)
        )
        m.denoise.modulation_tables = True

    if kind != "pi0":
        sub = {}
        for label, host, sync in (
            ("host_greedy", True, 1),
            ("graph_greedy", False, 1),
            ("graph_greedy_sync4", False, 4),
        ):
            m.subtask_host_greedy, m.subtask_sync_every = host, sync
            for n in (1, 16):
                sub[f"{label}_{n}tok_ms"] = _bench(
                    lambda n=n: m.generate_subtask(
                        images,
                        masks,
                        args.task,
                        max_new_tokens=n,
                        min_new_tokens=n,
                        img_emb=emb,
                    ),
                    args.repeat,
                )
            sub[f"{label}_per_token_ms"] = round(
                (sub[f"{label}_16tok_ms"]["median_ms"] - sub[f"{label}_1tok_ms"]["median_ms"]) / 15,
                3,
            )
            _, ids = m.generate_subtask(
                images, masks, args.task, max_new_tokens=16, min_new_tokens=16, return_ids=True
            )
            sub[f"{label}_ids16"] = ids
            text, ids_nat = m.generate_subtask(
                images, masks, args.task, max_new_tokens=48, return_ids=True
            )
            sub[f"{label}_natural"] = {"text": text, "ids": ids_nat}
        sub["ids_identical"] = (
            sub["host_greedy_ids16"] == sub["graph_greedy_ids16"] == sub["graph_greedy_sync4_ids16"]
            and sub["host_greedy_natural"]
            == sub["graph_greedy_natural"]
            == sub["graph_greedy_sync4_natural"]
        )
        m.subtask_host_greedy, m.subtask_sync_every = False, 1
        st["subtask"] = sub

    report["ok"] = bool(np.isfinite(np.asarray(ref)).all()) and all(
        v["actions_rel_vs_" + configs[0][0]] < 0.02 for v in fwd.values()
    )
    if kind != "pi0":
        report["ok"] = report["ok"] and st["subtask"]["ids_identical"]
    with open(os.path.join(args.out, "latency_levers.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
