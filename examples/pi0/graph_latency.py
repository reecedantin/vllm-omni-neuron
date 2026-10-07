#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""M3: per-graph warm-latency breakdown for the pi0.52 pipeline on one NeuronCore.

Measures, warm (after compile + one throwaway call each):
  * action prefix graph (Pi05PrefixGraph)      -- images + 200-tok prompt -> per-layer K/V
  * action denoise graph (Pi05DenoiseGraph)     -- one flow step, x `num_steps`
  * subtask text-prefix graph (Pi05TextPrefixGraph), one re-prefill call at a representative
    bucket, x `subtask_new_tokens` (the no-KV-cache decode cost)
  * full pipeline.forward() end to end (subtask decode + action sample)

    python examples/pi0/graph_latency.py --model <pi052 checkpoint dir> --out runs/m3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import types

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch


def _timeit(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.time()
        fn()
        out.append(time.time() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.environ.get("WEIGHTS", ""), "paligemma-3b-pt-224")
    )
    ap.add_argument("--task", default="pick up the red cube and place it in the bowl")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--subtask-new-tokens", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_neuron.envs import get_compile_backend_name
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.pi0 import NeuronPi05Pipeline

    od = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    for f in dataclasses.fields(OmniDiffusionConfig):
        setattr(od, f.name, None)
    od.model, od.dtype, od.model_config = args.model, "bfloat16", {}
    od.model_config = {"tokenizer": args.tokenizer}

    report: dict = {}
    pipe = NeuronPi05Pipeline(od_config=od)
    dev = torch.device("neuron", 0)
    pipe.to(dev)
    pipe.compile(get_compile_backend_name())
    m = pipe.model

    g = torch.Generator().manual_seed(0)
    r = pipe.config.image_resolution[0]
    cams = [k for k in pipe.config.input_features if "image" in k]
    pix = torch.rand(len(cams), 3, r, r, generator=g).to(m.vision_dtype).to(dev)
    img_valid = torch.ones(1, len(cams)).to(dev)
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang[0, :32] = torch.randint(2, 250000, (32,), generator=g)
    lang, lang_valid = lang.to(dev), (lang != 0).float().to(dev)
    noise = torch.randn(1, pipe.config.chunk_size, pipe.config.max_action_dim, generator=g)

    # Action prefix graph, warm.
    def run_prefix():
        with torch.no_grad():
            k, v, valid = m._prefix_fn(pix, img_valid, lang, lang_valid)
            k.to("cpu")  # sync

    run_prefix()  # discard first (compile already happened via pipe.compile(), but keep for safety)
    report["prefix_warm_s"] = _timeit(run_prefix, args.repeat)

    # One K/V to feed denoise.
    with torch.no_grad():
        k, v, valid = m._prefix_fn(pix, img_valid, lang, lang_valid)
    x_t = noise.to(dev)
    tc = m.time_conds(args.steps, 1)[0].to(dev)

    def run_denoise():
        with torch.no_grad():
            out = m._denoise_fn(x_t, tc, k, v, valid)
            out.to("cpu")

    run_denoise()
    report["denoise_step_warm_s"] = _timeit(run_denoise, args.repeat)
    report["denoise_full_schedule_s_est"] = [v * args.steps for v in report["denoise_step_warm_s"]]

    # Subtask text-prefix graph: one re-prefill call at a representative bucket.
    bucket = m.subtask_gen.buckets[1] if m.subtask_gen else 64
    text_tokens = torch.zeros(1, bucket, dtype=torch.long)
    text_tokens[0, :40] = torch.randint(2, 250000, (40,), generator=g)
    text_valid = (text_tokens != 0).float().to(dev)
    text_tokens = text_tokens.to(dev)
    text_pix = torch.rand(len(cams), 3, r, r, generator=g).to(m.vision_dtype).to(dev)

    def run_text_prefix():
        with torch.no_grad():
            out = m._text_prefix_fn(text_pix, img_valid, text_tokens, text_valid)
            out[:, -1:].to("cpu")

    run_text_prefix()
    report["subtask_text_prefix_uncached_warm_s"] = _timeit(run_text_prefix, args.repeat)
    report["subtask_decode_uncached_s_est"] = [
        v * args.subtask_new_tokens for v in report["subtask_text_prefix_uncached_warm_s"]
    ]

    # Image-prefix-cached path: one embed_images call (amortized over the whole decode) + N cheaper
    # re-prefills over the precomputed image embedding (no vision tower per token).
    def run_embed_images():
        with torch.no_grad():
            out = m._embed_images_fn(text_pix)
            out.to("cpu")

    run_embed_images()
    report["embed_images_warm_s"] = _timeit(run_embed_images, args.repeat)
    with torch.no_grad():
        img_emb = m._embed_images_fn(text_pix)

    def run_text_prefix_cached():
        with torch.no_grad():
            out = m._text_prefix_cached_fn(img_emb, img_valid, text_tokens, text_valid)
            out[:, -1:].to("cpu")

    run_text_prefix_cached()
    report["subtask_text_prefix_cached_warm_s"] = _timeit(run_text_prefix_cached, args.repeat)
    n = args.subtask_new_tokens
    report["subtask_decode_cached_s_est"] = [
        min(report["embed_images_warm_s"]) + v * n
        for v in report["subtask_text_prefix_cached_warm_s"]
    ]
    report["subtask_cache_speedup_x"] = min(report["subtask_decode_uncached_s_est"]) / min(
        report["subtask_decode_cached_s_est"]
    )

    # Full pipeline.forward(), warm (subtask decode + action sample).
    obs = {
        "prompt": args.task,
        **{k: (torch.rand(r, r, 3, generator=g)).numpy() for k in cams},
        "state": (torch.rand(pipe.config.state_dim, generator=g) * 2 - 1).numpy(),
    }
    sp = types.SimpleNamespace(
        extra_args={"robot_obs": dict(obs)},
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(1),
    )

    def run_forward():
        pipe.forward(types.SimpleNamespace(sampling_params=sp, prompts=[]))

    run_forward()  # first call already warm-ish (graphs compiled); discard
    report["pipeline_forward_warm_s"] = _timeit(run_forward, args.repeat)

    for k in (
        "prefix_warm_s",
        "denoise_step_warm_s",
        "subtask_text_prefix_uncached_warm_s",
        "subtask_text_prefix_cached_warm_s",
        "embed_images_warm_s",
        "pipeline_forward_warm_s",
    ):
        vals = report[k]
        report[k + "_min"] = min(vals)
        report[k + "_median"] = sorted(vals)[len(vals) // 2]

    report["ok"] = True
    with open(os.path.join(args.out, "graph_latency.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
