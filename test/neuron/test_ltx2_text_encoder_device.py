# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 text encoder parity on Neuron: the TP-sharded Gemma text tower (``text_encoder.py``)
against transformers' ``Gemma4UnifiedForConditionalGeneration`` on the CPU, three-way on the
49-state hidden stack of the prompt's tokens (diffusers' ``_get_gemma_prompt_embeds``):
CPU fp32 (reference), CPU bf16 (dtype-only band), Neuron bf16. Pass per prompt:
``device rel-L2 <= 2 x (CPU bf16 rel-L2) + 0.005``.

    python -m test.neuron.test_ltx2_text_encoder_device --model <LTX-2.5 dir> --mode reference --out <dir>
    torchrun --nproc_per_node 4 -m test.neuron.test_ltx2_text_encoder_device --model <dir> --mode device \\
        --out <dir>

As a pytest it skips unless ``LTX2_TEXT_PARITY_OUT`` points at a directory with both results.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytest
import torch

PROMPTS = (
    "A red fox walking through a snowy forest at dawn, the camera tracking alongside.",
    "Waves crashing on a rocky shore at sunset, seagulls calling overhead.",
)


def _tokens(model: str, prompt: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(model, "tokenizer"))
    tok.padding_side = "left"
    t = tok([prompt], padding="max_length", max_length=1024, truncation=True, return_tensors="pt")
    return tok, t.input_ids, t.attention_mask


def run_reference(model: str, out: str) -> None:
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    torch.set_num_threads(min(32, os.cpu_count() or 8))
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        te = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            os.path.join(model, "text_encoder"), torch_dtype=dtype
        ).eval()
        for i, p in enumerate(PROMPTS):
            _, ids, mask = _tokens(model, p)
            with torch.no_grad():
                hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            st = torch.stack(hs, dim=-1).flatten(2, 3)[0, mask[0].bool()].float()
            torch.save(st, os.path.join(out, f"ref_{name}_{i}.pt"))
        del te
    print("REFERENCE_DONE", flush=True)


def _pin_rank_core(local_rank: int) -> None:
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if "-" in cores and "," not in cores:
        lo, hi = (int(x) for x in cores.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(x) for x in cores.split(",")] if cores else []
    if local_rank < len(ids):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(ids[local_rank])


def run_device(model: str, out: str) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        _pin_rank_core(local_rank)
    import vllm_omni_neuron  # noqa: F401
    from vllm_omni_neuron.lite_compat import initialize

    initialize()
    torch.nn.functional.gelu = torch.ops.aten.gelu.default
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_neuron.envs import get_compile_backend_name, get_dist_backend

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend=get_dist_backend(),
    )
    initialize_model_parallel(world, 1)
    from vllm_omni_neuron.diffusion.models.ltx2.text_encoder import NeuronGemmaTextEncoder

    t0 = time.time()
    enc = NeuronGemmaTextEncoder(
        os.path.join(model, "text_encoder"), torch.device("neuron", 0), dtype=torch.bfloat16
    )
    load_s = time.time() - t0
    enc.compile(get_compile_backend_name())
    report: dict = {"tp": world, "load_s": round(load_s, 1)}
    report["weights_gib_per_core"] = round(enc.num_local_bytes() / 2**30, 2)
    ok = True
    for i, p in enumerate(PROMPTS):
        tok, _, _ = _tokens(model, p)
        times = []
        for _ in range(3):
            t0 = time.time()
            emb, mask = enc.encode_prompt(tok, p)
            times.append(round(time.time() - t0, 3))
        if rank != 0:
            continue
        dev = emb[0, mask[0].bool()].float()
        r32 = torch.load(os.path.join(out, f"ref_fp32_{i}.pt"))
        r16 = torch.load(os.path.join(out, f"ref_bf16_{i}.pt"))
        rel = float((dev - r32).norm() / r32.norm())
        band = float((r16 - r32).norm() / r32.norm())
        n = r32.shape[0]
        per = [
            float((dev.view(n, -1, 49)[..., k] - r32.view(n, -1, 49)[..., k]).norm())
            / float(r32.view(n, -1, 49)[..., k].norm())
            for k in range(49)
        ]
        bar = 2 * band + 0.005
        report[f"prompt{i}"] = {
            "tokens": n,
            "first_s": times[0],
            "warm_s": times[1:],
            "device_rel_l2": round(rel, 5),
            "cpu_bf16_band": round(band, 5),
            "bar": round(bar, 5),
            "max_layer_rel_l2": round(max(per), 5),
            "pass": rel <= bar,
        }
        ok &= rel <= bar
    if rank == 0:
        report["pass"] = bool(ok)
        with open(os.path.join(out, "text_parity.json"), "w") as f:
            json.dump(report, f, indent=1)
        print("TEXT_PARITY " + json.dumps(report), flush=True)


@pytest.mark.skipif(
    not os.environ.get("LTX2_TEXT_PARITY_OUT"), reason="set LTX2_TEXT_PARITY_OUT to a finished run"
)
def test_text_encoder_parity_device():
    with open(os.path.join(os.environ["LTX2_TEXT_PARITY_OUT"], "text_parity.json")) as f:
        assert json.load(f)["pass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("reference", "device"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (run_reference if a.mode == "reference" else run_device)(a.model, a.out)


if __name__ == "__main__":
    main()
