"""Microbenchmark one Cosmos3-Edge GEN forward at I2V 480p/121f geometry on ONE NeuronCore.

    NEURON_RT_VISIBLE_CORES=0 COSMOS3_EDGE_ATTN_QBLOCK=512 python bench_gen.py --mode gen --opt O1
    NEURON_RT_VISIBLE_CORES=1 python bench_gen.py --mode attn          # attention alone, one layer

Prints a JSON line: mode, qblock, opt, tokens, first_s, warm_s (median of 5).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch

W = os.environ.get("COSMOS3_EDGE_WEIGHTS", "nvidia/Cosmos3-Edge")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gen", "attn"], default="gen")
    p.add_argument("--opt", default="O1")
    p.add_argument("--geom", default="31,30,52")
    p.add_argument("--text", type=int, default=512)
    p.add_argument("--extra", default="", help="extra compiler args, space separated")
    p.add_argument("--skip-attn", action="store_true", help="replace attention by identity (cost attribution only)")
    a = p.parse_args()
    from vllm_neuron.envs import get_compile_backend_name

    t, h, w = (int(x) for x in a.geom.split(","))
    s_gen = t * (h // 2) * (w // 2)
    dev = torch.device("neuron", 0)
    bf = torch.bfloat16
    cargs = ["--model-type=transformer", "--auto-cast=none", f"-{a.opt}", *a.extra.split()]
    q = os.environ.get("COSMOS3_EDGE_ATTN_QBLOCK", "512")
    name = f"bench4s_{a.mode}{os.environ.get('BENCH_TAG', '')}{'_noattn' if a.skip_attn else ''}_{s_gen}_q{q}_{a.opt}{'_x' if a.extra else ''}"

    if a.mode == "attn":
        from vllm_omni_neuron.diffusion.models.cosmos3_edge.attention import edge_attention

        qq = torch.randn(1, 16, s_gen, 128, dtype=bf)
        kk = torch.randn(1, 8, s_gen + a.text, 128, dtype=bf)
        vv = torch.randn(1, 8, s_gen + a.text, 128, dtype=bf)
        from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import NeuronCosmos3EdgeGEN

        m = torch.zeros(1, a.text, dtype=torch.long)
        m[0, :300] = 1
        bias = NeuronCosmos3EdgeGEN.key_bias(m, s_gen)

        def f(q_, k_, v_, b_):
            for _ in range(4):  # 4 chained layers in one graph; tiny output -> no transfer noise
                q_ = f1(q_, k_, v_, b_)
            return q_.float().sum(dim=(-2, -1))  # full reduction: every row is live (a row slice let the compiler prune)

        def f1(q_, k_, v_, b_):
            if os.environ.get("BENCH_SDPA"):
                kr = k_.repeat_interleave(2, dim=1)
                vr = v_.repeat_interleave(2, dim=1)
                m = None if os.environ.get("BENCH_NO_BIAS") else b_.to(q_.dtype)
                return torch.nn.functional.scaled_dot_product_attention(q_, kr, vr, attn_mask=m)
            return edge_attention(q_, k_, v_, 128 ** -0.5, key_bias=None if os.environ.get("BENCH_NO_BIAS") else b_)

        args = [x.to(dev) for x in (qq, kk, vv, bias)]
        fn = torch.compile(f, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                           options={"model_name": name, "compiler_args": cargs})
    else:
        import socket

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel

        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        _ctx = set_current_vllm_config(VllmConfig())  # keep a reference: a GC'd generator exits the context
        _ctx.__enter__()
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
        initialize_model_parallel(1, 1)
        from vllm_omni_neuron.diffusion.models.cosmos3_edge import gen_tower as _gt
        from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
            EdgeGenConfig,
            NeuronCosmos3EdgeGEN,
        )

        if a.skip_attn:
            _gt.edge_attention = lambda q_, k_, v_, scale, **kw: q_

        cfg = EdgeGenConfig.from_model_dir(W)
        mask = torch.zeros(1, a.text, dtype=torch.long)
        mask[0, :300] = 1
        gen = NeuronCosmos3EdgeGEN(cfg, dtype=bf)
        gen.load_weights(W, dev)
        cg, sg = gen.rope_tables(mask, t, h, w, 24.0)
        kb = gen.key_bias(mask, s_gen)
        from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

        und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
        und.load_weights(W, "cpu")
        ids = torch.randint(1000, 100000, (1, a.text)) * mask
        cu, su = und.rope_tables(mask)
        with torch.no_grad():
            kv = [x.to(bf) for x in und(ids, cu, su)]
        del und
        args = [torch.randn(1, 48, t, h, w, dtype=bf), torch.tensor([500.0]), cg.to(bf), sg.to(bf), kb,
                torch.ones(1, s_gen, 1, dtype=bf), *kv]
        args = [x.contiguous().to(dev) for x in args]
        fn = torch.compile(gen, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                           options={"model_name": name, "compiler_args": cargs})

    with torch.no_grad():
        t0 = time.time()
        fn(*args).cpu()
        first = time.time() - t0
        ws = []
        for _ in range(int(os.environ.get("BENCH_REPS", "5"))):
            t0 = time.time()
            fn(*args).cpu()
            ws.append(time.time() - t0)
    print("BENCH " + json.dumps({"mode": a.mode, "qblock": q, "opt": a.opt, "extra": a.extra, "skip_attn": a.skip_attn, "tokens": s_gen,
                                 "first_s": round(first, 1), "warm_s": round(statistics.median(ws), 4), "tag": os.environ.get("BENCH_TAG", "")}), flush=True)


if __name__ == "__main__":
    main()
