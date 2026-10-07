# SPDX-License-Identifier: Apache-2.0
"""CPU reproduction of the TP=2 graphs (gloo, 2 processes): runs the tiny Alpamayo checkpoint in
bf16 through every device graph (``compile("eager")`` -> fullgraph trace) with tensor parallelism,
and compares rank 0's trajectory/tokens with the TP=1 run. Catches TP-only dtype mixes (the fp32
RowParallelLinear bias) that the TP=1 CPU path never exercises.

    VLLM_NEURON_CPU_MODE=1 python examples/alpamayo/tp2_cpu_check.py <checkpoint dir> <parity_ref.py dump>
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank: int, world: int, ckpt: str, ref_path: str, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("TP2_PORT", "29571"))
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    ref = torch.load(ref_path, weights_only=False)
    mi = ref["model_inputs"]
    m = NeuronAlpamayo1_5.from_pretrained(
        ckpt,
        dtype=getattr(torch, os.environ.get("TP2_DTYPE", "bfloat16")),
        tp_group=dist.group.WORLD if world > 1 else None,
    )
    m.compile("eager")
    r = m.get_action(
        dict(mi["tokenized_data"]),
        ego_history_xyz=mi["ego_history_xyz"],
        ego_history_rot=mi["ego_history_rot"],
        noise=ref["noise"],
    )
    if rank == 0:
        d = m._debug
        q.put(
            (
                r["generated"].numpy().copy(),
                r["pred_xyz"].numpy().copy(),
                {
                    "vision": d["image_embeds"].float().cpu().numpy().copy(),
                    "logits": d["prefill_logits_last"].float().numpy().copy(),
                    "v0": m._debug_steps["v"][0].float().numpy().copy(),
                },
            )
        )
    if world > 1:
        dist.destroy_process_group()


def run(world: int, ckpt: str, ref: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_run, args=(r, world, ckpt, ref, q)) for r in range(world)]
    for p in ps:
        p.start()
    g, x, d = q.get(timeout=1800)
    out = (torch.from_numpy(g), torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in d.items()})
    for p in ps:
        p.join()
        if p.exitcode:
            raise SystemExit(f"rank exited {p.exitcode}")
    return out


if __name__ == "__main__":
    ckpt, ref = sys.argv[1], sys.argv[2]
    g1, x1, d1 = run(1, ckpt, ref)
    g2, x2, d2 = run(int(os.environ.get("TP2_WORLD", "2")), ckpt, ref)
    rel = float((x2 - x1).norm() / x1.norm())
    parts = {k: float((d2[k] - d1[k]).norm() / d1[k].norm()) for k in d1}
    print(f"[tp2] tokens_equal={torch.equal(g1, g2)} traj_rel_tp2_vs_tp1={rel:.3e} parts={parts}")
    print(f"[tp2] tokens tp1={g1.tolist()} tp2={g2.tolist()}")
    sys.exit(0 if torch.equal(g1, g2) and rel < 2e-2 else 1)
