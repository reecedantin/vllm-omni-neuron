# SPDX-License-Identifier: Apache-2.0
"""Device probe for the shared block-sparse attention (diffusion/attention/block_sparse.py) at the
two consumers' per-rank shapes, on one LNC=2 logical core:

* ``fasth3``: 1344x768x124 at TP8 x CP8 -- 7 heads, 4736 query rows (37 tiles of 128), 37888 keys
  (590 x 64-row blocks, 37744 real), SLA top-15% = 88 blocks per query tile; also the same plan
  at 128-row key blocks (295 blocks, 44 kept).
* ``hunyuan``: 720p x 121 f at TP8 x CP8 -- 2 heads, 45 query tiles of 384 (17280 rows), 366 key
  tiles of 384 (360 video + 6 text = 140544 keys), 66 kept per query tile (text always).

Per shape: dense ``attention_cte`` baseline, L1 (dense + block bias), L2 (gather + dense with
bounds), L3 (index-list kernel). Reports ms per layer (warm, best of N), per-key efficiency vs
dense (dense ms per key / path ms per kept key), and rel-L2 vs the fp32 masked reference on a
query subset, next to the CPU-bf16 error of the same subset (the bf16 bar: device error should
sit at the CPU-bf16 level, not far above it).

    SMOKE_OUT=... python test/neuron/smoke_block_sparse_trn2.py --shapes fasth3,hunyuan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import smoke_platform_trn2 as S  # noqa: E402
import torch  # noqa: E402

from vllm_omni_neuron.diffusion.attention import block_sparse as BS  # noqa: E402

D = 128
SHAPES = {
    "fasth3": dict(
        heads=7, lq=4736, lk_real=37744, q_block=128, k_block=64, keep=0.15, text_blocks=0
    ),
    "fasth3-k128": dict(
        heads=7, lq=4736, lk_real=37744, q_block=128, k_block=128, keep=0.15, text_blocks=0
    ),
    "hunyuan": dict(
        heads=2, lq=17280, lk_real=140544, q_block=384, k_block=384, kept=66, text_blocks=6
    ),
    # CPU dry run of the harness (SMOKE_DEVICE=cpu NKI_SIMULATOR=1): same code paths, tiny shapes
    "dry-sla": dict(
        heads=1, lq=256, lk_real=1990, q_block=128, k_block=64, keep=0.3, text_blocks=0
    ),
    "dry-ssta": dict(
        heads=1, lq=768, lk_real=7 * 384 - 100, q_block=384, k_block=384, kept=4, text_blocks=2
    ),
}


def _plan(spec, q, k):
    lk = k.shape[1]
    nk = lk // spec["k_block"]
    if "keep" in spec:
        mask = BS.sla_block_mask(
            q, k, spec["q_block"], spec["k_block"], spec["keep"], lk_real=spec["lk_real"]
        )
    else:  # SSTA-like: text tiles always, the rest random (shape study; the selection is A6's)
        h, nq = q.shape[0], q.shape[1] // spec["q_block"]
        g = torch.Generator().manual_seed(0)
        mask = torch.zeros(h, nq, nk, dtype=torch.bool)
        n_text = spec["text_blocks"]
        mask[..., nk - n_text :] = True
        pick = (
            torch.rand(h, nq, nk - n_text, generator=g).topk(spec["kept"] - n_text, dim=-1).indices
        )
        mask[..., : nk - n_text].scatter_(2, pick, True)
    valid = torch.arange(lk) < spec["lk_real"]
    return BS.plan_from_block_mask(mask, spec["q_block"], spec["k_block"], key_valid=valid)


def _timed(fn, *args, reps=3):
    t0 = time.time()
    out = fn(*args)
    out_cpu = out.cpu()
    first = time.time() - t0
    best = None
    for _ in range(reps):
        t0 = time.time()
        fn(*args).cpu()
        best = min(best or 1e9, time.time() - t0)
    return out_cpu, first, best


def _rel(a, b):
    return S._rel(a, b)


def probe_shape(name, spec, dev, reps, out_dir, only):
    torch.manual_seed(0)
    h, lq = spec["heads"], spec["lq"]
    lk = -(-spec["lk_real"] // spec["k_block"]) * spec["k_block"]
    q = torch.randn(h, lq, D).to(torch.bfloat16)
    k = torch.randn(h, lk, D).to(torch.bfloat16)
    v = torch.randn(h, lk, D).to(torch.bfloat16)
    k[:, spec["lk_real"] :] = 0
    v[:, spec["lk_real"] :] = 0
    plan = _plan(spec, q, k)
    kept_keys = float(plan.counts.float().mean()) * plan.k_block
    row = {
        "shape": name,
        "heads": h,
        "lq": lq,
        "lk": lk,
        "lk_real": spec["lk_real"],
        "q_block": plan.q_block,
        "k_block": plan.k_block,
        "n_k_blocks": plan.n_k_blocks,
        "kp": plan.kp,
        "kept_fraction": round(plan.kept_fraction, 4),
        "kept_keys_per_query": round(kept_keys, 1),
    }
    # reference on a query subset (first 512 rows per head): fp32 math and CPU-bf16-rounded math
    sub = 512
    plan_sub = BS.BlockSparsePlan(
        plan.lists[:, : sub // plan.q_block] if plan.q_block <= sub else plan.lists[:, :1],
        plan.counts[:, : sub // plan.q_block] if plan.q_block <= sub else plan.counts[:, :1],
        plan.q_block,
        plan.k_block,
        plan.n_k_blocks,
        plan.key_valid,
    )
    sub = plan_sub.n_q_blocks * plan.q_block
    ref32 = BS.reference_attention(q[:, :sub].float(), k.float(), v.float(), plan_sub).float()
    dense_ref32 = torch.nn.functional.scaled_dot_product_attention(
        q[:, :sub].float(), k.float(), v.float()
    ).float()
    tok = plan_sub.token_mask(sub, lk)
    s16 = (q[:, :sub].float() @ k.float().transpose(1, 2)) * D**-0.5
    p16 = torch.softmax(s16.masked_fill(~tok, float("-inf")), -1).to(torch.bfloat16).float()
    ref_bf16 = (p16 @ v.float()).to(torch.bfloat16).float()
    row["cpu_bf16_rel"] = _rel(ref_bf16, ref32)
    del s16, p16, tok
    print(
        f"== {name}: kept {row['kept_fraction']:.3f} ({kept_keys:.0f} keys/query of {lk}); cpu-bf16 rel {row['cpu_bf16_rel']:.4f}",
        flush=True,
    )

    to = lambda t: t.to(dev).contiguous()  # noqa: E731
    scale = D**-0.5
    qs = to((q.float() * scale).to(torch.bfloat16))
    kd, vd = to(k), to(v)
    results = {}

    def record(key, fn):
        if only and key not in only:
            return
        torch._dynamo.reset()
        try:
            out, first, warm = fn()
            target = (
                dense_ref32 if key == "dense" else ref32
            )  # dense attends the pad keys too (zeros); compare on real keys only
            r = {
                "first_s": round(first, 1),
                "warm_ms": round(warm * 1e3, 2),
                "rel_vs_fp32": _rel(out[:, :sub], target),
                "shape": list(out.shape),
            }
            if key == "dense":
                r["note"] = "rel vs a DENSE fp32 reference (same zero pad keys)"
        except Exception as exc:  # noqa: BLE001
            r = {
                "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
                "trace": traceback.format_exc()[-1500:],
            }
        results[key] = r
        print(
            f"  {name} {key}: {json.dumps({kk: vv for kk, vv in r.items() if kk != 'trace'})[:400]}",
            flush=True,
        )
        row["paths"] = results
        with open(os.path.join(out_dir, f"block_sparse_{name}.json"), "w") as f:
            json.dump(row, f, indent=1)

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    # ---- dense attention_cte (the baseline) -- per head, one call each (the DiTs call it per layer with H batched)
    def dense():
        f = S._compile_on(
            dev, lambda a, b, c: wrap_nki(BS._dense_kernel)[2](q=a, k=b, v=c), f"bs_dense_{name}"
        )
        return _timed(f, qs, kd, vd, reps=reps)

    record("dense", dense)

    # ---- L1: dense + block bias, per head pair, query chunks (bias chunk x lk bf16 built on host)
    q_chunk = {"fasth3": 4736, "fasth3-k128": 4736, "hunyuan": 1920}.get(name, lq)  # must divide lq
    bias = BS.block_bias(plan, lq, lk)  # host, [H, lq, lk] bf16

    def _l1_run(bias_host, lq_, kd_, vd_, q_host_scaled, plan_ref, tag):
        """One L1 pass: pairs of heads, query chunks; every operand is built on the HOST (a sliced
        device tensor + .contiguous() is an eager op the Lite executor refuses: run 2 hunyuan)."""
        f = S._compile_on(
            dev,
            lambda a, b, c, bb: wrap_nki(BS._dense_bias_kernel)[2](q=a, k=b, v=c, bias=bb),
            f"bs_l1_{name}_{tag}",
        )
        pairs = [[h0, min(h0 + 1, h - 1)] for h0 in range(0, h, 2)]
        qc = q_chunk if lq_ == lq else lq_
        chunks = [(p, c0) for p in pairs for c0 in range(0, lq_, qc)]
        bias_dev = [to(bias_host[p, c0 : c0 + qc]) for p, c0 in chunks]
        q_dev = [to(q_host_scaled[p, c0 : c0 + qc]) for p, c0 in chunks]
        kv_dev = {tuple(p): (to(kd_[p]), to(vd_[p])) for p in pairs}

        def run_all():
            outs = []
            for (p, c0), bd, qd in zip(chunks, bias_dev, q_dev):
                kk, vv = kv_dev[tuple(p)]
                outs.append(f(qd, kk, vv, bd))
            per_pair = lq_ // qc
            rows = [
                torch.cat(outs[i * per_pair : (i + 1) * per_pair], 1) for i in range(len(pairs))
            ]
            rows[-1] = rows[-1] if pairs[-1][0] != pairs[-1][1] else rows[-1][:1]
            return torch.cat(rows, 0)

        return _timed(run_all, reps=reps)

    q_scaled_host = (q.float() * scale).to(torch.bfloat16)

    def l1():
        return _l1_run(bias, lq, k, v, q_scaled_host, plan, "full")

    def l1_zero_bias():
        # bias all zero: must reproduce the DENSE result; if it does, the kernel READS the bias at the
        # wrong rows/columns in the full case rather than ignoring it
        out, first, warm = _l1_run(torch.zeros_like(bias), lq, k, v, q_scaled_host, plan, "zero")
        results["l1_zero_bias_vs_dense_rel"] = _rel(out[:, :sub], dense_ref32)
        return out, first, warm

    def l1_small_lk():
        # keys truncated to 8192 (< the kernel's 10240 flash-attention threshold): exact here and
        # wrong at the full Lk points at the flash-section bias path
        lk_s = 8192
        if lk <= 10240:
            raise RuntimeError("skipped: Lk already below the flash-attention threshold")
        nk_s = lk_s // plan.k_block
        bm = plan.block_mask()[:, :, :nk_s]
        bm[..., 0] = True  # every query keeps at least one block
        plan_s = BS.plan_from_block_mask(
            bm, plan.q_block, plan.k_block, key_valid=plan.key_valid[:lk_s]
        )
        bias_s = BS.block_bias(plan_s, lq, lk_s)
        out, first, warm = _l1_run(
            bias_s, lq, k[:, :lk_s], v[:, :lk_s], q_scaled_host, plan_s, "smalllk"
        )
        ref_s = BS.reference_attention(
            q[:, :sub].float(), k[:, :lk_s].float(), v[:, :lk_s].float(), plan_s
        ).float()
        results["l1_small_lk_rel_vs_ref"] = _rel(out[:, :sub], ref_s)
        return out, first, warm

    record("l1_dense_bias", l1)
    record("l1_zero_bias", l1_zero_bias)
    record("l1_small_lk", l1_small_lk)

    # ---- L2: gather listed blocks (compiled torch gather) + attention_cte with bounds, batched over items
    lists_g, bound_max = BS._lists_gather_order(plan)
    kb = to(BS.blockify_kv(k, plan.k_block))
    vb = to(BS.blockify_kv(v, plan.k_block))
    lists_d = to(lists_g)
    nq, bs = plan.n_q_blocks, plan.q_block
    bmax = to(bound_max.reshape(h * nq, 1, 1).expand(h * nq, bs, 1).to(torch.int32))
    bmin = torch.zeros_like(bmax)
    q_items = to(qs.reshape(h * nq, bs, D))

    def l2():
        gather = S._compile_on(
            dev,
            lambda xb, li: BS.gather_blocks(xb, li).reshape(h * nq, plan.kp * plan.k_block, D),
            f"bs_l2_gather_{name}",
        )
        attn = S._compile_on(
            dev,
            lambda a, b, c, lo, hi: wrap_nki(BS._dense_bounded_kernel)[2](
                q=a, k=b, v=c, bound_min=lo, bound_max=hi
            ),
            f"bs_l2_attn_{name}",
        )

        def run_all():
            kg = gather(kb, lists_d)
            vg = gather(vb, lists_d)
            return attn(q_items, kg, vg, bmin, bmax).reshape(h, lq, D)

        out, first, warm = _timed(run_all, reps=reps)
        # kernel-only: gathered K/V staged once, attention timed alone (gather = total - this)
        kg, vg = gather(kb, lists_d), gather(vb, lists_d)
        _, _, a_warm = _timed(attn, q_items, kg, vg, bmin, bmax, reps=reps)
        results.setdefault("l2_parts", {})["attention_only_warm_ms"] = round(a_warm * 1e3, 2)
        results["l2_parts"]["gather_both_warm_ms"] = round((warm - a_warm) * 1e3, 2)
        return out, first, warm

    record("l2_gather_dense", l2)

    # ---- L3: index-list kernel. Variants: pb = kernel blocks per softmax pass (16 / 32 / 64), and a
    # DENSE-EQUIVALENT run (every block listed) that separates the kernel's own structure cost from
    # the sparsity-specific gather cost when compared per key with attention_cte.
    from vllm_omni_neuron.kernels.bsa_index_list import bsa_index_list

    def _l3_run(plan_, pb, tag):
        k3, v_ext, blocks, rowstart = BS.index_list_inputs(plan_, k, v, pb=pb)
        qT_h = (q.float() * scale).to(torch.bfloat16).transpose(1, 2)
        if (lq // 128) % 2:
            qT_h = torch.nn.functional.pad(qT_h, (0, 128))
            blocks = torch.cat([blocks, blocks[:, -1:]], dim=1)
            rowstart = torch.cat([rowstart, rowstart[:, -1:]], dim=1)
        results.setdefault("l3_layouts", {})[tag] = {
            "kernel_blocks_per_list": int(blocks.shape[2]),
            "passes": int(rowstart.shape[2]),
            "keys_per_pass": int(blocks.shape[2] // rowstart.shape[2]) * int(k3.shape[2]),
            "kernel_tiles": int(blocks.shape[1]),
        }
        qT, k3d, ved, bld, rsd = to(qT_h), to(k3), to(v_ext), to(blocks), to(rowstart)
        f = S._compile_on(
            dev,
            lambda a, b, c, d_, e: wrap_nki(bsa_index_list)[2](
                qT=a, k3=b, v_ext=c, blocks_i32=d_, rowstart_i32=e, pb=pb
            ),
            f"bs_l3_{name}_{tag}",
        )
        out, first, warm = _timed(f, qT, k3d, ved, bld, rsd, reps=reps)
        return out[:, :lq], first, warm

    record("l3_index_list", lambda: _l3_run(plan, 16, "pb16"))
    record("l3_pb32", lambda: _l3_run(plan, 32, "pb32"))
    if name.startswith("fasth3"):
        record("l3_pb64", lambda: _l3_run(plan, 64, "pb64"))
    dense_mask = torch.ones(h, plan.n_q_blocks, plan.n_k_blocks, dtype=torch.bool)
    plan_dense = BS.plan_from_block_mask(
        dense_mask, plan.q_block, plan.k_block, key_valid=plan.key_valid
    )

    def l3_dense_equiv():
        out, first, warm = _l3_run(plan_dense, 32, "dense")
        results["l3_dense_equiv_rel_vs_dense"] = _rel(out[:, :sub], dense_ref32)
        return out, first, warm

    if name.startswith(
        "fasth3"
    ):  # 590 blocks x 266 items: ~90 ms per call, fine; hunyuan would be ~5 s
        record("l3_dense_equiv", l3_dense_equiv)

    # ---- efficiency per key vs dense
    dense_ms = results.get("dense", {}).get("warm_ms")
    if dense_ms:
        for key in ("l1_dense_bias", "l2_gather_dense", "l3_index_list", "l3_pb32", "l3_pb64"):
            ms = results.get(key, {}).get("warm_ms")
            if ms:
                results[key]["speedup_vs_dense"] = round(dense_ms / ms, 3)
                results[key]["eff_per_key_vs_dense"] = round((dense_ms / lk) / (ms / kept_keys), 3)
        ms = results.get("l3_dense_equiv", {}).get("warm_ms")
        if ms:
            results["l3_dense_equiv"]["eff_per_key_vs_dense"] = round(
                dense_ms / ms, 3
            )  # same key count
    for key in list(results):
        if isinstance(results[key], dict) and "rel_vs_fp32" in results[key]:
            results[key]["rel_over_cpu_bf16"] = round(
                results[key]["rel_vs_fp32"] / max(row["cpu_bf16_rel"], 1e-9), 2
            )
    row["paths"] = results
    with open(os.path.join(out_dir, f"block_sparse_{name}.json"), "w") as f:
        json.dump(row, f, indent=1)
    print(
        f"SUMMARY {name} "
        + json.dumps(
            {
                k_: {
                    kk: vv
                    for kk, vv in v_.items()
                    if kk
                    in (
                        "warm_ms",
                        "speedup_vs_dense",
                        "eff_per_key_vs_dense",
                        "rel_vs_fp32",
                        "rel_over_cpu_bf16",
                        "error",
                    )
                }
                for k_, v_ in results.items()
                if isinstance(v_, dict)
            }
        ),
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="fasth3,fasth3-k128,hunyuan")
    ap.add_argument("--only", default="", help="comma list of path keys")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.environ.get("SMOKE_OUT", "."))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = S._device()
    only = set(args.only.split(",")) if args.only else None
    rows = []
    for name in args.shapes.split(","):
        rows.append(probe_shape(name, SHAPES[name], dev, args.reps, args.out, only))
    with open(os.path.join(args.out, "block_sparse_probe.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ok = all(
        "error" not in p for r in rows for p in r.get("paths", {}).values() if isinstance(p, dict)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
