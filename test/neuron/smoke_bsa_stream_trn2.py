# SPDX-License-Identifier: Apache-2.0
"""Device probe for the streaming block-sparse kernel (L4, ``kernels/bsa_stream.py``) at the two
consumers' per-rank shapes on one LNC=2 logical core, next to the dense ``attention_cte`` baseline.
Same plans, data and seeds as ``smoke_block_sparse_trn2.py`` (A0's probe of L1/L2/L3), so the rows
compare directly with its L2/L3 numbers.

Per shape: dense and L4 variants (pass width, V on a second DMA queue, deeper prefetch, a
dense-equivalent list = every block listed); warm wall time per call (best of N, output copied to
host, as the A0 probe times), device time per call from an in-graph chain (``t(3 calls) - t(1)``
over 2), per-key efficiency vs dense (dense ms per key / L4 ms per kept key), rel-L2 vs the fp32
masked reference on the first 512 query rows next to the CPU-bf16 error. Optionally a
``neuron-profile`` capture of the L4 NEFF (``--profile``).

    SMOKE_OUT=... python test/neuron/smoke_bsa_stream_trn2.py --shapes fasth3,fasth3-k128,hunyuan
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import smoke_block_sparse_trn2 as A0  # noqa: E402
import smoke_platform_trn2 as S  # noqa: E402
import torch  # noqa: E402

from vllm_omni_neuron.diffusion.attention import block_sparse as BS  # noqa: E402

D = 128
SHAPES = dict(A0.SHAPES)
# VSA (FastH3-VSA-DataFree) real per-rank geometry at 768p TP8xCP8: 672 tiles of 64 slots (660 video
# (4,4,4) cubes + 8 text/audio prefix + 4 empty) = 43008 slots, 84 query tiles per rank = 42 kernel items
# (pairs of 64-row tiles attending the UNION of both tiles' lists: 8 prefix + top-66 video each). The union
# size depends on the trained model's neighbour overlap; probed at both ends: identical halves (74 tiles)
# and disjoint halves (140). Each half is masked to its own tiles (kernel masked=1 mode), so the error is
# checked against the per-64-row-tile reference.
SHAPES["vsa-u74"] = dict(
    heads=7, lq=5376, lk_real=43008, q_block=128, k_block=64, vsa_union=74, useful_tiles=74
)
SHAPES["vsa-u140"] = dict(
    heads=7, lq=5376, lk_real=43008, q_block=128, k_block=64, vsa_union=140, useful_tiles=74
)
SHAPES["dry-vsa"] = dict(
    heads=2, lq=512, lk_real=40 * 64, q_block=128, k_block=64, vsa_union=12, useful_tiles=8
)
SHAPES["dry-sla128"] = dict(
    heads=3, lq=384, lk_real=20 * 128 - 30, q_block=128, k_block=128, keep=0.3, text_blocks=0
)
SHAPES["dry-sla"] = dict(
    heads=3, lq=384, lk_real=40 * 64 - 30, q_block=128, k_block=64, keep=0.3, text_blocks=0
)


def _rel(a, b):
    return S._rel(a, b)


def _wall(fn, args, reps):
    t0 = time.time()
    out = fn(*args)
    out_cpu = out.cpu()
    first = time.time() - t0
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        fn(*args).cpu()
        best = min(best, time.time() - t0)
    return out_cpu, first, best


def _chain_ms(dev, call, q_list, rest, name, reps):
    """Device ms per call from in-graph chains of 1 and 3 calls on distinct q (outputs summed, so
    nothing is CSE'd): (t3 - t1) / 2 removes launch, transfer and per-graph cost."""

    def mk(n):
        def fn(*a):
            qs, r = a[:n], a[n:]
            acc = call(qs[0], *r).float()
            for i in range(1, n):
                acc = acc + call(qs[i], *r).float()
            return acc.to(torch.bfloat16)

        return S._compile_on(dev, fn, f"{name}_x{n}")

    ts = {}
    for n in (1, 3):
        f = mk(n)
        args = list(q_list[:n]) + list(rest)
        f(*args).cpu()
        best = 1e9
        for _ in range(reps):
            t0 = time.time()
            f(*args).cpu()
            best = min(best, time.time() - t0)
        ts[n] = best
    return (ts[3] - ts[1]) / 2 * 1e3, ts


def _profile(tag, out_dir, since):
    """neuron-profile the newest NEFF written since ``since`` (the L4 graph): engine utilisation
    and DMA summary as text. Best effort; failures are recorded, not raised."""
    cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "")
    neffs = [
        p
        for p in glob.glob(os.path.join(cache, "**", "*.neff"), recursive=True)
        if os.path.getmtime(p) >= since
    ]
    if not neffs:
        return {"error": f"no NEFF newer than the L4 compile under {cache!r}"}
    neff = max(neffs, key=os.path.getmtime)
    if os.environ.get("PROFILE_DEFER") == "1":
        # the probe process holds the cores: capture after it exits (the job script reads this list)
        with open(os.path.join(out_dir, "profile_targets.txt"), "a") as f:
            f.write(f"{tag} {neff}\n")
        return {"neff": neff, "deferred": True}
    ntff = os.path.join(out_dir, f"profile_{tag}.ntff")
    res = {"neff": neff}
    # On PATH in the Neuron SDK environment (ships next to neuron-monitor / neuron-top).
    tool = shutil.which("neuron-explorer") or "neuron-explorer"
    try:
        cap = subprocess.run(
            [tool, "capture", "-n", neff, "-s", ntff],
            capture_output=True,
            text=True,
            timeout=900,
        )
        res["capture_rc"] = cap.returncode
        res["capture_tail"] = (cap.stdout + cap.stderr)[-1500:]
        view = subprocess.run(
            [tool, "view", "-n", neff, "-s", ntff, "--output-format", "summary-text"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        txt = view.stdout + view.stderr
        with open(os.path.join(out_dir, f"profile_{tag}.txt"), "w") as f:
            f.write(txt)
        res["summary_lines"] = [
            ln.strip()
            for ln in txt.splitlines()
            if any(
                k in ln.lower()
                for k in ("util", "dma", "total_time", "pe_", "act_", "dve_", "pool_", "busy")
            )
        ][:80]
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def _vsa_plan(spec, h, lq, lk):
    """Per-64-row-tile selections (prefix tiles + top-k video tiles), pairs either identical
    (union = useful_tiles) or disjoint (union = 2 * k_vid + prefix)."""
    nk, n_pairs = lk // 64, lq // 128
    n_prefix, n_empty = (8, 4) if nk > 100 else (2, 0)
    k_vid = spec["useful_tiles"] - n_prefix
    disjoint = spec["vsa_union"] > spec["useful_tiles"]
    g = torch.Generator().manual_seed(0)
    n_vid = nk - n_prefix - n_empty
    perm = torch.rand(h, n_pairs, n_vid, generator=g).argsort(-1) + n_prefix
    sel = torch.zeros(h, n_pairs, 2, nk, dtype=torch.bool)
    sel[..., :n_prefix] = True
    sel[:, :, 0].scatter_(2, perm[..., :k_vid], True)
    sel[:, :, 1].scatter_(2, perm[..., k_vid : 2 * k_vid] if disjoint else perm[..., :k_vid], True)
    sel64 = sel.reshape(h, 2 * n_pairs, nk)
    return BS.pair_union_plan(sel64, 64), sel64


def probe_shape(name, spec, dev, reps, out_dir, profile, variants="v2"):
    torch.manual_seed(0)
    h, lq = spec["heads"], spec["lq"]
    lk = -(-spec["lk_real"] // spec["k_block"]) * spec["k_block"]
    q = torch.randn(h, lq, D).to(torch.bfloat16)
    k = torch.randn(h, lk, D).to(torch.bfloat16)
    v = torch.randn(h, lk, D).to(torch.bfloat16)
    k[:, spec["lk_real"] :] = 0
    v[:, spec["lk_real"] :] = 0
    if "vsa_union" in spec:
        plan, sel64 = _vsa_plan(spec, h, lq, lk)
        useful = spec["useful_tiles"] * plan.k_block  # keys each 64-row query tile needs
    else:
        plan = A0._plan(spec, q, k)
        useful, sel64 = None, None
    kept_keys = float(plan.counts.float().mean()) * plan.k_block
    eff_keys = useful or kept_keys
    row = dict(
        shape=name, heads=h, lq=lq, lk=lk, lk_real=spec["lk_real"], q_block=plan.q_block,
        k_block=plan.k_block, n_k_blocks=plan.n_k_blocks, kp=plan.kp,
        kept_fraction=round(plan.kept_fraction, 4), kept_keys_per_query=round(kept_keys, 1),
    )  # fmt: skip
    sub = 512 if plan.q_block <= 512 else plan.q_block
    nqs = max(1, sub // plan.q_block)
    sub = nqs * plan.q_block

    def _sub_plan(p):
        return BS.BlockSparsePlan(
            p.lists[:, :nqs], p.counts[:, :nqs], p.q_block, p.k_block, p.n_k_blocks, p.key_valid
        )

    ref32 = BS.reference_attention(q[:, :sub].float(), k.float(), v.float(), _sub_plan(plan))
    if sel64 is not None:  # VSA: each 64-row tile attends only its own tiles
        p64 = BS.plan_from_block_mask(sel64[:, : 2 * nqs], 64, 64)
        ref32 = BS.reference_attention(q[:, :sub].float(), k.float(), v.float(), p64)
    dense_ref32 = torch.nn.functional.scaled_dot_product_attention(
        q[:, :sub].float(), k.float(), v.float()
    )
    tok = (p64 if sel64 is not None else _sub_plan(plan)).token_mask(sub, lk)
    s16 = (q[:, :sub].float() @ k.float().transpose(1, 2)) * D**-0.5
    p16 = torch.softmax(s16.masked_fill(~tok, float("-inf")), -1).to(torch.bfloat16).float()
    row["cpu_bf16_rel"] = _rel((p16 @ v.float()).to(torch.bfloat16), ref32)
    del s16, p16, tok
    print(
        f"== {name}: kept {row['kept_fraction']:.3f}; cpu-bf16 rel {row['cpu_bf16_rel']:.4f}",
        flush=True,
    )

    to = lambda t: t.to(dev).contiguous()  # noqa: E731
    scale = D**-0.5
    results = {}
    row["paths"] = results

    def save():
        with open(os.path.join(out_dir, f"bsa_stream_{name}.json"), "w") as f:
            json.dump(row, f, indent=1)

    def record(key, fn):
        torch._dynamo.reset()
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            r = {
                "error": f"{type(exc).__name__}: {str(exc)[-600:]}",
                "trace": traceback.format_exc()[-2000:],
            }
        results[key] = r
        print(
            f"  {name} {key}: {json.dumps({a: b for a, b in r.items() if a != 'trace'})[:600]}",
            flush=True,
        )
        save()

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.bsa_stream import bsa_stream

    # ---- dense attention_cte baseline
    g = torch.Generator().manual_seed(7)
    q_alt = [q] + [torch.randn(h, lq, D, generator=g).to(torch.bfloat16) for _ in range(2)]

    def dense_call(a, b, c):
        return wrap_nki(BS._dense_kernel)[2](q=a, k=b, v=c)

    def dense():
        qs = [to((x.float() * scale).to(torch.bfloat16)) for x in q_alt]
        kd, vd = to(k), to(v)
        f = S._compile_on(dev, dense_call, f"bss_dense_{name}")
        out, first, warm = _wall(f, (qs[0], kd, vd), reps)
        r = dict(
            first_s=round(first, 1),
            warm_ms=round(warm * 1e3, 2),
            rel_vs_fp32=_rel(out[:, :sub], dense_ref32),
        )
        if dev.type != "cpu":
            r["device_ms"], ts = _chain_ms(dev, dense_call, qs, (kd, vd), f"bss_dense_{name}", reps)
            r["device_ms"] = round(r["device_ms"], 3)
            r["chain_s"] = {str(a): round(b, 4) for a, b in ts.items()}
        return r

    record("dense", dense)

    # ---- L4 variants
    def l4(plan_, pb=None, nbuf=3, v_queue=0, chain=False, prof=None, ref=None, packed=False):
        ops = BS.stream_inputs(plan_, q, k, v, scale, pb, packed=packed)
        n_real = ops.pop("n_real_items")
        msk = sel64 is not None and plan_ is plan
        if msk:
            km, qi = BS.stream_half_masks(BS.stream_plan(plan_, ops["pb"], packed), sel64)
            ops["kmask"], ops["qind"] = km, qi
        qb, pbv = ops.pop("q_block"), ops.pop("pb")
        pk, kbk = ops.pop("packed", 0), ops.pop("k_block", 0)
        kp, n_pass = ops["lists"].shape[1], ops["bounds"].shape[2]
        layout = dict(
            pb=pbv,
            keys_per_pass=pbv * plan_.k_block,
            n_pass=n_pass,
            kp=kp,
            items=ops["lists"].shape[0],
            packed=pk,
        )
        dev_ops = {a: to(b) for a, b in ops.items()}

        def call(qT, kT_blk, v_blk, lists, bounds, *mk):
            extra = dict(kmask=mk[0], qind=mk[1], masked=1) if mk else {}
            return wrap_nki(bsa_stream)[2](
                qT=qT, kT_blk=kT_blk, v_blk=v_blk, lists=lists, bounds=bounds, q_block=qb, pb=pbv,
                nbuf=nbuf, v_queue=v_queue, packed=pk, k_block=kbk, **extra,
            )  # fmt: skip

        tag = f"{pbv}_{nbuf}_{v_queue}_{kp}_{pk}_{int(msk)}"
        since = time.time()
        f = S._compile_on(dev, call, f"bss_l4_{name}_{tag}")
        rest = (dev_ops["kT_blk"], dev_ops["v_blk"], dev_ops["lists"], dev_ops["bounds"])
        if msk:
            rest = rest + (dev_ops["kmask"], dev_ops["qind"])
        out, first, warm = _wall(f, (dev_ops["qT"],) + rest, reps)
        out = out[: n_real * qb].reshape(h, lq, D)
        target = ref if ref is not None else ref32
        r = dict(
            layout=layout,
            first_s=round(first, 1),
            warm_ms=round(warm * 1e3, 2),
            rel_vs_fp32=_rel(out[:, :sub], target),
        )
        if chain and dev.type != "cpu":
            qts = [dev_ops["qT"]]
            for x in q_alt[1:]:
                qts.append(to(BS.stream_inputs(plan_, x, k, v, scale, pb)["qT"]))
            ms, ts = _chain_ms(dev, call, qts, rest, f"bss_l4c_{name}_{tag}", reps)
            r["device_ms"] = round(ms, 3)
            r["chain_s"] = {str(a): round(b, 4) for a, b in ts.items()}
        # analytic K/V DMA volume and count per call
        n_items = ops["lists"].shape[0]
        r["kv_bytes"] = n_items * kp * plan_.k_block * D * 2 * 2
        r["kv_dmas"] = n_items * kp * (1 if pk else 2)
        if prof and dev.type != "cpu":
            r["profile"] = _profile(prof, out_dir, since)
        return r

    plan_dense = BS.plan_from_block_mask(
        torch.ones(h, plan.n_q_blocks, plan.n_k_blocks, dtype=torch.bool), plan.q_block, plan.k_block,
        key_valid=plan.key_valid,
    )  # fmt: skip
    dref = BS.reference_attention(q[:, :sub].float(), k.float(), v.float(), _sub_plan(plan_dense))
    can_pack = plan.k_block % 128 == 0
    prof_tag = f"{name}_best" if profile else None
    if variants == "v3":  # queue count (2 vs 3) for the small-block targets; VSA real geometry
        if can_pack:
            record(
                "l4_packed_vq1", lambda: l4(plan, packed=True, v_queue=1, chain=True, prof=prof_tag)
            )
            record("l4_packed_vq2", lambda: l4(plan, packed=True, v_queue=2, chain=True))
        else:
            record("l4_vq1", lambda: l4(plan, v_queue=1, chain=True, prof=prof_tag))
            record("l4_vq2", lambda: l4(plan, v_queue=2, chain=True))
    elif variants == "v1":
        record("l4", lambda: l4(plan, chain=True, prof=f"{name}_l4" if profile else None))
        record("l4_vq1", lambda: l4(plan, v_queue=1))
        record("l4_nb4", lambda: l4(plan, nbuf=4))
        record("l4_dense_equiv", lambda: l4(plan_dense, ref=dref))
    else:  # v2: two-queue split (+ packed K|V blocks where the block is a multiple of 128 rows)
        record(
            "l4_vq1", lambda: l4(plan, v_queue=1, chain=True, prof=None if can_pack else prof_tag)
        )
        if can_pack:
            record("l4_packed", lambda: l4(plan, packed=True))
            record(
                "l4_packed_vq1", lambda: l4(plan, packed=True, v_queue=1, chain=True, prof=prof_tag)
            )
            record("l4_packed_vq1_nb4", lambda: l4(plan, packed=True, v_queue=1, nbuf=4))
            record("l4_dense_equiv", lambda: l4(plan_dense, ref=dref, packed=True, v_queue=1))
        else:
            record("l4_dense_equiv", lambda: l4(plan_dense, ref=dref, v_queue=1))

    # ---- efficiency per key vs dense (wall and device-chain timings)
    dn = results.get("dense", {})
    lk_eff = lk  # dense works over all lk keys (pads included), as in the A0 probe
    for key, r in results.items():
        if key == "dense" or "warm_ms" not in r:
            continue
        keys = eff_keys if key != "l4_dense_equiv" else lk
        if dn.get("warm_ms"):
            r["eff_per_key_wall"] = round((dn["warm_ms"] / lk_eff) / (r["warm_ms"] / keys), 3)
            r["speedup_vs_dense_wall"] = round(dn["warm_ms"] / r["warm_ms"], 3)
        if dn.get("device_ms") and r.get("device_ms"):
            r["eff_per_key_device"] = round((dn["device_ms"] / lk_eff) / (r["device_ms"] / keys), 3)
            r["speedup_vs_dense_device"] = round(dn["device_ms"] / r["device_ms"], 3)
        if "rel_vs_fp32" in r:
            r["rel_over_cpu_bf16"] = round(r["rel_vs_fp32"] / max(row["cpu_bf16_rel"], 1e-9), 2)
    if "rel_vs_fp32" in dn:
        dn["rel_over_cpu_bf16"] = round(dn["rel_vs_fp32"] / max(row["cpu_bf16_rel"], 1e-9), 2)
    save()
    keep = ("warm_ms", "device_ms", "eff_per_key_wall", "eff_per_key_device", "speedup_vs_dense_wall",
            "rel_vs_fp32", "rel_over_cpu_bf16", "error")  # fmt: skip
    print(
        "SUMMARY "
        + name
        + " "
        + json.dumps({a: {x: y for x, y in b.items() if x in keep} for a, b in results.items()}),
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="fasth3,fasth3-k128,hunyuan")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--variants", default="v2", choices=["v1", "v2", "v3"])
    ap.add_argument("--out", default=os.environ.get("SMOKE_OUT", "."))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = S._device()
    rows = [
        probe_shape(n, SHAPES[n], dev, args.reps, args.out, args.profile, args.variants)
        for n in args.shapes.split(",")
    ]
    ok = all("error" not in p for r in rows for p in r["paths"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
