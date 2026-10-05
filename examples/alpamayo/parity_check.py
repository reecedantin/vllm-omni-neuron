# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer CPU parity of the Neuron Alpamayo port against an upstream dump
(``parity_ref.py`` for Alpamayo 1.5, ``parity_ref_super.py`` for Alpamayo 2 Super).

Runs with the plugin installed, in CPU mode (``VLLM_NEURON_CPU_MODE=1``)::

    python examples/alpamayo/parity_check.py --model <Alpamayo-1.5-10B dir> --ref parity_fp32.pt [--dtype float32]

Checks, in pipeline order: fused prompt ids; vision embeddings + DeepStack features; every saved
text-layer hidden state; last-position logits; the TEACHER-FORCED expert (reference CoC tokens +
reference noise) per Euler step and the final trajectory; then the free greedy rollout's tokens.
Prints one JSON summary line (``[parity] {...}``) and exits non-zero if a gate fails.
``compare()`` is importable (``test/unit/test_alpamayo_upstream_tiny.py`` uses it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def rel(a, b) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def compare(
    model_dir: str,
    ref_path: str,
    dtype: torch.dtype = torch.float32,
    traj_tol: float = 1e-3,
    greedy: bool = True,
) -> dict:
    os.environ["ALPAMAYO_TRACE"] = "1"  # keep the per-step expert inputs for comparison
    from vllm_omni_neuron.diffusion.models.alpamayo import backbone as bb
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    ref = torch.load(ref_path, weights_only=False)
    mi = ref["model_inputs"]
    tok = mi["tokenized_data"]
    P = int(ref["prompt_len"])
    report: dict = {}
    model = NeuronAlpamayo1_5.from_pretrained(model_dir, dtype=dtype)

    # 1) history fusion
    fused = model.fuse_history(tok["input_ids"], mi["ego_history_xyz"], mi["ego_history_rot"])
    report["fused_ids_equal"] = bool(torch.equal(fused, ref["fused_input_ids"]))

    # 2) prefill, recording every text layer's input (== HF hidden_states[i]) and the last output
    layer_in: list[torch.Tensor] = []
    layer_out: list[torch.Tensor] = []
    orig = bb._TextLayer.do_prefill

    def rec(self, x, *a, **k):
        layer_in.append(x[0, :P].detach().float())
        out = orig(self, x, *a, **k)
        layer_out.append(out[0][0, :P].detach().float())
        return out

    bb._TextLayer.do_prefill = rec
    try:
        res = model.get_action(
            dict(tok),
            ego_history_xyz=mi["ego_history_xyz"],
            ego_history_rot=mi["ego_history_rot"],
            force_tokens=ref["sequences"][0, P:],
            noise=ref["noise"],
        )
    finally:
        bb._TextLayer.do_prefill = orig
    dbg = model._debug
    report["vision_rel"] = rel(dbg["image_embeds"], ref["vision"]["image_embeds"])
    report["deepstack_rel"] = [
        rel(a, b) for a, b in zip(dbg["deepstack"], ref["vision"]["deepstack"])
    ]
    layers = {}
    for i, h in ref["hidden"].items():
        if i < len(layer_out):
            layers[int(i)] = rel(layer_in[i], h)
        else:  # HF's final entry: the last layer's output (normed or not, depending on version)
            last = layer_out[-1]
            layers[int(i)] = min(rel(last, h), rel(model.text.norm(last.to(dtype)).float(), h))
    report["layer_rel"] = layers
    lg, rlg = dbg["prefill_logits_last"][0].float(), ref["prefill_logits_last"].float().clone()
    start = int(model.cfg.head["traj_token_start_idx"])
    rlg[start : start + int(model.cfg.head["traj_vocab_size"])] = float("-inf")
    for tid in model.cfg.head.get("masked_token_ids", ()):  # Alpamayo 2 Super: text EOS masked
        rlg[int(tid)] = float("-inf")
    finite = torch.isfinite(rlg)
    report["logits_rel"] = rel(lg[finite], rlg[finite])
    report["logits_argmax_equal"] = int(lg.argmax()) == int(rlg.argmax())

    # 3) teacher-forced expert
    report["offset"] = [res["offset"], int(ref["offset"])]
    report["expert_v_rel"] = [rel(a, b) for a, b in zip(model._debug_steps["v"], ref["steps"]["v"])]
    report["sampled_action_rel"] = rel(res["actions"], ref["sampled_action"])
    report["traj_rel_teacher_forced"] = rel(res["pred_xyz"], ref["pred_xyz"][:, 0, 0])
    ok = (
        report["fused_ids_equal"]
        and report["offset"][0] == report["offset"][1]
        and report["traj_rel_teacher_forced"] <= traj_tol
    )

    # 4) free greedy rollout
    if greedy:
        g = model.get_action(
            dict(tok),
            ego_history_xyz=mi["ego_history_xyz"],
            ego_history_rot=mi["ego_history_rot"],
            noise=ref["noise"],
        )
        ref_gen = ref["sequences"][0, P:]
        report["greedy_tokens_equal"] = bool(torch.equal(g["generated"], ref_gen))
        report["greedy_n"] = [int(g["generated"].numel()), int(ref_gen.numel())]
        report["traj_rel_greedy"] = rel(g["pred_xyz"], ref["pred_xyz"][:, 0, 0])
        ok = ok and report["greedy_tokens_equal"]
    report["pass"] = bool(ok)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--traj-tol", type=float, default=1e-3)
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()
    r = compare(
        args.model, args.ref, getattr(torch, args.dtype), args.traj_tol, not args.skip_greedy
    )
    print("[parity]", json.dumps(r))
    sys.exit(0 if r["pass"] else 1)
