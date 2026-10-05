# SPDX-License-Identifier: Apache-2.0
"""Upstream (NVlabs ``alpamayo2_super``) DETERMINISTIC CPU reference for Alpamayo 2 Super parity.

The Alpamayo 2 Super counterpart of ``parity_ref.py``: runs upstream's own
``Alpamayo2Super.sample_trajectories_from_data`` with GREEDY decoding (``top_k=1``, so every
upstream logits processor and stopping criterion still applies) and a FIXED flow-matching initial
noise, in fp32 (``--dtype float32``, the parity target) or bf16, and dumps the intermediates the
Neuron port is compared against (same keys as ``parity_ref.py``, so ``parity_check.py`` reads both).

Inputs: ``--inputs`` reuses another dump's ``model_inputs``; otherwise a synthetic PhysicalAI-AV-
shaped sample (``--cameras`` x ``--frames`` smooth colour frames of ``--height`` x ``--width``,
a 16-waypoint ego history) goes through upstream's own ``helper.prepare_model_inputs``.

Private reference venv only (upstream needs hydra / transformers 4.57 on Python 3.12)::

    PYTHONPATH=<alpamayo2 repo>/src <venv>/bin/python examples/alpamayo/parity_ref_super.py \\
        --model <Alpamayo2-Super dir> --out parity_super_fp32.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--inputs", default=None, help="a dump whose model_inputs to reuse")
ap.add_argument("--cameras", type=int, default=6)
ap.add_argument("--frames", type=int, default=4)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument("--max-new-tokens", type=int, default=None, help="default: upstream's (256)")
ap.add_argument("--layers", default="all", help="'all', 'none' or a comma list of hidden_states")
args = ap.parse_args()
dtype = getattr(torch, args.dtype)

from alpamayo2_super import helper  # noqa: E402
from alpamayo2_super.diffusion import flow_matching as _fm  # noqa: E402
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.models.expert_utils import find_eos_offset  # noqa: E402
from alpamayo2_super.models.utils import fuse_traj_tokens  # noqa: E402

t0 = time.time()
model = Alpamayo2Super.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").eval()
model = model.to(dtype)
print(f"[ref] loaded {args.dtype} in {time.time() - t0:.1f}s", flush=True)
freqs = model.expert.action_in_proj.timestep_fourier_encoder.freqs
print(f"[ref] fourier freqs dtype {freqs.dtype} first {freqs.flatten()[:4].tolist()}")
print(f"[ref] vlm generation_config.eos_token_id {model.vlm.generation_config.eos_token_id}")

if args.inputs:
    model_inputs = torch.load(args.inputs, weights_only=False)["model_inputs"]
else:
    spec = importlib.util.spec_from_file_location(
        "_tiny_super",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "test",
            "unit",
            "test_alpamayo_super_upstream_tiny.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    data = mod.synthetic_sample(args.cameras, args.frames, args.height, args.width)
    model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
tok = model_inputs["tokenized_data"]
traj = {k: model_inputs[k] for k in ("ego_history_xyz", "ego_history_rot")}
fused = fuse_traj_tokens(
    model.history_traj_tokenizer,
    model.future_traj_tokenizer,
    tok["input_ids"].clone(),
    traj,
    model.config.traj_ids,
)
print(f"[ref] prompt {tuple(fused.shape)}; images {tuple(tok['image_grid_thw'].shape)}", flush=True)

# -- 1) prefill: vision outputs + every layer's hidden state for the prompt ------------------
vis_out: dict = {}


def _vis_hook(_m, _a, out):
    emb, deep = (out[0], out[1]) if isinstance(out, tuple) else (out, [])
    vis_out["image_embeds"] = emb.detach().float().clone()
    vis_out["deepstack"] = [d.detach().float().clone() for d in deep]


hidden: dict = {}
prefill_logits_last = None
if args.layers != "none":
    hv = model.vlm.model.visual.register_forward_hook(_vis_hook)
    t0 = time.time()
    with torch.no_grad():
        o = model.vlm(
            input_ids=fused,
            attention_mask=tok["attention_mask"],
            pixel_values=tok["pixel_values"].to(dtype),
            image_grid_thw=tok["image_grid_thw"],
            output_hidden_states=True,
            use_cache=False,
        )
    hv.remove()
    print(f"[ref] prefill in {time.time() - t0:.1f}s", flush=True)
    keep = (
        range(len(o.hidden_states))
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )
    hidden = {i: o.hidden_states[i][0].float().clone() for i in keep}
    prefill_logits_last = o.logits[0, -1].float().clone()
    del o

# -- 2) greedy rollout + flow matching with fixed noise ---------------------------------------
dims = tuple(model.expert.action_space.get_action_space_dims())
noise = torch.randn((1, *dims), generator=torch.Generator().manual_seed(args.noise_seed))
_orig_randn = torch.randn


def _randn(*shape, **kw):
    s = (
        tuple(shape[0])
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size))
        else tuple(shape)
    )
    if s == tuple(noise.shape):
        return noise.clone().to(kw.get("device") or "cpu")
    return _orig_randn(*shape, **kw)


steps: dict = {"action_in": [], "v": []}
h1 = model.expert.action_in_proj.register_forward_hook(
    lambda m, a, out: steps["action_in"].append(out.detach().float().clone())
)
h2 = model.expert.action_out_proj.register_forward_hook(
    lambda m, a, out: steps["v"].append(out.detach().float().clone())
)
captured: dict = {}
_orig_generate = model.vlm.generate


def _generate(*a, **kw):
    out = _orig_generate(*a, **kw)
    captured["sequences"] = out.sequences.clone()
    captured["prefill_seq_len"] = out.past_key_values.get_seq_length()
    captured["rope_deltas"] = model.vlm.model.rope_deltas.clone()
    return out


model.vlm.generate = _generate
_orig_euler = _fm.FlowMatching._euler


def _euler(self, *a, **kw):
    out = _orig_euler(self, *a, **kw)
    captured["sampled_action"] = out.detach().float().clone()
    return out


_fm.FlowMatching._euler = _euler
data_in = dict(model_inputs)
data_in["tokenized_data"] = dict(tok)
data_in["tokenized_data"]["pixel_values"] = tok["pixel_values"].to(dtype)
torch.randn = _randn
t0 = time.time()
with torch.no_grad():
    pred_xyz, pred_rot, _, extra = model.sample_trajectories_from_data(
        data_in,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        max_generation_length=args.max_new_tokens,
        return_extra=True,
    )
torch.randn = _orig_randn
h1.remove()
h2.remove()
print(f"[ref] greedy rollout + expert in {time.time() - t0:.1f}s", flush=True)
seq = captured["sequences"]
eos = int(model.config.traj_ids["future_start"])
offset = int(find_eos_offset(sequences=seq, eos_token_id=eos, device=seq.device, warn=False)[0])
n_gen = seq.shape[1] - fused.shape[1]
print(f"[ref] generated {n_gen} tokens, offset {offset}, cache {captured['prefill_seq_len']}")
print(f"[ref] CoC: {extra.get('cot', [[['']]]).reshape(-1)[0]!r:.300}")
print(f"[ref] pred_xyz[-1] {pred_xyz[0, 0, 0, -1].tolist()}")
torch.save(
    {
        "variant": "alpamayo2_super",
        "dtype": args.dtype,
        "model_inputs": model_inputs,
        "fused_input_ids": fused,
        "vision": vis_out,
        "hidden": hidden,
        "prefill_logits_last": prefill_logits_last,
        "sequences": seq,
        "offset": offset,
        "prompt_len": fused.shape[1],
        "prefill_seq_len": captured["prefill_seq_len"],
        "rope_deltas": captured["rope_deltas"],
        "noise": noise,
        "steps": steps,
        "sampled_action": captured["sampled_action"],
        "pred_xyz": pred_xyz.float(),
        "pred_rot": pred_rot.float(),
        "cot": extra.get("cot"),
        "noise_seed": args.noise_seed,
        "max_new_tokens": args.max_new_tokens,
    },
    args.out,
)
print(f"[ref] saved {args.out}")
