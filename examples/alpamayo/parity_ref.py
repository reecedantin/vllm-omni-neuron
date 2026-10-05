# SPDX-License-Identifier: Apache-2.0
"""Upstream (NVlabs alpamayo1_5) DETERMINISTIC fp32 CPU reference for layer-by-layer parity.

Unlike ``reference_1_5.py`` (which reproduces upstream's default *sampled* rollout: temperature 0.6,
top_p 0.98, bf16 autocast -- not comparable token-for-token), this runs upstream in fp32 with
GREEDY decoding (``top_k=1`` through upstream's own ``sample_trajectories_from_data_with_vlm_rollout``,
so every upstream logits processor / stopping criterion still applies) and a FIXED flow-matching
initial noise (``torch.Generator().manual_seed(noise_seed)``), and dumps the intermediates the
Neuron port is compared against:

* ``fused_input_ids`` -- prompt ids AFTER upstream's ``fuse_traj_tokens`` (history-trajectory
  tokens written into the ``<|traj_history|>`` placeholders);
* ``vision`` -- the vision tower's merged image embeddings and DeepStack features;
* ``hidden`` -- the text tower's per-layer hidden states for the prompt (``output_hidden_states``),
  and ``prefill_logits_last`` -- the logits at the last prompt position;
* ``sequences`` / ``offset`` / ``rope_deltas`` / ``prefill_seq_len`` -- the greedy rollout;
* ``noise``, per-Euler-step ``action_in`` / ``v`` (action_out_proj output), ``sampled_action``,
  ``pred_xyz`` / ``pred_rot``.

Private reference venv only (upstream needs hydra/scipy on Python 3.12)::

    <alpamayo1_5 venv>/bin/python examples/alpamayo/parity_ref.py \\
        --model <Alpamayo-1.5-10B dir> --backbone-config <Cosmos-Reason2-8B dir> \\
        --inputs ref_bf16_cpu.pt --out parity_fp32.pt
    # tiny checkpoint: --model <tiny dir> --backbone-config <tiny backbone dir> --build-inputs
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--backbone-config", required=True)
ap.add_argument("--inputs", default=None, help="a .pt holding model_inputs (e.g. ref_bf16_cpu.pt)")
ap.add_argument("--build-inputs", action="store_true", help="synthesize tiny-sized inputs instead")
ap.add_argument("--out", required=True)
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument(
    "--layers", default="all", help="'all' or comma list of hidden_states indices to save"
)
args = ap.parse_args()

cfg = json.load(open(f"{args.model}/config.json"))
cfg["vlm_name_or_path"] = args.backbone_config
cfg["attn_implementation"] = "sdpa"

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.config import Alpamayo1_5Config  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

c = Alpamayo1_5Config(**cfg)
t0 = time.time()
model = Alpamayo1_5.from_pretrained(args.model, config=c, dtype=torch.float32).float().eval()
print(f"[parity] loaded fp32 in {time.time() - t0:.1f}s", flush=True)

if args.inputs:
    model_inputs = torch.load(args.inputs, weights_only=False)["model_inputs"]
else:
    from PIL import Image

    helper.BASE_PROCESSOR_NAME = args.backbone_config
    rng = np.random.default_rng(0)
    n_cam, fpc, h, w = 2, 2, 64, 64
    n = n_cam * fpc
    frames = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]
    cam = torch.arange(n_cam)
    num_hist = c.tokens_per_history_traj
    image_content = helper._build_image_content(torch.zeros(n, 3, h, w), cam, fpc)
    it = iter(frames)
    for item in image_content:
        if item.get("type") == "image":
            item["image"] = next(it)
    hist = f"<|traj_history_start|>{'<|traj_history|>' * num_hist}<|traj_history_end|>"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content
            + [
                {
                    "type": "text",
                    "text": hist
                    + "output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    n_steps = (
        num_hist  # same as test_alpamayo_tiny_checkpoint (masked_scatter takes the first num_hist)
    )
    ego_history_xyz = torch.from_numpy(
        (rng.normal(size=(1, 1, n_steps, 3)) * 2.0).astype(np.float32)
    )
    q, _ = np.linalg.qr(rng.normal(size=(n_steps, 3, 3)).astype(np.float32))
    ego_history_rot = torch.from_numpy(q).unsqueeze(0).unsqueeze(0)
    model_inputs = {
        "tokenized_data": dict(inputs),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }

tok = model_inputs["tokenized_data"]
fused = model.fuse_traj_tokens(
    tok["input_ids"].clone(),
    {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    },
)
print(
    f"[parity] prompt {tuple(fused.shape)}; fused {(fused != tok['input_ids']).sum().item()} history tokens",
    flush=True,
)

# -- 1) prefill: vision outputs + every layer's hidden state for the prompt -----------------------
vis_out = {}


def _vis_hook(_m, _a, out):
    # Qwen3-VL visual returns (image_embeds, deepstack_feature_list) in transformers 4.57
    emb, deep = (out[0], out[1]) if isinstance(out, tuple) else (out, [])
    vis_out["image_embeds"] = emb.detach().float().clone()
    vis_out["deepstack"] = [d.detach().float().clone() for d in deep]


hv = model.vlm.model.visual.register_forward_hook(_vis_hook)
t0 = time.time()
with torch.no_grad():
    o = model.vlm(
        input_ids=fused,
        attention_mask=tok["attention_mask"],
        pixel_values=tok["pixel_values"],
        image_grid_thw=tok["image_grid_thw"],
        output_hidden_states=True,
        use_cache=False,
    )
hv.remove()
print(f"[parity] prefill fp32 in {time.time() - t0:.1f}s", flush=True)
keep = (
    range(len(o.hidden_states))
    if args.layers == "all"
    else [int(x) for x in args.layers.split(",")]
)
hidden = {i: o.hidden_states[i][0].float().clone() for i in keep}
prefill_logits_last = o.logits[0, -1].float().clone()
del o

# -- 2) greedy rollout + flow matching with fixed noise -------------------------------------------
n_wp = model.action_space.get_action_space_dims()[0]
noise = torch.randn(
    (1, *model.action_space.get_action_space_dims()),
    generator=torch.Generator().manual_seed(args.noise_seed),
)
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


steps = {"action_in": [], "v": []}
h1 = model.action_in_proj.register_forward_hook(
    lambda m, a, out: steps["action_in"].append(out.detach().float().clone())
)
h2 = model.action_out_proj.register_forward_hook(
    lambda m, a, out: steps["v"].append(out.detach().float().clone())
)
captured = {}
_orig_generate = model.vlm.generate


def _generate(*a, **kw):
    out = _orig_generate(*a, **kw)
    captured["sequences"] = out.sequences.clone()
    captured["prefill_seq_len"] = out.past_key_values.get_seq_length()
    captured["rope_deltas"] = model.vlm.model.rope_deltas.clone()
    return out


model.vlm.generate = _generate
from alpamayo1_5.diffusion import flow_matching as _fm  # noqa: E402

_orig_euler = _fm.FlowMatching._euler


def _euler(self, *a, **kw):
    out = _orig_euler(self, *a, **kw)
    captured["sampled_action"] = out.detach().float().clone()
    return out


_fm.FlowMatching._euler = _euler
torch.randn = _randn
t0 = time.time()
with torch.no_grad():
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        num_traj_samples=1,
        max_generation_length=c.tokens_per_future_traj,
        return_extra=True,
    )
torch.randn = _orig_randn
h1.remove()
h2.remove()
print(f"[parity] greedy rollout + expert in {time.time() - t0:.1f}s", flush=True)
eos = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
seq = captured["sequences"]
offset = int(model._find_eos_offset(sequences=seq, eos_token_id=eos, device=seq.device)[0])
print(
    f"[parity] generated {seq.shape[1] - fused.shape[1]} tokens, offset {offset}, cache {captured['prefill_seq_len']}"
)
print(f"[parity] CoC: {extra['cot'].reshape(-1)[0]!r:.200}")
print(f"[parity] pred_xyz[0,:3] {pred_xyz[0, 0, 0, :3].tolist()}")

torch.save(
    {
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
        "cot": extra["cot"],
        "noise_seed": args.noise_seed,
        "max_new_tokens": c.tokens_per_future_traj,
    },
    args.out,
)
print(f"[parity] saved {args.out}")
