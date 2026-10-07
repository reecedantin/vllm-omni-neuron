# SPDX-License-Identifier: Apache-2.0
"""Upstream (NVlabs alpamayo1_5) CPU reference for Alpamayo 1.5 parity checks.

Runs the REAL upstream model class end to end (both stages: VLM Chain-of-Causation rollout, then
the flow-matching expert) on synthetic but shape-correct inputs -- no real dataset clip needed, since
this validates the port's forward pass against upstream exactly, not driving-prediction quality.

Must run in the private CPU-only reference venv (upstream pins Python 3.12 and depends on hydra,
scipy, physical-ai-av; none of that belongs in the shared Neuron venv):

    <alpamayo1_5 venv>/bin/python examples/alpamayo/reference_1_5.py \
        --model <Alpamayo-1.5-10B dir> --backbone-config <Cosmos-Reason2-8B or Qwen3-VL-8B-Instruct dir> \
        --out ref_bf16_cpu.pt

Two upstream quirks this script works around, neither of which is a port bug:
* ``Alpamayo1_5Config.__init__`` builds the HF processor at CONSTRUCTION time (not load time), so
  ``vlm_name_or_path`` must already point at a local, offline-resolvable directory before the config
  object is built -- not after.
* ``config.json`` declares ``attn_implementation: flash_attention_2`` (CUDA-only); overridden to
  ``sdpa`` here, exactly as upstream's own ``Alpamayo1_5.__init__`` already forces for the expert
  ("The diffusion expert does not support FlashAttention 2").
* The flow-matching sampler draws its initial noise with a bare ``torch.randn(..., device=device)``
  (always fp32) and relies on CUDA autocast to reconcile that against the bf16 action-projection
  weights; the CPU equivalent is ``torch.autocast("cpu", dtype=torch.bfloat16)`` around the sampling
  call (upstream's own ``test_inference.py`` does the CUDA version of exactly this).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument(
    "--backbone-config",
    required=True,
    help="local nvidia/Cosmos-Reason2-8B (or Qwen3-VL-8B-Instruct) dir",
)
ap.add_argument(
    "--image-processor-config",
    default=None,
    help="local dir for the Qwen3-VL-2B-Instruct image processor helper.get_processor hardcodes "
    "upstream; defaults to --backbone-config's own vision bits if unset",
)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-cameras", type=int, default=4)
ap.add_argument("--frames-per-camera", type=int, default=4)
ap.add_argument("--image-hw", default="180,320")
ap.add_argument("--num-history-steps", type=int, default=16)
args = ap.parse_args()

cfg = json.load(open(f"{args.model}/config.json"))
cfg["vlm_name_or_path"] = (
    args.backbone_config
)  # must be set BEFORE Alpamayo1_5Config() -- see docstring
cfg["attn_implementation"] = "sdpa"

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.config import Alpamayo1_5Config  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

c = Alpamayo1_5Config(**cfg)
t0 = time.time()
model = Alpamayo1_5.from_pretrained(args.model, config=c, dtype=torch.bfloat16).eval()
print(
    f"[ref] loaded bf16 in {time.time() - t0:.1f}s, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
)

rng = np.random.default_rng(args.seed)
h, w = (int(x) for x in args.image_hw.split(","))
n = args.n_cameras * args.frames_per_camera
frames_np = rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)
frames_pil = [Image.fromarray(f) for f in frames_np]
camera_indices = torch.arange(args.n_cameras)

helper.BASE_PROCESSOR_NAME = args.image_processor_config or args.backbone_config
messages = helper.create_message(
    frames=torch.zeros(n, 3, h, w),
    camera_indices=camera_indices,
    num_frames_per_camera=args.frames_per_camera,
)
# create_message only uses `frames` for its ndim assert and (if camera_indices is None) iteration;
# with camera_indices set it never reads frame content, so patch the real PIL images in afterwards.
img_iter = iter(frames_pil)
for item in messages[1]["content"]:
    if item.get("type") == "image":
        item["image"] = next(img_iter)

processor = helper.get_processor(model.tokenizer)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

num_hist = args.num_history_steps
ego_history_xyz = torch.from_numpy((rng.normal(size=(1, 1, num_hist, 3)) * 2.0).astype(np.float32))
rand_mats = rng.normal(size=(num_hist, 3, 3)).astype(np.float32)
q, _ = np.linalg.qr(rand_mats)  # random valid rotation matrices
ego_history_rot = torch.from_numpy(q).unsqueeze(0).unsqueeze(0)

model_inputs = helper.to_device(
    {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    },
    "cpu",
)

torch.manual_seed(42)
t0 = time.time()
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=c.tokens_per_future_traj,
        return_extra=True,
    )
print(f"[ref] forward in {time.time() - t0:.1f}s")
print(f"[ref] pred_xyz {tuple(pred_xyz.shape)}, pred_rot {tuple(pred_rot.shape)}")
print(f"[ref] CoC[0]: {extra['cot'][0]!r:.200}")

torch.save(
    {
        "pred_xyz": pred_xyz.float(),
        "pred_rot": pred_rot.float(),
        "cot": extra["cot"],
        "model_inputs": model_inputs,
        "seed": args.seed,
    },
    args.out,
)
print(f"[ref] saved {args.out}")
