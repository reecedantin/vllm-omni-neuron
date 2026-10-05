# SPDX-License-Identifier: Apache-2.0
"""Upstream (vLLM-Omni ``Gr00tPolicy``) CPU reference for GR00T parity checks.

Builds a deterministic observation, runs it through upstream's processor + collator to get
the exact model inputs, then runs upstream ``Gr00tN1d7`` on CPU (fp32 and bf16) with a fixed
initial noise. Saves ``{inputs, noise, action_pred, action_pred_bf16, embodiment_tag}`` for
``device_check.py --reference``. CPU only.

Upstream resolves the backbone config and processor from the Hub ids
``nvidia/Cosmos-Reason2-2B`` / ``Qwen/Qwen3-VL-2B-Instruct``. Offline, point ``--hf-root`` at
a directory containing those two relative paths (e.g. symlinks to a local
Qwen3-VL-2B-Instruct checkout); the script runs from there.

    python examples/gr00t/reference.py --model /path/to/gr00t-n17 --hf-root /path/to/hfroot --out ref.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--hf-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--embodiment-tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
ap.add_argument("--prompt", default="pick up the red cube and put it in the bowl")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--image-hw", default="180,320", help="raw camera frame size H,W")
ap.add_argument("--obs", default=None,
                help="a real observation instead of random frames: an .npz with one uint8 [H,W,3] array per "
                     "video key, a float32 'state' vector split over the state keys in order, and 'task'")
args = ap.parse_args()
model_dir, out = os.path.abspath(args.model), os.path.abspath(args.out)
os.chdir(args.hf_root)

from vllm_omni.diffusion.models.gr00t.dataio.types import MessageType  # noqa: E402
from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy, _rec_to_dtype  # noqa: E402



class _RawTag:
    """Stand-in for an ``EmbodimentTag`` the upstream enum does not list (e.g. GR00T-H's surgical
    embodiments, which are in the checkpoint's processor config but not in vLLM-Omni's enum)."""

    def __init__(self, value: str):
        self.value, self.name = value, value.upper()

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return getattr(other, "value", other) == self.value


from vllm_omni.diffusion.models.gr00t.dataio import embodiment_tags as _et  # noqa: E402

_resolve = _et.EmbodimentTag.resolve.__func__ if hasattr(_et.EmbodimentTag.resolve, "__func__") else None


def _resolve_or_raw(cls, tag):
    try:
        return _resolve(cls, tag)
    except (ValueError, KeyError):
        print(f"embodiment {tag!r} is not in upstream's EmbodimentTag enum; using it as a raw tag")
        return _RawTag(str(tag))


if _resolve is not None:
    _et.EmbodimentTag.resolve = classmethod(_resolve_or_raw)

t0 = time.time()
policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=model_dir, device="cpu", strict=True)
print(f"upstream load {time.time() - t0:.1f}s")
mc = policy.modality_configs
tag = policy.embodiment_tag.value
norm = policy.processor.state_action_processor.norm_params[tag]
rng = np.random.default_rng(args.seed)
h, w = (int(x) for x in args.image_hw.split(","))
t_video = len(mc["video"].delta_indices)
obs = {
    "video": {k: rng.integers(0, 255, (1, t_video, h, w, 3), dtype=np.uint8) for k in mc["video"].modality_keys},
    "state": {},
    "language": {mc["language"].modality_keys[0]: [[args.prompt]]},
}
if args.obs:
    real = np.load(args.obs)
    obs["video"] = {k: real[k][None, None] for k in mc["video"].modality_keys}  # [B=1, T=1, H, W, 3]
    obs["language"] = {mc["language"].modality_keys[0]: [[str(real["task"])]]}
    vec, start = real["state"].astype(np.float32), 0
for k in mc["state"].modality_keys:
    dim = int(np.asarray(norm["state"][k]["dim"]))
    if args.obs:
        obs["state"][k] = vec[start:start + dim].reshape(1, 1, dim)
        start += dim
    else:
        obs["state"][k] = (rng.normal(size=(1, len(mc["state"].delta_indices), dim)) * 0.2).astype(np.float32)
policy.check_observation(obs)

# upstream _get_action steps 1-3 (unbatch -> processor -> collate)
step = policy._to_vla_step_data(policy._unbatch_observation(obs)[0])
collated = policy.collate_fn([policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])])
inputs = {k: v for k, v in collated["inputs"].items()}

model = policy.model
cfg = model.config
noise = torch.randn((1, cfg.action_horizon, cfg.max_action_dim), generator=torch.Generator().manual_seed(args.seed + 1))

import vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 as up  # noqa: E402

real_randn = torch.randn


def run(dtype):
    model.to(dtype)
    up.torch.randn = lambda *a, **k: noise.to(k.get("dtype") or torch.float32)
    try:
        t = time.time()
        with torch.inference_mode():
            pred = model.get_action(_rec_to_dtype(dict(inputs), dtype=dtype))["action_pred"].float()
        print(f"upstream {dtype} get_action {time.time() - t:.2f}s")
    finally:
        up.torch.randn = real_randn
    return pred


pred32 = run(torch.float32)
pred16 = run(torch.bfloat16)
decoded = policy.processor.decode_action(pred32.numpy(), policy.embodiment_tag,
                                         {k: v for k, v in obs["state"].items()})
inputs_cpu = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in inputs.items()}
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
torch.save({"inputs": inputs_cpu, "noise": noise, "action_pred": pred32, "action_pred_bf16": pred16,
            "decoded": decoded, "embodiment_tag": tag, "obs": obs}, out)
rel = ((pred16 - pred32).norm() / pred32.norm()).item()
print(f"saved {out}: action_pred {tuple(pred32.shape)}, upstream bf16-vs-fp32 rel {rel:.5f}, "
      f"seq {inputs['input_ids'].shape[1]}")
