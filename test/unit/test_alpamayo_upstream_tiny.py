# SPDX-License-Identifier: Apache-2.0
"""Upstream (NVlabs ``alpamayo1_5``) side of the Alpamayo 1.5 parity suite.

1. **Tiny checkpoint from upstream's own model class.** ``build_tiny_model`` shrinks the real
   backbone/expert configs (``tiny_models.shrink_layer_counts`` / ``apply_overrides``) and
   instantiates upstream's ``Alpamayo1_5`` against them, so module/parameter names, layer types and
   tensor layout are exactly what the port sees from a real checkpoint. As a script it writes the
   checkpoint (``model.safetensors`` + ``config.json``)::

       <alpamayo1_5 venv>/bin/python test/unit/test_alpamayo_upstream_tiny.py \\
           --backbone-config <Cosmos-Reason2-8B or Qwen3-VL-8B-Instruct dir> --out tiny-alpamayo1_5

2. **Upstream forward tests** (strict load round trip, both stages run): need upstream's own
   dependencies (Python 3.12, hydra, scipy, ``alpamayo1_5``) and ``$ALPAMAYO_BACKBONE_CONFIG``; they
   skip anywhere else.

3. **Port-vs-upstream parity gate** (component -> single step -> end to end): runs in the plugin's
   environment against a dump written by ``examples/alpamayo/parity_ref.py`` (upstream, fp32, greedy,
   fixed noise) when ``$ALPAMAYO_PARITY_MODEL`` (checkpoint dir) and ``$ALPAMAYO_PARITY_REF`` (dump)
   are set; it skips otherwise. Thresholds: every text-layer hidden state, the vision/DeepStack
   features and the last-position logits within 1e-4 relative L2; teacher-forced trajectory within
   1e-3; greedy Chain-of-Causation tokens identical.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pytest
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _load(name: str, *parts: str):
    """Import one file by path -- without putting its directory on ``sys.path`` (``vllm_omni_neuron/``
    holds a ``platform.py`` that would shadow the stdlib module in every spawned subprocess). The
    upstream environment cannot import the ``vllm_omni_neuron`` package itself, so these files are
    loaded standalone; both are dependency-light (stdlib / transformers)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, *parts))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


_tiny_models = _load("_alpamayo_tiny_models", "vllm_omni_neuron", "tiny_models.py")
apply_overrides, shrink_layer_counts = (
    _tiny_models.apply_overrides,
    _tiny_models.shrink_layer_counts,
)

# Real nvidia/Cosmos-Reason2-8B == Qwen/Qwen3-VL-8B-Instruct config (verified byte-identical,
# see vllm_omni_neuron/diffusion/models/alpamayo/config.py). Shrink depth/hidden here; the tiny
# backbone has its own tiny vocab too (independent of the real 151936/155697 split).
TINY_BACKBONE_OVERRIDES = {
    "text_config.hidden_size": 128,
    "text_config.intermediate_size": 256,
    "text_config.num_attention_heads": 4,
    "text_config.num_key_value_heads": 2,
    "text_config.head_dim": 32,
    # vocab_size is NOT shrunk: the real tokenizer (shared with the full model, see config.py's
    # byte-identical Cosmos-Reason2-8B / Qwen3-VL-8B-Instruct finding) produces real-range token
    # ids, so embed_tokens/lm_head must stay the real width even though every other dim is tiny.
    "vision_config.hidden_size": 64,
    "vision_config.intermediate_size": 128,
    "vision_config.num_heads": 4,
    "vision_config.out_hidden_size": 128,
    "vision_config.deepstack_visual_indexes": [1, 3, 5],
}
TINY_LAYERS = 4  # text_config.num_hidden_layers AND vision_config.depth, via shrink_layer_counts
TINY_EXPERT = {
    "head_dim": 32,
    "hidden_size": 96,
    "intermediate_size": 192,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}
TINY_TRAJ_VOCAB_SIZE = 32  # upstream's own tokenizer.add_tokens([f"<i{v}>" for v in range(this)])
# Real config: traj_tokenizer_cfg.num_bins=3000 (discrete FUTURE trajectory bins) + hist_traj_
# tokenizer's default num_bins=1000 (discrete HISTORY bins, reusing the same <i0>..<i3999> id block
# via hist_token_start_idx = traj_token_start_idx + traj_tokenizer.vocab_size) both fit inside
# traj_vocab_size=4000. Keep that same "both tokenizers' bins fit inside traj_vocab_size" shape at
# tiny scale instead of leaving hist_traj_tokenizer_cfg at its own (much larger) class default.
TINY_FUTURE_BINS = 20
TINY_HIST_BINS = 8
assert TINY_FUTURE_BINS + TINY_HIST_BINS <= TINY_TRAJ_VOCAB_SIZE


def _probe_tokenizer_ids(tokenizer_dir: str, traj_vocab_size: int, add_special_tokens: bool):
    """Run the SAME vocabulary-extension code the real checkpoint's tokenizer went through (see
    ``vllm_omni_neuron.diffusion.models.alpamayo.config.build_tokenizer``), at a smaller
    ``traj_vocab_size``, and read back the real resulting ids -- so the tiny checkpoint's
    ``traj_token_start_idx``/``traj_token_ids``/``vocab_size`` are exactly what upstream's own
    tokenizer would produce, not hand-picked numbers that could silently drift from the real
    extension logic."""
    build_tokenizer = _load(
        "_alpamayo_config", "vllm_omni_neuron", "diffusion", "models", "alpamayo", "config.py"
    ).build_tokenizer

    tok = build_tokenizer(tokenizer_dir, traj_vocab_size, add_special_tokens)
    return len(tok), tok.traj_token_start_idx, dict(tok.traj_token_ids)


# A real config.json's own key names (see alpamayo-1.5-10b/config.json): trimmed to what matters at
# tiny scale -- a short action horizon / history, few trajectory bins, fewer diffusion steps.
# traj_token_start_idx / traj_token_ids / vocab_size are filled in by build_tiny_alpamayo1_5_config
# (below) from a real tokenizer probe, not hardcoded here.
TINY_HEAD_OVERRIDES = {
    "tokens_per_future_traj": 8,
    "tokens_per_history_traj": 4,
    "traj_vocab_size": TINY_TRAJ_VOCAB_SIZE,
    "action_space_cfg": {
        "_target_": "alpamayo1_5.action_space.UnicycleAccelCurvatureActionSpace",
        "n_waypoints": 8,
        "dt": 0.1,
        "accel_bounds": [-9.8, 9.8],
        "curvature_bounds": [-0.33, 0.33],
        "accel_mean": 0.0,
        "accel_std": 1.0,
        "curvature_mean": 0.0,
        "curvature_std": 1.0,
        "a_lambda": 1e-4,
        "a_ridge": 1e-4,
        "kappa_lambda": 1e-4,
        "kappa_ridge": 1e-4,
        "theta_lambda": 1e-6,
        "theta_ridge": 1e-8,
        "v_lambda": 1e-6,
        "v_ridge": 1e-4,
    },
    "action_in_proj_cfg": {
        "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
        "hidden_size": 64,
        "max_freq": 100.0,
        "num_enc_layers": 2,
        "num_fourier_feats": 8,
    },
    "action_out_proj_cfg": {"_target_": "torch.nn.Linear"},
    "diffusion_cfg": {
        "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
        "int_method": "euler",
        "num_inference_steps": 3,
    },
    "hist_traj_tokenizer_cfg": {
        "_target_": "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer",
        "num_bins": TINY_HIST_BINS,
    },
    "traj_tokenizer_cfg": {
        "_recursive_": False,
        "_target_": "alpamayo1_5.action_space.discrete_action_space.DiscreteTrajectoryTokenizer",
        "action_space_cfg": {
            "_target_": "alpamayo1_5.action_space.UnicycleAccelCurvatureActionSpace"
        },
        "dims_max": [10, 10],
        "dims_min": [-10, -10],
        "num_bins": TINY_FUTURE_BINS,
    },
    "expert_cfg": TINY_EXPERT,
    "expert_non_causal_attention": True,
    "keep_same_dtype": True,
    "add_special_tokens": True,
}


def build_tiny_alpamayo1_5_config(
    backbone_config_dir: str, seed: int = 0, out_dir: str | None = None
):
    """Upstream ``Alpamayo1_5Config`` with every layer count / hidden dim shrunk, built OFFLINE
    against a local backbone config directory (``vlm_name_or_path`` must be set before
    construction -- see ``examples/alpamayo/reference_1_5.py`` for why). ``vocab_size`` and the
    trajectory-token ids are derived from a REAL run of the tokenizer vocabulary-extension code at
    ``TINY_TRAJ_VOCAB_SIZE`` (``_probe_tokenizer_ids``), not hand-picked -- so the tiny checkpoint's
    ``embed_tokens``/``lm_head`` width and the special-token ids it uses are exactly what the real
    extension logic would produce at that smaller trajectory vocabulary.

    ``out_dir``, if given, is where the tiny BACKBONE config directory is written (it must outlive
    this process if the resulting checkpoint will be read by a separate device job later); defaults
    to ``$TMPDIR`` (fine for same-process use, e.g. the CPU pytest fixture)."""
    from alpamayo1_5.config import Alpamayo1_5Config

    with open(os.path.join(backbone_config_dir, "config.json")) as f:
        backbone = json.load(f)
    backbone = apply_overrides(backbone, TINY_BACKBONE_OVERRIDES)
    backbone, _ = shrink_layer_counts(backbone, TINY_LAYERS)

    vocab_size, traj_token_start_idx, traj_token_ids = _probe_tokenizer_ids(
        backbone_config_dir, TINY_TRAJ_VOCAB_SIZE, TINY_HEAD_OVERRIDES["add_special_tokens"]
    )
    backbone["text_config"]["vocab_size"] = vocab_size

    # Alpamayo1_5Config pulls the backbone architecture from vlm_name_or_path at construction time
    # (see examples/alpamayo/reference_1_5.py docstring), so write the tiny backbone config to its
    # own directory and point vlm_name_or_path there, instead of overriding it after the fact.
    tiny_backbone_dir = os.path.join(
        out_dir or os.environ.get("TMPDIR", "/tmp"), f"tiny-cosmos-reason2-{seed}"
    )
    os.makedirs(tiny_backbone_dir, exist_ok=True)
    with open(os.path.join(tiny_backbone_dir, "config.json"), "w") as f:
        json.dump(backbone, f)
    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = os.path.join(backbone_config_dir, fn)
        if os.path.isfile(src) and not os.path.islink(os.path.join(tiny_backbone_dir, fn)):
            os.symlink(src, os.path.join(tiny_backbone_dir, fn))

    cfg = copy.deepcopy(TINY_HEAD_OVERRIDES)
    cfg["vlm_name_or_path"] = tiny_backbone_dir
    cfg["attn_implementation"] = "sdpa"
    cfg["vocab_size"] = vocab_size
    cfg["traj_token_start_idx"] = traj_token_start_idx
    cfg["traj_token_ids"] = traj_token_ids
    return Alpamayo1_5Config(**cfg), tiny_backbone_dir


def build_tiny_model(backbone_config_dir: str, seed: int = 0, out_dir: str | None = None):
    import torch as _torch
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    _torch.manual_seed(seed)
    config, tiny_backbone_dir = build_tiny_alpamayo1_5_config(
        backbone_config_dir, seed, out_dir=out_dir
    )
    model = Alpamayo1_5(config)
    return model.eval(), tiny_backbone_dir


_UPSTREAM = True
try:
    import alpamayo1_5  # noqa: F401
    import hydra  # noqa: F401
except ImportError:
    _UPSTREAM = False
_BACKBONE = os.environ.get("ALPAMAYO_BACKBONE_CONFIG", "")
_needs_upstream = pytest.mark.skipif(
    not (_UPSTREAM and os.path.isdir(_BACKBONE)),
    reason="needs the alpamayo1_5 package + $ALPAMAYO_BACKBONE_CONFIG (see module docstring)",
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    model, tiny_backbone_dir = build_tiny_model(
        _BACKBONE, seed=0, out_dir=str(tmp_path_factory.mktemp("tiny-bb"))
    )
    return model, tiny_backbone_dir


def _build_messages(
    helper_mod, camera_indices, frames_pil, num_history_tokens: int, n: int, h: int, w: int
):
    # helper.create_message hardcodes num_traj_token=48 (the REAL model's tokens_per_history_traj)
    # for the history placeholder; build the message by hand at the tiny config's own count.
    hist_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_history_tokens}<|traj_history_end|>"
    )
    prompt_text = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    image_content = helper_mod._build_image_content(
        torch.zeros(n, 3, h, w), camera_indices, n // camera_indices.shape[0]
    )
    img_iter = iter(frames_pil)
    for item in image_content:
        if item.get("type") == "image":
            item["image"] = next(img_iter)
    return [
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
            + [{"type": "text", "text": f"{hist_placeholder}{prompt_text}"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]


@_needs_upstream
def test_tiny_checkpoint_strict_load_and_architecture(tiny_checkpoint, tmp_path):
    """The tiny model's own state_dict must load back into itself with strict=True: proves the
    save/load round trip (and, by extension, the real-checkpoint load path the Neuron port uses)
    sees exactly the tensors the module tree declares -- no missing, no unexpected."""
    from safetensors.torch import load_file, save_file

    model, _ = tiny_checkpoint
    path = tmp_path / "model.safetensors"
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(path),
        metadata={"format": "pt"},
    )
    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=True)
    assert missing == [] and unexpected == []
    assert len(model.vlm.model.language_model.layers) == 4  # TINY_LAYERS
    assert len(model.vlm.model.visual.blocks) == 4


@_needs_upstream
def test_tiny_checkpoint_forward_both_stages(tiny_checkpoint):
    """VLM Chain-of-Causation rollout + flow-matching expert, both stages, on synthetic inputs,
    through upstream's own ``sample_trajectories_from_data_with_vlm_rollout``."""
    from alpamayo1_5 import helper
    from PIL import Image

    model, tiny_backbone_dir = tiny_checkpoint
    helper.BASE_PROCESSOR_NAME = tiny_backbone_dir

    rng = np.random.default_rng(0)
    n_cameras, frames_per_cam, h, w = 2, 2, 64, 64
    n = n_cameras * frames_per_cam
    frames_pil = [
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)
    ]
    camera_indices = torch.arange(n_cameras)

    num_hist = TINY_HEAD_OVERRIDES["tokens_per_history_traj"]
    messages = _build_messages(helper, camera_indices, frames_pil, num_hist, n, h, w)
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    ego_history_xyz = torch.from_numpy(
        (rng.normal(size=(1, 1, num_hist, 3)) * 2.0).astype(np.float32)
    )
    rand_mats = rng.normal(size=(num_hist, 3, 3)).astype(np.float32)
    q, _ = np.linalg.qr(rand_mats)
    ego_history_rot = torch.from_numpy(q).unsqueeze(0).unsqueeze(0)
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }

    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=model.config.tokens_per_future_traj,
            return_extra=True,
        )
    elapsed = time.time() - t0

    n_waypoints = model.config.action_space_cfg["n_waypoints"]
    assert tuple(pred_xyz.shape) == (1, 1, 1, n_waypoints, 3)
    assert tuple(pred_rot.shape) == (1, 1, 1, n_waypoints, 3, 3)
    assert torch.isfinite(pred_xyz).all() and torch.isfinite(pred_rot).all()
    assert "cot" in extra
    assert elapsed < 60, f"tiny forward took {elapsed:.1f}s, expected well under a minute"


_PARITY_MODEL = os.environ.get("ALPAMAYO_PARITY_MODEL", "")
_PARITY_REF = os.environ.get("ALPAMAYO_PARITY_REF", "")


@pytest.mark.skipif(
    not (os.path.isdir(_PARITY_MODEL) and os.path.isfile(_PARITY_REF)),
    reason="needs $ALPAMAYO_PARITY_MODEL and an examples/alpamayo/parity_ref.py dump in $ALPAMAYO_PARITY_REF",
)
def test_port_matches_upstream_layer_by_layer():
    os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
    compare = _load("_alpamayo_parity_check", "examples", "alpamayo", "parity_check.py").compare

    r = compare(_PARITY_MODEL, _PARITY_REF, torch.float32)
    # tier 1 -- components
    assert r["fused_ids_equal"]
    assert r["vision_rel"] < 1e-4 and max(r["deepstack_rel"]) < 1e-4, r
    assert max(r["layer_rel"].values()) < 1e-4, r["layer_rel"]
    # tier 2 -- one step: next-token logits and every expert velocity (teacher-forced)
    assert r["logits_rel"] < 1e-4 and r["logits_argmax_equal"], r
    assert max(r["expert_v_rel"]) < 1e-3, r["expert_v_rel"]
    # tier 3 -- end to end
    assert r["offset"][0] == r["offset"][1]
    assert r["traj_rel_teacher_forced"] < 1e-3, r
    assert r["greedy_tokens_equal"], r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--backbone-config",
        required=True,
        help="local Cosmos-Reason2-8B (or Qwen3-VL-8B-Instruct) config + tokenizer dir",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tiny_backbone_dir = build_tiny_model(a.backbone_config, a.seed, out_dir=a.out)
    model.config.vlm_name_or_path = (
        tiny_backbone_dir  # keep pointing at the tiny backbone, not the real one
    )

    # Not model.save_pretrained(): PreTrainedModel.save_pretrained diffs the config's generation
    # params against self.__class__() with NO ARGS, which falls back to Alpamayo1_5Config's own
    # built-in default vlm_name_or_path ("Qwen/Qwen3-VL-8B-Instruct") and tries to resolve it from
    # the Hub -- a network call this offline box refuses, unrelated to anything about THIS model
    # instance. Save the state dict + config directly instead; identical on-disk shape
    # (model.safetensors + config.json), just without that side effect.
    from safetensors.torch import save_file

    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f, indent=2, default=str)
    print(
        f"saved tiny Alpamayo 1.5 to {a.out}: {sum(p.numel() for p in model.parameters())} params, "
        f"backbone config at {tiny_backbone_dir}"
    )
