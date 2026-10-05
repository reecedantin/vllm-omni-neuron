# SPDX-License-Identifier: Apache-2.0
"""Upstream (NVlabs ``alpamayo2_super``) side of the Alpamayo 2 Super parity suite.

1. **Tiny checkpoint from upstream's own model class.** ``build_tiny_model`` takes the REAL
   ``nvidia/Alpamayo2-Super`` ``config.json`` (inline ``vlm_config`` + ``expert_config``), shrinks
   every width / depth (``TINY_*`` below) and instantiates upstream's ``Alpamayo2Super`` from it, so
   module/parameter names (``expert.expert.layers.*``, ``expert.action_in_proj.*``), layer types and
   tensor layout are exactly what the port sees from the real checkpoint. The trajectory vocabulary
   (1000 history + 3000 future bins, special tokens) is NOT shrunk: the tokenizer is the real one,
   so ``vocab_size`` and every ``traj_ids`` entry match the real checkpoint. As a script it writes
   the checkpoint (``model.safetensors`` + ``config.json`` + the tokenizer/processor files)::

       <alpamayo2_super venv>/bin/python test/unit/test_alpamayo_super_upstream_tiny.py \\
           --real-config <Alpamayo2-Super dir> --out tiny-alpamayo2-super

2. **Upstream forward test** (strict load round trip, both stages run): needs upstream's own
   dependencies (Python 3.12, hydra, transformers 4.57, ``alpamayo2_super``) and
   ``$ALPAMAYO2_SUPER_CONFIG`` (a directory holding the real ``config.json`` and tokenizer files);
   skips anywhere else.

3. **Port-vs-upstream parity gate**: runs in the plugin's environment against a dump written by
   ``examples/alpamayo/parity_ref_super.py`` (upstream, fp32, greedy, fixed noise) when
   ``$ALPAMAYO2_PARITY_MODEL`` (checkpoint dir) and ``$ALPAMAYO2_PARITY_REF`` (dump) are set;
   skips otherwise. Same thresholds as the Alpamayo 1.5 gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Files the upstream processor/tokenizer read from the checkpoint directory.
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "generation_config.json",
)

TINY_LAYERS = 4  # VLM text layers == expert layers (the expert reads one VLM cache per layer)
TINY_TEXT = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 8,  # GQA ratio 4 like the real 64 q / 8 KV heads ... scaled to 8 / 2
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_hidden_layers": TINY_LAYERS,
}
TINY_VISION = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "depth": TINY_LAYERS,
    "deepstack_visual_indexes": [0, 1, 2],
    "out_hidden_size": TINY_TEXT["hidden_size"],
}
TINY_EXPERT = {  # real: hidden 1536 < q width 16 x 128 = 2048; keep hidden != q width here too
    "hidden_size": 96,
    "intermediate_size": 192,
    "num_attention_heads": 4,
    "num_key_value_heads": TINY_TEXT["num_key_value_heads"],
    "head_dim": TINY_TEXT["head_dim"],
    "num_hidden_layers": TINY_LAYERS,
}
TINY_HEAD = {
    # small frames: 128x192 px -> 8x12 patches -> 24 LLM tokens per frame
    "min_pixels": 16384,
    "max_pixels": 32768,
    "action_in_proj_cfg": {"hidden_size": 64, "num_enc_layers": 2, "num_fourier_feats": 8},
    "n_waypoints": 16,
}


def tiny_config_dict(real_config: dict) -> dict:
    """The real ``config.json`` dict with every width / depth shrunk (``TINY_*``)."""
    cfg = copy.deepcopy(real_config)
    cfg["vlm_config"]["text_config"].update(TINY_TEXT)
    cfg["vlm_config"]["vision_config"].update(TINY_VISION)
    ex = cfg["expert_config"]
    ex["llm_config"].update(TINY_EXPERT)
    ex["expert_update_cfg"] = {
        k: TINY_EXPERT[k]
        for k in ("head_dim", "hidden_size", "intermediate_size", "num_attention_heads")
    }
    ex["action_in_proj_cfg"].update(TINY_HEAD["action_in_proj_cfg"])
    ex["action_space_cfg"]["n_waypoints"] = TINY_HEAD["n_waypoints"]
    cfg["future_traj_tokenizer_cfg"]["action_space_cfg"]["n_waypoints"] = TINY_HEAD["n_waypoints"]
    cfg["min_pixels"], cfg["max_pixels"] = TINY_HEAD["min_pixels"], TINY_HEAD["max_pixels"]
    return cfg


def build_tiny_model(real_dir: str, seed: int = 0):
    """Upstream ``Alpamayo2Super`` built from the shrunk real config. ``_name_or_path`` points at
    ``real_dir`` only so upstream can build its tokenizer (it reads the checkpoint's own files)."""
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    with open(os.path.join(real_dir, "config.json")) as f:
        raw = tiny_config_dict(json.load(f))
    for k in ("architectures", "transformers_version", "model_type"):
        raw.pop(k, None)
    config = Alpamayo2SuperConfig(**raw)
    config._name_or_path = real_dir
    config._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    model = Alpamayo2Super(config).float().eval()
    # random init leaves the RMSNorm weights at 1 and every bias at 0; perturb them so the parity
    # gate exercises them (a wrongly mapped norm or bias would otherwise compare equal)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 or name.endswith(".bias"):
                p.add_(0.1 * torch.randn(p.shape, generator=g))
    return model


def save_tiny(model, real_dir: str, out: str) -> None:
    """``model.safetensors`` + ``config.json`` + the real tokenizer/processor files (copied: the
    tiny checkpoint must be self-contained, like the real one)."""
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    # the real checkpoint omits the derived buffers upstream lists as ignorable (Fourier freqs, the
    # action-space normalization constants); keep the tiny one identical
    skip = [re.compile(p) for p in type(model)._keys_to_ignore_on_load_missing]
    sd = {
        k: v.contiguous()
        for k, v in model.state_dict().items()
        if not any(p.fullmatch(k) for p in skip)
    }
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg["architectures"] = ["Alpamayo2Super"]
    cfg.pop("_name_or_path", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    for fn in TOKENIZER_FILES:
        src = os.path.join(real_dir, fn)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, fn))


def synthetic_sample(n_cameras: int, n_frames: int, h: int, w: int, seed: int = 0) -> dict:
    """A PhysicalAI-AV-shaped sample (``load_physical_aiavdataset`` keys the trajectory prompt
    reads): smooth random colour fields as camera frames (uint8 ``[N_cam, N_frame, 3, H, W]``), the
    driving-profile camera ids, and a 16-waypoint ego history ending at the origin (~8 m/s, gentle
    curve) with yaw-only rotations."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    frames = np.empty((n_cameras, n_frames, 3, h, w), dtype=np.uint8)
    for c in range(n_cameras):
        for t in range(n_frames):
            a = rng.uniform(0.2, 1.0, size=(3, 3))
            img = a[:, 0, None, None] * yy + a[:, 1, None, None] * xx + a[:, 2, None, None]
            img = img + 0.05 * rng.standard_normal((3, h, w))
            frames[c, t] = np.clip(255 * img / img.max(), 0, 255).astype(np.uint8)
    t = np.arange(-15, 1) * 0.1
    yaw = 0.02 * t
    xyz = np.stack([8.0 * t, 0.5 * (8.0 * t) ** 2 * 0.004, np.zeros_like(t)], -1)
    rot = np.zeros((16, 3, 3), dtype=np.float32)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1] = (
        np.cos(yaw),
        -np.sin(yaw),
        np.sin(yaw),
        np.cos(yaw),
    )
    rot[:, 2, 2] = 1.0
    camera_ids = [0, 1, 2, 3, 5, 6][:n_cameras]
    return {
        "image_frames": torch.from_numpy(frames),
        "camera_indices": torch.tensor(camera_ids, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(xyz.astype(np.float32))[None, None],
        "ego_history_rot": torch.from_numpy(rot)[None, None],
    }


_UPSTREAM = True
try:
    import alpamayo2_super  # noqa: F401
    import hydra  # noqa: F401
except ImportError:
    _UPSTREAM = False
_REAL = os.environ.get("ALPAMAYO2_SUPER_CONFIG", "")
_needs_upstream = pytest.mark.skipif(
    not (_UPSTREAM and os.path.isfile(os.path.join(_REAL, "config.json"))),
    reason="needs the alpamayo2_super package + $ALPAMAYO2_SUPER_CONFIG (see module docstring)",
)


@_needs_upstream
def test_tiny_super_strict_load_and_both_stages(tmp_path):
    from alpamayo2_super import helper
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    model = build_tiny_model(_REAL)
    save_tiny(model, _REAL, str(tmp_path))
    back = Alpamayo2Super.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    sd, sd2 = model.state_dict(), back.state_dict()
    assert sd.keys() == sd2.keys() and all(torch.equal(sd[k], sd2[k]) for k in sd)
    data = synthetic_sample(2, 2, 128, 192)
    mi = helper.prepare_model_inputs(data, back.config, back.tokenizer)
    with torch.no_grad():
        xyz, rot, _ = back.sample_trajectories_from_data(
            mi, top_k=1, top_p=1.0, temperature=1.0, max_generation_length=16
        )
    assert tuple(xyz.shape) == (1, 1, 1, TINY_HEAD["n_waypoints"], 3)
    assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()


_PARITY_MODEL = os.environ.get("ALPAMAYO2_PARITY_MODEL", "")
_PARITY_REF = os.environ.get("ALPAMAYO2_PARITY_REF", "")


@pytest.mark.skipif(
    not (os.path.isdir(_PARITY_MODEL) and os.path.isfile(_PARITY_REF)),
    reason="needs $ALPAMAYO2_PARITY_MODEL and a parity_ref_super.py dump in $ALPAMAYO2_PARITY_REF",
)
def test_port_matches_upstream_super_layer_by_layer():
    import importlib.util

    os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
    spec = importlib.util.spec_from_file_location(
        "_alpamayo_parity_check", os.path.join(_ROOT, "examples", "alpamayo", "parity_check.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    r = mod.compare(_PARITY_MODEL, _PARITY_REF, torch.float32)
    assert r["fused_ids_equal"]
    assert r["vision_rel"] < 1e-4 and max(r["deepstack_rel"]) < 1e-4, r
    assert max(r["layer_rel"].values()) < 1e-4, r["layer_rel"]
    assert r["logits_rel"] < 1e-4 and r["logits_argmax_equal"], r
    assert max(r["expert_v_rel"]) < 1e-3, r["expert_v_rel"]
    assert r["offset"][0] == r["offset"][1]
    assert r["traj_rel_teacher_forced"] < 1e-3, r
    assert r["greedy_tokens_equal"], r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-config", required=True, help="the real Alpamayo2-Super checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = build_tiny_model(a.real_config, a.seed)
    save_tiny(m, a.real_config, a.out)
    print(
        f"saved tiny Alpamayo 2 Super to {a.out}: {sum(p.numel() for p in m.parameters())} params"
    )
