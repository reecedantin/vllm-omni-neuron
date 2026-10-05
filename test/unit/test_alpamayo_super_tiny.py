# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Alpamayo 2 Super side of the Neuron Alpamayo port, on a tiny random-weight
checkpoint written under the REAL Super tensor names (``expert.expert.*``,
``expert.action_in_proj.*``, ``expert.action_out_proj.*``, plus the derived buffers the real
checkpoint omits) with an inline ``vlm_config`` / ``expert_config`` like the real ``config.json``.
No weights, tokenizer, upstream package or Neuron device needed. Covered:

* config parsing: generation cap 256, text EOS masked, ``traj_ids`` mapping, expert widths;
* strict weight-name mapping (Super prefixes) round trip;
* 45-token-style history tokenization (``pad_origin_at_beginning=false``) at ``history_id0``;
* the rollout never emits the masked text EOS and runs to the 256-token cap without
  ``<traj_future_start>``;
* TP=2 (gloo, 2 processes, sliced per-rank weight reads) matches TP=1.

Parity against upstream ``alpamayo2_super`` lives in ``test_alpamayo_super_upstream_tiny.py``.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

IMAGE_TOKEN, VISION_START, VISION_END = 151655, 151652, 151653
TEXT_EOS = 151645
TRAJ_IDS = {  # nvidia/Alpamayo2-Super config.json traj_ids
    "future_end": 155683,
    "future_id0": 152669,
    "future_pad": 155685,
    "future_start": 155681,
    "history_end": 155676,
    "history_id0": 151669,
    "history_pad": 155684,
    "history_start": 155674,
}
VOCAB = 155776
N_HIST_WP = 4  # history waypoints -> (4 - 1) deltas x 3 axes = 9 history tokens
N_WP = 8

_TEXT = {
    "model_type": "qwen3_vl_text",
    "attention_bias": False,
    "eos_token_id": TEXT_EOS,
    "head_dim": 32,
    "hidden_act": "silu",
    "hidden_size": 128,
    "intermediate_size": 256,
    "max_position_embeddings": 262144,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000,
    "vocab_size": VOCAB,
    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [6, 5, 5], "rope_type": "default"},
}
TINY_SUPER = {
    "architectures": ["Alpamayo2Super"],
    "model_type": "alpamayo2_super",
    "tokens_per_future_traj": 128,
    "tokens_per_history_traj": 3 * (N_HIST_WP - 1),
    "history_vocab_size": 1000,
    "future_vocab_size": 3000,
    "traj_vocab_size": 4000,
    "traj_ids": TRAJ_IDS,
    "hist_traj_tokenizer_cfg": {
        "_target_": "alpamayo2_super.models.delta_tokenizer.DeltaTrajectoryTokenizer",
        "pad_origin_at_beginning": False,
    },
    "vlm_config": {
        "model_type": "qwen3_vl",
        "image_token_id": IMAGE_TOKEN,
        "video_token_id": 151656,
        "vision_start_token_id": VISION_START,
        "vision_end_token_id": VISION_END,
        "text_config": _TEXT,
        "vision_config": {
            "model_type": "qwen3_vl",
            "deepstack_visual_indexes": [0, 1],
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 64,
            "in_channels": 3,
            "intermediate_size": 128,
            "num_heads": 4,
            "num_position_embeddings": 2304,
            "out_hidden_size": 128,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    },
    "expert_config": {
        "llm_config": {
            **_TEXT,
            "hidden_size": 96,  # != q width (4 x 32), like the real 1536 vs 2048
            "intermediate_size": 192,
            "num_attention_heads": 4,
        },
        "action_space_cfg": {
            "n_waypoints": N_WP,
            "dt": 0.1,
            "accel_bounds": [-9.8, 9.8],
            "curvature_bounds": [-0.33, 0.33],
            "accel_mean": 0.03,
            "accel_std": 0.68,
            "curvature_mean": 0.0003,
            "curvature_std": 0.026,
        },
        "action_in_proj_cfg": {
            "hidden_size": 64,
            "max_freq": 100.0,
            "num_enc_layers": 2,
            "num_fourier_feats": 8,
        },
        "diffusion_cfg": {"int_method": "euler", "num_inference_steps": 3},
        "expert_non_causal_attention": True,
    },
}


def _checkpoint_key(name: str) -> str:
    """Inverse of the port's ``_map_key`` for the Super layout."""
    if name == "lm_head_weight":
        return "vlm.lm_head.weight"
    if name.startswith("vision."):
        return "vlm.model.visual." + name[len("vision.") :]
    if name.startswith("text."):
        return "vlm.model.language_model." + name[len("text.") :]
    if name.startswith("expert."):
        return "expert.expert." + name[len("expert.") :]
    return "expert." + name  # action_in_proj.* / action_out_proj.*


def build_tiny_super_checkpoint(path: str, seed: int = 0, eos_boost: bool = False) -> str:
    """Random weights under the real Super tensor names. ``eos_boost`` makes the LM head favour the
    text EOS, so only the logit mask keeps the rollout from emitting it."""
    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    os.makedirs(path, exist_ok=True)
    json.dump(TINY_SUPER, open(os.path.join(path, "config.json"), "w"))
    g = torch.Generator().manual_seed(seed)
    m = NeuronAlpamayo1_5(AlpamayoConfig.from_model_dir(path), dtype=torch.float32)
    sd = {}
    for name, p in m.state_dict().items():
        if name.endswith(".freqs"):
            continue
        if p.dim() == 1 and ("norm" in name or name.endswith("layernorm.weight")):
            t = 1.0 + 0.05 * torch.randn(p.shape, generator=g)
        else:
            t = 0.08 * torch.randn(p.shape, generator=g)
        sd[_checkpoint_key(name)] = t.contiguous()
    if eos_boost:
        sd["vlm.lm_head.weight"][TEXT_EOS] *= 50.0
    # derived buffers upstream's state_dict carries (and the released checkpoint omits)
    sd["expert.action_space.accel_mean"] = torch.tensor(0.03)
    sd["expert.action_in_proj.timestep_fourier_encoder.freqs"] = torch.ones(1, 4)
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    return path


def synthetic_observation(seed: int = 0, n_images: int = 2) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = [151644, 872, 198]
    for _ in range(n_images):
        ids += [VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END]
    n_hist = TINY_SUPER["tokens_per_history_traj"]
    ids += (
        [TRAJ_IDS["history_start"]]
        + [TRAJ_IDS["history_pad"]] * n_hist
        + [TRAJ_IDS["history_end"]]
        + [3838, 374, 279, 1790, 30]
    )
    input_ids = torch.tensor([ids])
    t = torch.arange(-(N_HIST_WP - 1), 1, dtype=torch.float32) * 0.1
    xyz = torch.stack([8.0 * t, 0.1 * t, torch.zeros_like(t)], -1)[None, None]
    return {
        "input_ids": input_ids.numpy(),
        "attention_mask": torch.ones_like(input_ids).numpy(),
        "pixel_values": torch.randn(16 * n_images, 3 * 2 * 16 * 16, generator=g).numpy(),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * n_images).numpy(),
        "ego_history_xyz": xyz.numpy(),
        "ego_history_rot": torch.eye(3).expand(1, 1, N_HIST_WP, 3, 3).numpy(),
    }


def _inputs(obs: dict) -> dict:
    return {
        k: torch.as_tensor(obs[k])
        for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return build_tiny_super_checkpoint(str(tmp_path_factory.mktemp("tiny-super")))


@pytest.fixture(scope="module")
def tiny_model(tiny_dir):
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    return NeuronAlpamayo1_5.from_pretrained(tiny_dir, dtype=torch.float32)


def test_super_config(tiny_dir):
    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    c = AlpamayoConfig.from_model_dir(tiny_dir)
    assert c.variant == "alpamayo2_super"
    assert c.max_new_tokens == 256  # upstream: max(256, tokens_per_future_traj)
    assert c.head["masked_token_ids"] == [TEXT_EOS] and c.head["stop_token_ids"] == [TEXT_EOS]
    assert c.head["traj_token_start_idx"] == TRAJ_IDS["history_id0"]
    assert c.head["hist_token_start_idx"] == TRAJ_IDS["history_id0"]
    assert c.head["traj_vocab_size"] == 4000 and c.head["fourier_freqs_bf16"] is False
    assert c.extra["traj_token_ids"]["history"] == TRAJ_IDS["history_pad"]
    assert c.extra["traj_token_ids"]["future_start"] == TRAJ_IDS["future_start"]
    assert c.expert["hidden_size"] == 96 and c.expert["num_hidden_layers"] == 2


def test_super_strict_weight_mapping(tiny_dir, tiny_model):
    from safetensors import safe_open

    with safe_open(os.path.join(tiny_dir, "model.safetensors"), "pt") as fh:
        keys = set(fh.keys())
    sd = tiny_model.state_dict()
    assert {_checkpoint_key(k) for k in sd if not k.endswith(".freqs")} <= keys
    with safe_open(os.path.join(tiny_dir, "model.safetensors"), "pt") as fh:
        w = fh.get_tensor("expert.expert.layers.1.self_attn.q_proj.weight")
    assert torch.equal(sd["expert.layers.1.self_attn.q_proj.weight"], w)
    freqs = tiny_model.action_in_proj.timestep_fourier_encoder.freqs  # rebuilt, fp32 (not 1s)
    assert torch.allclose(freqs, torch.logspace(0, 2, 4)[None])


def test_super_history_tokens(tiny_model):
    obs = synthetic_observation()
    ids = tiny_model.fuse_history(
        torch.as_tensor(obs["input_ids"]),
        torch.as_tensor(obs["ego_history_xyz"]),
        torch.as_tensor(obs["ego_history_rot"]),
    )
    xyz = obs["ego_history_xyz"][0, 0]
    d = xyz[1:] - xyz[:-1]  # consecutive deltas only (pad_origin_at_beginning=false)
    lo, hi = np.array([-4, -4, -10.0]), np.array([4, 4, 10.0])
    want = np.clip(np.round((d - lo) / (hi - lo) * 999), 0, 999).astype(np.int64).reshape(-1)
    hist = ids[0][(torch.as_tensor(obs["input_ids"])[0] == TRAJ_IDS["history_pad"])]
    assert hist.tolist() == (want + TRAJ_IDS["history_id0"]).tolist()


def test_super_rollout_masks_text_eos(tmp_path):
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    path = build_tiny_super_checkpoint(str(tmp_path / "eos"), eos_boost=True)
    m = NeuronAlpamayo1_5.from_pretrained(path, dtype=torch.float32)
    obs = synthetic_observation()
    lb = m._logit_bias()
    assert lb[0, TEXT_EOS] < -1e29 and lb[0, TRAJ_IDS["future_id0"]] < -1e29
    r = m.get_action(
        _inputs(obs),
        ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
        ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
        noise=torch.zeros(1, N_WP, 2),
    )
    gen = r["generated"]
    assert TEXT_EOS not in gen.tolist()
    if TRAJ_IDS["future_start"] not in gen.tolist():
        assert gen.numel() == 256 and r["offset"] == r["prompt_len"] + 256
    assert torch.isfinite(r["pred_xyz"]).all() and tuple(r["pred_xyz"].shape) == (1, N_WP, 3)


def _tp_rank(rank, path, port, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), VLLM_NEURON_CPU_MODE="1")
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    m = NeuronAlpamayo1_5.from_pretrained(path, dtype=torch.float32, tp_group=dist.group.WORLD)
    obs = synthetic_observation()
    r = m.get_action(
        _inputs(obs),
        ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
        ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
        noise=torch.zeros(1, N_WP, 2),
    )
    if rank == 0:
        q.put((r["generated"].numpy().copy(), r["pred_xyz"].numpy().copy()))
    dist.destroy_process_group()


def test_super_tp2_matches_tp1(tiny_dir, tiny_model):
    import torch.multiprocessing as mp

    obs = synthetic_observation()
    ref = tiny_model.get_action(
        _inputs(obs),
        ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
        ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
        noise=torch.zeros(1, N_WP, 2),
    )
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (os.getpid() + 500) % 1000
    ps = [ctx.Process(target=_tp_rank, args=(r, tiny_dir, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    gen, xyz = q.get(timeout=600)
    for p in ps:
        p.join()
        assert p.exitcode == 0
    assert torch.equal(torch.as_tensor(gen), ref["generated"])
    rel = float((torch.as_tensor(xyz) - ref["pred_xyz"]).norm() / ref["pred_xyz"].norm())
    assert rel < 1e-4


def test_super_config_copy_is_independent():
    """The inline config dict is reused by every fixture; parsing must not mutate it."""
    before = copy.deepcopy(TINY_SUPER)
    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig

    AlpamayoConfig._from_alpamayo2("/nonexistent", TINY_SUPER)
    assert TINY_SUPER == before
