# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Neuron Alpamayo 1.5 port on a tiny random-weight checkpoint.

The checkpoint is generated here from the port's own module tree and written under the REAL
checkpoint's tensor names (``vlm.model.visual.*``, ``vlm.model.language_model.*``, ``vlm.lm_head``,
``expert.*``, ``action_in_proj.*``, ``action_out_proj.*``) plus a tiny ``backbone_config.json``, so
no weights, tokenizer, upstream package or Neuron device are needed. Covered:

* strict weight-name mapping round trip;
* the static decode (fixed-length cache, masks, one-hot writes) reproduces a full causal recompute;
* the vision tower's fused-qkv TP interleave and the TP=2 graphs match TP=1 (gloo, 2 processes);
* the served contract: ``robot_obs`` numpy payload -> ``output["actions"]`` -> vLLM-Omni's own
  formatter -> ``multimodal_output["actions"]``;
* every device graph traces with ``fullgraph=True`` and does not recompile across steps.

Parity against the upstream NVlabs implementation lives in ``test_alpamayo_upstream_tiny.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

IMAGE_TOKEN, VISION_START, VISION_END = 151655, 151652, 151653
TRAJ = {
    "history": 151716,
    "future": 151717,
    "history_start": 151706,
    "future_start": 151713,
    "history_end": 151708,
    "future_end": 151715,
}

TINY_HEAD = {
    "architectures": ["Alpamayo1_5"],
    "model_type": "alpamayo1_5",
    "vocab_size": 151729,
    "expert_cfg": {
        "head_dim": 32,
        "hidden_size": 96,
        "intermediate_size": 192,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    },
    "action_space_cfg": {
        "n_waypoints": 8,
        "dt": 0.1,
        "accel_bounds": [-9.8, 9.8],
        "curvature_bounds": [-0.33, 0.33],
        "accel_mean": 0.0,
        "accel_std": 1.0,
        "curvature_mean": 0.0,
        "curvature_std": 1.0,
    },
    "action_in_proj_cfg": {
        "hidden_size": 64,
        "max_freq": 100.0,
        "num_enc_layers": 2,
        "num_fourier_feats": 8,
    },
    "diffusion_cfg": {"int_method": "euler", "num_inference_steps": 3},
    "tokens_per_future_traj": 8,
    "tokens_per_history_traj": 4,
    "traj_vocab_size": 32,
    "traj_token_start_idx": 151669,
    "traj_token_ids": TRAJ,
    "hist_traj_tokenizer_cfg": {
        "_target_": "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer",
        "num_bins": 8,
    },
    "traj_tokenizer_cfg": {"num_bins": 20},
    "expert_non_causal_attention": True,
}
TINY_BACKBONE = {
    "model_type": "qwen3_vl",
    "image_token_id": IMAGE_TOKEN,
    "video_token_id": 151656,
    "vision_start_token_id": VISION_START,
    "vision_end_token_id": VISION_END,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "attention_bias": False,
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 128,
        "intermediate_size": 256,
        "max_position_embeddings": 262144,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "vocab_size": 151729,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [6, 5, 5],
            "rope_type": "default",
        },
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "deepstack_visual_indexes": [1, 3],
        "depth": 4,
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
}


def _checkpoint_key(name: str) -> str:
    """Inverse of ``NeuronAlpamayo1_5._map_key``."""
    if name == "lm_head_weight":
        return "vlm.lm_head.weight"
    if name.startswith("vision."):
        return "vlm.model.visual." + name[len("vision.") :]
    if name.startswith("text."):
        return "vlm.model.language_model." + name[len("text.") :]
    return name


def build_tiny_checkpoint(path: str, seed: int = 0) -> str:
    """Random-weight Alpamayo-1.5-shaped checkpoint at ``path`` (config.json, backbone_config.json,
    model.safetensors under the real tensor names)."""
    from safetensors.torch import save_file

    from vllm_omni_neuron.diffusion.models.alpamayo.config import AlpamayoConfig
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    os.makedirs(path, exist_ok=True)
    json.dump(TINY_HEAD, open(os.path.join(path, "config.json"), "w"))
    json.dump(TINY_BACKBONE, open(os.path.join(path, "backbone_config.json"), "w"))
    g = torch.Generator().manual_seed(seed)
    m = NeuronAlpamayo1_5(AlpamayoConfig.from_model_dir(path), dtype=torch.float32)
    sd = {}
    for name, p in m.state_dict().items():
        if p.dim() == 1 and ("norm" in name or name.endswith("layernorm.weight")):
            t = 1.0 + 0.05 * torch.randn(p.shape, generator=g)
        else:
            t = 0.08 * torch.randn(p.shape, generator=g)
        sd[_checkpoint_key(name)] = t.contiguous()
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    return path


def synthetic_observation(seed: int = 0, n_images: int = 2) -> dict:
    """A prompt with ``n_images`` 4x4-patch images, the history placeholders and a short text tail,
    as numpy arrays (the shape a served request carries)."""
    g = torch.Generator().manual_seed(seed)
    ids = [151644, 872, 198]
    for _ in range(n_images):
        ids += [VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END]
    ids += (
        [TRAJ["history_start"]]
        + [TRAJ["history"]] * 4
        + [TRAJ["history_end"]]
        + [3838, 374, 279, 1790, 30]
    )
    input_ids = torch.tensor([ids])
    grid = torch.tensor([[1, 4, 4]] * n_images)
    return {
        "input_ids": input_ids.numpy(),
        "attention_mask": torch.ones_like(input_ids).numpy(),
        "pixel_values": torch.randn(16 * n_images, 3 * 2 * 16 * 16, generator=g).numpy(),
        "image_grid_thw": grid.numpy(),
        "ego_history_xyz": (torch.randn(1, 1, 2, 3, generator=g) * 2).numpy(),
        "ego_history_rot": torch.linalg.qr(torch.randn(1, 1, 2, 3, 3, generator=g))[0].numpy(),
    }


def _inputs(obs: dict) -> dict:
    return {
        k: torch.as_tensor(obs[k])
        for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return build_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny-alpamayo")))


@pytest.fixture(scope="module")
def tiny_model(tiny_dir):
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    return NeuronAlpamayo1_5.from_pretrained(tiny_dir, dtype=torch.float32)


def test_strict_weight_mapping(tiny_dir, tiny_model):
    from safetensors import safe_open

    with safe_open(os.path.join(tiny_dir, "model.safetensors"), "pt") as f:
        keys = set(f.keys())
    mine = {_checkpoint_key(k) for k in tiny_model.state_dict()}
    assert keys == mine
    with safe_open(os.path.join(tiny_dir, "model.safetensors"), "pt") as f:
        w = f.get_tensor("vlm.model.language_model.layers.0.self_attn.q_proj.weight")
    assert torch.equal(tiny_model.text.layers[0].self_attn.q_proj.weight, w)


def test_static_decode_matches_full_recompute(tiny_model):
    """Every greedy decode step's logits equal a from-scratch prefill over prompt + generated tokens."""
    m = tiny_model
    obs = synthetic_observation()
    inp = _inputs(obs)
    out = m.get_action(
        inp,
        ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
        ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
    )
    gen = out["generated"]
    assert gen.numel() >= 2
    fused = m.fuse_history(inp["input_ids"], torch.as_tensor(obs["ego_history_xyz"]), None)
    assert torch.equal(fused, inp["input_ids"])  # no history -> no fusion
    fused = m.fuse_history(
        inp["input_ids"],
        torch.as_tensor(obs["ego_history_xyz"]),
        torch.as_tensor(obs["ego_history_rot"]),
    )
    for n in range(1, gen.numel()):
        ids = torch.cat([fused, gen[:n][None]], 1)
        _, _, _, logits, _ = m._prefill(
            ids,
            torch.ones_like(ids),
            inp["pixel_values"],
            inp["image_grid_thw"],
            bucket=ids.shape[1],
        )
        assert int(logits.argmax(-1)[0]) == int(gen[n]), f"step {n}"
    # the trajectory-token range is never generated
    start, nv = TINY_HEAD["traj_token_start_idx"], TINY_HEAD["traj_vocab_size"]
    assert not ((gen >= start) & (gen < start + nv)).any()


def test_bucket_padding_is_exact(tiny_model):
    m = tiny_model
    obs = synthetic_observation(seed=1)
    inp = _inputs(obs)
    kw = dict(
        ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
        ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
        noise=torch.randn(1, 8, 2, generator=torch.Generator().manual_seed(3)),
    )
    n = inp["input_ids"].shape[1]
    a = m.get_action(inp, bucket=n, **kw)
    b = m.get_action(inp, bucket=n + 37, **kw)
    assert torch.equal(a["generated"], b["generated"])
    assert torch.allclose(a["pred_xyz"], b["pred_xyz"], rtol=1e-4, atol=1e-5)


def test_qkv_tp_interleave():
    from vllm_omni_neuron.diffusion.models.alpamayo.model import _interleave_qkv_for_tp

    d, tp = 8, 2
    w = torch.arange(3 * d)[:, None].float()  # rows: q0..q7 k0..k7 v0..v7
    shard0 = _interleave_qkv_for_tp(w, tp)[: 3 * d // tp, 0].long().tolist()
    assert shard0 == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]


def test_served_contract_through_omni_formatter(tiny_dir, tmp_path, monkeypatch):
    """robot_obs (numpy) -> pipeline.forward -> output['actions'] -> vLLM-Omni's formatter ->
    multimodal_output['actions'] with the trajectory, raw action and CoC text; with
    ALPAMAYO_RANK_DIGEST_DIR set, each request also writes this rank's output digest."""
    from vllm_omni.diffusion.output_formatter import (
        _build_multimodal_output,
        normalize_diffusion_postprocess_output,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_neuron.diffusion.models.alpamayo.pipeline import NeuronAlpamayo1_5Pipeline

    pipe = NeuronAlpamayo1_5Pipeline(
        od_config=types.SimpleNamespace(model=tiny_dir, model_config={})
    )
    pipe._model._tokenizer = types.SimpleNamespace(
        decode=lambda ids: " ".join(map(str, ids))
    )  # no tokenizer files
    req = types.SimpleNamespace(
        sampling_params=OmniDiffusionSamplingParams(
            seed=0, extra_args={"robot_obs": synthetic_observation()}
        ),
        request_id="r0",
    )
    d = pipe.forward(req)
    assert d.error is None, d.error
    mm = _build_multimodal_output(normalize_diffusion_postprocess_output(d.output, {}), None)
    res = mm["actions"]
    assert tuple(res["pred_xyz"].shape) == (1, 8, 3) and tuple(res["actions"].shape) == (1, 8, 2)
    assert isinstance(res["cot"][0], str) and res["generated"].ndim == 1
    monkeypatch.setenv("ALPAMAYO_RANK_DIGEST_DIR", str(tmp_path))
    again = pipe.forward(req).output["actions"]
    assert (again["pred_xyz"] == res["pred_xyz"]).all()  # same seed -> same trajectory
    digest = json.loads((tmp_path / "req001_rank00.json").read_text())
    want = hashlib.sha256(np.ascontiguousarray(res["pred_xyz"]).tobytes()).hexdigest()
    assert digest["pred_xyz"] == want and digest["cot"] == res["cot"][0]


def test_graphs_trace_fullgraph_without_recompiles(tiny_dir):
    import torch._dynamo

    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    torch._dynamo.reset()
    m = NeuronAlpamayo1_5.from_pretrained(tiny_dir, dtype=torch.float32)
    ref = m.get_action(_inputs(synthetic_observation()), noise=torch.zeros(1, 8, 2))
    m.compile("eager")
    with torch._dynamo.config.patch(error_on_recompile=True):
        got = m.get_action(_inputs(synthetic_observation()), noise=torch.zeros(1, 8, 2))
    assert torch.equal(got["generated"], ref["generated"])
    assert torch.allclose(got["actions"], ref["actions"], rtol=1e-5, atol=1e-6)


def _tp_rank(rank, path, port, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), VLLM_NEURON_CPU_MODE="1")
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    obs = synthetic_observation()
    out = {}
    for dtype in (torch.float32, torch.float64):
        m = NeuronAlpamayo1_5.from_pretrained(path, dtype=dtype, tp_group=dist.group.WORLD)
        r = m.get_action(
            _inputs(obs),
            ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
            ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
            noise=torch.zeros(1, 8, 2),
        )
        out[dtype] = (
            r["generated"].numpy().copy(),
            r["pred_xyz"].numpy().copy(),
            m._debug["image_embeds"].numpy().copy(),
        )
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def test_tp2_matches_tp1(tiny_dir, tiny_model):
    """TP=2 equals TP=1. The sharding itself is checked in float64, where the split reductions
    agree to rounding (measured rel 0 on the trajectory, 5e-16 on the vision embeds), so a
    sharding bug cannot hide under a tolerance. The float32 run only bounds the trajectory: the
    8-waypoint integration amplifies ~3e-5 action noise to ~1.3e-4 on pred_xyz, which is the
    float32 floor of the UNSHARDED model itself (TP=1 fp32 vs TP=1 fp64: 1.32e-4; TP=1 fp32 at 1
    vs 16 threads: 8.8e-5; TP=2 fp32 vs TP=1 fp32: 1.29e-4), so the fp32 bar is 2x that floor."""
    import torch.multiprocessing as mp

    from vllm_omni_neuron.diffusion.models.alpamayo.model import NeuronAlpamayo1_5

    obs = synthetic_observation()

    def tp1(model):
        r = model.get_action(
            _inputs(obs),
            ego_history_xyz=torch.as_tensor(obs["ego_history_xyz"]),
            ego_history_rot=torch.as_tensor(obs["ego_history_rot"]),
            noise=torch.zeros(1, 8, 2),
        )
        return r, model._debug["image_embeds"]

    refs = {
        torch.float32: tp1(tiny_model),
        torch.float64: tp1(NeuronAlpamayo1_5.from_pretrained(tiny_dir, dtype=torch.float64)),
    }
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    ps = [ctx.Process(target=_tp_rank, args=(r, tiny_dir, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    out = q.get(timeout=600)
    for p in ps:
        p.join()
        assert p.exitcode == 0
    rel = lambda a, b: float((torch.as_tensor(a) - b).norm() / b.norm())  # noqa: E731
    bars = {torch.float64: (1e-12, 1e-10), torch.float32: (1e-5, 3e-4)}  # (vision, pred_xyz)
    for dtype, (vis_bar, xyz_bar) in bars.items():
        (ref, ref_vis), (gen, xyz, vis) = refs[dtype], out[dtype]
        assert rel(vis, ref_vis) < vis_bar, dtype  # vision qkv sharding
        assert torch.equal(torch.as_tensor(gen), ref["generated"]), dtype
        assert rel(xyz, ref["pred_xyz"]) < xyz_bar, dtype
