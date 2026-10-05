# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight GR00T N1.7 / GR00T-H checkpoints, and CPU parity of the Neuron port
against upstream vLLM-Omni's ``Gr00tN1d7`` (the CUDA reference implementation, run on CPU).

The tiny checkpoint has the real model's exact structure -- same tensor names, layer types,
Qwen3-VL ViT with DeepStack mergers, mRoPE text decoder, AlternateVLDiT with interleaved
self/cross attention, embodiment-conditioned MLPs, the same ``config.json`` keys and on-disk
layout -- with shrunk widths and depths. It is built *from the upstream model classes*, so
loading it with ``strict=True`` checks the port's weight-name mapping.

Build one by hand (never commit the output)::

    python test/unit/test_gr00t_tiny.py --out /path/to/tiny-gr00t-n17 [--variant h]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys

import pytest
import torch

from vllm_omni_neuron.diffusion.models.gr00t.config import COSMOS_REASON2_2B

TINY_BACKBONE = copy.deepcopy(COSMOS_REASON2_2B)
TINY_BACKBONE["text_config"].update(
    hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    num_hidden_layers=4, rope_scaling={"mrope_interleaved": True, "mrope_section": [6, 5, 5], "rope_type": "default"})
TINY_BACKBONE["vision_config"].update(
    hidden_size=64, intermediate_size=128, num_heads=4, depth=6, deepstack_visual_indexes=[1, 3, 5], out_hidden_size=128)

# real N1.7 config.json keys, shrunk
TINY_HEAD = {
    "architectures": ["Gr00tN1d7"], "model_type": "Gr00tN1d7", "model_name": "nvidia/Cosmos-Reason2-2B",
    "action_horizon": 40, "add_pos_embed": True, "backbone_embedding_dim": 128, "hidden_size": 64,
    "input_embedding_dim": 96, "max_action_dim": 132, "max_state_dim": 132, "max_num_embodiments": 32,
    "max_seq_len": 1024, "num_inference_timesteps": 4, "num_timestep_buckets": 1000, "select_layer": 4,
    "use_alternate_vl_dit": True, "use_vlln": True, "use_vl_self_attention": True, "load_bf16": False,
    "state_history_length": 1, "attend_text_every_n_blocks": 2, "dtype": "bfloat16", "model_dtype": "bfloat16",
    "diffusion_model_cfg": {"attention_head_dim": 24, "dropout": 0.2, "final_dropout": True,
                            "interleave_self_attention": True, "norm_type": "ada_norm", "num_attention_heads": 4,
                            "num_layers": 4, "output_dim": 64, "positional_embeddings": None},
    "vl_self_attention_cfg": {"attention_head_dim": 32, "dropout": 0.2, "final_dropout": True,
                              "num_attention_heads": 4, "num_layers": 2, "positional_embeddings": None},
}

PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


def tiny_head_config(variant: str = "n17") -> dict:
    cfg = copy.deepcopy(TINY_HEAD)
    if variant == "h":  # GR00T-H-N1.7: no VL self-attention, 50-step chunks (see docs)
        cfg.update(action_horizon=50, use_vl_self_attention=False, vl_self_attention_cfg=None,
                   state_dropout_prob=0.0, use_soft_prompts=False, soft_prompt_num_tokens=32)
    return cfg


def build_upstream_model(model_dir: str, head_cfg: dict):
    """Upstream ``Gr00tN1d7`` on CPU fp32, backbone config read from ``<model_dir>/nvidia/Cosmos-Reason2-2B``.

    The upstream data collator (it loads the Qwen3-VL processor from the Hub) is stubbed: only
    the model is needed.
    """
    import vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 as up
    from vllm_omni.diffusion.models.gr00t.configs.gr00t_n1d7 import Gr00tN1d7Config

    cfg = dict(head_cfg)
    cfg["model_name"] = os.path.join(model_dir, "nvidia", "Cosmos-Reason2-2B")
    cfg["load_bf16"] = False
    conf = Gr00tN1d7Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    orig = up.Gr00tN1d7DataCollator
    up.Gr00tN1d7DataCollator = lambda **kw: None
    try:
        model = up.Gr00tN1d7(conf, transformers_loading_kwargs={"local_files_only": True})
    finally:
        up.Gr00tN1d7DataCollator = orig
    return model.eval()


def build_tiny_checkpoint(out: str, variant: str = "n17", seed: int = 0, processor_from: str | None = None) -> str:
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    head = tiny_head_config(variant)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(head, f, indent=2)
    with open(os.path.join(out, "backbone_config.json"), "w") as f:
        json.dump(TINY_BACKBONE, f, indent=2)
    hf_dir = os.path.join(out, "nvidia", "Cosmos-Reason2-2B")
    os.makedirs(hf_dir, exist_ok=True)
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump({**TINY_BACKBONE, "architectures": ["Qwen3VLForConditionalGeneration"]}, f, indent=2)
    torch.manual_seed(seed)
    model = build_upstream_model(out, head)
    with torch.no_grad():  # layer norms init to 1/0; perturb so they are exercised
        for n, p in model.named_parameters():
            if "norm" in n or "vlln" in n:
                p.add_(0.1 * torch.randn_like(p))
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    # upstream drops lm_head; real checkpoints carry it (tied to embed_tokens)
    sd["backbone.model.lm_head.weight"] = sd["backbone.model.model.language_model.embed_tokens.weight"].clone()
    if variant == "h":
        sd["action_head.dropout_prob_by_embodiment"] = torch.zeros(32, dtype=torch.bfloat16)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    if processor_from:
        for name in PROCESSOR_FILES:
            src = os.path.join(processor_from, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out, name))
    return out


# ----------------------------------------------------------------------------------------
# synthetic model inputs (the upstream collator's output layout)
# ----------------------------------------------------------------------------------------


def synthetic_inputs(n_images: int = 4, grid: int | tuple[int, int] = 16, n_text: int = 24, seed: int = 0,
                     embodiment: int = 17):
    """input_ids like ``<im_start>user\\n (<vs> <img>*(gh*gw/4) <ve>)*n  text <im_end>\\n``."""
    g = torch.Generator().manual_seed(seed)
    bb = COSMOS_REASON2_2B
    gh, gw = (grid, grid) if isinstance(grid, int) else grid
    per_img = (gh // 2) * (gw // 2)
    ids = [151644, 872, 198]
    for _ in range(n_images):
        ids += [bb["vision_start_token_id"]] + [bb["image_token_id"]] * per_img + [bb["vision_end_token_id"]]
    ids += torch.randint(1000, 100000, (n_text,), generator=g).tolist() + [151645, 198]
    input_ids = torch.tensor([ids])
    mm = (input_ids == bb["image_token_id"]).long()
    patch_dim = 3 * 2 * 16 * 16
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "mm_token_type_ids": mm,
        "pixel_values": torch.randn(n_images * gh * gw, patch_dim, generator=g),
        "image_grid_thw": torch.tensor([[1, gh, gw]] * n_images),
        "state": torch.randn(1, 1, 132, generator=g) * 0.5,
        "embodiment_id": torch.tensor([embodiment]),
    }


def upstream_get_action(model, inputs: dict, noise: torch.Tensor) -> torch.Tensor:
    """Run upstream ``Gr00tN1d7.get_action`` with a fixed initial noise."""
    import vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 as up
    from transformers.feature_extraction_utils import BatchFeature

    real_randn = torch.randn

    def fixed_randn(*args, **kwargs):
        return noise.to(kwargs.get("dtype") or torch.float32)

    dtype = next(model.parameters()).dtype
    batch = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
    up.torch.randn = fixed_randn
    try:
        with torch.no_grad():
            backbone_out = model.backbone(BatchFeature(data=dict(batch)))
            features = backbone_out["backbone_features"].clone()  # get_action overwrites it with vlln(...)
            out = model.action_head.get_action(backbone_out, BatchFeature(data=dict(batch)))
    finally:
        up.torch.randn = real_randn
    return out["action_pred"], features


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ----------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["n17", "h"])
def tiny(request, tmp_path_factory):
    pytest.importorskip("vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7")
    out = str(tmp_path_factory.mktemp(f"tiny-gr00t-{request.param}"))
    build_tiny_checkpoint(out, variant=request.param)
    return request.param, out


def test_tiny_layout_and_strict_load(tiny):
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    variant, path = tiny
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    assert (m.head.vl_self_attention is None) == (variant == "h")
    assert m.head.action_horizon == (50 if variant == "h" else 40)
    assert len(m.text.layers) == 4 and len(m.vision.blocks) == 6


def test_tiny_cpu_parity_vs_upstream(tiny):
    """fp32 CPU: backbone features and the 4-step action chunk match upstream to fp32 noise."""
    from safetensors.torch import load_file

    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    variant, path = tiny
    ref = build_upstream_model(path, tiny_head_config(variant))
    sd = load_file(os.path.join(path, "model.safetensors"))
    sd = {k: v.float() for k, v in sd.items() if k not in ("backbone.model.lm_head.weight",
                                                             "action_head.dropout_prob_by_embodiment")}
    ref.load_state_dict(sd, strict=True)
    mine = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    inputs = synthetic_inputs()
    noise = mine.draw_noise(1, torch.Generator().manual_seed(1), dtype=torch.float32)
    want_act, want_feat = upstream_get_action(ref, inputs, noise)
    got_feat, _ = mine.backbone_features(inputs)
    got_act = mine.get_action(inputs, noise=noise)["action_pred"]
    assert _rel(got_feat, want_feat) < 1e-4, _rel(got_feat, want_feat)
    assert _rel(got_act, want_act) < 1e-4, _rel(got_act, want_act)


def test_tiny_bucket_padding_is_exact(tiny):
    """Padding the VL sequence to a bucket must not change the actions."""
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    inputs = synthetic_inputs()
    real = inputs["input_ids"].shape[1]
    noise = m.draw_noise(1, torch.Generator().manual_seed(2), dtype=torch.float32)
    a = m.get_action(inputs, noise=noise, bucket=real)["action_pred"]
    b = m.get_action(inputs, noise=noise, bucket=real + 61)["action_pred"]
    assert _rel(b, a) < 1e-5


def test_tiny_image_bucket_padding_is_exact(tiny, monkeypatch):
    """Padding the image count to a bucket (zero images, never gathered) must not change the actions."""
    from vllm_omni_neuron.diffusion.models.gr00t import backbone
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    inputs = synthetic_inputs(n_images=3)
    noise = m.draw_noise(1, torch.Generator().manual_seed(3), dtype=torch.float32)
    monkeypatch.setattr(backbone, "IMAGE_BUCKETS", (3,))
    a = m.get_action(inputs, noise=noise)["action_pred"]
    assert m.stats["n_images"] == 3
    monkeypatch.setattr(backbone, "IMAGE_BUCKETS", (1, 2, 4))
    b = m.get_action(inputs, noise=noise)["action_pred"]
    assert m.stats["n_images"] == 4
    assert _rel(b, a) < 1e-5


def test_tiny_repeated_prompt_reuses_tables(tiny):
    """A repeated prompt hits the host text-table and device caches and gives the same actions;
    a new prompt misses and matches a fresh model."""
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    noise = m.draw_noise(1, torch.Generator().manual_seed(4), dtype=torch.float32)
    x0, x1 = synthetic_inputs(seed=0), synthetic_inputs(seed=1)
    a0 = m.get_action(x0, noise=noise)["action_pred"]
    again = m.get_action({**x0, "pixel_values": x1["pixel_values"]}, noise=noise)["action_pred"]
    assert len(m.prep._text_cache) == 1
    a1 = m.get_action(x1, noise=noise)["action_pred"]
    assert len(m.prep._text_cache) == 2
    fresh = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    assert torch.equal(a0, m.get_action(x0, noise=noise)["action_pred"])
    assert torch.equal(a1, fresh.get_action(x1, noise=noise)["action_pred"])
    assert torch.equal(again, fresh.get_action({**x0, "pixel_values": x1["pixel_values"]}, noise=noise)["action_pred"])


def test_tiny_fused_qkv_matches(tiny, monkeypatch):
    """Fused self-attention Q/K/V projections give the same actions."""
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    inputs = synthetic_inputs()
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    noise = m.draw_noise(1, torch.Generator().manual_seed(6), dtype=torch.float32)
    want = m.get_action(inputs, noise=noise)["action_pred"]
    monkeypatch.setenv("GR00T_HEAD_FUSE_QKV", "1")
    f = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    assert f.head.model.transformer_blocks[1].attn1.qkv_weight is not None
    assert _rel(f.get_action(inputs, noise=noise)["action_pred"], want) < 1e-6


def test_tiny_pretransposed_weights_match(tiny, monkeypatch):
    """Contiguous [in, out] weight copies (GR00T_PRETRANSPOSE=all) give the same actions, alone and
    after TP sharding refreshes them."""
    from vllm_omni_neuron.diffusion.models.gr00t.layers import PretransposedLinear
    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    inputs = synthetic_inputs()
    monkeypatch.setenv("GR00T_PRETRANSPOSE", "none")
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    assert not isinstance(m.head.model.proj_out_2, PretransposedLinear)
    noise = m.draw_noise(1, torch.Generator().manual_seed(7), dtype=torch.float32)
    want = m.get_action(inputs, noise=noise)["action_pred"]
    monkeypatch.setenv("GR00T_PRETRANSPOSE", "all")
    p = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    assert isinstance(p.head.model.proj_out_2, PretransposedLinear)
    assert isinstance(p.text.layers[0].mlp.down_proj, PretransposedLinear)
    assert _rel(p.get_action(inputs, noise=noise)["action_pred"], want) < 1e-6


def _tp_worker(rank, world, port, path, out, backbone=False):
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    if backbone:  # also exercise shard_linear refreshing the pretransposed weight copies
        os.environ["GR00T_PRETRANSPOSE"] = "all"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
        m.head.shard_tp(rank, world, dist.group.WORLD)
        if backbone:
            m.vision.shard_tp(rank, world, dist.group.WORLD)
            m.text.shard_tp(rank, world, dist.group.WORLD)
        noise = m.draw_noise(1, torch.Generator().manual_seed(5), dtype=torch.float32)
        act = m.get_action(synthetic_inputs(), noise=noise)["action_pred"]
        torch.save(act, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,backbone", [(2, False), (2, True), (4, False)])
def test_tiny_tp_matches_tp1(tiny, tmp_path, world, backbone):
    """Action head (and optionally the ViT + text decoder) sharded over gloo ranks == TP=1."""
    import socket

    import torch.multiprocessing as mp

    from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel

    _, path = tiny
    m = NeuronGr00tModel.from_pretrained(path, dtype=torch.float32)
    noise = m.draw_noise(1, torch.Generator().manual_seed(5), dtype=torch.float32)
    want = m.get_action(synthetic_inputs(), noise=noise)["action_pred"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "act")
    mp.spawn(_tp_worker, args=(world, port, path, out, backbone), nprocs=world, join=True)
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        assert _rel(got, want) < 1e-5, (r, _rel(got, want))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["n17", "h"], default="n17")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processor-from", default=None, help="copy processor/statistics JSONs from a real checkpoint")
    a = ap.parse_args()
    print(build_tiny_checkpoint(a.out, a.variant, a.seed, a.processor_from))
    sys.exit(0)
