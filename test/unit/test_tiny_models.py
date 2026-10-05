# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight checkpoints in a real on-disk layout (vllm_omni_neuron.tiny_models). CPU only."""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni_neuron import tiny_models as tm


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


@pytest.fixture(scope="module")
def fake_pipeline(tmp_path_factory):
    """A small stand-in 'real' checkpoint: sharded diffusers DiT, transformers text encoder, VAE,
    tokenizer/scheduler files, and a config-less weights folder described by the root config."""
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    root = tmp_path_factory.mktemp("real")
    torch.manual_seed(0)
    dit = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=64,
        num_layers=4,
        rope_max_seq_len=64,
    ).to(torch.bfloat16)
    dit.save_pretrained(root / "transformer", max_shard_size="40KB")
    enc = UMT5EncoderModel(
        UMT5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=3, num_heads=4)
    )
    enc.save_pretrained(root / "text_encoder")
    vae = AutoencoderKLWan(
        base_dim=16,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    vae.save_pretrained(root / "vae")
    _write_json(
        root / "scheduler" / "scheduler_config.json", {"_class_name": "UniPCMultistepScheduler"}
    )
    _write_json(root / "tokenizer" / "tokenizer_config.json", {"model_max_length": 16})
    (root / "tokenizer" / "spiece.model").write_bytes(b"\x00" * 128)
    _write_json(
        root / "model_index.json", {"_class_name": "WanPipeline", "transformer": ["diffusers", "X"]}
    )
    # config-less folder + root config / root index (Cosmos3-Edge style)
    vis = {
        f"model.visual.blocks.{i}.attn.weight": torch.randn(8, 8).to(torch.bfloat16)
        for i in range(3)
    }
    vis["model.visual.patch_embed.weight"] = torch.randn(8, 3)
    os.makedirs(root / "vision_encoder")
    save_file(vis, str(root / "vision_encoder" / "model.safetensors"))
    _write_json(
        root / "config.json",
        {
            "architectures": ["Fake"],
            "vision_config": {"depth": 3},
            "text_config": {"num_hidden_layers": 4},
        },
    )
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": {k: "vision_encoder/model.safetensors" for k in vis},
        },
    )
    (root / "README.md").write_text("readme")
    return root


def test_layer_lists_and_normalize():
    names = [
        "blocks.0.a.weight",
        "blocks.3.a.weight",
        "down.1.res.0.w",
        "down.1.res.1.w",
        "head.weight",
    ]
    assert tm.layer_lists(names) == {"blocks": 4, "down": 2, "down.1.res": 2}
    assert tm.normalize_key("blocks.12.attn.to_q.weight") == "blocks.#.attn.to_q.weight"


def test_shrink_layer_counts_recurses():
    cfg = {
        "num_layers": 30,
        "text_config": {"num_hidden_layers": 28, "hidden_size": 64},
        "depth": 1,
    }
    new, changed = tm.shrink_layer_counts(cfg, 2)
    assert (
        new["num_layers"] == 2
        and new["text_config"]["num_hidden_layers"] == 2
        and new["depth"] == 1
    )
    assert changed == {30: 2, 28: 2} and cfg["num_layers"] == 30


def test_synthesize_whole_pipeline(fake_pipeline, tmp_path):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5EncoderModel

    out = tmp_path / "tiny"
    man = tm.make_tiny_checkpoint(str(fake_pipeline), str(out), layers=2)
    comps = man["components"]
    assert set(comps) == {"transformer", "text_encoder", "vae", "vision_encoder"}
    for comp in comps:
        assert tm.compare_layout(str(out), str(fake_pipeline), comp)["ok"], comp

    dit = WanTransformer3DModel.from_pretrained(
        out / "transformer"
    )  # through the rewritten shard index
    assert len(dit.blocks) == 2 and dit.config.num_layers == 2
    assert {dt for _, dt in tm.read_headers(str(out / "transformer")).values()} == {torch.bfloat16}
    enc = UMT5EncoderModel.from_pretrained(out / "text_encoder")
    assert len(enc.encoder.block) == 2
    assert enc(torch.tensor([[1, 2, 3]])).last_hidden_state.shape == (1, 3, 32)
    AutoencoderKLWan.from_pretrained(out / "vae")  # untouched structure (no layer-count fields)

    vis = tm.read_headers(str(out / "vision_encoder"))
    assert sorted(k for k in vis if "blocks" in k) == [
        f"model.visual.blocks.{i}.attn.weight" for i in range(2)
    ]
    root_cfg = json.loads((out / "config.json").read_text())
    assert (
        root_cfg["vision_config"]["depth"] == 2
        and root_cfg["text_config"]["num_hidden_layers"] == 2
    )
    root_idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(root_idx["weight_map"]) == set(vis)
    assert set(root_idx["weight_map"].values()) == {"vision_encoder/model.safetensors"}

    assert (out / "tokenizer" / "spiece.model").exists() and (out / "README.md").exists()
    assert (out / "scheduler" / "scheduler_config.json").exists() and (
        out / "model_index.json"
    ).exists()
    assert not list((out / "transformer").glob("*-of-*"))


def test_instantiate_shrinks_dims(fake_pipeline, tmp_path):
    from diffusers import AutoencoderKLWan

    out = tmp_path / "tiny"
    tm.make_tiny_checkpoint(
        str(fake_pipeline),
        str(out),
        layers=2,
        modes={"vae": "instantiate"},
        overrides={"vae": {"base_dim": 8}},
    )
    vae = AutoencoderKLWan.from_pretrained(out / "vae")
    assert vae.config.base_dim == 8
    assert tm.compare_layout(str(out), str(fake_pipeline), "vae")["ok"]
    full = sum(
        p.numel() for p in AutoencoderKLWan.from_pretrained(fake_pipeline / "vae").parameters()
    )
    assert sum(p.numel() for p in vae.parameters()) < full / 2


def test_instantiate_keeps_source_dtypes(fake_pipeline, tmp_path):
    out = tmp_path / "tiny"
    tm.make_tiny_checkpoint(
        str(fake_pipeline), str(out), layers=1, modes={"transformer": "instantiate"}
    )
    hdr = tm.read_headers(str(out / "transformer"))
    assert {dt for _, dt in hdr.values()} == {torch.bfloat16}
    assert tm.compare_layout(str(out), str(fake_pipeline), "transformer")["ok"]


def test_compare_layout_reports_differences(fake_pipeline, tmp_path):
    out = tmp_path / "tiny"
    tm.make_tiny_checkpoint(str(fake_pipeline), str(out), layers=2)
    t = tm.read_headers(str(out / "vision_encoder"))
    tensors = {
        k: torch.zeros(s, dtype=torch.float32) for k, (s, _) in t.items() if "patch" not in k
    }
    tensors["extra.weight"] = torch.zeros(1)
    save_file(tensors, str(out / "vision_encoder" / "model.safetensors"))
    r = tm.compare_layout(str(out), str(fake_pipeline), "vision_encoder")
    assert (
        not r["ok"]
        and r["missing"] == ["model.visual.patch_embed.weight"]
        and r["extra"] == ["extra.weight"]
    )
    assert r["dtype_mismatch"] == ["model.visual.blocks.#.attn.weight"]


def test_refuses_non_empty_out(fake_pipeline, tmp_path):
    (tmp_path / "x").write_text("")
    with pytest.raises(FileExistsError):
        tm.make_tiny_checkpoint(str(fake_pipeline), str(tmp_path))


def test_cli(fake_pipeline, tmp_path, capsys):
    out = tmp_path / "cli"
    assert (
        tm.main(
            [
                str(fake_pipeline),
                str(out),
                "--layers",
                "1",
                "--mode",
                "vae=instantiate",
                "--set",
                "vae.base_dim=8",
            ]
        )
        == 0
    )
    rep = json.loads(capsys.readouterr().out)
    assert rep["vae"]["mode"] == "instantiate" and all(v["layout_ok"] for v in rep.values())
