# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the InternVLA-A1.5 port (tiny random checkpoint; no Neuron device needed).

* the matrix-product Gated DeltaNet chunk rule equals upstream's loop form;
* the checkpoint loader is strict and maps every upstream tensor name;
* end to end, the port's ``sample_actions`` matches upstream's (``INTERNVLA_REF_SRC`` = the
  InternVLA-A-series ``src`` dir; skipped otherwise) in fp32, and the bf16 port stays in the
  bf16 band of upstream's own bf16 run.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from vllm_omni_neuron.diffusion.models.internvla import InternVLAA15, InternVLAA15Runner
from vllm_omni_neuron.diffusion.models.internvla import preprocess as pp
from vllm_omni_neuron.diffusion.models.internvla.qwen3_5 import chunk_gated_delta_rule

from .test_internvla_a15_tiny_helper import make_tiny_checkpoint
from .test_internvla_a15_upstream_helper import build_upstream, ref_src, upstream_sample


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("internvla_tiny")), seed=0)


def _reference_chunk_rule(query, key, value, g, beta, chunk_size=64):
    """Upstream ``torch_chunk_gated_delta_rule`` (loop form), for comparison."""
    dt = query.dtype
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    b, h, s, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - s % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    query = query / dk**0.5
    v_beta, k_beta = value * beta[..., None], key * beta[..., None]
    query, key, value, k_beta, v_beta = [
        x.reshape(b, h, -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(b, h, -1, chunk_size).cumsum(-1)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), 0)
    decay = ((g[..., None] - g[..., None, :]).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + torch.eye(chunk_size)
    value = attn @ v_beta
    k_cum = attn @ (k_beta * g.exp()[..., None])
    st = torch.zeros(b, h, dk, dv)
    out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), 1)
    for i in range(value.shape[2]):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - k_cum[:, :, i] @ st
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ st + a @ v_new
        st = (
            st * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    out = out.reshape(b, h, -1, dv)[:, :, :s]
    return out.transpose(1, 2).contiguous().to(dt)


@pytest.mark.parametrize("seq", [37, 64, 150])
def test_chunk_rule_matches_upstream_loop(seq):
    g = torch.Generator().manual_seed(seq)
    b, h, dk, dv = 2, 4, 32, 48
    q, k = torch.randn(b, seq, h, dk, generator=g), torch.randn(b, seq, h, dk, generator=g)
    v = torch.randn(b, seq, h, dv, generator=g)
    g_in = -F.softplus(torch.randn(b, seq, h, generator=g)) * 0.5
    beta = torch.rand(b, seq, h, generator=g)
    ref = _reference_chunk_rule(q, k, v, g_in, beta)
    out = chunk_gated_delta_rule(q, k, v, g_in, beta)
    assert _rel(out, ref) < 1e-5


def test_device_tables_match_host_prepare(tiny_ckpt, monkeypatch):
    """The device path's cached tables (host token embedding with image rows zeroed, one-hot
    image placement) rebuild exactly the CPU path's prefix embeddings."""
    m = InternVLAA15.from_pretrained(tiny_ckpt, dtype=torch.float32)
    r = InternVLAA15Runner(m)
    batch = pp.synthetic_request(m.cfg, n_images=3, seed=4)
    h = r.prepare(batch)
    img = torch.randn(h["batch"] * h["m_img"], m.cfg.vlm.text.hidden_size)
    slots = torch.zeros(h["batch"], h["length"], img.shape[-1])
    slots[0].index_copy_(0, h["img_pos"][0], img)
    mask = h["image_mask"]
    x_ref = m.vlm.language_model.embed_tokens(h["input_ids"]) * (1 - mask) + slots * mask
    st = InternVLAA15Runner._device_tables(r, batch, None)
    assert st is InternVLAA15Runner._device_tables(r, batch, None)  # cached per prompt/layout
    x = st["text_emb"] + torch.matmul(st["place"], img.reshape(h["batch"], h["m_img"], -1))
    assert torch.equal(x, x_ref)


def test_device_tables_keep_vision_rope_fp32(tiny_ckpt):
    """On a device the host tables go in the model dtype, except the vision rope (upstream rotates
    the image queries/keys in fp32 from fp32 cos/sin) and the Euler step. Rounding the vision rope
    to bf16 made the device's action error ~1.4x the CPU-bf16 error end to end."""
    m = InternVLAA15.from_pretrained(tiny_ckpt, dtype=torch.bfloat16)
    r = InternVLAA15Runner(m)
    r.device = torch.device("meta")  # any non-CPU device takes the device-table casts
    st = r._device_tables(pp.synthetic_request(m.cfg, n_images=2, seed=3), None)
    assert st["vcos"].dtype == st["vsin"].dtype == st["dt"].dtype == torch.float32
    assert st["cos"].dtype == st["sin"].dtype == st["scos"].dtype == torch.bfloat16
    assert st["temb"][0].dtype == torch.bfloat16


def test_loader_strict(tiny_ckpt):
    m = InternVLAA15.from_pretrained(tiny_ckpt, dtype=torch.bfloat16)
    assert m.action_out_proj.weight.dtype == torch.float32
    assert m.action_in_proj.weight.dtype == torch.bfloat16
    emb = m.vlm.language_model.embed_tokens.weight
    assert emb.shape[0] == 250368 and not emb.is_meta


def test_prefix_padding_is_exact(tiny_ckpt):
    """Bucket padding (extra right-pad tokens) must not change the actions."""
    m = InternVLAA15.from_pretrained(tiny_ckpt, dtype=torch.float32)
    r = InternVLAA15Runner(m)
    batch = pp.synthetic_request(m.cfg, n_images=2, seed=1)
    noise = pp.initial_noise(m.cfg, seed=1)
    n = batch["input_ids"].shape[1]
    a = r.sample_actions(batch, noise, bucket=n)
    b = r.sample_actions(batch, noise, bucket=n + 77)
    assert _rel(b, a) < 1e-5


@pytest.mark.skipif(
    ref_src() is None, reason="set INTERNVLA_REF_SRC to the InternVLA-A-series src dir"
)
@pytest.mark.parametrize("n_images", [1, 3])
def test_matches_upstream(tiny_ckpt, n_images):
    vlm_cfg = os.path.join(tiny_ckpt, "vlm", "config.json")
    batch = pp.synthetic_request(
        InternVLAA15.from_pretrained(tiny_ckpt).cfg, n_images=n_images, seed=2
    )
    noise = pp.initial_noise(InternVLAA15.from_pretrained(tiny_ckpt).cfg, seed=2)
    res = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        up = upstream_sample(
            build_upstream(tiny_ckpt, vlm_cfg, dt, ref_src()), batch, noise, dt
        ).float()
        ours = InternVLAA15Runner(InternVLAA15.from_pretrained(tiny_ckpt, dtype=dt)).sample_actions(
            batch, noise
        )
        res[name] = (up, ours)
    up32, ours32 = res["fp32"]
    assert _rel(ours32, up32) < 1e-4, _rel(ours32, up32)
    up16, ours16 = res["bf16"]
    band = _rel(up16, up32)
    assert _rel(ours16, up32) <= max(2.0 * band, 1e-2), (_rel(ours16, up32), band)


def _upstream_processor(tok_dir):
    """The processor upstream's ``InternVLAA15ChatProcessorTransformFn`` loads
    (``Qwen3VLProcessor.from_pretrained("Qwen/Qwen3.5-2B")``), assembled offline from the
    tokenizer directory and Qwen3.5-2B's published ``preprocessor_config.json`` values."""
    from transformers import Qwen2VLImageProcessorFast
    from transformers.models.qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from vllm_omni_neuron.diffusion.models.internvla.pipeline_internvla import load_tokenizer

    tok = load_tokenizer(tok_dir)
    img = Qwen2VLImageProcessorFast(
        size={"longest_edge": 16777216, "shortest_edge": 65536},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    return Qwen3VLProcessor(
        image_processor=img,
        tokenizer=tok,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=tok.chat_template,
    ), tok


@pytest.mark.skipif(
    not os.environ.get("INTERNVLA_TOKENIZER"),
    reason="set INTERNVLA_TOKENIZER to a Qwen3.5 tokenizer dir",
)
@pytest.mark.parametrize("n_images,hw", [(1, (224, 224)), (3, (224, 224)), (2, (240, 320))])
def test_observation_request_matches_upstream_processor(tiny_ckpt, n_images, hw):
    """The served request equals what upstream's inference transform (``mode="eval"``, state as
    text) feeds the model: same token ids, same pixel patches and grid."""
    import numpy as np

    from vllm_omni_neuron.diffusion.models.internvla.pipeline_internvla import _robot_obs_to_batch

    cfg = InternVLAA15.from_pretrained(tiny_ckpt).cfg
    proc, tok = _upstream_processor(os.environ["INTERNVLA_TOKENIZER"])
    rng = np.random.default_rng(n_images)
    frames = [rng.integers(0, 256, (*hw, 3), dtype=np.uint8) for _ in range(n_images)]
    state = (rng.normal(size=(14,)) * 1.5).astype(np.float32)
    task = "put the red cube in the bowl"
    obs = {
        "images": [{"data": f.tobytes(), "shape": list(f.shape)} for f in frames],
        "state": state,
        "prompt": task,
    }
    ours = _robot_obs_to_batch(obs, cfg, tok)

    # upstream transform_internvla_a1_5.py, InternVLAA15ChatProcessorTransformFn.__call__, eval mode
    s = F.pad(torch.as_tensor(state), (0, 32 - state.shape[0])).numpy() / 3
    state_str = "State: " + " ".join(
        map(str, np.digitize(s, bins=np.linspace(-1, 1, 257)[:-1]) - 1)
    )
    user_text = (
        "Task: "
        + task
        + "; "
        + "Control Mode: <joint>"
        + "; "
        + state_str
        + "; Output: <Subtask, Action>"
    )
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user_text}]
    messages = [
        {"role": "system", "content": pp.SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [torch.as_tensor(f).permute(2, 0, 1).float() / 255.0 for f in frames]
    up = proc(text=[text], images=images, do_rescale=False, return_tensors="pt", padding=False)

    assert torch.equal(ours["input_ids"], up.input_ids), (
        ours["input_ids"].shape,
        up.input_ids.shape,
    )
    assert torch.equal(ours["image_grid_thw"], up.image_grid_thw)
    assert ours["pixel_values"].shape == up.pixel_values.shape
    assert (ours["pixel_values"] - up.pixel_values).abs().max() < 2e-2
    assert _rel(ours["pixel_values"], up.pixel_values) < 1e-3
