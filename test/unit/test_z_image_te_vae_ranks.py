# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the device text-encoder precision, its prompt cache, and the VAE tile dealing.

1. ``fp32_residual`` (fp32 residual stream and RMSNorms around bf16 matmuls) is the same math in
   fp32 and closer to the fp32 reference than plain bf16 (tiny checkpoint).
2. The encoder caches per token sequence: a repeated prompt is a hit with identical output.
3. Dealing the VAE decode tiles over 3 gloo ranks gives exactly the single-rank decode.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

SRC = os.environ.get(
    "Z_IMAGE_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", "/nonexistent"), "z-image-turbo")
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not os.path.isdir(os.path.join(SRC, "tokenizer")):
        pytest.skip("set Z_IMAGE_WEIGHTS to a Z-Image checkout (tokenizer + scheduler)")
    from .test_z_image_tiny_ckpt import make_tiny

    return make_tiny(SRC, str(tmp_path_factory.mktemp("tiny-z-image")))


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _tokens(tiny, prompts):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny, subfolder="tokenizer")
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in prompts
    ]
    ti = tok(texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    return ti.input_ids, ti.attention_mask.bool()


def _encode(tiny, dtype, ids, mask, **kw):
    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImageTextEncoder

    enc = NeuronZImageTextEncoder(tiny, dtype, **kw)
    enc.load()
    with torch.no_grad():
        h = enc(input_ids=ids, attention_mask=mask).hidden_states[0]
    return enc, [h[i][mask[i]].float() for i in range(h.shape[0])]


def test_text_encoder_fp32_residual(tiny):
    ids, mask = _tokens(tiny, ["a red fox in the snow, golden hour, detailed fur, photorealistic"])
    _, ref = _encode(tiny, torch.float32, ids, mask)
    _, same = _encode(tiny, torch.float32, ids, mask, fp32_residual=True)
    _, plain = _encode(tiny, torch.bfloat16, ids, mask, fp32_residual=False)
    _, mixed = _encode(tiny, torch.bfloat16, ids, mask, fp32_residual=True)
    assert _rel(same[0], ref[0]) < 1e-6
    e_plain, e_mixed = _rel(plain[0], ref[0]), _rel(mixed[0], ref[0])
    assert e_mixed < e_plain, (e_mixed, e_plain)


def test_text_encoder_cache(tiny):
    ids, mask = _tokens(tiny, ["a lighthouse at dusk"])
    enc, first = _encode(tiny, torch.float32, ids, mask)
    assert not enc.last_hit
    with torch.no_grad():
        again = enc(input_ids=ids, attention_mask=mask).hidden_states[0]
    assert enc.last_hit
    assert torch.equal(again[0][mask[0]].float(), first[0])
    ids2, mask2 = _tokens(tiny, ["a different prompt"])
    with torch.no_grad():
        enc(input_ids=ids2, attention_mask=mask2)
    assert not enc.last_hit


def _vae_rank_main(rank: int, port: int, tiny: str, out: str) -> None:
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.z_image import vae as vmod

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=3, rank=rank)
    vmod._stage_world = lambda: (rank, 3, dist.group.WORLD)
    v = vmod.NeuronAutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32)
    v.tile_lat = 8  # 20 x 20 latent -> a 3 x 3 grid with short, padded edge tiles
    z = torch.randn(1, 16, 20, 20, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        dealt = v.decode(z, return_dict=False)[0]
        v.deal = False
        single = v.decode(z, return_dict=False)[0]
    if rank == 0:
        torch.save({"dealt": dealt, "single": single}, out)
    dist.destroy_process_group()


def test_vae_dealt_decode_matches_single_rank(tiny, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "vae.pt")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    )
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(r), str(port), tiny, out], env=env
        )
        for r in range(3)
    ]
    assert [p.wait(timeout=600) for p in procs] == [0] * 3
    r = torch.load(out)
    assert r["dealt"].shape == (1, 3, 160, 160)
    assert torch.equal(r["dealt"], r["single"])


if __name__ == "__main__":  # a rank of test_vae_dealt_decode_matches_single_rank
    _vae_rank_main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
