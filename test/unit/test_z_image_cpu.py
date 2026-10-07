# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Neuron Z-Image components against diffusers / transformers (fp32, tiny model).

The tiny checkpoint is generated on the fly by ``make_tiny_z_image.py`` (structure-identical,
random weights); the real tokenizer + scheduler come from ``$Z_IMAGE_WEIGHTS`` (default
``$WEIGHTS/z-image-turbo``), so the suite skips where that is absent.
"""

from __future__ import annotations

import os

import pytest
import torch

from .test_z_image_tiny_ckpt import make_tiny

SRC = os.environ.get(
    "Z_IMAGE_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", "/nonexistent"), "z-image-turbo")
)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not os.path.isdir(os.path.join(SRC, "tokenizer")):
        pytest.skip("set Z_IMAGE_WEIGHTS to a Z-Image checkout (tokenizer + scheduler)")
    return make_tiny(SRC, str(tmp_path_factory.mktemp("tiny-z-image")))


def test_dit_matches_diffusers(tiny):
    from diffusers import ZImageTransformer2DModel

    from vllm_omni_neuron.diffusion.models.z_image.transformer import (
        NeuronZImageDiT,
        RopeTables,
        ZImageDiTConfig,
        prepare_dit_inputs,
        unpatchify,
    )

    ref = ZImageTransformer2DModel.from_pretrained(
        tiny, subfolder="transformer", torch_dtype=torch.float32
    ).eval()
    cfg = ZImageDiTConfig.from_model_dir(tiny)
    dit = NeuronZImageDiT(cfg, dtype=torch.float32, tp=(1, 0, None))
    dit.load_weights(tiny)
    torch.manual_seed(0)
    # B=2 (a CFG pair) with different caption lengths, neither a multiple of 32; a 40x24 latent gives
    # 240 image tokens (not a multiple of 32 either), exercising every pad path.
    imgs = [torch.randn(16, 1, 40, 24) for _ in range(2)]
    caps = [torch.randn(45, cfg.cap_feat_dim), torch.randn(7, cfg.cap_feat_dim)]
    t = torch.tensor([0.3, 0.3])
    with torch.no_grad():
        want = ref([i.clone() for i in imgs], t, [c.clone() for c in caps], return_dict=False)[0]
        for bucket, split in ((None, 0), (256, 0), (128, 1)):
            prep = prepare_dit_inputs(
                cfg, RopeTables(cfg), imgs, caps, torch.float32, cap_bucket=bucket
            )
            args = list(prep["args"])
            args[9] = t
            if split:
                dit.setup_runner(split)
                out = dit(*args)
                dit._runner = None
            else:
                out = dit(*args)
            got = unpatchify(cfg, out, prep["meta"])
            for g, w in zip(got, want):
                assert g.shape == w.shape
                assert _rel(g, w) < 1e-5, (bucket, split, _rel(g, w))


def test_text_encoder_matches_hf(tiny):
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImageTextEncoder

    ref = Qwen3ForCausalLM.from_pretrained(
        os.path.join(tiny, "text_encoder"), torch_dtype=torch.float32
    ).eval()
    tok = AutoTokenizer.from_pretrained(tiny, subfolder="tokenizer")
    enc = NeuronZImageTextEncoder(tiny, torch.float32)
    enc.load()
    prompts = ["a red fox in the snow, golden hour", ""]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in prompts
    ]
    ti = tok(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    m = ti.attention_mask.bool()
    with torch.no_grad():
        want = ref(
            input_ids=ti.input_ids, attention_mask=m, output_hidden_states=True
        ).hidden_states[-2]
        got = enc(
            input_ids=ti.input_ids, attention_mask=m, output_hidden_states=True
        ).hidden_states[-2]
    assert got.shape == want.shape
    for i in range(2):
        assert _rel(got[i][m[i]], want[i][m[i]]) < 1e-5


def test_vae_processor_matches_default(tiny):
    from diffusers import AutoencoderKL

    from vllm_omni_neuron.diffusion.models.z_image.vae import NeuronAutoencoderKL

    ref = AutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32).eval()
    ours = NeuronAutoencoderKL.from_pretrained(tiny, subfolder="vae", torch_dtype=torch.float32)
    z = torch.randn(1, 16, 16, 24)
    with torch.no_grad():
        want = ref.decode(z, return_dict=False)[0]
        got = ours.decode(z, return_dict=False)[0]
    assert _rel(got, want) < 1e-5


def test_pipeline_end_to_end_matches_diffusers(tiny):
    """Full T2I generation (CFG on, 3 steps) through diffusers' pipeline loop with our three
    components swapped in, against the pure diffusers pipeline, same seed."""
    from diffusers import ZImagePipeline

    from vllm_omni_neuron.diffusion.models.z_image.standalone import build_diffusers_pipeline

    ref = ZImagePipeline.from_pretrained(tiny, torch_dtype=torch.float32)
    ours = build_diffusers_pipeline(tiny, torch.float32, device="cpu", compile_backend=None)
    kw = dict(
        prompt="a lighthouse at dusk",
        height=256,
        width=192,
        num_inference_steps=3,
        guidance_scale=3.0,
        output_type="latent",
    )
    with torch.no_grad():
        want = ref(generator=torch.Generator().manual_seed(1), **kw).images
        got = ours(generator=torch.Generator().manual_seed(1), **kw).images
    assert _rel(got, want) < 1e-4
