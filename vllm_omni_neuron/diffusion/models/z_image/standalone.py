# SPDX-License-Identifier: Apache-2.0
"""Run the Neuron Z-Image components inside diffusers' own ``ZImagePipeline`` loop.

Used by the device checks and the CPU oracle (no vLLM engine needed): the diffusers pipeline
calls ``text_encoder(...).hidden_states[-2]``, ``transformer(latents, t, caps)[0]`` and
``vae.decode`` with the same signatures as upstream vLLM-Omni's pipeline, so the same three
facades serve both.
"""

from __future__ import annotations

import torch

from .pipeline_z_image import NeuronZImageTextEncoder, NeuronZImageTransformer
from .vae import NeuronAutoencoderKL


def build_components(model_path: str, dtype: torch.dtype):
    te = NeuronZImageTextEncoder(model_path, dtype)
    dit = NeuronZImageTransformer(model_path, dtype)
    vae = NeuronAutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    te.load()
    dit.load()
    return te, dit, vae


def build_diffusers_pipeline(
    model_path: str,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    compile_backend: str | None = None,
    components=None,
    te_on_host: bool = False,
):
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import AutoTokenizer

    te, dit, vae = components or build_components(model_path, dtype)
    dev = torch.device(device)
    if dev.type != "cpu":
        if not te_on_host:
            te.to(dev)
        dit.to(dev)
        vae.to(dev)
    if compile_backend:
        if not te_on_host:
            te.compile(compile_backend)
        dit.compile(compile_backend)
        vae.compile(compile_backend)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    tok = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    pipe = ZImagePipeline(scheduler=sched, vae=vae, text_encoder=te, tokenizer=tok, transformer=dit)
    return pipe
