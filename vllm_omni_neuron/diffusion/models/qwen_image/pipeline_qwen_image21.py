# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 text-to-image pipeline for Neuron (vLLM-Omni diffusion stage).

Host-side math follows diffusers' ``QwenImage21Pipeline`` (vendored in
``_vendor/pipeline_qwenimage21.py``): the same prompt template, system-token drop, latent
packing, flow-matching schedule (``calculate_shift`` + ``FlowMatchEulerDiscreteScheduler``) and
latent de-normalization. The installed vLLM-Omni has no Qwen-Image 2.1 pipeline, so this class
implements the stage contract (``forward(req) -> DiffusionOutput``) itself.

Four fixed-shape graphs run on the NeuronCore(s), each compiled on first use per shape:

1. ``text_encoder``  Qwen3-VL decoder stack -> pre-norm last hidden states (per text bucket);
2. ``dit_prefix``    the DiT over the prompt prefix at ``t = 0`` -> per-layer K/V (per bucket);
3. ``dit_target``    the DiT over the target tokens against the prefix K/V (per resolution),
                     once per denoising step and CFG branch;
4. ``vae_decode``    one image (per resolution).

The text encoder and the DiT are tensor-parallel over the stage's TP group; the VAE runs on
global rank 0 only. Initial noise is drawn in fp32 and cast, so a bf16 device run and an fp32
CPU reference start from the same noise for a seed.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ._vendor.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from .common import host
from .text_encoder_qwen3vl import NeuronQwen3VLTextEncoder, Qwen3VLTextConfig
from .transformer_qwenimage21 import (
    NeuronQwenImage21Transformer,
    QwenImage21DiTConfig,
    assemble_prefix_inputs,
    build_layout,
)
from .vae_qwenimage21 import NeuronQwenImage21VAE, skip_redundant_decode, tile_parallel_group

logger = logging.getLogger(__name__)

PROFILE = os.environ.get("QWEN_IMAGE_PROFILE", "0") == "1"
TEXT_BUCKETS = tuple(
    int(b) for b in os.environ.get("QWEN_IMAGE_TEXT_BUCKETS", "64,128,256,512,1024").split(",")
)
TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
VAE_SCALE_FACTOR = 16

PIPELINE_REGISTRY = [
    {
        # model_index.json `_class_name` of Qwen/Qwen-Image-2.1
        "model_arch": "QwenImage21Pipeline",
        "class_name": "NeuronQwenImage21Pipeline",
        "post_process_func_name": "get_qwen_image21_post_process_func",
    },
]


def _prof(name: str, t0: float) -> None:
    if PROFILE:
        print(f"[qwen21-prof] {name} {time.time() - t0:.4f}", flush=True)


def pick_bucket(n: int) -> int:
    for b in TEXT_BUCKETS:
        if n <= b:
            return b
    raise ValueError(
        f"sequence is {n} tokens; the largest bucket is {TEXT_BUCKETS[-1]} (QWEN_IMAGE_TEXT_BUCKETS)"
    )


def get_qwen_image21_post_process_func(od_config):
    from diffusers.image_processor import VaeImageProcessor

    processor = VaeImageProcessor(vae_scale_factor=VAE_SCALE_FACTOR)

    def post_process_func(images: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return images
        return processor.postprocess(images, output_type=output_type)

    return post_process_func


class NeuronQwenImage21Pipeline(nn.Module):
    supports_request_batch = False
    # The engine's warm-up request (512x512, 1 step) would compile graphs for a shape nobody asked
    # for; each graph compiles on the first real request of its shape instead.
    dummy_run_num_frames = 0

    def __init__(
        self,
        *,
        od_config=None,
        prefix: str = "",
        model_path: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen3VLProcessor

        self.od_config = od_config
        model = model_path or od_config.model
        self.model_path = model
        if dtype is None:
            dtype = getattr(od_config, "dtype", None) or torch.bfloat16
        self.dtype = dtype
        self.device = torch.device("cpu")  # host-side pipeline math stays on the CPU
        self._device = torch.device("cpu")  # where the compiled components live
        self.weights_sources = []  # our own sharded loaders, not the engine's weight iterator

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler"
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model, subfolder="processor")
        self.text_encoder = NeuronQwen3VLTextEncoder(
            Qwen3VLTextConfig.from_model_dir(model), dtype=dtype
        )
        self.transformer = NeuronQwenImage21Transformer(
            QwenImage21DiTConfig.from_model_dir(model), dtype=dtype
        )
        self.vae = NeuronQwenImage21VAE.from_pretrained(model, subfolder="vae", torch_dtype=dtype)
        self.latent_channels = self.vae.config.z_dim
        if self.transformer.cfg.in_channels != self.latent_channels:
            raise ValueError("transformer in_channels must equal the VAE z_dim")

        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
        sys_tokens = self.processor.apply_chat_template(
            sys_message, tokenize=True, return_dict=False
        )
        self._drop_idx = len(sys_tokens[0])
        self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

        self._te_fn = self.text_encoder
        self._prefix_fn = self.transformer.forward_prefix
        self._target_fn = self.transformer.forward_target
        self.stats: dict = {}
        self._target_s_sum, self._target_calls = 0.0, 0

    # -- lifecycle ------------------------------------------------------------------------
    def load_weights(self, weights=None):
        t0 = time.time()
        self.text_encoder.load_weights(self.model_path, "cpu")
        self.transformer.load_weights(self.model_path, "cpu")
        logger.info("Qwen-Image 2.1 weights loaded in %.1fs", time.time() - t0)
        return None

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            for m in (self.text_encoder, self.transformer):
                nn.Module.to(m, self._device)
            self.transformer.t_freqs = self.transformer.t_freqs.to(self._device)
            self.vae.to(self._device)
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(TRANSFORMER_COMPILER_ARGS)}

        self._te_fn = torch.compile(
            self.text_encoder, backend=backend, options=opts("qwen_image21_text_encoder"), **kw
        )
        self._prefix_fn = torch.compile(
            self.transformer.forward_prefix,
            backend=backend,
            options=opts("qwen_image21_dit_prefix"),
            **kw,
        )
        self._target_fn = torch.compile(
            self.transformer.forward_target,
            backend=backend,
            options=opts("qwen_image21_dit_target"),
            **kw,
        )
        if not skip_redundant_decode():  # every tile-decoding rank (or the output rank alone)
            self.vae.compile(backend, base)
        return self

    # -- prompt encoding ------------------------------------------------------------------
    def _dev(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return host(x, dtype).to(self._device)

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """-> (prompt_embeds ``[1, S, H]`` host, image_pad_mask ``[S]`` bool) for one prompt."""
        t0 = time.time()
        text = PROMPT_TEMPLATE_T2I.format(prompt if prompt else " ")
        ids = self.processor(text=[text], return_tensors="pt").input_ids  # [1, n]
        n = ids.shape[1]
        bucket = pick_bucket(n)
        padded = torch.zeros(1, bucket, dtype=torch.long)
        padded[:, :n] = ids
        valid = torch.zeros(1, bucket, dtype=torch.bool)
        valid[:, :n] = True
        te = self.text_encoder
        pos = torch.arange(bucket)[None].expand(3, 1, bucket)
        cos, sin = te.rope_tables(pos)
        hidden = self._te_fn(
            self._dev(te.embed(padded), self.dtype),
            self._dev(cos),
            self._dev(sin),
            self._dev(te.causal_bias(valid)),
        )
        hidden = host(hidden)[:, self._drop_idx : n]
        img_mask = ids[0, self._drop_idx :] == self._img_token_id
        _prof("text_encoder", t0)
        return hidden, img_mask

    # -- denoising ------------------------------------------------------------------------
    def _prefix_kv(self, embeds, img_mask, n_target_slots, img_shapes):
        cfg = self.transformer.cfg
        full_mask = torch.cat([img_mask, torch.ones(n_target_slots, dtype=torch.bool)])
        valid = torch.ones(1, embeds.shape[1], dtype=torch.bool)
        prefix_len = int(torch.where(img_mask, 4, 1).sum())
        lay = build_layout(cfg, full_mask, valid, img_shapes, bucket=pick_bucket(prefix_len))
        txt, img = assemble_prefix_inputs(lay, embeds, None, cfg.in_channels, self.dtype)
        t0 = time.time()
        kv = self._prefix_fn(
            self._dev(txt),
            self._dev(img),
            self._dev(lay.is_img, self.dtype),
            self._dev(lay.cos_p),
            self._dev(lay.sin_p),
            self._dev(lay.prefix_bias),
        )
        _prof("dit_prefix", t0)
        return (
            lay,
            tuple(kv),
            (self._dev(lay.cos_t), self._dev(lay.sin_t), self._dev(lay.target_bias)),
        )

    def _velocity(self, latents, t, branch):
        lay, kv, (cos_t, sin_t, tb) = branch
        ts = (t.expand(latents.shape[0]).to(latents.dtype) / 1000).float()
        t0 = time.time()
        out = self._target_fn(
            self._dev(latents, self.dtype), self._dev(ts), cos_t, sin_t, tb, *kv
        ).to("cpu")
        self._target_s_sum += time.time() - t0
        self._target_calls += 1
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 40,
        seed: int = 0,
        generator: torch.Generator | None = None,
        negative_prompt: str | None = None,
        true_cfg_scale: float = 1.0,
        sigmas=None,
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
    ) -> torch.Tensor:
        """One image. Returns the VAE output ``[1, C, H, W]`` in [-1, 1] (``output_type='pt'``)
        or the packed final latents (``'latent'``)."""
        t_all = time.time()
        self._target_s_sum, self._target_calls = 0.0, 0
        multiple = VAE_SCALE_FACTOR * 2
        height, width = height // multiple * multiple, width // multiple * multiple
        h, w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
        n_tok = h * w
        do_cfg = true_cfg_scale > 1 and negative_prompt is not None
        img_shapes = [(1, h, w)]

        t_te = time.time()
        emb, mask = self.encode_prompt(prompt)
        te_s = time.time() - t_te
        t_pfx = time.time()
        branches = [self._prefix_kv(emb, mask, n_tok // 4, img_shapes)]
        if do_cfg:
            nemb, nmask = self.encode_prompt(negative_prompt)
            branches.append(self._prefix_kv(nemb, nmask, n_tok // 4, img_shapes))
        prefix_s = time.time() - t_pfx

        c = self.latent_channels
        if latents is None:
            gen = generator if generator is not None else torch.Generator().manual_seed(seed)
            noise = torch.randn((1, 1, c, h, w), generator=gen, dtype=torch.float32)
            latents = noise.to(self.dtype).view(1, c, n_tok).transpose(1, 2)
        latents = latents.to(self.dtype)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        sc = self.scheduler.config
        mu = calculate_shift(
            n_tok,
            sc.get("base_image_seq_len", 256),
            sc.get("max_image_seq_len", 4096),
            sc.get("base_shift", 0.5),
            sc.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, num_inference_steps, "cpu", sigmas=sigmas, mu=mu
        )
        self.scheduler.set_begin_index(0)

        t_loop = time.time()
        for t in timesteps:
            noise_pred = self._velocity(latents, t, branches[0])
            if do_cfg:
                neg = self._velocity(latents, t, branches[1])
                noise_pred = neg + true_cfg_scale * (noise_pred - neg)
            latents = self.scheduler.step(
                noise_pred.to(latents.dtype), t, latents, return_dict=False
            )[0]
        self.stats = {
            "steps": len(timesteps),
            "loop_s": time.time() - t_loop,
            "text_encoder_s": round(te_s, 3),
            "dit_prefix_s": round(prefix_s, 3),
            "dit_target_s_sum": round(self._target_s_sum, 3),
            "dit_target_s_mean": round(self._target_s_sum / max(self._target_calls, 1), 4),
            "dit_target_calls": self._target_calls,
        }
        if output_type == "latent":
            return latents
        image = self.decode_latents(latents, height, width)
        self.stats["total_s"] = time.time() - t_all
        return image

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Packed final latents ``[1, h*w, C]`` -> image ``[1, C_out, H, W]`` in [-1, 1] (zeros on a
        TP rank that does not return the image)."""
        c = self.latent_channels
        h, w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
        z = latents.transpose(1, 2).reshape(1, c, 1, h, w).to(self.vae.dtype)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, c, 1, 1, 1).to(z.dtype)
        std = torch.tensor(self.vae.config.latents_std).view(1, c, 1, 1, 1).to(z.dtype)
        z = z * std + mean
        image = None
        if not skip_redundant_decode():
            image = self.vae.decode(z, return_dict=False, group=tile_parallel_group())[0]
            self.stats["vae_s"] = self.vae.last_decode_s
        if image is None:  # a non-output TP rank: its image is never returned
            return torch.zeros(1, self.vae.config.out_channels, height, width, dtype=self.vae.dtype)
        return image[:, :, 0]

    # -- vLLM-Omni stage contract -----------------------------------------------------------
    def forward(self, req):
        from vllm_omni.diffusion.data import DiffusionOutput

        if len(req.prompts) != 1:
            raise ValueError("NeuronQwenImage21Pipeline handles one prompt per request")
        p = req.prompts[0]
        prompt = p if isinstance(p, str) else (p.get("prompt") or "")
        negative = None if isinstance(p, str) else p.get("negative_prompt")
        if not isinstance(p, str) and p.get("multi_modal_data"):
            raise NotImplementedError(
                "Qwen-Image 2.1 on Neuron: image-conditioned generation is not supported yet"
            )
        sp = req.sampling_params
        gen = sp.generator[0] if isinstance(sp.generator, list) else sp.generator
        output_type = sp.output_type or "pil"
        image = self.generate(
            prompt,
            height=sp.height or 1024,
            width=sp.width or 1024,
            num_inference_steps=sp.num_inference_steps or 40,
            seed=sp.seed if sp.seed is not None else 0,
            generator=gen
            if isinstance(gen, torch.Generator) and gen.device.type == "cpu"
            else None,
            negative_prompt=negative,
            true_cfg_scale=sp.true_cfg_scale if sp.true_cfg_scale is not None else 1.0,
            sigmas=sp.sigmas,
            output_type="latent" if output_type == "latent" else "pt",
        )
        return DiffusionOutput(output=image)


__all__ = ["PIPELINE_REGISTRY", "NeuronQwenImage21Pipeline", "get_qwen_image21_post_process_func"]
