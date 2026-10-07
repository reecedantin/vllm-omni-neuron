# SPDX-License-Identifier: Apache-2.0
"""Neuron FLUX.2-dev pipeline: upstream vLLM-Omni's ``Flux2Pipeline`` with the text encoder and
the DiT swapped for NeuronCore components.

Everything above the two model calls stays upstream's: request parsing, the Mistral chat
template + tokenisation (always padded to ``max_sequence_length``, so the encoder graph has one
shape), the flow-matching scheduler and its empirical ``mu`` shift, latent packing / RoPE ids,
batch-norm de-normalisation and post-processing. That host-side math runs on the CPU
(``self._execution_device`` is CPU); the components that touch the NeuronCores are

* ``self.text_encoder`` -> :class:`.text_encoder.NeuronFlux2TextEncoder` (Mistral layers 0-29,
  TP-sharded, hidden states 10/20/30 out). Prompt embeddings are cached per prompt, so a repeated
  prompt (and the empty negative prompt) never re-runs the encoder.
* ``self.transformer`` -> :class:`.transformer_flux2.NeuronFlux2Transformer` (TP-sharded DiT).
* ``self.vae`` -> :class:`.vae_flux2.NeuronFlux2Vae`: decode as fixed-shape tiles with whole-image
  GroupNorm statistics, the tiles dealt across all ranks (TP x CP) and merged on rank 0.

The text encoder and DiT stay resident on the same cores (bf16 at TP=8: DiT 8.8 GiB/rank,
encoder 3.9 GiB/rank).
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import OrderedDict

import torch
from vllm.logger import init_logger
from vllm_omni.diffusion.models.flux2 import pipeline_flux2 as _up
from vllm_omni.diffusion.models.flux2.pipeline_flux2 import (
    Flux2Pipeline,
    get_flux2_post_process_func,
)

from .ops import log_info
from .text_encoder import NeuronFlux2TextEncoder
from .transformer_flux2 import NeuronFlux2Transformer
from .vae_flux2 import NeuronFlux2Vae

logger = init_logger(__name__)

_randn_tensor_up = _up.randn_tensor


def _randn_fp32(shape, generator=None, device=None, dtype=None, layout=None):
    """Draw initial noise in fp32, then cast: torch's RNG stream depends on the dtype it samples
    in (a bf16 draw is uncorrelated with an fp32 draw from the same seed, verified: CPU bf16 vs
    fp32 cos ~0 on the unpatched code), so without this a bf16 device run and an fp32 CPU
    reference start from different noise and parity numbers are meaningless. Same fix as the
    Cosmos3-Edge port (``pipeline_cosmos3_edge.py``)."""
    out = _randn_tensor_up(
        shape, generator=generator, device=device, dtype=torch.float32, layout=layout
    )
    return out.to(dtype) if dtype is not None else out


_up.randn_tensor = _randn_fp32

PIPELINE_REGISTRY = [
    {
        "model_arch": "Flux2Pipeline",
        "class_name": "NeuronFlux2Pipeline",
        "post_process_func_name": "get_flux2_post_process_func",
    },
]

EMBED_CACHE_SIZE = int(os.environ.get("FLUX2_EMBED_CACHE", "16"))


@contextlib.contextmanager
def _neuron_components(od_config):
    """Swap upstream's text encoder / DiT constructors (and its device) while its __init__ runs."""
    dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
    mc = dict(od_config.model_config or {})

    def make_text_encoder(config, prefix="", **kwargs):
        return NeuronFlux2TextEncoder(od_config.model, dtype=dtype, group=mc.get("te_group"))

    def make_transformer(quant_config=None, od_config=od_config, **kwargs):
        return NeuronFlux2Transformer(
            od_config.model,
            dtype=dtype,
            double_group=mc.get("double_group"),
            single_group=mc.get("single_group"),
        )

    saved = (_up.MistralEncoderModel, _up.Flux2Transformer2DModel, _up.get_local_device)
    _up.MistralEncoderModel = make_text_encoder
    _up.Flux2Transformer2DModel = make_transformer
    _up.get_local_device = lambda: torch.device("cpu")  # host-side pipeline math stays on the CPU
    try:
        yield
    finally:
        _up.MistralEncoderModel, _up.Flux2Transformer2DModel, _up.get_local_device = saved


class NeuronFlux2Pipeline(Flux2Pipeline):
    def __init__(self, *, od_config, prefix: str = ""):
        with _neuron_components(od_config):
            super().__init__(od_config=od_config, prefix=prefix)
        self.vae = NeuronFlux2Vae(self.vae.eval())
        # weights come from the components' own TP-sharded loaders, not the engine's iterator
        self.weights_sources = []
        self._embed_cache: OrderedDict = OrderedDict()
        self._device_target = None

    # -- weights / placement / compile ------------------------------------------------------
    def _target_device(self) -> torch.device:
        if self._device_target is not None:
            return self._device_target
        if os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1":
            return torch.device("cpu")
        try:
            from vllm_omni.diffusion.distributed.utils import get_local_device

            return get_local_device()
        except Exception:  # noqa: BLE001 - no device context (unit tests)
            return torch.device("cpu")

    def load_weights(self, weights=None):
        dev = self._target_device()
        t0 = time.time()
        self.text_encoder.load_weights(self.od_config.model, dev)
        self.transformer.load_weights(self.od_config.model, dev)
        log_info("flux2: weights loaded to %s in %.1fs", dev, time.time() - t0)
        return None

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device_target = torch.device(device)
            self.text_encoder.to(device)
            self.transformer.to(device)
            self.vae.to(device)
        return self

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Test plumbing: ``FLUX2_INIT_LATENTS=/path.pt`` injects a fixed fp32 noise tensor as the
        initial latents (cast to ``dtype``) instead of sampling, so a device run and the CPU fp32 /
        bf16 references share identical initial noise -- the only way to get a meaningful whole-
        pipeline parity number on a chaotic 4-step guidance-distilled flow. The stored tensor is the
        pre-pack ``(B, in_channels, H//2, W//2)`` noise (see examples/flux2/make_init_latents.py).
        Unset -> normal sampling. Documented in the model card."""
        path = os.environ.get("FLUX2_INIT_LATENTS")
        if path and latents is None:
            blob = torch.load(path, map_location="cpu")
            latents = (blob["latents"] if isinstance(blob, dict) else blob).to(torch.float32)
            log_info("flux2: injected init latents %s from %s", tuple(latents.shape), path)
        return super().prepare_latents(
            batch_size,
            num_latents_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents=latents,
        )

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1":
            return self  # CPU mode is the eager reference; the Neuron backends need device tensors
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self.text_encoder.compile(backend, options, **kwargs)
        self.transformer.compile(backend, options, **kwargs)
        self.vae.compile(backend, options, **kwargs)
        return self

    # -- engine warmup ------------------------------------------------------------------------
    def forward(self, req, *args, **kwargs):
        """Skip the engine's dummy warmup request: it would compile (and keep resident) a full
        DiT + encoder graph set for a geometry no real request asks for. Real requests compile
        their own geometry on first use."""
        if os.environ.get("FLUX2_RUN_WARMUP", "0") != "1" and req.is_dummy_run():
            from vllm_omni.diffusion.data import DiffusionOutput

            logger.info("flux2: skipping the engine warmup request")
            return DiffusionOutput(output=None)
        # Upstream Flux2Pipeline.forward takes output_type as a kwarg defaulting to "pil" and does
        # not read it from the request, so a per-request output_type=latent (parity / latent-only
        # callers) is otherwise ignored and the VAE always runs. Thread it through.
        sp = getattr(req, "sampling_params", None)
        if sp is not None and getattr(sp, "output_type", None) and "output_type" not in kwargs:
            kwargs["output_type"] = sp.output_type
        return super().forward(req, *args, **kwargs)

    # -- prompt-embedding cache -----------------------------------------------------------------
    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        max_sequence_length=512,
        text_encoder_out_layers=(10, 20, 30),
    ):
        if prompt_embeds is None and EMBED_CACHE_SIZE > 0:
            key = (
                tuple([prompt] if isinstance(prompt, str) else list(prompt or [""])),
                int(max_sequence_length),
                tuple(text_encoder_out_layers),
            )
            if key in self._embed_cache:
                self._embed_cache.move_to_end(key)
                prompt_embeds = self._embed_cache[key]
            else:
                prompt_embeds = self._get_mistral_3_small_prompt_embeds(
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    prompt=list(key[0]),
                    device=torch.device("cpu"),
                    max_sequence_length=max_sequence_length,
                    system_message=self.system_message,
                    hidden_states_layers=text_encoder_out_layers,
                )
                self._embed_cache[key] = prompt_embeds
                while len(self._embed_cache) > EMBED_CACHE_SIZE:
                    self._embed_cache.popitem(last=False)
        return super().encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )


__all__ = ["PIPELINE_REGISTRY", "NeuronFlux2Pipeline", "get_flux2_post_process_func"]
