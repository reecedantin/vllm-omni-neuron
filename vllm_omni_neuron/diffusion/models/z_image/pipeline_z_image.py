# SPDX-License-Identifier: Apache-2.0
"""Neuron Z-Image / Z-Image-Turbo pipeline.

Upstream vLLM-Omni's ``ZImagePipeline.forward`` runs unchanged on the host: request parsing,
chat-template tokenisation, CFG (with truncation / renormalisation), the flow-matching
scheduler, img2img latent prep and latent (de)normalisation. Three components are swapped for
NeuronCore-compiled ones with host-tensor interfaces:

* ``text_encoder`` -> :class:`NeuronZImageTextEncoder` (Qwen3, first N-1 layers, bucketed)
* ``transformer``  -> :class:`NeuronZImageTransformer` (static-shape DiT, caption buckets)
* ``vae``          -> :class:`NeuronAutoencoderKL` (diffusers AutoencoderKL decoder graph)
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    get_post_process_func as get_z_image_post_process_func,
)

from .cfg_forward import ZImageCFGForwardMixin
from .text_encoder import NeuronQwen3Encoder, Qwen3EncConfig, pick_text_bucket
from .transformer import (
    NeuronZImageDiT,
    RopeTables,
    ZImageDiTConfig,
    prepare_dit_inputs,
    unpatchify,
)
from .vae import NeuronAutoencoderKL

logger = logging.getLogger(__name__)

PROFILE = os.environ.get("Z_IMAGE_PROFILE", "0") == "1"
TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

PIPELINE_REGISTRY = [
    {
        "model_arch": "ZImagePipeline",
        "class_name": "NeuronZImagePipeline",
        "post_process_func_name": "get_z_image_post_process_func",
    },
]


def _prof(name: str, t0: float) -> None:
    if PROFILE:
        print(f"[zimage-prof] {name} {time.time() - t0:.4f} pid={os.getpid()}", flush=True)


def _offset_device(dev: torch.device, env: str) -> torch.device:
    """Optionally place a component on a sibling core of the same process (``Z_IMAGE_TE_CORE=1``)."""
    off = int(os.environ.get(env, "0") or 0)
    if dev.type == "cpu" or off == 0:
        return dev
    return torch.device(dev.type, (dev.index or 0) + off)


class ZImageDiffusionEngine(DiffusionEngine):
    """``DiffusionEngine`` with a warmup resolution that follows the stage config.

    Upstream's ``_dummy_run`` hardcodes 512x512 (``diffusion_engine.py``, no override hook for
    height/width — only ``num_frames`` has one, via ``model_cls.dummy_run_num_frames``). On a
    NeuronCore, warming up at a different resolution than the one served compiles and keeps
    resident a second set of DiT/VAE graphs. Set ``model_config.dummy_run_height`` /
    ``dummy_run_width`` in the stage YAML to the served resolution so the warmup compiles the graphs
    production uses. ``model_config.skip_dummy_run: true`` skips the warmup entirely.
    """

    def _dummy_run(self):
        mcfg = dict(self.od_config.model_config or {})
        if mcfg.get("skip_dummy_run"):
            logger.info("Skipping dummy warmup run (model_config.skip_dummy_run)")
            return
        h, w = mcfg.get("dummy_run_height"), mcfg.get("dummy_run_width")
        if h is None and w is None:
            return super()._dummy_run()
        import PIL.Image
        from vllm_omni.diffusion.io_support import (
            get_dummy_run_num_frames,
            image_color_format,
            supports_multimodal_input,
        )
        from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        height, width = int(h or 512), int(w or 512)
        prompt = {"prompt": "dummy run"}
        supports_image_input, supports_audio_input = supports_multimodal_input(self.od_config)
        if supports_image_input:
            color_format = image_color_format(self.od_config.model_class_name)
            prompt.setdefault("multi_modal_data", {})["image"] = PIL.Image.new(
                color_format, (width, height)
            )
        if supports_audio_input:
            import numpy as np

            prompt.setdefault("multi_modal_data", {})["audio"] = np.random.randn(32000).astype(
                "float32"
            )
        num_frames = get_dummy_run_num_frames(self.od_config.model_class_name, supports_audio_input)
        if num_frames <= 0:
            logger.info("Skipping dummy warmup run (num_frames=0)")
            return
        req = OmniDiffusionRequest(
            prompt=prompt,
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=1,
                num_frames=num_frames,
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info(
            "dummy run to warm up the model at %dx%d (stage-config resolution)", height, width
        )
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")


class NeuronZImageTextEncoder(nn.Module):
    """Drop-in for the HF ``Qwen3ForCausalLM`` the upstream pipeline calls.

    On the NeuronCore (tensor-parallel over the stage's TP group) it runs with an fp32 residual
    stream and fp32 RMSNorms around bf16 matmuls (``fp32_residual``, the default there): in plain
    bf16 its hidden states drift 1.5x further from fp32 than bf16 on CPU, which CFG 4 amplifies over
    the denoising loop (Z-Image base: 13.4 % final-latent rel-L2 vs a 10.4 % bar). ``on_host=True``
    keeps it on the CPU in bf16, eager: ``to()`` and ``compile()`` become no-ops.

    Outputs are cached per token sequence (``Z_IMAGE_TEXT_CACHE`` entries, default 64), so a repeated
    prompt and the empty negative prompt are encoded once.
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        subfolder: str = "text_encoder",
        on_host: bool = False,
        fp32_residual: bool | None = None,
    ):
        super().__init__()
        self.on_host = on_host
        self._fp32_residual = fp32_residual  # None: on for a NeuronCore placement, off on the host
        self.cfg = Qwen3EncConfig.from_model_dir(model_path, subfolder)
        self.model_path, self.subfolder = model_path, subfolder
        self.enc = NeuronQwen3Encoder(self.cfg, dtype=dtype, fp32_residual=bool(fp32_residual))
        self.dtype = dtype
        self._device = torch.device("cpu")
        self._fn = self.enc
        self._cache: OrderedDict = OrderedDict()
        self.cache_size = int(os.environ.get("Z_IMAGE_TEXT_CACHE", "64"))
        self.last_s = 0.0
        self.last_hit = False

    @property
    def device(self) -> torch.device:  # host-tensor interface: callers see a CPU module
        return torch.device("cpu")

    def load(self) -> None:
        self.enc.load_weights(self.model_path, self.subfolder)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and not self.on_host:
            self._device = _offset_device(torch.device(device), "Z_IMAGE_TE_CORE")
            for L in self.enc.layers:
                L.to(self._device)
            if self._fp32_residual is None:
                self.enc.fp32_residual = self._device.type != "cpu"
        return self

    def compile(self, backend, options=None, **kwargs) -> None:
        if self.on_host:
            return
        opts = {
            **(options or {}),
            "model_name": "z_image_text_encoder",
            "compiler_args": list(TRANSFORMER_COMPILER_ARGS),
        }
        self._fn = torch.compile(
            self.enc, backend=backend, options=opts, fullgraph=True, dynamic=False
        )

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
        t0 = time.time()
        ids = input_ids.cpu()
        full = ids.shape[1]
        real = int(attention_mask.sum(1).max().item()) if attention_mask is not None else full
        s = min(pick_text_bucket(real), full)
        # Rows past `real` only see padding through the causal mask, so the valid rows depend on the
        # first `real` ids alone; the key covers the whole bucket anyway (exact replay).
        key = (s, tuple(ids[:, :s].flatten().tolist()), ids.shape[0])
        h = self._cache.get(key)
        self.last_hit = h is not None
        if h is None:
            x = self.enc.embed(ids[:, :s])
            cos, sin = self.enc.rope(s)
            bias = self.enc.causal_bias(s)
            dev = self._device
            with torch.no_grad():
                h = (
                    self._fn(x.to(dev), cos.to(dev), sin.to(dev), bias.to(dev))
                    .to("cpu")
                    .to(self.dtype)
                )
            if self.cache_size > 0:
                self._cache[key] = h
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        if s < full:
            h = torch.cat([h, h.new_zeros(h.shape[0], full - s, h.shape[2])], dim=1)
        self.last_s = time.time() - t0
        _prof("text_encoder_hit" if self.last_hit else "text_encoder", t0)
        return SimpleNamespace(hidden_states=(h, None))


class NeuronZImageTransformer(nn.Module):
    """Drop-in for upstream ``ZImageTransformer2DModel``: same call signature, host tensors in/out."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        subfolder: str = "transformer",
        block_split: int | None = None,
    ):
        super().__init__()
        self.cfg = ZImageDiTConfig.from_model_dir(model_path, subfolder)
        self.model_path, self.subfolder = model_path, subfolder
        self.dit = NeuronZImageDiT(self.cfg, dtype=dtype)
        self.rope = RopeTables(self.cfg)
        self.dtype = dtype
        self.in_channels = self.cfg.in_channels
        self.out_channels = self.cfg.in_channels
        default_split = int(os.environ.get("Z_IMAGE_BLOCK_SPLIT", "5") or 0)
        self.split = default_split if block_split is None else int(block_split)
        if self.split and self.split < self.cfg.n_layers and self.cfg.n_layers % self.split:
            raise ValueError(f"block_split={self.split} must divide n_layers={self.cfg.n_layers}")
        self._device = torch.device("cpu")
        self._full_fn = self.dit
        self._pro_fn, self._epi_fn = self.dit.prologue, self.dit.epilogue
        self._compiled = False
        self.stats = {"calls": 0, "s": 0.0}

    @property
    def device(self) -> torch.device:  # host-tensor interface: callers see a CPU module
        return torch.device("cpu")

    def load(self) -> None:
        self.dit.load_weights(self.model_path, self.subfolder)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.dit.to(self._device)
            if getattr(self.dit, "_runner", None) is not None:
                self.dit._runner.refresh_weights()
        return self

    def compile(self, backend, options=None, **kwargs) -> None:
        base = dict(options or {})
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}

        def opts(name):
            return {**base, "model_name": name, "compiler_args": list(TRANSFORMER_COMPILER_ARGS)}

        if not self.split:  # whole DiT as one graph (small resolutions / tiny tests only)
            self._full_fn = torch.compile(
                self.dit, backend=backend, options=opts("z_image_dit"), **kw
            )
        else:  # prologue + shared N-block runner + epilogue
            self._pro_fn = torch.compile(
                self.dit.prologue, backend=backend, options=opts("z_image_dit_pro"), **kw
            )
            self._epi_fn = torch.compile(
                self.dit.epilogue, backend=backend, options=opts("z_image_dit_epi"), **kw
            )
            self.dit.setup_runner(
                self.split,
                compile_fn=lambda f: torch.compile(
                    f, backend=backend, options=opts("z_image_dit_blocks"), **kw
                ),
            )
        self._compiled = True

    def run_graph(self, args: tuple) -> torch.Tensor:
        (
            x_tok,
            x_pad,
            x_cos,
            x_sin,
            cap,
            cap_pad,
            c_cos,
            c_sin,
            cap_bias,
            t,
            u_cos,
            u_sin,
            u_bias,
        ) = args
        with torch.no_grad():
            if not self.split:
                return self._full_fn(*args)
            u, adaln = self._pro_fn(
                x_tok, x_pad, x_cos, x_sin, cap, cap_pad, c_cos, c_sin, cap_bias, t
            )
            u = self.dit.main_blocks(u, u_cos, u_sin, u_bias, adaln)
            return self._epi_fn(u, adaln)

    def forward(self, x, t, cap_feats, return_dict: bool = False, **kwargs):
        t0 = time.time()
        prep = prepare_dit_inputs(
            self.cfg,
            self.rope,
            [xi.detach().cpu() for xi in x],
            [c.detach().cpu() for c in cap_feats],
            self.dtype,
        )
        args = list(prep["args"])
        ts = t.detach().cpu().float().reshape(-1)
        if ts.numel() == 1 and len(x) > 1:
            ts = ts.expand(len(x))
        args[9] = ts.contiguous()
        dev = self._device
        args = tuple(a.contiguous().to(dev) for a in args)
        out = self.run_graph(args).to("cpu")
        res = unpatchify(self.cfg, out, prep["meta"])
        self.stats["calls"] += 1
        self.stats["s"] += time.time() - t0
        _prof("dit_call", t0)
        return (res,)


class NeuronZImagePipeline(ZImageCFGForwardMixin, ZImagePipeline):
    """Upstream ``ZImagePipeline`` with Neuron text encoder, DiT and VAE."""

    def __init__(self, *, od_config, prefix: str = ""):
        nn.Module.__init__(self)
        self.od_config = od_config
        self.weights_sources = []  # our own loaders read the checkpoint
        self._execution_device = torch.device("cpu")  # host-side pipeline math
        model = od_config.model
        dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        mcfg = dict(od_config.model_config or {})
        local = os.path.exists(model)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local
        )
        self.text_encoder = NeuronZImageTextEncoder(
            model,
            dtype,
            on_host=bool(mcfg.get("text_encoder_on_host", False)),
            fp32_residual=mcfg.get("text_encoder_fp32_residual"),
        )
        self.transformer = NeuronZImageTransformer(
            model, dtype, block_split=mcfg.get("block_split")
        )
        self.vae = NeuronAutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=dtype)
        if "vae_tile_lat" in mcfg:
            self.vae.tile_lat = int(mcfg["vae_tile_lat"])
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(
                od_config, "enable_diffusion_pipeline_profiler", False
            )
        )

    def load_weights(self, weights=None):
        t0 = time.time()
        self.text_encoder.load()
        self.transformer.load()
        logger.info("Z-Image: text encoder + DiT weights loaded in %.1fs", time.time() - t0)
        return None

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        self.transformer.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self.text_encoder.compile(backend, options)
        self.transformer.compile(backend, options, **kwargs)
        self.vae.compile(backend, options)
        return self


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronZImagePipeline",
    "NeuronZImageTextEncoder",
    "NeuronZImageTransformer",
    "ZImageDiffusionEngine",
    "get_z_image_post_process_func",
]
