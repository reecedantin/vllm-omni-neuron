# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo-1.5 image-to-video on Neuron (480p / 720p I2V, incl. the CFG-distilled checkpoints).

Same Neuron DiT / VAE / text path as :class:`NeuronHunyuanVideo15Pipeline`; on top of it, upstream
vLLM-Omni's ``HunyuanVideo15I2VPipeline`` semantics:

* **SigLIP image tokens.** ``feature_extractor`` + ``SiglipVisionModel`` (``last_hidden_state``, 729 x 1152)
  on the host of global rank 0, broadcast to every rank; the DiT receives them as ``image_embeds`` (non-zero
  -> the image stream is on, the 729 tokens are valid encoder keys placed first).
* **First-frame condition.** The input image is VAE-encoded (host encoder, ``argmax`` of the posterior, times
  the scaling factor) at the output resolution; the condition latents hold it in latent frame 0 and zeros
  elsewhere, the mask is 1 on frame 0. The DiT input is ``cat([latents, cond, mask])`` (65 channels).
* **Guidance.** The default guidance scale comes from the checkpoint's ``guider`` config: 6.0 for 480p/720p
  I2V, 1.0 for the CFG-distilled checkpoints, which then run ONE DiT forward per step (no negative branch,
  CFG-parallel unused). A request's ``guidance_scale`` still overrides it.
* **Flow shift** comes from the checkpoint's scheduler config (7.0 for the 720p I2V family).

The denoising loop is upstream's ``HunyuanVideo15I2VPipeline.forward`` itself (bound onto this class), so the
request handling, the sigma schedule and the conditioning layout cannot drift from the reference.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
from vllm_omni.diffusion.models.hunyuan_video import pipeline_hunyuan_video_1_5_i2v as _up_i2v

from .pipeline_hunyuanvideo15 import (
    NeuronHunyuanVideo15Pipeline,
    _global_rank,
    _host_threads,
    _prof,
)
from .transformer import local_model_dir

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # model_index.json `_class_name` of hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*_i2v*
        "model_arch": "HunyuanVideo15ImageToVideoPipeline",
        "class_name": "NeuronHunyuanVideo15I2VPipeline",
        "pre_process_func_name": "get_hunyuan_video_15_i2v_pre_process_func",
        "post_process_func_name": "get_hunyuan_video_15_i2v_post_process_func",
    },
]

get_hunyuan_video_15_i2v_pre_process_func = _up_i2v.get_hunyuan_video_15_i2v_pre_process_func
get_hunyuan_video_15_i2v_post_process_func = _up_i2v.get_hunyuan_video_15_i2v_post_process_func
_Upstream = _up_i2v.HunyuanVideo15I2VPipeline


def _broadcast(obj):
    """Object broadcast from global rank 0 over the world's host group (no-op on one rank)."""
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return obj
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    box = [obj]
    dist.broadcast_object_list(box, src=0, group=get_world_group().cpu_group)
    return box[0]


class NeuronHunyuanVideo15I2VPipeline(NeuronHunyuanVideo15Pipeline):
    """Upstream I2V semantics on the Neuron T2V pipeline (host SigLIP + VAE encoder on global rank 0)."""

    support_image_input = True
    color_format = "RGB"

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        from transformers import SiglipImageProcessor, SiglipVisionModel

        model = local_model_dir(od_config.model)
        local = os.path.exists(model)
        dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            model, subfolder="feature_extractor", local_files_only=local
        )
        self.image_encoder = None
        if _global_rank() == 0:
            self.image_encoder = SiglipVisionModel.from_pretrained(
                model, subfolder="image_encoder", local_files_only=local, torch_dtype=dtype
            ).eval()
        self._default_guidance = 6.0
        with open(os.path.join(model, "transformer", "config.json")) as f:
            self._target_size = int(json.load(f).get("target_size", 960))  # resolution bucket base
        gpath = os.path.join(model, "guider", "guider_config.json")
        if os.path.exists(gpath):
            with open(gpath) as f:
                g = json.load(f)
            self._default_guidance = (
                float(g.get("guidance_scale", 6.0)) if g.get("enabled", True) else 1.0
            )

    # -- conditioning (host, rank 0, broadcast) ---------------------------------------------------
    def _get_image_embeds(self, image, device):
        out = None
        if _global_rank() == 0:
            t0 = time.time()
            with torch.no_grad(), _host_threads("HV15_TEXT_THREADS", 32):
                out = _Upstream._get_image_embeds(self, image, torch.device("cpu")).detach().cpu()
            _prof("image_encoder", t0)
        return _broadcast(out).to(device)

    def _get_image_latents(self, image, height, width, device):
        out = None
        if _global_rank() == 0:
            t0 = time.time()
            with torch.no_grad():
                out = (
                    _Upstream._get_image_latents(self, image, height, width, torch.device("cpu"))
                    .float()
                    .cpu()
                )
            _prof("vae_encode", t0)
        return _broadcast(out).to(device)

    prepare_cond_latents_and_mask = _Upstream.prepare_cond_latents_and_mask
    _upstream_forward = _Upstream.forward

    def _prepare_image(self, req):
        """diffusers / Tencent semantics the vLLM-Omni pipeline skips: the output size defaults to the
        checkpoint's resolution bucket for the image's aspect (``target_size``), and the image is resized
        with a centre crop to exactly that size BEFORE both SigLIP and the VAE encoder see it."""
        import PIL.Image
        from diffusers.pipelines.hunyuan_video1_5.image_processor import (
            HunyuanVideo15ImageProcessor,
        )

        p = req.prompts[0]
        mm = dict(p.get("multi_modal_data") or {}) if isinstance(p, dict) else {}
        img = mm.get("image")
        if isinstance(img, list):
            img = img[0]
        if img is None:
            return
        if isinstance(img, str):
            img = PIL.Image.open(img)
        img = img.convert("RGB")
        proc = HunyuanVideo15ImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        sp = req.sampling_params
        if not sp.height or not sp.width:
            sp.height, sp.width = proc.calculate_default_height_width(
                img.size[1], img.size[0], self._target_size
            )
        mm["image"] = proc.resize(img, height=sp.height, width=sp.width, resize_mode="crop")
        # in place: the served engine's request batch rebuilds ``prompts`` on every access (a property over its
        # requests), so assigning ``req.prompts[0]`` would be lost; the prompt dict itself is shared
        if isinstance(p.get("multi_modal_data"), dict):
            p["multi_modal_data"]["image"] = mm["image"]
        else:
            p["multi_modal_data"] = mm

    def forward(self, req, *args, **kwargs):
        self._prepare_image(req)
        if not req.sampling_params.guidance_scale_provided:
            kwargs.setdefault("guidance_scale", self._default_guidance)
        out = self._upstream_forward(req, *args, **kwargs)
        if _global_rank() == 0:
            logger.info("hv15 i2v stats: %s", self.transformer.stats)
        return out


__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronHunyuanVideo15I2VPipeline",
    "get_hunyuan_video_15_i2v_post_process_func",
    "get_hunyuan_video_15_i2v_pre_process_func",
]
