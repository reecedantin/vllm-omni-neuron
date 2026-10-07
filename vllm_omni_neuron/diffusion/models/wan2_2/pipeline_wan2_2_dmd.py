# SPDX-License-Identifier: Apache-2.0
"""NeuronWanDMDPipeline: few-step DMD2 sampling for FastWan2.2-TI2V-5B on Neuron.

``FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`` is Wan2.2-TI2V-5B distilled with DMD2 to
three denoising steps (``[1000, 757, 522]``) without classifier-free guidance. Its sampler is
FastVideo's ``DmdDenoisingStage``, which is not an ODE solver: at every step the DiT's flow
prediction is turned into a clean-latent estimate and, before the next step, that estimate is
re-noised to the next timestep with fresh Gaussian noise::

    sigma_t = training_sigmas[argmin |training_timesteps - t|]     # shift-8 training table
    x0      = x_t - sigma_t * v_theta(x_t, t)
    x_next  = (1 - sigma_next) * x0 + sigma_next * eps,    eps ~ N(0, I)

The sigma table is FastVideo's flow-matching *training* noise table (``shift=8``,
``DMD_TRAINING_NOISE_SHIFT``), not the inference scheduler; the DiT is fed the raw timesteps.
Everything else (text encoding, latents, the TI2V VAE, decode) is the Wan2.2 Neuron pipeline.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx

from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2 import (
    NeuronWanPipeline,
    _compile_lite_helper,
)
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # FastVideo's model_index ``_class_name`` for FastWan2.2-TI2V-5B.
        "model_arch": "WanDMDPipeline",
        "class_name": "NeuronWanDMDPipeline",
        "pre_process_func_name": "get_wan22_pre_process_func",
        "post_process_func_name": "get_neuron_wan22_post_process_func",
    },
]

# FastWan2_2_TI2V_5B_Config.dmd_denoising_steps and DMD_TRAINING_NOISE_SHIFT (FastVideo).
DEFAULT_DMD_TIMESTEPS = (1000, 757, 522)
DMD_TRAINING_NOISE_SHIFT = 8.0


def dmd_training_sigma_table(
    shift: float = DMD_TRAINING_NOISE_SHIFT, num_train_timesteps: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    """FastVideo's FlowMatchEulerDiscreteScheduler(shift) ``(timesteps, sigmas)`` training table."""
    t = torch.linspace(1, num_train_timesteps, num_train_timesteps, dtype=torch.float32).flip(0)
    sigmas = t / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas * num_train_timesteps, sigmas


def dmd_sigmas_for(
    timesteps: tuple[int, ...] | list[int], shift: float = DMD_TRAINING_NOISE_SHIFT
) -> list[float]:
    """Nearest training-table sigma per DMD timestep (``pred_noise_to_pred_video``'s lookup)."""
    table_t, table_s = dmd_training_sigma_table(shift)
    table_t, table_s = table_t.double(), table_s.double()
    out = []
    for t in timesteps:
        idx = int(torch.argmin((table_t - float(t)).abs()))
        out.append(float(table_s[idx]))
    return out


def _dmd_step(
    latents: torch.Tensor,
    flow: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DMD step in fp32. ``sigmas`` = ``[sigma_t, sigma_next]`` (fp32, 2 elements).

    Returns ``(x0, x_next)`` in the latents' dtype; ``x_next`` re-noises ``x0`` to sigma_next.
    """
    x0 = latents.float() - sigmas[0] * flow.float()
    x_next = (1.0 - sigmas[1]) * x0 + sigmas[1] * noise.float()
    return x0.to(latents.dtype), x_next.to(latents.dtype)


def _read_dmd_timesteps(model: str) -> tuple[int, ...]:
    """``dmd_denoising_steps`` from model_index.json if a checkpoint sets it, else the default."""
    path = os.path.join(model, "model_index.json")
    if os.path.isfile(path):
        with open(path) as f:
            index = json.load(f)
        for key in ("dmd_denoising_steps", "dmd_timesteps"):
            if index.get(key):
                return tuple(int(t) for t in index[key])
        block = index.get("dmd2_config") or {}
        if block.get("denoising_timesteps"):
            return tuple(int(t) for t in block["denoising_timesteps"])
    return DEFAULT_DMD_TIMESTEPS


class NeuronWanDMDPipeline(NeuronWanPipeline):
    """Wan2.2 TI2V-5B DMD2 (FastWan) few-step pipeline: no CFG, fixed timesteps, re-noising."""

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.dmd_timesteps = _read_dmd_timesteps(od_config.model)
        override = (od_config.model_config or {}).get("dmd_denoising_steps")
        if override:
            self.dmd_timesteps = tuple(int(t) for t in override)
        self.dmd_sigmas = dmd_sigmas_for(self.dmd_timesteps)
        self._dmd_seed: int | None = None
        logger.info(
            "DMD2 sampler: timesteps=%s sigmas=%s",
            list(self.dmd_timesteps),
            [round(s, 5) for s in self.dmd_sigmas],
        )

    # -- request handling -----------------------------------------------------------------

    def _sanitize_dmd_request(self, req) -> None:
        """Force the distilled contract: no CFG, no negative prompt, DMD step count."""
        sp = req.sampling_params
        if sp.guidance_scale_provided and sp.guidance_scale not in (None, 1.0):
            logger.warning(
                "DMD2: ignoring guidance_scale=%s (distilled, no CFG)", sp.guidance_scale
            )
        sp.guidance_scale = 1.0
        sp.guidance_scale_provided = True
        if getattr(sp, "guidance_scale_2", None) is not None:
            sp.guidance_scale_2 = None
        sp.num_inference_steps = len(self.dmd_timesteps)
        extra = getattr(sp, "extra_args", None) or {}
        for key in ("sample_solver", "flow_shift"):
            extra.pop(key, None)
        # ``req.prompts`` is a read-only view over the batch's requests; edit those.
        for request in req.requests if hasattr(req, "requests") else [req]:
            p = request.prompt
            if isinstance(p, dict) and "negative_prompt" in p:
                request.prompt = {k: v for k, v in p.items() if k != "negative_prompt"}
        self._dmd_seed = sp.seed

    def forward(self, req):
        if getattr(req, "request_ids", None) != ["dummy_req_id"]:
            self._sanitize_dmd_request(req)
        return super().forward(req)

    # -- denoise loop ---------------------------------------------------------------------

    def _dmd_noise(self, shape, device, step: int) -> torch.Tensor:
        """Fresh re-noising draw for step ``step`` (fp32, CPU generator, then to device)."""
        seed = 0 if self._dmd_seed is None else int(self._dmd_seed)
        gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + 17 + step)
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)

    def _run_dmd_step(self, latents, flow, noise, sigma_t, sigma_next):
        sigmas = torch.tensor([sigma_t, sigma_next], dtype=torch.float32).to(latents.device)
        compiled = getattr(self, "_compiled_dmd_step", None)
        if compiled is None:
            return _dmd_step(latents, flow, noise, sigmas)
        return compiled(latents, flow, noise, sigmas)

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict | None,
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DMD2 loop (``timesteps``/guidance from the base forward are ignored)."""
        del timesteps, negative_prompt_embeds, guidance_low, guidance_high, boundary_timestep
        if latent_condition is not None:
            raise NotImplementedError("NeuronWanDMDPipeline serves text-to-video only")
        attention_kwargs = attention_kwargs or {}
        steps = list(self.dmd_timesteps)
        sigmas = self.dmd_sigmas + [0.0]
        self._num_timesteps = len(steps)
        model = self.transformer
        x0 = latents
        with self.progress_bar(total=len(steps)) as pbar:
            for i, t in enumerate(steps):
                self._current_timestep = t
                set_forward_context_denoise_step_idx(i)
                timestep = torch.full((latents.shape[0],), float(t), dtype=torch.float32)
                timestep = timestep.to(latents.device)
                flow = self.predict_noise(
                    current_model=model,
                    hidden_states=latents.to(dtype) if latents.dtype != dtype else latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )
                last = i == len(steps) - 1
                noise = (
                    torch.zeros(latents.shape, dtype=torch.float32).to(latents.device)
                    if last
                    else self._dmd_noise(tuple(latents.shape), latents.device, i)
                )
                x0, latents = self._run_dmd_step(latents, flow, noise, sigmas[i], sigmas[i + 1])
                pbar.update()
        self._current_timestep = None
        return x0

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        if is_lite_runtime():
            self._compiled_dmd_step = _compile_lite_helper(_dmd_step, *args, **kwargs)
        return self
