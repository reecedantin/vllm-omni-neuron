# SPDX-License-Identifier: Apache-2.0
"""Z-Image CFG via the shared ``NeuronCFGParallelMixin`` (docs/model-dev/onboarding-models.md):
mixed in AHEAD of the upstream pipeline, this gives sequential (batch-1,
two forwards/step) CFG for free at ``cfg_parallel_size==1`` and CFG-parallel (one branch/core)
at ``cfg_parallel_size==2``, with no separate code path for either.

Upstream's ``ZImagePipeline.forward`` builds CFG by concatenating positive+negative into one
batch-2 ``self.transformer(...)`` call (`latent_model_input.repeat(2,...)`) -- there is no call
through ``predict_noise_maybe_with_cfg`` to hook. This module overrides the denoising loop only:
everything else (prompt encoding, scheduler setup, latent prep, output) is the identical upstream
code, copied once rather than re-derived, so a diff against upstream shows exactly the CFG section
that changed.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    calculate_shift,
    retrieve_timesteps,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from ...distributed.cfg_parallel import NeuronCFGParallelMixin

_PROFILE = os.environ.get("Z_IMAGE_PROFILE", "0") == "1"


def _prof(name: str, t0: float) -> None:
    if _PROFILE:
        print(f"[zimage-prof] {name} {time.time() - t0:.4f} pid={os.getpid()}", flush=True)


class ZImageCFGForwardMixin(NeuronCFGParallelMixin):
    """Mix in ahead of ``NeuronZImagePipeline`` -> ``ZImagePipeline`` (MRO:
    ``class NeuronZImagePipeline(ZImageCFGForwardMixin, ZImagePipeline)``)."""

    def predict_noise(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """One batch-1 transformer call -> float32 noise prediction, shape [1, C, H, W]."""
        out = self.transformer(
            list(latent_model_input.unbind(dim=0)), timestep, prompt_embeds_list
        )[0]
        return torch.stack([o.float() for o in out], dim=0).squeeze(2)

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale: float,
        cfg_normalize: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Z-Image's exact combine: ``pos + scale*(pos-neg)`` (not ``neg + scale*(pos-neg)`` --
        Z-Image's CFG direction is relative to the POSITIVE branch), with its own max-norm clamp
        renormalization, reproduced bit-for-bit from upstream's inline loop body.

        The base ``CFGParallelMixin`` always wraps a single-tensor ``predict_noise`` result in a
        1-tuple before calling this (its multi-output support), so unwrap first."""
        pos = (
            positive_noise_pred[0]
            if isinstance(positive_noise_pred, tuple)
            else positive_noise_pred
        )
        neg = (
            negative_noise_pred[0]
            if isinstance(negative_noise_pred, tuple)
            else negative_noise_pred
        )
        pred = pos + true_cfg_scale * (pos - neg)
        if cfg_normalize and float(cfg_normalize) > 0.0:
            ori_pos_norm = torch.linalg.vector_norm(pos)
            new_pos_norm = torch.linalg.vector_norm(pred)
            max_new_norm = ori_pos_norm * float(cfg_normalize)
            scale = torch.where(
                new_pos_norm > max_new_norm,
                (max_new_norm / new_pos_norm.clamp(min=1e-12)).to(pred.dtype),
                pred.new_tensor(1.0),
            )
            pred = pred * scale
        return pred

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict,
        negative_kwargs: dict | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        kwargs: dict | None = None,
    ):
        """``cfg_parallel_size == 1``: the mixin's sequential path (two batch-1 calls, then
        :meth:`combine_cfg_noise`). ``cfg_parallel_size == 2``: one branch per CFG rank, then a
        HOST all-gather over the CFG group's gloo ``cpu_group`` (``host_all_gather``: parts in
        group-rank order) and the same :meth:`combine_cfg_noise` on every rank.

        Why not ``NeuronCFGParallelMixin``'s compiled device gather: that path (a) assumes
        ``predict_noise`` returns a device tensor, but this pipeline's transformer facade returns
        HOST tensors (the pipeline math runs on CPU, as in Cosmos3-Edge), and (b) hardcodes the
        ``neg + s*(pos-neg)`` combine with no renormalization, while Z-Image uses
        ``pos + s*(pos-neg)`` plus a max-norm clamp. The prediction is one [1, 16, H/8, W/8] fp32
        latent (1 MB at 1024 px) per step, negligible next to a multi-second DiT forward.
        """
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            get_classifier_free_guidance_rank,
            get_classifier_free_guidance_world_size,
        )

        try:
            cfg_size = get_classifier_free_guidance_world_size()
        except AssertionError:  # no CFG group initialised (standalone / tests)
            cfg_size = 1
        if not do_true_cfg or cfg_size <= 1:
            return super(NeuronCFGParallelMixin, self).predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
                output_slice,
                kwargs,
            )
        from ...distributed.parallel_state import host_all_gather

        group = get_cfg_group()
        rank = get_classifier_free_guidance_rank()
        local = self.predict_noise(**(positive_kwargs if rank == 0 else negative_kwargs))
        local = local.detach().to("cpu").contiguous()
        t0 = time.time()
        # Parts come back in CFG group-rank order (branch 0 = positive), also on a descending
        # physical-mesh CFG group, where a raw all_gather over cpu_group would swap the branches.
        pos, neg = host_all_gather(group, local)
        _prof("cfg_gather", t0)
        return self.combine_cfg_noise(pos, neg, true_cfg_scale, cfg_normalize)

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:  # noqa: C901
        """Identical to upstream ``ZImagePipeline.forward`` except the denoising loop's CFG
        section (originally one batch-2 ``self.transformer`` call + an inline combine) now goes
        through ``predict_noise`` / ``predict_noise_maybe_with_cfg`` / ``combine_cfg_noise``, each
        a batch-1 call. See the module docstring."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = [
                "" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts
            ]
        else:
            negative_prompt = None

        prompt_embeds = None
        negative_prompt_embeds = None
        image = None  # img2img not exercised by this override; falls back to upstream if needed
        if req.prompts and len(req.prompts) == 1:
            first_prompt = req.prompts[0]
            if not isinstance(first_prompt, str):
                raw_image = first_prompt.get("multi_modal_data", {}).get("image")
                if raw_image is not None:
                    import PIL.Image

                    image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else raw_image

        if image is not None:
            # img2img path is untested under this override; defer to upstream's forward.
            return super().forward(req)

        height = req.sampling_params.height or 1024
        width = req.sampling_params.width or 1024
        num_inference_steps = req.sampling_params.num_inference_steps or 50
        generator = req.sampling_params.generator
        sigmas = req.sampling_params.sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or 512
        guidance_scale = req.sampling_params.guidance_scale
        if not getattr(req.sampling_params, "guidance_scale_provided", True):
            # vLLM-Omni fills an omitted (or 0) guidance scale with 1.0, the "off" value of the
            # neg + g * (pos - neg) convention. Z-Image combines pos + g * (pos - neg), where off
            # is 0; 1.0 would run a real CFG pass (twice the DiT calls) and change the result,
            # e.g. for guidance-distilled Z-Image-Turbo.
            guidance_scale = 0.0
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else 1
        )
        latents = req.sampling_params.latents
        cfg_normalization = req.sampling_params.cfg_normalize
        cfg_truncation = req.sampling_params.extra_args.get("cfg_truncation", 1.0)
        output_type = req.sampling_params.output_type or "pil"

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(
                f"height/width must be divisible by {vae_scale} (got {height}x{width})"
            )

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._interrupt = False
        batch_size = len(prompt)

        t_req = time.time()
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [
                    npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)
                ]

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        timesteps_tensor = timesteps.to(device=device, dtype=torch.float32)
        t_norm_list = ((1000 - timesteps_tensor) / 1000).cpu().tolist()
        if not isinstance(t_norm_list, list):
            t_norm_list = [t_norm_list]

        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            timestep = (1000 - t.expand(latents.shape[0])) / 1000
            t_norm = t_norm_list[i]

            current_guidance_scale = self.guidance_scale
            if (
                self.do_classifier_free_guidance
                and cfg_truncation is not None
                and float(cfg_truncation) <= 1
                and t_norm > cfg_truncation
            ):
                current_guidance_scale = 0.0
            apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

            latents_typed = latents.to(self.od_config.dtype).unsqueeze(2)
            pos_kwargs = {
                "latent_model_input": latents_typed,
                "timestep": timestep,
                "prompt_embeds_list": prompt_embeds,
            }
            if apply_cfg:
                neg_kwargs = {
                    "latent_model_input": latents_typed,
                    "timestep": timestep,
                    "prompt_embeds_list": negative_prompt_embeds,
                }
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=pos_kwargs,
                    negative_kwargs=neg_kwargs,
                    cfg_normalize=cfg_normalization,
                )
            else:
                noise_pred = self.predict_noise(**pos_kwargs)

            noise_pred = -noise_pred
            latents = self.scheduler.step(
                noise_pred.to(torch.float32), t, latents, return_dict=False
            )[0]
            assert latents.dtype == torch.float32

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            t0 = time.time()
            image = self.vae.decode(latents, return_dict=False)[0]
            _prof("vae_decode", t0)

        _prof("pipeline_forward", t_req)
        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        return DiffusionOutput(output=image, stage_durations=stage_durations)
