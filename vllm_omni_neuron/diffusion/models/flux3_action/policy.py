# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action policy on Neuron: observation -> action chunk (+ predicted video latents).

Host orchestration mirrors upstream ``FluxActionPolicy`` (``flux_action/policy.py``) for the
``default`` inference profile (DROID): camera canvas, single-frame VAE encode of the observation,
state token, fixed-seed noise, Qwen3-VL caption context (cached per caption), two-pass CFG and
the Cosmos UniPC (order 2) solver. All of that host math is upstream's vendored code
(``_vendor``); the NeuronCore runs the DiT phases (:mod:`.dit`) and the VAE encoder
(:mod:`.video_vae`).

The conditioning latent is encoded from the observed frame alone, as upstream's
``prepare_inference()`` serving path does for every released package (the unprepared reference
path with ``single_frame_encode=False`` repeats the frame to a 45-frame clip instead; about 0.5 %
latent difference per upstream's own note).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ._vendor import normalization, packing, sampling
from ._vendor.config import PolicyConfig
from ._vendor.positional import batched_prc_action, batched_prc_vid, times_to_ids
from .dit import DiTDims, Flux3ActionDiT, load_policy_config
from .video_vae import VideoVAE

logger = logging.getLogger(__name__)

VEC_DIM = 768
BASE_REPO = "black-forest-labs/flux-3-action-base"


def resolve_component(spec: str | None, base_dir: str | None, kind: str) -> str:
    """A config's ``video_vae_id`` / ``text_encoder_id`` -> a local path.

    Resolution order: a local path in the spec; else ``base_dir`` (or ``$FLUX3_ACTION_BASE``), a local
    copy of ``black-forest-labs/flux-3-action-base``; else the Hugging Face Hub at the revision the
    spec pins (``repo[:file]@rev``), which also reads an offline cache under ``HF_HUB_OFFLINE=1``.
    """
    if spec and os.path.exists(spec):
        return spec
    default = {"video_vae": "video_vae.safetensors", "text_encoder": "text_encoder"}[kind]
    repo, name, revision = BASE_REPO, default, None
    if spec:
        body, _, revision = spec.partition("@")
        repo, _, fname = body.partition(":")
        name = fname or default
        revision = revision or None
    base_dir = base_dir or os.environ.get("FLUX3_ACTION_BASE")
    if base_dir:
        if repo != BASE_REPO:
            raise FileNotFoundError(f"{kind}: {spec!r} does not refer to {BASE_REPO}")
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download, snapshot_download

    if kind == "text_encoder":
        root = snapshot_download(repo, revision=revision, allow_patterns=[f"{name}/*"])
        return os.path.join(root, name)
    return hf_hub_download(repo, name, revision=revision)


@dataclass
class PolicyOutput:
    actions: Tensor  # (B, chunk, D) absolute commands in dataset units
    targets: Tensor  # (B, chunk, D) normalized model targets (before denormalization / integration)
    video_latents: Tensor  # (B, 96, n_pred, h, w) predicted (normalized) video latents
    cond_latents: Tensor  # (B, 96, 1, h, w) observation latent
    timing: dict[str, float] = field(default_factory=dict)


class NeuronFlux3ActionPolicy:
    """DROID-style (``inference_profile == "default"``) FLUX 3 Action policy."""

    def __init__(
        self,
        policy_dir: str,
        *,
        base_dir: str | None = None,
        device: str | torch.device = "cpu",
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group=None,
        text_encoder=None,
        dtype: torch.dtype = torch.bfloat16,
        vae_device: str | torch.device | None = None,
        cfg_size: int = 1,
        cfg_rank: int = 0,
        cfg_group=None,
        world_rank: int = 0,
        world_group=None,
        text_device: str | torch.device | None = None,
    ):
        raw = load_policy_config(policy_dir)
        self.raw_config = raw
        known = set(PolicyConfig.__dataclass_fields__)
        self.config = PolicyConfig(**{k: v for k, v in raw.items() if k in known})
        self.config.validate_inference()
        if self.config.inference_profile != "default":
            raise NotImplementedError(
                "only the default (DROID) inference profile is wired up so far"
            )
        if self.config.quantization is not None:
            raise NotImplementedError("FP8r packages are not supported; use a BF16 package")
        self.device = torch.device(device)
        # The VAE ENCODER runs on ``vae_device``: the NeuronCore (its neighborhood attention through
        # the plugin's shared halo-tiled op) or the host. The decoder (optional frame decode) always
        # runs on the host. With several ranks (TP and/or CFG-parallel) only world rank 0 encodes and
        # broadcasts the latent over ``world_group`` (a gloo CPU group), so the other ranks keep the
        # encoder on the host and never run it.
        self.vae_device = torch.device(vae_device) if vae_device is not None else self.device
        self.world_rank, self.world_group = world_rank, world_group
        if world_rank != 0:
            self.vae_device = torch.device("cpu")
        # CFG-parallel: the two guidance branches run on two TP replicas at once (cfg rank 0 the
        # conditional branch, cfg rank 1 the unconditional one), are exchanged over ``cfg_group`` (the
        # CFG GroupCoordinator, or anything with ``ranks`` / ``world_size`` / ``cpu_group``; the
        # DiT returns host tensors, so the exchange runs on its gloo CPU group) and combined
        # identically on every rank.
        if cfg_size not in (1, 2):
            raise ValueError(f"cfg_size must be 1 or 2, got {cfg_size}")
        self.cfg_size, self.cfg_rank, self.cfg_group = cfg_size, cfg_rank, cfg_group
        self.dtype = dtype
        self.tp_size, self.tp_rank, self.tp_group = tp_size, tp_rank, tp_group
        self.dims = DiTDims.from_policy_config(raw)
        self.dit = Flux3ActionDiT(
            self.dims, dtype=dtype, tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group
        )
        t0 = time.time()
        self.dit.load(os.path.join(policy_dir, "model.safetensors"), device=self.device)
        self.load_s = {"dit": time.time() - t0}
        t0 = time.time()
        vae_path = resolve_component(self.config.video_vae_id, base_dir, "video_vae")
        self.vae = VideoVAE.from_file(vae_path, dtype=dtype)
        if self.vae_device.type != "cpu":
            self.vae.model.encoder.to(self.vae_device)
            self.vae.prepare_device_attention(self.config.canvas_hw, self.vae_device, dtype)
        self.load_s["vae"] = time.time() - t0
        if text_encoder is None:
            from .text_encoder import Qwen3VLTextEncoder

            t0 = time.time()
            text_encoder = Qwen3VLTextEncoder(
                resolve_component(self.config.text_encoder_id, base_dir, "text_encoder")
            )
            self.load_s["text_encoder"] = time.time() - t0
        self.text_encoder = text_encoder
        # The caption encoder's 32 decoder layers run on ``text_device`` (default: the DiT's
        # device), head-sharded over the DiT's TP group: each CFG replica encodes for itself, so no
        # cross-replica exchange is needed. "cpu" keeps them on the host (the fallback).
        self.text_device = torch.device(text_device) if text_device is not None else self.device
        if self.text_device.type != "cpu" and hasattr(self.text_encoder, "to_device"):
            t0 = time.time()
            self.text_encoder.to_device(
                self.text_device, tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group
            )
            self.load_s["text_encoder_to_device"] = time.time() - t0
        else:
            self.text_device = torch.device("cpu")
        self._encoder_mu = self.vae.encode_frame_mu
        self._ctx_cache: dict[str, tuple[Tensor, Tensor]] = {}
        self.last_text_encode_s = 0.0
        self._prepared_cache: dict[tuple, Any] = {}

    # -- compile --------------------------------------------------------------------------
    def compile(
        self, backend: str, options: dict | None = None, *, compile_vae: bool = True
    ) -> None:
        self.dit.compile(backend, options)
        if self.text_device.type != "cpu":
            self.text_encoder.compile(backend, options)
        if not compile_vae or self.vae_device.type == "cpu":
            return  # VAE on host: left eager
        self.vae.compile_encoder(
            backend,
            options,
            [
                "--model-type=unet-inference",
                "--auto-cast=none",
                "--internal-max-instruction-limit=15000000",
                "-O1",
            ],
        )

    # -- helpers (upstream semantics) ---------------------------------------------------
    def _flip(self, x: Tensor) -> Tensor:
        dims = list(self.config.gripper_flip_dims)
        if not dims:
            return x
        x = x.clone()
        x[..., dims] = 1.0 - x[..., dims]
        return x

    def context(self, caption: str) -> tuple[Tensor, Tensor]:
        hit = self._ctx_cache.get(caption)
        if hit is None:
            t0 = time.time()
            ctx = self.text_encoder.encode([caption])[0].to(torch.bfloat16)
            self.last_text_encode_s = time.time() - t0
            hit = (ctx, packing.pack_text(ctx, VEC_DIM)["ctx_ids"])
            self._ctx_cache[caption] = hit
        return hit

    def _vae_encode(self, frame: Tensor) -> Tensor:
        """Run the VAE encoder: on the NeuronCore (compiled, normalized on the host) or on the host
        with an explicit intra-op thread count.

        Serving workers can run with a low intra-op thread count (OMP/MKL pinned per process). The host VAE encode is the dominant
        request stage (~29-41 s at 10 threads), and under TP only rank 0 runs it while ranks 1-3 wait
        in the broadcast, so rank 0 can use more cores. ``FLUX3_ACTION_VAE_THREADS`` (default 32)
        sets the count for this call only; the previous setting is restored afterwards.
        """
        if self.vae_device.type != "cpu":
            mu = self._encoder_mu(frame.to(self.vae_device).contiguous())
            return self.vae.normalize(mu.to("cpu"))
        n = int(os.environ.get("FLUX3_ACTION_VAE_THREADS", "32"))
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            t0 = time.time()
            out = self.vae.normalize(self._encoder_mu(frame))
            logger.info(
                "flux3_action host VAE encode: %.2fs on %d threads (was %d)",
                time.time() - t0,
                n,
                prev,
            )
            return out
        finally:
            torch.set_num_threads(prev)

    @torch.no_grad()
    def encode_observation(self, cams: Tensor) -> Tensor:
        """``cams (n_cams, 1, 3, H, W)`` uint8/float -> conditioning latent ``(1, 96, 1, h, w)`` (CPU, bf16)."""
        cfg = self.config
        frame = None
        if self.world_rank == 0:
            canvas = packing.materialize_video(
                cams, None, "cpu", layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw
            )
            frame = canvas[:, 0][None].to(torch.bfloat16)  # (1, 3, Hc, Wc)
        return self._encode_and_share(frame)

    def _encode_and_share(self, frame: Tensor | None) -> Tensor:
        """Encode ``frame (1, 3, Hc, Wc)`` on world rank 0 and share the latent with every rank.

        With several ranks (TP and/or CFG-parallel) only rank 0 runs the VAE encode
        (onboarding-models.md §1c: "only rank 0 instantiates ... other ranks skip it entirely") and
        broadcasts the small, fixed-shape latent; the other ranks pass ``None``.
        """
        cfg = self.config
        shape = (1, 96, 1, *cfg.latent_hw)
        if self.world_rank != 0:
            lat = torch.zeros(shape, dtype=torch.float32)
        else:
            full = self._vae_encode(frame)
            lat = full[..., : cfg.latent_hw[0], : cfg.latent_hw[1]].float().contiguous()
            assert tuple(lat.shape) == shape, (lat.shape, shape)
        if self.world_group is not None:
            # Same staging as Wan2.2 I2V: CPU fp32 over the world group's gloo cpu_group. A broadcast
            # over the Neuron device group outside a compiled graph is silently dropped under Lite.
            # Every rank reaches this call in lockstep (the startup dummy run included), so the
            # collective cannot deadlock.
            torch.distributed.broadcast(lat, src=0, group=self.world_group)
        return lat.to(torch.bfloat16).clone()

    def _cfg_gather(self, local: Tensor) -> tuple[Tensor, Tensor]:
        """(conditional, unconditional) branch outputs from the two CFG ranks, bit-exact in the
        branch dtype (exchanged as fp32 over the gloo CPU group, which widens bf16 exactly).

        ``host_all_gather`` returns the parts in group-rank order (``parts[0]`` from CFG rank 0, the
        conditional branch) even when the CFG group is descending, as on the Trn2 physical-mesh
        layouts, where a positional c10d gather would come back sorted and swap the branches."""
        from vllm_omni_neuron.diffusion.distributed.parallel_state import host_all_gather

        x = local.detach().to("cpu", torch.float32)
        cond, uncond = host_all_gather(self.cfg_group, x)
        return cond.to(local.dtype), uncond.to(local.dtype)

    # -- sampling -----------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self, cond_latent: Tensor, state_token: Tensor, caption: str, seed: int
    ) -> dict[str, Tensor]:
        """Joint video + action denoising (upstream ``_sample`` + ``_sample_prepared``)."""
        cfg, m = self.config, self.dims.action
        ak = f"x_{m}"
        video_cond, video_cond_ids = batched_prc_vid(
            cond_latent.to(torch.bfloat16), packing.video_time_ids(1, 0, 1, fps=cfg.fps)
        )
        values = state_token[:, None] * cfg.action_scale
        action_cond, action_cond_ids = batched_prc_action(
            values.transpose(1, 2), times_to_ids(torch.zeros(1, 1))
        )
        n_pred = packing.latent_frames(cfg.window_frames) - 1
        rng = torch.Generator().manual_seed(seed)
        video_noise = torch.randn(1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng)
        x_video, x_video_ids = batched_prc_vid(
            video_noise, packing.video_time_ids(n_pred, 1, 1, fps=cfg.fps)
        )
        times = packing.default_action_times(1, cfg.chunk_size, cfg.fps)
        action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
        x_action, x_action_ids = batched_prc_action(action_noise, times_to_ids(times))
        flow = {"x_video": x_video, ak: x_action}
        guidance = {
            "x_video": cfg.guidance_scale,
            ak: cfg.guidance_scale
            if cfg.guidance_scale_action is None
            else cfg.guidance_scale_action,
        }
        ctx_c = self.context(caption)
        ctx_uc = self.context("") if any(g != 1.0 for g in guidance.values()) else None

        def prepare(ctx):
            return self.dit.prepare(
                ctx[0],
                ctx[1],
                video_ids=x_video_ids,
                video_cond=video_cond,
                video_cond_ids=video_cond_ids,
                action_ids=x_action_ids,
                action_cond=action_cond.float(),
                action_cond_ids=action_cond_ids,
            )

        t0 = time.time()
        request = prepare(ctx_c)
        negative = prepare(ctx_uc) if ctx_uc is not None else None
        prepare_s = time.time() - t0
        _, ticks = sampling.cosmos_unipc_schedule(cfg.num_inference_steps, cfg.sampler_shift)
        times_s = (ticks.to(torch.float32) / 1000.0).tolist()
        t0 = time.time()
        steps = [self.dit.prepare_step(request, t, t) for t in times_s]
        step_prep_s = time.time() - t0
        phase = 0
        forward_s = 0.0

        cfg_parallel = negative is not None and self.cfg_size == 2
        gather_s = 0.0

        def predict(samples: dict[str, Tensor], _t) -> dict[str, Tensor]:
            nonlocal phase, forward_s, gather_s
            t0 = time.time()
            v_in, a_in = samples["x_video"].to(torch.bfloat16), samples[ak].to(torch.bfloat16)
            if cfg_parallel:
                branch = request if self.cfg_rank == 0 else negative
                lv, la = self.dit.forward(branch, steps[phase], v_in, a_in)
                tg = time.time()
                pv, nv = self._cfg_gather(lv)
                pa, na = self._cfg_gather(la)
                gather_s += time.time() - tg
                pv = nv + guidance["x_video"] * (pv - nv)
                pa = na + guidance[ak] * (pa - na)
            else:
                pv, pa = self.dit.forward(request, steps[phase], v_in, a_in)
                if negative is not None:
                    nv, na = self.dit.forward(negative, steps[phase], v_in, a_in)
                    pv = nv + guidance["x_video"] * (pv - nv)
                    pa = na + guidance[ak] * (pa - na)
            phase += 1  # CFG combine in the model dtype, as upstream does on its bf16 outputs
            forward_s += time.time() - t0
            return {"x_video": pv.float(), ak: pa.float()}

        if cfg.sampler == "cosmos_unipc":
            out = sampling.cosmos_unipc_order2(
                flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift
            )
        else:
            raise NotImplementedError(
                f"sampler {cfg.sampler!r}: only cosmos_unipc (the released DROID sampler)"
            )
        self._last_sample_timing = {
            "prepare_text_obs_s": prepare_s,
            "prepare_step_s": step_prep_s,
            "denoise_forward_s": forward_s,
            "num_steps": cfg.num_inference_steps,
            "cfg_branches": 2 if negative is not None else 1,
            "cfg_parallel": cfg_parallel,
            "cfg_gather_s": gather_s,
        }
        return out

    @torch.no_grad()
    def predict(self, batch: dict[str, Any], seed: int | None = None) -> PolicyOutput:
        """Observation batch (upstream keys: ``images.<camera>``, ``state``, ``task``) -> actions."""
        cfg = self.config
        seed = cfg.inference_seed if seed is None else seed
        cams = []
        for key in cfg.camera_order:
            img = batch[key]
            if img.ndim == 4:
                img = img[:, None]
            cams.append(img)
        cams = torch.stack(cams, dim=1)[:, :, :1]  # (B, n_cams, 1, 3, H, W)
        state = batch["state"].float()
        if state.ndim == 3:
            state = state[:, 0]
        b = state.shape[0]
        task = batch.get("task")
        captions = (
            [""] * b
            if task is None
            else (
                [task] * b if isinstance(task, str) else list(task) * (b if len(task) == 1 else 1)
            )
        )
        flipped = self._flip(state)
        token = normalization.normalize(flipped, cfg.state_normalization, cfg.normalization_clip)
        timing: dict[str, float] = {}
        targets, videos, conds = [], [], []
        for i in range(b):
            t0 = time.time()
            cond = self.encode_observation(cams[i])
            timing["vae_encode_s"] = time.time() - t0
            timing["vae_device"] = self.vae_device.type
            if self.vae_device.type == "cpu":
                timing["vae_threads"] = int(os.environ.get("FLUX3_ACTION_VAE_THREADS", "32"))
            t0 = time.time()
            new_captions = sum(c not in self._ctx_cache for c in {captions[i], ""})
            for c in {captions[i], ""}:
                self.context(c)
            timing["text_s"] = time.time() - t0
            timing["text_new_captions"] = new_captions
            timing["text_device"] = self.text_device.type
            t0 = time.time()
            out = self.sample(cond, token[i : i + 1], captions[i], seed)
            timing["denoise_s"] = time.time() - t0
            timing.update(getattr(self, "_last_sample_timing", {}))
            targets.append(out[f"x_{self.dims.action}"][0].float() / cfg.action_scale)
            n_pred = packing.latent_frames(cfg.window_frames) - 1
            h, w = cfg.latent_hw
            videos.append(out["x_video"][0].reshape(n_pred, h, w, -1).permute(3, 0, 1, 2))
            conds.append(cond[0])
        targets_t = torch.stack(targets)
        actions = []
        for tgt, s in zip(targets_t, flipped, strict=True):
            a = normalization.denormalize(tgt, cfg.action_normalization)
            if cfg.action_parameterization == "joint_delta":
                a = normalization.integrate(a, s, cfg.absolute_action_dims)
            actions.append(a)
        actions_t = self._flip(torch.stack(actions)).float()
        video_t = torch.stack(videos)
        if self.world_group is not None and os.environ.get("FLUX3_ACTION_RANK_CHECK") == "1":
            timing.update(self._rank_agreement(actions_t, video_t))
        return PolicyOutput(
            actions=actions_t,
            targets=targets_t,
            video_latents=video_t,
            cond_latents=torch.stack(conds),
            timing=timing,
        )

    def _rank_agreement(self, actions: Tensor, video: Tensor) -> dict[str, Any]:
        """Opt-in (``FLUX3_ACTION_RANK_CHECK=1``) all-rank check: every world rank's action chunk and
        a digest of its predicted video latents, gathered over the world gloo group. Every rank runs
        the full sampler (TP replicas and both CFG ranks combine the same gathered branches), so
        the outputs must be bit-identical; ``ranks_agree`` says whether they are."""
        import hashlib

        def digest(t: Tensor) -> str:
            return hashlib.sha256(
                t.detach().cpu().float().contiguous().numpy().tobytes()
            ).hexdigest()

        mine = (self.world_rank, actions.detach().cpu().float(), digest(video))
        n = torch.distributed.get_world_size(self.world_group)
        parts: list = [None] * n
        torch.distributed.all_gather_object(parts, mine, group=self.world_group)
        parts.sort(key=lambda p: p[0])
        ref = parts[0][1]
        max_rel = max(float((a - ref).norm() / ref.norm().clamp_min(1e-12)) for _, a, _ in parts)
        return {
            "rank_check_ranks": [r for r, _, _ in parts],
            "rank_action_digests": [digest(a)[:16] for _, a, _ in parts],
            "rank_video_digests": [d[:16] for _, _, d in parts],
            "rank_action_max_rel": max_rel,
            "ranks_agree": all(torch.equal(a, ref) and d == parts[0][2] for _, a, d in parts),
        }

    def decode_frames(
        self, out: PolicyOutput, index: int = 0, max_latent_frames: int | None = None
    ) -> Tensor:
        """Observation + predicted latents -> pixels ``(3, 4 * T_lat - 3, H, W)`` in [-1, 1] (host decode).

        ``max_latent_frames`` caps the number of latent frames decoded (cond + predicted) -- the full
        9-frame DROID decode on CPU is memory- and time-heavy because the VAE's 3D neighborhood
        attention runs over the whole pixel grid; a small cap (e.g. 2) gives a quick frame sample.
        """
        z = torch.cat([out.cond_latents[index], out.video_latents[index]], dim=1)
        if max_latent_frames is not None:
            z = z[:, :max_latent_frames]
        z = z[None].to(self.vae.model.decoder.proj_in.weight.dtype)
        return self.vae.decode(z)[0].float()  # the decoder always lives on the host

    @torch.no_grad()
    def predict_from_composite(
        self, composite: Tensor, state: Tensor, task: str, seed: int | None = None
    ) -> Tensor:
        """A DROID 540x640 composite (HWC uint8 or CHW float in [0,1]) + ``state (8,)`` -> ``(1, chunk, 8)``
        absolute commands. The placement equals the three-camera canvas, so a composite built like the
        training composite gives identical actions (upstream ``predict_from_composite``)."""
        cfg = self.config
        assert cfg.camera_layout == "droid", "composite inference requires the DROID layout"
        seed = cfg.inference_seed if seed is None else seed
        if composite.dtype == torch.uint8:
            composite = (
                composite.permute(2, 0, 1).float().div_(255.0)
                if composite.shape[-1] == 3
                else composite.float().div_(255.0)
            )
        canvas = packing.pad_composite(composite[None], cfg.canvas_hw)  # (3, 1, Hc, Wc)
        flipped = self._flip(state.float().reshape(1, -1))
        token = normalization.normalize(flipped, cfg.state_normalization, cfg.normalization_clip)
        frame = canvas[:, 0][None].to(torch.bfloat16)
        lat = self._encode_and_share(frame if self.world_rank == 0 else None)
        chunk = self.sample(lat, token, task, seed)
        tgt = chunk[f"x_{self.dims.action}"][0].float() / cfg.action_scale
        a = normalization.denormalize(tgt, cfg.action_normalization)
        if cfg.action_parameterization == "joint_delta":
            a = normalization.integrate(a, flipped[0], cfg.absolute_action_dims)
        return self._flip(a).float()[None]


def save_report(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
