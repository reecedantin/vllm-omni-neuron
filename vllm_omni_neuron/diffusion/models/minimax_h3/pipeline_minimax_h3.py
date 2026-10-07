# SPDX-License-Identifier: Apache-2.0
"""Neuron MiniMax-H3 / FastH3 pipeline (text -> video + audio, t2va).

The pinned vLLM-Omni 0.24 has no MiniMax-H3 pipeline to subclass, so this is a standalone pipeline over the
vendored diffusers modeling code (``_vendor/``), following diffusers' modular MiniMax-H3 pipeline step by step:

1. **Text** — the prompt verbatim (no chat template, no special tokens) through Qwen3-VL, read at
   ``hidden_states[50]`` (:mod:`.text_encoder`). Host CPU on the output-owner rank, broadcast to the stage ranks.
2. **Layout** — the packed ``[text | audio | video]`` sequence, RoPE tables and noise (:mod:`.layout`), on the host.
3. **Denoise** — one DiT forward per step on the NeuronCores (:mod:`.transformer`, TP x CP over the stage's cores),
   the two rectified-flow schedulers (video shift 12, audio shift 3) stepped on the host in fp32.
4. **Decode** — video VAE (ViT decoder) and audio VAE (BigVGAN) on the output-owner rank's host.

FastH3 and base MiniMax-H3 share the architecture; they differ in step count (FastH3: 4 forwards =
``num_inference_steps=5`` grid points, from the release's ``fastvideo_inference.json``) and weights.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import torch
import torch.nn as nn

from . import config as C
from .config import (
    BASE_NUM_INFERENCE_STEPS,
    BASE_TASKS,
    MiniMaxH3DiTConfig,
    base_sigmas,
    inference_contract,
    is_base_checkpoint,
    read_model_index,
    step_positions,
    text_encoder_layer,
    vsa_sparsity,
)
from .layout import build_layout, draw_noise, load_schedulers, unpack_audio, unpatchify_video
from .transformer import NeuronMiniMaxH3Transformer, host_adaln_tables

logger = logging.getLogger(__name__)

PIXEL_MEAN = (0.485, 0.456, 0.406)  # the video VAE works in ImageNet-normalized [0, 1] RGB
PIXEL_STD = (0.229, 0.224, 0.225)
AUDIO_SAMPLE_RATE = 32000

TRANSFORMER_COMPILER_ARGS = ["--model-type=transformer", "--auto-cast=none", "-O1"]

PIPELINE_REGISTRY = [
    {
        # FastH3 releases carry modular_model_index.json with this class name
        "model_arch": "MiniMaxH3ModularPipeline",
        "class_name": "NeuronMiniMaxH3Pipeline",
        "post_process_func_name": "get_minimax_h3_post_process_func",
    },
    {
        "model_arch": "MiniMaxH3Pipeline",
        "class_name": "NeuronMiniMaxH3Pipeline",
        "post_process_func_name": "get_minimax_h3_post_process_func",
    },
]


def get_minimax_h3_post_process_func(od_config=None):
    def post_process(output):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            return {
                "video": video,
                "audio": audio,
                "audio_sample_rate": AUDIO_SAMPLE_RATE,
                "fps": C.FPS,
            }
        return output

    return post_process


def _default_steps(model_path: str) -> int:
    """Sigma grid points for the release: FastH3's ``fastvideo_inference.json`` says 5 (4 forwards)."""
    p = os.path.join(model_path, "fastvideo_inference.json")
    if os.path.isfile(p):
        with open(p) as f:
            return int(json.load(f).get("num_inference_steps", 5))
    return int(read_model_index(model_path).get("num_inference_steps", 50))


def _stage_group():
    """The stage's whole world (TP x CP ranks) as a GroupCoordinator, or None outside a distributed run. The
    output rank broadcasts the prompt embeddings over it, and the tile-parallel video decode deals tiles over it."""
    for mod in (
        "vllm_omni.diffusion.distributed.parallel_state",
        "vllm.distributed.parallel_state",
    ):
        try:
            g = __import__(mod, fromlist=["get_world_group"]).get_world_group()
        except (AssertionError, ImportError, AttributeError):
            continue
        if g is not None:
            return g
    return None


class NeuronMiniMaxH3Pipeline(nn.Module):
    supports_request_batch = False

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.model_path = od_config.model
        mc = dict(od_config.model_config or {})
        self.dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        tdir = os.path.join(self.model_path, "transformer")
        self.cfg = MiniMaxH3DiTConfig.from_dir(tdir)
        # VSA-H3 students (FastH3 8-Step-V2) declare their sparsity in the inference contract; dense otherwise.
        # model_config.vsa = "off" runs a VSA checkpoint densely (an approximation; diagnostics only).
        sparsity = vsa_sparsity(self.model_path)
        if str(mc.get("vsa", "auto")).lower() == "off":
            sparsity = None
        self.transformer = NeuronMiniMaxH3Transformer(
            self.cfg, dtype=self.dtype, adaln=mc.get("adaln"), vsa_sparsity=sparsity
        )
        # step ladder:
        #   "contract" = the checkpoint's trained rungs (FastVideo's current behaviour for every FastH3 export),
        #   "linspace" = the scheduler's own linspace(1, 0, N) grid,
        #   "auto" (default) = linspace for the Preview-v1 4-step exports (their contract carries no scheduler
        #   shifts -- FastVideo's own marker for those "earlier exports"), contract for everything later.
        # auto pins the 4-step checkpoint to linspace because that is what its accepted parity gate (M1, vs
        # ref_fp32_256_neuronprompt.pt) and the accepted M3 timings ran on; the two grids differ only in the first
        # rung (0.999 vs 1.0 before the shift). 8-Step-V2 always runs its trained ladder.
        self.schedule = str(
            mc.get("schedule", os.environ.get("MINIMAX_H3_SCHEDULE", "auto"))
        ).lower()
        if self.schedule == "auto":
            has_shifts = "video_scheduler_shift" in inference_contract(self.model_path)
            self.schedule = "contract" if has_shifts else "linspace"
        self.host_adaln = (
            host_adaln_tables(tdir, self.cfg, self.dtype)
            if self.transformer.adaln == "host"
            else None
        )
        self.default_steps = int(mc.get("num_inference_steps") or _default_steps(self.model_path))
        # Base MiniMax-H3 (no FastVideo contract): the request contract of vLLM-Omni v0.30.0's MiniMax-H3 pipeline.
        # num_inference_steps counts denoiser evaluations (default 50, sigma boundaries linspace(1, 0, N + 1)
        # shifted by flow_shift / audio_flow_shift, default the checkpoint's 12 / 3), one conditional forward per
        # step: the checkpoint is guidance-distilled, so there is no negative branch and guidance_scale /
        # negative_prompt have no effect, as upstream. FastH3 students keep their own contract above.
        self.is_base = is_base_checkpoint(self.model_path)
        self._base_shifts: tuple[float, float] | None = None
        if self.is_base:
            self.default_steps = int(mc.get("num_inference_steps") or BASE_NUM_INFERENCE_STEPS)
        self.text_layer = text_encoder_layer(self.model_path)
        self.vae_mode = mc.get("vae", os.environ.get("MINIMAX_H3_VAE", "device"))  # device | cpu
        self.is_output_rank = self.transformer.tp_rank == 0 and self.transformer.cp_rank == 0
        # Tile-parallel video decode: every rank of the stage (TP x CP) holds the ViT decoder and decodes a share
        # of the (temporal chunk x spatial tile) work items; rank 0 gathers and blends. Device VAE only.
        ranks = self.transformer.tp_size * self.transformer.cp_size
        par = mc.get("vae_tile_parallel", os.environ.get("MINIMAX_H3_VAE_TILE_PARALLEL", "1"))
        self.vae_tile_parallel = (
            self.vae_mode == "device" and ranks > 1 and str(par).lower() in ("1", "true", "yes")
        )
        self.has_vae = self.is_output_rank or self.vae_tile_parallel
        # The audio VAE (host BigVGAN) decodes on another rank's process (the stage's last rank), in parallel with
        # the video decode, and its waveform is sent to the output rank: on rank 0 it would share one process's
        # torch thread pool with the video blend (1.9 s at 768p, 1.0 s alone). MINIMAX_H3_AUDIO_RANK=0 keeps it on
        # the output rank.
        g = _stage_group()
        self.world_rank = g.rank_in_group if g is not None else 0
        self.world_size = g.world_size if g is not None else 1
        last = (
            os.environ.get("MINIMAX_H3_AUDIO_RANK", "last") != "0"
            and self.vae_tile_parallel
            and self.world_size > 1
        )
        self.audio_rank = self.world_size - 1 if last else 0
        self.is_audio_rank = self.world_rank == self.audio_rank
        # Audio decode placement: "device" (default with tile-parallel device VAE) = BigVGAN on the NeuronCores in
        # fixed-length time windows (vae.audio_windows), one window per rank on the stage's last ranks after their
        # video tiles, sent to the output rank: 0.10 s per window with the time-folded activations, so audio leaves
        # the critical path (768p on 64 cores: 6.98 s warm against 7.88 s with the host decode). "host" (default
        # otherwise) = one host decode on `audio_rank`, overlapped with the video decode.
        default_audio = "device" if self.vae_tile_parallel else "host"
        self.audio_mode = str(
            mc.get("audio_vae", os.environ.get("MINIMAX_H3_AUDIO", default_audio))
        ).lower()
        # "host_windows" = the same windows decoded on the HOST of the last ranks, in parallel with each other and with
        # the video decode (one window ~0.4 s on 16 threads at 768p, against 1.9 s for the whole host decode).
        if self.audio_mode not in ("device", "host", "host_windows"):
            raise ValueError(
                f"audio_vae must be 'device', 'host' or 'host_windows', got {self.audio_mode!r}"
            )
        self._audio_dec = None
        # Video output: "uint8" (the clip's own 8-bit pixels, ~4x less to move out of the worker; what vLLM-Omni's
        # video encoder takes as integer frames) or "float" ([0, 1]).
        self.video_output = str(
            mc.get("video_output", os.environ.get("MINIMAX_H3_VIDEO_OUTPUT", "uint8"))
        ).lower()
        self.text_encoder = None
        # Text encoder placement: "device" = Qwen3-VL's first `text_layer` layers TP-sharded over the stage's TP
        # group and compiled per prompt-length bucket (every CP replica runs its own copy); "host" = the
        # transformers model on the output rank's CPU, broadcast to the other ranks (loaded on first use).
        self.text_mode = str(
            mc.get("text_encoder", os.environ.get("MINIMAX_H3_TEXT_ENCODER", "device"))
        ).lower()
        if self.text_mode not in ("device", "host"):
            raise ValueError(f"text_encoder must be 'device' or 'host', got {self.text_mode!r}")
        self.text_device = (
            None  # built on the first prompt that needs it (precomputed embeddings never do)
        )
        self._compile_backend: str | None = None
        # prompt -> embeddings, the last `cache` distinct prompts (the same on every rank: they see the same requests)
        self._embed_cache: dict[str, torch.Tensor] = {}
        self._embed_cache_size = int(mc.get("prompt_cache_size", 16))
        self.vae = None
        self.audio_vae = None
        self._device = torch.device("cpu")
        self._dit_fn = self.transformer
        self.stats: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------------------------------------------------
    def load_weights(self, weights=None):
        t0 = time.time()
        self.transformer.load_weights(os.path.join(self.model_path, "transformer"), "cpu")
        t1 = time.time()
        if self.has_vae:
            from ._vendor.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
            from ._vendor.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

            # Video: proj_in/proj_out/the block stack are NOT in _keep_in_fp32_modules, so loading at self.dtype
            # gives the released mixed precision (bf16 compute, fp32 norms/scales/encoder) -- matching what the
            # device wrapper feeds it. Audio: every cast-sensitive module is pinned, but BigVGAN's own convs are
            # not, and diffusers' own note says bf16 there measured ~20 dB quieter -- always load and run it fp32.
            self.vae = AutoencoderKLMiniMaxH3.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=self.dtype
            ).eval()
            if self.is_audio_rank or (self.audio_mode != "host" and self.has_vae):  # audio decoders
                self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                    self.model_path, subfolder="audio_vae", torch_dtype=torch.float32
                ).eval()
            self._raw_vae, self._raw_audio_vae = self.vae, self.audio_vae
        logger.info(
            "MiniMax-H3: DiT shard loaded in %.1fs, host components in %.1fs",
            t1 - t0,
            time.time() - t1,
        )
        return None  # components load themselves; nothing for the generic loader to check

    def post_load_weights(self) -> None:
        pass

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            self.transformer.to(self._device)  # host components stay on the CPU
            if self.has_vae and self.vae_mode == "device" and self._device.type != "cpu":
                from .vae import NeuronMiniMaxH3VideoVAE

                self.vae = NeuronMiniMaxH3VideoVAE(self._raw_vae, self._device, self.dtype)
                # The audio BigVGAN is wrapped on first use (_decode_windowed: fp32, weight norm folded, windowed);
                # with audio_vae: host it stays a plain host decode.
                self.audio_vae = self._raw_audio_vae
        return self

    def compile(self, *args, backend: str | None = None, options: dict | None = None, **kwargs):
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        opts = {
            **(options or {}),
            "model_name": "minimax_h3_dit",
            "compiler_args": list(TRANSFORMER_COMPILER_ARGS),
        }
        kw = {"fullgraph": kwargs.get("fullgraph", True), "dynamic": False}
        self._dit_fn = torch.compile(self.transformer, backend=backend, options=opts, **kw)
        self._compile_backend = backend  # the device text encoder compiles with it when first built
        return self

    def _prompt_embeds(self, prompt: str) -> torch.Tensor:
        """Embeddings for ``prompt`` on every rank: from the prompt cache, else the device encoder (every rank
        computes them) or the host encoder (output rank, then broadcast)."""
        hit = self._embed_cache.pop(prompt, None)
        if hit is None and self.text_mode == "device":
            hit = self._device_text_encoder().encode(prompt)
        elif hit is None:
            hit = self._broadcast(self._encode(prompt) if self.is_output_rank else None)
        self._embed_cache[prompt] = hit  # most recent last
        while len(self._embed_cache) > self._embed_cache_size:
            self._embed_cache.pop(next(iter(self._embed_cache)))
        return hit

    def _device_text_encoder(self):
        """The TP-sharded Qwen3-VL encoder on this rank: loaded, moved to the NeuronCore and wrapped for compile on
        first use (~6.1 GB of HBM per core at TP=8; every rank of the stage builds its shard)."""
        if self.text_device is None:
            from .text_encoder_device import DeviceTextEncoder

            t = self.transformer
            self.text_device = DeviceTextEncoder(
                self.model_path, self.text_layer, t.tp_size, t.tp_rank, t.tp_group, self.dtype
            ).to(self._device)
            if self._compile_backend is not None:
                self.text_device.compile(self._compile_backend, TRANSFORMER_COMPILER_ARGS)
        return self.text_device

    def _encode(self, prompt: str) -> torch.Tensor:
        """Host text encoder, loaded on first use (output-owner rank only): ~52 GB of host RAM for the 32B
        conditioner, which a request with precomputed embeddings never needs."""
        if self.text_encoder is None:
            from .text_encoder import H3TextEncoder

            self.text_encoder = H3TextEncoder(self.model_path, self.text_layer, self.dtype)
        # Lite pins the worker to one torch thread: ~7 s per prompt for the 32B encoder; 32 threads ~0.7 s
        n = int(os.environ.get("MINIMAX_H3_TEXT_THREADS", "32"))
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            return self.text_encoder.encode(prompt).cpu()
        finally:
            torch.set_num_threads(prev)

    # -- request parsing --------------------------------------------------------------------------------------
    @staticmethod
    def _prompt_text(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            return prompt.get("prompt") or ""
        raise ValueError(f"MiniMax-H3 takes one text prompt per request, got {type(prompt)}")

    def _broadcast(self, obj):
        g = _stage_group()
        if g is None or g.world_size == 1:
            return obj
        return g.broadcast_object(obj, src=0)

    def _rank_agreement(self, video_rows: torch.Tensor, audio_rows: torch.Tensor) -> dict:
        """Every stage rank steps its own copy of the latents (the DiT output is all-gathered to every rank) and
        decodes its share of the video tiles from it, so a rank that drifts corrupts its tiles. Digests each rank's
        final video / audio rows and compares them across the stage with the shared all-rank agreement check
        (``MINIMAX_H3_RANK_CHECK=1``; parity gates). ``ok`` means every rank holds bit-identical latents."""
        from vllm_omni_neuron.testing import check_rank_agreement

        g = _stage_group()
        group = None if g is None or g.world_size == 1 else g.cpu_group
        if group is None:  # one rank (or no distributed run): a 1-rank report
            from vllm_omni_neuron.testing.rank_agreement import compare_digests, outputs_digest

            report = compare_digests(
                [outputs_digest({"video_rows": video_rows, "audio_rows": audio_rows})]
            )
        else:
            report = check_rank_agreement(
                {"video_rows": video_rows, "audio_rows": audio_rows}, group=group
            )
        return report.to_json()

    # -- one request ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        return_latents: bool = False,
        prompt_embeds: torch.Tensor | None = None,
        sigmas: tuple[list[float], list[float]] | None = None,
    ) -> dict:
        t0 = time.time()
        if prompt_embeds is not None:
            embeds = self._broadcast(prompt_embeds.cpu())
        else:
            embeds = self._prompt_embeds(prompt)
        t_text = time.time() - t0

        layout = build_layout(int(embeds.shape[1]), height, width, num_frames)
        if generator is None:
            generator = torch.Generator().manual_seed(0 if seed is None else int(seed))
        video_rows, audio_rows = draw_noise(layout, generator)
        positions = (
            step_positions(self.model_path, num_inference_steps)
            if self.schedule == "contract" and sigmas is None
            else None
        )
        sched_v, sched_a = load_schedulers(self.model_path, num_inference_steps, positions, sigmas)
        cos, sin = layout.rotary(self.cfg.rope_freq_dim, self.cfg.rope_theta)
        p = self.cfg.patch_size
        grid = (
            layout.num_latent_frames // p[0],
            layout.latent_height // p[1],
            layout.latent_width // p[2],
        )
        self.transformer.set_layout(
            layout.num_text_tokens, layout.num_audio_rows, layout.num_video_rows, grid
        )

        dev, dt = self._device, self.dtype
        text_d = embeds.to(dt).contiguous().to(dev)
        cos, sin, shard = self.transformer.shard_inputs(
            cos, sin
        )  # this CP rank's rows (identity at cp=1)
        cos_d, sin_d = cos.contiguous().to(dev), sin.contiguous().to(dev)
        shard_d = None if shard is None else tuple(t.contiguous().to(dev) for t in shard)
        fwd_s = []
        vel_dump = []  # per-step velocities (MINIMAX_H3_DUMP_LATENTS): the single-step parity tier
        # Step replay (MINIMAX_H3_CAPTURE=<path>, MINIMAX_H3_CAPTURE_STEPS=0,25,49): the exact host inputs of the
        # chosen forwards and the device's velocities, for a CPU single-forward comparison at full size.
        capture_path = os.environ.get("MINIMAX_H3_CAPTURE") if self.is_output_rank else None
        capture_steps = {
            int(s) for s in os.environ.get("MINIMAX_H3_CAPTURE_STEPS", "0").split(",") if s.strip()
        }
        captured = []
        for i in range(len(sched_v.timesteps)):
            tv, ta = float(sched_v.timesteps[i]), float(sched_a.timesteps[i])
            ts = torch.tensor([tv, ta], dtype=torch.float32)
            if capture_path and i in capture_steps:
                captured.append(
                    {
                        "step": i,
                        "t_video": tv,
                        "t_audio": ta,
                        "video_rows": video_rows.clone(),
                        "audio_rows": audio_rows.clone(),
                    }
                )
            tables = None
            if self.host_adaln is not None:
                tables = self.host_adaln.tables(self.transformer.temb(ts)).contiguous().to(dev)
            t1 = time.time()
            vel_v, vel_a = self._dit_fn(
                text_d,
                audio_rows[None].contiguous().to(dev),
                video_rows[None].contiguous().to(dev),
                ts.to(dev),
                cos_d,
                sin_d,
                tables,
                shard_d,
            )
            vel_v, vel_a = vel_v.to("cpu").float()[0], vel_a.to("cpu").float()[0]
            fwd_s.append(time.time() - t1)
            if captured and captured[-1]["step"] == i:
                captured[-1].update(vel_video=vel_v.clone(), vel_audio=vel_a.clone())
            if os.environ.get("MINIMAX_H3_DUMP_LATENTS") and self.is_output_rank:
                vel_dump.append(
                    {
                        "t_video": tv,
                        "t_audio": ta,
                        "vel_video": vel_v.clone(),
                        "vel_audio": vel_a.clone(),
                    }
                )
            video_rows = sched_v.step(vel_v, sched_v.timesteps[i], video_rows, return_dict=False)[
                0
            ].float()
            audio_rows = sched_a.step(vel_a, sched_a.timesteps[i], audio_rows, return_dict=False)[
                0
            ].float()
        self.stats = {
            "text_s": t_text,
            "text_mode": "embeds" if prompt_embeds is not None else self.text_mode,
            "forward_s": fwd_s,
            "num_tokens": layout.sequence_length,
            "num_text_tokens": layout.num_text_tokens,
        }
        if os.environ.get("MINIMAX_H3_RANK_CHECK", "0") == "1":
            self.stats["rank_agreement"] = self._rank_agreement(video_rows, audio_rows)
            if self.is_output_rank:
                logger.info("MiniMax-H3 rank agreement: %s", self.stats["rank_agreement"])
        out = {"layout": layout, "video_rows": video_rows, "audio_rows": audio_rows}
        if captured:
            torch.save(
                {
                    "prompt": prompt,
                    "geom": (height, width, num_frames),
                    "seed": seed,
                    "prompt_embeds": embeds,
                    "sigmas_video": sched_v.sigmas.tolist(),
                    "sigmas_audio": sched_a.sigmas.tolist(),
                    "steps": captured,
                    "final_video_rows": video_rows,
                    "final_audio_rows": audio_rows,
                },
                capture_path,
            )
            os.environ.pop("MINIMAX_H3_CAPTURE", None)  # one request per capture
        dump = os.environ.get("MINIMAX_H3_DUMP_LATENTS")
        if dump and "%d" in dump:  # one file per request
            self._requests = getattr(self, "_requests", 0) + 1
            dump = dump % self._requests
        if (
            dump and self.is_output_rank
        ):  # debugging / parity: final latent rows + the request that made them
            # "steps" = the per-step velocities (gate_compare's single-step tier); the grid size has its own key
            torch.save(
                {
                    "prompt": prompt,
                    "geom": (height, width, num_frames),
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "prompt_embeds": embeds,
                    "video_rows": video_rows,
                    "audio_rows": audio_rows,
                    "layout": layout,
                    "steps": vel_dump,
                    "schedule": self.schedule,
                },
                dump,
            )
        if return_latents:
            return out
        if self.audio_mode != "host" and self.audio_vae is not None:
            return self._decode_windowed(out, video_rows, audio_rows, layout)
        if not self.is_output_rank:
            if (
                self.is_audio_rank
            ):  # decode the audio on a side thread while joining the video decode, then send it
                from concurrent.futures import ThreadPoolExecutor

                n = int(os.environ.get("MINIMAX_H3_AUDIO_THREADS", "32"))
                prev = torch.get_num_threads()
                torch.set_num_threads(n)
                try:
                    with ThreadPoolExecutor(1) as ex:
                        fut = ex.submit(self._timed, self.decode_audio, audio_rows, layout, False)
                        if self.vae_tile_parallel and self.vae is not None:
                            self.decode_video(video_rows, layout, threads=False)
                        audio, audio_s = fut.result()
                finally:
                    torch.set_num_threads(prev)
                self._send_audio(audio, audio_s)
            elif (
                self.vae_tile_parallel and self.vae is not None
            ):  # join rank 0's tile-parallel video decode
                self.decode_video(video_rows, layout)
            return out
        if self.audio_rank != 0:
            t2 = time.time()
            video = self.decode_video(video_rows, layout)
            t3 = time.time()
            audio, audio_s = self._recv_audio()
            self.stats["audio_rank"] = self.audio_rank
            out["video"], out["audio"] = video, audio
            self.stats.update(
                video_decode_s=t3 - t2, audio_decode_s=audio_s, decode_s=time.time() - t2
            )
            self._vae_stats()
            logger.info("MiniMax-H3 request: %s", self.stats)
            return out
        # The audio decode (host BigVGAN) overlaps the video decode: it runs on a side thread while this rank
        # enumerates, decodes its share of the video tiles and waits for the gather (torch releases the GIL).
        from concurrent.futures import ThreadPoolExecutor

        # torch's thread count is process-wide, so it is set once here for both (the two decodes do not touch it).
        n = int(os.environ.get("MINIMAX_H3_AUDIO_THREADS", "32"))
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        t2 = time.time()
        try:
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(self._timed, self.decode_audio, audio_rows, layout, False)
                video = self.decode_video(video_rows, layout, threads=False)
                t3 = time.time()
                audio, audio_s = fut.result()
        finally:
            torch.set_num_threads(prev)
        self.stats["audio_threads"] = n
        out["video"], out["audio"] = video, audio
        self.stats["video_decode_s"] = t3 - t2
        self.stats["audio_decode_s"] = audio_s
        self.stats["decode_s"] = time.time() - t2  # wall time of both, overlapped
        self._vae_stats()
        logger.info("MiniMax-H3 request: %s", self.stats)
        return out

    def _audio_owner(self, i: int) -> int:
        """Stage rank that decodes audio window ``i``: the last ranks, counting down, skipping rank 0."""
        if self.world_size == 1 or not self.vae_tile_parallel:
            return 0
        return self.world_size - 1 - (i % (self.world_size - 1))

    def _decode_windowed(self, out: dict, video_rows, audio_rows, layout) -> dict:
        """Video decode (tile-parallel), then each owner rank decodes its audio windows on its NeuronCore and sends
        them to the output rank, which stitches them. Every rank holds the final audio latents (the schedulers run
        on all ranks), so no latent broadcast is needed."""
        import torch.distributed as dist

        from .vae import NeuronMiniMaxH3AudioVAE, audio_windows

        t2 = time.time()
        if self._audio_dec is None:
            on_dev = self.vae_mode == "device" and self.audio_mode == "device"
            self._audio_dec = NeuronMiniMaxH3AudioVAE(
                self.audio_vae, self._device if on_dev else torch.device("cpu")
            )
        cfg = self.audio_vae.config
        z = unpack_audio(audio_rows, layout)
        z = z * torch.tensor(cfg.latents_std).view(1, -1, 1) + torch.tensor(cfg.latents_mean).view(
            1, -1, 1
        )
        chunk = int(os.environ.get("MINIMAX_H3_AUDIO_CHUNK", "26"))
        halo = int(os.environ.get("MINIMAX_H3_AUDIO_HALO", "16"))
        wins = audio_windows(z.shape[-1], chunk, halo)
        length = min(z.shape[-1], chunk + 2 * halo)
        pieces, secs = {}, []
        mine = [i for i in range(len(wins)) if self._audio_owner(i) == self.world_rank]

        def run_windows():
            for i in mine:
                ta = time.time()
                pieces[i] = self._audio_dec.decode_window(z, wins[i], length)
                secs.append(time.time() - ta)

        decode_v = self.is_output_rank or (self.vae_tile_parallel and self.vae is not None)
        video = None
        if (
            self.audio_mode == "host_windows" and mine
        ):  # host windows on a side thread, overlapping the video
            from concurrent.futures import ThreadPoolExecutor

            n = int(os.environ.get("MINIMAX_H3_AUDIO_WINDOW_THREADS", "16"))
            prev = torch.get_num_threads()
            torch.set_num_threads(n)
            try:
                with ThreadPoolExecutor(1) as ex:
                    fut = ex.submit(run_windows)
                    if decode_v:
                        video = self.decode_video(video_rows, layout, threads=False)
                    fut.result()
            finally:
                torch.set_num_threads(prev)
            t3 = time.time()
        else:
            if decode_v:
                video = self.decode_video(video_rows, layout)
            t3 = time.time()
            run_windows()
        g = _stage_group()
        if not self.is_output_rank:
            for i, w in pieces.items():
                dist.send(
                    torch.tensor([i, w.shape[-1], secs[0]], dtype=torch.float64),
                    dst=g.ranks[0],
                    group=g.cpu_group,
                )
                dist.send(w.contiguous(), dst=g.ranks[0], group=g.cpu_group)
            return out
        t4 = time.time()
        win_s = list(secs)
        for i in range(len(wins)):
            if i in pieces:
                continue
            head = torch.empty(3, dtype=torch.float64)
            dist.recv(head, src=g.ranks[self._audio_owner(i)], group=g.cpu_group)
            buf = torch.empty((z.shape[0], 1, int(head[1])), dtype=torch.float32)
            dist.recv(buf, src=g.ranks[self._audio_owner(i)], group=g.cpu_group)
            pieces[int(head[0])] = buf
            win_s.append(float(head[2]))
        audio = torch.cat([pieces[i] for i in range(len(wins))], dim=-1).permute(1, 0, 2)
        out["video"], out["audio"] = video, audio
        self.stats.update(
            video_decode_s=t3 - t2,
            audio_decode_s=time.time() - t3,
            decode_s=time.time() - t2,
            audio_windows=len(wins),
            audio_window_s=max(win_s) if win_s else 0.0,
            audio_wait_s=time.time() - t4,
        )
        self._vae_stats()
        logger.info("MiniMax-H3 request: %s", self.stats)
        return out

    def _vae_stats(self) -> None:
        if hasattr(
            self.vae, "tile_calls"
        ):  # VAE profiling: how many per-tile decoder graphs, and distinct shapes
            self.stats["vae_tile_calls"] = self.vae.tile_calls
            self.stats["vae_tile_shapes"] = {
                str(k): v for k, v in getattr(self.vae, "tile_shapes", {}).items()
            }
            self.stats["vae_timing"] = getattr(self.vae, "timing", None)

    def _send_audio(self, audio: torch.Tensor, seconds: float) -> None:
        """Audio rank -> output rank over the host (gloo) group: a header (shape, decode time), then the waveform."""
        import torch.distributed as dist

        g = _stage_group()
        head = torch.tensor([*audio.shape, seconds], dtype=torch.float64)
        dist.send(head, dst=g.ranks[0], group=g.cpu_group)
        dist.send(audio.float().contiguous(), dst=g.ranks[0], group=g.cpu_group)

    def _recv_audio(self) -> tuple[torch.Tensor, float]:
        import torch.distributed as dist

        g = _stage_group()
        head = torch.empty(4, dtype=torch.float64)
        dist.recv(head, src=g.ranks[self.audio_rank], group=g.cpu_group)
        audio = torch.empty(tuple(int(x) for x in head[:3].tolist()), dtype=torch.float32)
        dist.recv(audio, src=g.ranks[self.audio_rank], group=g.cpu_group)
        return audio, float(head[3])

    @staticmethod
    def _timed(fn, *args):
        t0 = time.time()
        with torch.no_grad():
            out = fn(*args)
        return out, time.time() - t0

    @torch.no_grad()
    def decode_video(
        self, video_rows: torch.Tensor, layout, threads: bool = True
    ) -> torch.Tensor | None:
        """-> ``(1, 3, F, H, W)`` on rank 0 (uint8 pixels, or float in ``[0, 1]`` with ``video_output: float``);
        ``None`` on the other ranks of a tile-parallel decode. ``threads``: widen Lite's single torch thread for the
        host work on the full clip after the tile loop, then restore; False when the caller already did."""
        if not threads:
            return self._decode_video(video_rows, layout)
        n = (
            int(os.environ.get("MINIMAX_H3_VAE_THREADS", "32"))
            if self.is_output_rank
            else torch.get_num_threads()
        )
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            return self._decode_video(video_rows, layout)
        finally:
            torch.set_num_threads(prev)

    def _decode_video(self, video_rows: torch.Tensor, layout) -> torch.Tensor | None:
        latents = unpatchify_video(video_rows, layout)
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        z = (latents * std + mean).float()
        pm = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
        ps = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
        if self.vae_tile_parallel and hasattr(self.vae, "_decode_tile_parallel"):
            # each rank turns its own tiles into 8-bit pixels (MINIMAX_H3_VAE_UINT8=0: ship 16-bit decoder output)
            to_unit = (
                (lambda v: v * ps + pm)
                if os.environ.get("MINIMAX_H3_VAE_UINT8", "1") != "0"
                else None
            )
            video = self.vae.decode(z, return_dict=False, tp_group=_stage_group(), to_unit=to_unit)[
                0
            ]
        else:
            video = self.vae.decode(z, return_dict=False)[0]
        if video is None:
            return None
        if getattr(self.vae, "output_uint8", False):  # already the clip's 8-bit pixels
            return video if self.video_output == "uint8" else video.float().div_(255.0)
        if getattr(self.vae, "applied_unit", False):
            video = video.clamp_(0, 1)
        else:
            video = (video.float() * ps + pm).clamp(0, 1)
        return video.mul(255.0).round_().to(torch.uint8) if self.video_output == "uint8" else video

    @torch.no_grad()
    def decode_audio(self, audio_rows: torch.Tensor, layout, threads: bool = True) -> torch.Tensor:
        """-> stereo ``(1, 2, samples)`` at 32 kHz. Host stage inside the worker: Lite pins the worker to 1 torch
        thread, which makes this BigVGAN decode several times slower than standalone, so ``threads`` widens it for
        the decode only (False when the caller already set the process-wide count)."""
        a = unpack_audio(audio_rows, layout)
        am = torch.tensor(self.audio_vae.config.latents_mean).view(1, -1, 1)
        astd = torch.tensor(self.audio_vae.config.latents_std).view(1, -1, 1)
        n = (
            int(os.environ.get("MINIMAX_H3_AUDIO_THREADS", "32"))
            if threads
            else torch.get_num_threads()
        )
        prev = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            audio = self.audio_vae.decode(a * astd + am, return_dict=False)[0]
        finally:
            torch.set_num_threads(prev)
        return audio.float().permute(1, 0, 2)

    def decode(
        self, video_rows: torch.Tensor, audio_rows: torch.Tensor, layout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> video ``(1, 3, F, H, W)`` in ``[0, 1]`` and stereo audio ``(1, 2, samples)`` at 32 kHz."""
        return self.decode_video(video_rows, layout), self.decode_audio(audio_rows, layout)

    def _base_request_sigmas(self, steps: int, extra: dict) -> tuple[list[float], list[float]]:
        """(video, audio) sigma boundaries of a base MiniMax-H3 request: ``steps`` denoiser evaluations, shifts
        from ``extra_args.flow_shift`` / ``extra_args.audio_flow_shift`` or the checkpoint's schedulers (12 / 3).
        Tasks other than t2va (FL2VA, Ref2VA) are rejected: not supported yet on Neuron."""
        task = str(extra.get("task", "t2va")).lower()
        if task not in BASE_TASKS:
            raise ValueError(
                f"MiniMax-H3 task {task!r} is not supported on Neuron yet (supported: {', '.join(BASE_TASKS)})"
            )
        if self._base_shifts is None:
            shifts = []
            for sub in ("scheduler", "audio_scheduler"):
                with open(os.path.join(self.model_path, sub, "scheduler_config.json")) as f:
                    shifts.append(float(json.load(f)["shift"]))
            self._base_shifts = tuple(shifts)
        video_shift = float(extra.get("flow_shift", self._base_shifts[0]))
        audio_shift = float(extra.get("audio_flow_shift", self._base_shifts[1]))
        return base_sigmas(steps, video_shift), base_sigmas(steps, audio_shift)

    def forward(self, req, **kwargs):
        from vllm_omni.diffusion.data import DiffusionOutput

        sp = req.sampling_params
        prompt = self._prompt_text(req.prompts[0])
        if (
            prompt == "dummy run"
            and (sp.num_inference_steps or 0) <= 1
            and (sp.num_frames or 1) <= 1
        ):
            # The engine's warm-up request (512x512, one frame, one step) is not a geometry this model serves;
            # compiling it would only add a graph. Real geometries compile on their first request.
            return DiffusionOutput(output=(torch.zeros(1, 3, 1, 32, 32), torch.zeros(1, 2, 800)))
        height = sp.height or 384
        width = sp.width or 640
        num_frames = sp.num_frames if sp.num_frames and sp.num_frames > 1 else 121
        steps = sp.num_inference_steps or self.default_steps
        if (
            steps < 2
        ):  # the engine's warm-up dummy run asks for 1 step; the schedule needs >= 2 grid points
            steps = 2
        gen = sp.generator[0] if isinstance(sp.generator, list) else sp.generator
        if gen is not None and gen.device.type != "cpu":
            gen = None
        extra = getattr(sp, "extra_args", None) or {}
        sigmas = None
        if self.is_base:
            sigmas = self._base_request_sigmas(steps, extra)
            steps = steps + 1  # generate() counts sigma grid points
        embeds = None
        if extra.get(
            "prompt_embeds_file"
        ):  # precomputed conditioning (parity gates, embedding caches)
            embeds = torch.load(
                extra["prompt_embeds_file"], map_location="cpu", weights_only=False
            )["prompt_embeds"]
        t0 = time.time()
        handoff = self._video_handoff()
        if handoff:
            self.vae.keep_clip_shm = True
        try:
            out = self.generate(
                prompt,
                height,
                width,
                num_frames,
                steps,
                seed=sp.seed,
                generator=gen,
                prompt_embeds=embeds,
                sigmas=sigmas,
            )
        finally:
            if handoff:
                self.vae.keep_clip_shm = False
        self.stats["pipeline_s"] = (
            time.time() - t0
        )  # the whole request inside the worker (output rank)
        if not self.is_output_rank:
            return DiffusionOutput(output=None)
        video = out["video"]
        name = getattr(video, "_h3_shm_name", None)
        if name is not None:
            video = self._shm_handle(video, name)
        return DiffusionOutput(
            output=(video, out["audio"]), custom_output={"minimax_h3_stats": self.stats}
        )

    def _video_handoff(self) -> bool:
        """Whether the composed uint8 clip (already a file in ``/dev/shm``) goes to the engine as vLLM-Omni's own
        shared-memory tensor handle, instead of being copied into a fresh segment by the worker's output packing
        (0.25 s for the 384 MB 768p clip; the engine's one copy out stays). ``MINIMAX_H3_VIDEO_HANDOFF=0`` off."""
        if os.environ.get("MINIMAX_H3_VIDEO_HANDOFF", "1") == "0" or not self.is_output_rank:
            return False
        if self.video_output != "uint8" or not getattr(self.vae, "output_uint8", False):
            return False
        try:
            from vllm_omni.diffusion.ipc import (
                _tensor_from_shm,  # noqa: F401  (the handle format's reader)
            )
        except ImportError:
            return False
        return hasattr(self.vae, "keep_clip_shm")

    @staticmethod
    def _shm_handle(video: torch.Tensor, name: str) -> dict:
        """vLLM-Omni's ``__tensor_shm__`` handle for the clip file ``/dev/shm/<name>``; the engine maps it, copies the
        clip out and unlinks it."""
        return {
            "__tensor_shm__": True,
            "name": name,
            "shape": list(video.shape),
            "torch_dtype": str(video.dtype),
            "numpy_dtype": "uint8",
            "nbytes": video.numel(),
        }
