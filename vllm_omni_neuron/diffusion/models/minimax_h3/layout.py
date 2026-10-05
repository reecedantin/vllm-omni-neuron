# SPDX-License-Identifier: Apache-2.0
"""Host-side construction of the MiniMax-H3 packed sequence (t2va).

Everything that is a pure function of (prompt length, canvas, frame count) is built on the CPU with diffusers' own
helpers (vendored), so the compiled DiT graph only sees static tensors: RoPE cos/sin, the AdaLN row indices and the
per-step timestep pair. The t2va layout is ``[text (N) | audio (2*A) | video (Nv)]``, three contiguous blocks, so the
device graph concatenates the modality streams instead of scattering rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import config as C
from ._vendor.mp_before_denoise import MiniMaxH3PrepareLayoutStep, patchify_video_latents
from ._vendor.mp_modular_pipeline import (
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from ._vendor.scheduling_minimax_h3 import MiniMaxH3Scheduler
from ._vendor.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

PATCH = (1, 2, 2)


@dataclass
class H3Layout:
    height: int
    width: int
    num_frames: int  # aligned to 17n + 5
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    num_text_tokens: int
    position_ids: torch.Tensor  # (L, 3) float64
    token_tags: torch.Tensor  # (L,) long
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor

    @property
    def num_video_rows(self) -> int:
        return int(self.video_indices.numel())

    @property
    def num_audio_rows(self) -> int:
        return int(self.audio_indices.numel())

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def geometry(self) -> tuple[int, int, int, int]:
        """What a compiled graph depends on: (text rows, audio rows, video rows, video latent width)."""
        return self.num_text_tokens, self.num_audio_rows, self.num_video_rows, self.latent_width

    def assert_contiguous(self) -> None:
        n, na, nv = self.num_text_tokens, self.num_audio_rows, self.num_video_rows
        assert torch.equal(self.text_indices, torch.arange(0, n)), (
            "text rows must lead the sequence"
        )
        assert torch.equal(self.audio_indices, torch.arange(n, n + na)), (
            "audio rows must follow text"
        )
        assert torch.equal(self.video_indices, torch.arange(n + na, n + na + nv)), (
            "video rows must be last"
        )

    def timestep_indices(self) -> torch.Tensor:
        """Row -> index into the 2-entry ``timestep`` tensor ``[t_video, t_audio]``.

        The reference collapses equal timesteps with ``torch.unique``; keeping two entries (identical when equal) is
        numerically the same and keeps every device tensor shape static across steps. Text rows take the video
        timestep, as in the reference.
        """
        idx = torch.zeros(self.sequence_length, dtype=torch.long)
        idx[self.audio_indices] = 1
        return idx

    def adaln_indices(self) -> torch.Tensor:
        return self.timestep_indices() * C.MODALITY_NUM + self.token_tags

    def rotary(self, rope_freq_dim: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``(L, 2*3*rope_freq_dim)`` fp32, computed exactly like the reference module."""
        rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)
        with torch.no_grad():
            cos, sin = rope(self.position_ids)
        return cos.contiguous(), sin.contiguous()


def build_layout(num_text_tokens: int, height: int, width: int, num_frames: int) -> H3Layout:
    """The t2va packed layout for a prompt of ``num_text_tokens`` tokens on a ``height x width`` canvas."""
    if height % 32 or width % 32:
        raise ValueError(f"height/width must be multiples of 32, got {height}x{width}")
    aligned = align_num_frames(num_frames, C.VAE_FRAMES_PER_CHUNK, C.VAE_LATENTS_PER_CHUNK)
    t_lat = video_latent_num_frames(aligned, C.VAE_FRAMES_PER_CHUNK, C.VAE_LATENTS_PER_CHUNK)
    h_lat, w_lat = height // C.VAE_SPATIAL, width // C.VAE_SPATIAL
    a_lat = audio_latent_num_frames(aligned, C.FPS)
    text_tags = torch.full((num_text_tokens,), C.TEXT_TAG, dtype=torch.long)
    position_ids, token_tags, video_indices, audio_indices, text_indices, n_cond_v, n_cond_a = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags,
            t_lat,
            h_lat,
            w_lat,
            a_lat,
            PATCH,
            C.AUDIO_CHANNELS,
            C.AUDIO_TAG,
            C.VIDEO_TAG,
            keyframe_anchors=(),
        )
    )
    assert n_cond_v == 0 and n_cond_a == 0
    layout = H3Layout(
        height,
        width,
        aligned,
        t_lat,
        h_lat,
        w_lat,
        a_lat,
        num_text_tokens,
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
    )
    layout.assert_contiguous()
    return layout


def draw_noise(
    layout: H3Layout, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same draw order as diffusers' MiniMaxH3PrepareLatentsStep: video as a latent tensor (then patchified), then
    the audio rows. Drawn in fp32 on the host."""
    video = torch.randn(
        (
            1,
            C.VIDEO_LATENT_CHANNELS,
            layout.num_latent_frames,
            layout.latent_height,
            layout.latent_width,
        ),
        generator=generator,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latents(video, PATCH)
    audio_rows = torch.randn(
        (layout.num_audio_rows, C.AUDIO_LATENT_CHANNELS), generator=generator, dtype=torch.float32
    )
    return video_rows, audio_rows


def unpatchify_video(rows: torch.Tensor, layout: H3Layout) -> torch.Tensor:
    """(Nv, 96) rows -> (1, 24, T, H, W) latents."""
    pt, ph, pw = PATCH
    c = C.VIDEO_LATENT_CHANNELS
    rows = rows.reshape(
        -1,
        layout.num_latent_frames // pt,
        layout.latent_height // ph,
        layout.latent_width // pw,
        c,
        pt,
        ph,
        pw,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        -1, c, layout.num_latent_frames, layout.latent_height, layout.latent_width
    ).contiguous()


def unpack_audio(rows: torch.Tensor, layout: H3Layout) -> torch.Tensor:
    """(2*A, 32) channel-major rows -> (2, 32, A), the audio VAE's batch-of-mono layout."""
    return (
        rows.reshape(C.AUDIO_CHANNELS, layout.num_audio_latents, -1).permute(0, 2, 1).contiguous()
    )


def load_schedulers(
    model_path: str, num_inference_steps: int, positions: tuple[float, ...] | None = None
) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
    """The video and audio schedulers (shifts from the checkpoint: 12 / 3 for the base and FastH3 4-step, 10 / 3 for
    8-Step-V2). ``num_inference_steps`` counts sigma grid points with the terminal 0 (FastH3 4-step: 5).

    ``positions``: an explicit pre-shift rung ladder ending in 0.0 (a distilled checkpoint's trained ladder,
    e.g. 0.999, 0.874, ..., 0.125, 0.0); each scheduler applies its own shift to it. None = the scheduler's linspace.
    """
    video = MiniMaxH3Scheduler.from_pretrained(model_path, subfolder="scheduler")
    audio = MiniMaxH3Scheduler.from_pretrained(model_path, subfolder="audio_scheduler")
    if positions is not None:
        base = torch.tensor(positions, dtype=torch.float32)
        for sch in (video, audio):
            sft = sch.shift
            sch.set_timesteps(sigmas=sft * base / (1 + (sft - 1) * base))
    else:
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
    if len(video.timesteps) != len(audio.timesteps):
        raise ValueError("video and audio schedules differ in length")
    return video, audio
