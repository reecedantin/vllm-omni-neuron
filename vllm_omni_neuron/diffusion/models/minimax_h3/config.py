# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 checkpoint facts the Neuron port keys off: the DiT config and the pipeline constants."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields

# Per-row modality tags (a checkpoint contract: they index the AdaLN table), from diffusers' modular pipeline.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
MODALITY_NUM = 3

FPS = 24
AUDIO_CHANNELS = 2  # stereo, packed channel-major as two blocks of audio rows
AUDIO_LATENT_CHANNELS = 32
VIDEO_LATENT_CHANNELS = 24
VAE_SPATIAL = 16
VAE_FRAMES_PER_CHUNK = 17  # video VAE clip_length
VAE_LATENTS_PER_CHUNK = 5
TEXT_ENCODER_LAYER = 50  # H3 conditions on Qwen3-VL hidden_states[50]


@dataclass
class MiniMaxH3DiTConfig:
    """``transformer/config.json`` of a MiniMax-H3 / FastH3 checkpoint (diffusers ``MiniMaxH3Transformer3DModel``)."""

    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        p = self.patch_size
        return self.in_channels * p[0] * p[1] * p[2]

    @property
    def rotary_dim(self) -> int:
        return 2 * 3 * self.rope_freq_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> MiniMaxH3DiTConfig:
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in cfg.items() if k in names}
        if "patch_size" in kw:
            kw["patch_size"] = tuple(kw["patch_size"])
        return cls(**kw)

    @classmethod
    def from_dir(cls, transformer_dir: str) -> MiniMaxH3DiTConfig:
        with open(os.path.join(transformer_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))


def read_model_index(model_path: str) -> dict:
    """``model_index.json`` (base H3) or ``modular_model_index.json`` (FastH3 releases), whichever exists."""
    for name in ("model_index.json", "modular_model_index.json"):
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    return {}


def text_encoder_layer(model_path: str) -> int:
    """Hidden state of the Qwen3-VL conditioner the DiT reads (50 for every release; a structure test checkpoint
    with fewer layers sets ``text_encoder_layer`` in its model index)."""
    return int(read_model_index(model_path).get("text_encoder_layer", TEXT_ENCODER_LAYER))


def inference_contract(model_path: str) -> dict:
    """FastVideo's ``fastvideo_inference.json`` (FastH3 releases), or ``{}``. Carries the trained step ladder
    (``dmd_denoising_steps``), ``num_inference_steps`` (grid points = forwards + 1), the scheduler shifts and, for
    VSA students, ``attention_backend == "VIDEO_SPARSE_ATTN_H3"`` with ``vsa_sparsity`` / ``vsa_tile_size``."""
    p = os.path.join(model_path, "fastvideo_inference.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)


def vsa_sparsity(model_path: str) -> float | None:
    """The VSA sparsity a checkpoint was trained with, or None for a dense checkpoint."""
    c = inference_contract(model_path)
    if c.get("attention_backend") != "VIDEO_SPARSE_ATTN_H3":
        return None
    tile = int(c.get("vsa_tile_size", 64))
    if tile != 64:
        raise NotImplementedError(
            f"VSA-H3 tile size {tile}: only the 64-token (4,4,4) tile is ported"
        )
    return float(c["vsa_sparsity"])


BASE_NUM_INFERENCE_STEPS = (
    50  # base MiniMax-H3 request default: 50 denoiser evaluations (vLLM-Omni v0.30.0)
)
BASE_TASKS = (
    "t2va",
)  # base tasks ported so far; FL2VA / Ref2VA (transformer_ref) are not supported yet


def is_base_checkpoint(model_path: str) -> bool:
    """True for the base MiniMax-H3 release (50-step, guidance-distilled), False for a FastH3 student. FastH3
    exports carry FastVideo's ``fastvideo_inference.json`` sampling contract; the base release has none."""
    return not os.path.isfile(os.path.join(model_path, "fastvideo_inference.json"))


def base_sigmas(num_inference_steps: int, shift: float) -> list[float]:
    """The base request's sigma boundaries, as vLLM-Omni v0.30.0 builds them (``minimax_h3_time_shift_sigmas``):
    ``num_inference_steps`` counts denoiser evaluations, so the grid is ``linspace(1, 0, N + 1)`` (both endpoints),
    pushed through the exponential shift ``s * x / (1 + (s - 1) * x)`` in fp32. N + 1 values, N forwards."""
    import torch

    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    if shift <= 0:
        raise ValueError(f"shift must be > 0, got {shift}")
    base = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1, dtype=torch.float32)
    shifted = float(shift) * base / (1 + (float(shift) - 1) * base)
    return [float(v) for v in shifted.tolist()]


def step_positions(model_path: str, num_inference_steps: int) -> tuple[float, ...] | None:
    """The trained rung ladder as pre-shift positions in [0, 1] closed with 0.0, when the checkpoint declares one
    and the request asks for exactly that many grid points; else None (the scheduler's own linspace grid)."""
    rungs = inference_contract(model_path).get("dmd_denoising_steps")
    if not rungs or num_inference_steps != len(rungs) + 1:
        return None
    return (*(float(r) / 1000.0 for r in rungs), 0.0)
