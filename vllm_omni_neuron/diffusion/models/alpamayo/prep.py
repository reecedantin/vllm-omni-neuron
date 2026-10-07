# SPDX-License-Identifier: Apache-2.0
"""Host-side preparation for Alpamayo 1.5: mRoPE position ids (via the real
``Qwen3VLModel.get_rope_index``), the causal/padding mask for prefill, and the expert's attention-mask/position-
id construction for stage 2 (mirrors upstream's ``_build_expert_pos_ids_and_attn_mask``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

MASK_VALUE = -30000.0  # finite: -inf poisons fully-masked rows and bf16 casts


def bias_from_keep(keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, 0.0, MASK_VALUE).to(torch.float32)


@dataclass
class BackboneInputs:
    pixels: torch.Tensor
    pos_index: torch.Tensor
    pos_weight: torch.Tensor
    vis_cos: torch.Tensor
    vis_sin: torch.Tensor
    n_images: int
    input_ids: torch.Tensor  # [1, bucket]
    image_index: torch.Tensor
    image_keep: torch.Tensor
    txt_cos: torch.Tensor  # [1, bucket, head_dim]
    txt_sin: torch.Tensor
    real_len: int
    rope_deltas: torch.Tensor  # [1, 1]
    last_position: torch.Tensor  # [3, 1] -- the (t,h,w) position of the last REAL token


class BackbonePrep:
    """Host math for the Alpamayo VLM backbone, delegated to ``transformers``' own Qwen3-VL
    helpers, also returning the ``rope_deltas`` that continue the mRoPE positions through the
    autoregressive decode steps and into the expert.
    """

    def __init__(self, hf_config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )

        self.cfg = hf_config
        vc = hf_config.vision_config
        self.merge = int(vc.spatial_merge_size)
        self.side = int(vc.num_position_embeddings**0.5)
        self.image_token_id = int(hf_config.image_token_id)
        self.vis_rope = Qwen3VLVisionRotaryEmbedding(vc)
        self.txt_rope = Qwen3VLTextRotaryEmbedding(hf_config.text_config)
        shim = SimpleNamespace(config=hf_config)
        shim.get_vision_position_ids = lambda *a, **k: Qwen3VLModel.get_vision_position_ids(
            shim, *a, **k
        )
        self._rope_index = lambda *a, **k: Qwen3VLModel.get_rope_index(shim, *a, **k)

    @torch.no_grad()
    def vision_tables(self, grid_thw: torch.Tensor):
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        idx, w = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self.merge,
        )
        pos = get_vision_position_ids(grid_thw, self.merge)
        cos, sin = self.vis_rope(torch.zeros(1), pos)
        return idx.long(), w.float(), cos.float(), sin.float()

    @torch.no_grad()
    def __call__(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, bucket: int
    ) -> BackboneInputs:
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "Alpamayo on Neuron serves one observation per call (batch 1)"
            )
        grid = image_grid_thw.long()
        n_images = int(grid[:, 0].sum())
        pidx, pw, vcos, vsin = self.vision_tables(grid)

        am = attention_mask.long()
        real = int(am.sum())
        if real > bucket:
            raise ValueError(f"prompt has {real} tokens, bucket is {bucket}")
        keep_tok = am[0].bool()
        ids = input_ids[:, keep_tok]
        mm = (ids == self.image_token_id).int()

        pos3, rope_deltas = self._rope_index(
            ids, mm, image_grid_thw=grid, attention_mask=torch.ones_like(ids)
        )
        pos = torch.zeros(3, 1, bucket, dtype=torch.long)
        pos[:, :, :real] = pos3
        tcos, tsin = self.txt_rope(torch.zeros(1, dtype=torch.float32), pos)

        pid = torch.zeros(1, bucket, dtype=torch.long)
        pid[:, :real] = ids
        valid = torch.zeros(1, bucket, dtype=torch.bool)
        valid[:, :real] = True
        img = (pid == self.image_token_id) & valid
        index = (img.long().cumsum(-1) - 1).clamp(min=0)

        last_position = pos3[:, 0, real - 1]  # [3] -- (t,h,w) of the last real prompt token
        return BackboneInputs(
            pixels=pixel_values,
            pos_index=pidx,
            pos_weight=pw,
            vis_cos=vcos,
            vis_sin=vsin,
            n_images=n_images,
            input_ids=pid,
            image_index=index,
            image_keep=img[..., None],
            txt_cos=tcos.float(),
            txt_sin=tsin.float(),
            real_len=real,
            rope_deltas=rope_deltas,
            last_position=last_position[:, None],
        )


@torch.no_grad()
def rope_cos_sin(hf_config, position_ids: torch.Tensor, dtype, device):
    """Text mRoPE tables for ``position_ids`` ``[3, n]`` -> cos/sin ``[1, n, head_dim]``."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    rope = Qwen3VLTextRotaryEmbedding(hf_config.text_config)
    cos, sin = rope(torch.zeros(1, dtype=torch.float32), position_ids[:, None, :])
    return cos.to(dtype).to(device), sin.to(dtype).to(device)


@torch.no_grad()
def prefill_bias(real_len: int, bucket: int) -> torch.Tensor:
    """``[1, 1, bucket, bucket]`` causal mask for a RIGHT-padded prompt: real rows see real columns
    ``<=`` themselves; padding columns are masked for every row (a padding row still sees itself, so
    no row is fully masked)."""
    i = torch.arange(bucket)
    keep = (i[None, :] <= i[:, None]) & ((i[None, :] < real_len) | (i[None, :] == i[:, None]))
    return bias_from_keep(keep)[None, None]


@torch.no_grad()
def decode_masks(pos: int, max_len: int, device):
    """For the decode token at cache slot ``pos``: the one-hot write mask ``[1, 1, max_len, 1]``
    (bool) and the attention bias ``[1, 1, 1, max_len]`` keeping slots ``0..pos``. Right-padding K/V
    the prefill left in slots ``>= real_len`` is either overwritten by then or still masked."""
    i = torch.arange(max_len)
    write_mask = (i == pos)[None, None, :, None]
    return write_mask.to(device), bias_from_keep(i <= pos)[None, None, None].to(device)


@torch.no_grad()
def expert_inputs(
    hf_config,
    offset: int,
    cache_valid: int,
    rope_delta: int,
    n_tokens: int,
    max_len: int,
    dtype,
    device,
):
    """mRoPE tables + attention bias for the expert's ``n_tokens`` action tokens, mirroring upstream's
    ``_build_expert_pos_ids_and_attn_mask``. Positions are ``offset + j + rope_delta`` on all three
    axes (``offset`` = sequence index right after ``<traj_future_start>``). Upstream's cropped
    ``DynamicCache`` holds ``cache_valid`` tokens and masks ``[offset, cache_valid)``; here the keys are
    the full fixed cache (``max_len``) followed by the ``n_tokens`` expert keys, so the kept columns
    are ``[0, min(offset, cache_valid))`` plus the last ``n_tokens``."""
    pos = (torch.arange(n_tokens) + offset + rope_delta)[None].expand(3, -1)
    cos, sin = rope_cos_sin(hf_config, pos, dtype, device)
    cols = torch.arange(max_len + n_tokens)
    keep = (cols < min(offset, cache_valid)) | (cols >= max_len)
    bias = bias_from_keep(keep)[None, None, None].expand(1, 1, n_tokens, -1).contiguous()
    return cos, sin, bias.to(device)


def _delta_tokenize(
    xyz: torch.Tensor, xyz_min, xyz_max, num_bins: int, pad_origin_at_beginning: bool = True
) -> torch.Tensor:
    """``DeltaTrajectoryTokenizer.encode`` (``predict_yaw=False``): per-step xyz deltas, binned
    uniformly over ``[min, max]``. With ``pad_origin_at_beginning`` (Alpamayo 1.5) the first delta is
    taken from the origin (T waypoints -> T deltas); without it (Alpamayo 2 Super's history, which
    already ends at the origin) only consecutive waypoints are differenced (T -> T - 1)."""
    d = torch.nn.functional.pad(xyz, [0, 0, 1, 0, 0, 0]) if pad_origin_at_beginning else xyz
    d = d[:, 1:] - d[:, :-1]
    lo = torch.tensor(xyz_min, dtype=d.dtype)
    hi = torch.tensor(xyz_max, dtype=d.dtype)
    idx = (((d - lo) / (hi - lo)) * (num_bins - 1)).round().long().clamp(0, num_bins - 1)
    return idx.reshape(idx.shape[0], -1)


@torch.no_grad()
def fuse_history_tokens(
    input_ids: torch.Tensor, ego_history_xyz: torch.Tensor, ego_history_rot: torch.Tensor, cfg
) -> torch.Tensor:
    """Upstream ``TrajectoryFusionMixin.fuse_traj_tokens`` for the default history tokenizer
    (``DeltaTrajectoryTokenizer``): tokenize ``ego_history_xyz`` ``[B, n_traj, T, 3]`` and write the
    ids into the ``<|traj_history|>`` placeholder slots, in order. Ids are offset by
    ``hist_token_start_idx``: Alpamayo 2 Super's ``traj_ids.history_id0``, or for Alpamayo 1.5
    ``traj_token_start_idx`` + the future tokenizer's ``num_bins``."""
    ex = cfg.extra
    hcfg = dict(ex.get("hist_traj_tokenizer_cfg") or {})
    target = hcfg.get("_target_", "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer")
    if not target.endswith("DeltaTrajectoryTokenizer") or hcfg.get("predict_yaw", False):
        raise NotImplementedError(
            f"history tokenizer {target} (predict_yaw={hcfg.get('predict_yaw')})"
        )
    if "hist_token_start_idx" in cfg.head:
        start = int(cfg.head["hist_token_start_idx"])
    else:
        start = int(cfg.head["traj_token_start_idx"])
        if ex.get("traj_tokenizer_cfg"):
            start += int(ex["traj_tokenizer_cfg"].get("num_bins", 0))
    del ego_history_rot  # the delta tokenizer encodes positions only
    xyz = torch.as_tensor(ego_history_xyz, dtype=torch.float32).flatten(0, 1)
    ids = (
        _delta_tokenize(
            xyz,
            tuple(hcfg.get("ego_xyz_min", (-4, -4, -10))),
            tuple(hcfg.get("ego_xyz_max", (4, 4, 10))),
            int(hcfg.get("num_bins", 1000)),
            bool(hcfg.get("pad_origin_at_beginning", True)),
        )
        + start
    )
    ids = ids.reshape(input_ids.shape[0], -1)
    placeholder = int(ex["traj_token_ids"]["history"])
    return input_ids.masked_scatter(input_ids == placeholder, ids.to(input_ids.dtype))
