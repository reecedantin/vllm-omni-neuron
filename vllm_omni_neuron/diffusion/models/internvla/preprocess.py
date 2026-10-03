# SPDX-License-Identifier: Apache-2.0
"""Host-side tables and inputs for the InternVLA-A1.5 graphs.

Everything here depends only on the request's *shape* (token layout, image grid, step count),
never on activations, so it runs on the CPU once per request and enters the compiled graphs as
plain tensors: mRoPE position ids and cos/sin, vision rotary and position-embedding taps,
additive attention biases, the flow-matching time schedule and its sinusoidal embeddings.
Each function mirrors the upstream code path it replaces (named in its docstring).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import InternVLAConfig, PolicyConfig, TextConfig, VisionConfig, VLMConfig

MASK_VALUE = -2.3819763e38  # upstream OPENPI_ATTENTION_MASK_VALUE


# -- text mRoPE -------------------------------------------------------------------------------


def rope_index(input_ids: torch.Tensor, attention_mask: torch.Tensor, grid_thw: list[list[int]], vlm: VLMConfig):
    """Upstream ``Qwen3_5Model.get_rope_index`` for images (no video). Returns ``[3, B, L]``."""
    merge = vlm.vision.spatial_merge_size
    b, length = input_ids.shape
    pos = torch.zeros(3, b, length, dtype=torch.long)
    img = 0
    for i in range(b):
        ids = input_ids[i][attention_mask[i] == 1].tolist()
        n_img = sum(
            1 for j, t in enumerate(ids[:-1]) if t == vlm.vision_start_token_id and ids[j + 1] == vlm.image_token_id
        )
        chunks, st = [], 0
        for _ in range(n_img):
            ed = ids.index(vlm.image_token_id, st)
            t, h, w = grid_thw[img]
            img += 1
            gt, gh, gw = t, h // merge, w // merge
            st_idx = chunks[-1].max().item() + 1 if chunks else 0
            text_len = ed - st
            chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            ti = torch.arange(gt).view(-1, 1).expand(-1, gh * gw).flatten()
            hi = torch.arange(gh).view(1, -1, 1).expand(gt, -1, gw).flatten()
            wi = torch.arange(gw).view(1, 1, -1).expand(gt, gh, -1).flatten()
            chunks.append(torch.stack([ti, hi, wi]) + text_len + st_idx)
            st = ed + gt * gh * gw
        if st < len(ids):
            st_idx = chunks[-1].max().item() + 1 if chunks else 0
            chunks.append(torch.arange(len(ids) - st).view(1, -1).expand(3, -1) + st_idx)
        pos[:, i, attention_mask[i] == 1] = torch.cat(chunks, dim=1)
    return pos


def text_rope(position_ids: torch.Tensor, text: TextConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream ``Qwen3_5TextRotaryEmbedding`` (interleaved mRoPE). ``[3,B,L]`` -> cos/sin
    ``[B, L, rotary_dim]`` fp32 (the graphs cast to the model dtype, as upstream does)."""
    r = text.rotary_dim
    inv_freq = 1.0 / (text.rope_theta ** (torch.arange(0, r, 2, dtype=torch.int64).float() / r))
    freqs = position_ids.float()[..., None] * inv_freq  # [3, B, L, r/2]
    ft = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        idx = slice(offset, text.mrope_section[dim] * 3, 3)
        ft[..., idx] = freqs[dim, ..., idx]
    emb = torch.cat([ft, ft], dim=-1)
    return emb.cos(), emb.sin()


# -- vision tables ----------------------------------------------------------------------------


def vision_tables(grid_h: int, grid_w: int, vision: VisionConfig):
    """Upstream ``Qwen3_5VisionModel.rot_pos_emb`` + ``fast_pos_embed_interpolate`` for one image
    grid (t=1), in merge-block order. Returns ``pe_idx [4,P]`` long, ``pe_w [4,P]`` fp32,
    ``cos/sin [P, head_dim]`` fp32."""
    m = vision.spatial_merge_size
    dim = vision.head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freq_table = torch.outer(torch.arange(max(grid_h, grid_w), dtype=torch.float), inv_freq)
    br, bc, ir, ic = torch.arange(grid_h // m), torch.arange(grid_w // m), torch.arange(m), torch.arange(m)
    row = (br[:, None, None, None] * m + ir[None, None, :, None]).expand(grid_h // m, grid_w // m, m, m).reshape(-1)
    col = (bc[None, :, None, None] * m + ic[None, None, None, :]).expand(grid_h // m, grid_w // m, m, m).reshape(-1)
    rot = freq_table[torch.stack([row, col], -1)].flatten(1)
    emb = torch.cat([rot, rot], dim=-1)

    n = int(math.isqrt(vision.num_position_embeddings))
    h_idx = torch.linspace(0, n - 1, grid_h)
    w_idx = torch.linspace(0, n - 1, grid_w)
    hf, wf = h_idx.int(), w_idx.int()
    hc, wc = (hf + 1).clip(max=n - 1), (wf + 1).clip(max=n - 1)
    dh, dw = h_idx - hf, w_idx - wf
    bh, bhc = hf * n, hc * n
    idx = torch.stack([
        (bh[:, None] + wf[None]).flatten(), (bh[:, None] + wc[None]).flatten(),
        (bhc[:, None] + wf[None]).flatten(), (bhc[:, None] + wc[None]).flatten(),
    ]).long()
    w = torch.stack([
        ((1 - dh)[:, None] * (1 - dw)[None]).flatten(), ((1 - dh)[:, None] * dw[None]).flatten(),
        (dh[:, None] * (1 - dw)[None]).flatten(), (dh[:, None] * dw[None]).flatten(),
    ])
    perm = torch.arange(grid_h * grid_w).view(grid_h // m, m, grid_w // m, m).permute(0, 2, 1, 3).flatten()
    return idx[:, perm].contiguous(), w[:, perm].contiguous(), emb.cos(), emb.sin()


def smart_resize(h: int, w: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 16777216):
    """Qwen2-VL ``smart_resize`` (the Qwen3-VL image processor defaults)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def images_to_patches(images: torch.Tensor, vision: VisionConfig, mean: float = 0.5, std: float = 0.5,
                      min_pixels: int = 65536, max_pixels: int = 16777216):
    """Qwen2-VL image processor for still images in ``[0, 1]`` (``do_rescale=False``, as the
    InternVLA transform calls it). ``images [N,3,H,W]`` -> ``patches [N, P, C*T*p*p]`` in
    merge-block order and the ``(grid_h, grid_w)`` patch grid."""
    p, t, m = vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size
    n, c, h, w = images.shape
    rh, rw = smart_resize(h, w, p * m, min_pixels, max_pixels)
    x = images.float()
    if (rh, rw) != (h, w):
        x = F.interpolate(x, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True)
    x = (x - mean) / std
    x = x[:, None].expand(n, t, c, rh, rw)
    gh, gw = rh // p, rw // p
    x = x.reshape(n, 1, t, c, gh // m, m, p, gw // m, m, p).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return x.reshape(n, gh * gw, c * t * p * p).contiguous(), (gh, gw)


# -- attention biases -------------------------------------------------------------------------


def prefix_bias(pad: torch.Tensor) -> torch.Tensor:
    """Causal over valid prefix tokens (upstream ``embed_prefix`` + ``make_att_2d_masks``)."""
    pad = pad.bool()
    length = pad.shape[1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    allowed = causal[None] & pad[:, None, :] & pad[:, :, None]
    return torch.where(allowed, 0.0, MASK_VALUE)[:, None].float()


def suffix_layout(policy: PolicyConfig) -> list[int]:
    """Suffix block starts (upstream ``embed_suffix``): [state?] [learnable x N] [action x chunk]."""
    att = [] if policy.tokenize_state else [1]
    att += [1] + [0] * (policy.num_learnable_tokens - 1)
    att += [1] + [0] * (policy.chunk_size - 1)
    return att


def suffix_bias(prefix_pad: torch.Tensor, policy: PolicyConfig, fast_mask: torch.Tensor | None = None):
    """Upstream ``denoise_step`` mask: suffix queries see valid (non-fast) prefix keys, plus
    suffix keys by block-causal order. ``[B, 1, S, L + S]`` fp32."""
    b, length = prefix_pad.shape
    att = torch.tensor(suffix_layout(policy))
    s = att.numel()
    cum = att.cumsum(0)
    blocks = (cum[None, :] <= cum[:, None])[None].expand(b, s, s)
    pre = prefix_pad.bool()
    if policy.block_action_attend_fast_tokens and fast_mask is not None:
        pre = pre & ~fast_mask.bool()
    allowed = torch.cat([pre[:, None, :].expand(b, s, length), blocks], dim=2)
    return torch.where(allowed, 0.0, MASK_VALUE)[:, None].float()


def suffix_positions(prefix_pos: torch.Tensor, suffix_len: int) -> torch.Tensor:
    """Upstream: ``arange(1, S+1) + max(prefix positions)`` on all three mRoPE axes."""
    mx = prefix_pos.max(dim=-1, keepdim=True).values  # [3, B, 1]
    return torch.arange(1, suffix_len + 1).view(1, 1, -1).expand(3, mx.shape[1], -1) + mx


# -- flow-matching schedule -------------------------------------------------------------------


def time_schedule(num_steps: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Upstream ``sample_actions`` Euler loop: fp32 ``time`` from 1.0 by ``dt=-1/steps`` while
    ``time >= -dt/2``. Returns the per-step times (0-dim fp32) and ``dt``."""
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32)
    time = torch.tensor(1.0, dtype=torch.float32)
    out = []
    while time >= -dt / 2:
        out.append(time.clone())
        time = time + dt
    return out, dt


def time_embedding(t: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """Upstream ``create_sinusoidal_pos_embedding`` (float64 math, ``t`` already in model dtype)."""
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** frac
    x = (1.0 / period * 2 * math.pi)[None, :] * t.double()[:, None]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


def time_embedding_table(policy: PolicyConfig, hidden: int, dtype: torch.dtype, batch: int = 1,
                         num_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``[steps, B, hidden]`` in ``dtype`` (upstream casts the timestep to the model dtype first)
    and ``dt``."""
    times, dt = time_schedule(num_steps or policy.num_inference_steps)
    rows = [time_embedding(t.expand(batch).to(dtype), hidden, policy.min_period, policy.max_period).to(dtype)
            for t in times]
    return torch.stack(rows), dt


# -- synthetic requests (tests, device smoke) -------------------------------------------------


def synthetic_request(cfg: InternVLAConfig, n_images: int = 3, grid: tuple[int, int] = (14, 14),
                      text_before: int = 12, text_after: int = 60, seed: int = 0, batch: int = 1):
    """A structurally valid request with random tokens/pixels: ``text_before`` tokens, then per
    image ``<vision_start> <image_pad> x N <vision_end>``, then ``text_after`` tokens.

    Returns a dict with upstream's batch keys: ``input_ids`` / ``attention_mask`` ``[B, L]``,
    ``pixel_values [B*n*P, C*T*p*p]``, ``image_grid_thw [B*n, 3]``, ``state [B, max_state_dim]``.
    """
    g = torch.Generator().manual_seed(seed)
    vlm, m = cfg.vlm, cfg.vlm.vision.spatial_merge_size
    gh, gw = grid
    n_tok = gh * gw // (m * m)
    rows = []
    for _ in range(batch):
        ids = torch.randint(1000, 200000, (text_before,), generator=g).tolist()
        for _ in range(n_images):
            ids += [vlm.vision_start_token_id] + [vlm.image_token_id] * n_tok + [vlm.vision_end_token_id]
        ids += torch.randint(1000, 200000, (text_after,), generator=g).tolist()
        rows.append(ids)
    input_ids = torch.tensor(rows, dtype=torch.long)
    v = vlm.vision
    pix = torch.rand(batch * n_images * gh * gw, v.in_channels * v.temporal_patch_size * v.patch_size**2,
                     generator=g) * 2 - 1
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pix,
        "image_grid_thw": torch.tensor([[1, gh, gw]] * (batch * n_images), dtype=torch.long),
        "state": torch.rand(batch, cfg.policy.max_state_dim, generator=g) * 2 - 1,
    }


def initial_noise(cfg: InternVLAConfig, batch: int = 1, seed: int = 0) -> torch.Tensor:
    """fp32 standard normal ``[B, chunk, max_action_dim]`` (upstream ``sample_noise``)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, cfg.policy.chunk_size, cfg.policy.max_action_dim, generator=g, dtype=torch.float32)
