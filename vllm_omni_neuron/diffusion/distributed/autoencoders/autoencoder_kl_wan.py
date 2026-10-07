# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Neuron-specific changes
# 1. WanRMS_norm: replaced F.normalize with manual x * rsqrt(mean(x^2)).
#    F.normalize generates a NaN-check (compare + select) that forces extra
#    CROSS_LANE_REDUCE synchronization on GpSimd (~67us per call, ~25 calls
#    in the decoder). The manual path eliminates the NaN guard entirely.
# 2. WanUpsample: replaced nn.Upsample(nearest-exact) with repeat_interleave.
#    Repeat_interleave maps to bulk tensor copies on the tensor engine. Also
#    eliminates the float32 cast that doubled memory traffic through the upsample.
# 3. Remove all to(cache_x.device) calls — feat_cache tensors stay on the Neuron
#    device throughout; no CPU↔device transfers needed.
# 4. WanResample: added first_chunk param to skip time_conv on first frame.
#    Replaces diffusers' None→"Rep" sentinel-based state machine with an
#    explicit flag, compatible with zero-initialized cache tensors.
# 5. NeuronWanDecoder3d: pads first-frame output temporally so first_chunk
#    and rest frames produce identical output shapes, enabling a single
#    compiled graph for both paths.
# 6. NeuronAutoencoderKLWan._decode: per-frame decode loop that slices z
#    on-device, strips first-frame padding on CPU, and keeps feat_cache on
#    Neuron device throughout (no CPU↔device transfer for cache).
# 7. Set feat_idx default value to None and explicitly pass [0] in NeuronWanDecoder3d.


import math
import os
import types

import nki
import nki.language as nl
import torch
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AvgDown3D,
    DupUp3D,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.experimental.conv.conv3d import conv3d
from nkilib.experimental.conv.conv3d_temporal_unroll import (
    conv3d_temporal_unroll,
    should_use_temporal_unroll,
)
from vllm_neuron.envs import get_compile_backend_name
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeExecutor,
)

from vllm_omni_neuron.lite_compat import (
    get_platform_target,
    is_lite_runtime,
    nki_op,
    register_process_group_replica_groups,
)
from vllm_omni_neuron.nc_generation import supports_nki

_VAE_ATTN_D_TILE_SIZE = 128
_VAE_ATTN_PAD_HEAD_DIM = 512
_VAE_ATTN_FLASH_THRESHOLD = 10 * 1024
# nkilib attention_cte's head-dim ceiling (_MAX_HEAD_DIM). The Wan VAE attention is single-head with
# head_dim = channels, so wider mid blocks (Wan2.2-TI2V-5B: 640 encoder / 1024 decoder) cannot use it.
_VAE_ATTN_MAX_HEAD_DIM = 512


def _vae_attn_can_use_nki(head_dim: int) -> bool:
    """attention_cte for the VAE mid-block attention: NeuronCore-v3+ and head_dim <= 512."""
    return supports_nki() and head_dim <= _VAE_ATTN_MAX_HEAD_DIM


def _vae_attention_explicit(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Single-head attention ``(N, S, D)`` without NKI: QK^T in the input dtype, fp32 scale + softmax,
    probabilities cast back for the PV matmul. Lowered by torch.compile on any NeuronCore."""
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth patchify, numerically identical to diffusers' ``patchify`` (verified bit-exact
    for every patch size/shape in ``test/unit/test_wan_vae_platform.py``), but built from only
    ADJACENT-axis ``transpose`` calls instead of one 7D ``permute(0, 1, 6, 4, 2, 3, 5)``.

    Why: that permute's non-adjacent axis reordering compiles on Trn2 to a ``DramToDramTranspose``
    the compiler rejects (``NCC_IDDT901`` assertion, every Wan2.2-5B-VAE encoder on NC-v3: Cosmos3
    Edge/Nano/Super I2V + action heads, Wan TI2V I2V — the chunked ``NeuronWanEncoder3d`` graph that
    runs patchify in-graph never compiled). A chain of single adjacent-axis transposes is the
    pattern every NC-v2/NC-v3 transpose kernel in this codebase already relies on (see e.g. the VAE
    attention padding and the Wan transformer's own axis juggling); it has never needed this
    workaround, so prefer it over the general permute on-device.
    """
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    b, c, f, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Height ({h}) and width ({w}) must be divisible by patch_size ({patch_size})"
        )
    # [b, c, f, h/p, p, w/p, p] (axes: b c f h' ph w' pw) -> walk pw to index 2, then ph to index 3,
    # each step one adjacent swap -> [b, c, pw, ph, f, h', w'].
    x = x.view(b, c, f, h // patch_size, patch_size, w // patch_size, patch_size)
    for i in range(6, 2, -1):
        x = x.transpose(i - 1, i)
    for i in range(5, 3, -1):
        x = x.transpose(i - 1, i)
    x = x.contiguous()
    return x.view(b, c * patch_size * patch_size, f, h // patch_size, w // patch_size)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`; same adjacent-transpose-chain construction, replacing diffusers'
    ``permute(0, 1, 4, 5, 3, 6, 2)``. See :func:`patchify` for why."""
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    b, c_patches, f, h, w = x.shape
    c = c_patches // (patch_size * patch_size)
    # [b, c, ph, pw, f, h, w] -> [b, c, f, h, pw, w, ph], one adjacent swap at a time.
    x = x.view(b, c, patch_size, patch_size, f, h, w)
    for i, j in ((3, 4), (2, 3), (4, 5), (3, 4), (4, 5), (5, 6)):
        x = x.transpose(i, j)
    x = x.contiguous()
    return x.view(b, c, f, h * patch_size, w * patch_size)


def _avg_down_3d_forward(self: AvgDown3D, x: torch.Tensor) -> torch.Tensor:
    """Replacement for ``AvgDown3D.forward`` (diffusers): same adjacent-transpose-chain trick as
    :func:`patchify`, this time for the encoder's space-to-depth-and-average downsample.

    diffusers builds this with ``permute(0, 1, 3, 5, 7, 2, 4, 6)`` -- another non-adjacent 8D
    permute, and the second ``DramToDramTranspose`` (``NCC_IDDT901``) found inside the Wan VAE
    encoder on Trn2 after the ``patchify`` fix (the decoder, which uses :class:`DupUp3D` not this
    class, compiled fine after the same patchify fix, isolating the remaining failure to the encoder's own
    downsample). Bit-exact against the original on CPU for every tested shape.
    """
    pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
    x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
    b, c, t, h, w = x.shape
    # [b, c, t/ft, ft, h/fs, fs, w/fs, fs] -> [b, c, ft, fs_h, fs_w, t/ft, h/fs, w/fs], one adjacent
    # swap at a time (replaces permute(0, 1, 3, 5, 7, 2, 4, 6)).
    x = x.view(
        b,
        c,
        t // self.factor_t,
        self.factor_t,
        h // self.factor_s,
        self.factor_s,
        w // self.factor_s,
        self.factor_s,
    )
    for i, j in ((2, 3), (4, 5), (3, 4), (6, 7), (5, 6), (4, 5)):
        x = x.transpose(i, j)
    x = x.contiguous()
    x = x.view(b, c * self.factor, t // self.factor_t, h // self.factor_s, w // self.factor_s)
    x = x.view(
        b,
        self.out_channels,
        self.group_size,
        t // self.factor_t,
        h // self.factor_s,
        w // self.factor_s,
    )
    return x.mean(dim=2)


def _dup_up_3d_forward(self: DupUp3D, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
    """Replacement for ``DupUp3D.forward`` (diffusers): the decoder-side counterpart of
    :func:`_avg_down_3d_forward`. The decoder already compiled on Trn2 before this change (its
    graphs apparently never hit the width/shape combination that trips the compiler), but the same
    non-adjacent permute (``permute(0, 1, 5, 2, 6, 3, 7, 4)``) is just as fragile in principle, so it
    gets the identical, verified-bit-exact fix for consistency and future-proofing rather than being
    left as the one remaining wide permute in this module.
    """
    x = x.repeat_interleave(self.repeats, dim=1)
    x = x.view(
        x.size(0),
        self.out_channels,
        self.factor_t,
        self.factor_s,
        self.factor_s,
        x.size(2),
        x.size(3),
        x.size(4),
    )
    # [b, c, ft, fs_h, fs_w, t, h, w] -> [b, c, t, ft, h, fs_h, w, fs_w], one adjacent swap at a
    # time (replaces permute(0, 1, 5, 2, 6, 3, 7, 4)).
    for i, j in ((4, 5), (3, 4), (2, 3), (5, 6), (4, 5), (6, 7)):
        x = x.transpose(i, j)
    x = x.contiguous()
    x = x.view(
        x.size(0),
        self.out_channels,
        x.size(2) * self.factor_t,
        x.size(4) * self.factor_s,
        x.size(6) * self.factor_s,
    )
    if first_chunk:
        x = x[:, :, self.factor_t - 1 :, :, :]
    return x


AvgDown3D.forward = _avg_down_3d_forward
DupUp3D.forward = _dup_up_3d_forward


def _should_pad_vae_attn_head_dim(seq_len: int, head_dim: int) -> bool:
    if not (_VAE_ATTN_D_TILE_SIZE < head_dim < _VAE_ATTN_PAD_HEAD_DIM):
        return False

    num_d_tiles = (head_dim + _VAE_ATTN_D_TILE_SIZE - 1) // _VAE_ATTN_D_TILE_SIZE
    flash_threshold = _VAE_ATTN_FLASH_THRESHOLD // num_d_tiles
    return seq_len > flash_threshold


@nki.jit
def _vae_attention_kernel(q, k, v):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,  # or pass scale=C**-0.5, see below
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def _vae_nki_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run the VAE attention kernel through ``wrap_nki``, the traceable NKI HOP.

    Registering the kernel as a custom op via ``lite_compat.nki_op`` instead installs
    the raw ``@nki.jit`` kernel as its own fake impl, so Dynamo executes it during
    fake-tensor tracing and imports ``torch_neuronx``, which the Lite image lacks.
    ``wrap_nki`` shape-infers in its meta impl, and is what the transformer uses.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(_vae_attention_kernel)  # kernel is already @nki.jit
    return wrapped[2](q, k, v)


# ---------------------------------------------------------------------------
# NKI conv3d for the decoder's 3x3x3 convolutions
# ---------------------------------------------------------------------------
#
# Dispatch is unconditional, matching every other NKI kernel in this repo.
# can_run_kernel honours VLLM_NEURON_DISABLE_NKI_KERNELS (the fleet-wide kill
# switch) and VLLM_NEURON_CPU_MODE, so anything finer is a code revert.
#
# Why only *some* call sites: measured in isolation, the generic conv3d kernel wins
# on the 3x3x3 resblock convs and is at or below parity on the 2D upsample sites --
# yet those 2D sites still ship, because in-graph they remove a compiler<->NKI
# round-trip: their marginal saving on top of the 3D group (+0.3350 s) exceeds the
# 3D group's own standalone saving (0.3161 s). Isolated numbers mispredict in both
# directions, so a site ships only when the full-model A/B says the group wins;
# everything else falls through to the compiler.
#
# conv_out is its own case: generic conv3d is ~0.41x there (PE-starved), and the
# small-C_out temporal-unroll variant clears its own gate only when tiling is OFF
# (it needs W_out > 512), so at the shipped tiled shapes conv_out runs on the
# compiler.
#
# Shape note: these convs are *causal*, and WanCausalConv3d.forward prepends 2
# cached frames while reducing pad_d_left to 0. So the convolution the hardware runs
# always has kernel padding (0, 0, 1, 1, 1, 1). That single static config covers
# every eligible site, which is what makes one traced kernel enough.

# Kernel padding order is (pad_d_l, pad_d_r, pad_h_t, pad_h_b, pad_w_l, pad_w_r).
_VAE_NKI_CONV_PADDING = (0, 0, 1, 1, 1, 1)

# Eligibility thresholds for the generic conv3d kernel: below these the contraction
# dim / PSUM partition dim are too underfilled to beat the compiler. Deliberately
# not env-tunable -- an unvalidated MIN_D_OUT=1 reachable from the environment is a
# footgun.
#
# MIN_D_OUT=2 rests on measurement, not on a "too little work per call" argument:
# relaxing it to 1 gave a +24.98% vae_decode regression end-to-end and a
# deterministic allreduce scheduling failure in the isolated harness (2/2 runs).
# MIN_CHANNELS=96 excludes conv_in (C_in = z_dim = 16); post_quant_conv is already
# excluded by its 1x1 kernel, before channels are considered. C_in only ever takes
# 16/96/192/384, so no site sits near the boundary and the exact value is untested.
_VAE_NKI_CONV_MIN_CHANNELS = 96
_VAE_NKI_CONV_MIN_D_OUT = 2

# TODO: most of the dispatch complexity below works around kernel specialisation
# rather than expressing a real policy. A more generic conv kernel would remove:
#   1. the shape thresholds here (kernel picks its own tiling);
#   2. the (kernel_size, stride, padding) preconditions in the
#      _can_use_nki_conv_kernel predicates -- these are trace-time constants, so
#      each config needs its own @nki.jit wrapper and nki_op name;
#   3. the conv3d vs conv3d_temporal_unroll choice in _maybe_nki_forward;
#   4. NkiConv2d's D=1 lifting, if a native 2D entry point existed.


@nki.jit
def _vae_conv3d_kernel(x, filters, bias):
    return conv3d(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
        lnc_shard=True,
    )


@nki.jit
def _vae_conv3d_temporal_unroll_kernel(x, filters, bias):
    return conv3d_temporal_unroll(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
    )


@nki_op("wan_vae::conv3d_3x3x3")
def _vae_nki_conv3d(x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_vae_conv3d_kernel)[2](x, filters, bias)


@nki_op("wan_vae::conv3d_3x3x3_temporal_unroll")
def _vae_nki_conv3d_temporal_unroll(
    x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_vae_conv3d_temporal_unroll_kernel)[2](x, filters, bias)


# ---------------------------------------------------------------------------
# One gate covers BOTH the 3D sites and the lifted-Conv2d sites, deliberately:
# they are not independently useful. On the isolated 16-rank harness the 2D group
# alone is worth nothing (-0.0044 s) while the two together beat the sum of their
# parts (interaction +0.3394 s). The 2D group's value is contextual -- converting a
# conv that sits BETWEEN existing NKI regions removes a compiler<->NKI round-trip
# instead of adding one -- so "3D only" and "2D only" are not configurations anyone
# should be able to select. Separate gates existed only to attribute the two halves
# during the A/B.
#
# Sites deliberately NOT routed: stride=2 downsample (encoder-only, so unreachable
# in decode) and 1x1 to_qkv/proj (~0.7 GMAC combined, measured -0.0255 s marginal).


def _pack_nki_filters(weight: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Permute ``weight`` into the NKI filter layout and materialize it contiguously.

    ``.contiguous()`` on a permuted **device** tensor raises ``Expected
    self.is_contiguous() to be true`` — the Neuron/Lite device kernels cannot
    restride in place. Weight packing happens once at compile time, so stage the
    permute+copy on CPU and move the packed result back to the original device.
    """
    src = weight.detach()
    device = src.device
    if device.type != "cpu":
        packed = src.cpu().permute(*dims).contiguous().to(src.dtype)
        return packed.to(device)
    return src.permute(*dims).contiguous().to(src.dtype)


class NkiConv2d(nn.Conv2d):
    """``nn.Conv2d`` that can route through the NKI **conv3d** kernel as ``D=1``.

    ``nkilib/experimental/conv/`` ships no conv2d kernel, so the only way to use NKI
    here is to lift the 2D conv to a degenerate 3D one: ``[N,C,H,W]`` ->
    ``[N,C,1,H,W]`` with a ``(1,K_h,K_w)`` filter, then squeeze ``D`` back out.

    Only the 3x3 stride-1 pad-1 upsample sites are routed (3 runtime modules); the
    rest are excluded on measured grounds, not on a shape limitation. Every routed
    site is ``D_out=1`` by construction, which is fine -- work scales with
    ``D_out x H_out x W_out`` and these carry ~117 GMAC combined, easily enough to
    amortise the graph boundary. Adjacency gives the sign, work gives the magnitude.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) dispatch test, checked at pack time.

        One shape only -- 3x3 stride-1 pad-1 -- so this is a bool, not a variant
        tag: there is exactly one kernel instantiation to dispatch to.
        """
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

        # Framework guard first (PR #148's pattern): honours the fleet-wide NKI
        # kill switch and CPU mode.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return False
        if tuple(self.dilation) != (1, 1) or self.groups != 1 or self.bias is None:
            return False
        if tuple(self.kernel_size) != (3, 3):
            return False
        if tuple(self.stride) != (1, 1) or tuple(self.padding) != (1, 1):
            return False
        # PE-starvation floor shared with the 3D path: C_out sits on the PSUM
        # partition dim, so small-C_out convs lose badly.
        return self.out_channels > 32 and self.in_channels >= _VAE_NKI_CONV_MIN_CHANNELS

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[1, K_h, K_w, C_in, C_out]``, D=1 lifted.

        Returns True if this site was packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 1, 0)).unsqueeze(0)
        return True

    def forward(self, x):
        if self._nki_packed_filters is None:
            return super().forward(x)
        packed = self._nki_packed_filters
        bias = self.bias.to(packed.dtype)
        # Lift to a degenerate 3D conv: D=1 in, D_out=1 out (K_d=1, d-pad=0).
        x_in = x.unsqueeze(2).to(packed.dtype)
        return _vae_nki_conv3d(x_in, packed, bias).squeeze(2)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CACHE_T = 2


def _own_code_object(fn, tag: str):
    """Return ``fn`` rebuilt on a private copy of its code object.

    Dynamo keys its compiled-frame cache (and the recompile limit) on the CODE object, so every
    closure made from the same ``def`` shares one cache however different its captured constants
    are. Giving each compiled graph its own code copy (``co_name`` tagged) gives it its own cache
    entry list: one compile, no shared limit. Closure cells, globals and defaults are kept.
    """
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_{tag}")
    return types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)


# trn2.48xlarge collective routability (see diffusion/distributed/parallel_state.py): 16 chips in
# a 4x4 torus, 4 logical cores per chip at LNC=2, rank r on chip r // 4. A collective is routable
# iff its member chips form a torus ring -- one chip, a torus-adjacent pair (including the wrap),
# a full row or column, or a rectangular block of them -- and every chip contributes the same
# core offsets. Three chips in a line are a path, not a ring, and the fabric rejects them
# ("no_hier no_mesh", surfacing as NRT_INVALID).
_TRN2_CORES_PER_CHIP = 4
_TRN2_CHIP_TORUS = (4, 4)
_TORUS_CHECKED_PLATFORMS = frozenset({"trn2"})


def _is_torus_ring_arc(values: list[int], ring: int) -> bool:
    """``values`` (sorted, distinct, in ``range(ring)``) form a ring: all of it, one node, or two
    torus-adjacent nodes (wrap included). Any other proper subset is a path."""
    if len(values) in (1, ring):
        return True
    if len(values) != 2:
        return False
    first, second = values
    return (second - first) % ring == 1 or (first - second) % ring == 1


def gather_group_torus_routable(
    ranks,
    *,
    cores_per_chip: int = _TRN2_CORES_PER_CHIP,
    torus: tuple[int, int] = _TRN2_CHIP_TORUS,
) -> bool:
    """Whether a replica group of global ``ranks`` is routable on the trn2 chip torus.

    Rank ``r`` sits on chip ``r // cores_per_chip`` and chips are numbered row-major on the
    ``torus`` grid. Routable iff every chip contributes the same core offsets and the chip set is
    a (cyclic) rectangle whose row set and column set are each a torus ring arc.
    """
    ranks = sorted({int(r) for r in ranks})
    if not ranks:
        return False
    by_chip: dict[int, list[int]] = {}
    for rank in ranks:
        by_chip.setdefault(rank // cores_per_chip, []).append(rank % cores_per_chip)
    chips = sorted(by_chip)
    if len(chips) == 1:
        return True
    offsets = {tuple(sorted(o)) for o in by_chip.values()}
    if len(offsets) != 1:
        return False
    torus_rows, torus_cols = torus
    if chips[-1] >= torus_rows * torus_cols:
        return False
    rows = sorted({chip // torus_cols for chip in chips})
    cols = sorted({chip % torus_cols for chip in chips})
    if len(chips) != len(rows) * len(cols):
        return False  # not a full rectangle of the row x column sets
    return _is_torus_ring_arc(rows, torus_rows) and _is_torus_ring_arc(cols, torus_cols)


VAE_GROUP_STRICT_ENV = "VLLM_OMNI_NEURON_VAE_GROUP_STRICT"


def check_vae_group_covers_world(group_world: int, global_world: int | None) -> None:
    """Warn (or refuse with ``VLLM_OMNI_NEURON_VAE_GROUP_STRICT=1``) when the VAE-parallel group is
    smaller than the job.

    A VAE group that excludes ranks is only correct if the excluded ranks never touch VAE outputs
    themselves. In practice they do: an I2V pipeline encodes the conditioning image on every rank,
    and a rank outside the group runs a different VAE object (different tiling, no group
    collective) and gets different latents -- the CP slices then denoise against inconsistent
    conditioning (observed as a visibly different clip, SSIM 0.94 vs the full-group run). The
    executor cannot see what the excluded ranks do, so by default it warns; a pipeline that
    guarantees the excluded ranks consume only broadcast results may keep the smaller group, and
    one that cannot should set the strict env (or refuse itself, as the Cosmos3 port does)."""
    if global_world is None or group_world >= global_world:
        return
    message = (
        f"VAE patch-parallel group has {group_world} ranks but the job has {global_world}: ranks "
        "outside the group must not encode/decode on their own (they would get different latents "
        "than the group). Use vae_patch_parallel_size == world size, or broadcast every VAE result "
        "to the excluded ranks."
    )
    if os.environ.get(VAE_GROUP_STRICT_ENV, "0") == "1":
        raise RuntimeError(message)
    logger.warning(message)


def _axis_blend_plan(starts, stride, blend, total):
    """Per-tile blend/crop plan along one axis, in output units: for tile ``k`` at ``starts[k]``,
    ``(keep_offset, keep_len, blend_n, cur_offset, prev_offset)``. Tile ``k`` keeps the output from
    where tile ``k-1``'s range ended to ``starts[k] + stride`` (the last tile: to ``total``); the
    first ``blend_n = min(blend, keep_len)`` samples of that range ramp from tile ``k-1`` (read at
    ``prev_offset`` inside it) into tile ``k`` (at ``cur_offset``). Evenly spaced starts give
    diffusers' crop-to-stride and ``prev[-blend:]`` / ``cur[:blend]``; a last tile pulled back to
    ``total - tile`` keeps only the remainder (same rule as ``vae_tiling.kept_ranges``)."""
    plan = []
    prev_end = 0
    for k, start in enumerate(starts):
        end = total if k == len(starts) - 1 else min(start + stride, total)
        begin = prev_end
        if begin < start:
            raise ValueError(f"tile {k} starts at {start}, after the covered range ends ({begin})")
        n = max(0, min(blend, end - begin)) if k > 0 else 0
        prev_offset = begin - starts[k - 1] if k > 0 else 0
        plan.append((begin - start, end - begin, n, begin - start, prev_offset))
        prev_end = end
    return tuple(plan)


class _ShapeDispatchedGather:
    """Callable standing in for one ``torch.compile``'d gather: compiles (and caches, through the
    executor's ``_compile_device_graph``) one graph per input shape, each on its own code object,
    and runs every call under ``no_grad`` so the grad mode never forces a recompile."""

    def __init__(self, executor, fn):
        self._executor = executor
        self._fn = fn

    def __call__(self, tensor):
        graph = self._executor._compile_device_graph("gather", tuple(tensor.shape), self._fn)
        with torch.no_grad():
            return graph(tensor)


class _NeuronDistributedVaeExecutor(DistributedVaeExecutor):
    """Route control metadata over CPU/Gloo and payload collectives over the device."""

    MAX_DEVICE_GATHER_BYTES = 512 * 1024 * 1024
    # Env override of the per-chunk gather budget (MiB). The gathered chunk is resident on EVERY
    # rank during the collective and on rank 0 during the blend, next to the DiT weights: a
    # 32-rank TI2V/Cosmos3 decode at 512 MiB sat at 22.5 GB of 24 per core. Smaller chunks mean
    # more collectives (each ~ms) and less HBM.
    GATHER_BUDGET_ENV = "VLLM_OMNI_NEURON_VAE_GATHER_MB"

    def gather_budget_bytes(self) -> int:
        raw = os.environ.get(self.GATHER_BUDGET_ENV)
        if raw:
            try:
                return max(1, int(raw)) * 1024 * 1024
            except ValueError:
                logger.warning("ignoring %s=%r (not an integer MiB)", self.GATHER_BUDGET_ENV, raw)
        return self.MAX_DEVICE_GATHER_BYTES

    @staticmethod
    def _await_device_tensor(tensor):
        """Wait for a device tensor without copying its payload to the host.

        ``tensor`` must be a BASE tensor (``storage_offset() == 0``). The Lite runtime's
        device-to-device ``copy_`` sizes the transfer as ``dst_bytes - src_storage_offset_bytes``,
        so a one-element probe of a view that starts past the beginning of its storage asks NRT
        for a negative byte count (``nrt_tensor_copy status=2`` / ``NRT_INVALID``, logged as
        ``Cannot copy 18446744073642442754 bytes ... to dst tensor of size 2``). Refuse such a
        view here, with the real reason, instead of letting the runtime report it in 2^64 form.
        """
        if tensor.device.type != "cpu" and tensor.storage_offset() != 0:
            raise RuntimeError(
                "_await_device_tensor needs a base device tensor (storage_offset 0), got a view "
                f"at element offset {tensor.storage_offset()} of shape {tuple(tensor.shape)}: the "
                "Lite device copy_ mis-sizes an offset source (nrt_tensor_copy NRT_INVALID)"
            )
        source = tensor.view(-1)[:1]
        torch.empty_like(source).copy_(source)

    def _compile_device_graph(self, name, key, fn):
        """Compile and cache a fixed-shape VAE payload graph.

        Every graph gets its own Dynamo cache: the per-geometry closures compiled here
        (``plane_chunk`` per chunk, ``slot_major`` per size, ``merge`` per grid) share ONE code
        object each, and Dynamo caches compiled frames per code object with a recompile limit
        (8 by default). A 64-rank decode cuts up to 7 chunks and a second resolution adds its own,
        so without isolation the limit trips: ``FailOnRecompileLimitHit`` under ``fullgraph=True``
        (verified on CPU at the 9th chunk). See :func:`_own_code_object`."""
        graphs = getattr(self, "_device_graphs", None)
        if graphs is None:
            graphs = {}
            self._device_graphs = graphs
        graph = graphs.get((name, key))
        if graph is None:
            graph = torch.compile(
                _own_code_object(fn, f"{name}_{len(graphs)}"),
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": f"wan_vae_{name}"},
            )
            graphs[(name, key)] = graph
        return graph

    def concat_tile_frames(self, frames):
        """Concatenate temporal tile outputs on the Neuron device."""

        def concat(*frame_outputs):
            return torch.cat(frame_outputs, dim=2)

        graph = self._compile_device_graph("tile_concat", len(frames), concat)
        result = graph(*frames)
        self._await_device_tensor(result)
        return result

    def _build_coord_index(self, meta_gather, grid_spec, tid_coord_map):
        """Map each grid coordinate to the ``(rank, slot)`` holding its tile."""
        coord_index = {}
        for rank in range(self.world_size):
            meta_src = meta_gather[rank]
            for idx in range(meta_src.shape[0]):
                tid = int(meta_src[idx, 0])
                if tid < 0:
                    continue
                coord_index[tid_coord_map[tid]] = (rank, idx)
        return coord_index

    def _compiled_blend_graph(
        self,
        gathered_shape,
        tile_sources,
        grid_height,
        grid_width,
        heights,
        widths,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
        row_starts=None,
        col_starts=None,
    ):
        """Compile (cached) the tile blend for one plane chunk.

        The single graph input is the gathered chunk ``[world, size, slots, H, W]`` (planes-major,
        straight from the collective); ``tile_sources`` lists, in raster order, the ``(rank, slot)``
        each grid tile lives at, and the graph takes the ``[rank, :, slot]`` view INSIDE itself. Keyed
        on the gathered shape, the tile map and the geometry.

        ``row_starts`` / ``col_starts`` (output units) place each tile; ``None`` means evenly spaced
        (``k * stride``). Each tile owns the output from where the previous tile's range ended to its
        own ``start + stride`` (the last: to the edge), and the first ``blend`` samples of that range
        are diffusers' linear ramp against the (already blended) previous tile -- so a last tile
        pulled back to end at the frame edge blends and crops correctly. On evenly spaced tiles this
        is exactly diffusers' ``blend_v``/``blend_h`` + crop-to-stride."""
        if row_starts is None:
            row_starts = tuple(row * stride_height for row in range(grid_height))
        if col_starts is None:
            col_starts = tuple(column * stride_width for column in range(grid_width))
        row_starts, col_starts = tuple(row_starts), tuple(col_starts)
        row_plan = _axis_blend_plan(row_starts, stride_height, blend_height, full_height)
        col_plan = _axis_blend_plan(col_starts, stride_width, blend_width, full_width)

        def ramp(n, like, shape):
            return (torch.arange(n, device=like.device, dtype=like.dtype) / n).reshape(shape)

        def blend_vertical(above, tile, plan):
            _, _, n, off_cur, off_prev = plan
            if n <= 0:  # stride == tile size: abutting tiles, nothing to blend
                return tile
            weights = ramp(n, tile, (n, 1))
            blended = (
                above[..., off_prev : off_prev + n, :] * (1.0 - weights)
                + tile[..., off_cur : off_cur + n, :] * weights
            )
            parts = (tile[..., :off_cur, :], blended, tile[..., off_cur + n :, :])
            return torch.cat([part for part in parts if part.shape[-2] > 0], dim=-2)

        def blend_horizontal(left, tile, plan):
            _, _, n, off_cur, off_prev = plan
            if n <= 0:
                return tile
            weights = ramp(n, tile, (1, n))
            blended = (
                left[..., off_prev : off_prev + n] * (1.0 - weights)
                + tile[..., off_cur : off_cur + n] * weights
            )
            parts = (tile[..., :off_cur], blended, tile[..., off_cur + n :])
            return torch.cat([part for part in parts if part.shape[-1] > 0], dim=-1)

        def merge(gathered):
            blended_tiles = []
            rows = []
            for row in range(grid_height):
                row_tiles = []
                keep_top, keep_rows = row_plan[row][:2]
                for column in range(grid_width):
                    index = row * grid_width + column
                    src_rank, src_slot = tile_sources[index]
                    tile = gathered[src_rank, :, src_slot, : heights[row], : widths[column]]
                    if row > 0:
                        tile = blend_vertical(
                            blended_tiles[index - grid_width], tile, row_plan[row]
                        )
                        if column > 0:
                            tile = tile.clone(memory_format=torch.contiguous_format)
                    if column > 0:
                        tile = blend_horizontal(blended_tiles[index - 1], tile, col_plan[column])
                    blended_tiles.append(tile)
                    keep_left, keep_cols = col_plan[column][:2]
                    row_tiles.append(
                        tile[
                            ..., keep_top : keep_top + keep_rows, keep_left : keep_left + keep_cols
                        ]
                    )
                rows.append(torch.cat(row_tiles, dim=-1))
            merged = torch.cat(rows, dim=-2)[..., :full_height, :full_width]
            return torch.clamp(merged, min=-1.0, max=1.0) if clamp else merged

        key = (
            tuple(gathered_shape),
            tuple(tile_sources),
            (grid_height, grid_width),
            heights,
            widths,
            (
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
            ),
            row_starts,
            col_starts,
        )
        # Name stays "merge" so its compilation remains observable to tests.
        return self._compile_device_graph("merge", key, merge)

    def _gather_chunk(self, compiled_gather, chunk, chunk_index):
        """Run one device all-gather with its post-gather barrier; returns the
        gathered ``[world, *chunk.shape]`` tensor on rank 0, ``None`` elsewhere."""
        gathered = compiled_gather(chunk)
        post_gather_error = None
        result = None
        try:
            self._await_device_tensor(gathered)
            if self.rank == 0:
                result = gathered
        except Exception as error:
            post_gather_error = error
        finally:
            if self.rank != 0:
                del gathered
        self._raise_if_device_gather_failed(post_gather_error, f"post-gather chunk {chunk_index}")
        return result

    def _stream_gather_planes(self, local_planes, chunk_planes, planes):
        """Yield the plane-chunk gathers, hoisting the Lite setup/precompile
        barriers out of the per-chunk loop (only post-gather stays per chunk)."""
        starts = list(range(0, planes, chunk_planes))

        if local_planes.device.type == "cpu" or not is_lite_runtime():
            for start in starts:
                size = min(chunk_planes, planes - start)
                per_rank = self.gather_tensors(local_planes.narrow(1, start, size).contiguous())
                if per_rank is None:
                    yield None
                    continue
                # same planes-major [world, size, slots, H, W] layout the device gather produces
                yield torch.stack(per_rank, dim=0).transpose(1, 2).contiguous()
            return

        yield from self._stream_gather_planes_lite(local_planes, chunk_planes, planes, starts)

    def _stream_gather_planes_lite(self, local_planes, chunk_planes, planes, starts):
        """Device (Lite) branch of :meth:`_stream_gather_planes`; split out so the chunk/layout
        logic is unit-testable on CPU with the compile and the collective stubbed."""
        setup_error = None
        try:
            compiled_gather = self._get_compiled_device_gather()
        except Exception as error:
            setup_error = error
        self._raise_if_device_gather_failed(setup_error, "setup")

        # The plane chunk used to be local_planes.narrow(1, start, size).contiguous() -- an EAGER
        # narrow + contiguous on a DEVICE tensor, which the Lite executor refuses ('Expected
        # self.is_contiguous()'; seen on Wan2.2-TI2V-5B 704p patch-parallel decode). The first fix cut ALL chunks in ONE compiled multi-output graph
        # (transpose -> split -> .contiguous() each). That breaks whenever ``slots == 1``: the
        # transpose is then a pure re-labelling (a size-1 axis moves, the layout stays contiguous),
        # ``torch.split`` yields contiguous views of the graph INPUT and ``.contiguous()`` is a
        # no-op, so AOT autograd classifies every chunk as an alias of the input and hands back
        # ``as_strided`` views of ``local_planes`` -- chunk k at element offset k*chunk_numel.
        # The Lite device ``copy_`` then mis-sizes the one-element wait probe of any chunk after
        # the first (see _await_device_tensor: 'Cannot copy 18446744073642442754 bytes ... dst
        # tensor of size 2' = 2 - 2^26 bytes = exactly two 256-plane chunks at 256x256 bf16 for a
        # 189-frame clip on 16 ranks; a 121-frame 704x1280 clip on 32 ranks hit 2 - 2^25).
        # Multi-slot layouts (more tiles than ranks) and single-chunk payloads never aliased,
        # which is why 16 ranks worked and 9-frame clips worked.
        #
        # Now: ONE compiled graph per chunk, whose only output is a ``clone`` -- a fresh,
        # contiguous base tensor in every layout, including slots == 1 -- cut lazily right
        # before its gather and dropped right after, so peak HBM is one chunk, not the whole
        # payload twice. The gathered chunk is yielded planes-major, ``[world, size, slots, H, W]``
        # (a reshape of the collective's output, no copy): the blend graph picks each tile's
        # ``[rank, :, slot]`` view INSIDE its own graph, so the old slot-major restore copy -- a
        # second budget-sized buffer resident on rank 0 next to the DiT weights -- is gone.
        slots, _, height, width = local_planes.shape
        world = self.world_size
        planes_key = tuple(local_planes.shape)

        def cut_chunk(
            start, size
        ):  # [slots, planes, H, W] -> [size, slots, H, W], fresh base tensor
            def fn(t):
                return (
                    t.transpose(0, 1)
                    .narrow(0, start, size)
                    .clone(memory_format=torch.contiguous_format)
                )

            chunk = self._compile_device_graph("plane_chunk", (planes_key, start, size), fn)(
                local_planes
            )
            self._check_gather_payload(chunk, (size, slots, height, width), local_planes)
            self._await_device_tensor(chunk)
            return chunk

        # Every chunk is chunk_planes wide except a possible shorter tail, so only the full shape
        # and (when it differs) the tail shape need precompiling. Cut those two chunks up front
        # for the preflight and hand them to their own gather iteration below instead of cutting
        # them twice (the tail stays resident until its turn: at most one extra chunk).
        sizes = [min(chunk_planes, planes)]
        tail = planes - starts[-1]
        if tail != sizes[0]:
            sizes.append(tail)
        cut_ahead = {}
        preflight = []
        for size in sizes:
            index = 0 if size == sizes[0] else len(starts) - 1
            cut_ahead[index] = cut_chunk(starts[index], size)
            source = cut_ahead[index].reshape(-1)
            preflight.append((0, source.numel(), source))
        self._precompile_lite_device_gather(compiled_gather, preflight, local_planes)
        del preflight

        for chunk_index, start in enumerate(starts):
            size = min(chunk_planes, planes - start)
            local_chunk = cut_ahead.pop(chunk_index, None)
            if local_chunk is None:
                local_chunk = cut_chunk(start, size)
            assert local_chunk.shape[0] == size, (local_chunk.shape, size)
            gathered = self._gather_chunk(compiled_gather, local_chunk.reshape(-1), chunk_index)
            del local_chunk  # the gather holds its own copy; free the cut before the blend
            if gathered is None:
                yield None
                continue
            yield gathered.reshape(world, size, slots, height, width)  # view, offset 0
            del gathered  # on resume, free the buffer before the next gather

    @staticmethod
    def _check_gather_payload(chunk, shape, source):
        """A chunk handed to the compiled gather must be a fresh contiguous base tensor of
        ``shape``: not a view (storage offset 0) and not sharing storage with ``source`` -- the
        aliasing that produced the negative-byte-count copy (see _stream_gather_planes_lite)."""
        if tuple(chunk.shape) != tuple(shape):
            raise RuntimeError(f"plane chunk has shape {tuple(chunk.shape)}, expected {shape}")
        if not chunk.is_contiguous() or chunk.storage_offset() != 0:
            raise RuntimeError(
                "plane chunk is a strided/offset view "
                f"(storage_offset={chunk.storage_offset()}, contiguous={chunk.is_contiguous()}); "
                "the compiled cut must return a fresh base tensor"
            )
        if chunk._is_view() or chunk.untyped_storage() is source.untyped_storage():
            raise RuntimeError(
                "plane chunk aliases the local plane tensor; the compiled cut must clone, not view"
            )
        chunk_ptr = chunk.untyped_storage().data_ptr()  # 0 on Neuron tensors: then undecidable
        if chunk_ptr and chunk_ptr == source.untyped_storage().data_ptr():
            raise RuntimeError(
                "plane chunk shares storage with the local plane tensor; the compiled cut must clone"
            )

    def gather_and_blend_tiles(
        self,
        local_tile_tensor,
        meta_gather,
        grid_spec,
        tid_coord_map,
        *,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
        to_host=False,
        row_starts=None,
        col_starts=None,
    ):
        """All-gather and blend the padded tiles, streamed over plane chunks.

        ``row_starts`` / ``col_starts``: each grid row's / column's start in output units (``None``:
        evenly spaced, ``k * stride``). The decode split passes them so its full-size last tile,
        pulled back to end at the frame edge, is placed and blended where it was cut.

        Instead of gathering the whole ``[world, slots, C, T, H, W]`` payload to
        rank 0 (~972 MiB at 720p, which OOMed the rank) then merging, this gathers
        only one channel*frame ("planes") slice at a time, blends it on rank 0, and
        frees it before the next. Every rank drives the per-chunk gather (collective); only
        rank 0 blends and returns a tensor, others return ``None``.

        Rank-0 HBM: one gathered chunk (<= :meth:`gather_budget_bytes`) plus, with
        ``to_host=False``, the merged chunks and their final join (2x the output while
        joining). With ``to_host=True`` each merged chunk is copied to the host as soon as it
        is blended and the join happens there, so the device holds one gathered chunk and one
        merged chunk at a time -- the mode for callers that move the result to the host anyway
        (patchified VAEs unpatchify on the host). The other ranks hold nothing past the gather.
        """
        slots = local_tile_tensor.shape[0]
        slot_shape = tuple(local_tile_tensor.shape[1:])
        slot_height, slot_width = slot_shape[-2], slot_shape[-1]
        frame_shape = slot_shape[:-2]
        planes = math.prod(frame_shape)
        local_planes = local_tile_tensor.reshape(slots, planes, slot_height, slot_width)

        # Take the largest plane chunk whose gathered form
        # ([world, slots, chunk_planes, H, W]) still fits the collective budget, so
        # each per-chunk gather is single-shot (no internal element-chunk join) and
        # the chunk count matches what the old full-payload gather already used --
        # no extra collectives. Smaller payloads gather whole in one chunk as before.
        plane_bytes = (
            self.world_size * slots * slot_height * slot_width * local_tile_tensor.element_size()
        )
        budget_planes = max(1, self.gather_budget_bytes() // plane_bytes) if plane_bytes else planes
        chunk_planes = max(1, min(planes, budget_planes))

        grid_height, grid_width = grid_spec.grid_shape
        coord_index = (
            self._build_coord_index(meta_gather, grid_spec, tid_coord_map)
            if self.rank == 0
            else None
        )
        if row_starts is None:
            row_starts = tuple(row * stride_height for row in range(grid_height))
        if col_starts is None:
            col_starts = tuple(column * stride_width for column in range(grid_width))
        row_starts, col_starts = tuple(row_starts), tuple(col_starts)
        if len(row_starts) != grid_height or len(col_starts) != grid_width:
            raise ValueError(
                f"{len(row_starts)}x{len(col_starts)} tile starts for a "
                f"{grid_height}x{grid_width} grid"
            )
        heights = tuple(min(slot_height, full_height - start) for start in row_starts)
        widths = tuple(min(slot_width, full_width - start) for start in col_starts)

        tile_sources = (
            tuple(
                coord_index[(row, column)]
                for row in range(grid_height)
                for column in range(grid_width)
            )
            if self.rank == 0
            else None
        )
        merged_chunks = []
        for gathered in self._stream_gather_planes(local_planes, chunk_planes, planes):
            if self.rank != 0:
                continue
            blend = self._compiled_blend_graph(
                tuple(gathered.shape),
                tile_sources,
                grid_height,
                grid_width,
                heights,
                widths,
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
                row_starts=row_starts,
                col_starts=col_starts,
            )
            merged = blend(gathered)
            self._await_device_tensor(merged)
            if to_host and merged.device.type != "cpu":
                merged = merged.to("cpu")  # base tensor (graph output): a plain D2H copy
            merged_chunks.append(merged)
            del gathered  # free the gathered chunk before the next collective

        if self.rank != 0:
            return None
        if len(merged_chunks) == 1:
            result = merged_chunks[0]
        elif merged_chunks[0].device.type == "cpu":
            result = torch.cat(merged_chunks, dim=0)
        else:

            def join(*parts):
                return torch.cat(parts, dim=0)

            shapes = tuple(tuple(part.shape) for part in merged_chunks)
            result = self._compile_device_graph("merge_join", shapes, join)(*merged_chunks)
            self._await_device_tensor(result)
        del merged_chunks
        return result.reshape(*frame_shape, full_height, full_width)

    def execute(self, z, operator, broadcast_result=True):
        """Tiled VAE decode/encode with a streamed gather+blend.

        Mirrors ``DistributedVaeExecutor.execute`` but only gathers the small tile
        metadata up front; ``operator.merge`` receives this rank's local padded
        tiles plus that metadata and streams the payload gather itself (see
        :meth:`gather_and_blend_tiles`), avoiding the full-payload gather buffer
        that OOMed a rank at 720p. Keep in lockstep with the base method.
        """
        pp_size = min(self.parallel_size, self.world_size)

        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if pp_size <= self.world_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim, z.device)
        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, z.device, output_dtype
        )

        meta_gather = self.gather_tensors(local_meta_tensor)

        result = operator.merge(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)
        if self.rank != 0:
            result = torch.empty(0, device=z.device)  # Dummy return for non-zero ranks.

        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
        return result

    @staticmethod
    def gather_replica_groups(world_size: int, global_world_size: int) -> list[list[int]]:
        """The replica-group partition registered for the compiled VAE gather: the global world
        split into consecutive blocks of ``world_size`` ranks (the VAE group is ranks
        ``0..world_size-1``; the other blocks are its SPMD images). Pure, for unit tests."""
        if world_size <= 0 or global_world_size % world_size != 0:
            raise ValueError(
                f"VAE gather group of {world_size} ranks does not divide a world of "
                f"{global_world_size}"
            )
        return [
            list(range(start, start + world_size))
            for start in range(0, global_world_size, world_size)
        ]

    @staticmethod
    def _check_gather_groups_routable(groups: list[list[int]], target: str | None = None) -> None:
        """Refuse, with the reason, a VAE gather group the trn2 torus cannot route.

        Without this the all-gather compiles and then every rank dies in NRT with a bare
        ``NRT_INVALID``. ``target`` defaults to the detected Lite platform; the check only applies
        to platforms whose topology is classified (``_TORUS_CHECKED_PLATFORMS``)."""
        if target is None:
            try:
                target = get_platform_target()
            except Exception:  # CPU host / undetectable platform: nothing to check against
                return
        if target not in _TORUS_CHECKED_PLATFORMS:
            return
        for group in groups:
            if not gather_group_torus_routable(group):
                raise RuntimeError(
                    f"VAE patch-parallel gather group {group[0]}..{group[-1]} ({len(group)} ranks) "
                    f"is not routable on the {target} chip torus: rank r is on chip r // "
                    f"{_TRN2_CORES_PER_CHIP}, and the group's chips must form a torus ring (one "
                    "chip, an adjacent pair, a full row of 4, or a block of full rows -- i.e. "
                    "1-4, 8, 16, 32 or 64 ranks from rank 0). Lower vae_patch_parallel_size to "
                    "one of those."
                )

    def _get_compiled_device_gather(self):
        compiled_gather = getattr(self, "_compiled_device_gather", None)
        if compiled_gather is None:
            group = self.group
            world_size = self.world_size

            if is_lite_runtime():
                group_name = getattr(group, "group_name", None)
                if group_name is None:
                    raise RuntimeError("VAE process group has no group_name for Lite compilation")
                group_ranks = torch.distributed.get_process_group_ranks(group)
                global_world_size = torch.distributed.get_world_size()
                if group_ranks != list(range(world_size)) or global_world_size % world_size != 0:
                    raise RuntimeError(
                        "VAE process group must be a contiguous rank-0 group whose size "
                        "evenly divides the global world for Lite compilation"
                    )
                replica_groups = self.gather_replica_groups(world_size, global_world_size)
                self._check_gather_groups_routable(replica_groups)
                register_process_group_replica_groups(group_name, replica_groups)

            def device_gather(tensor):
                gathered = funcol.all_gather_tensor(tensor, gather_dim=0, group=group)
                return gathered.reshape(world_size, *tensor.shape)

            # One compiled graph PER CHUNK SHAPE, each on its own code object (via
            # _compile_device_graph), instead of one torch.compile'd function recompiled for every
            # new shape: Dynamo caps recompiles per code object at 8 (fullgraph -> hard failure),
            # and a server that decodes several resolutions -- or a smoke that gathers a dozen chunk
            # shapes -- walks past that ("Lite VAE device gather failed during compiled gather
            # precompile", 8-rank smoke r3). The grad mode is pinned too: the precompile runs under
            # no_grad and a differing grad mode at the real call was another recompile per shape.
            compiled_gather = _ShapeDispatchedGather(self, device_gather)
            self._compiled_device_gather = compiled_gather
        return compiled_gather

    @staticmethod
    def _get_lite_gather_precompile_contexts():
        try:
            from libtorch_neuronx_lite.compile.lite_compile_only import (
                lite_compile_only,
            )
            from libtorch_neuronx_lite.compile.native_backend import (
                native_precompile,
            )
        except ImportError as error:
            raise RuntimeError(
                "Lite VAE gather precompile requires native_precompile() and "
                "lite_compile_only() from the pinned libtorch-neuronx-lite build"
            ) from error

        if not callable(native_precompile) or not callable(lite_compile_only):
            raise RuntimeError(
                "Lite VAE gather precompile requires callable native_precompile() and "
                "lite_compile_only() contexts from the pinned libtorch-neuronx-lite build"
            )
        return native_precompile, lite_compile_only

    def _raise_if_device_gather_failed(self, local_error, phase):
        failure = torch.tensor(
            [int(local_error is not None)],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            failure,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        if not failure.item():
            return

        message = f"Lite VAE device gather failed during {phase}"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _precompile_lite_device_gather(self, compiled_gather, chunks, tensor):
        precompiled_signatures = getattr(self, "_lite_gather_preflight_signatures", None)
        signature_chunks = {}
        for _, _, chunk in chunks:
            signature = (
                id(compiled_gather),
                tuple(chunk.shape),
                tensor.dtype,
                str(tensor.device),
                self.world_size,
                getattr(self.group, "group_name", None),
            )
            signature_chunks.setdefault(signature, chunk)

        needs_precompile = torch.tensor(
            [
                int(
                    precompiled_signatures is None
                    or any(
                        signature not in precompiled_signatures for signature in signature_chunks
                    )
                )
            ],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            needs_precompile,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        if not needs_precompile.item():
            return

        precompile_error = None
        try:
            native_precompile, lite_compile_only = self._get_lite_gather_precompile_contexts()
            with torch.no_grad(), lite_compile_only(), native_precompile():
                for chunk in signature_chunks.values():
                    placeholder = compiled_gather(chunk)
                    del placeholder
        except Exception as error:
            precompile_error = error
        self._raise_if_device_gather_failed(
            precompile_error,
            "compiled gather precompile",
        )

        if precompiled_signatures is None:
            precompiled_signatures = set()
            self._lite_gather_preflight_signatures = precompiled_signatures
        precompiled_signatures.update(signature_chunks)

    def _compute_global_padding_shape(self, local_results, output_ndim: int, device):
        global_padding_shape = super()._compute_global_padding_shape(
            local_results,
            output_ndim,
            device="cpu",
        )
        global_padding_shape[0] = max(global_padding_shape[0], 1)
        return global_padding_shape

    def _sync_final_result(self, rank0_result, output_ndim, output_device, output_dtype):
        """Broadcast result shape through Gloo and keep the payload on Neuron."""
        if self.rank == 0:
            shape_tensor = torch.tensor(tuple(rank0_result.shape), device="cpu", dtype=torch.int64)
        else:
            shape_tensor = torch.empty((output_ndim,), device="cpu", dtype=torch.int64)
        torch.distributed.broadcast(shape_tensor, src=0, group=self.group)

        if self.rank == 0:
            sync_result = rank0_result
        else:
            sync_result = torch.empty(
                tuple(shape_tensor.tolist()),
                device=output_device,
                dtype=output_dtype,
            )
        torch.distributed.broadcast(sync_result, src=0, group=self.group)
        self._await_device_tensor(sync_result)
        return sync_result

    def _pack_local_tiles(self, local_results, global_padding_shape, grid_spec, device, dtype):
        """Pad this rank's tiles to the slot shape and stack them, on device.

        Upstream writes each tile into a preallocated tensor by strided
        assignment, which the device kernels cannot do, so the pad-and-stack runs
        as a compiled graph instead. The metadata tensor stays on CPU: rank 0
        reads tile ids out of it as host ints.
        """
        slots = global_padding_shape[0]
        meta_tensor = torch.full(
            (slots, len(grid_spec.split_dims) + 1),
            -1,
            device="cpu",
            dtype=torch.int64,
        )
        for idx, (tid, t_tensor) in enumerate(local_results):
            meta_tensor[idx, 0] = tid
            for i, dim in enumerate(grid_spec.split_dims):
                meta_tensor[idx, i + 1] = t_tensor.shape[dim]

        if not local_results:
            return torch.zeros(global_padding_shape, device=device, dtype=dtype), meta_tensor

        slot_shape = tuple(global_padding_shape[1:])
        tiles = [tile for _, tile in local_results]
        key = (slot_shape, tuple(tuple(tile.shape) for tile in tiles), slots)

        def pack(*tile_tensors):
            padded = []
            for tile in tile_tensors:
                pad = []
                for axis in range(len(slot_shape) - 1, -1, -1):
                    pad.extend((0, slot_shape[axis] - tile.shape[axis]))
                padded.append(F.pad(tile, pad) if any(pad) else tile)
            stacked = torch.stack(padded, dim=0)
            if stacked.shape[0] < slots:
                empty = torch.zeros(
                    (slots - stacked.shape[0], *slot_shape),
                    dtype=stacked.dtype,
                    device=stacked.device,
                )
                stacked = torch.cat((stacked, empty), dim=0)
            return stacked

        payload = self._compile_device_graph("pack", key, pack)(*tiles)
        self._await_device_tensor(payload)
        return payload, meta_tensor


def _tail_cache(x):
    """Extract the last CACHE_T temporal frames from x for caching."""
    return x[:, :, -CACHE_T:, :, :].clone()


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0

    Neuron changes:
      - Removed .to(cache_x.device) in forward() to avoid cross-device copy
        errors within compiled graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

        # Set by install_nki_dispatch() when this site is NKI-eligible.
        # None => use the compiler path.
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) half of the NKI dispatch test.

        Checked at pack time so we only pre-pack filters we might actually use.
        The dynamic half -- the post-cache-prepend padding and D_out, which
        depend on the input -- is checked in forward().
        """
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

        # Framework guard first, per PR #148's pattern: honours the fleet-wide
        # VLLM_NEURON_DISABLE_NKI_KERNELS kill switch and VLLM_NEURON_CPU_MODE.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return False
        if tuple(self.kernel_size) != (3, 3, 3):
            return False
        if tuple(self.stride) != (1, 1, 1):
            return False
        if tuple(self.dilation) != (1, 1, 1):
            return False
        if self.groups != 1:
            return False
        if self.bias is None:
            return False
        # Spatial padding must be exactly 1 so the effective kernel padding is
        # the single static config the traced kernels were built for.
        if self._padding[0] != 1 or self._padding[2] != 1:
            return False
        # Either a fat conv (generic conv3d wins) or a narrow output conv that
        # the temporal-unroll variant can rescue. Everything between is where
        # the generic kernel measured *slower* than the compiler.
        min_ch = _VAE_NKI_CONV_MIN_CHANNELS
        wide = self.in_channels >= min_ch and self.out_channels >= min_ch
        narrow = self.out_channels <= 32
        return wide or narrow

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[K_d, K_h, K_w, C_in, C_out]``.

        Static half of the gate only; the dynamic half stays in ``forward()``, so a
        packed site can still take the compiler path. Returns True if packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 4, 1, 0))
        return True

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        nki_out = self._maybe_nki_forward(x, padding)
        if nki_out is not None:
            return nki_out

        x = F.pad(x, padding)
        return super().forward(x)

    def _maybe_nki_forward(self, x, padding):
        """Run an NKI conv kernel if this exact convolution is a measured win.

        Returns None to fall through to the compiler. Every condition here is
        resolvable at trace time (module attrs + static shapes), so the branch
        is decided during tracing and only one path is ever compiled.
        """
        if self._nki_packed_filters is None:
            return None
        # The cache prepend must have consumed all temporal padding, leaving the
        # (0, 0, 1, 1, 1, 1) config the kernels were traced with.
        if padding[4] != 0 or padding[5] != 0:
            return None

        d_out = x.shape[2] - self.kernel_size[0] + 1
        w_out = x.shape[4]
        if d_out < _VAE_NKI_CONV_MIN_D_OUT:
            return None

        x_in = x.to(self._nki_packed_filters.dtype)
        bias = self.bias.to(self._nki_packed_filters.dtype)

        if self.out_channels <= 32:
            # Generic conv3d is ~0.41x here (PE-starved: C_out sits on the PSUM
            # partition dim). The temporal-unroll variant stacks D_out positions
            # into one matmul and is the only profitable option -- and only when
            # its own gate agrees.
            if not should_use_temporal_unroll(
                self.out_channels, d_out, self.in_channels, self.kernel_size[0], w_out
            ):
                return None
            return _vae_nki_conv3d_temporal_unroll(x_in, self._nki_packed_filters, bias)

        return _vae_nki_conv3d(x_in, self._nki_packed_filters, bias)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.

    Neuron changes:
      - Replaced F.normalize with manual x * rsqrt(mean(x^2)) to eliminate
        NaN-guard CROSS_LANE_REDUCE synchronization (~67us per call).
    """

    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        # Manual RMS norm avoids F.normalize's NaN check (compare + select)
        # and its generic L2 path which generates expensive CROSS_LANE_REDUCE.
        # This is mathematically equivalent: F.normalize(x, dim) * scale
        #   = x / ||x||_2 * sqrt(D)
        #   = x / sqrt(mean(x^2))
        #   = x * rsqrt(mean(x^2))
        rms = x.to(torch.float32).pow(2).mean(dim, keepdim=True).clamp(min=1e-8).rsqrt()
        return (x * rms).to(self.gamma.dtype) * self.gamma + self.bias


class WanUpsample(nn.Module):
    r"""
    Perform 2x spatial upsampling via repeat_interleave.

    This avoids nn.Upsample(nearest-exact) which lowers to thousands of
    indirect DMA gather instructions on Neuron. repeat_interleave maps
    to efficient bulk tensor operations instead.

    Args:
        scale_factor: Spatial scale factor (tuple of 2 floats, e.g. (2.0, 2.0)).
        mode: Ignored, kept for API compatibility.

    Neuron changes:
      - Replaced nn.Upsample(nearest-exact) with repeat_interleave.
        nn.Upsample lowers to thousands of indirect DMA gathers on Neuron.
        repeat_interleave maps to efficient bulk tensor copies.
      - Removed the float32 cast (diffusers' WanUpsample does .float() then
        .type_as()) that doubled memory traffic.
    """

    def __init__(self, scale_factor=None, mode=None):
        super().__init__()
        if isinstance(scale_factor, (tuple, list)):
            self.scale_h = int(scale_factor[0])
            self.scale_w = int(scale_factor[1])
        else:
            self.scale_h = int(scale_factor)
            self.scale_w = int(scale_factor)

    def forward(self, x):
        return x.repeat_interleave(self.scale_h, dim=-2).repeat_interleave(self.scale_w, dim=-1)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.

    Neuron changes:
      - Added first_chunk param to forward(). For upsample3d mode, when
        first_chunk=True, skips time_conv and leaves zero cache untouched.
        This replaces diffusers' None->"Rep"->tensor state machine which
        is incompatible with torch.compile (no None/string cache values).
      - Replaced "feat_cache[idx] != 'Rep'" check with "not first_chunk".
      - Removed .to(cache_x.device) calls.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                # stays nn.Conv2d: encoder-only (the decoder uses upsample_mode),
                # so no decode test can exercise an NKI variant here. conv3d does
                # accept stride as a kwarg -- this is an unverifiability call, not
                # a kernel limitation.
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                # stays nn.Conv2d: encoder-only (the decoder uses upsample_mode),
                # so no decode test can exercise an NKI variant here. conv3d does
                # accept stride as a kwarg -- this is an unverifiability call, not
                # a kernel limitation.
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change 6: skip time_conv on first frame to match diffusers'
                    # None→"Rep" path. The zero-initialized cache is left untouched so
                    # that frame 1 sees time_conv(x, zeros) ≡ time_conv(x) (both produce
                    # [0,0,x] as effective conv input via causal zero-padding).
                    feat_idx[0] += 1
                else:
                    cache_x = _tail_cache(x)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not first_chunk:
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        # Replaces diffusers' "feat_cache[idx] != 'Rep'" check.
                        # With zero-init cache, first_chunk=False after frame 0 means
                        # the cache holds real data from the previous frame.
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if feat_cache[idx] is None:
                        # if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    # torch.stack over two advanced-indexed (non-contiguous) slices: the same
                    # pattern that raised RuntimeError: Expected self.is_contiguous() on the real
                    # device elsewhere in this fleet's code (neighborhood_attention's _window_axis,
                    # fixed the same way) -- force each slice contiguous before stacking.
                    x = torch.stack(
                        (x[:, 0, :, :, :, :].contiguous(), x[:, 1, :, :, :, :].contiguous()), 3
                    )
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change: on the first temporal chunk diffusers stores the
                    # full spatially-downsampled x as the cache and returns x unchanged
                    # (no time_conv). Later chunks only ever read the last frame
                    # (feat_cache[idx][:, :, -1:, :, :]), so store just that — a fixed
                    # [B, C, 1, H', W'] tensor — to keep the cache compile-stable.
                    # Replaces diffusers' "feat_cache[idx] is None" sentinel.
                    feat_cache[idx] = x[:, :, -1:, :, :].clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


class WanAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.norm = WanRMS_norm(dim)
        # stay nn.Conv2d: measured -0.0255 s marginal in the shipped context,
        # i.e. no benefit. Only ~0.7 GMAC combined -- too little work to amortise
        # a compiler<->NKI graph boundary, even though both sites flank the NKI
        # wan_vae::attention_cte kernel.
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x

        B, C, T, H, W = x.size()
        S = H * W

        # Diffusers Wan VAE attention layout:
        # apply attention independently per frame over spatial tokens.
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.norm(x)

        qkv = self.to_qkv(x)

        # Single-head layout:
        # q/k/v: (B*T, S, C)
        qkv = qkv.reshape(B * T, 3, C, S)
        # Single swap at a time (dim 0<->1, then dim 2<->3), not one combined permute([1,0,3,2]):
        # same reasoning as patchify/AvgDown3D/DupUp3D's adjacent-transpose-chain fix elsewhere in
        # this file -- a DramToDramTranspose (NCC_IDDT901) traced to the Cosmos3-Nano VAE encode
        # geometry (base_dim 160, 640x640) persisted
        # after those fixes, and this permute (already followed by .contiguous(), but combining two
        # independent axis swaps in one call) was the next candidate on this exact code path.
        qkv = qkv.transpose(0, 1).transpose(2, 3).contiguous()
        q, k, v = qkv.unbind(0)

        if x.device.type == "neuron" and not _vae_attn_can_use_nki(C):
            # No NKI (NeuronCore-v2: Inf2/Trn1), or a channel width attention_cte cannot take
            # (Wan2.2-TI2V-5B mid blocks: 640 / 1024 > its 512 head-dim limit): explicit
            # fp32-softmax attention that torch.compile lowers (single head, one frame's spatial
            # tokens per batch row).
            out = _vae_attention_explicit(q, k, v, self.scale)
        elif x.device.type == "neuron":
            # Keep scale based on the real Wan attention dim C.
            q = (q * self.scale).contiguous()
            k = k.contiguous()
            v = v.contiguous()

            # Existing attention_cte compile workaround:
            # full/no_tiling hits q/k/v = (1, 6240, 384), which fails in the
            # d=384 flash-section path. Padding D to 512 preserves QK math because
            # appended Q/K dims are zero, and we slice the extra V/output dims away.
            # TODO: Follow up with attention_cte kernel-side fix for long-sequence d=384
            # so this padding workaround can be removed.
            D_orig = q.shape[-1]
            padded_d = False

            if _should_pad_vae_attn_head_dim(q.shape[1], D_orig):
                pad_d = _VAE_ATTN_PAD_HEAD_DIM - D_orig
                q = F.pad(q, (0, pad_d)).contiguous()
                k = F.pad(k, (0, pad_d)).contiguous()
                v = F.pad(v, (0, pad_d)).contiguous()
                padded_d = True

            out = _vae_nki_attn(q, k, v)

            if padded_d:
                out = out[..., :D_orig]
        else:
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
            ).squeeze(1)

        x = out.permute(0, 2, 1).reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        # First residual block
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
                # TODO: remove clone() after torch-native fix for operator fusion precision loss in attention→resnet path
                # Workaround: prevent compiler from fusing attention output
                # directly into the next resnet. Without this barrier, the
                # torch-native backend's operator fusion keeps the attention
                # output in an internal buffer at reduced precision, causing
                # 2-5x error amplification through the subsequent RMS norm.
                x = x.clone()

            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class WanResidualDownBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            # Neuron change: thread first_chunk to WanResample so downsample3d can
            # use the explicit first-chunk cache path (replaces None sentinel).
            x = self.downsampler(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i]
                        if i != len(dim_mult) - 1
                        else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        # first_chunk drives the temporal-downsample cache path, so it is threaded
        # only to layers that accept it: WanResidualDownBlock (wan2.2 residual path)
        # and WanResample (wan2.1 non-residual downsample3d). Plain WanResidualBlock
        # / WanAttentionBlock layers are frame-local and take no first_chunk arg.
        for layer in self.down_blocks:
            if feat_cache is not None:
                if isinstance(layer, (WanResidualDownBlock, WanResample)):
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
                else:
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsampler() so WanResample can skip
        time_conv on the first frame. Diffusers does not pass first_chunk
        to the upsampler (relies on None/"Rep" cache state instead).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                # Neuron change 6: thread first_chunk to WanResample so it can
                # skip time_conv on the first frame (replaces diffusers' None/"Rep" state)
                x = self.upsampler(
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsamplers[0]() so WanResample can skip
        time_conv on the first frame.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                # Neuron change 6: thread first_chunk to WanResample
                x = self.upsamplers[0](
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class NeuronWanDecoder3d(nn.Module):
    """Wrapper that runs post_quant_conv + decoder in a single compiled graph.

    Flattens feat_cache into *args for torch.compile compatibility (compiled
    graphs cannot mutate Python list inputs). Returns updated cache as
    explicit outputs.
    """

    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def install_nki_conv_dispatch(self, target_device=None) -> int:
        """Pack filters for every NKI-capable submodule. Returns the number packed.

        Call after weights load and before ``torch.compile`` traces this wrapper.
        """
        packed = 0

        # Name must differ from the per-site install_nki_dispatch: apply() invokes
        # the callback on self as well as on children, so a shared name recurses.
        def _install(module: nn.Module) -> None:
            nonlocal packed
            if hasattr(module, "install_nki_dispatch"):
                packed += module.install_nki_dispatch(target_device)

        self.apply(_install)
        return packed

    def forward(self, z_frame, *flat_cache, first_chunk):
        x = self.post_quant_conv(z_frame)
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — don't rely on mutable default
        out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        return out, *feat_cache


class NeuronWanEncoder3d(nn.Module):
    """Wrapper that runs encoder + quant_conv in a single compiled graph.

    Encoder analogue of :class:`NeuronWanDecoder3d`. Flattens the encoder
    feat_cache into ``*args`` for torch.compile (compiled graphs cannot mutate
    Python list inputs) and returns the updated cache as explicit outputs.

    ``quant_conv`` is a 1x1x1 WanCausalConv3d (no temporal receptive field), so
    applying it per temporal chunk and concatenating the results is identical to
    applying it once over the whole latent — the same per-chunk factoring the
    decoder wrapper uses for ``post_quant_conv``.

    The encoder is called on two input temporal sizes (chunk 0 has T_in=1 with
    first_chunk=True; later chunks have T_in=4 with first_chunk=False), so
    torch.compile produces two specializations sharing one fixed-shape cache set.
    """

    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, x_chunk, *flat_cache, first_chunk, patch_size=None):
        # Wan2.2-TI2V-5B VAE: pixel -> patch layout (a per-frame space-to-depth) as part of the
        # compiled graph, so the whole encode stays on the NeuronCore. Purely spatial, so doing
        # it per temporal chunk equals patchifying the whole clip first.
        if patch_size is not None:
            x_chunk = patchify(x_chunk, patch_size=patch_size)
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — don't rely on mutable default
        out = self.encoder(
            x_chunk, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
        )
        out = self.quant_conv(out)
        return out, *feat_cache


class NeuronAutoencoderKLWan(AutoencoderKLWan):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False
    _group_offload_block_modules = ["quant_conv", "post_quant_conv", "encoder", "decoder"]
    # keys toignore when AlignDeviceHook moves inputs/outputs between devices
    # these are shared mutable state modified in-place
    _skip_keys = ["feat_cache", "feat_idx"]

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: list[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: list[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int | None = None,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
    ) -> None:
        # Skip AutoencoderKLWan.__init__: it is itself @register_to_config-decorated, so calling it
        # with no arguments re-registers its Wan2.1 defaults over this class's config (z_dim 16,
        # patch_size None, scale_factor_spatial 8, 16-channel latents_mean/std) and builds a
        # throwaway default-size VAE. Harmless for Wan2.1 checkpoints, wrong for Wan2.2-TI2V-5B
        # (Cosmos3-Edge): decode would skip unpatchify and de-normalise with the wrong statistics.
        super(AutoencoderKLWan, self).__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )

        self.spatial_compression_ratio = scale_factor_spatial

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        # NOTE: the pipeline path sets this from OmniDiffusionConfig.vae_use_tiling via
        # vllm_omni's registry (which overwrites whatever we set here), so enabling tiling
        # for the pipeline requires vae_use_tiling=True in the stage config, not this line.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

    def _init_feat_cache(self, z):
        """
        Pre-allocate feat_cache as zero tensors matching the decoder's data flow.
        Replaces diffusers' [None]*N initialization. Zero tensors are
        numerically equivalent to None for WanCausalConv3d (both produce
        the same causal zero-padding), and are required for Neuron
        compilation which needs fixed tensor shapes (no None/string sentinels).
        """
        device, dtype, B = z.device, z.dtype, z.shape[0]
        H, W = z.shape[3], z.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a zero cache tensor for a WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            # conv1, conv2, and possibly conv_shortcut
            # conv_shortcut is 1x1 (no temporal padding) but still needs a cache slot
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_resample(resample, h, w):
            if resample.mode == "upsample3d":
                # time_conv runs at pre-upsample resolution
                _add_conv_cache(resample.time_conv, h, w)
                h *= 2
                w *= 2
            elif resample.mode == "upsample2d":
                h *= 2
                w *= 2
            return h, w

        # conv_in
        _add_conv_cache(self.decoder.conv_in, H, W)

        # mid_block: resnet0, (attn + resnet) pairs
        for resnet in self.decoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # up_blocks
        for up_block in self.decoder.up_blocks:
            if isinstance(up_block, WanResidualUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsampler is not None:
                    H, W = _process_resample(up_block.upsampler, H, W)
            elif isinstance(up_block, WanUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsamplers is not None:
                    H, W = _process_resample(up_block.upsamplers[0], H, W)

        # conv_out
        _add_conv_cache(self.decoder.conv_out, H, W)

        return cache

    def _init_enc_feat_cache(self, x):
        """Pre-allocate the ENCODER feat_cache as fixed-shape zero tensors.

        Encoder analogue of :meth:`_init_feat_cache`. Walks conv_in -> down_blocks
        -> mid_block -> conv_out in ``feat_idx`` order, so entry N is the cache for
        the Nth WanCausalConv3d slot the encoder consumes.

        Standard WanCausalConv3d slots get a ``[B, in_ch, CACHE_T, h, w]`` zero
        cache — numerically equivalent to diffusers' ``None`` (both yield causal
        zero-padding), and required for Neuron compilation (no None sentinels).
        The ``downsample3d`` time_conv is special: its reworked forward stores only
        the last post-spatial-downsample frame, so it gets a single-frame
        ``[B, dim, 1, h//2, w//2]`` slot.

        Spatial dims SHRINK by 2 at each spatial/temporal downsample — the
        mirror image of the decoder's ×2 growth.
        """
        device, dtype, B = x.device, x.dtype, x.shape[0]
        H, W = x.shape[3], x.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a CACHE_T-frame zero cache for a standard WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_downsampler(resample, h, w):
            """Add the resample's cache slot (if any) and return the shrunk (h, w).

            Both downsample2d and downsample3d halve the spatial dims (ZeroPad2d +
            stride-2 Conv2d). Only downsample3d consumes a feat_cache slot, and its
            reworked forward keeps a single post-downsample frame.
            """
            if resample.mode == "downsample3d":
                h //= 2
                w //= 2
                cache.append(
                    torch.zeros(
                        B, resample.time_conv.in_channels, 1, h, w, device=device, dtype=dtype
                    )
                )
            elif resample.mode == "downsample2d":
                h //= 2
                w //= 2
            return h, w

        # conv_in
        _add_conv_cache(self.encoder.conv_in, H, W)

        # down_blocks
        for block in self.encoder.down_blocks:
            if isinstance(block, WanResidualDownBlock):
                for resnet in block.resnets:
                    _process_resblock(resnet, H, W)
                if block.downsampler is not None:
                    H, W = _process_downsampler(block.downsampler, H, W)
            elif isinstance(block, WanResidualBlock):
                _process_resblock(block, H, W)
            elif isinstance(block, WanResample):
                H, W = _process_downsampler(block, H, W)
            # WanAttentionBlock has no WanCausalConv3d -> no cache slot.

        # mid_block: resnet0, then (attn + resnet) pairs (attn adds no conv slot)
        for resnet in self.encoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # conv_out
        _add_conv_cache(self.encoder.conv_out, H, W)

        return cache

    def prepare_nki_conv_dispatch(self, target_device) -> int:
        """Pack NKI filters before the model moves to its Neuron device."""
        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        return decoder_wrapper.install_nki_conv_dispatch(target_device)

    def compile(self, *args, compile_encoder: bool = False, **compiler_kwargs):
        """Compile the VAE decoder (always) and optionally the encoder.

        The encoder is only needed by image-conditioned pipelines (e.g. I2V,
        which VAE-encodes the conditioning frame). ``compile_encoder`` is off by
        default so the text-to-video path pays no encoder compilation cost.
        """
        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        # Pack before tracing so the permute is a graph constant, not a per-call op.
        n_packed = decoder_wrapper.install_nki_conv_dispatch()
        logger.info("packed filters for %d NKI-eligible conv3d site(s)", n_packed)

        base_options = compiler_kwargs.get("options") or {}
        base_model_name = base_options.get("model_name", "wan_vae")

        first_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_first",
            },
        }
        rest_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_rest",
            },
        }

        self._compiled_decoder_first = torch.compile(
            decoder_wrapper, *args, **first_compiler_kwargs
        )
        self._compiled_decoder_rest = torch.compile(decoder_wrapper, *args, **rest_compiler_kwargs)

        if compile_encoder:
            encoder_wrapper = NeuronWanEncoder3d(self.encoder, self.quant_conv)
            self._compiled_encoder = torch.compile(encoder_wrapper, *args, **compiler_kwargs)

    def _decode(self, z, return_dict=True):
        """Decode latent frames one at a time through the compiled decoder.
        - z is sliced directly on device (no CPU round-trip for input)
        - feat_map stays on Neuron device throughout the loop
        - Only the output frame is transferred to CPU each iteration
        """
        _, _, num_frame, height, width = z.shape

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        feat_map = self._init_feat_cache(z)

        frames_out = []
        frame_indices = [torch.tensor([i], device=z.device) for i in range(num_frame)]
        for i in range(num_frame):
            frame = torch.index_select(z, 2, frame_indices[i])
            compiled_decoder = (
                self._compiled_decoder_first if i == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(i == 0))
            out_frame = result[0]
            feat_map = list(result[1:])
            frames_out.append(out_frame.cpu())

        out = torch.cat(frames_out, dim=2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)

        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    # Optional torch.distributed group over which tiled encode/decode deals tiles round-robin
    # (vae_tiling.run_tiles). None = this rank runs every tile. Every rank in the group must call
    # the same tiled_* method with the same input shape; only rank 0 of the group gets the result.
    tile_parallel_group = None

    def _graph_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Device/dtype the compiled graphs run on (the decoder's parameters)."""
        p = next(self.decoder.parameters())
        return p.device, p.dtype

    def _tile_decode_one(self, tile_z: torch.Tensor) -> torch.Tensor:
        """Decode one fixed-shape latent tile ``[B, C, T, th, tw]`` (a HOST tensor) frame by frame
        through the two compiled decoder specializations with a per-tile feat_cache; returns the
        pixel tile on CPU. Each frame is sliced on the host and moved whole, so every graph input is
        a fresh contiguous device tensor -- an eager ``narrow``/``contiguous`` on a DEVICE tensor is
        refused by this backend (``Expected self.is_contiguous()``, smoke round 13)."""
        device, dtype = self._graph_device_dtype()
        cache_ref = torch.empty(tile_z.shape, device=device, dtype=dtype)
        feat_map = self._init_feat_cache(cache_ref)
        frames = []
        for k in range(tile_z.shape[2]):
            frame = tile_z.narrow(2, k, 1).contiguous().to(dtype).to(device)
            decoder = self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
            result = decoder(frame, *feat_map, first_chunk=(k == 0))
            feat_map = list(result[1:])
            frames.append(result[0].cpu())
        return torch.cat(frames, dim=2)

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        r"""Spatially tiled decode through the compiled Neuron decoder, on the shared fixed-shape
        tiling helper (:mod:`vllm_omni_neuron.diffusion.layers.vae_tiling`).

        Differences from diffusers' ``tiled_decode`` (and from this method's previous version):

        * **Every tile has the same shape.** diffusers lets the last tile on each axis be narrower,
          which on Neuron means a separate NEFF (a cold compile of minutes) per distinct tile shape,
          times two decoder specializations. Here the last tile is pulled back to end at the
          boundary, so exactly two graphs (first/rest frame) serve the whole grid at any resolution.
          The blend reproduces diffusers' ``blend_v``/``blend_h`` ramps and crop-to-stride exactly
          for the evenly spaced tiles; the pulled-back tile keeps only the remainder.
        * **Static slicing.** Tiles are ``narrow`` views made contiguous before the graph boundary
          -- no data-dependent ``index_select`` on the device, and every graph input is a
          contiguous base tensor (the executor refuses strided views, smoke round 11).
        * **Optional tile parallelism.** With :attr:`tile_parallel_group` set, tiles deal
          round-robin across that group's ranks and gather to its rank 0; other ranks return ``None``.
        """
        from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, merge_tiles, run_tiles

        _, _, _, height, width = z.shape
        ratio = self.spatial_compression_ratio
        p = self.config.patch_size
        out_ratio = ratio if p is None else ratio // p  # decoder output is patchified when p is set
        tile = (self.tile_sample_min_height // ratio, self.tile_sample_min_width // ratio)
        stride = (self.tile_sample_stride_height // ratio, self.tile_sample_stride_width // ratio)
        grid = TileGrid.for_axes(
            total=(height, width), tile=tile, stride=stride, out_scale=out_ratio
        )
        # Tiles are cut on the HOST (one latent-sized transfer), then each frame of each tile goes
        # to the device as a fresh contiguous tensor inside _tile_decode_one. Slicing a device
        # tensor eagerly (narrow/index + contiguous) is what this backend refuses (round 13).
        zp = grid.pad_input(z.detach().cpu())
        tiles = run_tiles(
            grid,
            lambda n, idx: self._tile_decode_one(grid.slice_input(zp, idx)),
            group=self.tile_parallel_group,
        )
        if tiles is None:  # a non-root rank of the tile-parallel group
            return (None,) if not return_dict else DecoderOutput(sample=None)
        dec = merge_tiles(tiles, grid)
        if p is not None:
            dec = unpatchify(dec, patch_size=p)
        dec = torch.clamp(dec, min=-1.0, max=1.0)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z, return_dict=True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    # ------------------------------------------------------------------
    # Encoder (image/video -> latent), compiled analogue of the decoder.
    # Mirrors diffusers' AutoencoderKLWan._encode / tiled_encode but routes
    # every temporal chunk through the compiled NeuronWanEncoder3d graph with a
    # fixed-shape feat_cache (see _init_enc_feat_cache), instead of diffusers'
    # None/"Rep" mutable-state machine which torch.compile cannot trace.
    # ------------------------------------------------------------------
    def _encoder_module(self):
        """Return the compiled encoder or an uncompiled wrapper of the same modules."""
        compiled = getattr(self, "_compiled_encoder", None)
        if compiled is not None:
            return compiled
        if getattr(self, "_eager_encoder", None) is None:
            self._eager_encoder = NeuronWanEncoder3d(self.encoder, self.quant_conv)
        return self._eager_encoder

    def _encode(self, x):
        """Encode pixel frames to latent chunks through the Neuron encoder graph.

        Diffusers encodes in temporal chunks: chunk 0 is 1 pixel frame, each
        later chunk is 4 pixel frames, and every chunk emits 1 latent frame. The
        two distinct input temporal sizes (T_in=1, T_in=4) produce two compiled
        specializations that share one fixed-shape feat_cache set.
        """
        _, _, num_frame, height, width = x.shape
        p = self.config.patch_size

        tile_min_height = self.tile_sample_min_height
        tile_min_width = self.tile_sample_min_width
        if self.use_tiling and (width > tile_min_width or height > tile_min_height):
            # raw pixels in: tiles are cut in pixel space and patchify runs inside the encoder
            # graph per tile (NeuronWanEncoder3d.forward), exactly as the untiled path below.
            return self.tiled_encode(x)

        encoder = self._encoder_module()
        # Patchify (Wan2.2-TI2V-5B) runs inside the encoder graph, per chunk (see
        # NeuronWanEncoder3d.forward); the feat_cache is sized for the patchified layout.
        if p is not None:
            b, c = x.shape[:2]
            cache_ref = torch.empty(
                b, c * p * p, 1, height // p, width // p, device=x.device, dtype=x.dtype
            )
        else:
            cache_ref = x
        feat_map = self._init_enc_feat_cache(cache_ref)

        iter_ = 1 + (num_frame - 1) // 4
        enc_chunks = []
        for i in range(iter_):
            if i == 0:
                idx = torch.arange(0, 1, device=x.device)
            else:
                idx = torch.arange(1 + 4 * (i - 1), 1 + 4 * i, device=x.device)
            chunk = torch.index_select(x, 2, idx)
            result = encoder(chunk, *feat_map, first_chunk=(i == 0), patch_size=p)
            enc_chunks.append(result[0])
            feat_map = list(result[1:])

        enc = torch.cat(enc_chunks, dim=2)
        return enc

    @apply_forward_hook
    def encode(self, x, return_dict=True):
        """Encode a batch of images/videos into latents.

        Returns the same AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)
        shape as diffusers, so callers (e.g. I2V ``retrieve_latents(vae.encode(...),
        "argmax")``) work unchanged.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _tile_encode_one(self, tile_x: torch.Tensor) -> torch.Tensor:
        """Encode one fixed-shape pixel tile ``[B, C, T, th, tw]`` in diffusers' temporal chunks
        (1 frame, then 4 at a time) through the encoder graph with a per-tile feat_cache; returns
        the latent tile (the pre-split ``h``) on CPU. ``tile_x`` is a HOST tensor; chunks are sliced
        on the host and moved whole (see :meth:`_tile_decode_one`)."""
        device, dtype = self._graph_device_dtype()
        p = self.config.patch_size
        b, c, _, th, tw = tile_x.shape
        if (
            p is not None
        ):  # patchify runs inside the encoder graph; the cache is sized post-patchify
            cache_ref = torch.empty(b, c * p * p, 1, th // p, tw // p, device=device, dtype=dtype)
        else:
            cache_ref = torch.empty(b, c, 1, th, tw, device=device, dtype=dtype)
        feat_map = self._init_enc_feat_cache(cache_ref)
        encoder = self._encoder_module()
        chunks = []
        for k in range(1 + (tile_x.shape[2] - 1) // 4):
            start, length = (0, 1) if k == 0 else (1 + 4 * (k - 1), 4)
            chunk = tile_x.narrow(2, start, length).contiguous().to(dtype).to(device)
            result = encoder(chunk, *feat_map, first_chunk=(k == 0), patch_size=p)
            chunks.append(result[0].cpu())
            feat_map = list(result[1:])
        return torch.cat(chunks, dim=2)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Spatially tiled encode through the compiled encoder: the encode analogue of
        :meth:`tiled_decode` on the same fixed-shape helper (pixel tiles in, latent tiles out via
        ``in_scale``), so ONE encoder graph pair (first / steady chunk) serves every resolution and
        the tile size can sit below the single-graph compile threshold
        (Wan2.2-5B encoder on Trn2: 192 px OK, 208 px fails at the real width; set
        ``tile_sample_min_*`` / ``tile_sample_stride_*`` accordingly). ``x`` is raw pixels: patchify
        (when configured) runs inside the encoder graph per tile. Returns the latent ``h`` fed to
        ``DiagonalGaussianDistribution`` (``None`` on a non-root tile-parallel rank)."""
        from vllm_omni_neuron.diffusion.layers.vae_tiling import TileGrid, merge_tiles, run_tiles

        _, _, _, height, width = x.shape
        tile = (self.tile_sample_min_height, self.tile_sample_min_width)
        stride = (self.tile_sample_stride_height, self.tile_sample_stride_width)
        grid = TileGrid.for_axes(
            total=(height, width), tile=tile, stride=stride, in_scale=self.spatial_compression_ratio
        )
        xp = grid.pad_input(x.detach().cpu())  # host-side tiling, see tiled_decode
        tiles = run_tiles(
            grid,
            lambda n, idx: self._tile_encode_one(grid.slice_input(xp, idx)),
            group=self.tile_parallel_group,
        )
        if tiles is None:
            return None
        return merge_tiles(tiles, grid)


class DistributedAutoencoderKLWan(NeuronAutoencoderKLWan, DistributedVaeMixin):
    r"""
    Patch-parallel Neuron Wan VAE.

    Adds the vllm-omni distributed-VAE executor (see upstream
    ``vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan``) on top
    of :class:`NeuronAutoencoderKLWan`. When patch parallelism is enabled and
    the latent is large enough to tile, the spatial tiles produced by
    ``tile_split`` are distributed across the VAE-parallel process group; each
    rank decodes its assigned tiles and the results are gathered and blended on
    that group's rank 0 by ``tile_merge``.

    Neuron changes vs. the upstream ``DistributedAutoencoderKLWan``:
      - ``tile_exec`` runs each tile's frames through the compiled decoder
        (``self._compiled_decoder_first`` or ``self._compiled_decoder_rest``)
        with an explicit per-tile ``feat_cache`` allocated by
        ``_init_feat_cache`` — the same fixed-shape, no-None/-string cache path
        the non-distributed Neuron ``_decode`` uses — instead of the diffusers
        ``self._feat_map`` / ``self._conv_idx`` mutable-state machine, which
        is incompatible with torch.compile.
      - Per-tile frame slices use ``torch.index_select`` to match the rest of
        the Neuron decode path.
      - ``init_distributed`` accepts an explicit process group so the executor's
        collectives run over just the VAE-parallel ranks (a subgroup), rather
        than the whole DiT group as upstream does.
      - Shape statistics and integer tile metadata stay on CPU so the composite
        process group routes them over Gloo. Under Lite, decoded tiles are packed
        into a contiguous CPU buffer and transferred once to the requested Neuron
        device before the bounded compiled collective; rank 0 then reassembles
        contiguous CPU buffers before Omni creates cropped views.

    Collective-participation contract:
        When patch parallelism is active (``is_distributed_enabled()`` is True),
        ``tiled_decode`` runs ``gather`` / ``all_reduce`` / ``broadcast`` on the
        executor's process group. **Every** rank in that group MUST call
        ``decode`` in lockstep,
        or the group deadlocks. The pipeline builds this VAE (and calls
        ``decode``) only on the ranks that belong to the VAE-parallel subgroup;
        with ``broadcast_result=False`` only the subgroup's rank 0 receives the
        merged frames.
    """

    def init_distributed(self, group=None):
        """Create the distributed-VAE executor, optionally bound to ``group``.

        ``init_distributed`` is the only place ``distributed_executor`` is created,
        and both ``set_parallel_size`` and ``is_distributed_enabled`` dereference
        it, so it must be called before any decode.

        The upstream mixin binds the executor to the full DiT group. When
        ``group`` is provided we rebind it to that subgroup so the executor's
        collectives (gather/all_reduce/broadcast) run over only the VAE-parallel
        ranks; the remaining ranks never build this VAE and skip decode entirely.
        Sizing the subgroup to exactly the parallel degree also makes the
        executor's ``world_size == parallel_size``, so every participating rank
        is assigned a task bucket (no out-of-range rank) and the merged result
        lands on the subgroup's rank 0.
        """
        self.distributed_executor = _NeuronDistributedVaeExecutor()
        if group is not None:
            self.distributed_executor.group = group
            self.distributed_executor.world_size = torch.distributed.get_world_size(group)
            self.distributed_executor.rank = torch.distributed.get_rank(group)
        check_vae_group_covers_world(
            self.distributed_executor.world_size,
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else None,
        )

    def set_parallel_size(self, pp_size: int, mode: str = "tile") -> None:
        """Override to always use the executor's world_size as parallel size.

        The upstream executor builds an ``assigned`` list with exactly
        ``pp_size`` buckets and indexes it by ``self.rank``. If
        ``pp_size < world_size``, high-rank members get IndexError.
        Always use ``world_size`` (which equals the Neuron-valid subgroup
        size set in init_distributed) so every rank gets a bucket; surplus
        ranks receive no tiles but still join collectives.

        vllm-omni 0.24 passes ``mode`` (the ``vae_parallel_mode`` from the config,
        forwarded by registry.py); accept and forward it to the executor so the
        parallel mode is preserved.
        """
        super().set_parallel_size(self.distributed_executor.world_size, mode=mode)

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = (
                self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            )
            blend_width = (
                self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
            )
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        # Every tile is full size: the last tile on each axis is pulled back to end at the frame
        # edge (vae_tiling.tile_starts), as the single-process tiled_decode does. diffusers' grid
        # (range(0, H, stride)) leaves thin edge tiles -- 2 latent rows / 4 columns at 256/224/192
        # on a 30x52 latent -- which compile to their own decoder graphs and decoded with blocky
        # right/bottom-edge noise on trn2 (Cosmos3 32-core decode, frames >= 17). A latent smaller
        # than one tile is one tile of the latent's own size.
        from vllm_omni_neuron.diffusion.layers.vae_tiling import tile_starts

        latent_row_starts = tile_starts(height, tile_latent_min_height, tile_latent_stride_height)
        latent_col_starts = tile_starts(width, tile_latent_min_width, tile_latent_stride_width)
        out_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            out_ratio //= self.config.patch_size  # decoder output is patchified

        tiletask_list = []
        for row, i in enumerate(latent_row_starts):
            for column, j in enumerate(latent_col_starts):
                hi = min(height, i + tile_latent_min_height)
                wj = min(width, j + tile_latent_min_width)
                h_indices = torch.arange(i, hi, device=z.device)
                w_indices = torch.arange(j, wj, device=z.device)
                tile_z = torch.index_select(z, 3, h_indices)
                tile_z = torch.index_select(tile_z, 4, w_indices)
                # Slice each temporal frame; the compiled decoder consumes one
                # frame at a time (feat_cache carries temporal state across them).
                time_list = []
                frame_indices = [torch.tensor([k], device=z.device) for k in range(num_frames)]
                for k in range(num_frames):
                    time_list.append(torch.index_select(tile_z, 2, frame_indices[k]))
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (row, column),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
            # tile starts in decoder-output units, for the start-aware blend
            "row_starts": tuple(i * out_ratio for i in latent_row_starts),
            "col_starts": tuple(j * out_ratio for j in latent_col_starts),
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(
                tiletask_list[-1].grid_coord[0] + 1,
                tiletask_list[-1].grid_coord[1] + 1,
            ),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space via the compiled decoder.

        Each tile gets its own feat_cache (sized to the tile's spatial dims by
        _init_feat_cache) so the causal-conv temporal state stays correct across
        the tile's frames without cross-tile contamination.
        """
        # feat_cache shapes follow the tile's spatial dimensions.
        feat_map = self._init_feat_cache(task.tensor[0])
        time = []
        for k in range(len(task.tensor)):
            frame = task.tensor[k]
            compiled_decoder = (
                self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(k == 0))
            self.distributed_executor._await_device_tensor(result[0])
            time.append(result[0])
            feat_map = list(result[1:])
        return self.distributed_executor.concat_tile_frames(time)

    def tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend decoded tiles into a full image on the Neuron device.

        Patchified VAEs (Wan2.2 / Cosmos3-Edge, ``patch_size`` set): tiles are decoded and blended
        in the patchified layout -- ``tile_split`` already records strides/blends in those units,
        exactly like diffusers' ``tiled_decode`` -- then the merged frame is unpatchified and
        clamped once on the output rank.
        """
        patch = self.config.patch_size
        ts = grid_spec.tile_spec
        merged = self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["sample_height"],
            full_width=ts["sample_width"],
            stride_height=ts["tile_sample_stride_height"],
            stride_width=ts["tile_sample_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=patch is None,
            # Patchified output is unpatchified on the host anyway: stream each blended chunk to
            # the host as it is made, so rank 0 never holds the whole merged video on the device.
            to_host=patch is not None,
            row_starts=ts.get("row_starts"),
            col_starts=ts.get("col_starts"),
        )
        if patch is None or merged is None:
            return merged
        # unpatchify is a view/permute chain the Neuron eager path rejects; the merged frames leave
        # the device right after this anyway, so do it on the host.
        merged = merged.to("cpu").contiguous()
        return torch.clamp(unpatchify(merged, patch_size=patch), min=-1.0, max=1.0)

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    # ------------------------------------------------------------------
    # Patch-parallel ENCODE (image/video -> latent), the encoder analogue of the
    # tiled_decode operators above. Reuses the same DistributedVaeExecutor
    # (self.distributed_decoder, bound to the VAE-parallel subgroup): split the
    # pixel input into overlapping spatial tiles, encode each assigned tile's
    # temporal chunks through the compiled encoder, gather to rank 0, and blend
    # the latent tiles. Without this, encode ran serially on rank 0 while decode
    # sharded across the group — making the I2V image encode the dominant VAE
    # cost.
    # ------------------------------------------------------------------
    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split the pixel input into overlapping spatial tiles (encoder side).

        Mirrors the non-distributed :meth:`NeuronAutoencoderKLWan.tiled_encode`
        tiling: iterate the pixel-space grid with ``tile_sample_stride`` and crop
        ``tile_sample_min`` windows. Blend/stride/crop bookkeeping is recorded in
        latent space (the ``merge`` side blends the encoded latent tiles).
        """
        _, _, num_frames, height, width = x.shape

        # Latent-space compression ratio (patchify pre-shrinks by patch_size).
        encode_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_ratio
        latent_width = width // encode_ratio
        tile_latent_min_height = self.tile_sample_min_height // encode_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list = []
        for i in range(0, height, self.tile_sample_stride_height):
            for j in range(0, width, self.tile_sample_stride_width):
                hi = min(height, i + self.tile_sample_min_height)
                wj = min(width, j + self.tile_sample_min_width)
                h_indices = torch.arange(i, hi, device=x.device)
                w_indices = torch.arange(j, wj, device=x.device)
                tile_x = torch.index_select(x, 3, h_indices)
                tile_x = torch.index_select(tile_x, 4, w_indices)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (
                            i // self.tile_sample_stride_height,
                            j // self.tile_sample_stride_width,
                        ),
                        tile_x,
                        workload=tile_x.shape[3] * tile_x.shape[4],
                    )
                )
        tile_spec = {
            "latent_height": latent_height,
            "latent_width": latent_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_latent_stride_height": tile_latent_stride_height,
            "tile_latent_stride_width": tile_latent_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(
                tiletask_list[-1].grid_coord[0] + 1,
                tiletask_list[-1].grid_coord[1] + 1,
            ),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single pixel tile into a latent tile via the compiled encoder.

        Each tile gets its own feat_cache (sized to the tile's spatial dims by
        _init_enc_feat_cache) so the causal-conv temporal state stays correct
        across the tile's temporal chunks without cross-tile contamination —
        the encoder analogue of :meth:`tile_exec`.
        """
        encoder = self._encoder_module()
        tile_x = task.tensor
        feat_map = self._init_enc_feat_cache(tile_x)

        num_frames = tile_x.shape[2]
        iter_ = 1 + (num_frames - 1) // 4
        time = []
        for k in range(iter_):
            if k == 0:
                f_idx = torch.arange(0, 1, device=tile_x.device)
            else:
                f_idx = torch.arange(1 + 4 * (k - 1), 1 + 4 * k, device=tile_x.device)
            chunk = torch.index_select(tile_x, 2, f_idx)
            result = encoder(chunk, *feat_map, first_chunk=(k == 0))
            self.distributed_executor._await_device_tensor(result[0])
            time.append(result[0])
            feat_map = list(result[1:])
        return self.distributed_executor.concat_tile_frames(time)

    def encode_tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend encoded latent tiles into the full latent on Neuron."""
        ts = grid_spec.tile_spec
        return self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["latent_height"],
            full_width=ts["latent_width"],
            stride_height=ts["tile_latent_stride_height"],
            stride_width=ts["tile_latent_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=False,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)
        # The patch-parallel split/exec below works in the patchified layout (its encoder calls pass
        # no patch_size); the base-class tiled path now takes raw pixels and patchifies in-graph.
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        # broadcast_result=True: every VAE rank calls encode inside the I2V
        # prepare_latents and continues to normalize / retrieve the latent, so
        # all group members need the merged latent (not just rank 0 as in decode).
        return self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )
