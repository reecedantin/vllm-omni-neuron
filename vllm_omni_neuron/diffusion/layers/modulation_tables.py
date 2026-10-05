# SPDX-License-Identifier: Apache-2.0
"""Host-precomputed AdaLN / timestep-modulation tables for fixed step schedules.

In a DiT, each block's modulation (``adaln_proj(act(temb))`` -> shift/scale/gate chunks) depends only
on the step's timestep embedding. With a fixed schedule (few-step distilled models, or any model
served at a fixed step count) every modulation output is a constant per step, so it can be computed
once on the host and fed to the device graph as an input instead of keeping the modulation weights
in HBM. For a 33B video DiT this removed ~13B params (~24 GiB bf16) from the device, and the result
is bit-identical to the device path when the host reproduces the device's rounding
(:func:`bake_linear_modulation`). Tables are only valid for the schedule they were built for:
rebuild them whenever the step count / timesteps change (the cache is keyed on the embedding bytes,
so a different timestep simply misses).

Pieces:

* :class:`ModulationTables` -- per-layer tables for a given ``temb``, cached in memory (and optionally
  on disk), from any per-layer function, from ``nn.Linear`` modules, or streamed from a checkpoint.
* :meth:`ModulationTables.device_tables` -- ONE device tensor PER LAYER. Do not index a single stacked
  tensor inside a compiled graph: that bakes the layer offset into each graph, so N-block chunks that
  should be identical each compile (and keep resident) their own NEFF.
* :class:`HostModulation` -- an ``nn.Module`` that stands in for a block's modulation projection and
  returns the current table's chunks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LayerFn = Callable[[int, torch.Tensor], torch.Tensor]


@torch.no_grad()
def bake_linear_modulation(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    act: Callable[[torch.Tensor], torch.Tensor] | None = F.silu,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``linear(act(temb))`` with the device path's rounding: act in fp32, input cast to
    ``compute_dtype``, matmul with fp32 accumulation and one rounding to ``compute_dtype``, then the
    bias added in ``compute_dtype``. Returns ``compute_dtype``."""
    x = temb.float()
    if act is not None:
        x = act(x)
    x = x.to(compute_dtype).float()
    y = (x @ weight.to(compute_dtype).float().t()).to(compute_dtype)
    if bias is not None:
        y = y + bias.to(compute_dtype)
    return y


def temb_key(temb: torch.Tensor, fingerprint: str = "") -> str:
    """Stable cache key for a timestep embedding (exact bytes, fp32) + a model fingerprint."""
    h = hashlib.sha1(fingerprint.encode())
    h.update(str(tuple(temb.shape)).encode())
    h.update(temb.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:20]


class ModulationTables:
    """Per-layer modulation tables for a timestep embedding: ``tables(temb) -> (num_layers, *out)``.

    ``layer_fn(i, temb)`` returns layer ``i``'s modulation output. ``fingerprint`` must change when the
    weights change (it is part of the disk-cache key). ``cache_dir`` enables an on-disk cache.
    """

    def __init__(
        self,
        layer_fn: LayerFn,
        num_layers: int,
        *,
        fingerprint: str = "",
        cache_dir: str | None = None,
        out_shape: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.layer_fn = layer_fn
        self.num_layers = num_layers
        self.fingerprint = fingerprint
        self.cache_dir = cache_dir or ""
        self.out_shape = out_shape
        self._mem: dict[str, torch.Tensor] = {}
        self._dev: dict[tuple[str, str], list[torch.Tensor]] = {}

    @classmethod
    def from_linears(
        cls,
        linears: Sequence[nn.Linear],
        *,
        act=F.silu,
        compute_dtype: torch.dtype = torch.bfloat16,
        **kw,
    ) -> ModulationTables:
        def layer_fn(i, temb):
            lin = linears[i]
            return bake_linear_modulation(
                temb, lin.weight, lin.bias, act=act, compute_dtype=compute_dtype
            )

        return cls(layer_fn, len(linears), **kw)

    @classmethod
    def from_checkpoint(
        cls,
        get_tensor: Callable[[str], torch.Tensor],
        prefix_fmt: str,
        num_layers: int,
        *,
        act=F.silu,
        compute_dtype: torch.dtype = torch.bfloat16,
        bias: bool = True,
        **kw,
    ) -> ModulationTables:
        """Stream each layer's ``{prefix}.weight`` / ``{prefix}.bias`` from a checkpoint (e.g. a
        safetensors ``safe_open(...).get_tensor``), one layer at a time, so the modulation weights are
        never all resident. ``prefix_fmt`` is formatted with the layer index, e.g.
        ``"transformer_blocks.{}.adaln_proj.linear"``."""

        def layer_fn(i, temb):
            p = prefix_fmt.format(i)
            w = get_tensor(f"{p}.weight")
            b = get_tensor(f"{p}.bias") if bias else None
            return bake_linear_modulation(temb, w, b, act=act, compute_dtype=compute_dtype)

        return cls(layer_fn, num_layers, **kw)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"modulation_{key}.pt") if self.cache_dir else ""

    @torch.no_grad()
    def tables(self, temb: torch.Tensor) -> torch.Tensor:
        key = temb_key(temb, self.fingerprint)
        hit = self._mem.get(key)
        if hit is not None:
            return hit
        path = self._path(key)
        if path and os.path.exists(path):
            out = torch.load(path, map_location="cpu", weights_only=True)
        else:
            t0 = time.time()
            outs = []
            for i in range(self.num_layers):
                y = self.layer_fn(i, temb)
                outs.append(self.out_shape(y) if self.out_shape is not None else y)
            out = torch.stack(outs).contiguous()
            logger.info(
                "modulation tables: %d layers %s in %.1fs",
                self.num_layers,
                tuple(out.shape),
                time.time() - t0,
            )
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                torch.save(out, tmp)
                os.replace(tmp, path)
        self._mem[key] = out
        return out

    def precompute(self, tembs: Iterable[torch.Tensor]) -> None:
        """Build (and cache) the tables for every step of a fixed schedule up front."""
        for t in tembs:
            self.tables(t)

    def device_tables(self, temb: torch.Tensor, device: torch.device | str) -> list[torch.Tensor]:
        """This step's tables as one device tensor per layer (cached per step and device)."""
        key = (temb_key(temb, self.fingerprint), str(device))
        hit = self._dev.get(key)
        if hit is None:
            hit = [t.to(device) for t in self.tables(temb).unbind(0)]
            self._dev[key] = hit
        return hit

    def clear(self) -> None:
        self._mem.clear()
        self._dev.clear()


class HostModulation(nn.Module):
    """Stands in for a block's modulation projection: ``forward(*_)`` returns ``table.chunk(chunks, -1)``.

    The caller sets ``self.table`` (this layer's table for the step, already on the device) before the
    block runs -- or passes the table to the block as an explicit graph input and calls
    :meth:`split` directly, which is the form to use inside a compiled N-block graph.
    """

    def __init__(self, chunks: int, view: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.chunks = chunks
        self.view = view
        self.table: torch.Tensor | None = None

    def split(self, table: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.view is not None:
            table = table.view(*self.view)
        return table.chunk(self.chunks, dim=-1)

    def forward(self, *_args, **_kwargs) -> tuple[torch.Tensor, ...]:
        if self.table is None:
            raise RuntimeError("HostModulation.table is not set for this step")
        return self.split(self.table)
