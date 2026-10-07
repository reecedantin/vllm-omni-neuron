# SPDX-License-Identifier: Apache-2.0
"""N-block graph splitting: run a stack of identical transformer blocks as compiled N-block graphs
that all share ONE compiled graph.

Why: a whole-DiT graph can exceed the compiler's ceiling (SBUF overflow / instruction limit at long
sequences) and costs 30-80 GB of compiler host RAM; splitting into N-block graphs bounds both. But
the split only pays off if the chunks are the *same* graph. Compiling each chunk from its own
modules (or indexing a stacked per-layer tensor at a chunk-dependent offset) bakes chunk identity
into the trace, so every chunk compiles and keeps resident its own NEFF.

:class:`BlockGraphRunner` traces one *template* block with ``torch.func.functional_call`` and passes
each chunk's weights -- and any per-layer side inputs such as host-precomputed modulation tables --
as graph INPUTS. Every full chunk therefore lowers to the identical graph (one compile, one NEFF);
only a shorter trailing chunk (``num_blocks % group_size``) adds a second graph.

    runner = BlockGraphRunner(model.blocks, group_size=5,
                              compile_fn=lambda f: torch.compile(f, backend=backend, fullgraph=True,
                                                                 dynamic=False, options=opts))
    x = runner(x, temb, rope, per_layer=[(tables[i],) for i in range(len(model.blocks))])

The default calling convention is ``block(carry, *shared_args, *layer_args, **shared_kwargs)`` where
``carry`` (a tensor or tuple of tensors) is what the block returns; pass ``block_call`` to adapt any
other signature. Run under ``torch.no_grad()`` / ``torch.inference_mode()``: with autograd on, the
first chunk (grad-free input) and the rest (grad-carrying input) are two graphs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call

Carry = torch.Tensor | tuple[torch.Tensor, ...]
BlockCall = Callable[..., Carry]


def default_block_call(
    block: Callable[..., Carry],
    carry: Carry,
    shared_args: tuple,
    layer_args: tuple,
    shared_kwargs: dict[str, Any],
) -> Carry:
    """``block`` is the template block bound to one layer's weights; call it like the module."""
    return block(carry, *shared_args, *layer_args, **shared_kwargs)


def block_signature(block: nn.Module) -> tuple:
    """Structural identity of a block: class, and every parameter/buffer's name, shape and dtype."""
    tensors = list(block.named_parameters()) + list(block.named_buffers())
    return (type(block).__qualname__,) + tuple((n, tuple(t.shape), t.dtype) for n, t in tensors)


def _weightless_copy(block: nn.Module) -> nn.Module:
    """A copy of ``block`` whose parameters/buffers live on the meta device (no storage). The traced
    template must not own any real layer's tensors: dynamo guards on parameter identity, so a template
    that IS block 0 makes the first chunk a different graph from the others."""
    memo = {}
    for p in block.parameters():
        memo[id(p)] = nn.Parameter(
            torch.empty_like(p, device="meta"), requires_grad=p.requires_grad
        )
    for b in block.buffers():
        memo[id(b)] = torch.empty_like(b, device="meta")
    return copy.deepcopy(block, memo)


def plan_groups(num_blocks: int, group_size: int) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` chunks of ``group_size`` (0 = one chunk), the last one possibly shorter."""
    if num_blocks <= 0:
        return []
    if group_size <= 0 or group_size >= num_blocks:
        return [(0, num_blocks)]
    return [(s, min(s + group_size, num_blocks)) for s in range(0, num_blocks, group_size)]


class BlockGraphRunner(nn.Module):
    """Run ``blocks`` (structurally identical) as N-block graphs sharing one compiled template graph."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        group_size: int,
        *,
        compile_fn: Callable[[Callable], Callable] | None = None,
        block_call: BlockCall = default_block_call,
    ) -> None:
        super().__init__()
        blocks = list(blocks)
        if not blocks:
            raise ValueError("BlockGraphRunner needs at least one block")
        sig = block_signature(blocks[0])
        for i, b in enumerate(blocks[1:], 1):
            if block_signature(b) != sig:
                raise ValueError(
                    f"block {i} differs structurally from block 0; split the stack where it changes"
                )
        # Not registered as submodules: the runner borrows the caller's blocks (no duplicate state_dict keys).
        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_template", _weightless_copy(blocks[0]))
        self.group_size = group_size
        self.block_call = block_call
        self.groups = plan_groups(len(blocks), group_size)
        self._group_fn = compile_fn(self._run_group) if compile_fn is not None else self._run_group
        self.refresh_weights()

    def refresh_weights(self) -> None:
        """Re-read each block's tensors (call after moving the blocks to a device or reloading weights)."""
        self._weights = [
            {**dict(b.named_parameters()), **dict(b.named_buffers())} for b in self._blocks
        ]

    @property
    def num_graphs(self) -> int:
        """Distinct graphs this split needs: 1, or 2 when the last chunk is shorter."""
        return len({e - s for s, e in self.groups})

    def _run_group(
        self,
        weights: list[dict[str, torch.Tensor]],
        carry: Carry,
        shared_args: tuple,
        layer_args: list[tuple],
        shared_kwargs: dict[str, Any],
    ) -> Carry:
        template = self._template
        for w, la in zip(weights, layer_args):

            def block(*args, _w=w, **kwargs):
                return functional_call(template, _w, args, kwargs, strict=True)

            carry = self.block_call(block, carry, shared_args, la, shared_kwargs)
        return carry

    def forward(
        self, carry: Carry, *shared_args, per_layer: Sequence[tuple] | None = None, **shared_kwargs
    ) -> Carry:
        n = len(self._blocks)
        if per_layer is None:
            per_layer = [()] * n
        elif len(per_layer) != n:
            raise ValueError(f"per_layer has {len(per_layer)} entries for {n} blocks")
        for s, e in self.groups:
            carry = self._group_fn(
                self._weights[s:e],
                carry,
                shared_args,
                [tuple(a) for a in per_layer[s:e]],
                shared_kwargs,
            )
        return carry
