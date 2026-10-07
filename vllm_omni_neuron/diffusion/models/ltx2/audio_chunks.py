# SPDX-License-Identifier: Apache-2.0
"""Time-chunked, rank-parallel run of the LTX-2 / LTX-2.5 BWE vocoder (``LTX2VocoderWithBWE``).

The vocoder is a stack of local operators along time (dilated convolutions, anti-aliased
up/down-sampling, a causal STFT and the bandwidth extender), so an output sample depends only on a
bounded window of mel frames. The mel ``[B, C, T, n_mels]`` is cut into ``n`` contiguous spans;
each span is vocoded with ``margin`` extra frames of context on both sides and only its own span
of the output is kept (the vocoder emits a fixed number of samples per mel frame, 480 at 48 kHz).
Measured on the real LTX-2.5 vocoder (fp32, a 501-frame / 5 s mel): 4 spans, margin 32 -> output
rel-L2 7e-5 vs the single call (8 spans: 1.6e-4); margin 16 -> 5e-2. For comparison, bf16 vs fp32
of the single call is 0.18. Default margin 48.

With a ``group`` of ``n`` ranks (its own gloo group, so it can run on a worker thread next to the
video decode's collectives), rank 0 broadcasts the mel, every rank vocodes one span on its host
CPU and sends it back, and rank 0 concatenates. Rank 0 calls :meth:`__call__`, the others
:meth:`serve`. Without a group the spans run one after another (same result).
"""

from __future__ import annotations

import os

import torch

MARGIN = int(os.environ.get("LTX25_VOCODER_MARGIN", "48"))


def vocoder_spans(total: int, n: int, margin: int) -> list[tuple[int, int, int, int]]:
    """``(lo, a, b, hi)`` per span: keep frames ``[a, b)``, vocode ``[lo, hi)``."""
    n = max(1, min(n, total))
    bounds = [round(i * total / n) for i in range(n + 1)]
    return [
        (max(0, a - margin), a, b, min(total, b + margin))
        for a, b in zip(bounds, bounds[1:], strict=False)
    ]


def _vocode_span(fn, mel: torch.Tensor, span) -> torch.Tensor:
    lo, a, b, hi = span
    y = fn(mel[:, :, lo:hi])
    per_frame = y.shape[-1] // (hi - lo)
    return y[..., (a - lo) * per_frame : (b - lo) * per_frame].contiguous()


class ChunkedVocoder:
    """``vocoder_fn(mel) -> waveform`` over ``n`` time spans, optionally one span per rank."""

    _DTYPES = (torch.float32, torch.bfloat16, torch.float16)

    def __init__(self, n: int, margin: int = MARGIN, ranks: list[int] | None = None, group=None):
        self.n, self.margin = n, margin
        self.ranks, self.group = ranks, group  # global ranks of the span group, its gloo group

    def __call__(self, vocoder_fn, mel: torch.Tensor) -> torch.Tensor:
        spans = vocoder_spans(mel.shape[2], self.n, self.margin)
        if self.group is None:
            return torch.cat([_vocode_span(vocoder_fn, mel, s) for s in spans], dim=-1)
        import torch.distributed as dist

        mel = self._broadcast(mel)
        pieces = [_vocode_span(vocoder_fn, mel, spans[0])]
        for i in range(1, len(self.ranks)):
            shape = torch.zeros(4, dtype=torch.int64)
            dist.recv(shape, src=self.ranks[i], group=self.group)
            if int(shape[0]) == 0:  # this rank had no span (fewer frames than ranks)
                continue
            t = torch.empty(tuple(int(x) for x in shape[:3]), dtype=self._DTYPES[int(shape[3])])
            dist.recv(t.view(-1).view(torch.uint8), src=self.ranks[i], group=self.group)
            pieces.append(t)
        return torch.cat(pieces, dim=-1)

    def serve(self, vocoder_fn) -> None:
        """Non-zero ranks of the span group: receive the mel, vocode this rank's span, send it."""
        import torch.distributed as dist

        mel = self._broadcast(None)
        spans = vocoder_spans(mel.shape[2], self.n, self.margin)
        i = self.ranks.index(dist.get_rank())
        shape = torch.zeros(4, dtype=torch.int64)
        if i >= len(spans):
            dist.send(shape, dst=self.ranks[0], group=self.group)
            return
        y = _vocode_span(vocoder_fn, mel, spans[i])
        shape[:3] = torch.tensor(y.shape)
        shape[3] = self._DTYPES.index(y.dtype)
        dist.send(shape, dst=self.ranks[0], group=self.group)
        dist.send(y.view(-1).view(torch.uint8), dst=self.ranks[0], group=self.group)

    def _broadcast(self, mel: torch.Tensor | None) -> torch.Tensor:
        import torch.distributed as dist

        meta = torch.zeros(5, dtype=torch.int64)
        if mel is not None:
            meta[:4] = torch.tensor(mel.shape)
            meta[4] = self._DTYPES.index(mel.dtype)
        dist.broadcast(meta, src=self.ranks[0], group=self.group)
        shape, dtype = tuple(int(x) for x in meta[:4]), self._DTYPES[int(meta[4])]
        buf = mel.contiguous() if mel is not None else torch.empty(shape, dtype=dtype)
        dist.broadcast(buf.view(-1).view(torch.uint8), src=self.ranks[0], group=self.group)
        return buf
