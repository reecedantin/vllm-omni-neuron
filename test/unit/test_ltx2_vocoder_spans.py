# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the time-span vocoder run (``models/ltx2/audio_chunks.py``).

A stub "vocoder" with a bounded receptive field (a 1-D convolution over mel frames, then a fixed
number of samples per frame) is reproduced exactly by the span run once the margin covers the
receptive field, both sequentially and spread over gloo ranks (rank 0 broadcasts the mel and
concatenates). The real-vocoder numbers (margin 32: rel-L2 7e-5) are in the module docstring.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp

from vllm_omni_neuron.diffusion.models.ltx2.audio_chunks import ChunkedVocoder, vocoder_spans

RF = 5  # stub receptive field: frames on each side


def _stub_vocoder(mel: torch.Tensor) -> torch.Tensor:
    """[B, C, T, M] -> [B, C, T * 4]: a (2*RF+1)-tap moving sum over frames, 4 samples per frame."""
    b, c, t, m = mel.shape
    x = mel.float().sum(-1).reshape(b * c, 1, t)
    w = torch.arange(1, 2 * RF + 2, dtype=torch.float32).view(1, 1, -1)
    y = torch.nn.functional.conv1d(x, w, padding=RF).reshape(b, c, t)
    return y.repeat_interleave(4, dim=-1) * torch.tensor([1.0, -1.0, 0.5, 2.0]).repeat(t)


@pytest.mark.parametrize("total,n", [(501, 4), (501, 8), (37, 4), (3, 4)])
def test_spans_cover_once(total, n):
    spans = vocoder_spans(total, n, 7)
    assert spans[0][1] == 0 and spans[-1][2] == total
    assert all(s[2] == t[1] for s, t in zip(spans, spans[1:], strict=False))
    assert all(lo <= a < b <= hi for lo, a, b, hi in spans)


@pytest.mark.parametrize("n,margin,exact", [(4, RF, True), (8, RF + 3, True), (4, RF - 2, False)])
def test_sequential_spans_match_single_call(n, margin, exact):
    mel = torch.randn(1, 2, 101, 8)
    full = _stub_vocoder(mel)
    out = ChunkedVocoder(n, margin=margin)(_stub_vocoder, mel)
    assert out.shape == full.shape
    assert torch.allclose(out, full, atol=1e-4) == exact


def _worker(rank, world, port, out_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    g = dist.new_group(ranks=list(range(world)), backend="gloo")
    cv = ChunkedVocoder(world, margin=RF, ranks=list(range(world)), group=g)
    if rank == 0:
        mel = torch.randn(1, 2, 64, 8, generator=torch.Generator().manual_seed(0)).bfloat16()
        torch.save({"mel": mel, "out": cv(_stub_vocoder, mel)}, out_path)
    else:
        cv.serve(_stub_vocoder)
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_rank_parallel_spans_match_sequential(tmp_path, world):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "res.pt")
    mp.start_processes(
        _worker, args=(world, port, out), nprocs=world, join=True, start_method="spawn"
    )
    r = torch.load(out)
    ref = ChunkedVocoder(world, margin=RF)(_stub_vocoder, r["mel"])
    assert torch.equal(r["out"], ref)
    assert torch.allclose(r["out"], _stub_vocoder(r["mel"]), atol=1e-3)
