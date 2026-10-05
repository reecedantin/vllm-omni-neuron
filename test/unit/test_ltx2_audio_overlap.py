# SPDX-License-Identifier: Apache-2.0
"""CPU test for the LTX-2.5 served pipeline's audio/video decode overlap.

``NeuronLTX25Pipeline._install_audio_overlap`` starts the audio decode (audio VAE + vocoder) on a
worker thread when the video decode starts and hands its results to the pipeline's own two calls.
The outputs must be exactly the sequential ones, the audio must run while the video decode is
still in progress, and a request without the overlap path (no audio latents) must fall through.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from vllm_omni_neuron.diffusion.models.ltx2.pipeline_ltx25 import NeuronLTX25Pipeline


class _Vocoder(torch.nn.Module):
    def forward(self, mel):
        return mel.float().sum(dim=-1) * 0.5


def _make(events):
    def video_decode(z, temb=None, return_dict=True):
        events.append("video-start")
        time.sleep(0.2)  # the device tiles
        events.append("video-end")
        return (z * 2,)

    def audio_decode(z, return_dict=True):
        events.append("audio")
        return (z + 1,)

    class _Pipe:
        @staticmethod
        def _unpack_audio_latents(x):
            return x * 3

        def __call__(self, owner, v, a):
            al = self._unpack_audio_latents(a)
            video = owner.vae.decode(v, None, return_dict=False)[0]
            mel = owner.audio_vae.decode(al.to(owner.audio_vae.dtype), return_dict=False)[0]
            return video, owner.vocoder(mel)

    p = SimpleNamespace(
        _pipe=_Pipe(),
        vae=SimpleNamespace(decode=video_decode),
        audio_vae=SimpleNamespace(decode=audio_decode, dtype=torch.float32),
        vocoder=_Vocoder(),
    )
    p._install_audio_overlap = NeuronLTX25Pipeline._install_audio_overlap.__get__(p)
    return p


def test_audio_decode_overlaps_video_decode_and_matches_sequential():
    v, a = torch.randn(1, 4, 8), torch.randn(1, 2, 16)
    seq_events: list = []
    ref = _make(seq_events)
    ref_video, ref_wav = ref._pipe(ref, v, a)
    assert seq_events == ["video-start", "video-end", "audio"]

    events: list = []
    p = _make(events)
    p._install_audio_overlap()
    for _ in range(2):  # state is per request
        events.clear()
        video, wav = p._pipe(p, v, a)
        assert torch.equal(video, ref_video) and torch.equal(wav, ref_wav)
        assert events.index("audio") < events.index("video-end"), events


def test_audio_overlap_falls_through_without_latents():
    events: list = []
    p = _make(events)
    p._install_audio_overlap()
    z = torch.randn(1, 2, 16)
    mel = p.audio_vae.decode(z, return_dict=False)[0]
    assert torch.equal(mel, z + 1)
    assert torch.equal(p.vocoder(mel), _Vocoder()(mel))
