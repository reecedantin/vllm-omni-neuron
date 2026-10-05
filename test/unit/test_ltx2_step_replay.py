# SPDX-License-Identifier: Apache-2.0
"""CPU check of the teacher-forced step check (test/neuron/test_ltx2_fullsize_gate_device.py
``dump_transformer_steps`` + ``run_replay``) on the random-weight tiny LTX-2.5 transformer: the
Neuron transformer code run in fp32 on the CPU stands in for the device; its dumped steps replay
through the reference transformer inside the bar, and a corrupted dump fails."""

import json
import os

import torch

from test.neuron import test_ltx2_fullsize_gate_device as gate

from .test_ltx2_transformer import _cfg, make_inputs, tiny_dir  # noqa: F401


def _dump(tiny, out, steps=(0, 2)):
    from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import NeuronLTX2Transformer

    m = NeuronLTX2Transformer.from_dir(os.path.join(tiny, "transformer"), dtype=torch.float32)
    gate.dump_transformer_steps(m, str(out), steps)
    cfg = _cfg(tiny)
    with torch.no_grad():
        for i, sigma in enumerate((0.9, 0.7, 0.5)):
            m(**make_inputs(cfg, batch=1, seed=i, sigma=sigma), return_dict=False)
    return sorted(p.name for p in out.glob("step_*.pt"))


def test_replay_kwargs_keep_fp32_inputs():
    kw = {
        "hidden_states": torch.zeros(1, 2, dtype=torch.bfloat16),
        "video_coords": torch.full((1, 3, 2, 2), 1537.0),
        "timestep": torch.tensor([700.0]),
        "encoder_attention_mask": torch.ones(1, 2, dtype=torch.int64),
        "num_frames": 2,
    }
    for dtype in (torch.float32, torch.bfloat16):
        r = gate.replay_kwargs(kw, dtype)
        assert r["hidden_states"].dtype == dtype
        assert (
            r["video_coords"].dtype == torch.float32
            and float(r["video_coords"][0, 0, 0, 0]) == 1537.0
        )
        assert r["timestep"].dtype == torch.float32
        assert r["encoder_attention_mask"].dtype == torch.int64 and r["num_frames"] == 2


def test_replay_passes_for_matching_steps(tiny_dir, tmp_path):  # noqa: F811
    assert _dump(tiny_dir, tmp_path) == ["step_0.pt", "step_2.pt"]
    r = gate.run_replay(tiny_dir, str(tmp_path), threads=0)
    assert r["pass"], json.dumps(r)
    assert set(r["steps"]) == {0, 2}


def test_replay_fails_for_wrong_output(tiny_dir, tmp_path):  # noqa: F811
    _dump(tiny_dir, tmp_path, steps=(1,))
    p = tmp_path / "step_1.pt"
    d = torch.load(p, weights_only=False)
    d["out"][0] = d["out"][0] + d["out"][0].std()
    torch.save(d, p)
    assert not gate.run_replay(tiny_dir, str(tmp_path), threads=0)["pass"]
