# SPDX-License-Identifier: Apache-2.0
"""CPU parity: our UND + GEN towers vs upstream's full ``Cosmos3EdgeVFMTransformer.forward``.

Real checkpoint, fp32, T2I at 256x256 (16x16 latents, 64 tokens) and an I2V-style call
(conditioned first frame, 3 latent frames). Upstream trims the text to its real length; we
pad it to a bucket and mask, so this also checks that the padding is invisible.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")


class _Sdpa(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v, attn_metadata=None):
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal, enable_gqa=True)
        return out.transpose(1, 2)


@pytest.fixture(scope="module")
def upstream_tf(vllm_single_rank, edge_weights):
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor import transformer_cosmos3 as _tc

    # single process, no Ulysses sequence parallelism (omni's own SP groups are not initialized here)
    _tc._get_ulysses_state = lambda: (1, 0, None)
    _tc._is_sp_active = lambda: False

    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.transformer_cosmos3_edge import (
        Cosmos3EdgeVFMTransformer,
    )

    with open(os.path.join(edge_weights, "transformer", "config.json")) as f:
        cfg = json.load(f)
    od = SimpleNamespace(tf_model_config=cfg, dtype=torch.float32, model_config={}, custom_pipeline_args={},
                         quantization_config=None)
    tf = Cosmos3EdgeVFMTransformer(od)
    for layer in tf.language_model.layers:
        layer.self_attn.attn = _Sdpa(True)
    for layer in tf.gen_layers:
        layer.cross_attention.attn = _Sdpa(False)
    state = {}
    tdir = os.path.join(edge_weights, "transformer")
    for fn in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(tdir, fn), "pt") as f:
            for key in f.keys():
                name = Cosmos3OmniDiffusersPipeline._remap_ckpt_key("transformer." + key)
                if name:
                    state[name[len("transformer.") :]] = f.get_tensor(key).float()
    missing, unexpected = tf.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(("inv_freq" in m) or m.endswith("freqs") for m in missing), missing
    return tf.float().eval()


@pytest.fixture(scope="module")
def ours(vllm_single_rank, edge_weights):
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    cfg = EdgeGenConfig.from_model_dir(edge_weights)
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(edge_weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.float32)
    gen.load_weights(edge_weights, "cpu")
    return und, gen


@pytest.mark.parametrize("case", ["t2i", "i2v", "action"])
def test_gen_matches_upstream(case, upstream_tf, ours):
    und, gen = ours
    torch.manual_seed(1)
    real, bucket = 21, 32
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    t = 1 if case == "t2i" else 3
    h = w = 16
    lat = torch.randn(1, 48, t, h, w)
    ts = torch.tensor([637.0])
    fps = None if case == "t2i" else 24.0
    noisy = torch.ones(1, 1, t, 1, 1)
    if case != "t2i":
        noisy[:, :, 0] = 0  # first latent frame is the clean condition

    kw = {}
    s_act = 0
    if case == "action":
        s_act = 4
        act = torch.randn(1, s_act, gen.cfg.action_dim)
        act_mask = torch.ones(1, s_act, 1)
        kw = dict(action_latents=act, action_domain_ids=torch.tensor([3]), action_noisy_mask=act_mask,
                  action_start_frame_offset=1, action_fps=12.0)

    upstream_tf.reset_cache()
    with torch.no_grad():
        ref = upstream_tf(lat, ts, ids[:, :real], mask[:, :real], (t, h, w), fps=fps,
                          noisy_frame_mask=noisy if case != "t2i" else None, **kw)

        cos_u, sin_u = und.rope_tables(mask)
        kv = und(ids, cos_u, sin_u)
        cos_g, sin_g = gen.rope_tables(mask, t, h, w, fps, t_action=s_act, action_fps=12.0 if s_act else None)
        s_video = t * (h // 2) * (w // 2)
        kb = gen.key_bias(mask, s_video + s_act)
        nm = noisy[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, (h // 2) * (w // 2)).reshape(1, -1, 1)
        if case == "action":
            w_in, b_in, w_out, b_out = gen.domain_weights(3, "cpu")
            out = gen.forward_action(lat, ts, cos_g, sin_g, kb, nm, act, act_mask, w_in, b_in, w_out, b_out, *kv)
        else:
            out = gen(lat, ts, cos_g, sin_g, kb, nm, *kv)

    refs = ref if isinstance(ref, tuple) else (ref,)
    outs = out if isinstance(out, tuple) else (out,)
    for o, r in zip(outs, refs, strict=True):
        rel = ((o - r).norm() / r.norm()).item()
        assert rel < 1e-4, f"{case}: rel err {rel}"
