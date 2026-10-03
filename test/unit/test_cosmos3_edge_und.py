# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Cosmos3-Edge UND tower and attention dispatch (no Neuron device needed).

The oracle is upstream vLLM-Omni's own ``Cosmos3EdgeLanguageModel`` (vendored), loaded from
the real checkpoint through upstream's own key remap, run in fp32 on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from vllm_omni_neuron import nc_generation as nc_dispatch  # noqa: E402
from vllm_omni_neuron.diffusion.models.cosmos3_edge.attention import (  # noqa: E402
    key_padding_bias,
    torch_attention,
)


def _sdpa(q, k, v, scale, causal, valid=None):
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    mask = None
    if valid is not None:
        mask = valid[:, None, None, :]
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        tri = torch.ones(sq, sk, dtype=torch.bool).tril(sk - sq)
        mask = tri if mask is None else (mask & tri)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("qblock", [0, 16])
@pytest.mark.parametrize("masked", [False, True])
def test_torch_attention_matches_sdpa(causal, qblock, masked):
    torch.manual_seed(0)
    q = torch.randn(2, 8, 40, 64)
    k = torch.randn(2, 4, 40, 64)
    v = torch.randn(2, 4, 40, 64)
    valid = None
    bias = None
    if masked:
        valid = torch.ones(2, 40, dtype=torch.bool)
        valid[1, 30:] = False
        bias = key_padding_bias(valid)
    out = torch_attention(q, k, v, 0.125, causal=causal, key_bias=bias, qblock=qblock, qk_fp32=True)
    ref = _sdpa(q, k, v, 0.125, causal, valid)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_nc_dispatch_generation(monkeypatch):
    nc_dispatch.neuron_core_generation.cache_clear()
    monkeypatch.setenv("COSMOS3_NEURON_CORE_GEN", "2")
    assert nc_dispatch.neuron_core_generation() == 2
    assert not nc_dispatch.use_nki_kernels()  # NeuronCore-v2 (Inf2/Trn1): never NKI
    nc_dispatch.neuron_core_generation.cache_clear()
    monkeypatch.setenv("COSMOS3_NEURON_CORE_GEN", "3")
    assert nc_dispatch.neuron_core_generation() == 3
    assert not nc_dispatch.use_nki_kernels(torch.zeros(1))  # CPU tensor: never NKI
    monkeypatch.setenv("COSMOS3_EDGE_ATTN_IMPL", "torch")
    assert not nc_dispatch.use_nki_kernels()
    nc_dispatch.neuron_core_generation.cache_clear()


class _SdpaCausal(torch.nn.Module):
    def forward(self, q, k, v, attn_metadata=None):
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return out.transpose(1, 2)


def _upstream_language_model(weights: str):
    """Upstream Cosmos3EdgeLanguageModel in fp32 with the checkpoint loaded via upstream's remap."""
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.transformer_cosmos3_edge import (
        Cosmos3EdgeLanguageModel,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import EdgeTextConfig

    cfg = EdgeTextConfig.from_model_dir(weights)
    lm = Cosmos3EdgeLanguageModel(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        mrope_section=cfg.mrope_section,
        use_und_k_norm_for_gen=cfg.use_und_k_norm_for_gen,
        prefix="language_model",
    )
    state = {}
    tdir = os.path.join(weights, "transformer")
    prefix = "transformer.language_model."
    for fn in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(tdir, fn), "pt") as f:
            for key in f.keys():
                name = Cosmos3OmniDiffusersPipeline._remap_ckpt_key("transformer." + key)
                if name and name.startswith(prefix):
                    state[name[len(prefix) :]] = f.get_tensor(key).float()
    # Upstream's framework Attention dispatches on the platform; for a CPU oracle use torch's
    # reference SDPA (causal, GQA) with upstream's [B, S, H, D] layout instead.
    for layer in lm.layers:
        layer.self_attn.attn = _SdpaCausal()
    missing, unexpected = lm.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in m for m in missing), missing
    return lm.float().eval()


def test_und_matches_upstream_cpu(vllm_single_rank, edge_weights):
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import (
        EdgeTextConfig,
        NeuronCosmos3EdgeUND,
    )

    cfg = EdgeTextConfig.from_model_dir(edge_weights)
    ours = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    ours.load_weights(edge_weights, "cpu")
    ref = _upstream_language_model(edge_weights)

    torch.manual_seed(0)
    real, bucket = 23, 32
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1

    cos, sin = ours.rope_tables(mask)
    with torch.no_grad():
        out = ours(ids, cos, sin)
        ref_kv = ref(ids[:, :real], (cos[:, :real], sin[:, :real]))
    n = cfg.num_layers
    for i, (k_ref, v_ref) in enumerate(ref_kv):
        torch.testing.assert_close(out[i][:, :real], k_ref, rtol=2e-4, atol=2e-4, msg=f"K layer {i}")
        torch.testing.assert_close(out[n + i][:, :real], v_ref, rtol=2e-4, atol=2e-4, msg=f"V layer {i}")
