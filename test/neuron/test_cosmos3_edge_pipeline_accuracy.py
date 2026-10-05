# SPDX-License-Identifier: Apache-2.0
"""Tier 2 accuracy: one whole denoising step of the Cosmos3 pipeline (``num_steps=1``), no VAE decode.

The step is upstream's own ``Cosmos3OmniDiffusersPipeline.diffuse`` loop, run unchanged: the
conditional and unconditional CFG branches with their separate UND K/V caches, the guidance combine
and the flow-UniPC scheduler step. Only the transformer underneath it differs:

* baseline: upstream's ``Cosmos3VFMTransformer`` in fp32 on CPU (the reference implementation);
* expected: our ``NeuronCosmos3EdgeTransformer`` facade in bf16 on CPU, TP=1 (dtype error alone);
* actual: the same facade in bf16 on NeuronCores, tensor-parallel (``COSMOS3_TP``, default 4).

The denoised latent after the step is compared with ``assert_close_three_way``. Tier 1 checks the
towers one call at a time; this one adds the facade's host-side glue (text bucketing and padding
masks, per-branch K/V caching, RoPE tables, CFG combine, scheduler arithmetic) on top.

Env: ``COSMOS3_QWEN3_WEIGHTS`` (real Cosmos3-Nano), ``COSMOS3_TP`` (4), ``COSMOS3_STEP_TEST_GEOM``
(``1,32,32`` latent = T2I 512x512), ``COSMOS3_TEST_OUT``. Device ranks are pinned one per core from the
launch core set (``FLEET_CORE_LIST``, else ``NEURON_RT_VISIBLE_CORES``), before the Neuron runtime starts.
"""

from __future__ import annotations

import json
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from .test_cosmos3_edge_und_device import _cos, _neuron_available, _rel

pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")

WEIGHTS = os.environ.get("COSMOS3_QWEN3_WEIGHTS", "")
TP = int(os.environ.get("COSMOS3_TP", "4"))
PROMPT = "A red sports car parked on a wet city street at golden hour, photorealistic"
NEGATIVE = "blurry, low quality, distorted"
GUIDANCE = 7.0
FLOW_SHIFT = 3.0  # upstream COSMOS3_T2I_DEFAULT_FLOW_SHIFT
SEED = 1


def launch_core_list() -> list[int]:
    """Logical NeuronCore ids this process may use, from ``FLEET_CORE_LIST`` (``"32,33,34,35"``)
    or ``NEURON_RT_VISIBLE_CORES`` (``"32-35"`` / ``"32,33"``); ``[0, 1, ...]`` when neither is set."""
    spec = os.environ.get("FLEET_CORE_LIST") or os.environ.get("NEURON_RT_VISIBLE_CORES") or ""
    cores: list[int] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            cores.extend(range(lo, hi + 1))
        else:
            cores.append(int(part))
    return cores or list(range(64))


def pin_rank_core(rank: int, cores: list[int]) -> None:
    """One logical core per TP rank, set BEFORE anything initializes the Neuron runtime: with a shared
    multi-core visibility every spawned rank races for the same cores and nrt_init fails."""
    if rank >= len(cores):
        raise RuntimeError(f"rank {rank} has no core: launch core set {cores}")
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(cores[rank])
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ.pop("NEURON_VISIBLE_DEVICES", None)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_VLLM_CTX = []  # keep the config context alive: a collected context manager resets the config


def _init_vllm(rank: int, world: int, port: int) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    _VLLM_CTX.append(ctx)
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(world, 1)


def _inputs(weights: str, geom):
    """Token ids / masks for both CFG branches and the initial noise, identical in every process."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weights, subfolder="text_tokenizer", local_files_only=True)

    def enc(text):
        ids = torch.tensor([tok(text)["input_ids"]], dtype=torch.long)
        return ids, torch.ones_like(ids)

    cond_ids, cond_mask = enc(PROMPT)
    uncond_ids, uncond_mask = enc(NEGATIVE)
    t, h, w = geom
    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(1, 48, t, h, w, generator=gen, dtype=torch.float32)
    return cond_ids, cond_mask, uncond_ids, uncond_mask, latents


def _one_step(transformer, weights: str, geom, dtype) -> torch.Tensor:
    """Upstream ``diffuse`` for a single step, around ``transformer``; returns the stepped latent (CPU)."""
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor import pipeline_cosmos3 as up

    cond_ids, cond_mask, uncond_ids, uncond_mask, latents = _inputs(weights, geom)
    cfg = up.FlowUniPCMultistepScheduler.load_config(
        weights, subfolder="scheduler", local_files_only=True
    )
    scheduler = up.FlowUniPCMultistepScheduler.from_config(
        cfg, shift=1.0, use_dynamic_shifting=False, prediction_type="flow_prediction"
    )
    scheduler.set_timesteps(1, device="cpu", shift=FLOW_SHIFT)

    # The pipeline object without its __init__ (which would load tokenizer, VAE and weights): diffuse()
    # only needs the transformer, the scheduler and the bespoke (non-session) K/V-cache state.
    pipe = up.Cosmos3OmniDiffusersPipeline.__new__(up.Cosmos3OmniDiffusersPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.transformer = transformer
    pipe.scheduler = scheduler
    pipe._memory_manager = None
    pipe._bde_kv_state = None
    pipe._use_session_state = False
    pipe._cache_dit_requires_paired_cfg = False
    pipe._progress_bar_config = {"disable": True}
    t, h, w = geom
    with torch.no_grad():
        out = pipe.diffuse(
            latents=latents.to(dtype),
            timesteps=scheduler.timesteps,
            cond_ids=cond_ids,
            cond_mask=cond_mask,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
            guidance_scale=GUIDANCE,
            shared_kwargs=dict(video_shape=(t, h, w), fps=24.0),
            scheduler=scheduler,
            generator=torch.Generator().manual_seed(SEED),
        )
    return out.detach().to("cpu").float()


class _Sdpa(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v, attn_metadata=None):
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, enable_gqa=True
        )
        return out.transpose(1, 2)


def _upstream_transformer(weights: str):
    """Upstream ``Cosmos3VFMTransformer`` in fp32, loaded through upstream's own key remap."""
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor import transformer_cosmos3 as tc
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    tc._get_ulysses_state = lambda: (1, 0, None)
    tc._is_sp_active = lambda: False
    with open(os.path.join(weights, "transformer", "config.json")) as f:
        cfg = json.load(f)
    od = SimpleNamespace(
        tf_model_config=cfg,
        dtype=torch.float32,
        model_config={},
        custom_pipeline_args={"sound_gen": False},
        quantization_config=None,
    )
    tf = tc.Cosmos3VFMTransformer(od, temporal_compression_factor=4, sound_gen=False)
    for layer in tf.language_model.layers:
        layer.self_attn.attn = _Sdpa(True)
    for layer in tf.gen_layers:
        layer.cross_attention.attn = _Sdpa(False)
    state = {}
    tdir = os.path.join(weights, "transformer")
    for fn in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(tdir, fn), "pt") as f:
            for key in f.keys():
                if key.startswith("audio_"):
                    continue  # sound generation is not ported
                name = Cosmos3OmniDiffusersPipeline._remap_ckpt_key("transformer." + key)
                if name:
                    state[name[len("transformer.") :]] = f.get_tensor(key).float()
    missing, unexpected = tf.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    del state
    return tf.float().eval()


def _facade(weights: str, dtype, device):
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.pipeline_cosmos3_edge import (
        NeuronCosmos3EdgeTransformer,
    )

    tf = NeuronCosmos3EdgeTransformer(SimpleNamespace(model=weights, dtype=dtype))
    tf.load()
    return tf


def _cpu_refs(rank, weights, geom, out_path):
    """Baseline (upstream fp32) and expected (facade bf16, TP=1) in one CPU-only process."""
    os.environ["VLLM_NEURON_CPU_MODE"] = "1"
    _init_vllm(0, 1, _port())
    tf = _upstream_transformer(weights)
    base = _one_step(tf, weights, geom, torch.float32)
    del tf
    fac32 = _facade(weights, torch.float32, "cpu")
    port32 = _one_step(
        fac32, weights, geom, torch.float32
    )  # validates the port itself (should be ~1e-5)
    del fac32
    fac16 = _facade(weights, torch.bfloat16, "cpu")
    exp = _one_step(fac16, weights, geom, torch.bfloat16)
    torch.save({"baseline": base, "expected": exp, "port_fp32": port32}, out_path)


def _device_rank(rank, world, port, cores, weights, geom, out_path):
    pin_rank_core(rank, cores)
    _init_vllm(rank, world, port)
    from vllm_neuron.envs import get_compile_backend_name

    tf = _facade(weights, torch.bfloat16, "cpu")
    tf.to(torch.device("neuron", 0))
    tf.compile(get_compile_backend_name(), {})
    t0 = __import__("time").time()
    out = _one_step(tf, weights, geom, torch.bfloat16)
    first = __import__("time").time() - t0
    tf.reset_cache()
    out2 = _one_step(tf, weights, geom, torch.bfloat16)
    if rank == 0:
        torch.save({"actual": out, "actual2": out2, "first_s": first}, out_path)


def test_single_step_pipeline(tmp_path):
    if not WEIGHTS or not os.path.isdir(os.path.join(WEIGHTS, "transformer")):
        pytest.skip("set COSMOS3_QWEN3_WEIGHTS to a real Cosmos3-Nano checkout")
    from vllm_neuron.accuracy.testing import assert_close_three_way

    geom = tuple(int(x) for x in os.environ.get("COSMOS3_STEP_TEST_GEOM", "1,32,32").split(","))
    refs, dev = str(tmp_path / "refs.pt"), str(tmp_path / "dev.pt")
    cores = launch_core_list()[:TP]
    mp.spawn(_cpu_refs, args=(WEIGHTS, geom, refs), nprocs=1, join=True)
    mp.spawn(_device_rank, args=(TP, _port(), cores, WEIGHTS, geom, dev), nprocs=TP, join=True)
    r, d = torch.load(refs), torch.load(dev)
    base, exp, act = r["baseline"], r["expected"], d["actual"]
    report = {
        "weights": os.path.basename(os.path.normpath(WEIGHTS)),
        "geom": list(geom),
        "tp": TP,
        "cores": cores,
        "rel_port_fp32": _rel(r["port_fp32"], base),
        "rel_cpu_bf16": _rel(exp, base),
        "rel_dev": _rel(act, base),
        "cos_dev": _cos(act, base),
        "deterministic": bool(torch.equal(act, d["actual2"])),
        "device_first_step_s": d["first_s"],
    }
    try:
        res = assert_close_three_way(base, exp, act, name="single_step_latent")
        # ThreeWayAssertResult fields are numpy float32 (not JSON serializable): cast explicitly
        report["three_way"] = {
            "passed": True,
            "sigma_ratio": float(res.sigma_ratio),
            "bc": float(res.bc),
            "l2_ratio": float(res.l2_ratio),
            "linf_ratio": float(res.linf_ratio),
        }
    except AssertionError as exc:
        report["three_way"] = {"passed": False, "error": str(exc)[:2000]}
    out_dir = os.environ.get("COSMOS3_TEST_OUT", str(tmp_path))
    with open(os.path.join(out_dir, f"single_step_{report['weights']}_tp{TP}.json"), "w") as f:
        json.dump(
            report, f, indent=1, allow_nan=False
        )  # a NaN metric is a bug: fail, don't write NaN
    print("[single-step]", json.dumps(report))
    assert report["rel_port_fp32"] < 1e-3, (
        report
    )  # fp32 facade == upstream: the port itself is exact
    assert report["three_way"]["passed"], report
    assert report["deterministic"], report
