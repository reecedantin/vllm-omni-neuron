# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU parity of ``NeuronFlux2Pipeline`` vs diffusers ``Flux2Pipeline`` (tiny-flux2).

Same prompt, same initial latents, fp32, a few steps: the denoised latents must match. This
covers the pipeline glue the component tests do not -- prompt templating / tokenisation, the
three-layer embedding stack, RoPE ids, the scheduler's ``mu`` shift, guidance -- with TP=1 and
TP=2 (gloo ranks) for the Neuron side.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from .test_flux2_components import FLUX2_WEIGHTS, _free_port, rel, tiny_flux2  # noqa: F401

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

PROMPT = "a red apple on a wooden table, studio light"
H = W = 128
STEPS = 3
GUIDANCE = 4.0


def _latents(cfg_in_channels: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(1, cfg_in_channels, H // 16, W // 16, generator=g)


def _neuron_run(rank, world, port, weights, out_path):
    torch.set_num_threads(4)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_neuron.diffusion.models.flux2 import NeuronFlux2Pipeline

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
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
    initialize_model_parallel(tensor_parallel_size=world)
    od = OmniDiffusionConfig.from_kwargs(
        model=weights,
        dtype=torch.float32,
        model_class_name="Flux2Pipeline",
        parallel_config={"tensor_parallel_size": world},
    )
    pipe = NeuronFlux2Pipeline(od_config=od)
    pipe.load_weights()
    sp = OmniDiffusionSamplingParams(
        height=H, width=W, num_inference_steps=STEPS, guidance_scale=GUIDANCE, seed=0
    )
    req = DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=PROMPT, sampling_params=sp, request_id="t0")]
    )
    lat = _latents(pipe.transformer.cfg.in_channels)
    with torch.no_grad():
        out = pipe.forward(req, latents=lat, output_type="latent").output
    if rank == 0:
        torch.save(out, out_path)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(FLUX2_WEIGHTS, "tokenizer"))
    and not os.environ.get("FLUX2_TINY"),
    reason="needs the FLUX.2 tokenizer: set FLUX2_WEIGHTS (or FLUX2_TINY to a tiny checkout)",
)
@pytest.mark.parametrize("world", [1, 2])
def test_pipeline_matches_diffusers(world, tiny_flux2, tmp_path):  # noqa: F811
    from diffusers import Flux2Pipeline

    ref = Flux2Pipeline.from_pretrained(tiny_flux2, torch_dtype=torch.float32)
    lat = _latents(ref.transformer.config.in_channels)
    with torch.no_grad():
        packed = ref(
            prompt=PROMPT,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            latents=lat,
            output_type="latent",
        ).images
    ids = ref._prepare_latent_ids(lat)
    want = ref._unpack_latents_with_ids(packed, ids)
    mean = ref.vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(ref.vae.bn.running_var.view(1, -1, 1, 1) + ref.vae.config.batch_norm_eps)
    want = ref._unpatchify_latents(want * std + mean)

    path = tmp_path / "neuron.pt"
    mp.spawn(
        _neuron_run, args=(world, _free_port(), tiny_flux2, str(path)), nprocs=world, join=True
    )
    got = torch.load(path)
    assert got.shape == want.shape, (got.shape, want.shape)
    print(f"pipeline tp{world}: rel={rel(got, want):.3e} norm={float(want.norm()):.3f}")
    assert rel(got, want) < 1e-4, rel(got, want)
