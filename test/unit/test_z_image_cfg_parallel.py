# SPDX-License-Identifier: Apache-2.0
"""CPU parity for the CFG forward (``ZImageCFGForwardMixin``) on the tiny checkpoint.

1. Sequential CFG (``cfg_parallel_size=1``: two batch-1 DiT calls per step) vs upstream's batch-2
   CFG forward, same seed. These are two different code paths computing the same math.
2. CFG-parallel (``cfg_parallel_size=2``: one branch per rank, host all-gather over gloo, two CPU
   processes) vs sequential CFG: must be bit-identical, since every rank runs the same batch-1
   calls as the sequential path and the gather moves bytes only.

The end-to-end CPU test (``test_pipeline_end_to_end_matches_diffusers``) goes through the diffusers
pipeline loop (standalone.py), which uses upstream's inline batch-2 CFG, so only this file exercises
cfg_forward.py.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

SRC = os.environ.get(
    "Z_IMAGE_WEIGHTS", os.path.join(os.environ.get("WEIGHTS", "/nonexistent"), "z-image-turbo")
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not os.path.isdir(os.path.join(SRC, "tokenizer")):
        pytest.skip("set Z_IMAGE_WEIGHTS to a Z-Image checkout (tokenizer + scheduler)")
    from .test_z_image_tiny_ckpt import make_tiny

    return make_tiny(SRC, str(tmp_path_factory.mktemp("tiny-z-image")))


def _request_batch(prompt, negative_prompt, **sp_kw):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sp = OmniDiffusionSamplingParams(**sp_kw)
    req = OmniDiffusionRequest(
        prompt={"prompt": prompt, "negative_prompt": negative_prompt},
        sampling_params=sp,
        request_id="t",
    )
    return DiffusionRequestBatch(requests=[req])


@pytest.fixture(scope="module")
def vllm_single_rank():
    """World-size-1 vLLM-Omni distributed env, needed for get_cfg_group() at cfg_parallel_size=1.
    vllm_omni.diffusion.distributed.parallel_state keeps its OWN world-group state, separate from
    plain vLLM's -- its init_distributed_environment must be used, not vllm.distributed's."""
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(data_parallel_size=1, cfg_parallel_size=1, tensor_parallel_size=1)
    yield
    ctx.__exit__(None, None, None)


def _build_pipe(tiny: str):
    """NeuronZImagePipeline on the CPU in fp32, built without the engine (no od_config loader)."""
    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImagePipeline
    from vllm_omni_neuron.diffusion.models.z_image.standalone import build_components

    class FakeOdConfig:
        model = tiny
        dtype = torch.float32
        model_config: dict = {}
        revision = None
        quantization_config = None
        enable_diffusion_pipeline_profiler = False

    pipe = NeuronZImagePipeline.__new__(NeuronZImagePipeline)
    torch.nn.Module.__init__(pipe)
    te, dit, vae = build_components(tiny, torch.float32)
    pipe.od_config = FakeOdConfig()
    pipe.weights_sources = []
    pipe._execution_device = torch.device("cpu")
    pipe.text_encoder, pipe.transformer, pipe.vae = te, dit, vae
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(tiny, subfolder="scheduler")
    pipe.tokenizer = AutoTokenizer.from_pretrained(tiny, subfolder="tokenizer")
    pipe.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    pipe.image_processor = VaeImageProcessor(
        vae_scale_factor=pipe.vae_scale_factor * 2, do_convert_rgb=True
    )
    pipe.setup_diffusion_pipeline_profiler(enable_diffusion_pipeline_profiler=False)

    return pipe


SP_KW = dict(height=256, width=192, num_inference_steps=3, guidance_scale=3.0, output_type="latent")
PROMPT = "a lighthouse at dusk"


def test_sequential_cfg_matches_batch2_cfg(tiny, vllm_single_rank):
    pipe = _build_pipe(tiny)
    batch = _request_batch(PROMPT, "", **SP_KW)

    with torch.no_grad():
        torch.manual_seed(1)
        got = pipe.forward(batch).output  # sequential CFG (new override)

    from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

    with torch.no_grad():
        torch.manual_seed(1)
        want = ZImagePipeline.forward(pipe, batch).output  # upstream's batch-2 CFG, same weights

    assert got.shape == want.shape
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 1e-4, rel


def test_omitted_guidance_runs_without_cfg(tiny, vllm_single_rank):
    """guidance_scale 0 (Turbo) reaches the pipeline as vLLM-Omni's 1.0 sentinel; it must run one
    DiT call per step, not a CFG pair at scale 1."""
    pipe = _build_pipe(tiny)
    calls = []
    fwd = pipe.transformer.forward

    def count(x, *a, **k):
        calls.append(len(x))
        return fwd(x, *a, **k)

    pipe.transformer.forward = count
    batch = _request_batch(PROMPT, "", **{**SP_KW, "guidance_scale": 0.0})
    assert batch.requests[0].sampling_params.guidance_scale == 1.0  # the upstream sentinel
    with torch.no_grad():
        torch.manual_seed(1)
        pipe.forward(batch)
    assert calls == [1] * SP_KW["num_inference_steps"], calls


def _cfg_rank_main(rank: int, port: int, tiny: str, out: str) -> None:
    """One rank of a 2-rank CPU CFG-parallel run (``cfg_parallel_size=2``, TP 1); rank 0 saves the latent."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_classifier_free_guidance_world_size,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        initialize_model_parallel(data_parallel_size=1, cfg_parallel_size=2, tensor_parallel_size=1)
        assert get_classifier_free_guidance_world_size() == 2
        pipe = _build_pipe(tiny)
        with torch.no_grad():
            torch.manual_seed(1)
            lat = pipe.forward(_request_batch(PROMPT, "", **SP_KW)).output
        if rank == 0:
            torch.save(lat, out)


def test_cfg_parallel_matches_sequential(tiny, vllm_single_rank, tmp_path):
    """Two CPU processes with one CFG branch each give exactly the sequential-CFG latent."""
    pipe = _build_pipe(tiny)
    with torch.no_grad():
        torch.manual_seed(1)
        want = pipe.forward(_request_batch(PROMPT, "", **SP_KW)).output  # cfg size 1 here

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = str(tmp_path / "cfg2.pt")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    )
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(r), str(port), tiny, out], env=env
        )
        for r in range(2)
    ]
    rcs = [p.wait(timeout=600) for p in procs]
    assert rcs == [0, 0], rcs
    got = torch.load(out)
    assert got.shape == want.shape
    assert torch.equal(got, want), ((got - want).norm() / want.norm()).item()


# CFG groups over 4 gloo ranks, one ascending and one descending: the shape the Trn2 physical mesh
# gives CFG groups at TP8 x CP x CFG2 (e.g. [12, 8]), where c10d's sorted order swaps the branches.
_CFG_GROUPS = [[0, 2], [3, 1]]


def _branch_order_rank_main(rank: int, init_file: str, out_dir: str) -> None:
    """One rank: the CFG-parallel branch of ``predict_noise_maybe_with_cfg`` on a hand-built CFG
    coordinator. Branch 0 predicts 1, branch 1 predicts 0, so ``pos + s * (pos - neg)`` is 1 + s
    when the gather keeps group-rank order and -s when it swaps the branches."""
    from types import SimpleNamespace

    import torch.distributed as dist
    import vllm_omni.diffusion.distributed.parallel_state as ps

    from vllm_omni_neuron.diffusion.models.z_image.cfg_forward import ZImageCFGForwardMixin

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=4)
    coord = None
    for ranks in _CFG_GROUPS:
        pg = dist.new_group(ranks)  # every rank creates every group, as GroupCoordinator does
        if rank in ranks:
            coord = SimpleNamespace(
                ranks=ranks, world_size=2, rank_in_group=ranks.index(rank), cpu_group=pg
            )
    ps.get_cfg_group = lambda: coord
    ps.get_classifier_free_guidance_rank = lambda: coord.rank_in_group
    ps.get_classifier_free_guidance_world_size = lambda: 2

    class Stub(ZImageCFGForwardMixin):
        def predict_noise(self, value):
            return torch.full((1, 2, 3, 3), float(value))

    pred = Stub.__new__(Stub).predict_noise_maybe_with_cfg(
        True, 4.0, {"value": 1.0}, {"value": 0.0}, cfg_normalize=False
    )
    torch.save({"ranks": coord.ranks, "pred": pred}, os.path.join(out_dir, f"r{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.timeout(300)
def test_cfg_parallel_gather_keeps_branch_order_on_descending_group(tmp_path):
    """The CFG gather returns [positive, negative] on every rank, also on a descending CFG group."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    )
    init = str(tmp_path / "init")
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "order", str(r), init, str(tmp_path)],
            env=env,
        )
        for r in range(4)
    ]
    rcs = [p.wait(timeout=280) for p in procs]
    assert rcs == [0, 0, 0, 0], rcs
    for r in range(4):
        res = torch.load(tmp_path / f"r{r}.pt")
        assert torch.equal(res["pred"], torch.full((1, 2, 3, 3), 5.0)), (
            res["ranks"],
            res["pred"].flatten()[0],
        )


if __name__ == "__main__" and sys.argv[1] == "order":  # a rank of the branch-order test
    _branch_order_rank_main(int(sys.argv[2]), sys.argv[3], sys.argv[4])
elif __name__ == "__main__":  # a rank of test_cfg_parallel_matches_sequential
    _cfg_rank_main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
