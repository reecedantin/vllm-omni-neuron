# SPDX-License-Identifier: Apache-2.0
"""CPU end-to-end check of the Neuron HunyuanVideo-1.5 pipeline vs diffusers' reference pipeline
on the tiny checkpoint: same prompt embeddings, same denoised latents (fp32, eager, TP=1)."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

from .test_hunyuanvideo15_tiny_ckpt import tiny_ckpt  # noqa: E402,F401  (session fixture)

PROMPT = 'A cat walks on the grass, a sign reads "HELLO", realistic style.'


def spawn_with_free_port(fn, make_args, nprocs, attempts=4):
    """``mp.spawn`` with a freshly picked rendezvous port, retried when another process grabbed the
    port between picking and binding it (EADDRINUSE on a busy shared host)."""
    import torch.multiprocessing as mp

    for i in range(attempts):
        try:
            return mp.spawn(fn, args=make_args(_port()), nprocs=nprocs, join=True)
        except mp.ProcessRaisedException as e:
            if "EADDRINUSE" not in str(e) or i == attempts - 1:
                raise


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def single_rank():
    """World size 1 via vLLM-Omni's own init (TP + CFG + SP groups, as the diffusion worker does)."""
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    if not dist.is_initialized():
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(_port()),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")  # idempotent
    import vllm.distributed.parallel_state as vps

    if vps._TP is None:
        initialize_model_parallel(tensor_parallel_size=1)
    else:  # another module's fixture initialized vLLM's groups; add the omni-only ones
        import vllm_omni.diffusion.distributed.parallel_state as ops

        if ops._CFG is None:
            ops._CFG = vps._TP  # world size 1: any 1-rank group coordinator serves
    yield
    ctx.__exit__(None, None, None)


def _ours(ckpt):
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15 import NeuronHunyuanVideo15Pipeline

    od = SimpleNamespace(
        model=ckpt,
        dtype=torch.float32,
        model_config={"vae_dtype": "float32"},
        flow_shift=None,
        enable_diffusion_pipeline_profiler=False,
    )
    pipe = NeuronHunyuanVideo15Pipeline(od_config=od)
    pipe.load_weights()
    return pipe


def test_system_message_matches_diffusers(tiny_ckpt):  # noqa: F811
    import inspect

    from diffusers import HunyuanVideo15Pipeline

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import (
        SYSTEM_MESSAGE,
    )

    src = inspect.getsource(HunyuanVideo15Pipeline.__init__)
    ns: dict = {}
    body = src[src.index("self.system_message =") : src.index("# fmt: on")]
    exec(body.replace("self.", "ns_").strip().replace("ns_system_message", "x", 1), {}, ns)  # noqa: S102
    assert ns["x"] == SYSTEM_MESSAGE


def test_pipeline_latents_match_diffusers(tiny_ckpt, single_rank):  # noqa: F811
    from diffusers import HunyuanVideo15Pipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ref = HunyuanVideo15Pipeline.from_pretrained(tiny_ckpt, torch_dtype=torch.float32)
    ours = _ours(tiny_ckpt)

    h, w, f, steps = 64, 96, 9, 3
    lat0 = torch.randn(
        1, 32, (f - 1) // 4 + 1, h // 16, w // 16, generator=torch.Generator().manual_seed(7)
    )
    with torch.no_grad():
        want = ref(
            prompt=PROMPT,
            height=h,
            width=w,
            num_frames=f,
            num_inference_steps=steps,
            latents=lat0.clone(),
            output_type="latent",
        ).frames
        e_ref = ref.encode_prompt(prompt=PROMPT, device=torch.device("cpu"), dtype=torch.float32)
    e_ours = ours.encode_prompt(PROMPT, torch.device("cpu"), torch.float32)
    assert torch.equal(e_ref[0], e_ours[0]) and torch.equal(
        e_ref[2], e_ours[2]
    )  # MLLM / byT5 embeds

    sp = OmniDiffusionSamplingParams(
        height=h, width=w, num_frames=f, num_inference_steps=steps, latents=lat0.clone()
    )
    req = SimpleNamespace(prompts=[{"prompt": PROMPT}], sampling_params=sp)
    with torch.no_grad():
        got = ours.forward(req, output_type="latent").output
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    print(f"[pipeline-parity] latents rel-L2 {rel:.3e}")
    assert rel < 1e-5, rel

    with torch.no_grad():
        video = ours.decode_latents(got)
    assert video.shape == (1, 3, f, h, w) and torch.isfinite(video).all()


def test_encoder_cache_holds_its_key_tensors(tiny_ckpt, single_rank):  # noqa: F811
    """The encoder-prep cache is keyed on tensor addresses; each entry must keep its key tensors alive
    so a later tensor allocated at a freed address (e.g. the CFG negative branch right after the
    positive one) can never be served the cached entry of another tensor."""
    import gc

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import (
        NeuronHunyuanVideo15Transformer,
    )

    m = NeuronHunyuanVideo15Transformer(
        SimpleNamespace(model=tiny_ckpt, dtype=torch.float32, model_config={}, flow_shift=None)
    )
    text, tmask = torch.randn(1, 40, 64), torch.zeros(1, 40)
    tmask[0, :7] = 1
    text2, t2mask = torch.zeros(1, 16, 32), torch.zeros(1, 16)
    m._encoder(text, tmask, text2, t2mask, None, 24)
    ((srcs, _),) = m._enc_cache.values()
    assert srcs[0] is text and srcs[1] is tmask
    ptr = text.data_ptr()
    del text
    gc.collect()
    other = torch.randn(1, 40, 64)  # can never take the cached tensor's address: it is still alive
    assert other.data_ptr() != ptr
    _, _, _, _, _, tb_other, _, _ = m._encoder(other, tmask, text2, t2mask, None, 24)
    assert len(m._enc_cache) == 2


def _cfg_worker(rank, world, tp, cfg, port, ckpt, out_path, lat0, host_exchange=False):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as vps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        VLLM_NEURON_CPU_MODE="1",
    )
    vps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, backend="gloo")
    initialize_model_parallel(cfg_parallel_size=cfg, tensor_parallel_size=tp)
    import vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 as hv_pipe
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import _cfg_info

    if host_exchange:  # the non-routable-pair path (TP=8 x CFG=2 on 16 cores)
        hv_pipe._cfg_gather_on_device = lambda group: False

    assert _cfg_info()[1] == cfg  # the CFG-parallel branch is really exercised
    pipe = _ours(ckpt)
    assert pipe._tp_size == tp and pipe.transformer.dit.tp_size == tp
    sp = OmniDiffusionSamplingParams(
        height=64, width=96, num_frames=9, num_inference_steps=3, latents=lat0.clone()
    )
    req = SimpleNamespace(prompts=[{"prompt": PROMPT}], sampling_params=sp)
    with torch.no_grad():
        got = pipe.forward(req, output_type="latent").output
    torch.save(got, f"{out_path}.{rank}")


@pytest.mark.parametrize("tp,cfg,host_exchange", [(1, 2, False), (2, 2, False), (2, 2, True)])
def test_cfg_parallel_matches_sequential(tiny_ckpt, tmp_path, tp, cfg, host_exchange):  # noqa: F811
    """CFG-parallel (each CFG replica runs one branch, branch outputs exchanged in-graph or on the
    host, same combine on every rank) must give exactly the sequential-CFG latents, on every rank."""

    lat0 = torch.randn(1, 32, 3, 4, 6, generator=torch.Generator().manual_seed(7))
    world = tp * cfg
    out = str(tmp_path / f"lat_tp{tp}_cfg{cfg}_h{int(host_exchange)}")
    spawn_with_free_port(
        _cfg_worker, lambda port: (world, tp, cfg, port, tiny_ckpt, out, lat0, host_exchange), world
    )
    ref_out = str(tmp_path / "lat_ref")
    if not os.path.exists(f"{ref_out}.0"):
        spawn_with_free_port(_cfg_worker, lambda port: (1, 1, 1, port, tiny_ckpt, ref_out, lat0), 1)
    ref = torch.load(f"{ref_out}.0")
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-5, (tp, cfg, r, rel)


def _reverse_sp_groups(world, cp):
    """Rebuild the SP (CP) group from DESCENDING rank lists, like the physical-mesh layouts do
    (e.g. ``[12, 8]``): ``rank_in_group`` then disagrees with the sorted c10d group order."""
    import vllm_omni.diffusion.distributed.parallel_state as omni_ps

    groups = [list(range(s + cp - 1, s - 1, -1)) for s in range(0, world, cp)]
    rank = omni_ps.get_world_group().rank_in_group
    omni_ps._SP.destroy()
    ulysses_pg, ring_pg = omni_ps.set_seq_parallel_pg(
        sp_ulysses_degree=1, sp_ring_degree=cp, rank=rank, world_size=world, sp_group_ranks=groups
    )
    omni_ps._SP = omni_ps.init_model_parallel_group(
        group_ranks=groups,
        local_rank=rank,
        backend="gloo",
        parallel_mode="sequence",
        ulysses_group=ulysses_pg,
        ring_group=ring_pg,
    )
    assert omni_ps.get_sp_group().rank_in_group == groups[rank // cp].index(rank)


def _cp_worker(rank, world, tp, cp, cfg, port, ckpt, out_path, lat0, hw, reverse=False):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as vps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        VLLM_NEURON_CPU_MODE="1",
        HV15_BLOCKS_PER_GRAPH="2",
    )
    vps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, backend="gloo")
    initialize_model_parallel(
        cfg_parallel_size=cfg, tensor_parallel_size=tp, sequence_parallel_size=cp, ring_degree=cp
    )
    if reverse:
        _reverse_sp_groups(world, cp)
    pipe = _ours(ckpt)
    dit = pipe.transformer.dit
    assert dit.tp_size == tp and dit.cp_size == cp
    sp = OmniDiffusionSamplingParams(
        height=hw[0], width=hw[1], num_frames=9, num_inference_steps=2, latents=lat0.clone()
    )
    req = SimpleNamespace(prompts=[{"prompt": PROMPT}], sampling_params=sp)
    with torch.no_grad():
        got = pipe.forward(req, output_type="latent").output
    torch.save(got, f"{out_path}.{rank}")


@pytest.mark.parametrize(
    "tp,cp,cfg,hw",
    [(1, 2, 1, (64, 96)), (2, 2, 1, (64, 96)), (1, 8, 1, (64, 80)), (2, 2, 2, (64, 80))],
)
def test_context_parallel_matches_single_rank(tiny_ckpt, tmp_path, tp, cp, cfg, hw):  # noqa: F811
    """Context parallel (video tokens split across CP ranks, K/V all-gathered, encoder replicated)
    equals the single-rank pipeline on every rank, including a video length that needs padding
    (64x80: 3x4x5 = 60 latent tokens over 8 ranks)."""
    t, h, w = 3, hw[0] // 16, hw[1] // 16
    lat0 = torch.randn(1, 32, t, h, w, generator=torch.Generator().manual_seed(7))
    world = tp * cp * cfg
    out = str(tmp_path / f"lat_tp{tp}_cp{cp}_cfg{cfg}")
    spawn_with_free_port(
        _cp_worker, lambda port: (world, tp, cp, cfg, port, tiny_ckpt, out, lat0, hw), world
    )
    ref_out = str(tmp_path / "lat_ref")
    spawn_with_free_port(
        _cp_worker, lambda port: (1, 1, 1, 1, port, tiny_ckpt, ref_out, lat0, hw), 1
    )
    ref = torch.load(f"{ref_out}.0")
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-5, (tp, cp, cfg, r, rel)


@pytest.mark.parametrize("cp", [2, 4])
def test_context_parallel_descending_group_ranks(tiny_ckpt, tmp_path, cp):  # noqa: F811
    """CP over groups built from descending rank lists (the trn2 physical-mesh layouts): the in-graph
    K/V and output-token gathers must follow the CP rank order, not the sorted c10d order."""
    hw = (64, 96)
    lat0 = torch.randn(
        1, 32, 3, hw[0] // 16, hw[1] // 16, generator=torch.Generator().manual_seed(7)
    )
    out = str(tmp_path / "lat_rev")
    spawn_with_free_port(
        _cp_worker, lambda port: (cp, 1, cp, 1, port, tiny_ckpt, out, lat0, hw, True), cp
    )
    ref_out = str(tmp_path / "lat_ref")
    spawn_with_free_port(
        _cp_worker, lambda port: (1, 1, 1, 1, port, tiny_ckpt, ref_out, lat0, hw), 1
    )
    ref = torch.load(f"{ref_out}.0")
    for r in range(cp):
        got = torch.load(f"{out}.{r}")
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-5, (r, rel)


def _gather_worker(rank, world, port, out):
    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.transformer import group_all_gather

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
    )
    try:
        ranks = list(range(world - 1, -1, -1))  # [1, 0]: coordinator order != sorted c10d order
        pg = dist.new_group(ranks)

        def all_gather(x, dim):  # GroupCoordinator.all_gather on a (sorted) gloo device group
            parts = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(parts, x, group=pg)
            return torch.cat(parts, dim=dim)

        coord = SimpleNamespace(ranks=ranks, world_size=world, all_gather=all_gather)
        got = group_all_gather(coord, torch.tensor([[float(rank)]]), dim=1)
        torch.save(got.flatten().tolist(), os.path.join(out, f"g{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_group_all_gather_follows_coordinator_order(tmp_path):
    """Parts come back in ``coord.ranks`` order (global rank 1 first), as the in-graph collective on
    the NeuronCores, not in c10d's sorted order."""
    spawn_with_free_port(_gather_worker, lambda port: (2, port, str(tmp_path)), 2)
    for r in range(2):
        assert torch.load(tmp_path / f"g{r}.pt") == [1.0, 0.0]


def _reverse_cfg_groups(world):
    """CFG group from a DESCENDING rank list (``[1, 0]``): global rank 1 is CFG rank 0 (positive)."""
    import vllm_omni.diffusion.distributed.parallel_state as omni_ps

    rank = omni_ps.get_world_group().rank_in_group
    omni_ps._CFG.destroy()
    omni_ps._CFG = omni_ps.init_model_parallel_group(
        group_ranks=[list(range(world - 1, -1, -1))],
        local_rank=rank,
        backend="gloo",
        parallel_mode="classifier_free_guidance",
    )
    assert omni_ps.get_cfg_group().rank_in_group == world - 1 - rank


def _cfg_rev_worker(rank, world, port, ckpt, out_path, lat0):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as vps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        VLLM_NEURON_CPU_MODE="1",
        HV15_BLOCKS_PER_GRAPH="2",
    )
    vps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, backend="gloo")
    initialize_model_parallel(cfg_parallel_size=world, tensor_parallel_size=1)
    _reverse_cfg_groups(world)
    pipe = _ours(ckpt)
    sp = OmniDiffusionSamplingParams(
        height=64, width=96, num_frames=9, num_inference_steps=3, latents=lat0.clone()
    )
    req = SimpleNamespace(prompts=[{"prompt": PROMPT}], sampling_params=sp)
    with torch.no_grad():
        got = pipe.forward(req, output_type="latent").output
    torch.save(got, f"{out_path}.{rank}")


def test_cfg_parallel_descending_group_ranks(tiny_ckpt, tmp_path):  # noqa: F811
    """CFG-parallel over a descending CFG group: the in-graph branch gather follows CFG rank order,
    so positive/negative never swap; every rank equals sequential CFG."""
    lat0 = torch.randn(1, 32, 3, 4, 6, generator=torch.Generator().manual_seed(7))
    out = str(tmp_path / "lat_cfg_rev")
    spawn_with_free_port(_cfg_rev_worker, lambda port: (2, port, tiny_ckpt, out, lat0), 2)
    ref_out = str(tmp_path / "lat_ref")
    spawn_with_free_port(_cfg_worker, lambda port: (1, 1, 1, port, tiny_ckpt, ref_out, lat0), 1)
    ref = torch.load(f"{ref_out}.0")
    for r in range(2):
        got = torch.load(f"{out}.{r}")
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-5, (r, rel)


def _text_worker(rank, world, tp, port, ckpt, out_path, mode):
    import torch.distributed as dist
    import vllm.distributed.parallel_state as vps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        VLLM_NEURON_CPU_MODE="1",
        HV15_TEXT_ENCODER=mode,
        HV15_TEXT_TP=str(tp),
        HV15_TEXT_LAYERS_PER_GRAPH="1",
    )
    vps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, backend="gloo")
    initialize_model_parallel(tensor_parallel_size=world)
    pipe = _ours(ckpt)
    assert (pipe.text_tower is not None) == (mode == "device")
    out = []
    for _ in range(2):  # the second call is served by the per-prompt cache
        out.append(
            pipe.encode_prompt(
                PROMPT, torch.device("cpu"), torch.float32, None, do_classifier_free_guidance=True
            )
        )
    assert all(a is b for a, b in zip(out[0], out[1]))
    torch.save(out[0], f"{out_path}.{rank}")


@pytest.mark.parametrize("world,tp", [(2, 1), (2, 2), (4, 2)])
def test_device_text_tower_matches_host(tiny_ckpt, tmp_path, world, tp):  # noqa: F811
    """The TP-sharded Qwen2.5-VL tower (prompts dealt across text groups, results gathered on rank 0
    and broadcast) gives the host ``transformers`` encoder's embeddings on every rank."""
    out = str(tmp_path / f"emb_w{world}_tp{tp}")
    spawn_with_free_port(
        _text_worker, lambda port: (world, tp, port, tiny_ckpt, out, "device"), world
    )
    ref_out = str(tmp_path / "emb_ref")
    spawn_with_free_port(_text_worker, lambda port: (1, 1, port, tiny_ckpt, ref_out, "host"), 1)
    ref = torch.load(f"{ref_out}.0")
    for r in range(world):
        got = torch.load(f"{out}.{r}")
        for i in (0, 4):  # MLLM embeddings of the positive / negative prompt
            mask = ref[i + 1].bool()[0]
            a, b = got[i][0][mask].float(), ref[i][0][mask].float()
            rel = ((a - b).norm() / b.norm()).item()
            assert rel < 1e-5, (world, tp, r, i, rel)
            assert torch.equal(got[i + 1], ref[i + 1])


def test_cfg_gather_routing():
    """In-graph CFG exchange only for routable pairs: inside one aligned 8-core block, else the host
    (TP=8 x CFG=2 on 16 cores pairs rank r with r + 8, two chips apart)."""
    from vllm_omni_neuron.diffusion.models.hunyuanvideo15.pipeline_hunyuanvideo15 import (
        _cfg_gather_on_device,
    )

    def g(*ranks):
        return SimpleNamespace(ranks=list(ranks))

    assert _cfg_gather_on_device(None)
    assert _cfg_gather_on_device(g(0, 2))  # TP=2 x CFG=2, one chip
    assert _cfg_gather_on_device(g(3, 7))  # TP=4 x CFG=2, adjacent chip pair
    assert _cfg_gather_on_device(g(12, 8))  # descending pair inside one block
    assert not _cfg_gather_on_device(g(0, 8))  # TP=8 x CFG=2 on 16 cores, no mesh layout on CPU
    assert not _cfg_gather_on_device(g(7, 15))
