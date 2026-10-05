# SPDX-License-Identifier: Apache-2.0
"""CPU parity for the Qwen3-VL backbone (Cosmos3-Nano / Cosmos3-Super / Super 4-step) on a tiny
random checkpoint with the real structure (``vllm_omni_neuron.tiny_models``).

Oracle: upstream vLLM-Omni's own ``Cosmos3VFMTransformer`` (vendored), loaded through upstream's
own key remap, fp32 on CPU. Checks the UND K/V per layer, the full GEN forward (T2I, I2V, action),
and TP=2 / TP=4 on CPU gloo ranks against TP=1 (TP=4 > 2 K/V heads exercises KV-head
replication, the layout Super uses at TP=16).

The tiny checkpoint is built from a real checkpoint's structure: set ``COSMOS3_TINY_SRC`` to a
local Cosmos3-Nano / Super checkout (only its configs and safetensors headers are read).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

SRC = os.environ.get("COSMOS3_TINY_SRC", "")


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory):
    if not SRC or not os.path.isdir(os.path.join(SRC, "transformer")):
        pytest.skip("set COSMOS3_TINY_SRC to a local Cosmos3-Nano / Cosmos3-Super checkout")
    # Compose the tiny checkpoint from shared tiny_models component helpers (VAE / vision encoder,
    # shape-only synthesis) plus a shrunk-width transformer synthesized here: the vendored Cosmos3
    # MoT transformer is not a diffusers class (no from_config/save_pretrained) and tiny_models'
    # `synthesize` keeps full width, so neither shared mode produces a small transformer.
    from vllm_omni_neuron import tiny_models as tm

    out = str(tmp_path_factory.mktemp("tiny") / "cosmos3")
    os.makedirs(out, exist_ok=True)
    # top-level non-weight files (model_index.json, tokenizer configs, templates)
    for name in sorted(os.listdir(SRC)):
        p = os.path.join(SRC, name)
        if (
            os.path.isfile(p)
            and name.endswith((".json", ".txt", ".jinja"))
            and "safetensors" not in name
        ):
            shutil.copy2(p, os.path.join(out, name))
    for comp in ("text_tokenizer", "scheduler"):
        if os.path.isdir(os.path.join(SRC, comp)):
            shutil.copytree(os.path.join(SRC, comp), os.path.join(out, comp), dirs_exist_ok=True)
    if os.path.isdir(os.path.join(SRC, "sound_tokenizer")):
        os.makedirs(os.path.join(out, "sound_tokenizer"), exist_ok=True)
        shutil.copy2(
            os.path.join(SRC, "sound_tokenizer", "config.json"),
            os.path.join(out, "sound_tokenizer", "config.json"),
        )
    for comp in ("vae", "vision_encoder"):
        sdir = os.path.join(SRC, comp)
        if not os.path.isdir(sdir):
            continue
        cfg = tm._load_json(os.path.join(sdir, "config.json"))
        changed = tm.shrink_layer_counts(cfg, 2)[1] if cfg else {}
        tm.synthesize_component(sdir, os.path.join(out, comp), 2, changed, name=comp)
        shutil.copy2(os.path.join(sdir, "config.json"), os.path.join(out, comp, "config.json"))
    _synthesize_tiny_transformer(os.path.join(SRC, "transformer"), os.path.join(out, "transformer"))
    return out


def _synthesize_tiny_transformer(
    src: str, out: str, layers: int = 2, dims=(512, 768, 8, 2, 128), seed: int = 0
) -> None:
    """Random-weight transformer with the real on-disk (upstream, unprefixed) parameter NAMES and
    dtypes but shrunk width and ``layers`` layers. Shapes are derived from the Qwen3/Edge config
    roles, so no model class is needed. Mirrors the generality of tiny_models.synthesize, adding the
    width shrink the shared module intentionally does not do."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    h, inter, nh, nkv, d = dims
    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    cfg = {
        **src_cfg,
        "hidden_size": h,
        "intermediate_size": inter,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": d,
        "num_hidden_layers": layers,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    index = next(f for f in os.listdir(src) if f.endswith(".safetensors.index.json"))
    weight_map = json.load(open(os.path.join(src, index)))["weight_map"]
    gen = torch.Generator().manual_seed(seed)
    nq, nkvd = nh * d, nkv * d
    role = {
        "self_attn.to_q.weight": (nq, h),
        "self_attn.add_q_proj.weight": (nq, h),
        "self_attn.to_k.weight": (nkvd, h),
        "self_attn.add_k_proj.weight": (nkvd, h),
        "self_attn.to_v.weight": (nkvd, h),
        "self_attn.add_v_proj.weight": (nkvd, h),
        "self_attn.to_out.weight": (h, nq),
        "self_attn.to_add_out.weight": (h, nq),
        "mlp.gate_proj.weight": (inter, h),
        "mlp.up_proj.weight": (inter, h),
        "mlp.down_proj.weight": (h, inter),
        "mlp_moe_gen.gate_proj.weight": (inter, h),
        "mlp_moe_gen.up_proj.weight": (inter, h),
        "mlp_moe_gen.down_proj.weight": (h, inter),
    }
    tensors = {}
    for key in sorted(weight_map):
        m = re.match(r"layers\.(\d+)\.", key)
        if m and int(m.group(1)) >= layers:
            continue
        with safe_open(os.path.join(src, weight_map[key]), "pt") as fh:
            sl = fh.get_slice(key)
            src_shape, dt = list(sl.get_shape()), sl.get_dtype()
        name = re.sub(r"^layers\.\d+\.", "", key)
        if name in role:
            shape = list(role[name])
        elif name.startswith("self_attn.norm_") or name == "self_attn.k_norm_und_for_gen.weight":
            shape = [d]
        elif (
            name.endswith("layernorm.weight")
            or name.endswith("layernorm_moe_gen.weight")
            or name
            in (
                "norm.weight",
                "norm_moe_gen.weight",
                "action_modality_embed",
                "audio_modality_embed",
                "proj_in.bias",
                "time_embedder.linear_1.bias",
                "time_embedder.linear_2.bias",
                "audio_proj_in.bias",
            )
        ):
            shape = [h]
        elif name in ("embed_tokens.weight", "lm_head.weight"):
            shape = [src_shape[0], h]
        elif name == "proj_in.weight":
            shape = [h, src_shape[1]]
        elif name in ("proj_out.weight", "audio_proj_out.weight"):
            shape = [src_shape[0], h]
        elif name == "time_embedder.linear_1.weight":
            shape = [h, src_shape[1]]
        elif name == "time_embedder.linear_2.weight":
            shape = [h, h]
        elif name == "audio_proj_in.weight":
            shape = [h, src_shape[1]]
        elif name in ("action_proj_in.fc.weight", "action_proj_out.fc.weight"):
            shape = [src_shape[0], src_shape[1] // src_cfg["hidden_size"] * h]
        elif name == "action_proj_in.bias.weight":
            shape = [src_shape[0], h]
        else:  # proj_out.bias, action_proj_out.bias.weight, audio_proj_out.bias, ...
            shape = list(src_shape)
        std = (
            1.0
            if len(shape) == 1 and ("norm" in name or "layernorm" in name)
            else (shape[-1] ** -0.5)
        )
        t = (
            (1.0 + 0.1 * torch.randn(shape, generator=gen))
            if std == 1.0
            else torch.randn(shape, generator=gen) * std
        )
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}.get(
            dt, torch.bfloat16
        )
        tensors[key] = t.to(dtype).contiguous()
    keys = sorted(tensors)
    half = len(keys) // 2
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": keys[:half],
        "diffusion_pytorch_model-00002-of-00002.safetensors": keys[half:],
    }
    new_map, total = {}, 0
    for fn, ks in shards.items():
        save_file({k: tensors[k] for k in ks}, os.path.join(out, fn), metadata={"format": "pt"})
        for k in ks:
            new_map[k], total = fn, total + tensors[k].numel() * tensors[k].element_size()
    with open(os.path.join(out, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, f, indent=1)


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


@pytest.fixture(scope="module")
def upstream_tf(vllm_single_rank, tiny_weights):
    from safetensors import safe_open

    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor import transformer_cosmos3 as _tc

    _tc._get_ulysses_state = lambda: (1, 0, None)
    _tc._is_sp_active = lambda: False
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    with open(os.path.join(tiny_weights, "transformer", "config.json")) as f:
        cfg = json.load(f)
    od = SimpleNamespace(
        tf_model_config=cfg,
        dtype=torch.float32,
        model_config={},
        custom_pipeline_args={},
        quantization_config=None,
    )
    tf = _tc.Cosmos3VFMTransformer(od, temporal_compression_factor=4, sound_gen=False)
    for layer in tf.language_model.layers:
        layer.self_attn.attn = _Sdpa(True)
    for layer in tf.gen_layers:
        layer.cross_attention.attn = _Sdpa(False)
    state = {}
    tdir = os.path.join(tiny_weights, "transformer")
    for fn in sorted(f for f in os.listdir(tdir) if f.endswith(".safetensors")):
        with safe_open(os.path.join(tdir, fn), "pt") as f:
            for key in f.keys():
                if key.startswith("audio_"):
                    continue  # sound is not ported
                name = Cosmos3OmniDiffusersPipeline._remap_ckpt_key("transformer." + key)
                if name:
                    state[name[len("transformer.") :]] = f.get_tensor(key).float()
    missing, unexpected = tf.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(("inv_freq" in m) or m.endswith("freqs") for m in missing), missing
    return tf.float().eval()


@pytest.fixture(scope="module")
def ours(vllm_single_rank, tiny_weights):
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

    cfg = EdgeGenConfig.from_model_dir(tiny_weights)
    assert (
        cfg.backbone == "qwen3"
        and cfg.gated_mlp
        and cfg.und_qk_norm
        and not cfg.use_und_k_norm_for_gen
    )
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(tiny_weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.float32)
    gen.load_weights(tiny_weights, "cpu")
    return und, gen


def _prompt(real=21, bucket=32, seed=1):
    torch.manual_seed(seed)
    ids = torch.zeros(1, bucket, dtype=torch.long)
    ids[0, :real] = torch.randint(1000, 100000, (real,))
    mask = torch.zeros(1, bucket, dtype=torch.long)
    mask[0, :real] = 1
    return ids, mask, real


def test_und_matches_upstream(upstream_tf, ours):
    und, _ = ours
    ids, mask, real = _prompt()
    cos, sin = und.rope_tables(mask)
    with torch.no_grad():
        out = und(ids, cos, sin)
        ref_kv = upstream_tf.language_model(ids[:, :real], (cos[:, :real], sin[:, :real]))
    n = und.cfg.num_layers
    for i, (k_ref, v_ref) in enumerate(ref_kv):
        torch.testing.assert_close(
            out[i][:, :real], k_ref, rtol=2e-4, atol=2e-4, msg=f"K layer {i}"
        )
        torch.testing.assert_close(
            out[n + i][:, :real], v_ref, rtol=2e-4, atol=2e-4, msg=f"V layer {i}"
        )


@pytest.mark.parametrize("case", ["t2i", "i2v", "action"])
def test_gen_matches_upstream(case, upstream_tf, ours):
    und, gen = ours
    if case == "action" and not gen.cfg.action_gen:
        pytest.skip("checkpoint has no action head")
    ids, mask, real = _prompt()
    t = 1 if case == "t2i" else 3
    h = w = 16
    lat = torch.randn(1, 48, t, h, w)
    ts = torch.tensor([637.0])
    fps = None if case == "t2i" else 24.0
    noisy = torch.ones(1, 1, t, 1, 1)
    if case != "t2i":
        noisy[:, :, 0] = 0
    kw, s_act = {}, 0
    if case == "action":
        s_act = 4
        act = torch.randn(1, s_act, gen.cfg.action_dim)
        act_mask = torch.ones(1, s_act, 1)
        kw = dict(
            action_latents=act,
            action_domain_ids=torch.tensor([3]),
            action_noisy_mask=act_mask,
            action_start_frame_offset=1,
            action_fps=12.0,
        )
    upstream_tf.reset_cache()
    with torch.no_grad():
        ref = upstream_tf(
            lat,
            ts,
            ids[:, :real],
            mask[:, :real],
            (t, h, w),
            fps=fps,
            noisy_frame_mask=noisy if case != "t2i" else None,
            **kw,
        )
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu, su)
        cg, sg = gen.rope_tables(
            mask, t, h, w, fps, t_action=s_act, action_fps=12.0 if s_act else None
        )
        s_video = t * (h // 2) * (w // 2)
        kb = gen.key_bias(mask, s_video + s_act)
        nm = (
            noisy[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, (h // 2) * (w // 2)).reshape(1, -1, 1)
        )
        if case == "action":
            w_in, b_in, w_out, b_out = gen.domain_weights(3, "cpu")
            out = gen.forward_action(
                lat, ts, cg, sg, kb, nm, act, act_mask, w_in, b_in, w_out, b_out, *kv
            )
        else:
            out = gen(lat, ts, cg, sg, kb, nm, *kv)
    refs = ref if isinstance(ref, tuple) else (ref,)
    outs = out if isinstance(out, tuple) else (out,)
    for o, r in zip(outs, refs, strict=True):
        rel = ((o - r).norm() / r.norm()).item()
        assert rel < 1e-4, f"{case}: rel err {rel}"


ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")


def action_mode_masks(mode: str, num_frames: int, chunk: int):
    """Noisy masks of one action-mode request, from the vendored pipeline's own condition rules:
    video ``[1, 1, T_lat, 1, 1]`` and action ``[1, chunk, 1]`` (1 = denoised, 0 = clean condition).
    policy / forward_dynamics condition latent frame 0, inverse_dynamics every latent frame;
    forward_dynamics also conditions every action row."""
    from vllm_omni_neuron.diffusion.models.cosmos3_edge._vendor.action import (
        build_action_condition_mask,
        build_vision_condition_mask,
    )

    cpu = torch.device("cpu")
    vis = build_vision_condition_mask(mode, num_frames, 4, device=cpu, dtype=torch.float32)
    act = build_action_condition_mask(mode, chunk, device=cpu, dtype=torch.float32)
    return 1.0 - vis, 1.0 - act


@pytest.mark.parametrize("mode", ACTION_MODES)
def test_gen_action_modes_match_upstream(mode, upstream_tf, ours):
    """Each action mode's GEN call (DROID domain 8, 17 frames = 5 latent frames + a 16-row chunk)
    against the upstream transformer, video AND action outputs."""
    und, gen = ours
    if not gen.cfg.action_gen:
        pytest.skip("checkpoint has no action head")
    ids, mask, real = _prompt(seed=7)
    num_frames, chunk, h, w = 17, 16, 16, 16
    t = (num_frames - 1) // 4 + 1
    torch.manual_seed(11)
    lat = torch.randn(1, 48, t, h, w)
    act = torch.randn(1, chunk, gen.cfg.action_dim)
    ts = torch.tensor([512.0])
    noisy, act_mask = action_mode_masks(mode, num_frames, chunk)
    upstream_tf.reset_cache()
    with torch.no_grad():
        ref = upstream_tf(
            lat,
            ts,
            ids[:, :real],
            mask[:, :real],
            (t, h, w),
            fps=15.0,
            noisy_frame_mask=noisy,
            action_latents=act,
            action_domain_ids=torch.tensor([8]),
            action_noisy_mask=act_mask,
            action_start_frame_offset=1,
            action_fps=15.0,
        )
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu, su)
        cg, sg = gen.rope_tables(mask, t, h, w, 15.0, t_action=chunk, action_fps=15.0)
        hw = (h // 2) * (w // 2)
        kb = gen.key_bias(mask, t * hw + chunk)
        nm = noisy[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, hw).reshape(1, -1, 1)
        w_in, b_in, w_out, b_out = gen.domain_weights(8, "cpu")
        out = gen.forward_action(
            lat, ts, cg, sg, kb, nm, act, act_mask, w_in, b_in, w_out, b_out, *kv
        )
    for name, o, r in zip(("video", "action"), out, ref, strict=True):
        rel = ((o - r).norm() / r.norm()).item()
        assert rel < 1e-4, f"{mode} {name}: rel err {rel}"


# -- TP on CPU gloo ranks --------------------------------------------------------------------
def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tp_run(rank, world, port, weights, out_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

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
    initialize_model_parallel(world, 1)
    cfg = EdgeGenConfig.from_model_dir(weights)
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.float32)
    gen.load_weights(weights, "cpu")
    ids, mask, _ = _prompt(19, 32, 3)
    lat, ts = torch.randn(1, 48, 3, 16, 16), torch.tensor([421.0])
    t, h, w = lat.shape[2:]
    with torch.no_grad():
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu, su)
        cg, sg = gen.rope_tables(mask, t, h, w, 24.0)
        s_gen = t * (h // 2) * (w // 2)
        out = gen(lat, ts, cg, sg, gen.key_bias(mask, s_gen), torch.ones(1, s_gen, 1), *kv)
    if rank == 0:
        torch.save(out, out_path)


def test_tp_matches(tiny_weights, tmp_path):
    outs = {}
    for world in (1, 2, 4):
        p = tmp_path / f"tp{world}.pt"
        mp.spawn(_tp_run, args=(world, _port(), tiny_weights, str(p)), nprocs=world, join=True)
        outs[world] = torch.load(p)
    for world in (2, 4):
        rel = ((outs[world] - outs[1]).norm() / outs[1].norm()).item()
        assert rel < 1e-5, (world, rel)


def _cp_run(rank, world, port, weights, out_path, groups=None):
    """TP=1 x CP: each rank runs ``forward_cp`` on its token slice; its in-graph output gather must
    reproduce the unsplit ``forward`` (I2V-style noisy mask, so the slicing of every per-token input is
    exercised). ``groups`` = the CP groups as rank lists in GROUP-RANK order (default: one ascending
    group of ``world``); a descending list such as ``[3, 1]`` is the Trn2 physical-mesh shape, whose
    c10d group is sorted -- the gather must still follow the list. Every rank saves its own result."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_omni_neuron.diffusion.models.cosmos3_edge.gen_tower import (
        EdgeGenConfig,
        NeuronCosmos3EdgeGEN,
    )
    from vllm_omni_neuron.diffusion.models.cosmos3_edge.und_tower import NeuronCosmos3EdgeUND

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
    initialize_model_parallel(1, 1)
    groups = groups or [list(range(world))]
    pgs = [dist.new_group(g) for g in groups]  # every rank creates every group, same order
    mine = next(i for i, g in enumerate(groups) if rank in g)
    size, cp_rank = len(groups[mine]), groups[mine].index(rank)
    cfg = EdgeGenConfig.from_model_dir(weights)
    und = NeuronCosmos3EdgeUND(cfg, dtype=torch.float32)
    und.load_weights(weights, "cpu")
    gen = NeuronCosmos3EdgeGEN(cfg, dtype=torch.float32)
    gen.load_weights(weights, "cpu")
    gen.set_context_parallel(size, cp_rank, pgs[mine], groups[mine])
    ids, mask, _ = _prompt(19, 32, 3)
    torch.manual_seed(0)
    lat, ts = torch.randn(1, 48, 4, 16, 16), torch.tensor([421.0])
    t, h, w = lat.shape[2:]
    s_gen = t * (h // 2) * (w // 2)
    nm = torch.ones(1, s_gen, 1)
    nm[:, : (h // 2) * (w // 2)] = 0  # clean first latent frame (I2V)
    loc = s_gen // size
    sl = slice(cp_rank * loc, (cp_rank + 1) * loc)
    with torch.no_grad():
        cu, su = und.rope_tables(mask)
        kv = und(ids, cu, su)
        cg, sg = gen.rope_tables(mask, t, h, w, 24.0)
        kb = gen.key_bias(mask, s_gen)
        full = gen(lat, ts, cg, sg, kb, nm, *kv)
        tokens = gen.forward_cp(
            gen._patchify(lat)[:, sl], ts, cg[:, sl], sg[:, sl], kb, nm[:, sl], *kv
        )
        split = gen._unpatchify(tokens, t, h, w)
    torch.save({"full": full, "split": split}, f"{out_path}.{rank}")


def test_cp_matches_unsplit(tiny_weights, tmp_path):
    for world in (2, 4):
        p = tmp_path / f"cp{world}.pt"
        mp.spawn(_cp_run, args=(world, _port(), tiny_weights, str(p)), nprocs=world, join=True)
        for rank in range(world):
            r = torch.load(f"{p}.{rank}")
            rel = ((r["split"] - r["full"]).norm() / r["full"].norm()).item()
            assert rel < 1e-5, (world, rank, rel)


def test_cp_descending_group_ranks(tiny_weights, tmp_path):
    """Trn2 physical-mesh CP groups (TP8 x CP2: ``[0, 4] ... [12, 8]``): one ascending and one
    descending CP2 group over 4 ranks. A raw gather in c10d order swaps the slices on ``[3, 1]``."""
    p = tmp_path / "cpdesc.pt"
    mp.spawn(
        _cp_run, args=(4, _port(), tiny_weights, str(p), [[0, 2], [3, 1]]), nprocs=4, join=True
    )
    for rank in range(4):
        r = torch.load(f"{p}.{rank}")
        rel = ((r["split"] - r["full"]).norm() / r["full"].norm()).item()
        assert rel < 1e-5, (rank, rel)
