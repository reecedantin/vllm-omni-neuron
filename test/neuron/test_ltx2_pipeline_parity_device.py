# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 pipeline parity on Neuron: the served pipeline (``NeuronLTX25Pipeline``, TP over the visible
cores) vs the diffusers CPU pipeline with the reference transformer, same weights, prompt and seed.

Three-way, on the denoised latents (the VAE is checked separately, ``test_ltx2_vae_parity_device.py``):

* CPU fp32 reference (diffusers ``LTX2Pipeline`` + vendored ``LTX2VideoTransformer3DModel``);
* CPU bf16 (the same, bf16): the dtype-only error band;
* Neuron bf16: the served pipeline's ``forward(..., output_type="latent")``.

Pass: device rel-L2 <= 2 x (CPU bf16 rel-L2) + 0.005, per modality (video, audio). ``--steps 1`` is the
single-step tier (whole pipeline, one denoising step); the default 4 steps is the end-to-end latent tier.
The video path of the distilled schedule is bf16-sensitive at the model level (CPU bf16 alone diverges
~23% from fp32 over 4 steps), which is why the bar is relative to the band, not absolute.

Run as two steps (the reference needs ~140 GB of host RAM and no cores; the device run needs the cores):

    python -m test.neuron.test_ltx2_pipeline_parity_device --model <dir> --mode reference --out <dir> [--steps N]
    torchrun --nproc_per_node 4 -m test.neuron.test_ltx2_pipeline_parity_device --model <dir> --mode device \
        --out <dir> [--steps N] [--cp C]   # TP = nproc / C

As a pytest it skips unless ``LTX2_PARITY_OUT`` points at a directory with both results.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pytest
import torch

PROMPT = "A red fox walking through a snowy forest at dawn, the camera tracking alongside."
SHAPE = dict(height=192, width=256, num_frames=25, frame_rate=24.0, seed=42)


def _kwargs(steps: int) -> dict:
    return dict(
        prompt=PROMPT,
        height=SHAPE["height"],
        width=SHAPE["width"],
        num_frames=SHAPE["num_frames"],
        frame_rate=SHAPE["frame_rate"],
        num_inference_steps=steps,
        guidance_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        audio_guidance_scale=1.0,
        audio_stg_scale=0.0,
        audio_modality_scale=1.0,
        generator=torch.Generator().manual_seed(SHAPE["seed"]),
        output_type="latent",
        use_cross_timestep=True,
        return_dict=True,
    )


def _np(x):
    return np.asarray(x.detach().float().cpu() if torch.is_tensor(x) else x)


def _reference_pipeline(model: str, dtype):
    import inspect

    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel as Installed,
    )
    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

    from vllm_omni_neuron.diffusion.models.ltx2._vendor.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )

    with open(os.path.join(model, "transformer", "config.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    ok = inspect.signature(Installed.__init__).parameters
    placeholder = Installed(**{**{k: v for k, v in cfg.items() if k in ok}, "num_layers": 0}).to(
        dtype
    )
    pipe = LTX2Pipeline.from_pretrained(model, torch_dtype=dtype, transformer=placeholder)
    pipe.transformer = LTX2VideoTransformer3DModel.from_pretrained(
        os.path.join(model, "transformer"), torch_dtype=dtype
    ).eval()
    return pipe


def run_reference(model: str, out: str, steps: int) -> None:
    os.makedirs(out, exist_ok=True)
    meta = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        pipe = _reference_pipeline(model, dtype)
        t0 = time.time()
        with torch.no_grad():
            o = pipe(**_kwargs(steps))
        meta[f"{name}_s"] = round(time.time() - t0, 1)
        np.save(os.path.join(out, f"ref_{name}_s{steps}_video.npy"), _np(o.frames[0]))
        np.save(os.path.join(out, f"ref_{name}_s{steps}_audio.npy"), _np(o.audio[0]))
        del pipe
    print("REFERENCE_DONE " + json.dumps(meta), flush=True)


def _pin_rank_core(local_rank: int) -> None:
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if "-" in cores and "," not in cores:
        lo, hi = (int(x) for x in cores.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(x) for x in cores.split(",")] if cores else []
    if local_rank < len(ids):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(ids[local_rank])


def setup_device_pipeline(model: str, cp: int = 1):
    """Build the served pipeline class on this rank (torchrun: one pinned core per rank, TP =
    world / ``cp``), loaded and compiled. Returns ``(pipe, rank, world)``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        _pin_rank_core(local_rank)
    import vllm_omni_neuron  # noqa: F401
    from vllm_omni_neuron.lite_compat import initialize

    initialize()
    torch.nn.functional.gelu = torch.ops.aten.gelu.default
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_neuron.envs import get_compile_backend_name, get_dist_backend

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    ps.in_the_same_node_as = lambda pg, source_rank=0: [True] * dist.get_world_size(pg)
    if cp > 1:  # TP x CP, built the way the served worker builds it (diffusion_worker.py)
        from vllm_omni.diffusion.distributed.parallel_state import (
            init_distributed_environment as omni_init,
        )
        from vllm_omni.diffusion.distributed.parallel_state import (
            initialize_model_parallel as omni_mp,
        )

        from vllm_omni_neuron.diffusion.distributed.parallel_state import (
            override_groups_with_physical_mesh,
        )

        omni_init(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend=get_dist_backend(),
        )
        omni_mp(tensor_parallel_size=world // cp, ring_degree=cp, sequence_parallel_size=cp)
        override_groups_with_physical_mesh(tp_size=world // cp, cp_size=cp, cfg_size=1)
    else:
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend=get_dist_backend(),
        )
        initialize_model_parallel(world, 1)

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    from vllm_omni_neuron.diffusion.models.ltx2.pipeline_ltx25 import NeuronLTX25Pipeline

    pipe = NeuronLTX25Pipeline(od_config=OmniDiffusionConfig(model=model, dtype=torch.bfloat16))
    if world > 1:  # one core per rank (pinned above), so the transformer goes to that rank's core 0
        object.__setattr__(pipe.od_config, "device", torch.device("neuron", 0))
    pipe.load_weights()
    pipe.compile(backend=get_compile_backend_name())
    return pipe, rank, world


def device_request(prompt: str, **shape):
    """A single-request batch as the engine hands it to ``NeuronLTX25Pipeline.forward``."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sp = OmniDiffusionSamplingParams(**shape)
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(request_id="p", prompt=prompt, sampling_params=sp)]
    )


def run_device(model: str, out: str, steps: int, cp: int = 1) -> None:
    os.environ.setdefault("LTX2_VAE_DEVICE", "0")  # latents only: the VAE is not on this path
    pipe, rank, _ = setup_device_pipeline(model, cp)
    batch = device_request(
        PROMPT,
        height=SHAPE["height"],
        width=SHAPE["width"],
        num_frames=SHAPE["num_frames"],
        frame_rate=SHAPE["frame_rate"],
        num_inference_steps=steps,
        seed=SHAPE["seed"],
        output_type="latent",
    )
    t0 = time.time()
    res = pipe.forward(batch)
    dt = time.time() - t0
    if rank != 0:
        return
    np.save(os.path.join(out, f"dev_s{steps}_video.npy"), _np(res.output["video"]))
    np.save(os.path.join(out, f"dev_s{steps}_audio.npy"), _np(res.output["audio"]))
    report = compare(out, steps)
    report["device_request_s"] = round(dt, 1)
    print("PARITY " + json.dumps(report), flush=True)


def _rel_cos(ref, x):
    r, p = ref.astype(np.float64).ravel(), x.astype(np.float64).ravel()
    rn = max(np.linalg.norm(r), 1e-12)
    return float(np.linalg.norm(r - p) / rn), float(
        np.vdot(r, p) / (rn * max(np.linalg.norm(p), 1e-12))
    )


def compare(out: str, steps: int) -> dict:
    report: dict = {"steps": steps, "shape": SHAPE}
    ok = True
    for mod in ("video", "audio"):
        r32 = np.load(os.path.join(out, f"ref_fp32_s{steps}_{mod}.npy"))
        r16 = np.load(os.path.join(out, f"ref_bf16_s{steps}_{mod}.npy"))
        dev = np.load(os.path.join(out, f"dev_s{steps}_{mod}.npy")).reshape(r32.shape)
        band, _ = _rel_cos(r32, r16)
        rel, cos = _rel_cos(r32, dev)
        bar = 2.0 * band + 0.005
        report[mod] = {
            "device_rel_l2": round(rel, 5),
            "device_cos": round(cos, 6),
            "cpu_bf16_band": round(band, 5),
            "bar": round(bar, 5),
            "pass": bool(rel <= bar),
        }
        ok &= rel <= bar
    report["pass"] = bool(ok)
    with open(os.path.join(out, f"parity_s{steps}.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


@pytest.mark.skipif(
    not os.environ.get("LTX2_PARITY_OUT"), reason="set LTX2_PARITY_OUT to a reference+device run"
)
@pytest.mark.parametrize("steps", [1, 4])
def test_pipeline_parity_device(steps):
    out = os.environ["LTX2_PARITY_OUT"]
    if not os.path.exists(os.path.join(out, f"dev_s{steps}_video.npy")):
        pytest.skip(f"no device result for {steps} steps")
    assert compare(out, steps)["pass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("reference", "device"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cp", type=int, default=1, help="context-parallel degree (device mode)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "reference":
        run_reference(a.model, a.out, a.steps)
    else:
        run_device(a.model, a.out, a.steps, a.cp)


if __name__ == "__main__":
    main()
