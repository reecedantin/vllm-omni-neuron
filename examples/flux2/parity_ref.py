# SPDX-License-Identifier: Apache-2.0
"""M1 reference: the Neuron FLUX.2 pipeline on CPU (fp32 and bf16), compared to the device latents
dumped by ``run.py --parity-latents``.

Uses the SAME pipeline class (``NeuronFlux2Pipeline``) as the device run, in CPU mode, so the text
encoder and DiT math are identical and the only variable is device vs CPU execution -- the right
oracle for "does the compiled Neuron graph match this pipeline's own fp32 result". (Agreement with
diffusers/transformers is covered separately by the component unit tests.) All three runs share the
same fp32 initial noise via ``FLUX2_INIT_LATENTS`` and the same prompt, so the whole denoising path
excluding VAE decode is compared.

Run on the host under cpumode.sh (no Neuron):
    FLUX2_INIT_LATENTS=init.pt python examples/flux2/parity_ref.py \
        --model-path <dir> --device-latents dev.pt --out m1.json

Gate: device rel-L2 vs fp32 within 2x the CPU-bf16 rel-L2 + 0.5%, cosine >= 0.999.
"""

import argparse
import json
import socket

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip
import torch


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _cos(a, b):
    return float(
        torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    )


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reference_latents(model_path, meta, dtype):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_neuron.diffusion.models.flux2 import NeuronFlux2Pipeline

    od = OmniDiffusionConfig.from_kwargs(
        model=model_path,
        dtype=dtype,
        model_class_name="Flux2Pipeline",
        output_type="latent",
        parallel_config={"tensor_parallel_size": 1},
    )
    pipe = NeuronFlux2Pipeline(od_config=od)
    threads = torch.get_num_threads()
    pipe.load_weights()
    # the checkpoint loader's worker threads call torch.set_num_threads(1), which is process-wide:
    # without this restore the whole CPU reference runs on one core
    torch.set_num_threads(threads)
    sp = OmniDiffusionSamplingParams(
        height=int(meta["height"]),
        width=int(meta["width"]),
        num_inference_steps=int(meta["steps"]),
        guidance_scale=float(meta["guidance_scale"]),
        seed=int(meta["seed"]),
        output_type="latent",
    )
    req = DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=meta["prompt"], sampling_params=sp, request_id="r0")]
    )
    with torch.no_grad():
        out = pipe.forward(req).output
    return out.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device-latents", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--save-refs",
        default=None,
        help="also save {fp32, bf16, device} latents (.pt) for "
        "vllm_neuron.accuracy.testing.assert_close_three_way",
    )
    ap.add_argument(
        "--decode",
        action="store_true",
        help="also decode the fp32 / bf16 reference latents with the untiled diffusers VAE in the "
        "same dtype (host CPU, independent of the device tiled decode) and store them in "
        "--save-refs as fp32_image / bf16_image",
    )
    a = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    _ctx = set_current_vllm_config(VllmConfig())
    _ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_parallel_size=1)

    blob = torch.load(a.device_latents)
    dev = blob["latents"].float()

    out = {"shape": list(dev.shape)}
    ref32 = _reference_latents(a.model_path, blob, torch.float32)
    ref16 = _reference_latents(a.model_path, blob, torch.bfloat16)
    if dev.shape != ref32.shape:
        dev = dev.reshape(ref32.shape)
    out["rel_cpu_bf16"] = _rel(ref16, ref32)
    out["rel_dev"] = _rel(dev, ref32)
    out["cos_dev"] = _cos(dev, ref32)
    out["band"] = 2.0 * out["rel_cpu_bf16"] + 0.005
    out["pass"] = bool(out["rel_dev"] <= out["band"] and out["cos_dev"] >= 0.999)
    print("[parity] " + json.dumps(out))
    if a.save_refs:
        refs = {"fp32": ref32, "bf16": ref16, "device": dev}
        if a.decode:
            from diffusers import AutoencoderKLFlux2

            for key, lat, dt in (
                ("fp32_image", ref32, torch.float32),
                ("bf16_image", ref16, torch.bfloat16),
            ):
                vae = AutoencoderKLFlux2.from_pretrained(
                    a.model_path, subfolder="vae", torch_dtype=dt
                ).eval()
                with torch.no_grad():
                    refs[key] = vae.decode(lat.to(dt), return_dict=False)[0].float()
                del vae
        torch.save(refs, a.save_refs)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
