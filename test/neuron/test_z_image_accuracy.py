# SPDX-License-Identifier: Apache-2.0
"""Z-Image / Z-Image-Turbo accuracy on a NeuronCore, in the three tiers of
docs/model-dev/onboarding-models.md (Step 4).

1. Component three-way (``assert_close_three_way``: fp32 CPU / bf16 CPU / bf16 Neuron) for the
   DiT (one call, a CFG pair with different caption lengths), the Qwen3 text encoder and the
   host-tiled VAE decode.
2. Single-step pipeline (``num_inference_steps=1``) latent, same three-way, base with CFG and
   Turbo without.
3. End-to-end against an independent CPU fp32 reference, with CPU bf16 as the floor. Turbo: one
   sample (final-latent rel-L2 and decoded-image SSIM). Base (CFG 4, chaotic): per-step
   teacher-forced error, composition-flip count and median error ratio over 15 samples (see the
   tier 3 comment for the gate and its cost). Plus a repeatability check (two Neuron runs). Set
   ``Z_IMAGE_REF_DIR`` to cache the CPU references (hours of CPU time when cold).

Weights: ``Z_IMAGE_WEIGHTS`` (base) and ``Z_IMAGE_TURBO_WEIGHTS`` (Turbo), local checkouts.
Geometry: ``Z_IMAGE_TEST_HW`` (default 512). Each test places its own components on one NeuronCore
and they are not released within a process, so run one test per process, e.g.
``pytest --collect-only -q test/neuron/test_z_image_accuracy.py`` and one ``pytest <node id>`` each.
"""

from __future__ import annotations

import os

import pytest
import torch

HW = int(os.environ.get("Z_IMAGE_TEST_HW", "512"))
PROMPT = "A red sports car parked on a wet city street at golden hour, photorealistic"
SEED = 7


def _neuron_available() -> bool:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        import libtorch_neuronx_lite  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device")


def _weights(env: str) -> str:
    path = os.environ.get(env, "")
    if not path or not os.path.isdir(os.path.join(path, "transformer")):
        pytest.skip(f"set {env} to a local checkout")
    return path


@pytest.fixture(scope="module")
def base_weights() -> str:
    return _weights("Z_IMAGE_WEIGHTS")


@pytest.fixture(scope="module")
def turbo_weights() -> str:
    return _weights("Z_IMAGE_TURBO_WEIGHTS")


@pytest.fixture(scope="module")
def backend() -> str:
    from vllm_neuron.envs import get_compile_backend_name

    return get_compile_backend_name()


# Only built when a NeuronCore is present: without the Neuron runtime the "neuron" device type does not
# exist and the module would fail at collection instead of skipping.
DEV = torch.device("neuron", 0) if _neuron_available() else None


def _three_way(baseline, expected, actual, name):
    from vllm_neuron.accuracy.testing import assert_close_three_way

    res = assert_close_three_way(baseline, expected, actual, name=name)
    print(f"[{name}] {res}")
    return res


# -- tier 1: components ------------------------------------------------------------------------


def test_dit_three_way(base_weights, backend):
    from diffusers import ZImageTransformer2DModel

    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImageTransformer

    torch.manual_seed(SEED)
    lat = HW // 8
    imgs = [torch.randn(16, 1, lat, lat) for _ in range(2)]
    caps = [
        torch.randn(41, 2560),
        torch.randn(9, 2560),
    ]  # positive / empty-negative caption lengths
    t = torch.tensor([0.4, 0.4])

    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        ref = ZImageTransformer2DModel.from_pretrained(
            base_weights, subfolder="transformer", torch_dtype=dt
        ).eval()
        with torch.no_grad():
            outs[dt] = ref(
                [i.to(dt) for i in imgs], t, [c.to(dt) for c in caps], return_dict=False
            )[0]
        del ref
    dit = NeuronZImageTransformer(base_weights, torch.bfloat16)
    dit.load()
    dit.to(DEV)
    dit.compile(backend)
    with torch.no_grad():
        got = dit([i.to(torch.bfloat16) for i in imgs], t, [c.to(torch.bfloat16) for c in caps])[0]
    _three_way(
        [o.float() for o in outs[torch.float32]],
        [o.float() for o in outs[torch.bfloat16]],
        [g.float() for g in got],
        "z_image_dit",
    )


def test_text_encoder_three_way(base_weights, backend):
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from vllm_omni_neuron.diffusion.models.z_image.pipeline_z_image import NeuronZImageTextEncoder

    tok = AutoTokenizer.from_pretrained(base_weights, subfolder="tokenizer")
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in (PROMPT, "")
    ]
    ti = tok(texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    mask = ti.attention_mask.bool()

    def valid(h):
        return [h[i][mask[i]].float() for i in range(h.shape[0])]

    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        ref = Qwen3ForCausalLM.from_pretrained(
            os.path.join(base_weights, "text_encoder"), torch_dtype=dt
        ).eval()
        with torch.no_grad():
            outs[dt] = valid(
                ref(
                    input_ids=ti.input_ids, attention_mask=mask, output_hidden_states=True
                ).hidden_states[-2]
            )
        del ref
    enc = NeuronZImageTextEncoder(base_weights, torch.bfloat16)
    enc.load()
    enc.to(DEV)
    enc.compile(backend)
    with torch.no_grad():
        got = valid(enc(input_ids=ti.input_ids, attention_mask=mask).hidden_states[-2])
    _three_way(outs[torch.float32], outs[torch.bfloat16], got, "z_image_text_encoder")


def _realistic_latent(weights: str, hw: int) -> torch.Tensor:
    """A latent from encoding a smooth synthetic image (fp32 CPU), scaled as the pipeline feeds the
    decoder. Random normal latents are far out of distribution and make the decoder ill-conditioned."""
    from diffusers import AutoencoderKL

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, hw), torch.linspace(-1, 1, hw), indexing="ij")
    img = (
        torch.stack([torch.sin(3 * xx), torch.cos(2 * yy), torch.sin(2 * (xx + yy))], 0)[None] * 0.8
    )
    vae = AutoencoderKL.from_pretrained(weights, subfolder="vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        z = vae.encode(img).latent_dist.mode()
    return z


def test_vae_decode_three_way(base_weights, backend, monkeypatch):
    """The host-tiled decode as served (same tiling on all three sides), at 1024 px (3x3 tiles).
    Served dtype: the config's ``force_upcast`` keeps the decoder in fp32 on the NeuronCore; the
    three-way's dtype reference is still bf16 on CPU."""
    from vllm_omni_neuron.diffusion.models.z_image.vae import NeuronAutoencoderKL

    z = _realistic_latent(base_weights, 1024)
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        monkeypatch.setenv("Z_IMAGE_VAE_DTYPE", str(dt).split(".")[-1])
        vae = NeuronAutoencoderKL.from_pretrained(base_weights, subfolder="vae")
        outs[dt] = vae.decode(z, return_dict=False)[0].float()
        del vae
    monkeypatch.delenv("Z_IMAGE_VAE_DTYPE")
    vae = NeuronAutoencoderKL.from_pretrained(
        base_weights, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.to(DEV)
    vae.compile(backend)
    got = vae.decode(z, return_dict=False)[0].float()
    _three_way(outs[torch.float32], outs[torch.bfloat16], got, "z_image_vae_decode")


# -- tier 2: single step -----------------------------------------------------------------------


def _ref_latent(weights, dtype, **kw):
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(weights, torch_dtype=dtype)
    with torch.no_grad():
        out = pipe(
            prompt=PROMPT, generator=torch.Generator().manual_seed(SEED), output_type="latent", **kw
        ).images
    del pipe
    return out.float()


def _device_pipe(weights, backend):
    """The served configuration: text encoder (fp32 residual), DiT and VAE on the NeuronCore.
    ``Z_IMAGE_TEST_TE_ON_HOST=1`` keeps the text encoder on the host (``text_encoder_on_host``)."""
    from vllm_omni_neuron.diffusion.models.z_image.standalone import build_diffusers_pipeline

    return build_diffusers_pipeline(
        weights,
        torch.bfloat16,
        device=DEV,
        compile_backend=backend,
        te_on_host=os.environ.get("Z_IMAGE_TEST_TE_ON_HOST") == "1",
    )


@pytest.mark.parametrize("variant,guidance", [("base", 4.0), ("turbo", 0.0)])
def test_single_step_three_way(variant, guidance, base_weights, turbo_weights, backend):
    weights = base_weights if variant == "base" else turbo_weights
    kw = dict(height=HW, width=HW, num_inference_steps=1, guidance_scale=guidance)
    want32 = _ref_latent(weights, torch.float32, **kw)
    want16 = _ref_latent(weights, torch.bfloat16, **kw)
    pipe = _device_pipe(weights, backend)
    with torch.no_grad():
        got = pipe(
            prompt=PROMPT, generator=torch.Generator().manual_seed(SEED), output_type="latent", **kw
        ).images.float()
    _three_way(want32, want16, got, f"z_image_{variant}_single_step")


# -- tier 3: end-to-end ------------------------------------------------------------------------
#
# References: diffusers ZImagePipeline on the CPU in fp32 (the reference) and bf16 (the floor), same
# prompt / seed / schedule, 512 x 512.
#
# Z-Image-Turbo (guidance-distilled, 9 steps) is gated on one sample: Neuron vs fp32 at most 2x the
# CPU bf16 error + 0.5 %, on the final latent (rel-L2) and on the decoded image (1 - SSIM).
#
# Z-Image base (CFG 4, 20 steps) is chaotic: a ~0.2 % structured latent difference at step 4 can send
# the trajectory to a different composition (camera angle, car pose), and an fp32 run continued from
# that latent follows it. Either bf16 path flips on some samples (Neuron: car / 7, car / 2; CPU bf16:
# fox / 42), so a single final latent says nothing about the device. Base is gated on:
#   (a) per-step teacher-forced error: fp32's exact DiT inputs at every step, each path's own text
#       embeddings; the combined (CFG) noise prediction vs fp32, Neuron <= 2x CPU bf16 + 0.5 % at every
#       step, on the three samples that flipped (test_base_teacher_forced_per_step);
#   (b) composition flips over BASE_SAMPLES (15): Neuron <= CPU bf16 + 2 (test_base_flip_rate);
#   (c) median over BASE_SAMPLES of the final-latent rel-L2 ratio Neuron / CPU bf16 <= 1.5 (same test).
# Repeatability (two Neuron runs of one request) is a separate test.
#
# Cost: the CPU references dominate. On a trn2.48xlarge host one 512 px base sample needs an fp32 and a
# bf16 CPU run (about 8 minutes together), so the 15-sample set is about 2 hours cold and the three
# teacher-forced traces add about 30 minutes. Set Z_IMAGE_REF_DIR to a persistent directory: references
# are written there on the first run and reused afterwards (about 40 MB for the full set), leaving only
# the Neuron runs (about 6 minutes for the five tier-3 tests with warm compile caches). Delete the
# directory after changing the reference pipeline, the prompts or HW.

E2E = {
    "turbo": dict(num_inference_steps=9, guidance_scale=0.0),
    "base": dict(num_inference_steps=20, guidance_scale=4.0),
}
FOX = "A red fox standing in fresh snow at golden hour, photorealistic, detailed fur"
TURBO_SAMPLE = ("car", PROMPT, SEED)
BASE_SAMPLES = [("car", PROMPT, s) for s in (7, 1, 2, 3, 4, 5, 6, 8)] + [
    ("fox", FOX, s) for s in (42, 7, 1, 2, 3, 4, 5)
]
TF_SAMPLES = [("car", PROMPT, 7), ("car", PROMPT, 2), ("fox", FOX, 42)]
# A final latent this far from fp32 is a different composition. Labelled by eye on BASE_SAMPLES:
# the three flips sit at 47-64 %, every other sample (either path) at <= 29 %.
FLIP_REL_L2 = 0.40


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM of two [3, H, W] images in [0, 1] (11x11 Gaussian window, sigma 1.5)."""
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    w = g.expand(3, 1, 11, 11).double()
    a, b = a[None].double(), b[None].double()

    def f(x):
        return F.conv2d(x, w, groups=3)

    mu_a, mu_b = f(a), f(b)
    var_a, var_b, cov = f(a * a) - mu_a**2, f(b * b) - mu_b**2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return s.mean().item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _e2e_kw(variant: str, prompt: str) -> dict:
    return dict(prompt=prompt, height=HW, width=HW, output_type="latent", **E2E[variant])


def _cache_path(name: str) -> str:
    ref_dir = os.environ.get("Z_IMAGE_REF_DIR", "")
    if not ref_dir:
        return ""
    os.makedirs(ref_dir, exist_ok=True)
    return os.path.join(ref_dir, name)


def _cpu_reference(
    weights: str, variant: str, dtype: torch.dtype, tag: str, prompt: str, seed: int
) -> torch.Tensor:
    """Final latent of diffusers ZImagePipeline on the CPU; cached in Z_IMAGE_REF_DIR."""
    path = _cache_path(f"z_image_{variant}_{tag}_{seed}_{HW}_{str(dtype).split('.')[-1]}.pt")
    if path and os.path.exists(path):
        return torch.load(path)
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(weights, torch_dtype=dtype)
    with torch.no_grad():
        lat = pipe(
            generator=torch.Generator().manual_seed(seed), **_e2e_kw(variant, prompt)
        ).images.float()
    del pipe
    if path:
        torch.save(lat, path)
    return lat


def _neuron_latent(pipe, variant: str, prompt: str, seed: int) -> torch.Tensor:
    with torch.no_grad():
        return pipe(
            generator=torch.Generator().manual_seed(seed), **_e2e_kw(variant, prompt)
        ).images.float()


def _decode(vae, lat: torch.Tensor) -> torch.Tensor:
    """Latent -> [3, H, W] image in [0, 1], with the same fp32 CPU VAE for every latent compared."""
    with torch.no_grad():
        z = lat.float() / vae.config.scaling_factor + vae.config.shift_factor
        im = vae.decode(z, return_dict=False)[0][0]
    return ((im.clamp(-1, 1) + 1) / 2).float()


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def test_e2e_turbo_vs_cpu_reference(turbo_weights, backend):
    from diffusers import AutoencoderKL

    t, p, s = TURBO_SAMPLE
    ref32 = _cpu_reference(turbo_weights, "turbo", torch.float32, t, p, s)
    ref16 = _cpu_reference(turbo_weights, "turbo", torch.bfloat16, t, p, s)
    dev = _neuron_latent(_device_pipe(turbo_weights, backend), "turbo", p, s)
    vae = AutoencoderKL.from_pretrained(
        turbo_weights, subfolder="vae", torch_dtype=torch.float32
    ).eval()
    img32, img16, imgd = (_decode(vae, x) for x in (ref32, ref16, dev))
    l2_dev, l2_floor = _rel(dev, ref32), _rel(ref16, ref32)
    ds_dev, ds_floor = 1 - _ssim(imgd, img32), 1 - _ssim(img16, img32)
    print(
        f"[z_image_turbo_e2e {t}/{s}] latent rel-L2 vs fp32: neuron {l2_dev:.4f} cpu-bf16 {l2_floor:.4f} "
        f"(bar {2 * l2_floor + 0.005:.4f}); 1-SSIM neuron {ds_dev:.4f} cpu-bf16 {ds_floor:.4f}"
    )
    assert l2_dev <= 2 * l2_floor + 0.005, (l2_dev, l2_floor)
    assert ds_dev <= 2 * ds_floor + 0.005, (ds_dev, ds_floor)


def _cfg(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    g = E2E["base"]["guidance_scale"]
    return pos.float() + g * (pos.float() - neg.float())


def _dit_cfg(
    dit,
    lat: torch.Tensor,
    t: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One CFG pair through ``dit`` (cond + uncond branch), combined as the pipeline does."""
    with torch.no_grad():
        out = dit(
            [lat.to(dtype), lat.to(dtype)], t, [pos.to(dtype), neg.to(dtype)], return_dict=False
        )[0]
    return _cfg(out[0], out[1])


def _embeds(pipe, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (prompt, empty negative) text embeddings a pipeline feeds its DiT."""
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(
            prompt, device=pipe._execution_device, do_classifier_free_guidance=True
        )
    return pos[0].float().cpu(), neg[0].float().cpu()


def _tf_reference(weights: str, tag: str, prompt: str, seed: int) -> dict:
    """The fp32 CPU base trajectory, recorded at every DiT call (latent, timestep, combined noise
    prediction), and the CPU bf16 teacher-forced error at each of those calls (the bf16 DiT on fp32's
    exact latent and timestep, with the bf16 pipeline's own text embeddings). Cached in Z_IMAGE_REF_DIR."""
    path = _cache_path(f"z_image_base_tf_{tag}_{seed}_{HW}.pt")
    if path and os.path.exists(path):
        return torch.load(path)
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(weights, torch_dtype=torch.float32)
    calls, fwd = [], pipe.transformer.forward

    def record(x, t, caps, *a, **k):
        out = fwd(x, t, caps, *a, **k)
        o = out[0] if isinstance(out, tuple) else out.sample
        calls.append(
            {"x": x[0].detach().float().clone(), "t": t.detach().clone(), "want": _cfg(o[0], o[1])}
        )
        return out

    pipe.transformer.forward = record
    with torch.no_grad():
        pipe(generator=torch.Generator().manual_seed(seed), **_e2e_kw("base", prompt))
    del pipe
    bf = ZImagePipeline.from_pretrained(weights, torch_dtype=torch.bfloat16)
    pos, neg = _embeds(bf, prompt)
    bf16_err = [
        _rel(_dit_cfg(bf.transformer, c["x"], c["t"], pos, neg, torch.bfloat16), c["want"])
        for c in calls
    ]
    del bf
    ref = {"calls": calls, "bf16_err": bf16_err}
    if path:
        torch.save(ref, path)
    return ref


def test_base_teacher_forced_per_step(base_weights, backend):
    """Gate (a): the Neuron DiT's own error at every step of a chaotic trajectory stays at the CPU bf16 level."""
    pipe = _device_pipe(base_weights, backend)
    worst = []
    for t, p, s in TF_SAMPLES:
        ref = _tf_reference(base_weights, t, p, s)
        pos, neg = _embeds(pipe, p)
        for i, (c, floor) in enumerate(zip(ref["calls"], ref["bf16_err"])):
            err = _rel(
                _dit_cfg(pipe.transformer, c["x"], c["t"], pos, neg, torch.bfloat16), c["want"]
            )
            print(
                f"[z_image_base_tf {t}/{s} step {i}] combined rel-L2 vs fp32: neuron {err:.4f} "
                f"cpu-bf16 {floor:.4f} (bar {2 * floor + 0.005:.4f})"
            )
            worst.append((err - (2 * floor + 0.005), f"{t}/{s}", i, err, floor))
    margin, case, i, err, floor = max(worst)
    print(
        f"[z_image_base_tf] worst step {case} #{i}: neuron {err:.4f} cpu-bf16 {floor:.4f} (margin {margin:+.4f})"
    )
    assert margin <= 0, (case, i, err, floor)


def test_base_flip_rate(base_weights, backend):
    """Gates (b) and (c) over BASE_SAMPLES: composition flips and the median error ratio."""
    refs = {
        (t, s): (
            _cpu_reference(base_weights, "base", torch.float32, t, p, s),
            _cpu_reference(base_weights, "base", torch.bfloat16, t, p, s),
        )
        for t, p, s in BASE_SAMPLES
    }
    pipe = _device_pipe(base_weights, backend)
    flips = {"neuron": [], "cpu-bf16": []}
    ratios = []
    for t, p, s in BASE_SAMPLES:
        ref32, ref16 = refs[(t, s)]
        l2_dev, l2_floor = _rel(_neuron_latent(pipe, "base", p, s), ref32), _rel(ref16, ref32)
        ratios.append(l2_dev / l2_floor)
        for path, l2 in (("neuron", l2_dev), ("cpu-bf16", l2_floor)):
            if l2 > FLIP_REL_L2:
                flips[path].append(f"{t}/{s}")
        print(
            f"[z_image_base_e2e {t}/{s}] latent rel-L2 vs fp32: neuron {l2_dev:.4f} cpu-bf16 {l2_floor:.4f} "
            f"ratio {ratios[-1]:.2f}"
        )
    n_dev, n_floor = len(flips["neuron"]), len(flips["cpu-bf16"])
    med = _median(ratios)
    print(
        f"[z_image_base_e2e {len(BASE_SAMPLES)} samples] flips (rel-L2 > {FLIP_REL_L2}): neuron {n_dev} "
        f"{flips['neuron']}, cpu-bf16 {n_floor} {flips['cpu-bf16']} (bar {n_floor + 2}); "
        f"median ratio neuron / cpu-bf16 {med:.2f} (bar 1.5)"
    )
    assert n_dev <= n_floor + 2, flips
    assert med <= 1.5, ratios


@pytest.mark.parametrize("variant", ["turbo", "base"])
def test_e2e_repeatability(variant, base_weights, turbo_weights, backend):
    """Two Neuron runs of the same request give the same final latent."""
    weights = base_weights if variant == "base" else turbo_weights
    pipe = _device_pipe(weights, backend)
    a, b = (_neuron_latent(pipe, variant, PROMPT, SEED) for _ in range(2))
    print(f"[z_image_{variant}_repeat] rel-L2 run2 vs run1 {_rel(b, a):.2e}")
    assert _rel(b, a) < 1e-3, _rel(b, a)
