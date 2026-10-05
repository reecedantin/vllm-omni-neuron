# SPDX-License-Identifier: Apache-2.0
"""NeuronCore-generation detection and NKI dispatch gates (CPU only).

Pins the NC-v3 (Trn2) behaviour so NC-v2 support can never silently turn NKI off on Trn2.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni_neuron import nc_generation as ncg

_GEN_ENVS = ("VLLM_OMNI_NEURON_CORE_GEN", "COSMOS3_NEURON_CORE_GEN")
_IMPL_ENVS = ("VLLM_OMNI_NEURON_ATTN_IMPL", "COSMOS3_EDGE_ATTN_IMPL")


@pytest.fixture(autouse=True)
def _clean_detection(monkeypatch):
    for name in _GEN_ENVS + _IMPL_ENVS:
        monkeypatch.delenv(name, raising=False)
    ncg.neuron_core_generation.cache_clear()
    yield
    ncg.neuron_core_generation.cache_clear()


def _fake_target(monkeypatch, target):
    from vllm_omni_neuron import lite_compat

    def get_platform_target():
        if isinstance(target, Exception):
            raise target
        return target

    monkeypatch.setattr(lite_compat, "get_platform_target", get_platform_target)


def _fake_sysfs(tmp_path, monkeypatch, **files):
    arch = tmp_path / "neuron0" / "info" / "architecture"
    arch.mkdir(parents=True)
    for name, value in files.items():
        (arch / name).write_text(value + "\n")
    monkeypatch.setattr(ncg, "SYSFS_ARCH_GLOB", str(tmp_path / "neuron*" / "info" / "architecture"))


@pytest.mark.parametrize(
    "target,gen",
    [
        ("trn2", 3),
        ("trn2n", 3),
        ("TRN2", 3),
        ("trn3", 4),
        ("inf2", 2),
        ("trn1", 2),
        ("trn1n", 2),
        ("gpu", None),
    ],
)
def test_generation_from_target(target, gen):
    assert ncg.generation_from_target(target) == gen


@pytest.mark.parametrize("target,gen", [("trn2", 3), ("trn3", 4), ("inf2", 2), ("trn1n", 2)])
def test_detect_from_platform_target(monkeypatch, target, gen):
    _fake_target(monkeypatch, target)
    assert ncg.neuron_core_generation() == gen
    assert ncg.supports_nki() is (gen >= 3)


@pytest.mark.parametrize("env", _GEN_ENVS)
def test_env_override_wins(monkeypatch, env):
    _fake_target(monkeypatch, "trn2")
    monkeypatch.setenv(env, "2")
    assert ncg.neuron_core_generation() == 2
    assert not ncg.supports_nki()


@pytest.mark.parametrize(
    "files,gen",
    [
        ({"arch_type": "NDv3", "device_name": "Trainium2", "instance_type": "Trn2"}, 3),
        ({"arch_type": "NDv2", "device_name": "Inferentia2", "instance_type": "Inf2"}, 2),
        ({"device_name": "Trainium2"}, 3),
        ({"instance_type": "Trn1"}, 2),
    ],
)
def test_sysfs_fallback_when_runtime_unavailable(tmp_path, monkeypatch, files, gen):
    _fake_target(monkeypatch, RuntimeError("NRT not initialised"))
    _fake_sysfs(tmp_path, monkeypatch, **files)
    assert ncg.neuron_core_generation() == gen


def test_sysfs_fallback_on_unknown_target(tmp_path, monkeypatch):
    _fake_target(monkeypatch, "mystery")
    _fake_sysfs(tmp_path, monkeypatch, arch_type="NDv3")
    assert ncg.neuron_core_generation() == 3


def test_undetectable_defaults_to_v2_with_warning(tmp_path, monkeypatch, caplog):
    _fake_target(monkeypatch, ImportError("no lite"))
    monkeypatch.setattr(ncg, "SYSFS_ARCH_GLOB", str(tmp_path / "absent*"))
    with caplog.at_level("WARNING", logger=ncg.__name__):
        assert ncg.neuron_core_generation() == 2
    assert "undetectable" in caplog.text


def test_detection_is_cached(monkeypatch):
    calls = []

    def get_platform_target():
        calls.append(1)
        return "trn2"

    from vllm_omni_neuron import lite_compat

    monkeypatch.setattr(lite_compat, "get_platform_target", get_platform_target)
    for _ in range(3):
        assert ncg.neuron_core_generation() == 3
    assert len(calls) == 1


def test_use_nki_kernels_gates(monkeypatch):
    import vllm_neuron.utils.neuron_utils as nu

    monkeypatch.setattr(nu, "can_run_kernel", lambda t: True)
    _fake_target(monkeypatch, "trn2")
    assert ncg.use_nki_kernels()  # NC-v3, kernels enabled: unchanged Trn2 behaviour
    assert not ncg.use_nki_kernels(torch.zeros(1))  # CPU tensor: never NKI
    for env in _IMPL_ENVS:
        monkeypatch.setenv(env, "torch")
        assert not ncg.use_nki_kernels()
        monkeypatch.delenv(env)
    ncg.neuron_core_generation.cache_clear()
    _fake_target(monkeypatch, "inf2")
    assert not ncg.use_nki_kernels()  # NC-v2: never NKI, whatever vllm_neuron says


def test_use_nki_kernels_defers_to_vllm_neuron(monkeypatch):
    import vllm_neuron.utils.neuron_utils as nu

    monkeypatch.setattr(nu, "can_run_kernel", lambda t: False)
    _fake_target(monkeypatch, "trn2")
    assert not ncg.use_nki_kernels()


@pytest.mark.parametrize(
    "target,expected", [("trn2", True), ("trn3", True), ("inf2", False), ("trn1", False)]
)
def test_wan_can_run_kernel_gate(monkeypatch, target, expected):
    from vllm_omni_neuron.diffusion.models.wan2_2 import wan2_2_transformer as wt

    monkeypatch.setattr(wt, "_can_run_kernel_impl", lambda: (lambda tensor: True))
    _fake_target(monkeypatch, target)
    assert wt.can_run_kernel("neuron") is expected


def test_wan_can_run_kernel_respects_device_gate(monkeypatch):
    from vllm_omni_neuron.diffusion.models.wan2_2 import wan2_2_transformer as wt

    monkeypatch.setattr(wt, "_can_run_kernel_impl", lambda: (lambda tensor: False))
    _fake_target(monkeypatch, "trn2")
    assert wt.can_run_kernel(torch.zeros(1)) is False


def test_generation_constant_under_compile(monkeypatch):
    """The gate is baked into traced graphs, not traced (assume_constant_result)."""
    _fake_target(monkeypatch, "trn2")

    def f(x):
        return x + 1 if ncg.supports_nki() else x - 1

    out = torch.compile(f, backend="eager", fullgraph=True)(torch.zeros(2))
    torch.testing.assert_close(out, torch.ones(2))


@pytest.mark.parametrize("target,gb", [("trn2", 24), ("trn3", 36), ("inf2", 16), ("trn1", 16)])
def test_platform_hbm_per_core(monkeypatch, target, gb):
    from vllm_omni_neuron import platform as plat

    monkeypatch.setattr(plat, "get_platform_target", lambda: target)
    assert plat.NeuronOmniPlatform.get_device_total_memory(0) == gb * 1024**3


def test_platform_hbm_defaults_to_trn2(monkeypatch):
    from vllm_omni_neuron import platform as plat

    def boom():
        raise RuntimeError("undetectable")

    monkeypatch.setattr(plat, "get_platform_target", boom)
    assert plat.NeuronOmniPlatform.get_device_total_memory(0) == 24 * 1024**3


def test_platform_cpu_mode_reports_one_device(monkeypatch):
    from vllm_neuron import envs

    from vllm_omni_neuron import platform as plat

    monkeypatch.setattr(envs, "VLLM_NEURON_CPU_MODE", True, raising=False)
    assert plat.NeuronOmniPlatform.get_device_count() == 1
