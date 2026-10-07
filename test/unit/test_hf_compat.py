# SPDX-License-Identifier: Apache-2.0
"""vllm-omni discovers diffusers modular checkpoints (modular_model_index.json). CPU only."""

from __future__ import annotations

import json

import pytest

import vllm_omni_neuron  # noqa: F401  installs the fallback at import
from vllm_omni_neuron import hf_compat


def _ckpt(root, index_name, class_name):
    root.mkdir()
    (root / index_name).write_text(
        json.dumps({"_class_name": class_name, "_diffusers_version": "0.36.0"})
    )
    (root / "transformer").mkdir()
    (root / "transformer" / "config.json").write_text(json.dumps({"num_layers": 2}))
    return str(root)


@pytest.fixture
def modular(tmp_path):
    return _ckpt(tmp_path / "modular", "modular_model_index.json", "MiniMaxH3ModularPipeline")


@pytest.fixture
def both(tmp_path):
    path = _ckpt(tmp_path / "both", "model_index.json", "RealPipeline")
    (tmp_path / "both" / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "Other"})
    )
    return path


def test_installed_at_import_and_idempotent():
    import vllm.transformers_utils.config as cfg_mod

    fn = cfg_mod.get_hf_file_to_dict
    assert getattr(fn, hf_compat._MARK, None) is not None
    assert hf_compat.install_modular_model_index_fallback()
    assert cfg_mod.get_hf_file_to_dict is fn


def test_vllm_omni_resolves_modular_checkpoint(modular):
    from vllm_omni.diffusion.data import resolve_model_class_name
    from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

    assert resolve_model_class_name(modular) == "MiniMaxH3ModularPipeline"
    assert is_diffusion_model(modular)


def test_model_index_still_wins(both):
    from vllm_omni.diffusion.data import resolve_model_class_name

    assert resolve_model_class_name(both) == "RealPipeline"


def test_other_files_untouched(modular):
    from vllm.transformers_utils.config import get_hf_file_to_dict

    assert get_hf_file_to_dict("transformer/config.json", modular) == {"num_layers": 2}
    assert get_hf_file_to_dict("config.json", modular) is None


def test_uninstall_restores(modular):
    from vllm_omni.diffusion.utils import hf_utils

    hf_compat.uninstall_modular_model_index_fallback()
    try:
        assert hf_utils.get_hf_file_to_dict("model_index.json", modular) is None
    finally:
        hf_compat.install_modular_model_index_fallback()
    assert (
        hf_utils.get_hf_file_to_dict("model_index.json", modular)["_class_name"]
        == "MiniMaxH3ModularPipeline"
    )
