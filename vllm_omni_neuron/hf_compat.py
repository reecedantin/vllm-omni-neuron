# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-format compatibility shims for vllm-omni's model discovery.

Diffusers *modular* pipelines (for example the FastH3 / MiniMax-H3 releases) ship
``modular_model_index.json`` instead of ``model_index.json``. It carries the same ``_class_name`` /
``_diffusers_version`` fields, but vllm-omni 0.24 only reads ``model_index.json``
(``OmniDiffusionConfig.enrich_config``, ``resolve_model_class_name``, ``is_diffusion_model``), so
``Omni(model=...)`` / ``vllm serve`` fail with "Could not find config.json or model_index.json".

:func:`install_modular_model_index_fallback` wraps vllm's ``get_hf_file_to_dict``: when
``model_index.json`` is absent it returns ``modular_model_index.json`` instead. Every other file and
every checkpoint that has a ``model_index.json`` behaves exactly as before.
"""

from __future__ import annotations

import functools
import logging
import sys

logger = logging.getLogger(__name__)

MODEL_INDEX = "model_index.json"
MODULAR_MODEL_INDEX = "modular_model_index.json"
_MARK = "_vllm_omni_neuron_modular_fallback"


def _wrap(orig):
    @functools.wraps(orig)
    def get_hf_file_to_dict(file_name, model, revision="main"):
        out = orig(file_name, model, revision)
        if out is None and file_name == MODEL_INDEX:
            out = orig(MODULAR_MODEL_INDEX, model, revision)
            if out is not None:
                logger.debug("%s: using %s in place of %s", model, MODULAR_MODEL_INDEX, MODEL_INDEX)
        return out

    setattr(get_hf_file_to_dict, _MARK, orig)
    return get_hf_file_to_dict


def install_modular_model_index_fallback() -> bool:
    """Install the fallback (idempotent). Returns False when vllm's helper is unavailable."""
    try:
        import vllm.transformers_utils.config as cfg_mod
    except ImportError:
        return False
    orig = cfg_mod.get_hf_file_to_dict
    if getattr(orig, _MARK, None) is not None:
        return True
    wrapped = _wrap(orig)
    cfg_mod.get_hf_file_to_dict = wrapped
    # Modules that already did ``from vllm.transformers_utils.config import get_hf_file_to_dict``.
    for name, mod in list(sys.modules.items()):
        if (
            mod is not None
            and name.startswith(("vllm_omni", "vllm."))
            and getattr(mod, "get_hf_file_to_dict", None) is orig
        ):
            mod.get_hf_file_to_dict = wrapped
    return True


def uninstall_modular_model_index_fallback() -> None:
    """Undo :func:`install_modular_model_index_fallback` (tests)."""
    try:
        import vllm.transformers_utils.config as cfg_mod
    except ImportError:
        return
    wrapped = cfg_mod.get_hf_file_to_dict
    orig = getattr(wrapped, _MARK, None)
    if orig is None:
        return
    cfg_mod.get_hf_file_to_dict = orig
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "get_hf_file_to_dict", None) is wrapped:
            mod.get_hf_file_to_dict = orig
