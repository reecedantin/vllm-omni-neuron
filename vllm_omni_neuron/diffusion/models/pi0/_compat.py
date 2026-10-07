# SPDX-License-Identifier: Apache-2.0
"""vllm-omni 0.24 compatibility for LeRobot checkpoints.

``OmniDiffusionConfig.enrich_config`` in vllm-omni 0.24 resolves a non-diffusers checkpoint from
``config.json``'s ``model_type`` / ``architectures``. A LeRobot policy ``config.json`` carries
neither -- only ``type`` (``pi0`` / ``pi05`` / ``pi052``) -- so engine start-up raises
``model_index.json not found`` before any pipeline is built. Upstream vllm-omni main resolves
``type: pi0`` / ``type: pi05`` in that same method (``Pi0Pipeline`` / ``Pi05Pipeline``); this
backports that branch, plus ``pi052``, which shares the pi0.5 pipeline.

The patch only acts when the original method raises and the checkpoint's ``config.json`` has one
of these ``type`` values; every other model takes the original path unchanged. It is installed
when this package is imported (the plugin imports every model package at registration, in every
engine process) and is a no-op on a vllm-omni that already handles these types.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_LEROBOT_TYPES = {"pi0": "Pi0Pipeline", "pi05": "Pi05Pipeline", "pi052": "Pi05Pipeline"}


def install() -> None:
    try:
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    except ImportError:  # vllm-omni not importable (docs builds, partial installs)
        return
    original = OmniDiffusionConfig.enrich_config
    if getattr(original, "_pi0_lerobot_compat", False):
        return

    @functools.wraps(original)
    def enrich_config(self) -> None:
        try:
            return original(self)
        except (OSError, ValueError, FileNotFoundError):
            from vllm.transformers_utils.config import get_hf_file_to_dict

            cfg = get_hf_file_to_dict("config.json", self.model) or {}
            pipeline = _LEROBOT_TYPES.get(cfg.get("type"))
            if pipeline is None:
                raise
            if self.model_class_name is None:
                self.model_class_name = pipeline
            self.set_tf_model_config(TransformerConfig())
            self.update_multimodal_support()
            logger.info(
                "LeRobot %s checkpoint resolved to %s", cfg.get("type"), self.model_class_name
            )

    enrich_config._pi0_lerobot_compat = True
    OmniDiffusionConfig.enrich_config = enrich_config
