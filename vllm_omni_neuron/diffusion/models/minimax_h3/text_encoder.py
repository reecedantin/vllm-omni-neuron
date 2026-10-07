# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's text conditioner: Qwen3-VL-32B, read at ``hidden_states[50]``.

Runs on the host CPU in this first version (one process, the output-owner rank) and is broadcast to the other TP
ranks. Only the first ``text_encoder_layer + 1`` decoder layers are instantiated: ``hidden_states[50]`` is the raw
output of decoder layer 50, and a stack truncated to exactly 50 layers would return it post-norm (diffusers'
``get_qwen3vl_prompt_embeds`` refuses that for the same reason), so one extra layer is the price of reusing the
reference encode verbatim.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from ._vendor.mp_encoders import get_qwen3vl_prompt_embeds

logger = logging.getLogger(__name__)


class H3TextEncoder:
    def __init__(self, model_path: str, layer: int, dtype: torch.dtype = torch.bfloat16):
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )

        t0 = time.time()
        te_dir = os.path.join(model_path, "text_encoder")
        cfg = AutoConfig.from_pretrained(te_dir)
        self.layer = layer
        keep = min(cfg.text_config.num_hidden_layers, layer + 1)
        cfg.text_config.num_hidden_layers = keep
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            te_dir, config=cfg, dtype=dtype
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self.processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_path, "processor"))
        logger.info(
            "MiniMax-H3 text encoder: %d/%d Qwen3-VL layers loaded in %.1fs",
            keep,
            AutoConfig.from_pretrained(te_dir).text_config.num_hidden_layers,
            time.time() - t0,
        )

    def token_ids(self, prompt: str) -> list[int]:
        """t2va presentation: the prompt verbatim, no chat template, no special tokens."""
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """-> ``(1, num_tokens, text_dim)`` in the encoder's dtype."""
        ids = self.token_ids(prompt)
        return get_qwen3vl_prompt_embeds(
            self.model,
            self.processor,
            ids,
            {},
            text_encoder_layer=self.layer,
            device=torch.device("cpu"),
            dtype=self.model.dtype,
        )
