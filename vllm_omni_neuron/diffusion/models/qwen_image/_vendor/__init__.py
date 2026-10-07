# SPDX-License-Identifier: Apache-2.0
"""Vendored Qwen-Image 2.1 modeling code from huggingface/diffusers (Apache-2.0).

Source: https://github.com/huggingface/diffusers at 8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd
(``models/transformers/transformer_qwenimage21.py``, ``models/autoencoders/autoencoder_kl_qwenimage21.py``,
``pipelines/qwenimage21/pipeline_qwenimage21.py``). The installed diffusers release does not ship
Qwen-Image 2.1 yet. Local changes: relative imports rewritten to absolute ``diffusers`` imports.
These files are the CPU reference for the Neuron implementation; drop them once a diffusers
release includes Qwen-Image 2.1.
"""
