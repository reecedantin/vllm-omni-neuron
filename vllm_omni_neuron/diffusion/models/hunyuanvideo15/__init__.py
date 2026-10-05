# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of the HunyuanVideo-1.5 diffusion pipelines (480p / 720p T2V and I2V)."""

from .pipeline_hunyuanvideo15 import (
    PIPELINE_REGISTRY as _T2V_PIPELINES,
)
from .pipeline_hunyuanvideo15 import (
    NeuronHunyuanVideo15Pipeline,
    get_hunyuan_video_15_post_process_func,
)
from .pipeline_hunyuanvideo15_i2v import (
    PIPELINE_REGISTRY as _I2V_PIPELINES,
)
from .pipeline_hunyuanvideo15_i2v import (
    NeuronHunyuanVideo15I2VPipeline,
    get_hunyuan_video_15_i2v_post_process_func,
    get_hunyuan_video_15_i2v_pre_process_func,
)

PIPELINE_REGISTRY = [*_T2V_PIPELINES, *_I2V_PIPELINES]

__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronHunyuanVideo15I2VPipeline",
    "NeuronHunyuanVideo15Pipeline",
    "get_hunyuan_video_15_i2v_post_process_func",
    "get_hunyuan_video_15_i2v_pre_process_func",
    "get_hunyuan_video_15_post_process_func",
]
