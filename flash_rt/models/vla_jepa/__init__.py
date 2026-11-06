"""FlashRT — LeRobot VLA-JEPA (Qwen3-VL-2B backbone + DiT-B flow action head).

See :mod:`flash_rt.models.vla_jepa.pipeline_rtx` for the static RTX pipeline and
:mod:`flash_rt.frontends.torch.vla_jepa_rtx` for the weight/framework layer.
"""
from __future__ import annotations

__all__ = ["VlaJepaRtxPipeline"]


def __getattr__(name: str):
    if name == "VlaJepaRtxPipeline":
        from .pipeline_rtx import VlaJepaRtxPipeline

        return VlaJepaRtxPipeline
    raise AttributeError(name)
