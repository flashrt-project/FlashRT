"""FlashRT -- MiniMax-Remover video inpainting pipeline.

MiniMax-Remover is a flow-matching video inpainting Transformer used for
subtitle / object removal. This package ships the NVFP4 (W4A4)
kernelized inference pipeline (``MiniMaxRemoverPipeline``): the
transformer Linears are rewritten as NVFP4 GEMMs over the generic
FlashRT SM120 kernels, the per-block norm / gate / residual / gelu ops
become fused Triton kernels, attention moves to FA2 / SageAttention, and
the N-step flow-matching loop is captured as a single CUDA Graph. The VAE
encode / decode run unchanged from the loaded diffusers model.
"""

from flash_rt.models.minimax_remover.pipeline import (
    MiniMaxRemoverPipeline,
    _load_kernels,
)

__all__ = ["MiniMaxRemoverPipeline"]
