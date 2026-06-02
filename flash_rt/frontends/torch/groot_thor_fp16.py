"""FlashRT -- GROOT N1.6 Thor full-FP16 reference frontend.

A non-quantized FP16 A/B reference for the FP8 production frontend
(:class:`GrootTorchFrontendThor`). It is the SAME fully-kernelized, CUDA-graph
pipeline as production — SigLIP, Qwen3 and the DiT all run through the
flash_rt kernels — with the per-tensor FP8 GEMMs swapped for FP16 GEMMs
(``use_fp8=False`` threads into ``siglip_forward``, ``CKernelQwen3`` and
``CKernelDiTHead``). No PyTorch matmul anywhere; the only difference from
production is the GEMM precision. Useful for validating FP8 cosine against a
kernel FP16 baseline.
"""

from __future__ import annotations

from flash_rt.frontends.torch.groot_thor import GrootTorchFrontendThor


class GrootTorchFrontendThorFP16(GrootTorchFrontendThor):
    """N1.6 Thor full-FP16 reference. Requires ``use_fp8=False``."""

    def __init__(self, checkpoint, num_views=2, autotune=3,
                 embodiment_tag="new_embodiment", use_fp8=False):
        if use_fp8:
            raise ValueError(
                "GrootTorchFrontendThorFP16 is a full-FP16 reference and "
                "requires use_fp8=False (use GrootTorchFrontendThor for FP8).")
        super().__init__(checkpoint, num_views=num_views, autotune=autotune,
                         embodiment_tag=embodiment_tag, use_fp8=False)
