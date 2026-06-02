"""FlashRT -- GROOT N1.7 Thor full-FP16 reference frontend.

A non-quantized FP16 A/B reference for the FP8 Thor serving frontend
(:class:`GrootN17TorchFrontendThorFP8`). It runs the identical fully-kernelized
ViT -> DeepStack -> LLM -> vlln -> VL-self-attn backbone, with every stage's
GEMMs executed through the cuBLASLt ``fp16_nn`` path on the shadow weights
instead of per-tensor FP8. There is no PyTorch matmul/attention on the feature
path and no activation calibration; the only difference from the FP8 frontend
is GEMM precision. The DiT action head is already bf16 in both frontends.

Useful for validating the FP8 backbone cosine against a kernel FP16 baseline.
"""

from __future__ import annotations

from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
    GrootN17TorchFrontendThorFP8,
)


class GrootN17TorchFrontendThorFP16(GrootN17TorchFrontendThorFP8):
    """N1.7 Thor full-FP16 reference backbone (ViT/DeepStack/LLM/VL-self-attn).

    Flips the shared ``_run_kernel_backbone`` to feed every stage its fp16
    shadow weights through ``fp16_nn`` (``_KBB_USE_FP8 = False``). The LLM
    runs fully fp16 — ``PROTECT_LLM_FP16`` is forced to all 16 layers here
    (the FP8 frontend defaults it to empty), since this reference computes
    no FP8 activation scales.
    """

    _KBB_USE_FP8 = False
    PROTECT_LLM_FP16 = tuple(range(16))

    def _ensure_act_scales(self, aux: dict) -> None:
        """No activation calibration in the FP16 reference.

        The FP8 frontend calibrates per-tensor activation scales here and frees
        the fp16 shadow weights afterwards. The FP16 path uses no activation
        scales, so this only makes sure the shadow weights — the fp16 GEMM
        source — stay resident.
        """
        if not hasattr(self, "_fp16_shadow_weights"):
            self._load_fp16_shadow_weights()
