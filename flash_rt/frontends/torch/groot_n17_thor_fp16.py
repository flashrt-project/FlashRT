"""FlashRT -- GROOT N1.7 Thor full-FP16 reference frontend.

A fully non-quantized A/B reference for the FP8 Thor serving frontend
(:class:`GrootN17TorchFrontendThorFP8`). It runs the identical fully-kernelized
pipeline with **no FP8 anywhere**:
  * the backbone (ViT -> DeepStack -> LLM -> vlln -> VL-self-attn) GEMMs go
    through the cuBLASLt ``fp16_nn`` path on the shadow weights
    (``_KBB_USE_FP8 = False``);
  * the DiT action head stays bf16 — the FP8 FFN / fused-QKV path is disabled
    (``_DIT_USE_FP8 = False``), so the captured DiT graph uses bf16 GEMMs.
There is no activation calibration and no PyTorch matmul/attention on the
feature path. Useful as a non-quantized accuracy baseline for the whole model.
"""

from __future__ import annotations

from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
    GrootN17TorchFrontendThorFP8,
)


class GrootN17TorchFrontendThorFP16(GrootN17TorchFrontendThorFP8):
    """N1.7 Thor full-FP16 reference (no FP8 anywhere).

    Backbone GEMMs run in FP16 (``_KBB_USE_FP8 = False``) and the DiT action
    head stays bf16 (``_DIT_USE_FP8 = False``) — a fully non-quantized A/B
    baseline for the whole model.

    Flips the shared ``_run_kernel_backbone`` to feed every backbone stage its
    fp16 shadow weights through ``fp16_nn``. The LLM runs fully fp16 —
    ``PROTECT_LLM_FP16`` is forced to all 16 layers here (the FP8 frontend
    defaults it to empty), since this reference computes no FP8 activation
    scales.
    """

    _KBB_USE_FP8 = False
    _DIT_USE_FP8 = False
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
