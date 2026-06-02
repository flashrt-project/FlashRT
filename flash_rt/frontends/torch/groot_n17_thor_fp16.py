"""FlashRT -- GROOT N1.7 Thor FP16 *backbone* reference frontend.

This is NOT a full-model FP16 path. It is an A/B reference for the FP8 Thor
serving frontend (:class:`GrootN17TorchFrontendThorFP8`) that runs only the
vision/language *backbone* (ViT -> DeepStack -> LLM -> vlln -> VL-self-attn)
in FP16: every backbone-stage GEMM goes through the cuBLASLt ``fp16_nn`` path
on the shadow weights instead of per-tensor FP8, with no activation
calibration. The DiT action head is unchanged — it runs the same production
FP8 CUDA graph as the default frontend.

Useful for validating the FP8 backbone cosine against a kernel FP16 baseline.
"""

from __future__ import annotations

from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
    GrootN17TorchFrontendThorFP8,
)


class GrootN17TorchFrontendThorFP16(GrootN17TorchFrontendThorFP8):
    """N1.7 Thor FP16 *backbone* reference (ViT/DeepStack/LLM/VL-self-attn).

    Backbone GEMMs only run in FP16; the DiT action head stays on the
    production FP8 graph. This is not a full-model FP16 path.

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
