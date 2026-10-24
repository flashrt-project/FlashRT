"""FlashRT -- PyTorch frontend for Qwen/Qwen3-4B (dense BF16) on Jetson Thor.

Batch-1 causal prefill -> logits that matches HuggingFace ``Qwen3ForCausalLM``
in BF16. Same layer sequence as RTX (``flash_rt.models.qwen3.prefill_bf16``),
bound to the Thor target in ``flash_rt.models.qwen3.pipeline_thor``: the
vendored FlashAttention-4 forward for attention, cuBLASLt GEMMs and the
``*_hf_bf16`` norm / RoPE / SiLU kernels::

    from flash_rt.frontends.torch.qwen3_4b_thor import Qwen3_4BTorchFrontendThor
    fe = Qwen3_4BTorchFrontendThor("<path-to-Qwen3-4B>")
    logits = fe.forward_prefill([...token ids...])   # [S, vocab] bf16

See ``docs/qwen3_4b_bf16.md``.
"""
from __future__ import annotations

from flash_rt.frontends.torch._qwen3_4b_bf16_runtime import Qwen3_4BBf16Frontend
from flash_rt.models.qwen3.pipeline_thor import QWEN3_4B_THOR

__all__ = ["Qwen3_4BTorchFrontendThor"]


class Qwen3_4BTorchFrontendThor(Qwen3_4BBf16Frontend):
    """Qwen3-4B BF16 prefill on Jetson AGX Thor (SM110), FA4 attention."""

    TARGET = QWEN3_4B_THOR
