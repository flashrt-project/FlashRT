"""FlashRT -- PyTorch frontend for Qwen/Qwen3-4B (dense BF16) on RTX.

Batch-1 causal prefill -> logits that matches HuggingFace ``Qwen3ForCausalLM``
in BF16. The layer sequence is ``flash_rt.models.qwen3.prefill_bf16``; each
class here binds one validated device target from
``flash_rt.models.qwen3.pipeline_rtx`` (FA2 causal attention, cuBLASLt
GEMMs, the ``*_hf_bf16`` norm / RoPE / SiLU kernels)::

    from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx
    fe = Qwen3_4BTorchFrontendRtx("<path-to-Qwen3-4B>", device="cuda:0")
    logits = fe.forward_prefill([...token ids...])   # [S, vocab] bf16

A text decoder, so it is constructed directly rather than through
``load_model``. See ``docs/qwen3_4b_bf16.md``.
"""
from __future__ import annotations

from flash_rt.frontends.torch._qwen3_4b_bf16_runtime import Qwen3_4BBf16Frontend
from flash_rt.models.qwen3.pipeline_rtx import QWEN3_4B_RTX_SM89, QWEN3_4B_RTX_SM120

__all__ = ["Qwen3_4BTorchFrontendRtx", "Qwen3_4BTorchFrontendRtxSm89"]


class Qwen3_4BTorchFrontendRtx(Qwen3_4BBf16Frontend):
    """Qwen3-4B BF16 prefill on RTX 50-series (SM120), FA2 attention."""

    TARGET = QWEN3_4B_RTX_SM120


class Qwen3_4BTorchFrontendRtxSm89(Qwen3_4BBf16Frontend):
    """Qwen3-4B BF16 prefill on RTX 40-series (SM89), FA2 attention."""

    TARGET = QWEN3_4B_RTX_SM89
