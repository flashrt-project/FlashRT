"""FlashRT -- Qwen3.6-27B model pipelines.

Per the unified pipeline_<hw>.py contract:
    pipeline_rtx.py   - RTX SM120 Qwen36Pipeline (Blackwell consumer)
    pipeline_rocm.py  - ROCm Qwen36 metadata / owned-buffer staging

Phase plan (see docs/qwen36_integration.md when written; for now the
adaptation plan lives in project memory):
    Phase 1: PyTorch-eager wrapper around HF AutoModelForCausalLM
             (this commit). No fvk kernels yet -- proves the frontend
             contract end-to-end and locks down the Phase-0 cosine
             fixture as the regression baseline.
    Phase 2: replace full-attn (16 layers) + MLP (64 layers) + RMSNorm
             + RoPE + SwiGLU with fvk kernel calls. Linear-attn stays
             PyTorch eager.
    Phase 3: write csrc/kernels/{gated_deltanet,causal_conv1d_qwen36}
             + csrc/gemm/fp8_block128_gemm. Replace the remaining 48
             linear-attn layers and the FP8 GEMM path.
    Phase 4: KV cache + sampling + CUDA graph + clean decode loop.
    Phase 5: vision tower (only if multimodal).
    Phase 6: MTP speculative decode (optional).
"""

from flash_rt.models.qwen36.pipeline_rocm import Qwen36RocmDims
from flash_rt.models.qwen36.pipeline_rtx import Qwen36Pipeline

__all__ = [
    'Qwen36Pipeline',
    'Qwen36RocmDims',
]
