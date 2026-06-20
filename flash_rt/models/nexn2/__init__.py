"""FlashRT -- Nex-N2-mini model pipelines.

Per the unified pipeline_<hw>.py contract:
    pipeline_rtx.py  - RTX SM120 Nexn2Pipeline (Blackwell consumer)

Nex-N2-mini is the MoE sibling of the dense qwen36 family
(architectures=Qwen3_5MoeForConditionalGeneration / model_type=qwen3_5_moe):
hybrid Gated-DeltaNet + softmax-attention with a fine-grained 256-expert MoE
FFN, a native 1-layer MTP head, and a SigLIP-style vision tower.

Phase plan:
    Phase 1: PyTorch-eager wrapper around the HF reference model
             (this commit). No fvk kernels yet -- locks the frontend
             contract end-to-end and the Phase-0 cosine fixture as the
             regression baseline.
    Phase 2: replace full-attn + GDN linear-attn + MoE + RMSNorm + RoPE
             with fvk kernel calls (reuse the qwen36 GDN / attention /
             partial-RoPE families; add the MoE router + grouped GEMM).
    Phase 3: NVFP4 weights + FP8 KV cache + CUDA graph + decode loop.
    Phase 4: MTP / DFlash speculative decode.
    Phase 5: vision tower (multimodal, optional).
"""

from flash_rt.models.nexn2.pipeline_rtx import Nexn2Pipeline

__all__ = [
    'Nexn2Pipeline',
]
