"""FlashRT -- Nex-N2-mini model pipelines.

    pipeline_rtx.py  - Nexn2Dims (static dims) + Nexn2Pipeline (BF16 HF
                       reference, used only for kernelized=False).

Nex-N2-mini is the MoE sibling of the dense qwen36 family
(architectures=Qwen3_5MoeForConditionalGeneration / model_type=qwen3_5_moe):
hybrid Gated-DeltaNet + softmax-attention with a fine-grained 256-expert MoE
FFN. The production NVFP4 kernel forward/decode (CUDA-graph, chunked
long-context prefill) lives in flash_rt.frontends.torch._nexn2_rtx_*.
"""

from flash_rt.models.nexn2.pipeline_rtx import Nexn2Pipeline

__all__ = [
    'Nexn2Pipeline',
]
