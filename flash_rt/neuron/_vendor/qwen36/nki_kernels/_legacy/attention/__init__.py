# Vendored gpt-oss attention pipeline. See `nki_kernels/_vendor_meta.md` for
# the rationale; the gpt-oss megakernel imports `attention_block_tkg` from
# here instead of nkilib so that 24 per-layer invocations don't produce
# duplicate op names. Every direct `nl.ndarray(name=...)` /
# `nisa.dma_copy(name=...)` call in these files has been rewritten to prepend
# `sbm.get_name_prefix()`. The megakernel sets
# `sbm.set_name_prefix(f"L{layer_idx}_attn_")` before each invocation.
#
# The Qwen3-MoE fused attention kernel (`attn_fused_nki.py`) lives in
# `megakernels/qwen3_moe/` since it is qwen-specific.
from .attention_block_tkg import attention_block_tkg

__all__ = ["attention_block_tkg"]
