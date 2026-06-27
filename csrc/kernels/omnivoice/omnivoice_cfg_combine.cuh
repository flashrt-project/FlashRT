// ================================================================
// FlashRT — OmniVoice fused kernels.
//
// Replaces 3× torch log_softmax in the MaskGIT CFG path:
//   c_lp = log_softmax(c_logits)        with mask_col → -inf
//   u_lp = log_softmax(u_logits)        (NOT masked)
//   out  = log_softmax(c_lp + gs*(c_lp - u_lp))
//
// Warp-per-row, BF16 I/O, FP32 compute, shfl_xor reductions.
// Covers up to CC_ITERS*32 columns (1536).
// ================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt { namespace kernels {

// (kept in namespace for organization; .cu defines at global scope via using)
}}

void omnivoice_cfg_logsoftmax_bf16(
    const __nv_bfloat16* c_logits,   // [rows, cols]
    const __nv_bfloat16* u_logits,   // [rows, cols]
    __nv_bfloat16* out,              // [rows, cols]
    int rows, int cols, int mask_col, float guidance_scale,
    cudaStream_t stream);
