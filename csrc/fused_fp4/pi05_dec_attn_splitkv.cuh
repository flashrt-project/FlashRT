// ============================================================================
//  FlashRT — Pi0.5 decoder attention, split-KV single pass + fused NVFP4 out.
//
//  Replaces the four-launch chain (cuBLAS QK^T, softmax, cuBLAS PV, FP4
//  quantize) for the action-expert shape: 10 query tokens x 8 heads, one
//  shared KV head, head_dim 256, ~800 keys. Kernel 1 tiles the keys across
//  CTAs and writes unnormalised partial outputs with per-row (max, sum);
//  kernel 2 merges the partials, rounds through fp16 exactly as the chain's
//  output buffer did, and emits the packed e2m1 block + CUTLASS SFA byte with
//  the same rounding as quantize_fp4_dynamic_sfa_fp16_vec.
//
//  Additive: the cuBLAS chain remains the default.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace fp4 {

// Workspace bytes required by pi05_dec_attn_splitkv_fp4 (any S_kv).
size_t pi05_dec_attn_splitkv_ws_bytes();

// Q: (S*NH, HD) fp16 (RoPE already applied); K, V: (S_kv, HD) fp16.
// dst_packed: (S, NH*HD/2) bytes; dst_sfa: CUTLASS SFA layout for (S, NH*HD).
// Requires HD == 256 and S*NH == 80.
int pi05_dec_attn_splitkv_fp4(
    const void* Q, const void* K, const void* V, void* workspace,
    void* dst_packed, void* dst_sfa,
    int S, int S_kv, int NH, int HD, float attn_scale,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
