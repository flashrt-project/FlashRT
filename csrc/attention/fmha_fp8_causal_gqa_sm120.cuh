// SPDX-License-Identifier: Apache-2.0
//
// FP8 (e4m3) causal GQA FlashAttention for SM120a (RTX 5090 / Blackwell
// consumer). Q, K and V are raw e4m3 (direct cast, no per-token scales:
// e4m3's exponent range covers the post-norm Q/K magnitudes, see the
// per-layer cos >= 0.988 / end-to-end argmax parity validation). The
// full mainloop runs on the fp8 m16n8k32 MMA with fp32 accumulate and
// online softmax; P is requantised to e4m3 in shared memory for the
// P*V product. K/V tiles are cp.async double-buffered and the V
// transpose staging uses a padded row stride so the scatter stores are
// shared-memory bank-conflict free.
//
// Layout contract (matches the qwen3 attention backend buffers):
//   q        : (Lq, Hq, 128)  e4m3, row stride Hq*128
//   k, v     : (Lk, Hkv, 128) e4m3, row stride Hkv*128
//   out      : (Lq, Hq, 128)  bf16, row stride Hq*128
// Causal self-attention from position 0 (Lq == Lk).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace attention {

// Returns 0 on success; non-zero when the shape is outside the kernel's
// support envelope (caller should fall back to FA2):
//   Lq == Lk, Lq a multiple of 128, Hq == 32, Hkv == 8, head_dim == 128.
int fmha_fp8_causal_gqa_nhd_d128(
    const void*  q_fp8,
    const void*  k_fp8,
    const void*  v_fp8,
    void*        out_bf16,
    int          Lq,
    int          Lk,
    int          num_q_heads,
    int          num_kv_heads,
    float        softmax_scale,
    cudaStream_t stream);

// Variant emitting O directly as NVFP4 (packed (Lq, Hq*128/2) u8 + swizzled
// ue4m3 group scales — the o_proj GEMM's A-operand format), skipping the
// bf16 O round-trip and the standalone quantize launch. Byte-identical to
// the [bf16 O + quantize kernel] chain (values round through bf16 first).
int fmha_fp8_causal_gqa_nhd_d128_fp4out(
    const void*  q_fp8,
    const void*  k_fp8,
    const void*  v_fp8,
    void*        out_fp4,
    void*        out_sf,
    int          Lq,
    int          Lk,
    int          num_q_heads,
    int          num_kv_heads,
    float        softmax_scale,
    cudaStream_t stream);

// 64-row / 64-key tile (4 warps, two CTAs per SM). Same layout contract;
// Lq must be a multiple of 64. Finer online-softmax tiling: 98.3 vs
// 114.7 us/layer at L=768, cosine 0.99996 against the 128-row tile.
int fmha_fp8_causal_gqa_q64_nhd_d128(
    const void*  q_fp8,
    const void*  k_fp8,
    const void*  v_fp8,
    void*        out_bf16,
    int          Lq,
    int          Lk,
    int          num_q_heads,
    int          num_kv_heads,
    float        softmax_scale,
    cudaStream_t stream);

// Same 64x64 tile with two producer warps that transpose the V tile while
// the four compute warps run QK; math and output are unchanged.
int fmha_fp8_causal_gqa_q64_nhd_d128_fp4out_wsp(
    const void* q_fp8, const void* k_fp8, const void* v_fp8,
    void* out_fp4, void* out_sf,
    int Lq, int Lk, int num_q_heads, int num_kv_heads,
    float softmax_scale, cudaStream_t stream);

int fmha_fp8_causal_gqa_q64_nhd_d128_fp4out(
    const void*  q_fp8,
    const void*  k_fp8,
    const void*  v_fp8,
    void*        out_fp4,
    void*        out_sf,
    int          Lq,
    int          Lk,
    int          num_q_heads,
    int          num_kv_heads,
    float        softmax_scale,
    cudaStream_t stream);

}  // namespace attention
}  // namespace flash_rt
