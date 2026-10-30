// SPDX-License-Identifier: Apache-2.0
// ============================================================================
//  FlashRT — MolmoAct2 action-expert persistent layer kernel (sm_120a).
//
//  One cooperative launch runs all action-expert transformer layers of one
//  flow-matching step (M = horizon <= 32 rows, D = 768, 8 heads x 96,
//  FFN 3072). The phases of a layer are separated by grid barriers:
//
//    A  modulated RMSNorm -> FP8 (per CTA)   + FP8 QKV GEMM
//    B  QKV split + bias + q/k RMSNorm + RoPE + self attention (per head)
//    C  NVFP4 W4A4 self out-proj + bias, gate, residual
//    E  per (head, KV split): modulated RMSNorm -> FP8, the head's FP8
//       cross-query projection, query bias + RMSNorm, split-KV cross
//       attention; the head's split CTAs then combine one quarter each
//    F  NVFP4 W4A4 cross out-proj + bias, gate, residual
//    G  modulated RMSNorm -> FP8 (per CTA)   + FP8 up/gate GEMM + SwiGLU
//       -> NVFP4 down operand
//    H  NVFP4 W4A4 down-proj + bias, gate, residual
//
//  Precision and per-element rounding follow the multi-launch route
//  (FP8 qkv/cq/up/gate, skinny NVFP4 out/down, BF16 attention with FP32
//  softmax); weights are read in fragment-major layouts so every lane
//  streams 16 contiguous bytes per MMA chunk. While a layer runs, the last
//  `prefetch` CTAs stream the next layer's weights and cross-attention
//  context into L2 with bulk prefetches, so DRAM traffic overlaps the
//  barrier chain of the compute CTAs.
// ============================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt {
namespace molmoact2 {

constexpr int kMkMaxPrefetch = 12;

struct ActionMkLayer {
    const uint8_t* qkv_w;        // FP8 [2304, 768], fragment-major
    const __nv_bfloat16* qkv_b;  // [2304]
    const uint8_t* so_w;         // skinny NVFP4 packed [768, 768]
    const uint8_t* so_sf;
    const __nv_bfloat16* so_b;
    const uint8_t* cq_w;         // FP8 [768, 768], fragment-major
    const __nv_bfloat16* cq_b;
    const uint8_t* co_w;
    const uint8_t* co_sf;
    const __nv_bfloat16* co_b;
    const uint8_t* up_w;         // FP8 [3072, 768], fragment-major
    const uint8_t* gate_w;
    const __nv_bfloat16* up_b;
    const __nv_bfloat16* gate_b;
    const uint8_t* dn_w;         // skinny NVFP4 packed [768, 3072]
    const uint8_t* dn_sf;
    const __nv_bfloat16* dn_b;
    const __nv_bfloat16* ctx_k;  // head-major [8][ctx_rows][104] bf16 (96 used)
    const __nv_bfloat16* ctx_v;
    // L2 prefetch regions (this layer's weights + context)
    const void* pf_ptr[kMkMaxPrefetch];
    unsigned long long pf_bytes[kMkMaxPrefetch];
    int pf_count;
};

// Static per-(step, layer) scalars, frozen after calibration.
struct ActionMkScales {
    float qkv_qs;     // FP8 activation scale of the qkv operand
    float qkv_alpha;  // qkv_qs * weight scale
    float cq_qs;
    float cq_alpha;
    float ffn_qs;     // shared up/gate FP8 activation scale
    float up_alpha;
    float gate_alpha;
    float so_ags;     // NVFP4 activation global scale
    float so_alpha;   // so_ags * weight global scale
    float co_ags;
    float co_alpha;
    float dn_ags;
    float dn_alpha;
    float pad[3];
};

struct ActionMkParams {
    const ActionMkLayer* layers;   // device [num_layers]
    const ActionMkScales* scales;  // device [num_layers] for this step
    const __nv_bfloat16* mod_self;   // [num_layers, 3, 768] for this step
    const __nv_bfloat16* mod_cross;
    const __nv_bfloat16* mod_ffn;
    const __nv_bfloat16* rope_cos;   // [rows, 96]
    const __nv_bfloat16* rope_sin;
    __nv_bfloat16* x;      // [rows, 768] residual stream (in/out)
    __nv_bfloat16* qkv;    // [rows, 2304] scratch
    uint8_t* attn_fp4;     // [rows, 384] attention output as NVFP4 out-proj operand
    uint8_t* attn_sf;      // [rows, 48]
    uint8_t* act_fp4;      // [rows, 1536] NVFP4 down operand
    uint8_t* act_sf;       // [rows, 192]
    float* xo_part;        // [8, splits, 32, 96]
    float* xo_lse;         // [8, splits, 32]
    unsigned* counters;    // [16]: per-head split arrivals, prefetch progress (zeroed once)
    unsigned* bar;         // 64-bit grid-barrier arrival counter (8-byte aligned, zeroed once)
    const ActionMkLayer* next_first;  // layer prefetched after the last one
    int num_layers;
    int rows;
    int ctx_len;
    int ctx_rows;          // padded rows per head block of ctx_k / ctx_v
    float eps;
    int prefetch;          // CTAs dedicated to L2 prefetch (0: none)
};

// Returns 0 on success.
int action_mk_layers(const ActionMkParams& params, int grid,
                     cudaStream_t stream);
// Dynamic shared memory per CTA.
int action_mk_smem_bytes();
// Repack an e4m3 weight ([N, K], or [K, N] with kn != 0) into the
// fragment-major layout the kernel streams (N * K bytes).
int action_mk_pack_fp8(const void* w, void* out, int N, int K, int kn,
                       cudaStream_t stream);

}  // namespace molmoact2
}  // namespace flash_rt
