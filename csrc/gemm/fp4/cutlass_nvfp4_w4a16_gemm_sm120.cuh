// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS-based NVFP4 W4A16 GEMM for SM120a (RTX 5090 / Blackwell
// consumer GeForce). Native block-scaled FP4 GEMM matching the Qwen3.6
// NVFP4 ckpt schema (compressed-tensors `nvfp4-pack-quantized` format).
//
// Wraps NVIDIA's verified template from
// third_party/cutlass/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/
// sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu — using `OpClassBlockScaledTensorOp`
// + `nv_float4_t<float_e2m1_t>` + `float_ue4m3_t` group scales, BF16
// output. SM_120 + SM_121 (RTX 5090 / 5080) gated.
//
// Schema (matches both A=act and B=weight after per-token NVFP4 quant):
//   * elements    : 4-bit FP e2m1, packed two per byte
//   * group scale : FP8 ue4m3, one scale per 16-element group
//                   (`group_size = 16` per the ckpt config.json)
//   * global scale: a single FP32 per tensor (fed via the epilogue's
//                   alpha so we get D = sf_global_a * sf_global_b *
//                   (A * B) with a single multiply instead of a
//                   per-tile rescale)
//
// Caller responsibilities:
//   * A_packed, B_packed are u8 arrays viewing FP4 e2m1 (2x packed).
//     A is row-major (M, K/2 byte-pairs). B is column-major weight
//     view; we accept the natural HF row-major (N, K/2) layout and
//     reinterpret as ColumnMajor (K, N) — same memory.
//   * SFA, SFB are FP8 ue4m3 with the CUTLASS Sm1xx blockscaled tile
//     atom layout. The activation quantizer (`quantize_fp4_dynamic_*`)
//     is responsible for emitting SFA in this layout; the weight
//     loader does the same SFB transform once at load.
//
// Constraints (verified by `can_implement` at runtime):
//   * K must be a multiple of 16 (group size).
//   * Pointer alignments: A/B 16 bytes (32 FP4 elements), C/D 16 bytes
//     (8 BF16 elements).
//   * M unrestricted. We pick a tile shape by M (small-M variant
//     coming once profiled — first cut uses the unit test's
//     <128,128,256> for all M and lets CUTLASS handle padding).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// NVFP4 W4A16 GEMM, BF16 output, SM120a (RTX 5090).
//
//   A_packed  : (M, K/2)        u8  row-major   (FP4 e2m1, 2 per byte)
//   B_packed  : (N, K/2)        u8  row-major   (FP4 e2m1, 2 per byte)
//                                              — read as ColumnMajor (K, N)
//   D_bf16    : (M, N)          bf16 row-major
//   SFA       : (M, K/16)       e4m3 (CUTLASS blockscaled atom layout)
//   SFB       : (N, K/16)       e4m3 (CUTLASS blockscaled atom layout)
//   alpha     : fp32 scalar = act_global_scale * w_global_scale
//
// Stream-safe; per-shape arguments + workspace cached internally
// (mirrors the FP8 sm_120 kernel).
void fp4_w4a16_gemm_sm120_bf16out(
    const void*  A_packed,    // (M, K/2)        u8
    const void*  B_packed,    // (N, K/2)        u8
    void*        D_bf16,      // (M, N)          bf16
    int M, int N, int K,
    const void*  SFA,         // (M, K/16)       e4m3 (Sm1xx blockscaled layout)
    const void*  SFB,         // (N, K/16)       e4m3 (Sm1xx blockscaled layout)
    float        alpha,       // = sf_global_a * sf_global_b
    cudaStream_t stream);

// Wide-N variant: TileShape <128, 256, 128>. For shapes with very
// large N (lm_head N=248320, MLP gate/up N=17408) where the wider N
// tile uses fewer waves and hits ~88%/66% peak BW vs ~64%/56% for
// the default <128,128,256> tile. For small/medium N (<= 6144) the
// default kernel is faster — caller dispatches by shape.
void fp4_w4a16_gemm_sm120_bf16out_widen(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// Same tile shape as the default kernel, but with
// KernelTmaWarpSpecializedPingpong. Kept as an explicit opt-in variant so
// callers can A/B schedule effects per shape without perturbing the default.
void fp4_w4a16_gemm_sm120_bf16out_pingpong(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
