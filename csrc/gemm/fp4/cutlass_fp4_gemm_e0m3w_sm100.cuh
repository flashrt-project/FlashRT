// ============================================================================
//  FlashRT — block-scaled GEMM with E0M3 (uniform INT4) weights on SM110.
//
//  The tcgen05 block-scaled MMA decodes its 3-bit instruction-descriptor
//  element-format field at run time; value 1 is the documented E2M1, value 0
//  decodes the sign-magnitude uniform INT4 grid (E0M3, magnitudes 0..7).
//  This runner instantiates the CUTLASS runtime-datatype collective
//  (type_erased_dynamic_nv_float4_t) and issues A as E2M1 (activations,
//  produced by the existing NVFP4 quantizers) against B as E0M3 weights
//  (produced by quantize_e0m3_dynamic_sfa_fp16). Scale factors stay per-16
//  UE4M3 in the standard SFA/SFB tile-interleaved layouts on both operands.
//
//  Tile = 128x64x256 (the production decoder projection tile, variant v10).
//  Additive: does NOT modify cutlass_fp4_gemm_variants.cu or any existing
//  GEMM path.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// D[M,N] = alpha * (A x SFA) @ (B_e0m3 x SFB)^T + beta * D
//   A: packed 4-bit [M, K/2] row-major + SFA tile-interleaved UE4M3;
//      element format selected by a_format (1 = E2M1 default, 0 = E0M3)
//   B: packed e0m3 [N, K/2] (column-major operand) + SFB tile-interleaved
//   D: fp16 [M, N] row-major
// Returns 0 on success; CUTLASS status codes are or-ed with 0x10000 /
// 0x20000 / 0x30000 for can_implement / initialize / run failures.
int cutlass_fp4_gemm_e0m3w(
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream, int a_format = 1);

}  // namespace fp4
}  // namespace flash_rt
