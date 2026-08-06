// ============================================================================
//  FlashRT — NVFP4 GEMM with fused GeGLU epilogue over a column-interleaved
//  gate/up weight, FP4 (e2m1) packed output + SFD tile-interleaved.
//
//  B holds gate and up rows pairwise interleaved along N (B_il[2j] = gate[j],
//  B_il[2j+1] = up[j]); the epilogue computes gelu(gate)*up per column pair
//  and duplicates the result into both columns (full-width output), replacing
//  the separate gate GEMM + up GEMM + combiner chain with one kernel.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// A: [M, K] NVFP4 packed row-major + SFA (tile-interleaved).
// B_il: [N_il, K] NVFP4 packed column-major + SFB, gate/up pairwise
//       interleaved along N (N_il = 2 * projection width, must be even).
// D: [M, N_il] NVFP4 packed row-major + SFD (tile-interleaved), each column
//    pair holding the duplicated gelu(gate)*up value.
// Returns 0 on success; CUTLASS status | stage flag otherwise.
int cutlass_fp4_gemm_geglu_il(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_packed,
    void*       D_SFD,
    int M, int N_il, int K,
    cudaStream_t stream);

// Half-width variant: the epilogue quantizes gelu(gate)*up at compact
// granularity (16 unique values per scale block, combiner-equivalent) and
// writes compact_packed [M, N_il/2] + compact_sfa (SFA tile-atom layout on
// (M, N_il/2)) directly; D_dummy [M, N_il] receives zeros and can be one
// small buffer shared across layers. The downstream GEMM consumes the
// compact outputs with its original K = N_il/2 weight.
int cutlass_fp4_gemm_geglu_il_hw(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream);

// Skinny-M variant on the decoder GEMM tile (128x64x256): same contract
// as cutlass_fp4_gemm_geglu_il_hw with CTA parallelism suited to tiny M.
int cutlass_fp4_gemm_geglu_il_hw_v10(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
