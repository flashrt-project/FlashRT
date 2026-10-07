// ============================================================================
//  FlashRT — NVFP4 (e2m1 + UE4M3 block-scale, block=16) GEMM for Thor SM110.
//
//  Minimal C-style entry points used by the pybind layer. Kernel bodies live
//  in cutlass_fp4_gemm.cu and are based on CUTLASS example 72a
//  (Blackwell NVFP4 × NVFP4 → bfloat16) rewritten to emit fp16.
//
//  This header is intentionally free of any CUTLASS includes so it can be
//  pulled into bindings.cpp and standalone test harnesses without paying
//  the CUTLASS compile cost there.
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {

// Invariants (enforced at call sites and / or static_assert'd):
//   - block_size == 16 (fixed; matches NVFP4 spec and hardware TCGEN05_MXF4_MMA)
//   - M, K multiples of 16
//   - N, K multiples of 16
//   - Row-major A [M, K], Column-major B [N, K] (natural for GEMM)
//   - Output D fp16 [M, N], row-major
//
//  Packing:
//   - A_fp4_packed: uint8 [M, K/2]  — each byte = 2 int4 elements (low nibble
//     holds element 2i, high nibble holds 2i+1)
//   - SFA / SFB: fp8 (UE4M3 bitpattern via torch.float8_e4m3fn positive range)
//     [M, K/16] or [N, K/16] — one scale per 16 elements along K.
//
// Returns 0 on success, nonzero CUTLASS/CUDA error code otherwise.

int cutlass_fp4_sq_fp16(
    void const* A_fp4_packed,   // device ptr, uint8 [M, K/2]
    void const* SFA,            // device ptr, fp8  [M, K/16]
    void const* B_fp4_packed,   // device ptr, uint8 [N, K/2]
    void const* SFB,            // device ptr, fp8  [N, K/16]
    void* D_fp16,               // device ptr, fp16 [M, N]
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

// Runtime capability check — cheap, no-arg.
bool has_nvfp4_sm110();

// Parametric variant dispatcher for tile/schedule tuning experiments.
int cutlass_fp4_gemm_variant(int idx,
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream);
const char* cutlass_fp4_gemm_variant_name(int idx);
int cutlass_fp4_gemm_num_variants();

} // namespace fp4
} // namespace flash_rt
