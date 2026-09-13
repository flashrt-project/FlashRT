// ============================================================================
//  FlashRT — NVFP4 x NVFP4 GEMM for M <= 16 (Pi0.5 Thor action expert).
//
//  The CUTLASS block-scaled kernels reach ~50-75% of DRAM bandwidth at M=10
//  because tile N >= 64 leaves N/64 CTAs (16 for N=1024) on a 20-SM part.
//  This kernel streams the weight with 16-column CTAs whose 8 warps split K
//  and reduce through shared memory; operands are dequantised to fp16 with
//  their UE4M3 block scales folded in (exact: e2m1 x e4m3 fits fp16) and fed
//  to mma.sync.m16n8k16 with fp32 accumulation.
//
//  Layouts match the CUTLASS runners: A/B packed e2m1 [rows][K/2] (even k in
//  the low nibble), scale factors in the Sm1xx 128x4 blocked layout.
//  Requires M <= 16, N % 16 == 0, K % 128 == 0.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// D[M][N] fp16 = A[M][K] * B[N][K]^T  (block-scaled NVFP4 operands).
int nvfp4_m16_gemm_fp16out(const void* A, const void* SFA, const void* B,
                           const void* SFB, void* D, int M, int N, int K,
                           cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
