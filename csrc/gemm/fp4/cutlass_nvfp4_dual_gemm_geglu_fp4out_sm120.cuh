// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 fused GeGLU GEMM, SM120a.  Computes, in one launch and
// without materializing the bf16 gate/up activations to HBM:
//
//     D[m, j] = pack_FP4( silu(A_fp4 @ Wgate_fp4^T) * (A_fp4 @ Wup_fp4^T) )
//
// i.e. the gate and up projections share the same A operand, are accumulated
// per-CTA so gate[j] and up[j] are register-coresident, and the epilogue does
// the GeGLU multiply + per-16-block NVFP4 requantization.  This is the
// "dual-accumulator" realization of the GeGLU fold (CUTLASS example 45
// `LeftSiLUAndMul` math on the blockscaled SM120 collective): the output N
// equals each projection's N (= intermediate), so the blockscaled-FP4 store
// epilogue stays standard (no half-width / cross-column store surgery).
//
//   A_packed   : (M, K/2)        uint8  NVFP4 packed (cutlass-swizzled SF)
//   Bgate/Bup  : (Ninter, K/2)   uint8  NVFP4 packed weights
//   SFA        : (M*K/16)        e4m3
//   SFBgate/up : (Ninter*K/16)   e4m3
//   D_packed   : (M, Ninter/2)   uint8  NVFP4 packed GeGLU output
//   SFD        : (M*Ninter/16)   e4m3   output SF, cutlass-swizzled layout
//   alpha_gate : float32         = sf_global_a * sf_global_bgate
//   alpha_up   : float32         = sf_global_a * sf_global_bup
//
// MILESTONE NOTE (build scaffold): the initial implementation validates the
// build/binding/instantiation loop and the SiLu activation + FP4-out epilogue
// with a SINGLE projection (silu(A@Bgate) -> FP4, Bup ignored).  The dual
// mainloop (Bup second accumulator + GeGLU multiply) lands incrementally on
// top of this exact interface so callers never change.
//
// Stream-safe; per-shape workspace cached internally.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

void fp4_w4a16_dual_gemm_geglu_fp4out_sm120(
    const void*  A_packed,
    const void*  Bgate_packed,
    const void*  Bup_packed,
    const void*  SFA,
    const void*  SFBgate,
    const void*  SFBup,
    void*        D_packed,
    void*        SFD,
    int M, int N, int K,
    float        alpha_gate,
    float        alpha_up,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
