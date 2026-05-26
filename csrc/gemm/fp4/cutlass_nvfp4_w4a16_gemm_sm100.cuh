// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS-based NVFP4 W4A16 GEMM for sm_100-class Blackwell (datacenter
// SM100 / Jetson AGX Thor SM110). Block-scaled FP4 GEMM matching the
// Qwen3.6 NVFP4 ckpt schema (compressed-tensors `nvfp4-pack-quantized`).
//
// Sibling of cutlass_nvfp4_w4a16_gemm_sm120.cuh. The two differ only in
// CUTLASS arch dispatch and kernel-schedule policy:
//   - sm120: arch::Sm120 + KernelTmaWarpSpecializedCooperative /
//            KernelTmaWarpSpecializedPingpong
//   - sm100: arch::Sm100 + KernelScheduleAuto
// On Thor (sm_110a) the Sm100 dispatch path produces the correct
// blockscaled tcgen05 mainloop. Verified in dev_log/step0.
//
// Wire-format contract is identical to the sm120 variant, so the
// Python-side weight/scale layout and the activation quantizer are
// reused unchanged. The pybind layer binds these Thor symbols under
// the existing public names ``fp4_w4a16_gemm_sm120_bf16out*`` so the
// Qwen3.6 frontend code path does not need any hardware fork.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Default tile <128,128,256>, cluster <1,1,1>, KernelScheduleAuto.
void fp4_w4a16_gemm_sm100_bf16out(
    const void*  A_packed,    // (M, K/2)        u8  row-major
    const void*  B_packed,    // (N, K/2)        u8  row-major (read as ColMajor (K,N))
    void*        D_bf16,      // (M, N)          bf16 row-major
    int M, int N, int K,
    const void*  SFA,         // (M, K/16)       e4m3 (Sm1xx blockscaled atom layout)
    const void*  SFB,         // (N, K/16)       e4m3 (Sm1xx blockscaled atom layout)
    float        alpha,       // = sf_global_a * sf_global_b
    cudaStream_t stream);

// Wide-N tile <128,256,128>, cluster <1,1,1>, KernelScheduleAuto.
// For shapes with very large N (lm_head, MLP gate/up).
void fp4_w4a16_gemm_sm100_bf16out_widen(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// Default tile <128,128,256>, cluster <1,1,1>, KernelScheduleAuto.
// Kept as a separate symbol so callers can A/B against the default
// variant after the tile sweep produces a Thor-tuned schedule.
void fp4_w4a16_gemm_sm100_bf16out_pingpong(
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
