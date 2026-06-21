// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini M=1 W4A16 GEMV for SM120 (NVFP4 weight x BF16 activation).
//
// The dense decode projections (GDN/full-attn/out/shared) are the largest
// single bucket of decode-step HBM traffic. The BF16 MLP GEMV
// (nexn2_bf16_matvec_bf16) already hits ~87% of peak BW, so the only way to
// shrink that bucket is to read 4-bit weights instead of 16-bit ones.
//
// Unlike the W4A4 matvec (fp4_w4a4_matvec_sm120), the activation here stays
// full BF16 (no per-tensor / per-block activation quant), so there is no
// activation-quant error -- this is the high-precision NVFP4 path. The weight
// is the standard swizzled NVFP4 produced by bf16_weight_to_nvfp4_swizzled:
//   W[n,k] = global_scale * ue4m3(SFB[n, k/16]) * fp4_e2m1(nibble[n,k])
// so out[n] = global_scale * sum_kb ue4m3(SFB) * sum_j fp4(nibble) * x_bf16.
//
// Layout mirrors nexn2_bf16_gemv: 1 warp / output row, x staged once in smem
// and shared across the 8 warps of the block, UNROLL packed-weight loads in
// flight per warp so the K loop stays bandwidth-bound. All add-only.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(1,N) = (x(1,K) bf16) . (W(N,K) NVFP4)^T, fp32 accumulate, bf16 out.
//   x_bf16    : (K,)            bf16 activation (unquantized)
//   W_packed  : (N, K/2)        NVFP4 e2m1 nibbles (row-major)
//   SFB       : swizzled        UE4M3 block scales (Sm120 tile-interleaved)
//   out       : (N,)            bf16
//   alpha     : weight per-tensor global_scale (out_global_scale)
// K must be a multiple of 16. Returns 0 on success, nonzero on arg error.
int nexn2_w4a16_matvec_bf16(
    const void*  x_bf16,
    const void*  W_packed,
    const void*  SFB,
    void*        out,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
