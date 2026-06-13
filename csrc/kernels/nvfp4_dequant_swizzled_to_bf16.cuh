// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (e2m1 packed + UE4M3 swizzled block scales) -> BF16 weight
// dequantization for SM120a/SM121a.
//
// Produces a dense BF16 weight from the same packed+swizzled artifacts the
// W4A4 GEMM consumes, so a BF16-activation matmul over the result realizes
// true W4A16 numerics (no activation quantization). Used by the MiniMax-M3
// quality ladder: W4A4 decode (cos ~0.80 E2E) -> W4A16 (cos ~0.89) by
// dequantizing the weight and keeping the activation in BF16.
//
//   B_packed : (N, K/2)   u8   row-major  (FP4 e2m1, 2 per byte; low
//                                           nibble = even k, high = odd k)
//   SFB      : (N, K/16)  u8   UE4M3, CUTLASS Sm1xx blockscaled swizzle
//   D_bf16   : (N, K)     bf16 row-major
//   alpha    : per-tensor global dequant scale (the value emitted by
//              bf16_weight_to_nvfp4_swizzled as out_global_scale)
//
// K must be a multiple of 16. N, K arbitrary otherwise.

#pragma once

#include <cstdint>

namespace flash_rt {
namespace quantize {

void nvfp4_dequant_swizzled_to_bf16(const uint8_t* B_packed,
                                    const uint8_t* SFB,
                                    void* D_bf16,
                                    int N, int K,
                                    float alpha,
                                    void* stream);

}  // namespace quantize
}  // namespace flash_rt
