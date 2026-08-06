#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

// M=1 decode GEMV with NVFP4 (e2m1) weights and BF16 activation (W4A16).
//   x   : [K]        bf16 activation
//   Wp  : [N, K/2]   uint8, packed e2m1 (low nibble = even col, high = odd)
//   Ws  : [N, K/16]  bf16 per-16-element block scales
//   out : [N]        bf16
// Weight bytes/elem = 0.5 (packed) + 0.125 (bf16 scale/16) = 0.625 vs bf16 2.0
// -> ~3.2x less HBM traffic, the decode-throughput lever on bandwidth-bound Thor.
void qwen3_vl_w4_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream);

}  // namespace flash_rt::kernels
