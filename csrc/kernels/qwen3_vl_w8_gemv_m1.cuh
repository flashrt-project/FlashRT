#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

// M=1 decode GEMV with FP8 e4m3 weights and BF16 activation (W8A16).
//   x   : [K]        bf16 activation
//   Wp  : [N, K]     e4m3 (1 byte/weight)
//   Ws  : [N, K/16]  bf16 per-16-element block scales
//   out : [N]        bf16
// Weight bytes/elem = 1.0 (e4m3) + 0.125 (bf16 scale/16) = 1.125 vs bf16 2.0
// -> ~1.8x less HBM traffic. e4m3->float is a single hardware conversion, so
// (unlike the FP4 path) this kernel stays bandwidth-bound on Thor's LPDDR5x.
void qwen3_vl_w8_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream);

}  // namespace flash_rt::kernels
