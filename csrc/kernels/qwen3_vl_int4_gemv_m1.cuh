#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

// M=1 decode GEMV with INT4 symmetric weights and BF16 activation (W4A16).
//   x   : [K]         bf16 activation
//   Wp  : [N, K/2]    two int4 per byte (low nibble = even elem, high = odd);
//                     each nibble is 2's-complement in [-7, 7]
//   Ws  : [N, K/16]   bf16 per-16-element block scales (scale = amax/7)
//   out : [N]         bf16
// Half the weight bytes of int8 (0.5 B/elem + 0.125 B scale). Nibble unpack is
// a cheap shift + sign-extend + hardware int->float I2F — Ampere-friendly
// (contrast the e2m1 `qwen3_vl_w4_gemv_m1`, whose LUT/fp8 path is ALU-bound on
// sm_87). For the weight-bandwidth-bound M=1 decode this ~halves weight HBM
// traffic vs int8. Precision is coarser (15 levels) — validate per task.
void qwen3_vl_int4_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream);

}  // namespace flash_rt::kernels
