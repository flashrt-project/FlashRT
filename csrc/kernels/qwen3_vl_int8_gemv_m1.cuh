#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

// M=1 decode GEMV with INT8 symmetric weights and BF16 activation (W8A16).
//   x   : [K]        bf16 activation
//   Wp  : [N, K]     int8 (stored as uint8 bytes; 1 byte/weight)
//   Ws  : [N, K/16]  bf16 per-16-element block scales (scale = amax/127)
//   out : [N]        bf16
// Same byte traffic as the e4m3 W8 kernel (1.125 B/elem vs bf16 2.0), but the
// dequant is int8->float via a hardware I2F instruction. On Ampere (sm_87,
// Jetson Orin) there is NO hardware FP8 conversion, so the e4m3 kernel's
// fp8->half2 expand is a software bit-sequence that makes it ALU-bound; the
// int8 dequant keeps this GEMV bandwidth-bound on pre-sm89 GPUs.
void qwen3_vl_int8_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream);

}  // namespace flash_rt::kernels
