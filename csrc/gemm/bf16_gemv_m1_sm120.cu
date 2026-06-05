// SPDX-License-Identifier: Apache-2.0
//
// Dedicated M=1 BF16 GEMV for sm_120a decode (batch=1 token) — the all-BF16
// sibling of fp8_gemv_m1_sm120.cu, for hosts that cannot run FP8.
//
// One warp per output row n: each warp reduces B[n] against A and writes D[n].
// B is read in 16-byte coalesced uint4 (8 bf16) chunks, stride 32 across the
// warp; the partial sums are folded by a warp-shuffle reduction. A is read from
// global (no smem staging): at BF16 the FP8 variant's smem stage would be 2x
// the bytes (K=9728 -> 19.5 KB/block) and cap blocks/SM under
// launch_bounds(.,8); A is tiny (<=19 KB) and stays hot in L2 across blocks, so
// reading it from global keeps smem=0 and occupancy maximal — the all-BF16
// decode is HBM-bound on the weight read, which this saturates.

#include "bf16_gemv_m1_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace gemv_m1 {

namespace {

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 8)
void gemv_bf16_m1_kernel(
    const __nv_bfloat16* __restrict__ A,   // [K]
    const __nv_bfloat16* __restrict__ B,   // [N, K]
    __nv_bfloat16* __restrict__ D,         // [N]
    int N, int K, float alpha)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K8   = K >> 3;                // # of 16-byte (uint4 = 8 bf16) groups

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const uint4* A8   = reinterpret_cast<const uint4*>(A);
    const uint4* Brow = reinterpret_cast<const uint4*>(B) + (size_t)n * K8;
    float acc = 0.0f;
    for (int i = lane; i < K8; i += 32) {
        uint4 bpack = Brow[i];
        uint4 apack = A8[i];
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bpack);
        const __nv_bfloat16* ap = reinterpret_cast<const __nv_bfloat16*>(&apack);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += __bfloat162float(ap[j]) * __bfloat162float(bp[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) D[n] = __float2bfloat16(acc * alpha);
}

template <int W>
int launch_(const void* A, const void* B, void* D,
            int /*M*/, int N, int K, float alpha, cudaStream_t stream) {
    dim3 grid((N + W - 1) / W);
    gemv_bf16_m1_kernel<W><<<grid, W * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B),
        reinterpret_cast<__nv_bfloat16*>(D), N, K, alpha);
    return 0;
}

}  // namespace

#define DEFINE(NAME, W)                                                         \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,          \
           float alpha, cudaStream_t stream) {                                  \
    return launch_<W>(A, B, D, M, N, K, alpha, stream);                         \
  }

DEFINE(gemv_bf16_m1_w4,  4)
DEFINE(gemv_bf16_m1_w8,  8)
DEFINE(gemv_bf16_m1_w16, 16)

#undef DEFINE

}  // namespace gemv_m1
}  // namespace gemm
}  // namespace flash_rt
