// SPDX-License-Identifier: Apache-2.0
//
// Dedicated M=1 FP8 e4m3 -> BF16 GEMV for sm_120a decode (batch=1 token).
//
// The hand-tuned MMA GEMMs pad M=1 to BLOCK_M=16 (m16n8k32), computing 16
// rows to use 1 — fine for compute (memory-bound) but the N=2560 shapes only
// spawn N/BLOCK_N blocks and starve the SMs. This GEMV assigns one warp per
// output row: A[1,K] is staged once into smem (hot in L2 across blocks), each
// warp streams its B row in 16-byte coalesced chunks and warp-reduces the dot
// product. BLOCK_N effectively 1-per-warp => N/WARPS_PER_BLOCK blocks (e.g.
// N=2560, W=8 -> 320 blocks) saturates occupancy without a split-K reduction.

#include "fp8_gemv_m1_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace gemv_m1 {

namespace {

// One warp per output row n. A staged in smem as raw fp8 (K bytes). B row read
// in uint4 (16 fp8) coalesced chunks, stride 32 across the warp. K assumed a
// multiple of 16 (all Higgs/Qwen3 GEMM K: 2560/4096/9728).
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 8)
void gemv_fp8_m1_kernel(
    const __nv_fp8_e4m3* __restrict__ A,   // [K]
    const __nv_fp8_e4m3* __restrict__ B,   // [N, K]
    __nv_bfloat16* __restrict__ D,         // [N]
    int N, int K, float alpha)
{
    extern __shared__ __nv_fp8_e4m3 sA[];   // [K]
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int threads = WARPS_PER_BLOCK * 32;
    const int K16     = K >> 4;             // # of 16-byte (uint4) groups

    // Cooperatively stage A into smem, 16 bytes per thread.
    uint4* sA16 = reinterpret_cast<uint4*>(sA);
    const uint4* A16 = reinterpret_cast<const uint4*>(A);
    for (int i = tid; i < K16; i += threads) sA16[i] = A16[i];
    __syncthreads();

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const uint4* Brow = reinterpret_cast<const uint4*>(B) + (size_t)n * K16;
    const __nv_fp8_e4m3* sAf = sA;
    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        uint4 bpack = Brow[i];
        const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&bpack);
        const __nv_fp8_e4m3* ap = sAf + (i << 4);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc += float(ap[j]) * float(bp[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) D[n] = __float2bfloat16(acc * alpha);
}

// GEMV with fused residual accumulate: D[n] += acc * alpha (in-place into the
// residual stream). The residual is per-element local — no cross-block
// dependency like the norm — so it folds into the epilogue for free, removing
// the separate residual_add launch. Each output n is written by one warp lane.
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 8)
void gemv_fp8_m1_resadd_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    __nv_bfloat16* __restrict__ D,   // residual stream, accumulated in place
    int N, int K, float alpha)
{
    extern __shared__ __nv_fp8_e4m3 sA[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int threads = WARPS_PER_BLOCK * 32, K16 = K >> 4;
    uint4* sA16 = reinterpret_cast<uint4*>(sA);
    const uint4* A16 = reinterpret_cast<const uint4*>(A);
    for (int i = tid; i < K16; i += threads) sA16[i] = A16[i];
    __syncthreads();
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;
    const uint4* Brow = reinterpret_cast<const uint4*>(B) + (size_t)n * K16;
    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        uint4 bpack = Brow[i];
        const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&bpack);
        const __nv_fp8_e4m3* ap = sA + (i << 4);
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc += float(ap[j]) * float(bp[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) D[n] = __float2bfloat16(__bfloat162float(D[n]) + acc * alpha);
}

template <int W>
int launch_resadd_(const void* A, const void* B, void* D,
                   int N, int K, float alpha, cudaStream_t stream) {
    dim3 grid((N + W - 1) / W);
    size_t smem = (size_t)K * sizeof(__nv_fp8_e4m3);
    gemv_fp8_m1_resadd_kernel<W><<<grid, W * 32, smem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D), N, K, alpha);
    return 0;
}

template <int W>
int launch_(const void* A, const void* B, void* D,
            int /*M*/, int N, int K, float alpha, cudaStream_t stream) {
    dim3 grid((N + W - 1) / W);
    dim3 block(W * 32);
    size_t smem = (size_t)K * sizeof(__nv_fp8_e4m3);
    gemv_fp8_m1_kernel<W><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        N, K, alpha);
    return 0;
}

}  // namespace

#define DEFINE(NAME, W)                                                         \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,          \
           float alpha, cudaStream_t stream) {                                  \
    return launch_<W>(A, B, D, M, N, K, alpha, stream);                         \
  }

DEFINE(gemv_fp8_m1_w4,  4)
DEFINE(gemv_fp8_m1_w8,  8)
DEFINE(gemv_fp8_m1_w16, 16)

#define DEFINE_RA(NAME, W)                                                      \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,          \
           float alpha, cudaStream_t stream) {                                  \
    return launch_resadd_<W>(A, B, D, N, K, alpha, stream);                     \
  }

DEFINE_RA(gemv_fp8_m1_resadd_w4, 4)
DEFINE_RA(gemv_fp8_m1_resadd_w8, 8)

#undef DEFINE
#undef DEFINE_RA

}  // namespace gemv_m1
}  // namespace gemm
}  // namespace flash_rt
