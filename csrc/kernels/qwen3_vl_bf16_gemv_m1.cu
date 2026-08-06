#include "qwen3_vl_bf16_gemv_m1.cuh"

#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

template<int K_FIXED>
__global__ __launch_bounds__(kThreads) void qwen3_vl_bf16_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int N) {
    // No shared-memory staging of x: x is tiny (K bf16 = 4-12 KB) and stays
    // hot in L2 across the many weight-row blocks, so reading it from global
    // keeps smem=0 and occupancy maximal — the decode is HBM-bound on the
    // weight read, which this saturates (mirrors bf16_gemv_m1_sm120).
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= N) return;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    const int4* w_row_i4 = reinterpret_cast<const int4*>(W + n * K_FIXED);
    constexpr int K_INT4 = K_FIXED / 8;

    float acc = 0.0f;
    #pragma unroll 1
    for (int i4 = lane; i4 < K_INT4; i4 += 32) {
        int4 wv = w_row_i4[i4];
        int4 xv = x_i4[i4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&wv)[k]));
            __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv)[k]));
            float2 wf = __bfloat1622float2(wb);
            float2 xf = __bfloat1622float2(xb);
            acc = fmaf(xf.x, wf.x, acc);
            acc = fmaf(xf.y, wf.y, acc);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[n] = __float2bfloat16(acc);
    }
}

}  // namespace

void qwen3_vl_bf16_gemv_m1(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int N,
    int K,
    cudaStream_t stream) {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if (K == 2048) {
        qwen3_vl_bf16_gemv_m1_kernel<2048>
            <<<grid, kThreads, 0, stream>>>(x, W, out, N);
    } else if (K == 6144) {
        qwen3_vl_bf16_gemv_m1_kernel<6144>
            <<<grid, kThreads, 0, stream>>>(x, W, out, N);
    } else {
        throw std::runtime_error(
            "qwen3_vl_bf16_gemv_m1 supports only K=2048 or K=6144, got K=" +
            std::to_string(K));
    }
}

}  // namespace flash_rt::kernels
