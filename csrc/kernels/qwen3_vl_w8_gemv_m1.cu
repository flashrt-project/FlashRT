#include "qwen3_vl_w8_gemv_m1.cuh"

#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

// One warp per output row n. Each lane strides over the row's 16-element
// blocks (16 e4m3 bytes = one uint4), converts each e4m3->float (hardware),
// dots with the BF16 activation (x is tiny, hot in L2), applies the per-block
// scale once per group. fp32 accumulation: e4m3 dequant values reach ~448, so
// half2 accumulation would overflow the half range — W8 uses fp32. Already
// bandwidth-bound (~219 GB/s of Thor's ~229 achievable), so no smem/SIMD win.
__global__ __launch_bounds__(kThreads) void qwen3_vl_w8_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ Wp,
    const __nv_bfloat16* __restrict__ Ws,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp;
    if (n >= N) return;

    const int NG = K >> 4;                          // K / 16 blocks
    const uint8_t* wrow = Wp + (size_t)n * K;
    const __nv_bfloat16* srow = Ws + (size_t)n * NG;

    float acc = 0.0f;
    for (int g = lane; g < NG; g += 32) {
        uint4 wp = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 16);
        const __nv_fp8x2_e4m3* w2 =
            reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);       // 8 pairs
        const __nv_bfloat162* x2 =
            reinterpret_cast<const __nv_bfloat162*>(x + (size_t)g * 16);
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 wf = __half22float2(static_cast<__half2>(w2[j]));
            float2 xf = __bfloat1622float2(x2[j]);
            part = fmaf(wf.x, xf.x, part);
            part = fmaf(wf.y, xf.y, part);
        }
        acc = fmaf(part, __bfloat162float(srow[g]), acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[n] = __float2bfloat16(acc);
    }
}

}  // namespace

void qwen3_vl_w8_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream) {
    if (K <= 0 || (K % 16) != 0) {
        throw std::runtime_error(
            "qwen3_vl_w8_gemv_m1 requires K to be a positive multiple of 16 "
            "(one uint4 weight load and one block scale per 16 elements), "
            "got K=" + std::to_string(K));
    }
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    qwen3_vl_w8_gemv_m1_kernel<<<grid, kThreads, 0, stream>>>(
        x, Wp, Ws, out, N, K);
}

}  // namespace flash_rt::kernels
