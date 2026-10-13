// Generic fp32-weight linear projections for small shapes. See linear_f32.cuh.

#include "linear_f32.cuh"

#include <stdexcept>

namespace flash_rt {
namespace kernels {

__global__ void linear_f32_in_bf16_out_kernel(const float* __restrict__ x,
                                              const float* __restrict__ W,
                                              const float* __restrict__ b,
                                              __nv_bfloat16* __restrict__ out,
                                              int rows, int N, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * N) return;
    const int t = i / N, n = i % N;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(x[t * K + k], W[n * K + k], acc);
    out[i] = __float2bfloat16(acc + b[n]);
}

__global__ void linear_bf16_in_f32_out_kernel(const __nv_bfloat16* __restrict__ x,
                                              const float* __restrict__ W,
                                              const float* __restrict__ b,
                                              float* __restrict__ out,
                                              int rows, int N, int K) {
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= rows * N) return;
    const int t = o / N, n = o % N;
    float acc = 0.f;
    for (int k = lane; k < K; k += 32)
        acc = fmaf(__bfloat162float(x[(long long)t * K + k]), W[(long long)n * K + k], acc);
    for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc + b[n];
}

void linear_f32_in_bf16_out(const float* x, const float* W, const float* b,
                            __nv_bfloat16* out, int rows, int N, int K,
                            cudaStream_t stream) {
    // An empty call is a no-op and needs no pointers: a zero-element tensor can
    // legitimately report a null data_ptr, so the null check has to come second.
    if (rows <= 0 || N <= 0 || K <= 0) return;
    if (!x || !W || !b || !out)
        throw std::runtime_error("linear_f32_in_bf16_out: null pointer argument");
    linear_f32_in_bf16_out_kernel<<<(rows * N + 255) / 256, 256, 0, stream>>>(
        x, W, b, out, rows, N, K);
}

void linear_bf16_in_f32_out(const __nv_bfloat16* x, const float* W, const float* b,
                            float* out, int rows, int N, int K,
                            cudaStream_t stream) {
    // An empty call is a no-op and needs no pointers: a zero-element tensor can
    // legitimately report a null data_ptr, so the null check has to come second.
    if (rows <= 0 || N <= 0 || K <= 0) return;
    if (!x || !W || !b || !out)
        throw std::runtime_error("linear_bf16_in_f32_out: null pointer argument");
    // 8 warps per block, one output element per warp
    linear_bf16_in_f32_out_kernel<<<(rows * N + 7) / 8, 256, 0, stream>>>(
        x, W, b, out, rows, N, K);
}

}  // namespace kernels
}  // namespace flash_rt
