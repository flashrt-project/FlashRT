#include "qwen3_vl_int4_gemv_m1.cuh"

#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr int kUnroll = 4;                           // independent weight loads in flight
                                                     // (must divide NG/32; NG/32 = 4 (K=2048) / 12 (K=6144))

__device__ __forceinline__ float dot_block(const uint8_t* wb, int4 xa, int4 xb) {
    float p = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 f0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(
            &(reinterpret_cast<int*>(&xa)[k])));
        int b0 = static_cast<int>(wb[k]);
        int lo0 = (static_cast<int>(static_cast<int8_t>(b0 << 4))) >> 4;
        int hi0 = (static_cast<int>(static_cast<int8_t>(b0))) >> 4;
        p = fmaf(static_cast<float>(lo0), f0.x, p);
        p = fmaf(static_cast<float>(hi0), f0.y, p);
        float2 f1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(
            &(reinterpret_cast<int*>(&xb)[k])));
        int b1 = static_cast<int>(wb[k + 4]);
        int lo1 = (static_cast<int>(static_cast<int8_t>(b1 << 4))) >> 4;
        int hi1 = (static_cast<int>(static_cast<int8_t>(b1))) >> 4;
        p = fmaf(static_cast<float>(lo1), f1.x, p);
        p = fmaf(static_cast<float>(hi1), f1.y, p);
    }
    return p;
}

// One warp per output row. The int4 GEMV is latency-bound on Orin LPDDR5 (ncu:
// ~88% occupancy, both mem/compute pipes moderate). To hide weight-load latency
// we keep max warps (1 row/warp) and instead lift instruction-level parallelism:
// each lane issues kUnroll INDEPENDENT uint2 weight loads (+ their activation)
// up front, then does the kUnroll dot-products — so several LPDDR reads are in
// flight at once. Nibble unpack: shift+sign-extend+I2F; per-16 bf16 scale; fp32
// accumulation. NG is a multiple of 32*kUnroll for K in {2048,6144} (no tail).
template<int K_FIXED>
__global__ __launch_bounds__(kThreads) void qwen3_vl_int4_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ Wp,
    const __nv_bfloat16* __restrict__ Ws,
    __nv_bfloat16* __restrict__ out,
    int N) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp;
    if (n >= N) return;

    constexpr int NG = K_FIXED >> 4;                 // per-16 blocks
    const uint2* wrow = reinterpret_cast<const uint2*>(
        Wp + (size_t)n * (K_FIXED / 2));
    const __nv_bfloat16* srow = Ws + (size_t)n * NG;
    const int4* x_i4 = reinterpret_cast<const int4*>(x);

    float acc = 0.0f;
    for (int g0 = lane; g0 < NG; g0 += 32 * kUnroll) {
        uint2 wp[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wp[u] = wrow[g0 + u * 32];            // independent HBM loads in flight
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int g = g0 + u * 32;
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wp[u]);
            float part = dot_block(wb, x_i4[g * 2], x_i4[g * 2 + 1]);
            acc = fmaf(part, __bfloat162float(srow[g]), acc);
        }
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

void qwen3_vl_int4_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream) {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if (K == 2048) {
        qwen3_vl_int4_gemv_m1_kernel<2048>
            <<<grid, kThreads, 0, stream>>>(x, Wp, Ws, out, N);
    } else if (K == 6144) {
        qwen3_vl_int4_gemv_m1_kernel<6144>
            <<<grid, kThreads, 0, stream>>>(x, Wp, Ws, out, N);
    } else {
        throw std::runtime_error(
            "qwen3_vl_int4_gemv_m1 supports only K=2048 or K=6144, got K=" +
            std::to_string(K));
    }
}

}  // namespace flash_rt::kernels
