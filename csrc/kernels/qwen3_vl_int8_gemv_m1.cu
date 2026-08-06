#include "qwen3_vl_int8_gemv_m1.cuh"

#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr int kRows = 1;                             // output rows per warp

// One warp computes kRows output rows, templated on R so the activation can be
// loaded once per K-block and reused across rows (register blocking) if a wider
// tile ever wins. kRows=1 measured best on Orin's 16 SMs, so today each warp
// owns one row.
// Weights dequant via hardware int8->float I2F (no HW FP8 on Ampere sm_87);
// per-16 bf16 block scale applied once per (row, block); fp32 accumulation.
// Measured non-wins (do not re-add): smem-staging x (occupancy loss on 16 SMs);
// half2/__hfma2 accumulation (int8 per-16 partial can overflow half on outlier
// activations — fp32 is the safe choice).
template<int K_FIXED, int R>
__global__ __launch_bounds__(kThreads) void qwen3_vl_int8_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ Wp,
    const __nv_bfloat16* __restrict__ Ws,
    __nv_bfloat16* __restrict__ out,
    int N) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * kWarpsPerBlock + warp) * R;
    if (n0 >= N) return;

    constexpr int NG = K_FIXED >> 4;                 // K / 16 blocks
    const int4* x_i4 = reinterpret_cast<const int4*>(x);

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (int g = lane; g < NG; g += 32) {
        // Activation: 16 bf16 -> 16 float, loaded once, reused for all R rows.
        int4 xv0 = x_i4[g * 2];
        int4 xv1 = x_i4[g * 2 + 1];
        float xf[16];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 xb0 = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv0)[k]));
            float2 f0 = __bfloat1622float2(xb0);
            xf[2 * k] = f0.x; xf[2 * k + 1] = f0.y;
            __nv_bfloat162 xb1 = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv1)[k]));
            float2 f1 = __bfloat1622float2(xb1);
            xf[8 + 2 * k] = f1.x; xf[8 + 2 * k + 1] = f1.y;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const int nn = n0 + r;
            if (nn >= N) break;
            const uint4* wrow = reinterpret_cast<const uint4*>(
                Wp + (size_t)nn * K_FIXED);
            uint4 wp = wrow[g];                       // 16 int8 (16 B)
            const int8_t* w1 = reinterpret_cast<const int8_t*>(&wp);
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                part = fmaf(static_cast<float>(w1[j]), xf[j], part);
            }
            acc[r] = fmaf(part, __bfloat162float(Ws[(size_t)nn * NG + g]),
                          acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float a = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, off);
        }
        if (lane == 0 && (n0 + r) < N) {
            out[n0 + r] = __float2bfloat16(a);
        }
    }
}

}  // namespace

void qwen3_vl_int8_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream) {
    const int rows_per_block = kWarpsPerBlock * kRows;
    dim3 grid((N + rows_per_block - 1) / rows_per_block);
    if (K == 2048) {
        qwen3_vl_int8_gemv_m1_kernel<2048, kRows>
            <<<grid, kThreads, 0, stream>>>(x, Wp, Ws, out, N);
    } else if (K == 6144) {
        qwen3_vl_int8_gemv_m1_kernel<6144, kRows>
            <<<grid, kThreads, 0, stream>>>(x, Wp, Ws, out, N);
    } else {
        throw std::runtime_error(
            "qwen3_vl_int8_gemv_m1 supports only K=2048 or K=6144, got K=" +
            std::to_string(K));
    }
}

}  // namespace flash_rt::kernels
