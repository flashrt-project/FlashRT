#include "qwen3_vl_w4_gemv_m1.cuh"

#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

// Dequant 4 e2m1 nibbles (packed in the low 16 bits of `w`, one nibble per
// 4-bit field) to 4 e4m3 bytes (packed in the returned uint32), branchless via
// a __byte_perm 8-entry magnitude LUT + sign OR. Processing 4 nibbles per
// __byte_perm (which produces 4 bytes) — vs 1 byte/2 nibbles — halves the
// dequant ALU so the GEMV approaches the memory-bandwidth bound.
__device__ __forceinline__ unsigned dequant4(unsigned w) {
    unsigned mag = __byte_perm(0x3C383000u, 0x4C484440u, w & 0x7777u);
    unsigned sgn = ((w & 0x8u) << 4) | ((w & 0x80u) << 8)
                 | ((w & 0x800u) << 12) | ((w & 0x8000u) << 16);
    return mag | sgn;
}

// One warp per output row n. The activation x is shared across all rows, so
// the block cooperatively stages x (bf16 -> half) into dynamic shared memory
// ONCE (eliminating per-warp re-conversion + L2 re-reads). The dot product
// uses __hfma2 half2 SIMD (weight half2 from fp8x2->half2, x half2 from smem);
// per group the 8 pairs accumulate into a half2, then reduce to float and
// apply the per-block scale in fp32. This drops the compute floor so the GEMV
// reaches the memory-bandwidth bound.
__global__ __launch_bounds__(kThreads) void qwen3_vl_w4_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ Wp,
    const __nv_bfloat16* __restrict__ Ws,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
    extern __shared__ __half2 xsh[];                // K/2 half2 (= K halfs)
    const int nh2 = K >> 1;
    const __nv_bfloat162* x2g =
        reinterpret_cast<const __nv_bfloat162*>(x);
    for (int i = threadIdx.x; i < nh2; i += kThreads) {
        xsh[i] = __float22half2_rn(__bfloat1622float2(x2g[i]));
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp;
    if (n >= N) return;

    const int NG = K >> 4;                          // K / 16 blocks
    const uint8_t* wrow = Wp + (size_t)n * (K >> 1);
    const __nv_bfloat16* srow = Ws + (size_t)n * NG;

    float acc = 0.0f;
    for (int g = lane; g < NG; g += 32) {
        uint2 wp = *reinterpret_cast<const uint2*>(wrow + (size_t)g * 8);
        unsigned e[4] = {dequant4(wp.x), dequant4(wp.x >> 16),
                         dequant4(wp.y), dequant4(wp.y >> 16)};
        const __half2* xg = xsh + g * 8;            // 8 half2 pairs / group
        __half2 acc2 = __floats2half2_rn(0.0f, 0.0f);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            __nv_fp8x2_e4m3 f8;
            f8.__x = (unsigned short)((p & 1) ? (e[p >> 1] >> 16)
                                              : (e[p >> 1] & 0xFFFFu));
            acc2 = __hfma2(static_cast<__half2>(f8), xg[p], acc2);
        }
        float part = __low2float(acc2) + __high2float(acc2);
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

void qwen3_vl_w4_gemv_m1(
    const __nv_bfloat16* x, const uint8_t* Wp, const __nv_bfloat16* Ws,
    __nv_bfloat16* out, int N, int K, cudaStream_t stream) {
    if (K <= 0 || (K % 16) != 0) {
        throw std::runtime_error(
            "qwen3_vl_w4_gemv_m1 requires K to be a positive multiple of 16 "
            "(one uint2 weight load and one block scale per 16 elements), "
            "got K=" + std::to_string(K));
    }
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    size_t smem = (size_t)K * sizeof(__half);
    // The activation is staged in dynamic shared memory; past the 48 KB
    // default the launch would fail with an opaque error at the next sync.
    if (smem > 48u * 1024u) {
        throw std::runtime_error(
            "qwen3_vl_w4_gemv_m1 stages K halves in dynamic shared memory and "
            "so supports K <= 24576 (48 KB); got K=" + std::to_string(K));
    }
    qwen3_vl_w4_gemv_m1_kernel<<<grid, kThreads, smem, stream>>>(
        x, Wp, Ws, out, N, K);
}

}  // namespace flash_rt::kernels
