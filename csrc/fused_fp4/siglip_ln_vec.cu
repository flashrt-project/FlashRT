// ============================================================================
//  Vectorized SigLIP LayerNorm kernels — register-resident, one DRAM pass.
//
//  Layout: 128 threads per row; thread t owns the 16 contiguous elements of
//  quant block t (two 16-byte loads). Mean and variance reduce across the
//  active threads; normalize/quantize then runs entirely from registers.
//  The FP8 variant stores its 16 output bytes as one 16-byte transaction;
//  the FP4 variant packs 8 bytes + one scale byte with no cross-thread
//  traffic (the per-16 quant block is thread-local).
// ============================================================================
#include "fused_fp4/siglip_ln_vec.cuh"

#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

namespace {

using CfgLNV = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int kLNVecThreads = 128;

__device__ __forceinline__ uint8_t fp32_to_e2m1_lnv(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__device__ __forceinline__ float block_sum_lnv(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < (kLNVecThreads >> 5)) ? sh[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

// Loads this thread's 16-element block into vals[]; returns sum. Inactive
// threads (blk >= n_blocks) contribute zeros.
__device__ __forceinline__ float load_block16(
    const __half* row, int blk, int n_blocks, float vals[16]) {
    float s = 0.f;
    if (blk < n_blocks) {
        const int4* p4 = reinterpret_cast<const int4*>(row) + 2 * blk;
        const int4 rawv[2] = {p4[0], p4[1]};
        const __half* h = reinterpret_cast<const __half*>(rawv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = __half2float(h[i]);
            s += vals[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) vals[i] = 0.f;
    }
    return s;
}

__global__ void layer_norm_fp8_vec_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    uint8_t* __restrict__ out,
    int D, float eps) {
    const int r = blockIdx.x;
    const int blk = threadIdx.x;
    const int n_blocks = D >> 4;
    const __half* row = x + static_cast<long>(r) * D;
    __shared__ float sh[32];

    float vals[16];
    const float s = load_block16(row, blk, n_blocks, vals);
    const float mean = block_sum_lnv(s, sh) / D;

    float var = 0.f;
    if (blk < n_blocks) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float d = vals[i] - mean;
            var += d * d;
        }
    }
    const float rstd = rsqrtf(block_sum_lnv(var, sh) / D + eps);

    if (blk < n_blocks) {
        const int4* g4 = reinterpret_cast<const int4*>(gamma) + 2 * blk;
        const int4* b4 = reinterpret_cast<const int4*>(beta) + 2 * blk;
        const int4 gr[2] = {g4[0], g4[1]};
        const int4 br[2] = {b4[0], b4[1]};
        const __half* gh = reinterpret_cast<const __half*>(gr);
        const __half* bh = reinterpret_cast<const __half*>(br);
        uint4 packed_out;
        uint8_t* ob = reinterpret_cast<uint8_t*>(&packed_out);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float normed =
                (vals[i] - mean) * rstd * __half2float(gh[i]) +
                __half2float(bh[i]);
            const __nv_fp8_e4m3 q = __nv_fp8_e4m3(normed);
            ob[i] = *reinterpret_cast<const uint8_t*>(&q);
        }
        reinterpret_cast<uint4*>(out + static_cast<long>(r) * D)[blk] =
            packed_out;
    }
}

template <class LayoutSF>
__global__ void layer_norm_mul_fp4_sfa_vec_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    const __half* __restrict__ inv_s,  // nullptr for the plain path
    uint2* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D, float eps) {
    const int r = blockIdx.x;
    const int blk = threadIdx.x;
    const int n_blocks = D >> 4;
    const __half* row = x + static_cast<long>(r) * D;
    __shared__ float sh[32];

    float vals[16];
    const float s = load_block16(row, blk, n_blocks, vals);
    const float mean = block_sum_lnv(s, sh) / D;

    float var = 0.f;
    if (blk < n_blocks) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float d = vals[i] - mean;
            var += d * d;
        }
    }
    const float rstd = rsqrtf(block_sum_lnv(var, sh) / D + eps);

    if (blk >= n_blocks) return;

    const int4* g4 = reinterpret_cast<const int4*>(gamma) + 2 * blk;
    const int4* b4 = reinterpret_cast<const int4*>(beta) + 2 * blk;
    const int4 gr[2] = {g4[0], g4[1]};
    const int4 br[2] = {b4[0], b4[1]};
    const __half* gh = reinterpret_cast<const __half*>(gr);
    const __half* bh = reinterpret_cast<const __half*>(br);
    int4 ir[2];
    const __half* ih = nullptr;
    if (inv_s != nullptr) {
        const int4* i4 = reinterpret_cast<const int4*>(inv_s) + 2 * blk;
        ir[0] = i4[0];
        ir[1] = i4[1];
        ih = reinterpret_cast<const __half*>(ir);
    }

    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float normed = (vals[i] - mean) * rstd * __half2float(gh[i]) +
                       __half2float(bh[i]);
        if (inv_s != nullptr) normed *= __half2float(ih[i]);
        // fp16 rounding matches the two-step (fp16 LayerNorm, then
        // quantize) reference, same as the original fused kernel.
        vals[i] = __half2float(__float2half(normed));
        const float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float bs_dq = static_cast<float>(bs_q);
    dst_sfa[layout(r, blk * 16, 0)] = *reinterpret_cast<uint8_t*>(&bs_q);
    const float inv_bs = 1.f / bs_dq;
    uint2 outb;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&outb);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t lo = fp32_to_e2m1_lnv(vals[2 * p] * inv_bs);
        const uint8_t hi = fp32_to_e2m1_lnv(vals[2 * p + 1] * inv_bs);
        ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    packed[static_cast<long>(r) * n_blocks + blk] = outb;
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int layer_norm_fp8_vec_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    void* out_fp8, int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (dim % 16 != 0 || (dim >> 4) > kLNVecThreads) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(gamma) & 15) ||
        (reinterpret_cast<uintptr_t>(beta) & 15) ||
        (reinterpret_cast<uintptr_t>(out_fp8) & 15)) return -1;
    layer_norm_fp8_vec_kernel<<<seq_len, kLNVecThreads, 0, stream>>>(
        x, gamma, beta, reinterpret_cast<uint8_t*>(out_fp8), dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)x; (void)gamma; (void)beta; (void)out_fp8;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

int layer_norm_mul_fp4_sfa_vec_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    const __half* inv_s,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (dim % 16 != 0 || (dim >> 4) > kLNVecThreads) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(gamma) & 15) ||
        (reinterpret_cast<uintptr_t>(beta) & 15) ||
        (reinterpret_cast<uintptr_t>(inv_s) & 15) ||
        (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgLNV::tile_atom_to_shape_SFA(shape);
    layer_norm_mul_fp4_sfa_vec_kernel<<<seq_len, kLNVecThreads, 0, stream>>>(
        x, gamma, beta, inv_s,
        reinterpret_cast<uint2*>(packed),
        reinterpret_cast<uint8_t*>(sfa),
        layout, dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)x; (void)gamma; (void)beta; (void)inv_s; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
