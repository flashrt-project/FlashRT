// ============================================================================
//  FlashRT — fused LayerNorm (gamma/beta) + NVFP4 quantize + SFA write.
//
//  SigLIP FFN front-end: replaces layer_norm_fp8 when the Up GEMM consumes
//  NVFP4 activations. The normalized value is rounded through fp16 before
//  quantization, so the output is bit-identical to running a fp16
//  LayerNorm followed by quantize_fp4_dynamic_sfa_fp16.
//
//  Grid: (S,) one CTA per row; 16-element quant blocks are handled one per
//  thread with 16-byte loads.
// ============================================================================
#include "layer_norm_fp4_sfa.cuh"

#include <cuda_fp16.h>
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

using CfgLN = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_ln(float x) {
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

__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < ((blockDim.x + 31) >> 5)) ? sh[lane] : 0.f;
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

template <class LayoutSF>
__global__ void layer_norm_fp4_sfa_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    const __half* __restrict__ inv_s,  // [D] AWQ inverse scales, or nullptr
    uint2* __restrict__ packed,        // [S, D/16] 8-byte blocks
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D, float eps) {
    const int r = blockIdx.x;
    const __half2* row2 =
        reinterpret_cast<const __half2*>(x + static_cast<long>(r) * D);
    const int D2 = D >> 1;
    __shared__ float sh[32];

    float s = 0.f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        const float2 v = __half22float2(row2[i]);
        s += v.x + v.y;
    }
    const float mean = block_sum(s, sh) / D;

    float var = 0.f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        const float2 v = __half22float2(row2[i]);
        const float d0 = v.x - mean, d1 = v.y - mean;
        var += d0 * d0 + d1 * d1;
    }
    const float rstd = rsqrtf(block_sum(var, sh) / D + eps);

    const int n_blocks = D >> 4;
    const int4* x4 = reinterpret_cast<const int4*>(x + static_cast<long>(r) * D);
    const int4* g4 = reinterpret_cast<const int4*>(gamma);
    const int4* b4 = reinterpret_cast<const int4*>(beta);
    for (int blk = threadIdx.x; blk < n_blocks; blk += blockDim.x) {
        const int4 xr[2] = {x4[2 * blk], x4[2 * blk + 1]};
        const int4 gr[2] = {g4[2 * blk], g4[2 * blk + 1]};
        const int4 br[2] = {b4[2 * blk], b4[2 * blk + 1]};
        const __half* xh = reinterpret_cast<const __half*>(xr);
        const __half* gh = reinterpret_cast<const __half*>(gr);
        const __half* bh = reinterpret_cast<const __half*>(br);
        int4 ir[2];
        const __half* ih = nullptr;
        if (inv_s != nullptr) {
            const int4* i4 = reinterpret_cast<const int4*>(inv_s);
            ir[0] = i4[2 * blk];
            ir[1] = i4[2 * blk + 1];
            ih = reinterpret_cast<const __half*>(ir);
        }
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float normed =
                (__half2float(xh[i]) - mean) * rstd * __half2float(gh[i]) +
                __half2float(bh[i]);
            if (inv_s != nullptr)
                normed *= __half2float(ih[i]);
            // fp16 rounding to stay bit-identical with the two-step
            // (fp16 LayerNorm, then quantize) reference.
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
        uint2 out;
        uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint8_t lo = fp32_to_e2m1_ln(vals[2 * p] * inv_bs);
            const uint8_t hi = fp32_to_e2m1_ln(vals[2 * p + 1] * inv_bs);
            ob[p] = static_cast<uint8_t>(lo | (hi << 4));
        }
        packed[static_cast<long>(r) * n_blocks + blk] = out;
    }
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int layer_norm_mul_fp4_sfa_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    const __half* inv_s,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (dim % 16 != 0 || dim % 2 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(gamma) & 15) ||
        (reinterpret_cast<uintptr_t>(beta) & 15) ||
        (reinterpret_cast<uintptr_t>(inv_s) & 15) ||
        (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgLN::tile_atom_to_shape_SFA(shape);
    layer_norm_fp4_sfa_kernel<<<seq_len, 128, 0, stream>>>(
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
