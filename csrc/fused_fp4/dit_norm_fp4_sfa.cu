// ============================================================================
//  Fused DiT norms (bf16) + NVFP4 quantize + SFA write. See header.
//
//  Grid: (S,) one CTA per row, 128 threads. Row mean/var via the standard
//  two-pass block reduction; 16-element quant blocks one per thread with
//  16-byte loads. Scale selection and e2m1 rounding are identical to
//  quantize_fp4_dynamic_sfa_bf16_vec.
// ============================================================================
#include "dit_norm_fp4_sfa.cuh"

#include <cuda_bf16.h>
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

using CfgDN = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_dn(float x) {
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

__device__ __forceinline__ float block_sum_dn(float v, float* sh) {
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

// HAS_MOD=true: AdaLN (modulate with (1+scale), shift). false: plain LN.
template <bool HAS_MOD, class LayoutSF>
__global__ void dit_norm_fp4_sfa_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,   // [D] or nullptr
    const __nv_bfloat16* __restrict__ shift,   // [D] or nullptr
    uint2* __restrict__ packed,                // [S, D/16] 8-byte blocks
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D, float eps) {
    const int r = blockIdx.x;
    const __nv_bfloat162* row2 =
        reinterpret_cast<const __nv_bfloat162*>(x + static_cast<long>(r) * D);
    const int D2 = D >> 1;
    __shared__ float sh[32];

    float s = 0.f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        const __nv_bfloat162 v = row2[i];
        s += __bfloat162float(v.x) + __bfloat162float(v.y);
    }
    const float mean = block_sum_dn(s, sh) / D;

    float var = 0.f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        const __nv_bfloat162 v = row2[i];
        const float d0 = __bfloat162float(v.x) - mean;
        const float d1 = __bfloat162float(v.y) - mean;
        var += d0 * d0 + d1 * d1;
    }
    const float rstd = rsqrtf(block_sum_dn(var, sh) / D + eps);

    const int n_blocks = D >> 4;
    const int4* x4 = reinterpret_cast<const int4*>(x + static_cast<long>(r) * D);
    const int4* s4 = reinterpret_cast<const int4*>(scale);
    const int4* h4 = reinterpret_cast<const int4*>(shift);
    for (int blk = threadIdx.x; blk < n_blocks; blk += blockDim.x) {
        const int4 xr[2] = {x4[2 * blk], x4[2 * blk + 1]};
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(xr);
        int4 sr[2], hr[2];
        const __nv_bfloat16* sc = nullptr;
        const __nv_bfloat16* sf = nullptr;
        if (HAS_MOD) {
            sr[0] = s4[2 * blk]; sr[1] = s4[2 * blk + 1];
            hr[0] = h4[2 * blk]; hr[1] = h4[2 * blk + 1];
            sc = reinterpret_cast<const __nv_bfloat16*>(sr);
            sf = reinterpret_cast<const __nv_bfloat16*>(hr);
        }
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float normed = (__bfloat162float(xh[i]) - mean) * rstd;
            if (HAS_MOD)
                normed = normed * (1.0f + __bfloat162float(sc[i])) +
                         __bfloat162float(sf[i]);
            // bf16 rounding keeps this bit-identical to the two-step
            // (bf16 norm kernel, then quantize) reference chain.
            vals[i] = __bfloat162float(__float2bfloat16(normed));
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
            const uint8_t lo = fp32_to_e2m1_dn(vals[2 * p] * inv_bs);
            const uint8_t hi = fp32_to_e2m1_dn(vals[2 * p + 1] * inv_bs);
            ob[p] = static_cast<uint8_t>(lo | (hi << 4));
        }
        packed[static_cast<long>(r) * n_blocks + blk] = out;
    }
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

static int check_dn_args(const void* x, void* packed, int dim) {
    if (dim % 16 != 0 || dim % 2 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
    return 0;
}

int ada_layer_norm_fp4_sfa_bf16(
    const void* x, const void* scale, const void* shift,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (check_dn_args(x, packed, dim) != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(scale) & 15) ||
        (reinterpret_cast<uintptr_t>(shift) & 15)) return -1;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgDN::tile_atom_to_shape_SFA(shape);
    dit_norm_fp4_sfa_kernel<true><<<seq_len, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<uint2*>(packed),
        reinterpret_cast<uint8_t*>(sfa),
        layout, dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)x; (void)scale; (void)shift; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

int layer_norm_no_affine_fp4_sfa_bf16(
    const void* x, void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (check_dn_args(x, packed, dim) != 0) return -1;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = CfgDN::tile_atom_to_shape_SFA(shape);
    dit_norm_fp4_sfa_kernel<false><<<seq_len, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        nullptr, nullptr,
        reinterpret_cast<uint2*>(packed),
        reinterpret_cast<uint8_t*>(sfa),
        layout, dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)x; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)eps; (void)stream;
    return -2;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
