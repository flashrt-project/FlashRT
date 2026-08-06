// ============================================================================
//  Vectorized gate_silu_mul + fp4_quant + SFA (bit-exact with the F4 v2
//  register-only kernel in silu_mul_fp4_sfa_v2.cu).
//
//  Identical arithmetic and evaluation order; only the access width
//  changes: the 16 gate and 16 up halves arrive as two 16-byte loads each
//  and the packed block leaves as one 8-byte store, instead of 16 4-byte
//  loads and 8 single-byte stores. The scalar variant is memory-latency
//  bound at the decoder shape (S=10, H=4096).
//
//  Additive: new entry point; the v2 kernel remains for unaligned shapes.
// ============================================================================
#include "fused_fp4/norm_silu_fp4_sfa.cuh"

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

using CfgVec = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_gvec(float x) {
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

__device__ __forceinline__ float silu_gelu_mul_gvec(float g, float u) {
    // Same constants as csrc/kernels/activation.cu gate_silu_mul_merged_kernel.
    float gelu = g / (1.0f + expf(-1.5957691216057308f * g *
                                  (1.0f + 0.044715f * g * g)));
    return gelu * u;
}

template <class LayoutSF>
__global__ void gvec_silu_mul_fp4_sfa_kernel(
    const __half* __restrict__ merged,  // [S, 2H]
    uint2* __restrict__ packed,         // [S, H/2] bytes as uint2 per block
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    const __half* merged_row = merged + row * 2 * H;
    const int4* gate4 = reinterpret_cast<const int4*>(merged_row + col_base);
    const int4* up4 =
        reinterpret_cast<const int4*>(merged_row + H + col_base);

    const int4 graw[2] = {gate4[0], gate4[1]};
    const int4 uraw[2] = {up4[0], up4[1]};
    const __half2* g2 = reinterpret_cast<const __half2*>(graw);
    const __half2* u2 = reinterpret_cast<const __half2*>(uraw);

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float v0 = silu_gelu_mul_gvec(__half2float(g2[i].x),
                                            __half2float(u2[i].x));
        const float v1 = silu_gelu_mul_gvec(__half2float(g2[i].y),
                                            __half2float(u2[i].y));
        vals[2 * i] = v0;
        vals[2 * i + 1] = v1;
        const float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float bs_dq = static_cast<float>(bs_q);

    dst_sfa[layout(row, col_base, 0)] = *reinterpret_cast<uint8_t*>(&bs_q);

    const float inv_bs = 1.f / bs_dq;
    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t lo = fp32_to_e2m1_gvec(vals[2 * p] * inv_bs);
        const uint8_t hi = fp32_to_e2m1_gvec(vals[2 * p + 1] * inv_bs);
        ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    packed[row * n_blocks + block_idx] = out;
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int gate_silu_mul_fp4_sfa_vec_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    if (half_dim % 16 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(merged) & 15) ||
        (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
    const int n_blocks = half_dim / 16;
    const int threads = 128;
    dim3 grid(seq_len, (n_blocks + threads - 1) / threads);

    auto shape = cute::make_shape(seq_len, 1, half_dim, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFA(shape);
    gvec_silu_mul_fp4_sfa_kernel<<<grid, threads, 0, stream>>>(
        merged, reinterpret_cast<uint2*>(packed), sfa, layout, half_dim);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)merged; (void)packed; (void)sfa;
    (void)seq_len; (void)half_dim; (void)stream;
    return -2;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
