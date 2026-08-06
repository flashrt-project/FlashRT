// ============================================================================
//  E0M3 (uniform INT4) quantize + CUTLASS SFA/SFB tile-interleaved scales.
//
//  Mirrors the structure of quantize_fp4_sfa.cu with the E2M1 element
//  encoder replaced by the sign-magnitude integer grid. Device helpers are
//  duplicated locally to stay additive (no linkage against the existing
//  quantizer translation units).
// ============================================================================
#include "quantize_e0m3_sfa.cuh"

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
namespace fp4 {

#if FV_HAVE_CUTLASS

namespace {

using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// Sign-magnitude uniform INT4: code = s|mmm, value = (s ? -1 : 1) * mmm.
// Round-to-nearest integer, clamp magnitude to 7; -0 normalizes to +0.
__device__ __forceinline__ uint8_t fp32_to_e0m3(float x) {
    int mag = __float2int_rn(fabsf(x));
    if (mag > 7) mag = 7;
    uint8_t sign = (x < 0.f && mag > 0) ? 0x8u : 0x0u;
    return sign | static_cast<uint8_t>(mag);
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3_e0m3(float x) {
    return __nv_fp8_e4m3(fmaxf(x, 0.f));
}

template <class LayoutSF>
__global__ void kernel_quantize_e0m3_sfa(
    const __half* __restrict__ src,
    uint8_t* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int N, int D) {
    const int row = blockIdx.y;
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_blocks = D / 16;
    if (row >= N || block_idx >= n_blocks) return;

    const __half* block_src = src + row * D + block_idx * 16;
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __half2float(block_src[i]);
        float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }

    // IEEE-exact division/reciprocal: --use_fast_math would otherwise
    // lower these to approximate instructions, nudging exact fp8 rounding
    // ties and making the output build-flag-dependent.
    float desired = __fdiv_rn(amax, 7.f);
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 scale_q = quantize_ue4m3_e0m3(desired);
    float scale_dq = static_cast<float>(scale_q);
    dst_sfa[layout(row, block_idx * 16, 0)] =
        *reinterpret_cast<uint8_t*>(&scale_q);

    const float inv = __frcp_rn(scale_dq);
    const int out_base = row * (D / 2) + block_idx * 8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e0m3(vals[2 * p] * inv);
        uint8_t hi = fp32_to_e0m3(vals[2 * p + 1] * inv);
        dst_packed[out_base + p] = lo | (hi << 4);
    }
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int quantize_e0m3_dynamic_sfa_fp16(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (D % 16 != 0) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);

  auto shape = cute::make_shape(
      is_sfb ? 1 : N,
      is_sfb ? N : 1,
      D, 1);

  if (is_sfb) {
    auto layout = Cfg::tile_atom_to_shape_SFB(shape);
    kernel_quantize_e0m3_sfa<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D);
  } else {
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    kernel_quantize_e0m3_sfa<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(src_fp16),
        reinterpret_cast<uint8_t*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D);
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_fp16; (void)dst_packed; (void)dst_sfa;
  (void)N; (void)D; (void)is_sfb; (void)stream;
  return -2;
#endif
}

}  // namespace fp4
}  // namespace flash_rt
