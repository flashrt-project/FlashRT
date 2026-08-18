// ============================================================================
//  Fused SiLU(gate) * up (bf16) + NVFP4 quantize + SFA write.
//  One thread per 16-element block: two int4 loads per operand, silu in
//  fp32 rounded to bf16, bf16 multiply, then the standard per-block scale
//  selection + e2m1 rounding + tile-interleaved SFA byte.
// ============================================================================
#include "silu_mul_fp4_sfa_bf16.cuh"

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

using CfgSM = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_sm(float x) {
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

template <class LayoutSF>
__global__ void kernel_silu_mul_fp4_sfa_bf16(
    const int4* __restrict__ gate,
    const int4* __restrict__ up,
    uint2* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int N, int D8) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int n_blocks = D8 >> 1;
  if (row >= N || block_idx >= n_blocks) return;

  const int4 g0 = gate[row * D8 + 2 * block_idx];
  const int4 g1 = gate[row * D8 + 2 * block_idx + 1];
  const int4 u0 = up[row * D8 + 2 * block_idx];
  const int4 u1 = up[row * D8 + 2 * block_idx + 1];
  const __nv_bfloat16* gh0 = reinterpret_cast<const __nv_bfloat16*>(&g0);
  const __nv_bfloat16* gh1 = reinterpret_cast<const __nv_bfloat16*>(&g1);
  const __nv_bfloat16* uh0 = reinterpret_cast<const __nv_bfloat16*>(&u0);
  const __nv_bfloat16* uh1 = reinterpret_cast<const __nv_bfloat16*>(&u1);

  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float g[2] = {__bfloat162float(gh0[i]), __bfloat162float(gh1[i])};
    const __nv_bfloat16 u[2] = {uh0[i], uh1[i]};
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float s = g[h] / (1.f + expf(-g[h]));         // silu fp32
      const __nv_bfloat16 sb = __float2bfloat16(s);       // round like torch
      const __nv_bfloat16 prod = __hmul(sb, u[h]);        // bf16 multiply
      vals[h * 8 + i] = __bfloat162float(prod);
      const float a = fabsf(vals[h * 8 + i]);
      if (a > amax) amax = a;
    }
  }

  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
  const float bs_dq = static_cast<float>(bs_q);

  dst_sfa[layout(row, block_idx * 16, 0)] =
      *reinterpret_cast<uint8_t*>(&bs_q);

  const float inv_bs = 1.f / bs_dq;
  uint2 out;
  uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint8_t lo = fp32_to_e2m1_sm(vals[2 * p] * inv_bs);
    const uint8_t hi = fp32_to_e2m1_sm(vals[2 * p + 1] * inv_bs);
    ob[p] = static_cast<uint8_t>(lo | (hi << 4));
  }
  dst_packed[row * n_blocks + block_idx] = out;
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int silu_mul_fp4_sfa_bf16(
    const void* gate, const void* up, void* packed, void* sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (D % 16 != 0) return -1;
  if ((reinterpret_cast<uintptr_t>(gate) & 15) ||
      (reinterpret_cast<uintptr_t>(up) & 15) ||
      (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);

  auto shape = cute::make_shape(
      is_sfb ? 1 : N,
      is_sfb ? N : 1,
      D, 1);

  if (is_sfb) {
    auto layout = CfgSM::tile_atom_to_shape_SFB(shape);
    kernel_silu_mul_fp4_sfa_bf16<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(gate),
        reinterpret_cast<const int4*>(up),
        reinterpret_cast<uint2*>(packed),
        reinterpret_cast<uint8_t*>(sfa),
        layout, N, D >> 3);
  } else {
    auto layout = CfgSM::tile_atom_to_shape_SFA(shape);
    kernel_silu_mul_fp4_sfa_bf16<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(gate),
        reinterpret_cast<const int4*>(up),
        reinterpret_cast<uint2*>(packed),
        reinterpret_cast<uint8_t*>(sfa),
        layout, N, D >> 3);
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)gate; (void)up; (void)packed; (void)sfa;
  (void)N; (void)D; (void)is_sfb; (void)stream;
  return -2;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
