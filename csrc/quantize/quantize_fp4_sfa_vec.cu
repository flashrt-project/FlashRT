// ============================================================================
//  Vectorized fused FP4 quantize + CUTLASS SFA/SFB tile-interleaved scales.
//
//  Bit-exact drop-in for quantize_fp4_dynamic_sfa_fp16 (quantize_fp4_sfa.cu):
//  identical per-block scale selection and e2m1 rounding, but each thread
//  moves its 16 source halves with two 16-byte loads and stores the packed
//  block as one 8-byte store instead of 16 scalar loads + 8 byte stores.
//  The scalar kernel is memory-latency-bound at decoder activation sizes.
//
//  Additive: does not modify quantize_fp4_sfa.cu.
// ============================================================================
#include "quantize_fp4_sfa.cuh"

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

using CfgVec = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1_vec(float x) {
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
__global__ void kernel_quantize_fp4_sfa_vec(
    const int4* __restrict__ src,      // fp16 [N, D] as int4 (8 halves)
    uint2* __restrict__ dst_packed,    // [N, D/2] bytes as uint2 (1 block)
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int N, int D8) {                   // D8 = D / 8 int4 chunks per row
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int n_blocks = D8 >> 1;        // 16 halves per block
  if (row >= N || block_idx >= n_blocks) return;

  const int4 raw0 = src[row * D8 + 2 * block_idx];
  const int4 raw1 = src[row * D8 + 2 * block_idx + 1];
  const __half* h0 = reinterpret_cast<const __half*>(&raw0);
  const __half* h1 = reinterpret_cast<const __half*>(&raw1);

  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    vals[i] = __half2float(h0[i]);
    vals[8 + i] = __half2float(h1[i]);
  }
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float a = fabsf(vals[i]);
    if (a > amax) amax = a;
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
    const uint8_t lo = fp32_to_e2m1_vec(vals[2 * p] * inv_bs);
    const uint8_t hi = fp32_to_e2m1_vec(vals[2 * p + 1] * inv_bs);
    ob[p] = static_cast<uint8_t>(lo | (hi << 4));
  }
  dst_packed[row * n_blocks + block_idx] = out;
}

}  // namespace

#endif  // FV_HAVE_CUTLASS

int quantize_fp4_dynamic_sfa_fp16_vec(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (D % 16 != 0) return -1;
  // 16-byte aligned rows and 8-byte aligned packed rows.
  if ((reinterpret_cast<uintptr_t>(src_fp16) & 15) ||
      (reinterpret_cast<uintptr_t>(dst_packed) & 7)) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);

  auto shape = cute::make_shape(
      is_sfb ? 1 : N,
      is_sfb ? N : 1,
      D, 1);

  if (is_sfb) {
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    kernel_quantize_fp4_sfa_vec<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(src_fp16),
        reinterpret_cast<uint2*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D >> 3);
  } else {
    auto layout = CfgVec::tile_atom_to_shape_SFA(shape);
    kernel_quantize_fp4_sfa_vec<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(src_fp16),
        reinterpret_cast<uint2*>(dst_packed),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, N, D >> 3);
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_fp16; (void)dst_packed; (void)dst_sfa;
  (void)N; (void)D; (void)is_sfb; (void)stream;
  return -2;
#endif
}

}  // namespace fp4
}  // namespace flash_rt
