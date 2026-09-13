// See pi05_rowops_v2.cuh.
#include "fused_fp4/pi05_rowops_v2.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_rt {
namespace fused_fp4 {
namespace {

using CfgSF = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int ROWS_PER_CTA = 8;
constexpr int THREADS = ROWS_PER_CTA * 32;
constexpr int MAX_BPL = 4;   // blocks of 16 per lane => D <= 2048

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void load16(const __half* p, float* v) {
  const int4 a = reinterpret_cast<const int4*>(p)[0];
  const int4 b = reinterpret_cast<const int4*>(p)[1];
  const __half* ha = reinterpret_cast<const __half*>(&a);
  const __half* hb = reinterpret_cast<const __half*>(&b);
  #pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = __half2float(ha[i]); v[8 + i] = __half2float(hb[i]); }
}

__device__ __forceinline__ void store16_half(__half* p, const float* v) {
  int4 a, b;
  __half* ha = reinterpret_cast<__half*>(&a);
  __half* hb = reinterpret_cast<__half*>(&b);
  #pragma unroll
  for (int i = 0; i < 8; ++i) { ha[i] = __float2half(v[i]); hb[i] = __float2half(v[8 + i]); }
  reinterpret_cast<int4*>(p)[0] = a;
  reinterpret_cast<int4*>(p)[1] = b;
}

// vals must already carry the fp16 rounding of the reference path.
template <class LayoutSF>
__device__ __forceinline__ void quant16_store(
    const float* vals, int row, int blk, int nb,
    uint2* __restrict__ packed, uint8_t* __restrict__ sfa, LayoutSF layout) {
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(vals[i]));
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
  const float bs_dq = static_cast<float>(bs_q);
  sfa[layout(row, blk * 16, 0)] = *reinterpret_cast<uint8_t*>(&bs_q);
  const float inv_bs = 1.f / bs_dq;
  uint2 out;
  uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    ob[p] = __nv_cvt_float2_to_fp4x2(
        make_float2(vals[2 * p] * inv_bs, vals[2 * p + 1] * inv_bs),
        __NV_E2M1, cudaRoundNearest);
  }
  packed[static_cast<size_t>(row) * nb + blk] = out;
}

__device__ __forceinline__ void store16_fp8(uint8_t* p, const float* v) {
  uint4 out;
  uint16_t* o2 = reinterpret_cast<uint16_t*>(&out);
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    o2[i] = __nv_cvt_float2_to_fp8x2(make_float2(v[2 * i], v[2 * i + 1]),
                                     __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint4*>(p) = out;
}

enum Mode { kResRmsMulFp4 = 0, kQuantFp4 = 1, kResRmsFp8 = 2, kRmsFp8 = 3,
            kLnMulFp4 = 4, kLnFp8 = 5, kRmsMulFp4 = 6 };

template <int MODE, int BPL, class LayoutSF>
__global__ void __launch_bounds__(THREADS)
rowops_kernel(__half* __restrict__ residual, const __half* __restrict__ x,
              const __half* __restrict__ gamma, const __half* __restrict__ beta,
              const __half* __restrict__ inv_s, const float* __restrict__ descale,
              uint2* __restrict__ packed, uint8_t* __restrict__ sfa,
              uint8_t* __restrict__ out_fp8, LayoutSF layout,
              int S, int D, float eps) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * ROWS_PER_CTA + warp;
  if (row >= S) return;
  const int nb = D >> 4;
  const size_t rowoff = static_cast<size_t>(row) * D;

  float v[BPL][16];
  float acc = 0.f;
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    const int b = lane + 32 * j;
    if (b < nb) {
      if (MODE == kResRmsMulFp4 || MODE == kResRmsFp8) {
        float r[16];
        load16(residual + rowoff + b * 16, r);
        load16(x + rowoff + b * 16, v[j]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) { v[j][i] += r[i]; acc += v[j][i] * v[j][i]; }
        store16_half(residual + rowoff + b * 16, v[j]);
      } else {
        load16(x + rowoff + b * 16, v[j]);
        if (MODE == kRmsFp8 || MODE == kRmsMulFp4) {
          #pragma unroll
          for (int i = 0; i < 16; ++i) acc += v[j][i] * v[j][i];
        } else if (MODE == kLnMulFp4 || MODE == kLnFp8) {
          #pragma unroll
          for (int i = 0; i < 16; ++i) acc += v[j][i];
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 16; ++i) v[j][i] = 0.f;
    }
  }

  float scale = 1.f, mean = 0.f, rstd = 1.f;
  if (MODE == kResRmsMulFp4 || MODE == kResRmsFp8 || MODE == kRmsFp8 || MODE == kRmsMulFp4) {
    const float ssq = warp_sum(acc);
    scale = __frsqrt_rn(ssq / D + 1e-6f);
    if (MODE == kResRmsFp8 || MODE == kRmsFp8) scale /= fmaxf(*descale, 1e-12f);
  } else if (MODE == kLnMulFp4 || MODE == kLnFp8) {
    mean = warp_sum(acc) / D;
    float var = 0.f;
    #pragma unroll
    for (int j = 0; j < BPL; ++j) {
      if (lane + 32 * j < nb) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) { const float d = v[j][i] - mean; var += d * d; }
      }
    }
    rstd = rsqrtf(warp_sum(var) / D + eps);
  }

  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    const int b = lane + 32 * j;
    if (b >= nb) continue;
    float vals[16];
    if (MODE == kResRmsMulFp4 || MODE == kRmsMulFp4) {
      float is[16];
      if (inv_s != nullptr) load16(inv_s + b * 16, is);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        float t = v[j][i] * scale;
        if (inv_s != nullptr) t *= is[i];
        vals[i] = __half2float(__float2half(t));
      }
      quant16_store(vals, row, b, nb, packed, sfa, layout);
    } else if (MODE == kQuantFp4) {
      quant16_store(v[j], row, b, nb, packed, sfa, layout);
    } else if (MODE == kResRmsFp8 || MODE == kRmsFp8) {
      #pragma unroll
      for (int i = 0; i < 16; ++i)
        vals[i] = fminf(fmaxf(v[j][i] * scale, -448.f), 448.f);
      store16_fp8(out_fp8 + rowoff + b * 16, vals);
    } else {
      float g[16], bt[16], is[16];
      load16(gamma + b * 16, g);
      load16(beta + b * 16, bt);
      if (MODE == kLnMulFp4 && inv_s != nullptr) load16(inv_s + b * 16, is);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        float t = (v[j][i] - mean) * rstd * g[i] + bt[i];
        if (MODE == kLnMulFp4) {
          if (inv_s != nullptr) t *= is[i];
          vals[i] = __half2float(__float2half(t));
        } else {
          vals[i] = t;
        }
      }
      if (MODE == kLnMulFp4) quant16_store(vals, row, b, nb, packed, sfa, layout);
      else store16_fp8(out_fp8 + rowoff + b * 16, vals);
    }
  }
}

template <int MODE>
int launch(__half* residual, const __half* x, const __half* gamma,
           const __half* beta, const __half* inv_s, const float* descale,
           void* packed, void* sfa, void* out_fp8, int S, int D, float eps,
           cudaStream_t stream) {
  if (D % 16 != 0 || D > 16 * 32 * MAX_BPL || S <= 0) return -1;
  auto check = [](const void* p, int a) { return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1)); };
  if (check(residual, 16) || check(x, 16) || check(gamma, 16) || check(beta, 16) ||
      check(inv_s, 16) || check(packed, 8) || check(out_fp8, 16)) return -1;
  auto layout = CfgSF::tile_atom_to_shape_SFA(cute::make_shape(S, 1, D, 1));
  const int nb = D / 16;
  const int bpl = (nb + 31) / 32;
  const dim3 grid((S + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
  auto* pk = reinterpret_cast<uint2*>(packed);
  auto* sf = reinterpret_cast<uint8_t*>(sfa);
  auto* o8 = reinterpret_cast<uint8_t*>(out_fp8);
  switch (bpl) {
    case 1: rowops_kernel<MODE, 1><<<grid, THREADS, 0, stream>>>(residual, x, gamma, beta, inv_s, descale, pk, sf, o8, layout, S, D, eps); break;
    case 2: rowops_kernel<MODE, 2><<<grid, THREADS, 0, stream>>>(residual, x, gamma, beta, inv_s, descale, pk, sf, o8, layout, S, D, eps); break;
    case 3: rowops_kernel<MODE, 3><<<grid, THREADS, 0, stream>>>(residual, x, gamma, beta, inv_s, descale, pk, sf, o8, layout, S, D, eps); break;
    case 4: rowops_kernel<MODE, 4><<<grid, THREADS, 0, stream>>>(residual, x, gamma, beta, inv_s, descale, pk, sf, o8, layout, S, D, eps); break;
    default: return -1;
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace

int rowops_residual_rms_mul_fp4_sfa_v2(__half* residual, const __half* x, const __half* inv_s,
                                       void* packed, void* sfa, int S, int D, cudaStream_t stream) {
  return launch<kResRmsMulFp4>(residual, x, nullptr, nullptr, inv_s, nullptr, packed, sfa, nullptr, S, D, 0.f, stream);
}
int rowops_rms_mul_fp4_sfa_v2(const __half* x, const __half* inv_s, void* packed, void* sfa,
                              int S, int D, cudaStream_t stream) {
  return launch<kRmsMulFp4>(nullptr, x, nullptr, nullptr, inv_s, nullptr, packed, sfa, nullptr, S, D, 0.f, stream);
}
int rowops_quantize_fp4_sfa_v2(const __half* src, void* packed, void* sfa, int N, int D, cudaStream_t stream) {
  return launch<kQuantFp4>(nullptr, src, nullptr, nullptr, nullptr, nullptr, packed, sfa, nullptr, N, D, 0.f, stream);
}
int rowops_residual_rms_fp8_v2(__half* residual, const __half* x, void* out_fp8, int S, int D,
                               const float* descale, cudaStream_t stream) {
  return launch<kResRmsFp8>(residual, x, nullptr, nullptr, nullptr, descale, nullptr, nullptr, out_fp8, S, D, 0.f, stream);
}
int rowops_rms_fp8_v2(const __half* x, void* out_fp8, int S, int D, const float* descale, cudaStream_t stream) {
  return launch<kRmsFp8>(nullptr, x, nullptr, nullptr, nullptr, descale, nullptr, nullptr, out_fp8, S, D, 0.f, stream);
}
int rowops_layer_norm_mul_fp4_sfa_v2(const __half* x, const __half* gamma, const __half* beta, const __half* inv_s,
                                     void* packed, void* sfa, int S, int D, float eps, cudaStream_t stream) {
  return launch<kLnMulFp4>(nullptr, x, gamma, beta, inv_s, nullptr, packed, sfa, nullptr, S, D, eps, stream);
}
int rowops_layer_norm_fp8_v2(const __half* x, const __half* gamma, const __half* beta, void* out_fp8,
                             int S, int D, float eps, cudaStream_t stream) {
  return launch<kLnFp8>(nullptr, x, gamma, beta, nullptr, nullptr, nullptr, nullptr, out_fp8, S, D, eps, stream);
}

}  // namespace fused_fp4
}  // namespace flash_rt
