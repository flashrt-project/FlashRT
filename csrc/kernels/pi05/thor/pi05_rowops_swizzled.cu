// ============================================================================
//  FlashRT — v5 warp-per-row activation kernels (see pi05_rowops_v5.cuh).
//
//  Nsight on the v2 kernels inside the pipeline: L2 hit 99%, IPC 2.5-3.0 of 4
//  issue slots while the SMs are active — instruction-issue bound, so the
//  instruction count per element is the lever. Relative to v2:
//    * the e2m1 midpoint handling (three compares, a select and a multiply per
//      element) is folded into the block reciprocal: stepping 1/scale down two
//      ulps moves every exact midpoint strictly below the tie, so the
//      round-to-nearest-even hardware cvt yields the reference chain's
//      toward-zero result at those points at no per-element cost;
//    * the block-scale address is the closed form of the CUTLASS
//      Sm1xxBlockScaledConfig<16> SFA layout (row base once, three integer ops
//      per block) instead of the generic layout evaluation;
//    * per-column tables are fp32 in a lane-major order (four 16-byte loads per
//      block, 4 L1 lines per warp-wide load) instead of fp16 + sixteen unpack
//      instructions;
//    * the e4m3 path relies on the saturating cvt instead of an explicit clamp.
//
//  fp8 outputs and block scales are bit-identical to v2; e2m1 values differ
//  only within a few fp32 ulps above a midpoint (rounded toward zero here).
// ============================================================================
#include "kernels/pi05/thor/pi05_rowops_swizzled.cuh"
#include "kernels/pi05/thor/pdl.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace flash_rt {
namespace fused_fp4 {
namespace {

constexpr int ROWS_PER_CTA = 8;
constexpr int THREADS = ROWS_PER_CTA * 32;
constexpr int MAX_BPL = 4;   // blocks of 16 per lane => D <= 2048

enum Mode { kQuantFp4 = 1, kRmsFp8 = 3, kLnMulFp4 = 4, kLnFp8 = 5, kRmsMulFp4 = 6 };

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Reduction in exactly the original vectorized-LayerNorm order (as in v2).
template <int BPL>
__device__ __forceinline__ float ln_block_sum(const float* part, int lane) {
  float w[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int j = 0; j < BPL; ++j) w[j] = warp_sum(part[j]);
  float v[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = (i < BPL) ? w[i] : 0.f;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float nv[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) nv[i] = v[i] + v[i ^ o];
    #pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = nv[i];
  }
  (void)lane;
  return v[0];
}

// Sum of squares in exactly the order of the 256-thread RMS kernels for D == 2048 (as in v2).
__device__ __forceinline__ float rms_ssq_exact2048(const float v[4][16], int lane) {
  float p[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float ssq = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) ssq += v[j][2 * i] * v[j][2 * i] + v[j][2 * i + 1] * v[j][2 * i + 1];
    p[i] = ssq;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) p[i] += __shfl_xor_sync(0xffffffffu, p[i], 2);
  #pragma unroll
  for (int i = 0; i < 8; ++i) p[i] += __shfl_xor_sync(0xffffffffu, p[i], 1);
  float q[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) q[i] = p[i] + p[i ^ 4];
  #pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = q[i] + q[i ^ 2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) q[i] = p[i] + p[i ^ 1];
  const float wsum = q[0];
  float w[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) w[k] = __shfl_sync(0xffffffffu, wsum, 4 * k);
  float sl[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) sl[i] = (i < 8) ? w[i] : 0.f;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float nv[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) nv[i] = sl[i] + sl[i ^ o];
    #pragma unroll
    for (int i = 0; i < 32; ++i) sl[i] = nv[i];
  }
  (void)lane;
  return sl[0];
}

__device__ __forceinline__ void load16(const __half* p, float* v) {
  const int4 a = reinterpret_cast<const int4*>(p)[0];
  const int4 b = reinterpret_cast<const int4*>(p)[1];
  const __half* ha = reinterpret_cast<const __half*>(&a);
  const __half* hb = reinterpret_cast<const __half*>(&b);
  #pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = __half2float(ha[i]); v[8 + i] = __half2float(hb[i]); }
}

// Per-column fp32 tables are stored lane-major for the warp's access pattern: block b = lane + 32*j,
// quarter q (4 floats) lives at float4 index (j*4 + q)*32 + lane, so each warp-wide 16-byte load
// covers 512 contiguous bytes (4 L1 lines) instead of 32 lanes striding 64 bytes apart (16 lines).
__device__ __forceinline__ void load16f_swz(const float* table, int b, float* v) {
  const float4* q4 = reinterpret_cast<const float4*>(table) + ((b >> 5) * 4) * 32 + (b & 31);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 a = q4[i * 32];
    v[4 * i] = a.x; v[4 * i + 1] = a.y; v[4 * i + 2] = a.z; v[4 * i + 3] = a.w;
  }
}

__device__ __forceinline__ float half_at(const uint4* r, int i) {
  return __half2float(reinterpret_cast<const __half*>(r)[i]);
}

// e4m3 store; the saturating cvt clamps finite input to +-448 (v2 clamped explicitly first).
__device__ __forceinline__ void store16_fp8(uint8_t* p, const float* v) {
  uint4 out;
  uint16_t* o2 = reinterpret_cast<uint16_t*>(&out);
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    o2[i] = __nv_cvt_float2_to_fp8x2(make_float2(v[2 * i], v[2 * i + 1]),
                                     __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint4*>(p) = out;
}

// SFA index of CUTLASS Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA((S,1,D,1)), closed form
// (verified against cute for the pipeline shapes):
//   idx(row, blk) = ((row/128) * (nb/4) + blk/4) * 512 + (row%32) * 16 + ((row/32)%4) * 4 + blk%4
__device__ __forceinline__ uint8_t* sf_row_ptr(uint8_t* sfa, int row, int nb) {
  return sfa + (static_cast<size_t>(row >> 7) * static_cast<size_t>(nb >> 2)) * 512
             + ((row & 31) << 4) + (((row >> 5) & 3) << 2);
}
__device__ __forceinline__ int sf_blk_off(int blk) { return ((blk >> 2) << 9) | (blk & 3); }

// vals must already carry the fp16 rounding of the reference path.
__device__ __forceinline__ void quant16_slim(const float* vals, uint2* __restrict__ packed_row,
                                             int blk, uint8_t* __restrict__ sf_row) {
  float amax = fabsf(vals[0]);
  #pragma unroll
  for (int i = 1; i < 16; ++i) amax = fmaxf(amax, fabsf(vals[i]));
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  const __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(desired);
  const float bs_dq = static_cast<float>(bs_q);
  sf_row[sf_blk_off(blk)] = *reinterpret_cast<const uint8_t*>(&bs_q);
  // Exact e2m1 midpoints round toward zero in the reference chain but ties-to-even in the
  // hardware cvt; two ulps off the reciprocal put every exact midpoint strictly below the tie.
  const float inv_bs = __int_as_float(__float_as_int(1.f / bs_dq) - 2);
  uint2 out;
  uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
  #pragma unroll
  for (int p = 0; p < 8; ++p)
    ob[p] = __nv_cvt_float2_to_fp4x2(make_float2(vals[2 * p] * inv_bs, vals[2 * p + 1] * inv_bs),
                                     __NV_E2M1, cudaRoundNearest);
  packed_row[blk] = out;
}

template <int MODE, int BPL>
__global__ void __launch_bounds__(THREADS)
rowops_v5_kernel(const __half* __restrict__ x, const float* __restrict__ inv_s,
                 const float* __restrict__ descale, uint2* __restrict__ packed,
                 uint8_t* __restrict__ sfa, uint8_t* __restrict__ out_fp8, int S, int D) {
  static_assert(MODE == kQuantFp4 || MODE == kRmsFp8 || MODE == kRmsMulFp4, "row modes only");
  flashrt_pdl_wait_and_trigger();
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
      load16(x + rowoff + b * 16, v[j]);
      if (MODE != kQuantFp4) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc += v[j][i] * v[j][i];
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 16; ++i) v[j][i] = 0.f;
    }
  }

  float scale = 1.f;
  if (MODE == kRmsFp8 || MODE == kRmsMulFp4) {
    float ssq;
    if (BPL == 4 && D == 2048) ssq = rms_ssq_exact2048(v, lane);   // bit-exact vs the originals
    else ssq = warp_sum(acc);
    scale = __frsqrt_rn(ssq / D + 1e-6f);
    if (MODE == kRmsFp8) scale /= fmaxf(*descale, 1e-12f);
  }

  uint2* packed_row = (MODE == kRmsFp8) ? nullptr : packed + static_cast<size_t>(row) * nb;
  uint8_t* sf_row = (MODE == kRmsFp8) ? nullptr : sf_row_ptr(sfa, row, nb);
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    const int b = lane + 32 * j;
    if (b >= nb) continue;
    if (MODE == kRmsMulFp4) {
      float vals[16];
      if (inv_s != nullptr) {
        float is[16];
        load16f_swz(inv_s, b, is);
        #pragma unroll
        for (int i = 0; i < 16; ++i) vals[i] = __half2float(__float2half(v[j][i] * scale * is[i]));
      } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) vals[i] = __half2float(__float2half(v[j][i] * scale));
      }
      quant16_slim(vals, packed_row, b, sf_row);
    } else if (MODE == kQuantFp4) {
      quant16_slim(v[j], packed_row, b, sf_row);
    } else {
      float vals[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) vals[i] = v[j][i] * scale;
      store16_fp8(out_fp8 + rowoff + b * 16, vals);
    }
  }
}

// LayerNorm rows with the row kept as packed halves (as rowops_ln_kernel in v2), fp32 tables.
template <int MODE, int BPL>
__global__ void __launch_bounds__(THREADS)
rowops_ln_v5_kernel(const __half* __restrict__ x, const float* __restrict__ gamma,
                    const float* __restrict__ beta, const float* __restrict__ inv_s,
                    uint2* __restrict__ packed, uint8_t* __restrict__ sfa,
                    uint8_t* __restrict__ out_fp8, int S, int D, float eps) {
  static_assert(MODE == kLnMulFp4 || MODE == kLnFp8, "LN modes only");
  flashrt_pdl_wait_and_trigger();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * ROWS_PER_CTA + warp;
  if (row >= S) return;
  const int nb = D >> 4;
  const size_t rowoff = static_cast<size_t>(row) * D;
  uint4 raw[BPL][2];
  float part[BPL];
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    const int b = lane + 32 * j;
    part[j] = 0.f;
    if (b < nb) {
      raw[j][0] = reinterpret_cast<const uint4*>(x + rowoff + b * 16)[0];
      raw[j][1] = reinterpret_cast<const uint4*>(x + rowoff + b * 16)[1];
    } else {
      raw[j][0] = make_uint4(0u, 0u, 0u, 0u); raw[j][1] = raw[j][0];
    }
  }
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    if (lane + 32 * j < nb) {
      float s16 = 0.f;
      #pragma unroll
      for (int i = 0; i < 16; ++i) s16 += half_at(raw[j], i);
      part[j] = s16;
    }
  }
  const float mean = ln_block_sum<BPL>(part, lane) / D;
  float vpart[BPL];
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    float var = 0.f;
    if (lane + 32 * j < nb) {
      #pragma unroll
      for (int i = 0; i < 16; ++i) { const float d = half_at(raw[j], i) - mean; var += d * d; }
    }
    vpart[j] = var;
  }
  const float rstd = rsqrtf(ln_block_sum<BPL>(vpart, lane) / D + eps);

  uint2* packed_row = (MODE == kLnFp8) ? nullptr : packed + static_cast<size_t>(row) * nb;
  uint8_t* sf_row = (MODE == kLnFp8) ? nullptr : sf_row_ptr(sfa, row, nb);
  #pragma unroll
  for (int j = 0; j < BPL; ++j) {
    const int b = lane + 32 * j;
    if (b >= nb) continue;
    float g[16], bt[16], vals[16];
    load16f_swz(gamma, b, g);
    load16f_swz(beta, b, bt);
    if (MODE == kLnMulFp4) {
      const bool has_is = (inv_s != nullptr);
      float is[16];
      if (has_is) load16f_swz(inv_s, b, is);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        float t = (half_at(raw[j], i) - mean) * rstd * g[i] + bt[i];
        if (has_is) t *= is[i];
        vals[i] = __half2float(__float2half(t));
      }
      quant16_slim(vals, packed_row, b, sf_row);
    } else {
      #pragma unroll
      for (int i = 0; i < 16; ++i) vals[i] = (half_at(raw[j], i) - mean) * rstd * g[i] + bt[i];
      store16_fp8(out_fp8 + rowoff + b * 16, vals);
    }
  }
}

inline bool misaligned(const void* p, int a) {
  return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1));
}

template <int MODE>
int launch_v5(const __half* x, const float* inv_s, const float* descale, void* packed, void* sfa,
              void* out_fp8, int S, int D, cudaStream_t stream) {
  if (D % 16 != 0 || D > 16 * 32 * MAX_BPL || S <= 0) return -1;
  if (misaligned(x, 16) || misaligned(inv_s, 16) || misaligned(packed, 8) || misaligned(out_fp8, 16)) return -1;
  const int nb = D / 16;
  const int bpl = (nb + 31) / 32;
  const dim3 grid((S + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
  auto* pk = reinterpret_cast<uint2*>(packed);
  auto* sf = reinterpret_cast<uint8_t*>(sfa);
  auto* o8 = reinterpret_cast<uint8_t*>(out_fp8);
  switch (bpl) {
    case 1: flash_rt::fp4::launch_maybe_pdl(rowops_v5_kernel<MODE, 1>, grid, dim3(THREADS), 0, stream, x, inv_s, descale, pk, sf, o8, S, D); break;
    case 2: flash_rt::fp4::launch_maybe_pdl(rowops_v5_kernel<MODE, 2>, grid, dim3(THREADS), 0, stream, x, inv_s, descale, pk, sf, o8, S, D); break;
    case 3: flash_rt::fp4::launch_maybe_pdl(rowops_v5_kernel<MODE, 3>, grid, dim3(THREADS), 0, stream, x, inv_s, descale, pk, sf, o8, S, D); break;
    case 4: flash_rt::fp4::launch_maybe_pdl(rowops_v5_kernel<MODE, 4>, grid, dim3(THREADS), 0, stream, x, inv_s, descale, pk, sf, o8, S, D); break;
    default: return -1;
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

template <int MODE>
int launch_ln_v5(const __half* x, const float* gamma, const float* beta, const float* inv_s,
                 void* packed, void* sfa, void* out_fp8, int S, int D, float eps, cudaStream_t stream) {
  if (D % 16 != 0 || D > 16 * 32 * MAX_BPL || S <= 0) return -1;
  if (misaligned(x, 16) || misaligned(gamma, 16) || misaligned(beta, 16) || misaligned(inv_s, 16) ||
      misaligned(packed, 8) || misaligned(out_fp8, 16)) return -1;
  if (gamma == nullptr || beta == nullptr) return -1;
  const int nb = D / 16;
  const int bpl = (nb + 31) / 32;
  const dim3 grid((S + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
  auto* pk = reinterpret_cast<uint2*>(packed);
  auto* sf = reinterpret_cast<uint8_t*>(sfa);
  auto* o8 = reinterpret_cast<uint8_t*>(out_fp8);
  switch (bpl) {
    case 1: flash_rt::fp4::launch_maybe_pdl(rowops_ln_v5_kernel<MODE, 1>, grid, dim3(THREADS), 0, stream, x, gamma, beta, inv_s, pk, sf, o8, S, D, eps); break;
    case 2: flash_rt::fp4::launch_maybe_pdl(rowops_ln_v5_kernel<MODE, 2>, grid, dim3(THREADS), 0, stream, x, gamma, beta, inv_s, pk, sf, o8, S, D, eps); break;
    case 3: flash_rt::fp4::launch_maybe_pdl(rowops_ln_v5_kernel<MODE, 3>, grid, dim3(THREADS), 0, stream, x, gamma, beta, inv_s, pk, sf, o8, S, D, eps); break;
    case 4: flash_rt::fp4::launch_maybe_pdl(rowops_ln_v5_kernel<MODE, 4>, grid, dim3(THREADS), 0, stream, x, gamma, beta, inv_s, pk, sf, o8, S, D, eps); break;
    default: return -1;
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace

int pi05_row_rms_fp8_swizzled(const __half* x, void* out_fp8, int S, int D, const float* descale,
                      cudaStream_t stream) {
  if (descale == nullptr) return -1;
  return launch_v5<kRmsFp8>(x, nullptr, descale, nullptr, nullptr, out_fp8, S, D, stream);
}
int pi05_row_quantize_fp4_sfa_swizzled(const __half* src, void* packed, void* sfa, int N, int D,
                               cudaStream_t stream) {
  return launch_v5<kQuantFp4>(src, nullptr, nullptr, packed, sfa, nullptr, N, D, stream);
}
int pi05_row_rms_mul_fp4_sfa_swizzled(const __half* x, const float* inv_s, void* packed, void* sfa,
                              int S, int D, cudaStream_t stream) {
  return launch_v5<kRmsMulFp4>(x, inv_s, nullptr, packed, sfa, nullptr, S, D, stream);
}
int pi05_row_layer_norm_fp8_swizzled(const __half* x, const float* gamma, const float* beta,
                             void* out_fp8, int S, int D, float eps, cudaStream_t stream) {
  return launch_ln_v5<kLnFp8>(x, gamma, beta, nullptr, nullptr, nullptr, out_fp8, S, D, eps, stream);
}
int pi05_row_layer_norm_mul_fp4_sfa_swizzled(const __half* x, const float* gamma, const float* beta,
                                     const float* inv_s, void* packed, void* sfa, int S, int D,
                                     float eps, cudaStream_t stream) {
  return launch_ln_v5<kLnMulFp4>(x, gamma, beta, inv_s, packed, sfa, nullptr, S, D, eps, stream);
}

}  // namespace fused_fp4
}  // namespace flash_rt
