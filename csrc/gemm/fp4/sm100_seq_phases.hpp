// ============================================================================
//  FlashRT — elementwise phases run inside the persistent GEMM sequence kernel
//  by its 128 epilogue threads (sm100_gemm_seq_persistent_kernel.hpp).
//
//  kPhaseGateResAdarms reproduces pi05_gate_res_adarms_fp4_sfa_register_kernel
//  (csrc/fused_fp4/norm_silu_fp4_sfa.cu) bit for bit: the 256-thread row
//  partition, the per-thread segment order, the warp xor trees and the
//  16-lane block quantize are the originals'; each epilogue thread stands in
//  for original threads t and t+128.
// ============================================================================
#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace cutlass::gemm::kernel {

struct SeqPhaseArgs {
  const __half* x = nullptr;          // GEMM output feeding the residual (fg)
  const __half* prev_gate = nullptr;  // (S, D)
  __half* residual = nullptr;         // (S, D), updated in place
  const __half* style = nullptr;      // (S, 3D): scale, shift, gate
  uint8_t* packed = nullptr;          // (S, D/2) e2m1 out
  uint8_t* sfa = nullptr;             // UE4M3 SFA (Sm1xx layout for (S, D)) out
  __half* gate = nullptr;             // (S, D) out
  int S = 0;
  int D = 0;
  // kind 3 (private AdaRMS after a mode-2 GEMM epilogue): residual already holds x_new,
  // partials holds 32 per-row sum-of-squares partials; the quantized rows go to this
  // cluster's slot of the next problem's activation operand.
  const float* partials = nullptr;    // (S, 32)
  uint8_t* slot_packed = nullptr;     // slots x (16 rows x D/2)
  uint8_t* slot_sfa = nullptr;        // slots x (128 x D/16)
  long slot_packed_pitch = 0;
  long slot_sfa_pitch = 0;
};

struct SeqPhaseSmem {
  float reduction[8];
  float rstd;
  float rstd_rows[32];
};

namespace seq_phase_detail {

__device__ __forceinline__ int sfa_index_rows_lt_32(int row, int kb, int D) {
  // Sm1xx SFA layout for an (S <= 32, D) operand: ((row>>7)*(D/64) + (kb>>2))*512 + (row&31)*16 + ((row>>5)&3)*4 + (kb&3)
  return ((row >> 7) * (D / 64) + (kb >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (kb & 3);
}

// pi05_quantize_register_value<true>: 16-lane block amax -> UE4M3 scale -> hardware e2m1x2 pack.
__device__ __forceinline__ void quantize_block_lane(float value, uint8_t* packed_row, uint8_t* dst_sfa,
                                                    int row, int block_idx, int lane_in_block, int D) {
  float amax = fabsf(value);
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 16));
  }
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
  const float inv_bs = 1.f / static_cast<float>(bs_q);
  if (lane_in_block == 0) {
    dst_sfa[sfa_index_rows_lt_32(row, block_idx, D)] = *reinterpret_cast<uint8_t*>(&bs_q);
  }
  const float next = __shfl_down_sync(0xffffffffu, value, 1, 16);
  if ((lane_in_block & 1) == 0) {
    packed_row[block_idx * 8 + lane_in_block / 2] = static_cast<uint8_t>(
        __nv_cvt_float2_to_fp4x2(make_float2(value * inv_bs, next * inv_bs), __NV_E2M1, cudaRoundNearest));
  }
}

}  // namespace seq_phase_detail

// Run by NumEpi (=128) threads; `sync()` is a barrier over exactly those threads.
// Rows are strided over the CTAs of the grid (row = cta, cta + num_ctas, ...).
// All of a row's inputs are loaded into registers before any store so the
// (cold, per-layer) style rows and the residual stream are fetched with one
// round of memory latency rather than a load/store chain.
template <class SyncFn>
__device__ __forceinline__ void seq_phase_gate_res_adarms(SeqPhaseArgs const& a, SeqPhaseSmem& sm, int e, SyncFn&& sync, int cta, int num_ctas) {
  const int D = a.D;
  const int lane = e & 31;
  const int warp = e >> 5;              // real warp 0..3 (original warps warp and warp+4)
  const __half* __restrict__ x_p = a.x;
  const __half* __restrict__ pg_p = a.prev_gate;
  __half* __restrict__ res_p = a.residual;
  const __half* __restrict__ style_p = a.style;
  __half* __restrict__ gate_p = a.gate;
  uint8_t* __restrict__ sfa_p = a.sfa;
  for (int row = cta; row < a.S; row += num_ctas) {
    const __half* __restrict__ sc = style_p + row * 3 * D;
    const __half* __restrict__ sh = sc + D;
    const __half* __restrict__ gt = sh + D;
    uint8_t* __restrict__ packed_row = a.packed + row * (D / 2);
    float r_res[2][4], r_x[2][4], r_pg[2][4], r_sc[2][4], r_sh[2][4];
    __half r_gt[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = e + 128 * h;
      #pragma unroll
      for (int segment = 0; segment < 4; ++segment) {
        const int i = t + segment * 256;
        const int elem = row * D + i;
        r_res[h][segment] = __half2float(res_p[elem]);
        r_x[h][segment] = __half2float(x_p[elem]);
        r_pg[h][segment] = __half2float(pg_p[elem]);
        r_sc[h][segment] = __half2float(sc[i]);
        r_sh[h][segment] = __half2float(sh[i]);
        r_gt[h][segment] = gt[i];
      }
    }
    float values[2][4];
    float sum_sq[2] = {0.f, 0.f};
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = e + 128 * h;
      #pragma unroll
      for (int segment = 0; segment < 4; ++segment) {
        const int i = t + segment * 256;
        const int elem = row * D + i;
        const float value = r_res[h][segment] + r_x[h][segment] * r_pg[h][segment];
        const __half rounded = __float2half(value);
        res_p[elem] = rounded;
        values[h][segment] = __half2float(rounded);
        sum_sq[h] += value * value;
      }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq[h] += __shfl_xor_sync(0xffffffffu, sum_sq[h], offset);
      }
    }
    if (lane == 0) { sm.reduction[warp] = sum_sq[0]; sm.reduction[warp + 4] = sum_sq[1]; }
    sync();
    if (warp == 0) {
      float s = lane < 8 ? sm.reduction[lane] : 0.f;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, offset);
      }
      if (lane == 0) sm.rstd = rsqrtf(s / D + 1e-6f);
    }
    sync();
    const float rstd = sm.rstd;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = e + 128 * h;
      const int lane_in_block = t & 15;
      const int block_group = t >> 4;
      #pragma unroll
      for (int segment = 0; segment < 4; ++segment) {
        const int i = t + segment * 256;
        const int elem = row * D + i;
        const float normed = values[h][segment] * rstd * (1.f + r_sc[h][segment]) + r_sh[h][segment];
        const __half rounded = __float2half(normed);
        gate_p[elem] = r_gt[h][segment];
        seq_phase_detail::quantize_block_lane(__half2float(rounded), packed_row, sfa_p, row,
                                              segment * 16 + block_group, lane_in_block, D);
      }
    }
    sync();   // reduction[] / rstd reuse on the next row
  }
}

// gate_res_fp16_kernel: residual[i] = half(residual[i] + x[i] * prev_gate[i]) over S*D elements.
__device__ __forceinline__ void seq_phase_gate_res(SeqPhaseArgs const& a, int e, int cta, int num_ctas) {
  const int n = a.S * a.D;
  for (int i = cta * 128 + e; i < n; i += num_ctas * 128) {
    const float v = __half2float(a.residual[i]) + __half2float(a.x[i]) * __half2float(a.prev_gate[i]);
    a.residual[i] = __float2half(v);
  }
}


// Private AdaRMS: every CTA quantizes all rows into its cluster's slot. rstd comes from the
// epilogue's partials (a different fp32 summation order than the row kernel, same value up
// to rounding); the elementwise part and the block quantize are the row kernel's.
template <class SyncFn>
__device__ __forceinline__ void seq_phase_private_adarms(SeqPhaseArgs const& a, SeqPhaseSmem& sm, int e, SyncFn&& sync, int slot) {
  const int D = a.D;
  if (e < a.S) {
    float ssq = 0.f;
    const float* __restrict__ pr = a.partials + e * 32;
    #pragma unroll
    for (int k = 0; k < 32; ++k) ssq += pr[k];
    sm.rstd_rows[e] = rsqrtf(ssq / D + 1e-6f);
  }
  sync();
  uint8_t* __restrict__ packed = a.slot_packed + slot * a.slot_packed_pitch;
  uint8_t* __restrict__ sfa = a.slot_sfa + slot * a.slot_sfa_pitch;
  const __half* __restrict__ xr = a.residual;
  const __half* __restrict__ style = a.style;
  const int nblk = a.S * (D / 16);
  for (int blk = e; blk < nblk; blk += 128) {
    const int row = blk / (D / 16), b = blk % (D / 16);
    const float rstd = sm.rstd_rows[row];
    const __half* xb = xr + row * D + b * 16;
    const __half* sc = style + row * 3 * D + b * 16;
    const __half* sh = sc + D;
    uint4 vx[2], vsc[2], vsh[2];
    vx[0] = *reinterpret_cast<const uint4*>(xb); vx[1] = *reinterpret_cast<const uint4*>(xb + 8);
    vsc[0] = *reinterpret_cast<const uint4*>(sc); vsc[1] = *reinterpret_cast<const uint4*>(sc + 8);
    vsh[0] = *reinterpret_cast<const uint4*>(sh); vsh[1] = *reinterpret_cast<const uint4*>(sh + 8);
    const __half* hx = reinterpret_cast<const __half*>(vx);
    const __half* hsc = reinterpret_cast<const __half*>(vsc);
    const __half* hsh = reinterpret_cast<const __half*>(vsh);
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float normed = __half2float(hx[i]) * rstd * (1.f + __half2float(hsc[i])) + __half2float(hsh[i]);
      vals[i] = __half2float(__float2half(normed));
      amax = fmaxf(amax, fabsf(vals[i]));
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float inv_bs = 1.f / static_cast<float>(bs_q);
    sfa[seq_phase_detail::sfa_index_rows_lt_32(row, b, D)] = *reinterpret_cast<uint8_t*>(&bs_q);
    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
      ob[q] = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(make_float2(vals[2 * q] * inv_bs, vals[2 * q + 1] * inv_bs),
                                                            __NV_E2M1, cudaRoundNearest));
    }
    *reinterpret_cast<uint2*>(packed + row * (D / 2) + b * 8) = out;
  }
  sync();
}

}  // namespace cutlass::gemm::kernel
