// Phase CTAs for the forked SM100 GEMM kernel (Thor decoder): the grid gets
// extra clusters that skip the GEMM entirely, wait until every GEMM CTA has
// landed its epilogue stores, then run an element-wise phase (the gated
// residual + AdaRMS + NVFP4 quantize that feeds the next GEMM) in parallel,
// one query row per CTA. The consumer GEMM is launched at this kernel's PDL
// trigger, so its weight stream runs while the phase computes instead of
// waiting for a separate small kernel (PDL only overlaps one generation).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "cutlass/arch/barrier.h"
#include "gemm/fp4/sm100_seq_phases.hpp"

namespace flash_rt {
namespace fp4 {

using cutlass::gemm::kernel::SeqPhaseArgs;
using cutlass::gemm::kernel::SeqPhaseSmem;

struct PhaseCtaParams {
  int kind = 0;                  // 0: none; 1: gate_res_adarms; 2: gate_res (last layer)
  int num_gemm_ctas = 0;         // arrivals the phase waits for
  int num_phase_ctas = 0;        // extra CTAs appended to the grid (multiple of the cluster size)
  int cluster_size = 1;
  unsigned* counter = nullptr;   // [0] GEMM arrivals, [1] phase CTAs done; both reset by the last phase CTA
  int dbg = 0;                   // probes: 1 phase CTAs skip the arrival wait, 2 GEMM CTAs skip the arrival, 4 skip the phase compute
  SeqPhaseArgs args{};
};

__device__ __forceinline__ unsigned phase_ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Called by every epilogue thread of a GEMM CTA after store_tail(): completes
// this thread's bulk (TMA) stores in full, orders them against generic-proxy
// readers in other CTAs, then one thread announces the arrival.
__device__ __forceinline__ void phase_gemm_arrive(PhaseCtaParams const& p, bool first_epilogue_thread, uint32_t epilogue_threads, int barrier_id) {
  if (p.dbg & 2) return;
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence();
  cutlass::arch::NamedBarrier::sync(epilogue_threads, barrier_id);
  if (first_epilogue_thread) atomicAdd(p.counter, 1u);
}

// Row-prefetch form of seq_phase_gate_res_adarms for the in-GEMM phase: the
// residual, previous gate and style rows do not depend on the GEMM, so they
// are loaded before the arrival barrier; only fg (the GEMM output) is read
// after it. Same arithmetic and reduction order as the sequence-kernel port.
template <int MAX_ROWS>
struct PhaseRowPrefetch {
  float r_res[MAX_ROWS][2][4], r_pg[MAX_ROWS][2][4], r_sc[MAX_ROWS][2][4], r_sh[MAX_ROWS][2][4];
  __half r_gt[MAX_ROWS][2][4];
  int rows[MAX_ROWS];
  int nrows = 0;
  __device__ __forceinline__ void load(SeqPhaseArgs const& a, int e, int cta, int num_ctas) {
    const int D = a.D;
    nrows = 0;
    #pragma unroll
    for (int k = 0; k < MAX_ROWS; ++k) {
      const int row = cta + k * num_ctas;
      rows[k] = row;
      if (row < a.S) {
        nrows = k + 1;
        const __half* sc = a.style + row * 3 * D;
        const __half* sh = sc + D;
        const __half* gt = sh + D;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int t = e + 128 * h;
          #pragma unroll
          for (int segment = 0; segment < 4; ++segment) {
            const int i = t + segment * 256;
            const int elem = row * D + i;
            r_res[k][h][segment] = __half2float(a.residual[elem]);
            r_pg[k][h][segment] = __half2float(a.prev_gate[elem]);
            r_sc[k][h][segment] = __half2float(sc[i]);
            r_sh[k][h][segment] = __half2float(sh[i]);
            r_gt[k][h][segment] = gt[i];
          }
        }
      }
    }
  }
  template <class SyncFn>
  __device__ __forceinline__ void run(SeqPhaseArgs const& a, SeqPhaseSmem& sm, int e, SyncFn&& sync) {
    const int D = a.D;
    const int lane = e & 31;
    const int warp = e >> 5;
    #pragma unroll
    for (int k = 0; k < MAX_ROWS; ++k) {
      if (k >= nrows) break;
      const int row = rows[k];
      uint8_t* packed_row = a.packed + row * (D / 2);
      float r_x[2][4];
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = e + 128 * h;
        #pragma unroll
        for (int segment = 0; segment < 4; ++segment) r_x[h][segment] = __half2float(a.x[row * D + t + segment * 256]);
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
          const float value = r_res[k][h][segment] + r_x[h][segment] * r_pg[k][h][segment];
          const __half rounded = __float2half(value);
          a.residual[elem] = rounded;
          values[h][segment] = __half2float(rounded);
          sum_sq[h] += value * value;
        }
      }
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq[h] += __shfl_xor_sync(0xffffffffu, sum_sq[h], offset);
      }
      if (lane == 0) { sm.reduction[warp] = sum_sq[0]; sm.reduction[warp + 4] = sum_sq[1]; }
      sync();
      if (warp == 0) {
        float sacc = lane < 8 ? sm.reduction[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sacc += __shfl_xor_sync(0xffffffffu, sacc, offset);
        if (lane == 0) sm.rstd = rsqrtf(sacc / D + 1e-6f);
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
          const float normed = values[h][segment] * rstd * (1.f + r_sc[k][h][segment]) + r_sh[k][h][segment];
          const __half rounded = __float2half(normed);
          a.gate[elem] = r_gt[k][h][segment];
          cutlass::gemm::kernel::seq_phase_detail::quantize_block_lane(__half2float(rounded), packed_row, a.sfa, row,
                                                                      segment * 16 + block_group, lane_in_block, D);
        }
      }
      sync();
    }
  }
};

// In-GEMM form (num_phase_ctas == 0): every GEMM CTA's epilogue warps complete
// and publish their stores, meet the other GEMM CTAs at a grid-wide arrival
// count, then compute the phase for rows cta, cta + num_gemm_ctas, ... . No
// extra CTAs, so the dependent GEMM (launched at this kernel's trigger) keeps
// every SM this kernel does not use.
__device__ __forceinline__ void phase_in_gemm(PhaseCtaParams const& p, int cta, int e, uint32_t epilogue_threads, char* smem) {
  if (p.dbg & 2) return;
  PhaseRowPrefetch<2> pre;
  const bool prefetch = (p.kind == 1) && !(p.dbg & 4) && e < 128 && (p.args.S <= 2 * p.num_gemm_ctas);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");   // the rows below come from earlier kernels
#endif
  if (prefetch) pre.load(p.args, e, cta, p.num_gemm_ctas);
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence();
  cutlass::arch::NamedBarrier::sync(epilogue_threads, 1);
  if (e == 0) {
    atomicAdd(p.counter, 1u);
    if (!(p.dbg & 1)) {
      while (phase_ld_acquire(p.counter) < static_cast<unsigned>(p.num_gemm_ctas)) __nanosleep(64);
    }
  }
  cutlass::arch::NamedBarrier::sync(epilogue_threads, 1);
  __threadfence();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (!(p.dbg & 4) && e < 128) {
    SeqPhaseSmem& sm = *reinterpret_cast<SeqPhaseSmem*>(smem);
    auto sync = []() { cutlass::arch::NamedBarrier::sync(128, 0); };
    if (prefetch) pre.run(p.args, sm, e, sync);
    else if (p.kind == 1) cutlass::gemm::kernel::seq_phase_gate_res_adarms(p.args, sm, e, sync, cta, p.num_gemm_ctas);
    else if (p.kind == 2) cutlass::gemm::kernel::seq_phase_gate_res(p.args, e, cta, p.num_gemm_ctas);
  }
  cutlass::arch::NamedBarrier::sync(epilogue_threads, 1);
  if (e == 0) {
    __threadfence();
    if (atomicAdd(p.counter + 1, 1u) == static_cast<unsigned>(p.num_gemm_ctas - 1)) {
      p.counter[0] = 0u;
      p.counter[1] = 0u;
      __threadfence();
    }
  }
}

// Body of a phase CTA (all threads of the CTA enter; only the first 128 compute).
__device__ __forceinline__ void phase_cta_run(PhaseCtaParams const& p, int phase_cta_index, char* smem_buf) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Trigger first: the dependent GEMM may launch and stream its weights while
  // this phase runs (its own griddepcontrol.wait still covers the phase output).
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int tid = threadIdx.x;
  if (tid == 0 && !(p.dbg & 1)) {
    while (phase_ld_acquire(p.counter) < static_cast<unsigned>(p.num_gemm_ctas)) __nanosleep(128);
  }
  __syncthreads();
  __threadfence();
  if (tid < 128 && !(p.dbg & 4)) {
    SeqPhaseSmem& sm = *reinterpret_cast<SeqPhaseSmem*>(smem_buf);
    auto sync = []() { cutlass::arch::NamedBarrier::sync(128, 0); };   // user barrier 0 (hardware id 8)
    if (p.kind == 1) cutlass::gemm::kernel::seq_phase_gate_res_adarms(p.args, sm, tid, sync, phase_cta_index, p.num_phase_ctas);
    else if (p.kind == 2) cutlass::gemm::kernel::seq_phase_gate_res(p.args, tid, phase_cta_index, p.num_phase_ctas);
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.counter + 1, 1u) == static_cast<unsigned>(p.num_phase_ctas - 1)) {
      p.counter[0] = 0u;
      p.counter[1] = 0u;
      __threadfence();
    }
  }
}

}  // namespace fp4
}  // namespace flash_rt
