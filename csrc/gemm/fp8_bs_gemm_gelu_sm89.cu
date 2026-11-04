// SPDX-License-Identifier: Apache-2.0
//
// FFN-up GEMM with the bias + gelu-tanh + FP8 block-128 quantiser fused into
// the epilogue: fp8_bs_gemm_cutlass_f16blk_k64_sm89 (64x128x64 tile, FP16
// block sums folded every two K iterations) whose accumulators, instead of
// being stored as bf16, go through the arithmetic of
// gelu_tanh_bias_to_fp8_block128_warp_bf16 on the bf16-rounded GEMM output:
//
//   x = bf16(acc), b = bf16(x + bias), g = bf16(gelu_tanh(b)),
//   s = max(amax_128(|g|) / 448, 1e-12), q = e4m3(clamp(g / s, +-448))
//
// The CTA's 128 output columns are exactly one quantisation block, so the
// per-row amax is a quad shuffle plus a four-warp exchange through shared
// memory. This file is compiled with the flags of the standalone quantiser's
// library (flush-to-zero, approximate division) so that the gelu and the
// division round the same way; the bf16 GEMM output, every fp8 byte and every
// scale are bit-identical to the GEMM followed by the standalone kernel.
//
// The same mainloop also serves the attention / cross-attention output
// projections with the gated residual update in the epilogue (see
// res_gemm_kernel).
#include "fp8_bs_gemm_cutlass_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"

#include "bs_mma_multistage.h"

namespace flash_rt {
namespace gemm {
namespace bs_cutlass_sm89 {

namespace {

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr float kFp8Max = 448.0f;

// Same expression as the standalone quantiser (fp8_per_token_block_quant.cu).
__device__ __forceinline__ float gelu_tanh_f32(float x) {
  return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

template <class TbShape, class WarpShape, int Stages, int KIterPerFold>
struct GeluQGemm {
  static_assert(TbShape::kN == 128, "one quantisation block per CTA column tile");
  static_assert(TbShape::kK * KIterPerFold == 128, "one fold per 128-K scale block");
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 8, ElementAcc, float>;
  using Stock = cutlass::gemm::kernel::DefaultGemm<
      ElementA, LayoutA, 16, ElementB, LayoutB, 16, ElementC, LayoutC, ElementAcc,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      TbShape, WarpShape, InstructionShape, EpilogueOp, Swizzle, 3, false,
      cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone,
      false, false, false, cutlass::layout::NoPermute>;
  using StockMma = typename Stock::Mma;
  using Mma = cutlass::gemm::threadblock::BlockScaleMmaMultistage<
      typename StockMma::Shape,
      typename StockMma::IteratorA, typename StockMma::SmemIteratorA, StockMma::kCacheOpA,
      typename StockMma::IteratorB, typename StockMma::SmemIteratorB, StockMma::kCacheOpB,
      ElementC, LayoutC, typename StockMma::Policy, Stages, true, KIterPerFold>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreads = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    const __nv_bfloat16* bias;
    __nv_fp8_e4m3* out;
    float* out_scale;
    const float* act_scale;
    const float* w_scale;
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    float part[TbShape::kM][WarpCount::kN];
  };
};

template <class G>
__global__ void __launch_bounds__(G::kThreads)
gelu_q_gemm_kernel(typename G::Params p) {
  using Mma = typename G::Mma;
  using WarpCount = typename G::WarpCount;
  using Shape = typename Mma::Shape;
  extern __shared__ int smem_raw[];
  auto& ss = *reinterpret_cast<typename G::SharedStorage*>(smem_raw);

  const int M = p.problem.m(), N = p.problem.n(), K = p.problem.k();
  const cutlass::gemm::GemmCoord tile(blockIdx.x, blockIdx.y, 0);
  const int m0 = tile.m() * Shape::kM, n0 = tile.n() * Shape::kN;
  const int tid = threadIdx.x;
  const int warp = __shfl_sync(0xffffffffu, tid / 32, 0);
  const int lane = tid % 32;

  typename Mma::IteratorA iterator_A(p.params_A, p.ref_A.data(), {M, K}, tid, {m0, 0});
  typename Mma::IteratorB iterator_B(p.params_B, p.ref_B.data(), {K, N}, tid, {0, n0});
  Mma mma(ss.main_loop, tid, warp, lane);
  mma.set_block_scales(tile, p.act_scale, p.w_scale, K >> 7);
  typename Mma::FragmentC acc;
  acc.clear();
  const int k_iters = (K + Shape::kK - 1) / Shape::kK;
  mma(k_iters, acc, iterator_A, iterator_B, acc);
  __syncthreads();   // the epilogue reuses the mainloop's shared memory

  constexpr int kRowIters = Mma::Operator::MmaIterations::kRow;
  constexpr int kColIters = Mma::Operator::MmaIterations::kColumn;
  const int warp_mn = warp % (WarpCount::kM * WarpCount::kN);
  const int warp_m = warp_mn % WarpCount::kM;
  const int warp_n = warp_mn / WarpCount::kM;
  const int rt = warp_m * Mma::Operator::Shape::kM + (lane >> 2);       // row in tile
  const int ct = warp_n * Mma::Operator::Shape::kN + 2 * (lane & 3);    // column in tile

  float v[kRowIters * kColIters * 4];
  float pmax[kRowIters][2];
#pragma unroll
  for (int m = 0; m < kRowIters; ++m) pmax[m][0] = pmax[m][1] = 0.0f;
#pragma unroll
  for (int s = 0; s < kRowIters * kColIters; ++s) {
    const int m = s % kRowIters, n = s / kRowIters;
    const int c = n0 + ct + n * 8;
    const float b0 = __bfloat162float(p.bias[c]), b1 = __bfloat162float(p.bias[c + 1]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x = __bfloat162float(__float2bfloat16_rn(acc[s * 4 + e]));
      const float b = __bfloat162float(__float2bfloat16(x + ((e & 1) ? b1 : b0)));
      const float g = __bfloat162float(__float2bfloat16(gelu_tanh_f32(b)));
      v[s * 4 + e] = g;
      pmax[m][e >> 1] = fmaxf(pmax[m][e >> 1], fabsf(g));
    }
  }
#pragma unroll
  for (int m = 0; m < kRowIters; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float a = pmax[m][h];
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
      if ((lane & 3) == 0) ss.part[rt + m * 16 + h * 8][warp_n] = a;
    }
  __syncthreads();
  const int nkb = N >> 7;
#pragma unroll
  for (int m = 0; m < kRowIters; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = rt + m * 16 + h * 8;
      float amax = 0.0f;
#pragma unroll
      for (int w = 0; w < WarpCount::kN; ++w) amax = fmaxf(amax, ss.part[r][w]);
      const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
      const int row = m0 + r;
      if (row < M) {
#pragma unroll
        for (int n = 0; n < kColIters; ++n) {
          const int s = n * kRowIters + m;
          float q0 = v[s * 4 + h * 2] / sc, q1 = v[s * 4 + h * 2 + 1] / sc;
          q0 = fminf(fmaxf(q0, -kFp8Max), kFp8Max);
          q1 = fminf(fmaxf(q1, -kFp8Max), kFp8Max);
          const __nv_fp8_e4m3 f0(q0), f1(q1);
          const uint16_t pair = (uint16_t)f0.__x | ((uint16_t)f1.__x << 8);
          *reinterpret_cast<uint16_t*>(p.out + (size_t)row * N + n0 + ct + n * 8) = pair;
        }
        if (warp_n == 0 && (lane & 3) == 0) p.out_scale[(size_t)row * nkb + (n0 >> 7)] = sc;
      }
    }
}

// Residual epilogue: the bf16-rounded GEMM output y updates the residual
// stream in place with the arithmetic of wan_gated_residual_sstp_bias_bf16
// (gate = ss_row + tp, sstp) or wan_gated_residual_bias_bf16 (no gate):
//   y' = bf16(y + bias);  h = bf16(h + y' * gate)  /  h = bf16(h + y')
// Each output element belongs to exactly one thread, so the in-place update
// has no race.
template <class G>
__global__ void __launch_bounds__(G::kThreads)
res_gemm_kernel(typename G::Params p, __nv_bfloat16* __restrict__ residual,
                const float* __restrict__ ss_row, const float* __restrict__ tp,
                int tp_bcast) {
  using Mma = typename G::Mma;
  using WarpCount = typename G::WarpCount;
  using Shape = typename Mma::Shape;
  extern __shared__ int smem_raw[];
  auto& ss = *reinterpret_cast<typename G::SharedStorage*>(smem_raw);

  const int M = p.problem.m(), N = p.problem.n(), K = p.problem.k();
  const cutlass::gemm::GemmCoord tile(blockIdx.x, blockIdx.y, 0);
  const int m0 = tile.m() * Shape::kM, n0 = tile.n() * Shape::kN;
  const int tid = threadIdx.x;
  const int warp = __shfl_sync(0xffffffffu, tid / 32, 0);
  const int lane = tid % 32;

  typename Mma::IteratorA iterator_A(p.params_A, p.ref_A.data(), {M, K}, tid, {m0, 0});
  typename Mma::IteratorB iterator_B(p.params_B, p.ref_B.data(), {K, N}, tid, {0, n0});
  Mma mma(ss.main_loop, tid, warp, lane);
  mma.set_block_scales(tile, p.act_scale, p.w_scale, K >> 7);
  typename Mma::FragmentC acc;
  acc.clear();
  const int k_iters = (K + Shape::kK - 1) / Shape::kK;
  mma(k_iters, acc, iterator_A, iterator_B, acc);

  constexpr int kRowIters = Mma::Operator::MmaIterations::kRow;
  constexpr int kColIters = Mma::Operator::MmaIterations::kColumn;
  const int warp_mn = warp % (WarpCount::kM * WarpCount::kN);
  const int warp_m = warp_mn % WarpCount::kM;
  const int warp_n = warp_mn / WarpCount::kM;
  const int rt = warp_m * Mma::Operator::Shape::kM + (lane >> 2);
  const int ct = warp_n * Mma::Operator::Shape::kN + 2 * (lane & 3);
  const __nv_bfloat16* __restrict__ bias = p.bias;
  constexpr int kAtoms = kRowIters * kColIters;
  // every load first (the residual rows, bias, gate), then every store: the
  // in-place residual write would otherwise order each load behind the
  // previous element's store
  __nv_bfloat162 rin[kAtoms][2];
  float2 bv[kColIters], sv[kColIters];
  float2 tv[kAtoms][2];
#pragma unroll
  for (int n = 0; n < kColIters; ++n) {
    const int c = n0 + ct + n * 8;
    bv[n] = bias != nullptr ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + c))
                            : make_float2(0.f, 0.f);
    sv[n] = ss_row != nullptr ? *reinterpret_cast<const float2*>(ss_row + c) : make_float2(0.f, 0.f);
  }
#pragma unroll
  for (int s = 0; s < kAtoms; ++s) {
    const int m = s % kRowIters, n = s / kRowIters;
    const int c = n0 + ct + n * 8;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = min(m0 + rt + m * 16 + h * 8, M - 1);
      const size_t i = (size_t)row * N + c;
      rin[s][h] = *reinterpret_cast<const __nv_bfloat162*>(residual + i);
      tv[s][h] = ss_row != nullptr
                     ? *reinterpret_cast<const float2*>(
                           tp + (tp_bcast == 1 ? (size_t)c
                                 : tp_bcast > 1 ? (size_t)(row - row % tp_bcast) * N + c : i))
                     : make_float2(0.f, 0.f);
    }
  }
#pragma unroll
  for (int s = 0; s < kAtoms; ++s) {
    const int m = s % kRowIters, n = s / kRowIters;
    const int c = n0 + ct + n * 8;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + rt + m * 16 + h * 8;
      if (row >= M) continue;
      const size_t i = (size_t)row * N + c;
      const float bb[2] = {bv[n].x, bv[n].y}, ssv[2] = {sv[n].x, sv[n].y};
      const float tt[2] = {tv[s][h].x, tv[s][h].y};
      float y[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float xb = __bfloat162float(__float2bfloat16_rn(acc[s * 4 + h * 2 + e]));
        if (bias != nullptr)
          xb = __bfloat162float(__float2bfloat16_rn(__fadd_rn(xb, bb[e])));
        if (ss_row != nullptr) xb = __fmul_rn(xb, __fadd_rn(ssv[e], tt[e]));
        y[e] = xb;
      }
      const float2 rf = __bfloat1622float2(rin[s][h]);
      const float h0 = (ss_row != nullptr) ? __fadd_rn(rf.x, y[0]) : rf.x + y[0];
      const float h1 = (ss_row != nullptr) ? __fadd_rn(rf.y, y[1]) : rf.y + y[1];
      __nv_bfloat162 o;
      o.x = __float2bfloat16_rn(h0);
      o.y = __float2bfloat16_rn(h1);
      *reinterpret_cast<__nv_bfloat162*>(residual + i) = o;
    }
  }
}

template <class G>
int run_res(const void* A, const void* B, int M, int N, int K, const float* act_scale,
            const float* w_scale, void* residual, const void* bias, const float* ss_row,
            const float* tp, int tp_bcast, cudaStream_t stream) {
  using Mma = typename G::Mma;
  typename G::Params p;
  p.problem = cutlass::gemm::GemmCoord(M, N, K);
  typename Mma::IteratorA::TensorRef ref_A(
      reinterpret_cast<ElementA*>(const_cast<void*>(A)), K);
  typename Mma::IteratorB::TensorRef ref_B(
      reinterpret_cast<ElementB*>(const_cast<void*>(B)), K);
  p.ref_A = ref_A;
  p.ref_B = ref_B;
  p.params_A = typename Mma::IteratorA::Params(ref_A.layout());
  p.params_B = typename Mma::IteratorB::Params(ref_B.layout());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias);
  p.out = nullptr;
  p.out_scale = nullptr;
  p.act_scale = act_scale;
  p.w_scale = w_scale;
  const int smem = int(sizeof(typename G::SharedStorage));
  auto kfn = res_gemm_kernel<G>;
  static bool attr = false;
  if (!attr) {
    if (cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
        cudaSuccess)
      return -5;
    attr = true;
  }
  dim3 grid((M + Mma::Shape::kM - 1) / Mma::Shape::kM, N / Mma::Shape::kN);
  kfn<<<grid, G::kThreads, smem, stream>>>(
      p, reinterpret_cast<__nv_bfloat16*>(residual), ss_row, tp, tp_bcast);
  return (int)cudaGetLastError();
}

template <class G>
int run_gelu_q(const void* A, const void* B, const void* bias, void* out, float* out_scale,
               int M, int N, int K, const float* act_scale, const float* w_scale,
               cudaStream_t stream) {
  using Mma = typename G::Mma;
  typename G::Params p;
  p.problem = cutlass::gemm::GemmCoord(M, N, K);
  typename Mma::IteratorA::TensorRef ref_A(
      reinterpret_cast<ElementA*>(const_cast<void*>(A)), K);
  typename Mma::IteratorB::TensorRef ref_B(
      reinterpret_cast<ElementB*>(const_cast<void*>(B)), K);
  p.ref_A = ref_A;
  p.ref_B = ref_B;
  p.params_A = typename Mma::IteratorA::Params(ref_A.layout());
  p.params_B = typename Mma::IteratorB::Params(ref_B.layout());
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias);
  p.out = reinterpret_cast<__nv_fp8_e4m3*>(out);
  p.out_scale = out_scale;
  p.act_scale = act_scale;
  p.w_scale = w_scale;
  const int smem = int(sizeof(typename G::SharedStorage));
  auto kfn = gelu_q_gemm_kernel<G>;
  static bool attr = false;
  if (!attr) {
    if (cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
        cudaSuccess)
      return -5;
    attr = true;
  }
  dim3 grid((M + Mma::Shape::kM - 1) / Mma::Shape::kM, N / Mma::Shape::kN);
  kfn<<<grid, G::kThreads, smem, stream>>>(p);
  return (int)cudaGetLastError();
}

using CfgGeluK64 = GeluQGemm<cutlass::gemm::GemmShape<64, 128, 64>,
                             cutlass::gemm::GemmShape<32, 32, 64>, 4, 2>;

}  // namespace

int fp8_bs_gemm_cutlass_f16blk_k64_residual_sm89(const void* A, const void* B, int M, int N,
                                                 int K, const float* act_scale,
                                                 const float* w_scale, void* residual,
                                                 const void* bias, const float* ss_row,
                                                 const float* tp, int tp_bcast,
                                                 cudaStream_t stream) {
  if ((K & 127) || (N & 127) || (M & 15)) return -2;
  if (!residual || !act_scale || !w_scale || ((ss_row == nullptr) != (tp == nullptr)))
    return -1;
  return run_res<CfgGeluK64>(A, B, M, N, K, act_scale, w_scale, residual, bias, ss_row, tp,
                             tp_bcast, stream);
}

int fp8_bs_gemm_cutlass_f16blk_k64_gelu_fp8_sm89(const void* A, const void* B,
                                                 const void* bias, void* out,
                                                 float* out_scale, int M, int N, int K,
                                                 const float* act_scale,
                                                 const float* w_scale,
                                                 cudaStream_t stream) {
  if ((K & 127) || (N & 127) || (M & 15)) return -2;
  if (!bias || !out || !out_scale || !act_scale || !w_scale) return -1;
  return run_gelu_q<CfgGeluK64>(A, B, bias, out, out_scale, M, N, K, act_scale, w_scale,
                                stream);
}

}  // namespace bs_cutlass_sm89
}  // namespace gemm
}  // namespace flash_rt
