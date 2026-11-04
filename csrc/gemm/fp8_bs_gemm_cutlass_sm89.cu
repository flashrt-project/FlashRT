// SPDX-License-Identifier: Apache-2.0
//
// See the header. The stock CUTLASS 2.x types are used for everything except
// the mainloop, which is the in-tree copy in bs_mma_multistage.h with the
// per-k-block scale fold.
#include "fp8_bs_gemm_cutlass_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"

#include "bs_gemm_kernel.h"
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

template <class TbShape, class WarpShape, int Stages, int EpilogueCount,
          bool F16Block = false, int KIterPerFold = 1>
struct BSGemm {
  static_assert(TbShape::kK * KIterPerFold == 128, "one fold per 128-K scale block");
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, EpilogueCount, ElementAcc, float>;

  // Stock CUTLASS instantiation: its iterators, policy, epilogue and SMEM
  // layout are reused unchanged; only the mainloop class is replaced.
  using Stock = cutlass::gemm::kernel::DefaultGemm<
      ElementA, LayoutA, 16, ElementB, LayoutB, 16, ElementC, LayoutC, ElementAcc,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      TbShape, WarpShape, InstructionShape, EpilogueOp, Swizzle, 3, false,
      cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone,
      false, false, false, cutlass::layout::NoPermute>;

  using StockMma = typename Stock::Mma;

  using Mma = cutlass::gemm::threadblock::BlockScaleMmaMultistage<
      typename StockMma::Shape,
      typename StockMma::IteratorA, typename StockMma::SmemIteratorA,
      StockMma::kCacheOpA,
      typename StockMma::IteratorB, typename StockMma::SmemIteratorB,
      StockMma::kCacheOpB,
      ElementC, LayoutC, typename StockMma::Policy, Stages, F16Block, KIterPerFold>;

  using Epilogue = typename Stock::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::BlockScaleGemm<Mma, Epilogue, Swizzle, false>;

  static int run(const void* A, const void* B, void* D, int M, int N, int K,
                 const float* act_scale, const float* w_scale,
                 cudaStream_t stream) {
    using TensorRefA = typename Mma::IteratorA::TensorRef;
    using TensorRefB = typename Mma::IteratorB::TensorRef;
    using TensorRefD = typename Epilogue::OutputTileIterator::TensorRef;

    cutlass::gemm::GemmCoord problem(M, N, K);
    cutlass::gemm::GemmCoord grid_shape = Swizzle::get_tiled_shape(
        problem, {TbShape::kM, TbShape::kN, TbShape::kK}, 1);

    typename GemmKernel::Params params(
        problem, grid_shape,
        TensorRefA(reinterpret_cast<ElementA*>(const_cast<void*>(A)), K),
        TensorRefB(reinterpret_cast<ElementB*>(const_cast<void*>(B)), K),
        TensorRefD(reinterpret_cast<ElementC*>(D), N),
        TensorRefD(reinterpret_cast<ElementC*>(D), N),
        typename EpilogueOp::Params(1.0f, 0.0f),
        nullptr, nullptr, nullptr, nullptr);
    params.block_scale_act = act_scale;
    params.block_scale_w = w_scale;
    params.block_scale_k_blocks = K >> 7;

    dim3 grid = Swizzle().get_grid_shape(grid_shape);
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      static bool attr_set = false;
      if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            cutlass::Kernel<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (e != cudaSuccess) return -5;
        attr_set = true;
      }
    }
    cutlass::Kernel<GemmKernel><<<grid, GemmKernel::kThreadCount, smem_size,
                                  stream>>>(params);
    return (int)cudaGetLastError();
  }
};

using CfgQkv = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                      cutlass::gemm::GemmShape<32, 32, 64>, 3, 4>;
using CfgOut = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                      cutlass::gemm::GemmShape<32, 32, 64>, 3, 2>;
// Threadblock K must equal the 128-wide scale block for the fold.
using CfgUp = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                     cutlass::gemm::GemmShape<32, 32, 64>, 3, 4>;
using CfgDown = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                       cutlass::gemm::GemmShape<32, 32, 64>, 3, 4>;
// Stage-depth variants for the M=128 (weight-bound) video shapes. The 3-stage
// configuration was tuned at M=256; at M=128 fewer stages mean more resident
// CTAs and a deeper L2 weight stream.
using CfgQkvS2 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                        cutlass::gemm::GemmShape<32, 32, 64>, 2, 4>;
using CfgOutS2 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                        cutlass::gemm::GemmShape<32, 32, 64>, 2, 2>;
using CfgUpS2 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                       cutlass::gemm::GemmShape<32, 32, 64>, 2, 4>;
using CfgDownS2 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 32, 64>, 2, 4>;
using CfgQkvS4 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                        cutlass::gemm::GemmShape<32, 32, 64>, 4, 4>;
using CfgOutS4 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                        cutlass::gemm::GemmShape<32, 32, 64>, 4, 2>;
using CfgUpS4 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                       cutlass::gemm::GemmShape<32, 32, 64>, 4, 4>;
using CfgDownS4 = BSGemm<cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 32, 64>, 4, 4>;

// FP16 block-sum variants (weights quantized to |w| <= 1 per tile). Measured
// on real block-0 weights at M=256, L2 flushed (FP32-sum kernel above ->
// these): qkv 209.4 -> 166.2 us, 3072^2 78.1 -> 60.0, ffn up 315.1 -> 235.5,
// ffn down 305.9 -> 221.7. The 64x128 CTA with 32x64 warps wins every shape
// except ffn up, which takes the 128x128 CTA with 64x64 warps.
using CfgF16 = BSGemm<cutlass::gemm::GemmShape<64, 128, 128>,
                      cutlass::gemm::GemmShape<32, 64, 64>, 3, 4, true>;
using CfgF16Up = BSGemm<cutlass::gemm::GemmShape<128, 128, 128>,
                        cutlass::gemm::GemmShape<64, 64, 64>, 3, 4, true>;
// 64-deep K tile, folded every two iterations: four stages of a 64x128 tile
// fit in 48 KB, so two CTAs (16 warps) share an SM. M=256 on real block-0
// weights, L2 flushed (CfgF16/CfgF16Up -> these): qkv 165.4 -> 148.5 us,
// 3072^2 59.7 -> 54.6, ffn up 234.4 -> 217.1, ffn down 219.5 -> 216.5
// (64x32 warps); per block-forward 798 -> 746 us. Same block sums and fold;
// the outputs agree with the 128-deep kernels to one bf16 ulp.
using CfgF16K64 = BSGemm<cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>, 4, 4, true, 2>;
using CfgF16K64Down = BSGemm<cutlass::gemm::GemmShape<64, 128, 64>,
                             cutlass::gemm::GemmShape<64, 32, 64>, 4, 4, true, 2>;

}  // namespace

int fp8_bs_gemm_cutlass_f16blk_sm89(const void* A, const void* B, void* D,
                                    int M, int N, int K,
                                    const float* act_scale, const float* w_scale,
                                    cudaStream_t stream) {
  // M need only be a multiple of 16: rows past M are masked by the CUTLASS
  // iterators, but the fold reads act_scale up to the 64-row (128 for the
  // ffn-up tile) CTA edge, so the caller's act_scale buffer must cover it.
  if ((K & 127) || (N & 127) || (M & 15)) return -2;
  if (N == 14336 && (M & 127) == 0)
    return CfgF16Up::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  return CfgF16::run(A, B, D, M, N, K, act_scale, w_scale, stream);
}

int fp8_bs_gemm_cutlass_f16blk_k64_sm89(const void* A, const void* B, void* D,
                                        int M, int N, int K,
                                        const float* act_scale, const float* w_scale,
                                        cudaStream_t stream) {
  // Same operand contract as fp8_bs_gemm_cutlass_f16blk_sm89 (the fold reads
  // act_scale up to the 64-row CTA edge).
  if ((K & 127) || (N & 127) || (M & 15)) return -2;
  if (N == 3072 && K > 3072)
    return CfgF16K64Down::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  return CfgF16K64::run(A, B, D, M, N, K, act_scale, w_scale, stream);
}

int fp8_bs_gemm_cutlass_sm89(const void* A, const void* B, void* D,
                             int M, int N, int K,
                             const float* act_scale, const float* w_scale,
                             cudaStream_t stream) {
  // M must fill whole 64-row CTA tiles: the fold reads act_scale rows
  // up to the tile edge, which are only valid when M is a multiple of 64.
  if ((K & 127) || (N & 63) || (M & 63)) return -2;
  // At the CFG-elided M=128 video shapes the 4-stage pipe wins for the two
  // N=3072 shapes (measured 1.056x attn1 out, 1.168x ffn down, bit-equal);
  // qkv/ffn up stay on the 3-stage pipe (s4 loses 11-14% there).
  if (N == 9216 && K == 3072) return CfgQkv::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 3072) {
    if (M == 128) return CfgOutS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
    return CfgOut::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  }
  if (N == 14336 && K == 3072) return CfgUp::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 14336) {
    if (M == 128) return CfgDownS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
    return CfgDown::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  }
  return -3;
}

int fp8_bs_gemm_cutlass_sm89_s2(const void* A, const void* B, void* D,
                                int M, int N, int K,
                                const float* act_scale, const float* w_scale,
                                cudaStream_t stream) {
  if ((K & 127) || (N & 63) || (M & 63)) return -2;
  if (N == 9216 && K == 3072) return CfgQkvS2::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 3072) return CfgOutS2::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 14336 && K == 3072) return CfgUpS2::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 14336) return CfgDownS2::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  return -3;
}

int fp8_bs_gemm_cutlass_sm89_s4(const void* A, const void* B, void* D,
                                int M, int N, int K,
                                const float* act_scale, const float* w_scale,
                                cudaStream_t stream) {
  if ((K & 127) || (N & 63) || (M & 63)) return -2;
  if (N == 9216 && K == 3072) return CfgQkvS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 3072) return CfgOutS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 14336 && K == 3072) return CfgUpS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  if (N == 3072 && K == 14336) return CfgDownS4::run(A, B, D, M, N, K, act_scale, w_scale, stream);
  return -3;
}

}  // namespace bs_cutlass_sm89
}  // namespace gemm
}  // namespace flash_rt
