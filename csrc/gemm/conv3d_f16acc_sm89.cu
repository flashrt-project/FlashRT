// SPDX-License-Identifier: Apache-2.0
//
// See the header. The Wan VAE encoder spends its time in 3x3x3 convolutions at
// 160-640 channels; cuDNN runs them on the FP32-accumulating BF16 MMA, which
// GeForce Ada issues at half the FP16-accumulate rate. This is the stock
// CUTLASS 2.x implicit-GEMM conv3d (optimized iterators) with FP16 operands and
// FP16 accumulation; measured on the VAE shapes it is ~2x the cuDNN BF16 kernel
// with a relative error of 3-7e-3 (bf16 output rounding alone: 2.9e-3).
#include "conv3d_f16acc_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace flash_rt {
namespace gemm {
namespace conv3d_sm89 {

namespace {

using namespace cutlass;

template <int BM, int BN, int BK, int WM, int WN, int Stages>
struct ConvF16Acc {
  using Kernel = typename conv::kernel::DefaultConv3dFprop<
      half_t, layout::TensorNDHWC, half_t, layout::TensorNDHWC, bfloat16_t,
      layout::TensorNDHWC, half_t, arch::OpClassTensorOp, arch::Sm80,
      cutlass::gemm::GemmShape<BM, BN, BK>, cutlass::gemm::GemmShape<WM, WN, BK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      epilogue::thread::LinearCombination<bfloat16_t, 8, half_t, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, arch::OpMultiplyAdd,
      conv::IteratorAlgorithm::kOptimized>::Kernel;
  using Op = conv::device::ImplicitGemmConvolution<Kernel>;

  static int run(const void* x, const void* w, const void* bias, void* y, int N, int D, int H,
                 int W, int C, int Ko, int T, int R, int pad, cudaStream_t stream) {
    const int Z = D - T + 1, P = H + 2 * pad - R + 1, Q = W + 2 * pad - R + 1;
    conv::Conv3dProblemSize problem(N, D, H, W, C, Ko, T, R, R, Z, P, Q, 0, pad, pad, 1, 1, 1,
                                    1, 1, 1, conv::Mode::kCrossCorrelation, 1, 1);
    // The bias is the epilogue's source operand with zero strides: every
    // output position reads bias[k].
    layout::TensorNDHWC bias_layout(layout::TensorNDHWC::Stride(0));
    typename Op::Arguments args(
        problem,
        {reinterpret_cast<half_t*>(const_cast<void*>(x)),
         layout::TensorNDHWC::packed({N, D, H, W, C})},
        {reinterpret_cast<half_t*>(const_cast<void*>(w)),
         layout::TensorNDHWC::packed({Ko, T, R, R, C})},
        {reinterpret_cast<bfloat16_t*>(const_cast<void*>(bias)), bias_layout},
        {reinterpret_cast<bfloat16_t*>(y), layout::TensorNDHWC::packed({N, Z, P, Q, Ko})},
        {1.f, bias != nullptr ? 1.f : 0.f});
    Op op;
    if (op.can_implement(args) != Status::kSuccess) return -1;
    if (op.get_workspace_size(args) != 0) return -2;
    if (op.initialize(args, nullptr, stream) != Status::kSuccess) return -3;
    return op.run(stream) == Status::kSuccess ? 0 : -4;
  }
};

// Tiles measured on the VAE shapes (fp16 accumulate, us, 4060 Ti):
//   C=160 64x64x18  128x256 3413   64x128 3722   128x128 4572
//   C=320 32x32x18  64x128 2919    128x128 3409
//   C=640 16x16x10  64x128 1253    128x128x64 1292
//   C=640  8x8x6    64x128  217    128x128 297
using TileWide = ConvF16Acc<128, 256, 32, 64, 64, 3>;
using TileNarrow = ConvF16Acc<64, 128, 32, 32, 64, 4>;

__global__ void time_cat_kernel(const __nv_bfloat162* __restrict__ cache, long long cache_n,
                                const __nv_bfloat162* __restrict__ x, long long x_n,
                                __half2* __restrict__ out, long long total) {
  // per batch element: cache_n then x_n bf16 pairs
  const long long per = cache_n + x_n;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const long long n = i / per, o = i - n * per;
    const __nv_bfloat162 v = o < cache_n ? cache[n * cache_n + o] : x[n * x_n + (o - cache_n)];
    out[i] = __float22half2_rn(__bfloat1622float2(v));
  }
}

}  // namespace

int conv3d_ndhwc_f16acc_bf16out(const void* x, const void* w, const void* bias, void* y,
                                int N, int D, int H, int W, int C, int Ko, int T, int R,
                                int pad, cudaStream_t stream) {
  if ((C & 7) || (Ko & 7) || T != R || (T != 1 && T != 3) || D < T) return -10;
  if (H >= 64) return TileWide::run(x, w, bias, y, N, D, H, W, C, Ko, T, R, pad, stream);
  return TileNarrow::run(x, w, bias, y, N, D, H, W, C, Ko, T, R, pad, stream);
}

int ndhwc_time_cat_bf16_to_f16(const void* cache, int Tc, const void* x, int Tx, void* out,
                               int N, long long frame, cudaStream_t stream) {
  if ((frame & 1) || Tx <= 0 || Tc < 0) return -1;
  const long long cache_n = Tc * frame / 2, x_n = Tx * frame / 2;
  const long long total = N * (cache_n + x_n);
  const int threads = 256;
  const int blocks = (int)((total + threads - 1) / threads < 4096 ? (total + threads - 1) / threads
                                                                  : 4096);
  time_cat_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(cache), cache_n,
      reinterpret_cast<const __nv_bfloat162*>(x), x_n, reinterpret_cast<__half2*>(out), total);
  return (int)cudaGetLastError();
}

}  // namespace conv3d_sm89
}  // namespace gemm
}  // namespace flash_rt
