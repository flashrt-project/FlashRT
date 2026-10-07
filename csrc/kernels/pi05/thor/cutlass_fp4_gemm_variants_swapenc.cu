// NVFP4 GEMM variants with the operands swapped for the encoder projections.
//
// The encoder projections have M=784 activation rows (three views): with the
// 256-row 2-SM tile the fourth M tile carries 16 useful rows but costs a full
// wave (32 cluster tiles on 10 cluster slots = 3.2 waves). Computing
// D^T = W * X^T puts the 784 rows on the N axis where the CTA tile can be 192:
// 8 x 5 = 40 cluster tiles, exactly 4 waves. D is stored column-major, which
// is byte-for-byte the row-major (M, N) output the callers already expect.
// The clusters along M' multicast the activation tile to the CTA pairs.
#include "kernels/pi05/thor/cutlass_fp4_gemm_variants_mcast.cuh"

namespace flash_rt {
namespace fp4 {
namespace variants_swapenc {
using namespace cute;
using variants_mcast::Variant;
using E0 = Variant<Shape<_256,_192,_128>, Shape<_2,_1,_1>, true>;   // 2-SM, 8x5 = 40 cluster tiles = 4 waves
using E1 = Variant<Shape<_256,_192,_128>, Shape<_4,_1,_1>, true>;   // + activations multicast x2 (2 pairs along M')
using E2 = Variant<Shape<_256,_192,_256>, Shape<_4,_1,_1>, true>;   // same, k-tile 256
}  // namespace variants_swapenc

// Public entry keeps the (activation A, weight B, M, N, K) convention of cutlass_fp4_gemm_variant.
int cutlass_fp4_gemm_variant_swapenc(int idx, void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using namespace variants_swapenc;
  switch (idx) {
    case 0: return E0::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 1: return E1::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 2: return E2::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    default: return -99;
  }
}
}  // namespace fp4
}  // namespace flash_rt
