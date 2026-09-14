// NVFP4 GEMM variants for the large (encoder / SigLIP) problem shapes:
// static persistent scheduling through the forked kernel layer (no CLC) and
// wider TMA-multicast clusters. Experimental; selected by index only.
#include "gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cuh"

namespace flash_rt {
namespace fp4 {
namespace variants_persist {
using namespace cute;
using variants_earlyb::Variant;
using SPS = cutlass::gemm::StaticPersistentScheduler;
using P0 = Variant<Shape<_128,_256,_256>, Shape<_1,_1,_1>, true, 0, false, 0, false, false, SPS>;   // v8 tile, static persistent
using P1 = Variant<Shape<_256,_256,_256>, Shape<_2,_1,_1>, true, 0, false, 0, false, false, SPS>;   // v38 tile (2-SM), static persistent
using P2 = Variant<Shape<_256,_256,_256>, Shape<_4,_1,_1>, true, 0, false, 0, false, false, SPS>;   // 2-SM, cluster 4x1 (B multicast x2), static persistent
using P3 = Variant<Shape<_256,_256,_256>, Shape<_2,_2,_1>, true, 0, false, 0, false, false, SPS>;   // 2-SM, cluster 2x2 (A multicast x2), static persistent
}  // namespace variants_persist

int cutlass_fp4_gemm_variant_persist_a(int idx, void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using namespace variants_persist;
  switch (idx) {
    case 0: return P0::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 1: return P1::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 2: return P2::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 3: return P3::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    default: return -99;
  }
}
}  // namespace fp4
}  // namespace flash_rt
