// NVFP4 GEMM variants whose mainloop streams the weight tiles before the
// programmatic-dependency wait (see sm100_blockscaled_mma_earlyb.hpp).
// The kernel-level GDC waits are compiled out here on purpose: the fork's
// load() performs the wait after the weight prefetch.
#include "gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cuh"

namespace flash_rt {
namespace fp4 {
namespace { int g_weight_evict_first = 0; }
void set_weight_evict_first(int on) { g_weight_evict_first = on ? 1 : 0; }
int get_weight_evict_first() { return g_weight_evict_first; }
namespace variants_earlyb {
using namespace cute;

using E0 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>>;   // v10 tile
using E1 = Variant<Shape<_128,_128,_256>, Shape<_1,_1,_1>>;   // v7 tile
using E2 = Variant<Shape<_128,_256,_256>, Shape<_1,_1,_1>>;   // v8 tile
using E3 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, true>;   // v10 tile through the forked (sequence) kernel
using E4 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, false, 2>;   // v10 tile, 2 stages (probe)
using E5 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, false, 4>;   // v10 tile, 4 stages (probe)
using E10 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, false, 0, true>;   // swapped operands, weights streamed before the PDL wait
using E11 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true>;   // swapped operands, 2-SM UMMA, weights streamed before the PDL wait
using E12 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true, 2>;   // ... only 2 weight k-tiles before the wait
using E13 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true, 3>;   // ... 3 weight k-tiles before the wait
using E14 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, false, 0, true, 2>;   // 1-SM swapped, 2 weight k-tiles before the wait
using E15 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true, 2, true, true>;    // 2 early, dependents triggered from the MMA warp
using E16 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true, 0, true, true>;    // all early, trigger from the MMA warp
using E17 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, false, 0, true, 0, false, true>;   // activations early (control), trigger from the MMA warp
using E18 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>, true, 0, false, 0, false, false, cutlass::gemm::StaticPersistentScheduler>;   // forked kernel + static persistent scheduler (no CLC)
using E19 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>, true, 0, true, 3, true, false, cutlass::gemm::StaticPersistentScheduler>;    // v28 configuration through the forked kernel + static scheduler
}  // namespace variants_earlyb

int cutlass_fp4_gemm_variant_earlyb(int idx, void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using namespace variants_earlyb;
  switch (idx) {
    case 0: return E0::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 1: return E1::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 2: return E2::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 3: return E3::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 4: return E4::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 5: return E5::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 10: return E10::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 11: return E11::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 12: return E12::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 13: return E13::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 14: return E14::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 15: return E15::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 16: return E16::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 17: return E17::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 18: return E18::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 19: return E19::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    default: return -99;
  }
}
}  // namespace fp4
}  // namespace flash_rt
