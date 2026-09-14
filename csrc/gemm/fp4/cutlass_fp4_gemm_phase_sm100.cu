// Decoder NVFP4 GEMM (variant-28 configuration) launched together with the
// element-wise phase that consumes its output: extra clusters run the phase
// once every GEMM CTA has stored (see sm100_gemm_phase_cta.hpp), so the next
// GEMM can be launched at this kernel's PDL trigger.
#include "gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_phase_sm100.cuh"
#include "gemm/fp4/sm100_gemm_phase_cta.hpp"
#include <cuda_fp16.h>

namespace flash_rt {
namespace fp4 {
namespace variants_phase {
using namespace cute;
using variants_earlyb::Variant;
// Swapped operands, 2-SM 256x64x256 tile, 3 weight k-tiles before the PDL wait,
// forked kernel + static scheduler sized to one tile per cluster.
using PV = Variant<Shape<_256, _64, _256>, Shape<_2, _1, _1>, true, 0, true, 3, true, false, cutlass::gemm::StaticPersistentScheduler>;
}  // namespace variants_phase

int cutlass_fp4_gemm_phase(void const* A, void const* SFA, void const* B, void const* SFB, void* D,
                           int M, int N, int K, float alpha, float beta, cudaStream_t stream,
                           PhaseHostArgs const& h) {
  if (h.kind == 0 || h.counter == nullptr || h.num_phase_ctas < 0 || (h.num_phase_ctas & 1)) return -11;
  if (M > 64) return -12;   // one activation (N-)tile after the swap
  PhaseCtaParams ph;
  ph.kind = h.kind;
  ph.num_phase_ctas = h.num_phase_ctas;
  ph.cluster_size = 2;
  ph.num_gemm_ctas = 2 * ((N + 255) / 256);   // weight rows after the swap, one 256-row tile per cluster
  if (h.num_phase_ctas == 0 && ph.num_gemm_ctas > 20) return -13;   // in-GEMM phase needs every GEMM CTA co-resident
  ph.counter = static_cast<unsigned*>(h.counter);
  ph.args.x = static_cast<const __half*>(h.x);
  ph.args.prev_gate = static_cast<const __half*>(h.prev_gate);
  ph.args.residual = static_cast<__half*>(h.residual);
  ph.args.style = static_cast<const __half*>(h.style);
  ph.args.packed = static_cast<uint8_t*>(h.packed);
  ph.args.sfa = static_cast<uint8_t*>(h.sfa);
  ph.args.gate = static_cast<__half*>(h.gate_out);
  ph.args.S = h.S;
  ph.args.D = h.D;
  ph.dbg = h.dbg;
  return variants_phase::PV::run_phase(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream, ph);
}

}  // namespace fp4
}  // namespace flash_rt
