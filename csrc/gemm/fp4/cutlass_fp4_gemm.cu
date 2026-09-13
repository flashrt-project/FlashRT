// ============================================================================
//  FlashRT — NVFP4 GEMM impl. Based on CUTLASS 4.x example 72a
//  (Blackwell NVFP4 × NVFP4 → bfloat16), modified for fp16 output and to
//  expose a C-style entry point consumable from pybind/bindings.cpp.
//
//  Build target guard: only compiled when ENABLE_SM100_CUTLASS=ON
//  (i.e. Thor SM110 or Blackwell SM100+). Arch flag must be -arch=sm_110a
//  or -arch=sm_100a because TCGEN05_MXF4_MMA requires the 'a' suffix.
// ============================================================================

#include "cutlass_fp4_gemm.cuh"
#include "fused_fp4/pdl.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/tensor_ref.h"
#  include "cutlass/epilogue/thread/linear_combination.h"
#  include "cutlass/gemm/dispatch_policy.hpp"
#  include "cutlass/gemm/collective/collective_builder.hpp"
#  include "cutlass/epilogue/collective/collective_builder.hpp"
#  include "cutlass/gemm/device/gemm_universal_adapter.h"
#  include "cutlass/gemm/kernel/gemm_universal.hpp"
#  include "cutlass/util/packed_stride.hpp"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_FP4_HAVE_CUTLASS 1
#else
#  define FV_FP4_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fp4 {

bool has_nvfp4_sm110() {
#if FV_FP4_HAVE_CUTLASS
  return true;
#else
  return false;
#endif
}

#if FV_FP4_HAVE_CUTLASS

using namespace cute;

// ── Kernel configuration (mirrors CUTLASS 72a but fp16 output) ───────────────
using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::half_t;      // fp16 output (Pi0.5 downstream expects fp16)
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

// Tuned for Pi0.5 decoder: small M (SQ=10), wide N (2H=16384 for Gate+Up, or
// H=8192 for Down). MmaTileShape picked to cover small-M case efficiently.
// We may add shape-specific kernel instantiations (e.g. _wide, _sq) later —
// for now single "sq" variant to validate end-to-end numerics.
using MmaTileShape  = Shape<_128, _128, _128>;
using ClusterShape  = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

#endif // FV_FP4_HAVE_CUTLASS

// ──────────────────────────────────────────────────────────────────────────────
int cutlass_fp4_sq_fp16(
    void const* A_fp4_packed,
    void const* SFA,
    void const* B_fp4_packed,
    void const* SFB,
    void* D_fp16,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
#if FV_FP4_HAVE_CUTLASS
  // Derive strides and layouts for this call.
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  // The CUTLASS kernel expects typed pointers. A_fp4_packed is a packed
  // byte stream (2 int4 per byte); CUTLASS treats this as an ElementA::DataType
  // array which is `cutlass::float_e2m1_t` stored at 4 bits each. We just
  // reinterpret_cast the device byte pointer.
  using EA_data = typename ElementA::DataType;
  using EA_sf   = typename ElementA::ScaleFactorType;
  using EB_data = typename ElementB::DataType;
  using EB_sf   = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      { // Mainloop
          reinterpret_cast<EA_data const*>(A_fp4_packed), stride_A,
          reinterpret_cast<EB_data const*>(B_fp4_packed), stride_B,
          reinterpret_cast<EA_sf   const*>(SFA),          layout_SFA,
          reinterpret_cast<EB_sf   const*>(SFB),          layout_SFB
      },
      { // Epilogue
          {alpha, beta},
          // For beta=0 we don't read C. The CUTLASS API still requires a ptr,
          // so reuse D for C (unused when beta=0).
          reinterpret_cast<ElementC*>(D_fp16), stride_C,
          reinterpret_cast<ElementD*>(D_fp16), stride_D
      }
  };

  Gemm gemm;

  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x10000;
  }

  size_t ws_sz = Gemm::get_workspace_size(args);
  void* ws_ptr = nullptr;
  if (ws_sz > 0) {
    if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) return -1;
  }

  st = gemm.initialize(args, ws_ptr, stream);
  if (st != cutlass::Status::kSuccess) {
    if (ws_ptr) cudaFree(ws_ptr);
    return static_cast<int>(st) | 0x20000;
  }

  st = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
  if (ws_ptr) cudaFree(ws_ptr);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x30000;
  }
  return 0;
#else
  // Build-time: not compiled with CUTLASS SM100 support. Fail loud.
  (void)A_fp4_packed; (void)SFA; (void)B_fp4_packed; (void)SFB;
  (void)D_fp16; (void)M; (void)N; (void)K; (void)alpha; (void)beta; (void)stream;
  return -2;
#endif
}

} // namespace fp4
} // namespace flash_rt
