// ============================================================================
//  FlashRT — programmatic dependent launch (PDL) switch for the Thor path.
//
//  When enabled, CUTLASS GEMMs are launched with launch_with_pdl and the
//  small activation kernels with cudaLaunchAttributeProgrammaticStreamSerialization;
//  every such kernel executes griddepcontrol.wait before touching its inputs
//  and griddepcontrol.launch_dependents right after, so the next kernel's
//  prologue (CTA scheduling, TMA descriptor prefetch, barrier init) overlaps
//  the current kernel's tail. Off by default; process-wide.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <utility>

namespace flash_rt {
namespace fp4 {

bool& pdl_flag();
inline bool pdl_launch() { return pdl_flag(); }

template <typename... KArgs, typename... Args>
inline cudaError_t launch_maybe_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block,
                                    size_t smem, cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl_launch() ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

}  // namespace fp4
}  // namespace flash_rt

#if defined(__CUDACC__)
// No-ops when the grid was not launched with a programmatic dependency.
__device__ __forceinline__ void flashrt_pdl_wait_and_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#endif
