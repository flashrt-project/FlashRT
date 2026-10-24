// ============================================================================
//  FlashRT — programmatic dependent launch (PDL) switch for the SM120 paths.
//
//  The same contract as csrc/kernels/pi05/thor/pi05_pdl.cuh, for the kernels
//  the X-VLA RTX pipeline launches: when enabled, CUTLASS GEMMs are launched
//  with launch_with_pdl (their producer warps already execute
//  griddepcontrol.wait under CUTLASS_ENABLE_GDC_FOR_SM100, which the vendored
//  CUTLASS also honours on sm_120a), and the elementwise / norm kernels with
//  cudaLaunchAttributeProgrammaticStreamSerialization. Every such kernel
//  executes griddepcontrol.wait before its first global access and
//  griddepcontrol.launch_dependents right after, so the next node's launch and
//  prologue overlap the current node's tail inside the CUDA graph.
//
//  Off by default and process-wide (one flag per shared object: the inline
//  function's static is merged by the linker across translation units).
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <utility>

namespace flash_rt {
namespace pdl {

inline bool& flag() {
  static bool enabled = false;
  return enabled;
}

template <typename... KArgs, typename... Args>
inline cudaError_t launch(void (*kernel)(KArgs...), dim3 grid, dim3 block,
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
  cfg.numAttrs = flag() ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

}  // namespace pdl
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
