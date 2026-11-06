// SPDX-License-Identifier: Apache-2.0
#pragma once
// Programmatic dependent launch (PDL) helpers for the activation producers
// that sit between the SM120 CUTLASS GEMMs (built with grid-dependency
// control, CUTLASS_ENABLE_GDC_FOR_SM100).
//
// Contract: a kernel launched with the PDL attribute calls pdl_wait() before
// its first global memory access (reads of producer data and writes alike),
// so it may start while its predecessor drains but never touches memory
// before that predecessor has completed. pdl_trigger() lets the successor
// start its own prologue early. Both are no-ops for a kernel launched
// without the attribute, so these kernels stay correct for every caller,
// and a successor launched without the attribute is fully serialized.
#include "kernel_policy.h"

#include <cuda_runtime.h>
#include <cstdlib>

namespace flash_rt {
namespace pdl {

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Launch with the PDL attribute. The default is off, preserving the
// pre-existing stream-serialized behavior; a model frontend that has
// validated PDL calls set_kernel_fast_paths(..., pdl=true), and
// FLASHRT_PDL=1/0 remains an explicit override. sm_90+ is required.
inline bool enabled() {
    static const int env = [] {
        const char* e = std::getenv("FLASHRT_PDL");
        return e ? std::atoi(e) : -1;
    }();
    if (env >= 0) return env != 0;
    if (!flash_rt::kernels::pdl_default()) return false;
    static const bool sm90 = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        return major >= 9;
    }();
    return sm90;
}

template <typename... KArgs, typename... Args>
inline cudaError_t launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem,
                          cudaStream_t stream, bool pdl, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

}  // namespace pdl
}  // namespace flash_rt
