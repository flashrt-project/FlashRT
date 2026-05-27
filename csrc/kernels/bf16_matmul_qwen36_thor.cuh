// Thor-only bf16 row-major matmul specializations for Qwen3.6.
//
// These symbols are an additive, hardware-isolated companion to the
// shared `bf16_matmul_qwen36_bf16` family. They use kernel
// configurations (dynamic shared memory > 99 KB) that exceed the
// per-block opt-in limit on SM120-class GPUs and must never be
// dispatched on those devices. The implementation queries
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` and a checked
// `cudaFuncSetAttribute` at first use; if either is insufficient the
// caller is expected to fall back to the shared kernel. The Thor
// frontend opts in via a per-class hook; the RTX frontend keeps the
// shared dispatch untouched.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

// MTP prompt-tail fc kernel: output (M, N), input (M, K=10240),
// weight (N, K=10240), bf16, row-major. Reuses W across an
// M_TILE block to cut W bandwidth by 1/M_TILE while preserving the
// per-output fma order (bit-identical to the shared generic kernel).
//
// Returns 0 on successful launch via the M-tile fast path, or a
// non-zero value when the device does not support the required
// dynamic shared memory (caller must fall back). Never throws.
int bf16_matmul_qwen36_thor_mtp_fc_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
