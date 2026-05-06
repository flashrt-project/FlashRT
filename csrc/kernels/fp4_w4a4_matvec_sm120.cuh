// SPDX-License-Identifier: Apache-2.0
//
// Custom NVFP4 W4A4 M=1 matvec for SM120 (RTX 5090).
//
// Why a hand-rolled kernel: CUTLASS' NVFP4 W4A4 GEMM (sm120 block-scaled
// tensor op) is tiled assuming M ≥ 16 — at M=1 most SMs sit idle and the
// effective HBM BW utilization measured at ~30% (16.4 µs to read 9 MiB
// at 4096×4096 = 555 GiB/s vs RTX 5090 peak 1.79 TB/s). LLM decode is
// always M=1 per step, so this single inefficiency caps the whole
// decode tok/s. This kernel specializes for M=1 with one warp per
// output row tile + K parallelism inside the warp, targeting ~70%+ HBM
// BW utilization → ~2× decode speedup.
//
// Schema (matches the existing `fp4_w4a16_gemm_sm120_bf16out`):
//   A_packed  : (1, K/2)        u8   FP4 e2m1 packed (2 elts/byte)
//                                    activation: BF16 → NVFP4 dynamic at runtime
//   B_packed  : (N, K/2)        u8   FP4 e2m1 packed weight, row-major
//   SFA       : (1, K/16)       u8   UE4M3 per-block SF, swizzled (Sm1xx layout)
//   SFB       : (N, K/16)       u8   UE4M3 per-block SF, swizzled (Sm1xx layout)
//   D_bf16    : (1, N)          bf16 output
//   alpha     : fp32 scalar     = 1 / weight_global_scale
//                                  (activation global scale absorbed by
//                                   the dynamic activation quantizer)
//
// Constraints (caller-checked):
//   * M = 1 always (this kernel is M=1 specialized).
//   * K must be a multiple of 16 × 32 = 512 (one warp covers K).
//     Verified: K ∈ {4096, 12288} for Qwen3-8B.
//   * N must be a multiple of ROWS_PER_BLOCK (32). True for all Qwen3
//     shapes (N ∈ {1024, 4096, 12288}).
//   * Pointer alignments: 8-byte for packed (uint64 vector loads),
//     2-byte for D output. SF pointers: 1-byte (byte-granular reads).
//
// SF swizzle layout (matches `nvfp4_sf_linear_to_swizzled`):
//   For row r, k_block b:
//     rb = r >> 7       ; ri = r & 127
//     cb = b >> 2       ; ci = b & 3
//     super_idx = rb * n_col_super + cb
//     inner_off = (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + ci
//     SF byte at (super_idx * 512 + inner_off)

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// NVFP4 W4A4 matvec, M=1 specialized, BF16 output, SM120.
//
// Returns 0 on success; nonzero on caller-side argument error.
// Internally this is fire-and-forget: success is reported by host
// argument validation only (the device kernel itself never errors).
//
// One launch per call; lifetime is the caller's stream.
int fp4_w4a4_matvec_sm120_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// Initialize the device-side UE4M3 lookup table (one-time, idempotent).
// Called by the binding init; the kernel won't run before this returns.
void fp4_w4a4_matvec_init_luts();

}  // namespace gemm
}  // namespace flash_rt
