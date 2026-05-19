// SPDX-License-Identifier: Apache-2.0
//
// G7.23 v4 — v3 with x cached in REGISTERS (bf162 packed) instead of smem.
//
// v3 hit 1300 GB/s on the C=256 main shape (4 CTAs/SM with 24 KB smem)
// but only 280 GB/s on the C=512 heavy shape because sm_x dominated
// smem (48 KB) → 2 CTAs/SM → 25% warp occupancy → undermasked latency.
//
// v4 drops sm_x entirely. Each thread caches its own c-stripe values
// (up to 128 BF16 = 64 __nv_bfloat162 packed pairs) in register array
// during pass 1, then re-reads from regs in pass 2. Smem now holds
// only sm_out (FP8) + sm_red (reduce):
//
//   v3 smem ≈ 96·C + 1 KB    (sm_x + sm_out + sm_red)
//   v4 smem ≈ 32·C + 1 KB    (sm_out + sm_red)
//
//   Per-shape smem & expected occupancy (CTAs/SM):
//     C= 256: v3 24 KB → 4   /  v4  9 KB → 8+ (likely cap by reg)
//     C= 512: v3 48 KB → 2   /  v4 17 KB → 4
//     C=1024: v3 97 KB → 1   /  v4 33 KB → 2
//
// Reg pressure: __nv_bfloat162[64] = 64 32-bit regs/thread for C=1024
// case (worst). Combined with pipeline regs ~30 → ~95 regs/thread.
// 256 threads × 95 = 24K regs/CTA, SM has 65K → 2 CTAs/SM by reg
// for C=1024 (matches the smem-derived 2 CTAs/SM ceiling). For
// C ≤ 512 the reg array is small enough that smem becomes the
// limiter (4 CTAs/SM).
//
// To prevent register-array spills to local memory, the c-loop is
// fully #pragma unroll and bounded by a compile-time max constant
// kMaxBf162 = 64 with runtime mask via my_n_c.
//
// All other passes (NCDHW read, sum_sq reduce via sm_red, RMS·γ·SiLU·
// quant compute, uint32 vec coalesced write) match v3.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

int bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
    const void*  x_bf16,
    const void*  gamma_bf16,
    void*        y_fp8,
    int B, int C, int T, int H, int W,
    float act_scale,
    float eps,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
