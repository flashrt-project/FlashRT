// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini / qwen3_5_moe fused MoE unpermute (weighted gather-sum), sm120.
// out[t, :] = sum_k tw[t,k] * d_dn[rows[t,k], :]  over k in [0, TOPK).
//
// d_dn holds the routed experts' down-projection outputs in tile-sorted layout;
// rows[t,k] is the d_dn row for token t's k-th expert, tw[t,k] its router
// weight. Fusing the gather + weighted sum writes the (S, HID) result directly,
// without materialising the (S, TOPK, HID) intermediate that the torch
// gather+sum allocates (4 GB at S=32k -> the long-context prefill wall). The
// per-token k-sum runs in a fixed order, so the result is bit-reproducible.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

int moe_weighted_sum_sm120_bf16(
    const void*  d_dn,        // (num_rows, dn_stride) bf16
    const void*  rows,        // (S * TOPK) int32, d_dn row per (token, k)
    const void*  tw,          // (S * TOPK) fp32, router weight per (token, k)
    void*        out,         // (S, HID) fp32
    int          S,
    int          TOPK,
    int          HID,
    int          dn_stride,   // d_dn row stride in elements (>= HID)
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
