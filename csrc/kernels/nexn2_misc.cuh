// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini (qwen3_5_moe) layout-glue kernels for SM120a / RTX 5090.
//
// Nex-N2 shares the Qwen3.6 Gated DeltaNet / full-attn math but differs in
// head counts, so the qwen36 split/broadcast kernels (hardwired 16->48 and
// NQ=24) do not apply. These are the Nex-N2 variants:
//   * linear-attn: num_k_heads = 16, num_v_heads = 32 (Q/K broadcast 2x),
//     head_dim = 128; conv_out row = Q(16*128) K(16*128) V(32*128) = 8192.
//   * full-attn: num_q_heads = 16, head_dim = 256; q_proj row =
//     [q_pre(256), gate(256)] per head = 16*512 = 8192.
//
// All other GDN/full-attn kernels (gating, recurrent, gated-norm, partial
// RoPE) are head-count-parameterised and reused as-is.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Split + broadcast linear-attention conv output, 16 K-heads -> 32 V-heads.
// conv_out: (S, 8192) = Q(16*128), K(16*128), V(32*128).
// q32/k32/v32: contiguous (S, 32, 128); Q/K head h sourced from floor(h/2).
void nexn2_lin_split_qkv_broadcast_bf16(
    const void* conv_out,
    void*       q32,
    void*       k32,
    void*       v32,
    int S,
    cudaStream_t stream);

// Split full-attention q_proj output (16 Q-heads).
// q_proj: (S, 16, 512) = [q_pre(256), gate(256)] per head.
// q_pre:  (S, 16, 256), contiguous.
// gate:   (S, 16*256), contiguous.
void nexn2_split_q_gate_bf16(
    const void* q_proj,
    void*       q_pre,
    void*       gate,
    int S,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
