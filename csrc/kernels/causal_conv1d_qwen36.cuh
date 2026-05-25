// SPDX-License-Identifier: Apache-2.0
//
// Causal depthwise 1D convolution for Qwen3.6 linear-attention input
// projection (Phase 3.2). Two variants matching the HF
// causal_conv1d_fn / causal_conv1d_update API:
//
//   * Multi-token forward (prefill / chunk):
//       y[b, s, c] = silu(bias[c] +
//           sum_{i=0..k-1} x[b, s + i - (k-1), c] * w[c, i])
//     with implicit zero left-pad (causal).
//
//   * Single-token decode update:
//       state cache holds last (k-1) tokens per channel.
//       y[b, 0, c] = silu(bias[c] + dot(state[b, c, :], w[c, :k-1])
//                                + x_new[b, c] * w[c, k-1])
//       state[b, c, 0:k-1-1] = state[b, c, 1:k-1]
//       state[b, c, k-2]    = x_new[b, c]
//
// SiLU activation is fused into the epilogue (saves one pass over the
// (B, S, conv_dim=10240) tensor — small, but every bit counts toward
// the "0 overhead" goal).
//
// Qwen3.6 specifics: kernel_size=4, conv_dim=10240, bf16 throughout.
// Generic kernel handles any k <= 8 (compile-time loop unrolled).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Multi-token forward, applies SiLU when ``apply_silu`` is true.
//
//   x         : (B, S, conv_dim) bf16 row-major
//   w         : (conv_dim, k)    bf16 row-major  (depthwise)
//   bias      : (conv_dim,)      bf16 or nullptr
//   out       : (B, S, conv_dim) bf16 row-major
//   k must be <= 8 (compile-time max).
void causal_conv1d_qwen36_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       out,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

// Single-token decode update with persistent state cache.
//
//   x_new     : (B, conv_dim)            bf16 row-major
//   w         : (conv_dim, k)            bf16 row-major
//   bias      : (conv_dim,)              bf16 or nullptr
//   out       : (B, conv_dim)            bf16 row-major
//   state     : (B, conv_dim, k-1)       bf16 row-major  (in/out)
//                ^ last (k-1) tokens, indexed [oldest .. newest-1].
void causal_conv1d_qwen36_update_bf16(
    const void* x_new,
    const void* w,
    const void* bias,
    void*       out,
    void*       state,
    int B, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

// In/out-state variant for K-iter chained per-step save (A2c-3).
void causal_conv1d_qwen36_update_inout_bf16(
    const void* x_new,
    const void* w,
    const void* bias,
    void*       out,
    const void* state_in,
    void*       state_out,
    int B, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

// Multi-token decode update with persistent state cache.
// Processes S consecutive tokens and writes only the final state, so
// speculative verify paths that need per-step state snapshots should
// keep using causal_conv1d_qwen36_update_inout_bf16.
void causal_conv1d_qwen36_update_chunk_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       out,
    void*       state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

// Parallel prefill variant: computes each (S, channel) output
// independently, then updates the final state in a second tiny kernel.
// This trades extra global loads for much higher S-dimension
// parallelism on long chunks.
void causal_conv1d_qwen36_update_chunk_parallel_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       out,
    void*       state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

// Parallel prefill variant for the Qwen3.6 WY path. Computes the same
// depthwise conv output as causal_conv1d_qwen36_update_chunk_parallel_bf16,
// but writes directly to split GQA buffers:
//   q16: (B, S, 16, 128), k16: (B, S, 16, 128), v48: (B, S, 48, 128).
// This avoids materializing and rereading the full (B, S, 10240) conv tensor.
void causal_conv1d_qwen36_update_chunk_parallel_gqa_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       q16,
    void*       k16,
    void*       v48,
    void*       state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
