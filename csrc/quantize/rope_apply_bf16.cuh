// SPDX-License-Identifier: Apache-2.0
//
// G7.15 — Fused 3D RoPE apply for Wan video Q/K. Replaces the Python
// chain (to fp64 → reshape → view_as_complex → multiply broadcast →
// view_as_real → reshape → .float() = 5-6 launches per call) with a
// single CUDA kernel.
//
// Math (per (b, t, n, c_pair) for t < seq_len):
//   re_y = re_x * re_f - im_x * im_f
//   im_y = re_x * im_f + im_x * re_f
// where re_x/im_x are read from in_bf16[(b, t, n, 2c)/(2c+1)] and
// re_f/im_f from freqs_*[t, c]. For t >= seq_len, in is copied to out.
//
// Layout: row-major (B, T, N, head_dim) for both in and out. head_dim
// is 2 * c_complex where c_complex = freqs.shape[1].

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// in_bf16    : (B, T, N, head_dim) bf16 row-major
// freqs_re   : (seq_len, c_complex) fp32 row-major (real part of complex freq grid)
// freqs_im   : (seq_len, c_complex) fp32 row-major (imag part)
// out_fp32   : (B, T, N, head_dim) fp32 row-major (matches upstream .float() return)
// head_dim   : must equal 2 * c_complex
void rope_apply_bf16_to_fp32(
    const void*  in_bf16,
    const float* freqs_re,
    const float* freqs_im,
    void*        out_fp32,
    int B, int T, int N, int head_dim, int seq_len,
    cudaStream_t stream);

// G7.16 — bf16 output variant. Same math; output stays in bf16 so the
// downstream cat([q_video_rope, action_q, und_q]) can keep bf16 and
// FA2 picks its bf16 tensor-core fast path instead of fp32 fallback.
void rope_apply_bf16_to_bf16(
    const void*  in_bf16,
    const float* freqs_re,
    const float* freqs_im,
    void*        out_bf16,
    int B, int T, int N, int head_dim, int seq_len,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
