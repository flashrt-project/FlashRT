// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 action-expert weight-only int8 GEMM (W8A16) for M <= 16.
//
// Weight-only: the activations stay bf16, the weights are int8 with one fp32
// scale per output channel, and the reduction is fp32 on bf16 tensor cores.
// FlashRT's other int8 path is W8A8 and is not built for GPU_ARCH=120.
//
// Shapes: M <= 16, N % 16 == 0, K % 32 == 0. The weights must first be run
// through w8a16_pack; the layout it produces is what the GEMMs read.
//
// x is [M, K] bf16, wp the packed weights, ws the fp32 per-channel scales,
// out [M, N] bf16 unless stated otherwise.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace dm05 {

// Repack an (N, K) int8 weight into the fragment order the GEMMs below read.
// `out` must hold N*K bytes, the same count as the input.
void w8a16_pack(const int8_t* w, int8_t* out, int N, int K, cudaStream_t stream);

// out[m, n] = bf16( sum_k x[m, k] * dequant(w[n, k]) ).
void w8a16_tc(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
              __nv_bfloat16* out, int M, int N, int K, int warps, cudaStream_t stream);

// The same GEMM over a weight whose N is the concatenation of three segments,
// each written to its own buffer with its own leading dimension: rows
// [0, n0) go to o0, [n0, n0+n1) to o1, the rest to o2. This is what lets a fused
// q|k|v GEMM write q to its buffer and k, v straight into the rows of a cache.
void w8a16_tc3(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
               __nv_bfloat16* o0, int n0, int ld0,
               __nv_bfloat16* o1, int n1, int ld1,
               __nv_bfloat16* o2, int n2, int ld2,
               int M, int K, int warps, cudaStream_t stream);

// Gated MLP epilogue over a fused [gate | up] weight: the weight's N is 2 * ff,
// block rows 0..7 are gate channels and 8..15 the matching up channels, and the
// output is bf16(bf16(gelu_tanh(bf16(gate))) * bf16(up)) with the separate gate
// GEMM, up GEMM and gelu-multiply roundings, at out[m, c] for c in [0, ff).
void w8a16_tc_glu(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
                  __nv_bfloat16* out, int ff, int M, int K, int warps, cudaStream_t stream);

}  // namespace dm05
}  // namespace flash_rt
