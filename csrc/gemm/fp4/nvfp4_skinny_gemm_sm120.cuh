// SPDX-License-Identifier: Apache-2.0
#pragma once
// Skinny NVFP4 (W4A4) GEMM family for small-row, weight-streaming shapes on
// sm_120a (action decoders and DiT action heads: M <= 48 rows per CTA).
//
// Each CTA streams BN weight rows x KC of K into mma fragments (registers)
// before the programmatic-dependent-launch wait, so the weight read of a GEMM
// overlaps the tail of its producer, then runs block-scaled e2m1 mma.sync
// (kind::mxf4nvf4, m16n8k64, UE4M3 scale per 16 elements) over up to three
// 16-row tiles of A and writes FP32 partial sums per K split. A fused consumer
// sums the splits and applies alpha = (*a_gs) * (*w_gs) (see
// nvfp4_skinny_consumers_sm120.cuh); quantization of the next GEMM's operand
// happens inside that consumer, never in a standalone kernel.
//
// Layouts:
//   activation  e2m1 [M, K/2] row-major, scales ue4m3 [M, K/16] row-major.
//   weight      fragment-major: for every (8-column tile nt, 128-wide K chunk
//               kc) 512 bytes = 32 lanes x 16 bytes; lane (t0 = lane & 3,
//               t1 = lane >> 2) holds column nt*8+t1, K elements
//               kc*128 + s*64 + r*32 + 8*t0 + [0, 8) at byte 8*s + 4*r
//               (s, r in {0, 1}). Scales: for every (nt, kc) 8 columns x 8
//               bytes (the 4 block scales of each 64-wide half). pack_weight
//               produces it from a natural [N, K] NVFP4 quantization.
//   partials    fp32 [K / KC][M][N].
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_skinny {

// Tile configurations (BN x KC x warps); the row tiling (16, 32 or 48 rows per
// CTA) follows M.
int config_count();
int config_k_chunk(int cfg);
bool config_supports(int cfg, int N, int K);

// partials[s] = A(M, Ks) @ W(N, Ks)^T. Returns cudaError_t as int.
int gemm(const void* A, const void* SFA, const void* W, const void* SFB, float* partials, int M, int N, int K,
         int cfg, bool pdl, cudaStream_t stream);

// Same with a BF16 activation [M, K] quantized to NVFP4 on load (global scale *a_gs).
int gemm_bf16_act(const __nv_bfloat16* A, const float* a_gs, const void* W, const void* SFB, float* partials,
                  int M, int N, int K, int cfg, bool pdl, cudaStream_t stream);

// Setup time: natural NVFP4 [N, K/2] + [N, K/16] into the fragment-major
// layout. N % 8 == 0, K % 128 == 0.
int pack_weight(const uint8_t* w, const uint8_t* sf, uint8_t* w_packed, uint8_t* sf_packed, int N, int K,
                cudaStream_t stream);

// Setup time: BF16 [N, K] -> natural NVFP4 with global scale *gs
// (scales ue4m3_ceil(amax16 / (6 gs))). mse_steps > 0 additionally tries that
// many smaller UE4M3 scale codes per block and keeps the one with the least
// squared error.
int quantize_weight(const __nv_bfloat16* w, const float* gs, uint8_t* out, uint8_t* out_sf, int N, int K,
                    int mse_steps, cudaStream_t stream);

}  // namespace nvfp4_skinny
}  // namespace flash_rt
