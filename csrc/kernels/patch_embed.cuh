// ================================================================
// FlashRT — Patch embedding kernel declarations
// GPU im2col + fused bias + positional embedding
// ================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// GPU im2col: (nv, 224, 224, 3) → (nv*256, 588)
// Pure strided copy, bit-exact, no computation.
void patch_im2col(const half* input, half* output, int nv,
                  cudaStream_t stream = 0);

// GPU im2col with exact uint8 -> FP16 normalization through a 256-entry LUT.
void patch_im2col_uint8(const uint8_t* input, const half* lut, half* output,
                        int nv, cudaStream_t stream = 0);

// Add bias + positional embedding to patch GEMM output (FP16)
// output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
void patch_embed_bias_pos(half* output, const half* bias, const half* pos_emb,
                          int S, int D, int S_per_view,
                          cudaStream_t stream = 0);
