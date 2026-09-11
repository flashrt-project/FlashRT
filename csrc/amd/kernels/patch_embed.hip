// ================================================================
// FlashRT AMD — Patch embedding kernels (FP16, CDNA wave64)
//
// 1. im2col: (nv, 224, 224, 3) → (nv*256, 588) strided copy
// 2. bias_pos: output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
//
// Mirrors csrc/kernels/patch_embed.cu host-wrapper signatures exactly.
// ================================================================

#include "common_hip.h"

// ── GPU im2col for SigLIP patch embedding ──
// Input:  (nv, 224, 224, 3) FP16, row-major NHWC
// Output: (nv*256, 588) FP16, each row = one 14×14×3 patch flattened
//
// Equivalent to:
//   img.reshape(nv, 16, 14, 16, 14, 3)
//      .transpose(0, 1, 3, 2, 4, 5)
//      .reshape(nv*256, 588)
__global__ void patch_im2col_kernel(
    const __half* __restrict__ input,   // (nv, 224, 224, 3)
    __half* __restrict__ output,        // (nv*256, 588)
    int nv)
{
    // Total output elements = nv * 256 * 588
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nv * 256 * 588;
    if (idx >= total) return;

    // Decode output index
    int patch_idx = idx / 588;         // which patch [0, nv*256)
    int feat_idx  = idx % 588;         // which feature [0, 588)

    int batch = patch_idx / 256;       // which view
    int local_patch = patch_idx % 256; // patch within view
    int ph = local_patch / 16;         // patch row [0, 16)
    int pw = local_patch % 16;         // patch col [0, 16)

    int pxh = feat_idx / 42;           // pixel row within patch [0, 14), 42=14*3
    int pxw = (feat_idx % 42) / 3;     // pixel col within patch [0, 14)
    int c   = feat_idx % 3;            // channel [0, 3)

    // Source index in (nv, 224, 224, 3) row-major
    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    output[idx] = input[src];
}

void patch_im2col(const __half* input, __half* output, int nv, hipStream_t stream)
{
    int total = nv * 256 * 588;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_im2col_kernel<<<blocks, threads, 0, stream>>>(input, output, nv);
}

// ── Bias + positional embedding ──

__global__ void patch_embed_bias_pos_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    const __half* __restrict__ pos_emb,
    int S, int D, int S_per_view)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * D;
    if (idx >= total) return;

    int i = idx / D;
    int j = idx % D;
    int pos_i = i % S_per_view;

    float v = __half2float(output[idx])
            + __half2float(bias[j])
            + __half2float(pos_emb[pos_i * D + j]);
    output[idx] = __float2half(v);
}

void patch_embed_bias_pos(__half* output, const __half* bias, const __half* pos_emb,
                          int S, int D, int S_per_view, hipStream_t stream)
{
    int total = S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_embed_bias_pos_kernel<<<blocks, threads, 0, stream>>>(
        output, bias, pos_emb, S, D, S_per_view);
}
