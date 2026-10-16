// ================================================================
// FlashRT — Patch embedding kernels
//
// 1. im2col: (nv, 224, 224, 3) → (nv*256, 588) strided copy
// 2. bias_pos: output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
// ================================================================

#include "patch_embed.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

// ── GPU im2col for SigLIP patch embedding ──
// Input:  (nv, 224, 224, 3) FP16, row-major NHWC
// Output: (nv*256, 588) FP16, each row = one 14×14×3 patch flattened
//
// Equivalent to:
//   img.reshape(nv, 16, 14, 16, 14, 3)
//      .transpose(0, 1, 3, 2, 4, 5)
//      .reshape(nv*256, 588)
__global__ void patch_im2col_kernel(
    const half* __restrict__ input,   // (nv, 224, 224, 3)
    half* __restrict__ output,        // (nv*256, 588)
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
    int pxw = (feat_idx % 42) / 3;    // pixel col within patch [0, 14)
    int c   = feat_idx % 3;            // channel [0, 3)

    // Source index in (nv, 224, 224, 3) row-major
    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    output[idx] = input[src];
}

void patch_im2col(const half* input, half* output, int nv, cudaStream_t stream)
{
    int total = nv * 256 * 588;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_im2col_kernel<<<blocks, threads, 0, stream>>>(input, output, nv);
}

__global__ void patch_im2col_uint8_kernel(
    const uint8_t* __restrict__ input,
    const half* __restrict__ lut,
    half* __restrict__ output,
    int nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nv * 256 * 588;
    if (idx >= total) return;

    int patch_idx = idx / 588;
    int feat_idx = idx % 588;
    int batch = patch_idx / 256;
    int local_patch = patch_idx % 256;
    int ph = local_patch / 16;
    int pw = local_patch % 16;
    int pxh = feat_idx / 42;
    int pxw = (feat_idx % 42) / 3;
    int c = feat_idx % 3;
    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    output[idx] = lut[input[src]];
}

void patch_im2col_uint8(const uint8_t* input, const half* lut, half* output,
                        int nv, cudaStream_t stream)
{
    int total = nv * 256 * 588;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_im2col_uint8_kernel<<<blocks, threads, 0, stream>>>(
        input, lut, output, nv);
}

// ── Bias + positional embedding ──

__global__ void patch_embed_bias_pos_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ pos_emb,
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

void patch_embed_bias_pos(half* output, const half* bias, const half* pos_emb,
                          int S, int D, int S_per_view, cudaStream_t stream)
{
    int total = S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_embed_bias_pos_kernel<<<blocks, threads, 0, stream>>>(
        output, bias, pos_emb, S, D, S_per_view);
}


// ── bf16 NCHW patch im2col, parameterised over the patch size ──
//
//   in:  (nv, C, H, W) bf16, planar NCHW
//   out: (nv * gh * gw, C * P * P) bf16, one row per patch
//
// The column order is (c, kh, kw), which is exactly
// conv.weight.reshape(out_ch, C * P * P), so the patch embedding is one GEMM
// against the transposed convolution weight. Patches are numbered row-major
// over the (gh, gw) grid, matching the convolution output's flatten(2).
//
// Distinct from patch_im2col above on four counts -- dtype (bf16 not fp16),
// layout (NCHW not NHWC), a parametrised patch size rather than a fixed 224,
// and a transposed feature order -- so neither substitutes for the other.
__global__ void patch_im2col_nchw_bf16_kernel(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    int nv, int C, int H, int W, int P) {
    const int gh = H / P, gw = W / P;
    const int feat = C * P * P;
    const long long total = (long long)nv * gh * gw * feat;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int f = (int)(idx % feat);
        const long long patch = idx / feat;
        const int v = (int)(patch / (gh * gw));
        const int pl = (int)(patch % (gh * gw));
        const int ph = pl / gw, pw = pl % gw;
        const int c = f / (P * P);
        const int kh = (f / P) % P;
        const int kw = f % P;
        const long long src = (((long long)v * C + c) * H + (ph * P + kh)) * W + (pw * P + kw);
        out[idx] = in[src];
    }
}

void patch_im2col_nchw_bf16(const __nv_bfloat16* input, __nv_bfloat16* output,
                            int nv, int C, int H, int W, int P, cudaStream_t stream) {
    if (nv <= 0 || C <= 0 || H <= 0 || W <= 0 || P <= 0) return;
    if (!input || !output)
        throw std::runtime_error("patch_im2col_nchw_bf16: null pointer argument");
    if (H % P != 0 || W % P != 0)
        throw std::runtime_error("patch_im2col_nchw_bf16: H and W must be multiples of P");
    const long long total = (long long)nv * (H / P) * (W / P) * C * P * P;
    const int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535LL * 8) blocks = 65535LL * 8;
    patch_im2col_nchw_bf16_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, nv, C, H, W, P);
}
