// ================================================================
// FlashRT — split-half ("rotate_half" / GPT-NeoX) RoPE on Q and K
//
// Applies rotary position embedding using the rotate_half convention
// that pairs element d with d + head_dim/2, matching HuggingFace's
// apply_rotary_pos_emb:
//
//     out[..., d]        = x[..., d]      * cos[d] - x[..., d+half] * sin[d]
//     out[..., d+half]   = x[..., d+half] * cos[d] + x[..., d]      * sin[d]
//
// This is distinct from rope.cu::rope_apply, which uses the interleaved
// (d, d+1) convention. The kernel is model-agnostic and reusable by any
// attention site whose RoPE uses the rotate_half form (Qwen3 backbones,
// SigLIP-style ViT towers, etc.).
//
//   q_in / q_out : (rows, q_heads, head_dim) bf16  (in-place allowed)
//   k_in / k_out : (rows, k_heads, head_dim) bf16  (in-place allowed)
//   cos  / sin   : (rows, head_dim/2)        bf16
//
// One thread handles one (row, head, d<half) triple per tensor; Q and K
// are rotated in the same launch (k_heads may differ from q_heads).
// ================================================================

#include <cuda_bf16.h>

#include "common.cuh"

namespace flash_rt {
namespace kernels {

__global__ void rope_neox_qk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ cos_tab,
    const __nv_bfloat16* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows, int q_heads, int k_heads, int head_dim) {
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_total = rows * q_heads * half;
    const int k_total = rows * k_heads * half;

    if (idx < q_total) {
        const int d = idx % half;
        const int tmp = idx / half;
        const int head = tmp % q_heads;
        const int row = tmp / q_heads;
        const int base = (row * q_heads + head) * head_dim;
        const float c = to_f32(cos_tab[row * half + d]);
        const float s = to_f32(sin_tab[row * half + d]);
        const float x0 = to_f32(q_in[base + d]);
        const float x1 = to_f32(q_in[base + d + half]);
        q_out[base + d] = from_f32<__nv_bfloat16>(x0 * c - x1 * s);
        q_out[base + d + half] = from_f32<__nv_bfloat16>(x1 * c + x0 * s);
    }

    if (idx < k_total) {
        const int d = idx % half;
        const int tmp = idx / half;
        const int head = tmp % k_heads;
        const int row = tmp / k_heads;
        const int base = (row * k_heads + head) * head_dim;
        const float c = to_f32(cos_tab[row * half + d]);
        const float s = to_f32(sin_tab[row * half + d]);
        const float x0 = to_f32(k_in[base + d]);
        const float x1 = to_f32(k_in[base + d + half]);
        k_out[base + d] = from_f32<__nv_bfloat16>(x0 * c - x1 * s);
        k_out[base + d + half] = from_f32<__nv_bfloat16>(x1 * c + x0 * s);
    }
}

void rope_neox_qk_bf16(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    int rows, int q_heads, int k_heads, int head_dim,
    cudaStream_t stream) {
    const int half = head_dim / 2;
    const int q_total = rows * q_heads * half;
    const int k_total = rows * k_heads * half;
    const int total = q_total > k_total ? q_total : k_total;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    rope_neox_qk_kernel<<<blocks, threads, 0, stream>>>(
        q_in, k_in, cos_tab, sin_tab, q_out, k_out,
        rows, q_heads, k_heads, head_dim);
}

}  // namespace kernels
}  // namespace flash_rt
