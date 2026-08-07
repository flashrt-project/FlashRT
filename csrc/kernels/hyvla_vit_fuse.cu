// ================================================================
// FlashRT — Hy-VLA Orin ViT fusion kernels
//
// hyvla_vit_add_layer_norm_bf16:
//   residual += x_add          (bf16 round, in-place — matches torch add)
//   out       = LayerNorm(residual)
// Fuses the ViT post-attention residual add with the following LayerNorm
// (and, across blocks, the previous block's MLP residual with the entry
// LayerNorm), removing one full read+write pass per site.
//
// Precision contract: the add rounds to bf16 exactly like torch's
// elementwise add; the LayerNorm is bit-identical to this repo's
// layer_norm_kernel (fp32 two-pass mean/var, rsqrtf, single bf16 round).
// ================================================================

#include "hyvla_vit_fuse.cuh"
#include "common.cuh"

__global__ void hyvla_vit_add_layer_norm_bf16_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x_add,
        const __nv_bfloat16* __restrict__ ln_weight,
        const __nv_bfloat16* __restrict__ ln_bias,
        __nv_bfloat16* __restrict__ out,
        int dim, float eps) {
    extern __shared__ float partial[];

    int row = blockIdx.x;
    using T2 = __nv_bfloat162;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* add2 = reinterpret_cast<const T2*>(x_add + (size_t)row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* b2 = reinterpret_cast<const T2*>(ln_bias);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    // Pass 1: residual = bf16(residual + x_add) to global, sum for mean.
    // Re-reading residual from global in passes 2/3 keeps this bit-equal
    // to running torch add then layer_norm_kernel sequentially.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], av = add2[i];
        __nv_bfloat16 r0 = from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(av.x));
        __nv_bfloat16 r1 = from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(av.y));
        res2[i] = make_packed2<__nv_bfloat16>(r0, r1);
        local_sum += to_f32(r0) + to_f32(r1);
    }
    float mean = block_reduce_sum(local_sum, partial) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, partial) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i], wv = w2[i], bv = b2[i];
        float n0 = (to_f32(val.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float n1 = (to_f32(val.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(n0), from_f32<__nv_bfloat16>(n1));
    }
}

extern "C" void hyvla_vit_add_layer_norm_bf16(
        void* residual, const void* x_add,
        const void* ln_weight, const void* ln_bias,
        void* out, int rows, int dim, float eps, cudaStream_t stream) {
    int smem = 256 * sizeof(float);
    hyvla_vit_add_layer_norm_bf16_kernel<<<rows, 256, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x_add),
        reinterpret_cast<const __nv_bfloat16*>(ln_weight),
        reinterpret_cast<const __nv_bfloat16*>(ln_bias),
        reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
}
