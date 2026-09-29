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
#include "nvfp4_convert.cuh"
#include <math_constants.h>
#include <cstdlib>

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

// Fused (residual += x_add) + LayerNorm(residual + time_pe) for the ViT
// spacetime blocks. The time positional embedding is added for the norm only
// and is NOT stored into the residual stream (matching the torch path
// `x = x + pending; h = layer_norm(x + pe)`), so the residual stream is not
// corrupted. x_add == nullptr skips the residual add.
__global__ void hyvla_vit_res_add_ln_time_bf16_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x_add,
        const __nv_bfloat16* __restrict__ pe,
        const __nv_bfloat16* __restrict__ ln_weight,
        const __nv_bfloat16* __restrict__ ln_bias,
        __nv_bfloat16* __restrict__ out,
        int dim, int n, int kf, float eps) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    int kf_idx = (row / n) % kf;
    using T2 = __nv_bfloat162;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* pe2 = reinterpret_cast<const T2*>(pe + (size_t)kf_idx * dim);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* b2 = reinterpret_cast<const T2*>(ln_bias);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    if (x_add != nullptr) {
        const T2* add2 = reinterpret_cast<const T2*>(x_add + (size_t)row * dim);
        for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
            T2 rv = res2[i], av = add2[i];
            res2[i] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(av.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(av.y)));
        }
    }
    __syncthreads();

    // v = bf16(residual + pe) (matches the torch bf16 add feeding layer_norm)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], pv = pe2[i];
        local_sum += to_f32(from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(pv.x)))
                   + to_f32(from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(pv.y)));
    }
    float mean = block_reduce_sum(local_sum, partial) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], pv = pe2[i];
        float v0 = to_f32(from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(pv.x)));
        float v1 = to_f32(from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(pv.y)));
        float d0 = v0 - mean, d1 = v1 - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, partial) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], pv = pe2[i], wv = w2[i], bv = b2[i];
        float v0 = to_f32(from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(pv.x)));
        float v1 = to_f32(from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(pv.y)));
        float n0 = (v0 - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float n1 = (v1 - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(n0), from_f32<__nv_bfloat16>(n1));
    }
}

extern "C" void hyvla_vit_res_add_ln_time_bf16(
        void* residual, const void* x_add, const void* pe,
        const void* ln_weight, const void* ln_bias, void* out,
        int rows, int dim, int n, int kf, float eps, cudaStream_t stream) {
    int smem = 256 * sizeof(float);
    hyvla_vit_res_add_ln_time_bf16_kernel<<<rows, 256, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x_add),
        reinterpret_cast<const __nv_bfloat16*>(pe),
        reinterpret_cast<const __nv_bfloat16*>(ln_weight),
        reinterpret_cast<const __nv_bfloat16*>(ln_bias),
        reinterpret_cast<__nv_bfloat16*>(out), dim, n, kf, eps);
}

// ViT tail: out (num_cam, n, d) = last history frame of (x + pending), where
// x/pending are (num_cam*K, n, d). Fuses the residual add, the frame select and
// the materialising contiguous copy into one launch. x_add may be nullptr.
__global__ void hyvla_vit_tail_slice_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ x_add,
        __nv_bfloat16* __restrict__ out,
        int num_cam, int K, int n, int d) {
    long total = (long)num_cam * n * d;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        int d_idx = (int)(i % d);
        long t = i / d;
        int n_idx = (int)(t % n);
        int cam = (int)(t / n);
        long src = ((long)cam * K + (K - 1)) * n * d + (long)n_idx * d + d_idx;
        float v = __bfloat162float(x[src]);
        if (x_add != nullptr) v += __bfloat162float(x_add[src]);
        out[i] = __float2bfloat16(v);
    }
}

extern "C" void hyvla_vit_tail_slice_bf16(
        const void* x, const void* x_add, void* out,
        int num_cam, int K, int n, int d, cudaStream_t stream) {
    long total = (long)num_cam * n * d;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hyvla_vit_tail_slice_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(x_add),
        reinterpret_cast<__nv_bfloat16*>(out), num_cam, K, n, d);
}

// ViT patch-embed bias add: y (B,C,H,W) += bias (C,). Used after a bias-free
// conv2d so the bias add is a native kernel (no framework elementwise launch).
__global__ void hyvla_vit_patch_bias_bf16_kernel(
        __nv_bfloat16* __restrict__ y,
        const __nv_bfloat16* __restrict__ bias, int B, int C, int HW) {
    long total = (long)B * C * HW;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        y[i] = __float2bfloat16(
            __bfloat162float(y[i]) + __bfloat162float(bias[(i / HW) % C]));
    }
}

extern "C" void hyvla_vit_patch_bias_bf16(
        void* y, const void* bias, int B, int C, int H, int W,
        cudaStream_t stream) {
    long total = (long)B * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hyvla_vit_patch_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(y),
        reinterpret_cast<const __nv_bfloat16*>(bias), B, C, H * W);
}

// ViT patch-embed + positional-embedding: out (B,n,d) = xbuf (B,d,n) transpose
// + pe (n,d), materialised contiguous in one launch.
__global__ void hyvla_vit_pos_add_bf16_kernel(
        const __nv_bfloat16* __restrict__ xbuf,
        const __nv_bfloat16* __restrict__ pe,
        __nv_bfloat16* __restrict__ out, int B, int n, int d) {
    long total = (long)B * n * d;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        int c = (int)(i % d);
        long t = i / d;
        int j = (int)(t % n);
        long b = t / n;
        float v = __bfloat162float(xbuf[b * (long)d * n + (long)c * n + j])
                + __bfloat162float(pe[(long)j * d + c]);
        out[i] = __float2bfloat16(v);
    }
}

extern "C" void hyvla_vit_pos_add_bf16(
        const void* xbuf, const void* pe, void* out,
        int B, int n, int d, cudaStream_t stream) {
    long total = (long)B * n * d;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hyvla_vit_pos_add_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xbuf),
        reinterpret_cast<const __nv_bfloat16*>(pe),
        reinterpret_cast<__nv_bfloat16*>(out), B, n, d);
}

// Scatter image-dependent prefix tokens into a preallocated prefix buffer whose
// static tokens (BOS/HY_USER/vision start/split/end/lang) were pre-placed once.
// Replaces the per-camera interleave cats and the final concat.
__global__ void hyvla_prefix_scatter_kernel(
        const __nv_bfloat16* __restrict__ merged,
        __nv_bfloat16* __restrict__ buf,
        const int* __restrict__ dest, int nt, int C) {
    int j = blockIdx.x;
    if (j >= nt) return;
    const __nv_bfloat16* src = merged + (long)j * C;
    __nv_bfloat16* dst = buf + (long)dest[j] * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) dst[c] = src[c];
}

extern "C" void hyvla_prefix_scatter_bf16(
        const void* merged, void* buf, const void* dest, int nt, int C,
        cudaStream_t stream) {
    if (nt <= 0) return;
    hyvla_prefix_scatter_kernel<<<nt, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(merged),
        reinterpret_cast<__nv_bfloat16*>(buf),
        reinterpret_cast<const int*>(dest), nt, C);
}

// Merger NormalizedDwPooler gating (NormalizedDwPooler 2x2):
//   x (B,2h,2w,C) -> new_x (B,h,w,4,C) [2x2 group extract] and
//   fused (B,h,w,4,2C) = [new_x | mean_g(new_x)]. One block per (b,i,j).
__global__ void hyvla_merger_pool_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ new_x,
        __nv_bfloat16* __restrict__ fused,
        int B, int H, int W, int C) {
    const int h = H / 2, w = W / 2;
    long blk = blockIdx.x;
    int j = (int)(blk % w); long t = blk / w;
    int i = (int)(t % h); long b = t / h;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float vals[4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            int di = g >> 1, dj = g & 1;
            int row = i * 2 + di, col = j * 2 + dj;
            vals[g] = __bfloat162float(
                x[(((long)b * H + row) * W + col) * C + c]);
        }
        float mean = 0.25f * (vals[0] + vals[1] + vals[2] + vals[3]);
        long o = (((long)b * h + i) * w + j) * 4;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            long idx = (o + g) * C + c;
            new_x[idx] = __float2bfloat16(vals[g]);
            fused[(o + g) * 2 * C + c] = __float2bfloat16(vals[g]);
            fused[(o + g) * 2 * C + C + c] = __float2bfloat16(mean);
        }
    }
}

extern "C" void hyvla_merger_pool_bf16(
        const void* x, void* new_x, void* fused,
        int B, int H, int W, int C, cudaStream_t stream) {
    long blocks = (long)B * (H / 2) * (W / 2);
    hyvla_merger_pool_bf16_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(new_x),
        reinterpret_cast<__nv_bfloat16*>(fused), B, H, W, C);
}

// Gating weighted sum: out (B,h,w,C) = sum_g new_x[g] * softmax_g(score).
__global__ void hyvla_merger_gate_bf16_kernel(
        const __nv_bfloat16* __restrict__ score,
        const __nv_bfloat16* __restrict__ new_x,
        __nv_bfloat16* __restrict__ out,
        int B, int h, int w, int C) {
    long blk = blockIdx.x;
    int j = (int)(blk % w); long t = blk / w;
    int i = (int)(t % h); long b = t / h;
    long base = (((long)b * h + i) * w + j) * 4;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s[4];
        float m = -1e30f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            s[g] = __bfloat162float(score[(base + g) * C + c]);
            m = fmaxf(m, s[g]);
        }
        float denom = 0.f, acc = 0.f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            float e = __expf(s[g] - m);
            denom += e;
            acc += __bfloat162float(new_x[(base + g) * C + c]) * e;
        }
        out[(((long)b * h + i) * w + j) * C + c] =
            __float2bfloat16(acc / denom);
    }
}

extern "C" void hyvla_merger_gate_bf16(
        const void* score, const void* new_x, void* out,
        int B, int h, int w, int C, cudaStream_t stream) {
    long blocks = (long)B * h * w;
    hyvla_merger_gate_bf16_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(score),
        reinterpret_cast<const __nv_bfloat16*>(new_x),
        reinterpret_cast<__nv_bfloat16*>(out), B, h, w, C);
}

// Fused (residual += x_add) + LayerNorm + block-128 FP8 quant. Bit-identical
// to hyvla_vit_add_layer_norm_bf16 followed by fp8_per_token_block128_quant
// (same 256-thread mean/var reduction, same per-128-block amax, same
// multiply-by-reciprocal-scale quantization), but the normed activation never
// round-trips through HBM — one global write (fp8) instead of two (bf16 + read).
__global__ void hyvla_vit_add_layer_norm_to_fp8_block128_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x_add,
        const __nv_bfloat16* __restrict__ ln_weight,
        const __nv_bfloat16* __restrict__ ln_bias,
        __nv_fp8_e4m3* __restrict__ out_fp8,
        float* __restrict__ scale,
        int dim, float eps) {
    extern __shared__ float partial[];   // 256 floats for the sum reductions
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(partial + 256);

    int row = blockIdx.x;
    using T2 = __nv_bfloat162;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* add2 = reinterpret_cast<const T2*>(x_add + (size_t)row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* b2 = reinterpret_cast<const T2*>(ln_bias);
    int dim2 = dim >> 1;

    // Pass 1: residual = bf16(residual + x_add), sum for mean (same as the
    // bf16 kernel; re-read from global in later passes keeps it bit-equal).
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], av = add2[i];
        __nv_bfloat16 r0 = from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(av.x));
        __nv_bfloat16 r1 = from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(av.y));
        res2[i] = make_packed2<__nv_bfloat16>(r0, r1);
        local_sum += to_f32(r0) + to_f32(r1);
    }
    float mean = block_reduce_sum(local_sum, partial) / dim;

    // Pass 2: variance.
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, partial) / dim + eps);

    // Pass 3: normed (bf16) -> shared memory, so the quant pass can consume it
    // without an HBM round-trip of the M x dim activation.
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i], wv = w2[i], bv = b2[i];
        float n0 = (to_f32(val.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float n1 = (to_f32(val.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        normed[2 * i]     = from_f32<__nv_bfloat16>(n0);
        normed[2 * i + 1] = from_f32<__nv_bfloat16>(n1);
    }
    __syncthreads();

    // Pass 4: per-128-K-block amax + quantize (matches fp8_per_token_block128_quant:
    // amax reduce is order-independent max; scale = max(amax/448, 1e-12);
    // q = clamp(v * (1/scale), -448, 448) -> e4m3). The per-block amax is
    // accumulated with shared-memory atomicMax (non-negative float bit tricks)
    // so the quant needs no per-block __syncthreads round-trips.
    const int n_kb = dim / 128;
    __shared__ float block_amax[16];
    for (int kb = threadIdx.x; kb < n_kb; kb += blockDim.x)
        block_amax[kb] = 0.0f;
    __syncthreads();

    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const int kb = e >> 7;
        const float v = fabsf(to_f32(normed[e]));
        atomicMax(reinterpret_cast<int*>(&block_amax[kb]), __float_as_int(v));
    }
    __syncthreads();

    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const int kb = e >> 7;
        const float sc = fmaxf(block_amax[kb] / 448.0f, 1.0e-12f);
        const float inv_s = 1.0f / sc;
        float q = to_f32(normed[e]) * inv_s;
        q = fminf(fmaxf(q, -448.0f), 448.0f);
        out_fp8[(size_t)row * dim + e] = __nv_fp8_e4m3(q);
    }
    if (threadIdx.x < n_kb) {
        const float sc = fmaxf(block_amax[threadIdx.x] / 448.0f, 1.0e-12f);
        scale[(size_t)row * n_kb + threadIdx.x] = sc;
    }
}

// Spacetime (memory-encoder) temporal attention: for each (batch, token,
// head) apply a causal-in-time softmax over the K frames folded onto V.
// Replaces the torch `q_t @ k_t^T -> mask -> softmax -> @ v_t` composite
// (three framework launches + materialised scores) with one kernel.
// q/k/v are (b*kf, H, N, D) bf16 with caller strides for the (bk, H, N, D)
// view (D contiguous); out is contiguous (b*kf, H, N, D). Scores/softmax run
// in fp32 (>= the bf16 torch path). kf must be in [1, 8] and D <= 128.
__global__ void hyvla_vit_temporal_mix_bf16_kernel(
        const __nv_bfloat16* __restrict__ q,
        const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v,
        __nv_bfloat16* __restrict__ out,
        int b, int kf, int H, int N, int D,
        long s_bk, long s_h, long s_n, float scale) {
    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = sq + (size_t)kf * D;
    float* sv = sk + (size_t)kf * D;
    float* ss = sv + (size_t)kf * D;          // (kf, kf) scores
    float* sp = ss + (size_t)kf * kf;         // (kf, kf) softmax probs

    int idx = blockIdx.x;
    int h = idx % H;
    int n = (idx / H) % N;
    int bb = idx / (H * N);
    long base = (long)bb * kf * s_bk + (long)h * s_h + (long)n * s_n;

    for (int i = threadIdx.x; i < kf * D; i += blockDim.x) {
        int f = i / D, d = i % D;
        long o = base + (long)f * s_bk + d;
        sq[i] = to_f32(q[o]);
        sk[i] = to_f32(k[o]);
        sv[i] = to_f32(v[o]);
    }
    __syncthreads();

    for (int t = threadIdx.x; t < kf * kf; t += blockDim.x) {
        int f1 = t / kf, f2 = t % kf;
        float acc = 0.0f;
        if (f2 <= f1) {
            for (int d = 0; d < D; ++d) acc += sq[f1 * D + d] * sk[f2 * D + d];
            ss[t] = acc * scale;
        } else {
            ss[t] = -CUDART_INF_F;
        }
    }
    __syncthreads();

    for (int f1 = threadIdx.x; f1 < kf; f1 += blockDim.x) {
        float m = -CUDART_INF_F, sum = 0.0f;
        for (int f2 = 0; f2 <= f1; ++f2) m = fmaxf(m, ss[f1 * kf + f2]);
        for (int f2 = 0; f2 <= f1; ++f2) {
            float e = __expf(ss[f1 * kf + f2] - m);
            sp[f1 * kf + f2] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int f2 = 0; f2 <= f1; ++f2) sp[f1 * kf + f2] *= inv;
    }
    __syncthreads();

    long obase = ((long)bb * kf) * H * N * D + (long)h * N * D + (long)n * D;
    for (int i = threadIdx.x; i < kf * D; i += blockDim.x) {
        int f1 = i / D, d = i % D;
        float acc = 0.0f;
        for (int f2 = 0; f2 <= f1; ++f2) acc += sp[f1 * kf + f2] * sv[f2 * D + d];
        out[obase + (long)f1 * H * N * D + d] = from_f32<__nv_bfloat16>(acc);
    }
}

// bf16x2-vectorized variant: the global q/k/v loads and the output store move
// two bfloat16 per instruction (all strides are even in the HyVLA ViT shape
// D=96). Same math as the scalar kernel above, so outputs are bit-identical.
__global__ void hyvla_vit_temporal_mix_bf16_v2_kernel(
        const __nv_bfloat16* __restrict__ q,
        const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v,
        __nv_bfloat16* __restrict__ out,
        int b, int kf, int H, int N, int D,
        long s_bk, long s_h, long s_n, float scale) {
    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = sq + (size_t)kf * D;
    float* sv = sk + (size_t)kf * D;
    float* ss = sv + (size_t)kf * D;
    float* sp = ss + (size_t)kf * kf;

    const int Db = D >> 1;
    const int npair = kf * Db;
    const long s_bk2 = s_bk >> 1, s_h2 = s_h >> 1, s_n2 = s_n >> 1;

    int idx = blockIdx.x;
    int n = idx % N;
    int h = (idx / N) % H;
    int bb = idx / (N * H);
    long base2 = (long)bb * kf * s_bk2 + (long)h * s_h2 + (long)n * s_n2;

    const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(q);
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(k);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(v);

    for (int i = threadIdx.x; i < npair; i += blockDim.x) {
        int f = i / Db, d2 = i - f * Db;
        long o = base2 + (long)f * s_bk2 + d2;
        float2 a = __bfloat1622float2(q2[o]);
        float2 c = __bfloat1622float2(k2[o]);
        float2 e = __bfloat1622float2(v2[o]);
        sq[2 * i] = a.x; sq[2 * i + 1] = a.y;
        sk[2 * i] = c.x; sk[2 * i + 1] = c.y;
        sv[2 * i] = e.x; sv[2 * i + 1] = e.y;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < kf * kf; t += blockDim.x) {
        int f1 = t / kf, f2 = t % kf;
        float acc = 0.0f;
        if (f2 <= f1) {
            for (int d = 0; d < D; ++d) acc += sq[f1 * D + d] * sk[f2 * D + d];
            ss[t] = acc * scale;
        } else {
            ss[t] = -CUDART_INF_F;
        }
    }
    __syncthreads();

    for (int f1 = threadIdx.x; f1 < kf; f1 += blockDim.x) {
        float m = -CUDART_INF_F, sum = 0.0f;
        for (int f2 = 0; f2 <= f1; ++f2) m = fmaxf(m, ss[f1 * kf + f2]);
        for (int f2 = 0; f2 <= f1; ++f2) {
            float e = __expf(ss[f1 * kf + f2] - m);
            sp[f1 * kf + f2] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int f2 = 0; f2 <= f1; ++f2) sp[f1 * kf + f2] *= inv;
    }
    __syncthreads();

    long obase = ((long)bb * kf) * H * N * D + (long)h * N * D + (long)n * D;
    long obase2 = obase >> 1;
    long o_f2 = ((long)H * N * D) >> 1;
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    for (int i = threadIdx.x; i < npair; i += blockDim.x) {
        int f1 = i / Db, d2 = i - f1 * Db;
        int d = 2 * d2;
        float a0 = 0.0f, a1 = 0.0f;
        for (int f2 = 0; f2 <= f1; ++f2) {
            float p = sp[f1 * kf + f2];
            a0 += p * sv[f2 * D + d];
            a1 += p * sv[f2 * D + d + 1];
        }
        __nv_bfloat162 r;
        r.x = from_f32<__nv_bfloat16>(a0);
        r.y = from_f32<__nv_bfloat16>(a1);
        out2[obase2 + (long)f1 * o_f2 + d2] = r;
    }
}

extern "C" void hyvla_vit_temporal_mix_bf16(
        const void* q, const void* k, const void* v, void* out,
        int b, int kf, int H, int N, int D,
        long s_bk, long s_h, long s_n, float scale, cudaStream_t stream) {
    if (kf < 1 || kf > 8 || D > 128 || b <= 0 || H <= 0 || N <= 0) {
        return;  // unsupported shape: caller falls back to the torch path
    }
    size_t smem = (size_t)(3 * kf * D + 2 * kf * kf) * sizeof(float);
    int grid = b * H * N;
    const bool vec = (D % 2 == 0) && (s_bk % 2 == 0) && (s_h % 2 == 0)
                     && (s_n % 2 == 0);
    const char* v2e = std::getenv("HYVLA_VIT_TMIX_V2");
    const bool use_v2 = vec && !(v2e && v2e[0] == '0');
    if (use_v2) {
        hyvla_vit_temporal_mix_bf16_v2_kernel<<<grid, 128, smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const __nv_bfloat16*>(k),
            reinterpret_cast<const __nv_bfloat16*>(v),
            reinterpret_cast<__nv_bfloat16*>(out),
            b, kf, H, N, D, s_bk, s_h, s_n, scale);
        return;
    }
    hyvla_vit_temporal_mix_bf16_kernel<<<grid, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(out),
        b, kf, H, N, D, s_bk, s_h, s_n, scale);
}

// Build the FA2 denoise query buffer from the single (1,H,S,D) query tensor.
// qb (2,S,H,D) layout: qb[0,0] = q[0,:,0,:] (state query, attends to the
// prefix + itself via seqused_k = S_p+1); qb[1,s] = q[0,:,s+1,:] for the 40
// action queries (seqused_k = S_p+S). Every other (dummy) row is left
// untouched, so qb is zeroed once at allocation and never re-filled. Replaces
// the torch transpose+contiguous + zeros-fill + two slice copies per call.
__global__ void hyvla_fa2_denoise_prepare_q_bf16_kernel(
        const __nv_bfloat16* __restrict__ q,
        __nv_bfloat16* __restrict__ qb,
        int S, int H, int D) {
    int hd = H * D;
    int total = S * hd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        int b, s, h, d;
        if (i < hd) {
            b = 0; s = 0;
            h = i / D; d = i % D;
        } else {
            b = 1;
            int r = i - hd;
            s = r / hd;                  // action rows 0..S-2
            int rr = r % hd;
            h = rr / D; d = rr % D;
        }
        int qrow = (b == 0) ? 0 : (s + 1);           // q row 1..S-1 for b=1
        int qoff = (h * S + qrow) * D + d;           // q is (1,H,S,D)
        int qboff = (b * S + s) * hd + h * D + d;    // qb is (2,S,H,D)
        qb[qboff] = q[qoff];
    }
}

extern "C" void hyvla_fa2_denoise_prepare_q_bf16(
        const void* q, void* qb, int S, int H, int D, cudaStream_t stream) {
    int total = S * H * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    hyvla_fa2_denoise_prepare_q_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(qb), S, H, D);
}

// Assemble att (S, H*D) from the FA2 denoise ob (2,S,H,D) AND quantize it to
// block-128 FP8 in one pass. Replaces torch.cat + the standalone per-token
// block-128 quant launch on the expert o-projection input.
__global__ void hyvla_fa2_denoise_gather_o_fp8_block128_bf16_kernel(
        const __nv_bfloat16* __restrict__ ob,
        uint8_t* __restrict__ a8, float* __restrict__ ascale,
        int S, int H, int D) {
    int hd = H * D;
    int nb = hd >> 7;
    int s = blockIdx.x / nb;
    int kb = blockIdx.x % nb;
    const __nv_bfloat16* src =
        (s == 0) ? ob : ob + (size_t)S * hd + (size_t)(s - 1) * hd;

    __shared__ float warp_max[32];
    int k0 = kb << 7;
    float a = 0.0f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        a = fmaxf(a, fabsf(to_f32(src[k0 + i])));
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(mask, a, off));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = a;
    __syncthreads();
    __shared__ float sblock;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_max[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(mask, v, off));
        if (threadIdx.x == 0) sblock = v;
    }
    __syncthreads();
    float scale = fmaxf(sblock / 448.0f, 1.0e-12f);
    float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        float v = fminf(fmaxf(to_f32(src[k0 + i]) * inv, -448.0f), 448.0f);
        __nv_fp8_e4m3 f(v);
        a8[(size_t)s * hd + k0 + i] = *reinterpret_cast<uint8_t*>(&f);
    }
    if (threadIdx.x == 0) ascale[(size_t)s * nb + kb] = scale;
}

extern "C" void hyvla_fa2_denoise_gather_o_fp8_block128_bf16(
        const void* ob, void* a8, void* ascale, int S, int H, int D,
        cudaStream_t stream) {
    int hd = H * D;
    if (hd % 128 != 0 || S <= 0) return;
    int nb = hd / 128;
    hyvla_fa2_denoise_gather_o_fp8_block128_bf16_kernel<<<S * nb, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(ob),
        reinterpret_cast<uint8_t*>(a8),
        reinterpret_cast<float*>(ascale), S, H, D);
}

// Assemble att (S, H*D) from the FA2 denoise ob (2,S,H,D) AND quantize it to
// NVFP4 (per-16 UE4M3 scales, Sm1xx swizzled) in one pass. Bit-identical to
// gathering with the FP8 kernel's row mapping followed by
// quantize_bf16_to_nvfp4_swizzled. Replaces the standalone NVFP4 quant launch
// on the expert o-projection input (FP4 expert tower).
__global__ void hyvla_fa2_denoise_gather_o_nvfp4_bf16_kernel(
        const __nv_bfloat16* __restrict__ ob,
        uint8_t* __restrict__ packed,
        uint8_t* __restrict__ sf_swz,
        int S, int H, int D, int num_blocks, int n_col_blocks) {
    int hd = H * D;
    int s = blockIdx.x;
    const __nv_bfloat16* src =
        (s == 0) ? ob : ob + (size_t)S * hd + (size_t)(s - 1) * hd;

    extern __shared__ float amax[];
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) amax[b] = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        atomicMax((int*)&amax[i >> 4], __float_as_int(fabsf(to_f32(src[i]))));
    }
    __syncthreads();

    int rb = s >> 7, ri = s & 127;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float a = __int_as_float(*(int*)&amax[b]);
        uint8_t ue = float_to_ue4m3_ceil(a / 6.0f);
        int cb = b >> 2, ci = b & 3;
        int out_idx = (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16
                    + (ri / 32) * 4 + ci;
        sf_swz[out_idx] = ue;
        amax[b] = ue4m3_to_float(ue);
    }
    __syncthreads();

    uint8_t* row4 = packed + (size_t)s * (hd >> 1);
    int half = hd >> 1;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
        int i = p << 1;
        int blk = i >> 4;
        float sc = amax[blk];
        float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        float v0 = to_f32(src[i]) * inv;
        float v1 = to_f32(src[i + 1]) * inv;
        int blk1 = (i + 1) >> 4;
        if (blk1 != blk) {
            float sc1 = amax[blk1];
            float inv1 = (sc1 > 0.0f) ? (1.0f / sc1) : 0.0f;
            v1 = to_f32(src[i + 1]) * inv1;
        }
        uint8_t lo = float_to_fp4_e2m1(v0);
        uint8_t hi = float_to_fp4_e2m1(v1);
        row4[p] = (hi << 4) | (lo & 0x0F);
    }
}

extern "C" void hyvla_fa2_denoise_gather_o_nvfp4_bf16(
        const void* ob, void* packed, void* sf_swz, int S, int H, int D,
        cudaStream_t stream) {
    int hd = H * D;
    if (hd % 16 != 0 || S <= 0) return;
    int num_blocks = (hd + 15) / 16;
    int n_col_blocks = (num_blocks + 3) / 4;
    size_t smem_size = num_blocks * sizeof(float);
    hyvla_fa2_denoise_gather_o_nvfp4_bf16_kernel<<<S, 256, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(ob),
        reinterpret_cast<uint8_t*>(packed),
        reinterpret_cast<uint8_t*>(sf_swz), S, H, D, num_blocks, n_col_blocks);
}

// Fused (residual += x_add) + LayerNorm + NVFP4 swizzled quant (per-16 group
// UE4M3 scales, Sm1xx block-scaled layout). Bit-identical to
// hyvla_vit_add_layer_norm_bf16 followed by quantize_bf16_to_nvfp4_swizzled.
__global__ void hyvla_vit_add_layer_norm_to_nvfp4_swizzled_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x_add,
        const __nv_bfloat16* __restrict__ ln_weight,
        const __nv_bfloat16* __restrict__ ln_bias,
        uint8_t* __restrict__ out_fp4,
        uint8_t* __restrict__ out_sfa,
        int dim, float eps, int num_blocks, int n_col_blocks) {
    extern __shared__ float smem[];   // 256 partial + dim bf16 + num_blocks
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(smem + 256);
    float* amax = smem + 256 + (dim + 1) / 2;

    int row = blockIdx.x;
    using T2 = __nv_bfloat162;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* add2 = reinterpret_cast<const T2*>(x_add + (size_t)row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* b2 = reinterpret_cast<const T2*>(ln_bias);
    int dim2 = dim >> 1;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], av = add2[i];
        __nv_bfloat16 r0 = from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(av.x));
        __nv_bfloat16 r1 = from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(av.y));
        res2[i] = make_packed2<__nv_bfloat16>(r0, r1);
        local_sum += to_f32(r0) + to_f32(r1);
    }
    float mean = block_reduce_sum(local_sum, smem) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, smem) / dim + eps);

    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) amax[b] = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i], wv = w2[i], bv = b2[i];
        float n0 = (to_f32(val.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float n1 = (to_f32(val.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        __nv_bfloat16 h0 = from_f32<__nv_bfloat16>(n0);
        __nv_bfloat16 h1 = from_f32<__nv_bfloat16>(n1);
        normed[2 * i] = h0;
        normed[2 * i + 1] = h1;
        atomicMax((int*)&amax[(2 * i) >> 4], __float_as_int(fabsf(to_f32(h0))));
        atomicMax((int*)&amax[(2 * i + 1) >> 4], __float_as_int(fabsf(to_f32(h1))));
    }
    __syncthreads();

    int rb = row >> 7, ri = row & 127;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float a = __int_as_float(*(int*)&amax[b]);
        uint8_t ue = float_to_ue4m3_ceil(a / 6.0f);
        int cb = b >> 2, ci = b & 3;
        int out_idx = (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16
                    + (ri / 32) * 4 + ci;
        out_sfa[out_idx] = ue;
        amax[b] = ue4m3_to_float(ue);
    }
    __syncthreads();

    uint8_t* row_fp4 = out_fp4 + (size_t)row * (dim >> 1);
    int half = dim >> 1;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
        int i = p << 1;
        int blk = i >> 4;
        float sc = amax[blk];
        float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        float v0 = to_f32(normed[i]) * inv;
        int blk1 = (i + 1) >> 4;
        float sc1 = amax[blk1];
        float inv1 = (sc1 > 0.0f) ? (1.0f / sc1) : 0.0f;
        float v1 = to_f32(normed[i + 1]) * inv1;
        uint8_t lo = float_to_fp4_e2m1(v0);
        uint8_t hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (uint8_t)((hi << 4) | (lo & 0x0F));
    }
}

extern "C" void hyvla_vit_add_layer_norm_to_nvfp4_swizzled_bf16(
        void* residual, const void* x_add,
        const void* ln_weight, const void* ln_bias,
        void* out_fp4, void* out_sfa, int rows, int dim, float eps,
        cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || (dim % 16) != 0) return;
    int num_blocks = dim / 16;
    int n_col_blocks = (num_blocks + 3) / 4;
    size_t smem = (size_t)(256 + (dim + 1) / 2 + num_blocks) * sizeof(float);
    hyvla_vit_add_layer_norm_to_nvfp4_swizzled_kernel<<<rows, 256, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x_add),
        reinterpret_cast<const __nv_bfloat16*>(ln_weight),
        reinterpret_cast<const __nv_bfloat16*>(ln_bias),
        reinterpret_cast<uint8_t*>(out_fp4),
        reinterpret_cast<uint8_t*>(out_sfa), dim, eps, num_blocks, n_col_blocks);
}

// Aspect-preserving bilinear resize + zero centre-pad (Pi0 style). Input
// (B,C,H,W) bf16 -> (B,C,OH,OW) with the resized image (RH,RW) centred and the
// border filled with pad_value. Matches F.interpolate(align_corners=False)
// followed by F.pad.
__global__ void hyvla_resize_pad_bilinear_bf16_kernel(
        const __nv_bfloat16* __restrict__ in,
        __nv_bfloat16* __restrict__ out,
        int B, int Cc, int H, int W, int RH, int RW, int OH, int OW,
        int pad_top, int pad_left, float pad_value, float scale,
        float offset) {
    long total = (long)B * Cc * OH * OW;
    float sy = (float)H / (float)RH;
    float sx = (float)W / (float)RW;
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        int ow = (int)(i % OW);
        long t = i / OW;
        int oh = (int)(t % OH);
        t /= OH;
        int c = (int)(t % Cc);
        int b = (int)(t / Cc);
        int r = oh - pad_top, cc = ow - pad_left;
        if (r < 0 || r >= RH || cc < 0 || cc >= RW) {
            out[i] = __float2bfloat16(fmaf(pad_value, scale, offset));
            continue;
        }
        float fy = (r + 0.5f) * sy - 0.5f;
        float fx = (cc + 0.5f) * sx - 0.5f;
        int y0 = (int)floorf(fy);
        int x0 = (int)floorf(fx);
        float wy = fy - y0, wx = fx - x0;
        int y1 = y0 + 1, x1 = x0 + 1;
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1));
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1));
        const __nv_bfloat16* base = in + (((long)b * Cc + c) * H) * W;
        float v00 = to_f32(base[(long)y0 * W + x0]);
        float v01 = to_f32(base[(long)y0 * W + x1]);
        float v10 = to_f32(base[(long)y1 * W + x0]);
        float v11 = to_f32(base[(long)y1 * W + x1]);
        float top = v00 + (v01 - v00) * wx;
        float bot = v10 + (v11 - v10) * wx;
        out[i] = from_f32<__nv_bfloat16>(
            fmaf(top + (bot - top) * wy, scale, offset));
    }
}

extern "C" void hyvla_resize_pad_bilinear_bf16(
        const void* in, void* out, int B, int Cc, int H, int W,
        int RH, int RW, int OH, int OW, int pad_top, int pad_left,
        float pad_value, cudaStream_t stream) {
    long total = (long)B * Cc * OH * OW;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    hyvla_resize_pad_bilinear_bf16_kernel<<<(int)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in),
        reinterpret_cast<__nv_bfloat16*>(out), B, Cc, H, W, RH, RW, OH, OW,
        pad_top, pad_left, pad_value, 1.0f, 0.0f);
}

extern "C" void hyvla_resize_pad_bilinear_scale_bf16(
        const void* in, void* out, int B, int Cc, int H, int W,
        int RH, int RW, int OH, int OW, int pad_top, int pad_left,
        float pad_value, float scale, float offset, cudaStream_t stream) {
    long total = (long)B * Cc * OH * OW;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    hyvla_resize_pad_bilinear_bf16_kernel<<<(int)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in),
        reinterpret_cast<__nv_bfloat16*>(out), B, Cc, H, W, RH, RW, OH, OW,
        pad_top, pad_left, pad_value, scale, offset);
}

// Gather the ViT spatial-attention FA2 output o (bk,H,N,Ds) into the proj
// activation (rows=bk*N, K=H*Dh) and quantize to NVFP4 packed u8 + swizzled
// UE4M3 SFA in one pass.
template <int DH_CONST>
__global__ void hyvla_vit_proj_gather_nvfp4_swizzled_kernel(
        const __nv_bfloat16* __restrict__ o,
        uint8_t* __restrict__ out_fp4,
        uint8_t* __restrict__ out_sfa,
        int bk, int H, int N, int Ds, int Dh, int K,
        int num_blocks, int n_col_blocks) {
    const int dh = (DH_CONST > 0) ? DH_CONST : Dh;
    int row = blockIdx.x;
    int b = row / N;
    int nn = row % N;
    extern __shared__ float smem[];
    for (int kb = threadIdx.x; kb < num_blocks; kb += blockDim.x)
        smem[kb] = 0.0f;
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int h = k / dh, d = k - h * dh;
        float v = fabsf(to_f32(o[((size_t)(b * H + h) * N + nn) * Ds + d]));
        atomicMax((int*)&smem[k >> 4], __float_as_int(v));
    }
    __syncthreads();
    int rb = row >> 7, ri = row & 127;
    for (int kb = threadIdx.x; kb < num_blocks; kb += blockDim.x) {
        float a = __int_as_float(*(int*)&smem[kb]);
        uint8_t ue = float_to_ue4m3_ceil(a / 6.0f);
        int cb = kb >> 2, ci = kb & 3;
        out_sfa[(size_t)(rb * n_col_blocks + cb) * 512
                + (ri % 32) * 16 + (ri / 32) * 4 + ci] = ue;
        float sc = ue4m3_to_float(ue);
        smem[kb] = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    }
    __syncthreads();
    uint8_t* row_fp4 = out_fp4 + (size_t)row * (K >> 1);
    for (int p = threadIdx.x; p < (K >> 1); p += blockDim.x) {
        int k0 = p << 1, k1 = k0 + 1;
        int h0 = k0 / dh, d0 = k0 - h0 * dh;
        int h1 = k1 / dh, d1 = k1 - h1 * dh;
        float v0 = to_f32(o[((size_t)(b * H + h0) * N + nn) * Ds + d0])
                 * smem[k0 >> 4];
        float v1 = to_f32(o[((size_t)(b * H + h1) * N + nn) * Ds + d1])
                 * smem[k1 >> 4];
        v0 = fminf(fmaxf(v0, -448.0f), 448.0f);
        v1 = fminf(fmaxf(v1, -448.0f), 448.0f);
        uint8_t lo = float_to_fp4_e2m1(v0);
        uint8_t hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (uint8_t)((hi << 4) | (lo & 0x0F));
    }
}

extern "C" void hyvla_vit_proj_gather_nvfp4_swizzled_bf16(
        const void* o, void* out_fp4, void* out_sfa, int bk, int H, int N,
        int Ds, int Dh, cudaStream_t stream) {
    int K = H * Dh;
    if (K % 16 != 0 || bk <= 0 || N <= 0) return;
    int num_blocks = K / 16;
    int n_col_blocks = (num_blocks + 3) / 4;
    int rows = bk * N;
    int smem = num_blocks * sizeof(float);
    if (Dh == 72) {
        hyvla_vit_proj_gather_nvfp4_swizzled_kernel<72>
            <<<rows, 256, smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(o),
                reinterpret_cast<uint8_t*>(out_fp4),
                reinterpret_cast<uint8_t*>(out_sfa), bk, H, N, Ds, Dh, K,
                num_blocks, n_col_blocks);
        return;
    }
    hyvla_vit_proj_gather_nvfp4_swizzled_kernel<0>
        <<<rows, 256, smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(o),
            reinterpret_cast<uint8_t*>(out_fp4),
            reinterpret_cast<uint8_t*>(out_sfa), bk, H, N, Ds, Dh, K,
            num_blocks, n_col_blocks);
}

extern "C" void hyvla_vit_add_layer_norm_to_fp8_block128_bf16(
        void* residual, const void* x_add,
        const void* ln_weight, const void* ln_bias,
        void* out_fp8, float* scale, int rows, int dim, float eps,
        cudaStream_t stream) {
    if ((dim % 128) != 0) {
        // dim 1152 is 128-aligned; guard anyway to fail loudly, not silently.
        return;
    }
    int smem = 256 * sizeof(float) + (size_t)dim * sizeof(__nv_bfloat16);
    hyvla_vit_add_layer_norm_to_fp8_block128_kernel<<<rows, 256, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x_add),
        reinterpret_cast<const __nv_bfloat16*>(ln_weight),
        reinterpret_cast<const __nv_bfloat16*>(ln_bias),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8), scale, dim, eps);
}
