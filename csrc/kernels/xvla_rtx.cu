// FlashRT — X-VLA (soft-prompted flow-matching VLA) RTX helpers.
#include "xvla_rtx.cuh"

// The repository's own NVFP4 rounding and block reduction, so the fused
// LayerNorm-quantize below packs exactly what the entry it replaces packs.
#include "common.cuh"
#include "block_reduce_safe.cuh"
#include "nvfp4_convert.cuh"
#include "pdl_switch.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace xvla {

// ── out = a + b (elementwise, bf16) ──
// Folds a constant table (positional embeddings) into a sequence buffer.
__global__ void add2_kernel(const __nv_bfloat16* __restrict__ a,
                            const __nv_bfloat16* __restrict__ b,
                            __nv_bfloat16* __restrict__ out, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
}

void add2_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
               __nv_bfloat16* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    flash_rt::pdl::launch(add2_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, a, b, out, n);
}

// ── action-token assembly, part 1: interpolant + time broadcast ──
//
//   act_tok[r, d]   = x1[r, d] * t + prev[r, d] * (1 - t)   for d < dim
//   act_tok[r, g0]  = act_tok[r, g1] = 0                    (EE6D gripper)
//   act_tok[r, toff + k] = time[k]                          for k < time_dim
//
// The interpolation, the gripper zeroing and the per-step timestep embedding
// all land in one launch, writing straight into the transformer's token buffer.
__global__ void flow_step_kernel(const __nv_bfloat16* __restrict__ x1,
                                 const __nv_bfloat16* __restrict__ prev,
                                 __nv_bfloat16* __restrict__ act_tok,
                                 const __nv_bfloat16* __restrict__ time,
                                 float t, float one_minus_t, int dim, int stride,
                                 int g0, int g1, int toff, int time_dim,
                                 int per_row, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = idx / per_row;
    int d = idx - r * per_row;
    if (d < dim) {
        float v;
        if (d == g0 || d == g1) {
            v = 0.0f;  // gripper logits are not part of the interpolant
        } else {
            v = __bfloat162float(x1[r * dim + d]) * t
              + __bfloat162float(prev[r * dim + d]) * one_minus_t;
        }
        act_tok[r * stride + d] = __float2bfloat16(v);
    }
    if (d < time_dim) {
        act_tok[r * stride + toff + d] = time[d];
    }
}

void flow_step_bf16(const __nv_bfloat16* x1, const __nv_bfloat16* prev,
                    __nv_bfloat16* act_tok, const __nv_bfloat16* time, float t,
                    int rows, int dim, int stride, int g0, int g1, int toff,
                    int time_dim, cudaStream_t stream) {
    int per_row = dim > time_dim ? dim : time_dim;
    int total = rows * per_row;
    if (total <= 0) return;
    flash_rt::pdl::launch(flow_step_kernel, dim3((total + 255) / 256), dim3(256), 0, stream,
        x1, prev, act_tok, time, t, 1.0f - t, dim, stride, g0, g1, toff,
        time_dim, per_row, total);
}

// ── action-token assembly, part 2: broadcast the proprio vector ──
//
//   act_tok[r, poff + d] = (d < n_state && d not gripper) ? state[d] : 0
//
// `state` is the raw fp32 proprio reading; padding to the model's
// max_state_dim and the gripper zeroing both happen here.
__global__ void proprio_kernel(const float* __restrict__ state,
                               __nv_bfloat16* __restrict__ act_tok, int rows,
                               int stride, int poff, int dim_proprio,
                               int n_state, int g0, int g1, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = idx / dim_proprio;
    int d = idx - r * dim_proprio;
    float v = 0.0f;
    if (d < n_state && d != g0 && d != g1) v = state[d];
    act_tok[r * stride + poff + d] = __float2bfloat16(v);
}

void proprio_bf16(const float* state, __nv_bfloat16* act_tok, int rows,
                  int stride, int poff, int dim_proprio, int n_state, int g0,
                  int g1, cudaStream_t stream) {
    int total = rows * dim_proprio;
    if (total <= 0) return;
    flash_rt::pdl::launch(proprio_kernel, dim3((total + 255) / 256), dim3(256), 0, stream,
        state, act_tok, rows, stride, poff, dim_proprio, n_state, g0, g1, total);
}

// ── Chunk finalisation: sigmoid the gripper logits and widen to fp32 ──
__global__ void finalize_kernel(const __nv_bfloat16* __restrict__ in,
                                float* __restrict__ out, int dim, int stride,
                                int g0, int g1, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = idx / dim;
    int d = idx - r * dim;
    float v = __bfloat162float(in[r * stride + d]);
    if (d == g0 || d == g1) v = 1.0f / (1.0f + __expf(-v));
    out[idx] = v;
}

void finalize_bf16(const __nv_bfloat16* in, float* out, int rows, int dim,
                   int stride, int g0, int g1, cudaStream_t stream) {
    int total = rows * dim;
    if (total <= 0) return;
    flash_rt::pdl::launch(finalize_kernel, dim3((total + 255) / 256), dim3(256), 0, stream,
        in, out, dim, stride, g0, g1, total);
}

// ── fp16 -> bf16 cast ──
// The DaViT tower runs fp16 (its 72 sequential GEMMs are what limit action
// fidelity); the encoder and flow-matching transformer stay bf16, so the
// pooled vision tokens are narrowed once here.
__global__ void cast_f16_bf16_kernel(const __half* __restrict__ in,
                                     __nv_bfloat16* __restrict__ out, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2bfloat16(__half2float(in[idx]));
}

void cast_f16_to_bf16(const __half* in, __nv_bfloat16* out, int n,
                      cudaStream_t stream) {
    if (n <= 0) return;
    flash_rt::pdl::launch(cast_f16_bf16_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, in, out, n);
}

// ── out = in (fp16) ──
__global__ void copy_f16_kernel(const __half* __restrict__ in,
                                __half* __restrict__ out, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = in[idx];
}

void copy_f16(const __half* in, __half* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    flash_rt::pdl::launch(copy_f16_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, in, out, n);
}

// ── LayerNorm (+ optional residual add) -> NVFP4 swizzled quantize ──────────
//
// Every one of the 24 transformer blocks runs `x += sublayer(x); LayerNorm(x);
// quantize`. Keeping those as three launches costs 480 residual adds plus 480
// LayerNorm-quantizes per inference, and the repository's
// `layer_norm_to_nvfp4_swizzled_bf16` reads its input three times (mean pass,
// variance pass, normalize pass).
//
// These two kernels do it in one launch and one read of `x`: the row is staged
// in shared memory, the residual add (when present) is folded into the staging
// pass, and the reduction order is kept identical to the fused entry it
// replaces so the packed output is unchanged bit for bit.
//
// `weight`/`bias` are the LayerNorm's affine terms; `eps` uses the reference's
// variance convention (var = sum of squared deviations, divided by cols).
// Fold (optional) + LayerNorm + NVFP4 quantize. NT is the threads-per-row
// template parameter and it is not cosmetic: the LayerNorm's reduction order
// decides whether the encoder can use this kernel at all. With 128 threads the
// fused entry measured 0.09 ms faster than the three-launch chain but dropped
// the delivery gate to 0.9846 min, because the encoder is post-norm and twelve
// layers amplify a reduction-order difference into the gripper's decision
// boundary; at 256 threads, matching csrc/kernels/norm.cu's layer_norm_kernel,
// the same fusion passes. The denoise loop keeps 128, which f5054539 measured
// as its optimum. The quantize passes that follow are the ones
// quantize_bf16_to_nvfp4_swizzled uses.
//
// norm_out (optional) receives the normalised bf16 row itself; it may alias x
// because every read of x happens before the pass that writes it.
template <bool ADD, int NT>
__global__ void ln_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ h,
                                __nv_bfloat16* __restrict__ residual_out,
                                __nv_bfloat16* __restrict__ norm_out,
                                const __nv_bfloat16* __restrict__ weight,
                                const __nv_bfloat16* __restrict__ bias,
                                uint8_t* __restrict__ packed,
                                uint8_t* __restrict__ sf_swz, int cols,
                                int num_blocks, int n_col_blocks, float eps) {
    flashrt_pdl_wait_and_trigger();
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int dim2 = cols >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + (size_t)row * cols);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(h + (size_t)row * cols);
    __nv_bfloat162* post2 = residual_out
        ? reinterpret_cast<__nv_bfloat162*>(residual_out + (size_t)row * cols) : nullptr;
    __nv_bfloat162* norm2 = norm_out
        ? reinterpret_cast<__nv_bfloat162*>(norm_out + (size_t)row * cols) : nullptr;

    extern __shared__ float smem_dyn[];
    float* warp_red = smem_dyn;
    float* sf_smem = smem_dyn + 32;
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(sf_smem + num_blocks);

    // Pass 1: the (optionally folded) row, kept as bf16 so the statistics are
    // taken over exactly the values layer_norm_kernel would have read.
    //
    // The row is read with an explicit 4-deep prefetch: the trip count is
    // runtime (dim2 / blockDim.x), so the compiler cannot unroll the loop by
    // itself, and the delivered form issued each iteration's x/h loads only
    // after the previous iteration's smem stores. At the denoise fold site
    // (cols=1024, NT=128, exactly 4 iterations) that left the staging pass
    // latency-bound at ~261 GB/s; the prefetch issues all eight loads of the
    // four iterations before consuming any of them. The per-thread order of
    // `local_sum` additions is unchanged, so the reduction tree -- and the
    // packed output -- stay bit-identical.
    constexpr int kPrefetch = 4;
    float local_sum = 0.0f;
    int i = t;
    for (; i + (kPrefetch - 1) * (int)blockDim.x < dim2; i += kPrefetch * (int)blockDim.x) {
        __nv_bfloat162 xs[kPrefetch];
        __nv_bfloat162 hs[kPrefetch];
        #pragma unroll
        for (int k = 0; k < kPrefetch; ++k) {
            xs[k] = x2[i + k * (int)blockDim.x];
        }
        if (ADD) {
            #pragma unroll
            for (int k = 0; k < kPrefetch; ++k) {
                hs[k] = h2[i + k * (int)blockDim.x];
            }
        }
        #pragma unroll
        for (int k = 0; k < kPrefetch; ++k) {
            const int idx = i + k * (int)blockDim.x;
            __nv_bfloat162 xi = xs[k];
            if (ADD) {
                __nv_bfloat162 hi = hs[k];
                xi.x = __float2bfloat16(__bfloat162float(xi.x) + __bfloat162float(hi.x));
                xi.y = __float2bfloat16(__bfloat162float(xi.y) + __bfloat162float(hi.y));
                if (post2 != nullptr) { post2[idx] = xi; }
            }
            reinterpret_cast<__nv_bfloat162*>(normed)[idx] = xi;
            local_sum += __bfloat162float(xi.x) + __bfloat162float(xi.y);
        }
    }
    for (; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xi = x2[i];
        if (ADD) {
            __nv_bfloat162 hi = h2[i];
            xi.x = __float2bfloat16(__bfloat162float(xi.x) + __bfloat162float(hi.x));
            xi.y = __float2bfloat16(__bfloat162float(xi.y) + __bfloat162float(hi.y));
            if (post2 != nullptr) { post2[i] = xi; }
        }
        reinterpret_cast<__nv_bfloat162*>(normed)[i] = xi;
        local_sum += __bfloat162float(xi.x) + __bfloat162float(xi.y);
    }
    const float mean = block_reduce_sum_safe(local_sum, warp_red) / cols;

    // Pass 2: variance about that mean, same accumulation shape.
    float local_var = 0.0f;
    for (int i = t; i < dim2; i += blockDim.x) {
        float d0 = __bfloat162float(normed[2 * i]) - mean;
        float d1 = __bfloat162float(normed[2 * i + 1]) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    const float inv_std = rsqrtf(block_reduce_sum_safe(local_var, warp_red) / cols + eps);

    for (int b = t; b < num_blocks; b += blockDim.x) sf_smem[b] = 0.f;
    __syncthreads();

    // Pass 3: affine (bf16 rounding, as layer_norm_kernel does) + per-16 amax.
    for (int i = t; i < dim2; i += blockDim.x) {
        __nv_bfloat162 wv = reinterpret_cast<const __nv_bfloat162*>(weight)[i];
        __nv_bfloat162 bv = reinterpret_cast<const __nv_bfloat162*>(bias)[i];
        float v0 = (__bfloat162float(normed[2 * i]) - mean) * inv_std *
                       __bfloat162float(wv.x) + __bfloat162float(bv.x);
        float v1 = (__bfloat162float(normed[2 * i + 1]) - mean) * inv_std *
                       __bfloat162float(wv.y) + __bfloat162float(bv.y);
        __nv_bfloat16 n0 = __float2bfloat16(v0), n1 = __float2bfloat16(v1);
        reinterpret_cast<__nv_bfloat162*>(normed)[i] = __nv_bfloat162(n0, n1);
        if (norm2 != nullptr) { norm2[i] = __nv_bfloat162(n0, n1); }
        atomicMax((int*)&sf_smem[(2 * i) >> 4], __float_as_int(fabsf(__bfloat162float(n0))));
        atomicMax((int*)&sf_smem[(2 * i + 1) >> 4], __float_as_int(fabsf(__bfloat162float(n1))));
    }
    __syncthreads();

    const int rb = row / 128, ri = row % 128;
    for (int b = t; b < num_blocks; b += blockDim.x) {
        uint8_t ue = float_to_ue4m3_ceil(__int_as_float(*(int*)&sf_smem[b]) / 6.0f);
        sf_swz[(rb * n_col_blocks + (b >> 2)) * 512 + (ri % 32) * 16 +
               (ri / 32) * 4 + (b & 3)] = ue;
        sf_smem[b] = ue4m3_to_float(ue);
    }
    __syncthreads();

    uint8_t* row_fp4 = packed + (size_t)row * (cols >> 1);
    const int npairs = cols >> 1;
    int p = t;
    // Four pairs per thread, one 32-bit store: the delivered form issued one
    // byte store (and two shared-memory scale reads) per pair, four serialized
    // rounds at the denoise shape. The packed bytes are computed by the same
    // expressions from the same smem values, so the output is byte-identical;
    // only which thread writes which byte changes. Guarded on a full-width
    // grid so the narrow tower shapes keep the fully-parallel scalar mapping.
    if (npairs >= 4 * (int)blockDim.x) {
        const __nv_bfloat162* n2 = reinterpret_cast<const __nv_bfloat162*>(normed);
        const int vec_lim = npairs & ~3;
        for (p = t * 4; p + 3 < vec_lim; p += 4 * (int)blockDim.x) {
            uint32_t word = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int pp = p + k;
                const int i = pp << 1;
                float s0 = sf_smem[i >> 4], s1 = sf_smem[(i + 1) >> 4];
                const __nv_bfloat162 nv = n2[pp];
                float v0 = __bfloat162float(nv.x) * (s0 > 0.f ? 1.f / s0 : 0.f);
                float v1 = __bfloat162float(nv.y) * (s1 > 0.f ? 1.f / s1 : 0.f);
                // The branchless encoder is bit-identical to float_to_fp4_e2m1
                // (same thresholds, same result) but the branchy form diverges
                // across a warp and serializes.  Measured on the denoise shape:
                // 5.58 -> 5.33 us eager, byte-identical output and scales.
                const uint8_t b =
                    (uint8_t)((float_to_fp4_e2m1_branchless(v1) << 4) |
                              (float_to_fp4_e2m1_branchless(v0) & 0x0F));
                word |= (uint32_t)b << (8 * k);
            }
            *reinterpret_cast<uint32_t*>(row_fp4 + p) = word;
        }
        p = vec_lim + t;
    }
    for (; p < npairs; p += blockDim.x) {
        int i = p << 1;
        float s0 = sf_smem[i >> 4], s1 = sf_smem[(i + 1) >> 4];
        float v0 = __bfloat162float(normed[i]) * (s0 > 0.f ? 1.f / s0 : 0.f);
        float v1 = __bfloat162float(normed[i + 1]) * (s1 > 0.f ? 1.f / s1 : 0.f);
        row_fp4[p] = (uint8_t)((float_to_fp4_e2m1_branchless(v1) << 4) |
                               (float_to_fp4_e2m1_branchless(v0) & 0x0F));
    }
}

// LayerNorm that emits the normalised bf16 row and nothing else.
//
// The DaViT conv-embed norms hand bf16 to the depthwise convolution, so they
// cannot use the fused norm+quantize entry. The repository's layer_norm_kernel
// runs one block per row with NT threads and one element per thread, which at
// the tower's dim <= 256 shapes (dim2 <= 128) leaves half the block idle while
// both halves pay for the two block reductions -- the same asymmetry that cost
// the fused entry 1.62x before it was fixed. This is the fused kernel's
// arithmetic verbatim (same staging, same block_reduce_sum, same bf16 rounding
// of the affine result, same variance convention) minus the quantize, so at
// dim2 <= NT it reproduces layer_norm_kernel bit for bit for the same reason
// the fused entry does: the idle warps contribute exactly 0.0f and p + 0.0f == p
// is exact, leaving the tree unchanged.
template <int NT>
__global__ void ln_only_kernel(const __nv_bfloat16* __restrict__ x,
                               __nv_bfloat16* __restrict__ out,
                               const __nv_bfloat16* __restrict__ weight,
                               const __nv_bfloat16* __restrict__ bias,
                               int cols, float eps) {
    flashrt_pdl_wait_and_trigger();
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int dim2 = cols >> 1;
    const __nv_bfloat162* x2 =
        reinterpret_cast<const __nv_bfloat162*>(x + (size_t)row * cols);
    __nv_bfloat162* o2 =
        reinterpret_cast<__nv_bfloat162*>(out + (size_t)row * cols);

    extern __shared__ float smem_dyn[];
    float* warp_red = smem_dyn;
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(smem_dyn + 32);

    float local_sum = 0.0f;
    for (int i = t; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xi = x2[i];
        normed[2 * i] = xi.x;
        normed[2 * i + 1] = xi.y;
        local_sum += __bfloat162float(xi.x) + __bfloat162float(xi.y);
    }
    const float mean = block_reduce_sum_safe(local_sum, warp_red) / cols;

    float local_var = 0.0f;
    for (int i = t; i < dim2; i += blockDim.x) {
        float d0 = __bfloat162float(normed[2 * i]) - mean;
        float d1 = __bfloat162float(normed[2 * i + 1]) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    const float inv_std =
        rsqrtf(block_reduce_sum_safe(local_var, warp_red) / cols + eps);

    for (int i = t; i < dim2; i += blockDim.x) {
        __nv_bfloat162 wv = reinterpret_cast<const __nv_bfloat162*>(weight)[i];
        __nv_bfloat162 bv = reinterpret_cast<const __nv_bfloat162*>(bias)[i];
        float v0 = (__bfloat162float(normed[2 * i]) - mean) * inv_std *
                       __bfloat162float(wv.x) + __bfloat162float(bv.x);
        float v1 = (__bfloat162float(normed[2 * i + 1]) - mean) * inv_std *
                       __bfloat162float(wv.y) + __bfloat162float(bv.y);
        o2[i] = __nv_bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
    }
}

void ln_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
             const __nv_bfloat16* w, const __nv_bfloat16* b, int rows, int cols,
             float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    size_t smem = 32 * sizeof(float) + (size_t)cols * 2;
    // NT=128 whenever the live half of the row fits inside it, which is exactly
    // when the 128- and 256-thread trees are the same expression.
    if ((cols >> 1) <= 128) {
        flash_rt::pdl::launch(ln_only_kernel<128>, dim3(rows), dim3(128), smem, stream, x, out, w, b, cols, eps);
        return;
    }
    flash_rt::pdl::launch(ln_only_kernel<256>, dim3(rows), dim3(256), smem, stream, x, out, w, b, cols, eps);
}

template <bool ADD, int NT = 128>
static void ln_quant_launch(const __nv_bfloat16* x, const __nv_bfloat16* h,
                            __nv_bfloat16* residual_out,
                            const __nv_bfloat16* w, const __nv_bfloat16* b,
                            uint8_t* packed, uint8_t* sf_swz, int rows, int cols,
                            float eps, cudaStream_t stream) {
    int num_blocks = (cols + 15) / 16;
    size_t smem = (size_t)(32 + num_blocks) * sizeof(float) + (size_t)cols * 2;
    // NT is not cosmetic: the LayerNorm's reduction order is what decides
    // whether a consumer's numerics survive the delivery gate. 256 matches
    // csrc/kernels/norm.cu's layer_norm_kernel, which is what the encoder's
    // post-norm stream and the DaViT tower's norms need; 128 is the tuned
    // optimum for the denoise loop's shapes (measured at f5054539).
    flash_rt::pdl::launch(ln_quant_kernel<ADD, NT>, dim3(rows), dim3(NT), smem,
                          stream, x, h, residual_out,
                          static_cast<__nv_bfloat16*>(nullptr), w, b, packed,
                          sf_swz, cols, num_blocks, (num_blocks + 3) / 4, eps);
}

void ln_quant_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                   const __nv_bfloat16* b, uint8_t* packed, uint8_t* sf_swz,
                   int rows, int cols, float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    ln_quant_launch<false>((const __nv_bfloat16*)x, nullptr, (__nv_bfloat16*)x, w, b,
                           packed, sf_swz, rows, cols, eps, stream);
}

void add_ln_quant_bf16(__nv_bfloat16* residual, const __nv_bfloat16* h,
                       const __nv_bfloat16* w, const __nv_bfloat16* b,
                       uint8_t* packed, uint8_t* sf_swz, int rows, int cols,
                       float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    ln_quant_launch<true>(residual, h, residual, w, b, packed, sf_swz, rows, cols,
                          eps, stream);
}

// No fold, 256 threads: the same reduction order as layer_norm_kernel, fused
// with the NVFP4 quantize. The DaViT tower's two norms use this so that fusing
// them into the GEMM operand does not move the delivery gate.
//
// At cols <= 256 the block is half idle: dim2 is at most 128 pairs, so only
// the first 128 threads read `x` at all, and the idle warps contribute exactly
// 0.0f to every partial. `p + 0.0f == p` is exact, so the eight-warp tree
// that 256 threads build out of four live and four zero warp partials is the
// same expression the four-warp tree builds directly, and a 128-thread launch
// is byte-identical here. That is the DaViT stage 0 shape (6272 rows, 256
// columns), where halving the block measures 1.58x on the kernel and 0.13 ms
// over the equal-scope pipeline, with the packed activation and the swizzled
// scale factors unchanged. Above 256 columns each thread accumulates more than
// one pair, the order genuinely changes, and the 256-thread launch is what
// layer_norm_kernel compatibility requires.
void ln_quant_bf16_nt256(const __nv_bfloat16* x, const __nv_bfloat16* w,
                         const __nv_bfloat16* b, uint8_t* packed,
                         uint8_t* sf_swz, int rows, int cols, float eps,
                         cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols <= 256) {
        ln_quant_launch<false, 128>(x, nullptr, nullptr, w, b, packed, sf_swz,
                                    rows, cols, eps, stream);
        return;
    }
    ln_quant_launch<false, 256>(x, nullptr, nullptr, w, b, packed, sf_swz, rows,
                                cols, eps, stream);
}

void add_ln_quant_norm_bf16(__nv_bfloat16* residual, const __nv_bfloat16* h,
                            __nv_bfloat16* norm_out, const __nv_bfloat16* w,
                            const __nv_bfloat16* b, uint8_t* packed,
                            uint8_t* sf_swz, int rows, int cols, float eps,
                            cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int num_blocks = (cols + 15) / 16;
    size_t smem = (size_t)(32 + num_blocks) * sizeof(float) + (size_t)cols * 2;
    flash_rt::pdl::launch(ln_quant_kernel<true, 256>, dim3(rows), dim3(256), smem,
                          stream, residual, h, residual, norm_out, w, b, packed,
                          sf_swz, cols, num_blocks, (num_blocks + 3) / 4, eps);
}

// ── out = in (bf16) ──
__global__ void copy_kernel(const __nv_bfloat16* __restrict__ in,
                            __nv_bfloat16* __restrict__ out, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = in[idx];
}

void copy_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int n,
               cudaStream_t stream) {
    if (n <= 0) return;
    flash_rt::pdl::launch(copy_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, in, out, n);
}

// ── action decoder: one warp per (row, output) dot product ──
// out[m,n] = bias[n] + sum_k x[m,k] * w[n,k], fp32 accumulation, warp tree.
__global__ void action_decoder_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
    flashrt_pdl_wait_and_trigger();
    int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int mn = wid;
    if (mn >= M * N) return;
    int m = mn / N;
    int n = mn - m * N;
    const __nv_bfloat16* xr = x + (size_t)m * K;
    const __nv_bfloat16* wr = w + (size_t)n * K;
    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += 64) {
        __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(xr + k);
        __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162*>(wr + k);
        acc = fmaf(__bfloat162float(xv.x), __bfloat162float(wv.x), acc);
        acc = fmaf(__bfloat162float(xv.y), __bfloat162float(wv.y), acc);
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        out[(size_t)m * N + n] = __float2bfloat16(acc + __bfloat162float(bias[n]));
}

void action_decoder_gemv_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                              const __nv_bfloat16* bias, __nv_bfloat16* out,
                              int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int warps = M * N;
    int blocks = (warps + 3) / 4;
    flash_rt::pdl::launch(action_decoder_gemv_kernel, dim3(blocks), dim3(128), 0, stream, x, w, bias, out, M, N, K);
}

}  // namespace xvla
}  // namespace flash_rt
