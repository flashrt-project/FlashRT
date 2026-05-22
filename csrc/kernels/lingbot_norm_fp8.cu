#include "lingbot_common.cuh"

__device__ __forceinline__ float lingbot_block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    return smem[0];
}

__global__ void lingbot_ada_rms_residual_fp8_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,    // [B, S, D] — mutated to residual+x
    const __nv_bfloat16* __restrict__ x,     // [B, S, D]
    const __nv_bfloat16* __restrict__ rms_weight,  // [D]
    const __nv_bfloat16* __restrict__ gamma, // [B, D]
    const __nv_bfloat16* __restrict__ beta,  // [B, D]
    __nv_fp8_e4m3* __restrict__ out_fp8,     // [B, S, D]
    const float* __restrict__ act_scale,     // [1] scalar
    int B, int S, int D, float eps)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int row_offset = (b * S + s) * D;
    int bd_offset = b * D;

    extern __shared__ float smem[];

    // Pass 1: y = residual + x, accumulate sum of squares.
    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float r = __bfloat162float(residual[row_offset + d]);
        float xv = __bfloat162float(x[row_offset + d]);
        float y = r + xv;
        residual[row_offset + d] = __float2bfloat16_rn(y);
        local_sum_sq += y * y;
    }

    float total_sum_sq = lingbot_block_reduce_sum(local_sum_sq, smem);
    float rsqrt_var = rsqrtf(total_sum_sq / (float)D + eps);
    float inv_scale = 1.0f / __ldg(act_scale);

    // Pass 2: norm + FiLM + FP8 quantize. residual now holds y.
    // Round through bf16 between the FiLM result and the FP8 quantize
    // to MATCH the eager `ada_rms_norm` → `linear_fp8`-internal-quantize
    // pipeline exactly. Without this intermediate cast our output is
    // slightly fresher (fewer rounding steps) but drifts from the
    // bf16-trained baseline by ~0.005 cos.
    for (int d = tid; d < D; d += blockDim.x) {
        float y = __bfloat162float(residual[row_offset + d]);
        float w = __bfloat162float(rms_weight[d]);
        float g = __bfloat162float(gamma[bd_offset + d]);
        float bt = __bfloat162float(beta[bd_offset + d]);
        float norm_v = y * rsqrt_var * w;
        float film_v = (1.0f + g) * norm_v + bt;
        float film_bf16 = __bfloat162float(__float2bfloat16_rn(film_v));
        float fp8_v = film_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[row_offset + d] = __nv_fp8_e4m3(fp8_v);
    }
}

// =============================================================================
//  Pybind wrappers (raw pointer entry; LingBot kernel_ops.py owns lifetime)
// =============================================================================

void lingbot_ada_rms_residual_fp8_bf16(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, float eps, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(S, B);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = BLOCK_THREADS * sizeof(float);
    lingbot_ada_rms_residual_fp8_bf16_kernel<<<grid, block, smem_bytes, to_stream(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(rms_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        B, S, D, eps
    );
}

// =============================================================================
//  Kernel: per-sample γ/β AdaRMSNorm + FP8 (no residual)
//  Used by the Expert ``input_layernorm`` site (the input to Q/K/V
//  projections). Same math as ada_rms_residual_fp8 but skips the
//  residual+x step — hidden state already holds the layer input.
// =============================================================================
__global__ void lingbot_ada_rms_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,     // [B, S, D] read-only
    const __nv_bfloat16* __restrict__ rms_weight,
    const __nv_bfloat16* __restrict__ gamma, // [B, D]
    const __nv_bfloat16* __restrict__ beta,  // [B, D]
    __nv_fp8_e4m3* __restrict__ out_fp8,
    const float* __restrict__ act_scale,
    int B, int S, int D, float eps)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int row_offset = (b * S + s) * D;
    int bd_offset = b * D;

    extern __shared__ float smem[];

    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[row_offset + d]);
        local_sum_sq += v * v;
    }
    float total_sum_sq = lingbot_block_reduce_sum(local_sum_sq, smem);
    float rsqrt_var = rsqrtf(total_sum_sq / (float)D + eps);
    float inv_scale = 1.0f / __ldg(act_scale);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[row_offset + d]);
        float w = __bfloat162float(rms_weight[d]);
        float g = __bfloat162float(gamma[bd_offset + d]);
        float bt = __bfloat162float(beta[bd_offset + d]);
        float norm_v = v * rsqrt_var * w;
        float film_v = (1.0f + g) * norm_v + bt;
        // Round through bf16 to match eager's ada_rms_norm→linear_fp8
        // quantize pipeline (see ada_rms_residual variant for rationale).
        float film_bf16 = __bfloat162float(__float2bfloat16_rn(film_v));
        float fp8_v = film_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[row_offset + d] = __nv_fp8_e4m3(fp8_v);
    }
}

// =============================================================================
//  M-padded variants: identical math but write a [B, PADM, D] FP8 output
//  (PADM >= S) with rows [S, PADM) zero-filled, so the downstream FP8 GEMM
//  reads M=PADM directly and skips linear_fp8_from_fp8's pad copy. The same
//  norm output feeds q/k/v (3 GEMMs) or gate/up (2 GEMMs) — one pre-padded
//  buffer replaces 3 (resp. 2) redundant 51->64 copies per layer.
// =============================================================================
__global__ void lingbot_ada_rms_fp8_mpad_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,     // [B, S, D] read-only
    const __nv_bfloat16* __restrict__ rms_weight,
    const __nv_bfloat16* __restrict__ gamma, // [B, D]
    const __nv_bfloat16* __restrict__ beta,  // [B, D]
    __nv_fp8_e4m3* __restrict__ out_fp8,     // [B, PADM, D]
    const float* __restrict__ act_scale,
    int B, int S, int PADM, int D, float eps)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int out_off = (b * PADM + s) * D;
    if (s >= S) {
        for (int d = tid; d < D; d += blockDim.x)
            out_fp8[out_off + d] = __nv_fp8_e4m3(0.0f);
        return;
    }
    int row_offset = (b * S + s) * D;
    int bd_offset = b * D;
    extern __shared__ float smem[];
    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[row_offset + d]);
        local_sum_sq += v * v;
    }
    float total_sum_sq = lingbot_block_reduce_sum(local_sum_sq, smem);
    float rsqrt_var = rsqrtf(total_sum_sq / (float)D + eps);
    float inv_scale = 1.0f / __ldg(act_scale);
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(x[row_offset + d]);
        float w = __bfloat162float(rms_weight[d]);
        float g = __bfloat162float(gamma[bd_offset + d]);
        float bt = __bfloat162float(beta[bd_offset + d]);
        float norm_v = v * rsqrt_var * w;
        float film_v = (1.0f + g) * norm_v + bt;
        float film_bf16 = __bfloat162float(__float2bfloat16_rn(film_v));
        float fp8_v = film_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[out_off + d] = __nv_fp8_e4m3(fp8_v);
    }
}

__global__ void lingbot_ada_rms_residual_fp8_mpad_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,    // [B, S, D] mutated to residual+x
    const __nv_bfloat16* __restrict__ x,     // [B, S, D]
    const __nv_bfloat16* __restrict__ rms_weight,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_fp8_e4m3* __restrict__ out_fp8,     // [B, PADM, D]
    const float* __restrict__ act_scale,
    int B, int S, int PADM, int D, float eps)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int out_off = (b * PADM + s) * D;
    if (s >= S) {
        for (int d = tid; d < D; d += blockDim.x)
            out_fp8[out_off + d] = __nv_fp8_e4m3(0.0f);
        return;
    }
    int row_offset = (b * S + s) * D;
    int bd_offset = b * D;
    extern __shared__ float smem[];
    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float r = __bfloat162float(residual[row_offset + d]);
        float xv = __bfloat162float(x[row_offset + d]);
        float y = r + xv;
        residual[row_offset + d] = __float2bfloat16_rn(y);
        local_sum_sq += y * y;
    }
    float total_sum_sq = lingbot_block_reduce_sum(local_sum_sq, smem);
    float rsqrt_var = rsqrtf(total_sum_sq / (float)D + eps);
    float inv_scale = 1.0f / __ldg(act_scale);
    for (int d = tid; d < D; d += blockDim.x) {
        float y = __bfloat162float(residual[row_offset + d]);
        float w = __bfloat162float(rms_weight[d]);
        float g = __bfloat162float(gamma[bd_offset + d]);
        float bt = __bfloat162float(beta[bd_offset + d]);
        float norm_v = y * rsqrt_var * w;
        float film_v = (1.0f + g) * norm_v + bt;
        float film_bf16 = __bfloat162float(__float2bfloat16_rn(film_v));
        float fp8_v = film_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[out_off + d] = __nv_fp8_e4m3(fp8_v);
    }
}

void lingbot_ada_rms_fp8_mpad_bf16(
    uintptr_t x, uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int PADM, int D, float eps, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(PADM, B);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = BLOCK_THREADS * sizeof(float);
    lingbot_ada_rms_fp8_mpad_bf16_kernel<<<grid, block, smem_bytes, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(rms_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        B, S, PADM, D, eps);
}

void lingbot_ada_rms_residual_fp8_mpad_bf16(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int PADM, int D, float eps, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(PADM, B);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = BLOCK_THREADS * sizeof(float);
    lingbot_ada_rms_residual_fp8_mpad_bf16_kernel<<<grid, block, smem_bytes, to_stream(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(rms_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        B, S, PADM, D, eps);
}

// =============================================================================
//  Kernel: AdaRMS + residual, fp16 OUTPUT (M-padded), no FP8 quant.
//  For the denoise FP4 path (G53): the post-attn AdaRMS feeds the FP4 gate_up
//  GEMM. Outputting fp16 directly lets the FP4 quant run on it WITHOUT the
//  fp8->fp16 dequant round-trip (4 tiny launch-bound kernels, ~18us/layer).
//  Same math as the fp8 mpad variant minus the fp8 scale/clamp.
// =============================================================================
__global__ void lingbot_ada_rms_residual_fp16_mpad_kernel(
    __nv_bfloat16* __restrict__ residual,    // [B, S, D] mutated to residual+x
    const __nv_bfloat16* __restrict__ x,     // [B, S, D]
    const __nv_bfloat16* __restrict__ rms_weight,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __half* __restrict__ out_fp16,           // [B, PADM, D]
    int B, int S, int PADM, int D, float eps)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int out_off = (b * PADM + s) * D;
    if (s >= S) {
        for (int d = tid; d < D; d += blockDim.x)
            out_fp16[out_off + d] = __float2half_rn(0.0f);
        return;
    }
    int row_offset = (b * S + s) * D;
    int bd_offset = b * D;
    extern __shared__ float smem[];
    float local_sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float r = __bfloat162float(residual[row_offset + d]);
        float xv = __bfloat162float(x[row_offset + d]);
        float y = r + xv;
        residual[row_offset + d] = __float2bfloat16_rn(y);
        local_sum_sq += y * y;
    }
    float total_sum_sq = lingbot_block_reduce_sum(local_sum_sq, smem);
    float rsqrt_var = rsqrtf(total_sum_sq / (float)D + eps);
    for (int d = tid; d < D; d += blockDim.x) {
        float y = __bfloat162float(residual[row_offset + d]);
        float w = __bfloat162float(rms_weight[d]);
        float g = __bfloat162float(gamma[bd_offset + d]);
        float bt = __bfloat162float(beta[bd_offset + d]);
        float norm_v = y * rsqrt_var * w;
        float film_v = (1.0f + g) * norm_v + bt;
        out_fp16[out_off + d] = __float2half_rn(film_v);
    }
}

void lingbot_ada_rms_residual_fp16_mpad(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp16, int B, int S, int PADM, int D, float eps, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(PADM, B);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = BLOCK_THREADS * sizeof(float);
    lingbot_ada_rms_residual_fp16_mpad_kernel<<<grid, block, smem_bytes, to_stream(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(rms_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__half*>(out_fp16),
        B, S, PADM, D, eps);
}

// =============================================================================
//  Kernel: bf16 in-place RoPE (LingBot split-half variant)
// =============================================================================
//   For each (b, s, h):
//     x1, x2 = x[..., :half], x[..., half:]
//     x[..., :half] = x1 * cos - x2 * sin
//     x[..., half:] = x2 * cos + x1 * sin
//   cos, sin have shape [B, S, half] (per-(b,s) shared across heads).
//
// Replaces 5 launches per call (x.to(fp32) + apply + .to(bf16) + repeat for K)
// with one in-place pass that does the math in fp32 internally.
void lingbot_ada_rms_fp8_bf16(
    uintptr_t x, uintptr_t rms_weight,
    uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, float eps, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(S, B);
    dim3 block(BLOCK_THREADS);
    size_t smem_bytes = BLOCK_THREADS * sizeof(float);
    lingbot_ada_rms_fp8_bf16_kernel<<<grid, block, smem_bytes, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(rms_weight),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        B, S, D, eps
    );
}


// =============================================================================
//  Kernel: fused QKV bias-add + RoPE (Expert denoise QKV post-processing)
// =============================================================================
// Replaces, per Expert layer, the 5 separate launches
//   add_bias(q) + add_bias(k) + add_bias(v) + rope(q) + rope(k)
// with ONE kernel. q/k get (bias add → split-half RoPE); v gets bias only.
// q has NHQ heads, k/v have NHKV heads, all head_dim HD. Inputs are the raw
// (bias-free) bf16 GEMM outputs [M, NH*HD]; cos/sin are [M, HD/2] fp32.
//
// Grid:  (NHQ + 2*NHKV, M)   — one block per (head-slot, token)
// Block: min(HD/2, 128) threads
//   slot < NHQ            -> q head (slot)
//   NHQ <= slot < NHQ+NHKV-> k head (slot-NHQ)
//   else                 -> v head (slot-NHQ-NHKV)
