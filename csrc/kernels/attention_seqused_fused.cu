// ================================================================
// FlashRT — seqused attention with the mask folded into softmax.
//
// attention_qkv_fp16_seqused runs QK^T, a standalone -inf mask
// kernel over rows [valid, S_kv_max), softmax, then AV. The v2 chain
// folds the seqused bound into the softmax kernel itself: max/sum
// run over [0, valid) and positions beyond valid are written as
// zero probabilities (identical to exp(-65504 - max) ~ 0 in the
// reference), removing one kernel per call.
//
// Additive: attention_cublas.cu is untouched.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "attention_seqused_fused.cuh"

namespace {

#define SMV2_WARP_SIZE 32
#define SMV2_MAX_COLS 1024
#define SMV2_ITERS (SMV2_MAX_COLS / SMV2_WARP_SIZE)

// Any S_kv_max is supported: rows up to SMV2_MAX_COLS use the register-tiled
// softmax below, wider rows use the multi-pass variant after it.
//
// One warp per logits row. The register-to-column mapping, the reduction
// order, and the arithmetic are all identical to softmax_fp16_kernel; the
// only difference is that out-of-range columns take -1e30 at load time
// instead of being read back as the -65504 a separate mask kernel wrote.
// Both exponentiate to exactly zero, so the stored probabilities are
// bit-identical to the mask + softmax chain.
__global__ void softmax_fp16_seqused_kernel(
    __half* data, int rows, int cols, const int* __restrict__ seqused_k) {
    int lane = threadIdx.x % SMV2_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    int valid = seqused_k[0];
    if (valid < 0) valid = 0;
    if (valid > cols) valid = cols;

    __half* src = data + row * cols;
    int cols2 = cols / 2;
    __half2* src2 = reinterpret_cast<__half2*>(src);

    float reg[SMV2_ITERS];
    float mx = -1e30f;
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS / 2; it++) {
        int c2 = it * SMV2_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2 = src2[c2];
            reg[it*2]   = (2 * c2     < valid) ? __half2float(v2.x) : -1e30f;
            reg[it*2+1] = (2 * c2 + 1 < valid) ? __half2float(v2.y) : -1e30f;
            mx = fmaxf(mx, fmaxf(reg[it*2], reg[it*2+1]));
        } else {
            reg[it*2] = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        float v = (cols - 1 < valid) ? __half2float(src[cols-1]) : -1e30f;
        reg[SMV2_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    float sm = 0.f;
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SMV2_ITERS / 2; it++) {
        int c2 = it * SMV2_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2;
            v2.x = __float2half(reg[it*2] * inv);
            v2.y = __float2half(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2half(reg[SMV2_ITERS-1] * inv);
    }
}

// Register-tiled softmax above covers at most SMV2_MAX_COLS columns. Wider
// logits rows go through this multi-pass variant, which keeps no
// per-column registers and is therefore correct at any width: without it,
// columns past the tile were left unnormalized and still fed the PV GEMM.
__global__ void softmax_fp16_seqused_wide_kernel(
    __half* data, int rows, int cols, const int* __restrict__ seqused_k) {
    const int lane = threadIdx.x % SMV2_WARP_SIZE;
    const int row = blockIdx.x;
    if (row >= rows) return;

    int valid = seqused_k[0];
    if (valid < 0) valid = 0;
    if (valid > cols) valid = cols;

    __half* src = data + (long)row * cols;

    float mx = -1e30f;
    for (int c = lane; c < cols; c += SMV2_WARP_SIZE) {
        const float v = (c < valid) ? __half2float(src[c]) : -1e30f;
        mx = fmaxf(mx, v);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));

    float sum = 0.f;
    for (int c = lane; c < cols; c += SMV2_WARP_SIZE) {
        const float v = (c < valid) ? __half2float(src[c]) : -1e30f;
        const float e = __expf(v - mx);
        src[c] = __float2half(e);
        sum += e;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = 1.0f / sum;

    for (int c = lane; c < cols; c += SMV2_WARP_SIZE)
        src[c] = __float2half(__half2float(src[c]) * inv);
}

}  // namespace

void attention_qkv_fp16_seqused_v2(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv_max, S * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, S_kv_max,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (S_kv_max <= SMV2_MAX_COLS) {
        softmax_fp16_seqused_kernel<<<S * NH, SMV2_WARP_SIZE, 0, stream>>>(
            logits, S * NH, S_kv_max, seqused_k);
    } else {
        softmax_fp16_seqused_wide_kernel<<<S * NH, SMV2_WARP_SIZE, 0, stream>>>(
            logits, S * NH, S_kv_max, seqused_k);
    }

    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv_max,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, S_kv_max,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
