// Decomposed tiny-M decode attention for the pi0.5 action expert (Thor).
//
// For q_tokens ≤ 16 over a padded f16 KV of token rows, flash attention's
// stream-k kernel plus its fixup pass is slower than the classic
// decomposition the FlashRT torch pipeline uses: one QK^T GEMM, a masked
// softmax over the KV axis, and one PV GEMM.
//
// All GQA query heads share the single KV head, so instead of a batched
// GEMM per head both contractions run as one wide GEMM over the
// n_head*n_tok query rows. Rows are ordered t-major (row r = t*n_head + h):
// with that ordering the PV output column j lands at byte offset j*hd in
// the flash-attention node's [hd, n_head, n_tok] destination, i.e. the
// GEMM writes the fp32 result contiguously with no strided-C penalty (the
// dispatch layer guarantees the destination is contiguous). The t-major
// order also makes the q gather read the [hd, n_head, n_tok]-contiguous
// Q buffer sequentially.
//
// Numerics follow ggml's fattn contract: scores = scale * q.k + mask (f16
// mask, slope 1 as max_bias must be 0), softmax in fp32 with running max.

#include "fr_kernels.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace ggml_cuda_flashrt {

namespace {

// gather the permuted f32 Q view into contiguous f16 rows [n_tok*n_head, hd]
// (row r = t*n_head + h), applying nothing else (scale folds into QK alpha)
__global__ void kernel_q_gather_f16(const float * __restrict__ q,
                                    __half * __restrict__ out,
                                    int hd, int n_head,
                                    int64_t s_d, int64_t s_tok, int64_t s_head) {
    const int r = blockIdx.x;          // t*n_head + h
    const int h = r % n_head;
    const int t = r / n_head;
    const float * src = q + (int64_t) h * s_head + (int64_t) t * s_tok;
    __half * dst = out + (int64_t) r * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        dst[d] = __float2half(src[(int64_t) d * s_d]);
    }
}

// in-place masked softmax over rows of [n_tok*n_head, n_kv] f16 scores.
// mask element for (kv, t) at mask + kv + t*mask_stride (f16, -inf on pads).
__global__ void kernel_mask_softmax_f16(__half * __restrict__ scores,
                                        const __half * __restrict__ mask,
                                        int n_kv, int n_head, int64_t mask_stride) {
    const int r = blockIdx.x;          // t*n_head + h
    const int t = r / n_head;
    __half * row = scores + (int64_t) r * n_kv;
    const __half * mrow = mask + (int64_t) t * mask_stride;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        const float v = __half2float(row[i]) + __half2float(mrow[i]);
        m = fmaxf(m, v);
    }
    __shared__ float red[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        }
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        const float v = __half2float(row[i]) + __half2float(mrow[i]);
        const float e = expf(v - m);
        row[i] = __float2half(e);
        sum += e;
    }
    __shared__ float red2[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    if (threadIdx.x % 32 == 0) red2[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < blockDim.x / 32) ? red2[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (threadIdx.x == 0) red2[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red2[0];
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        row[i] = __float2half(__half2float(row[i]) * inv);
    }
}

} // namespace

int decode_attn_decomposed(void * cublas_handle,
                           const float * q, int64_t q_sd, int64_t q_stok, int64_t q_shead,
                           const void * k_f16_rows,   // [n_kv, hd] f16 rows
                           const void * v_f16_rows,   // [n_kv, hd] f16 rows
                           const void * mask_f16, int64_t mask_stride,
                           float * dst, int64_t dst_stok, int64_t dst_shead,
                           void * q16_ws, int q16_ready, void * scores_ws,
                           int hd, int n_tok, int n_head, int n_kv,
                           float scale, cudaStream_t stream) {
    // the contiguous PV store below requires the [hd, n_head, n_tok] dst
    // to be dense; the dispatch layer checks the same before fusing
    if (dst_shead != hd || dst_stok != (int64_t) hd * n_head) {
        return -1;
    }
    cublasHandle_t handle = (cublasHandle_t) cublas_handle;
    const int R = n_head * n_tok;

    if (!q16_ready) {
        kernel_q_gather_f16<<<R, 128, 0, stream>>>(
            q, (__half *) q16_ws, hd, n_head, q_sd, q_stok, q_shead);
    }

    cublasSetStream(handle, stream);
    // scores_col[n_kv, R] = K_col^T [n_kv, hd] x Q16_col [hd, R]
    const float beta0 = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_kv, R, hd,
        &scale,
        k_f16_rows, CUDA_R_16F, hd,
        q16_ws,     CUDA_R_16F, hd,
        &beta0,
        scores_ws,  CUDA_R_16F, n_kv,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return -100 - (int) st;

    kernel_mask_softmax_f16<<<R, 256, 0, stream>>>(
        (__half *) scores_ws, (const __half *) mask_f16, n_kv, n_head, mask_stride);

    // dst_col[hd, R] (dense, column r = t*n_head + h at offset r*hd) =
    //   V_col [hd, n_kv] x P_col [n_kv, R]
    const float one = 1.0f;
    st = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hd, R, n_kv,
        &one,
        v_f16_rows, CUDA_R_16F, hd,
        scores_ws,  CUDA_R_16F, n_kv,
        &beta0,
        dst, CUDA_R_32F, hd,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return -200 - (int) st;

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt
