// SPDX-License-Identifier: Apache-2.0
// G7.19 — Fused QKV split + WanRMSNorm + 3D RoPE for Wan video Q/K.
// See header for spec.

#include "qkv_split_norm_rope_bf16.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

__global__ void qkv_split_norm_rope_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,  // (B, L, 3*dim)
    const __nv_bfloat16* __restrict__ norm_q_w,    // (dim,)
    const __nv_bfloat16* __restrict__ norm_k_w,    // (dim,)
    const float*          __restrict__ freqs_re,   // (seq_len, D_h/2)
    const float*          __restrict__ freqs_im,
    __nv_bfloat16*       __restrict__ q_rope_out,  // (B, L, N*D_h) row-major
    __nv_bfloat16*       __restrict__ k_rope_out,
    int B, int L, int N, int D_h, int seq_len, float eps)
{
  // 1 block per (b, t). Threads stride over `dim = N * D_h`.
  const int row = blockIdx.x;          // = b * L + t
  const int t   = row % L;
  const int dim = N * D_h;
  const int dim2 = dim >> 1;            // bf162 vector count

  const long long row_off  = (long long)row * 3 * dim;
  const long long out_row  = (long long)row * dim;
  const long long fr_row   = (long long)t * (D_h >> 1);

  const __nv_bfloat162* qkv2 = reinterpret_cast<const __nv_bfloat162*>(
      packed_qkv + row_off);                     // q at [0..dim2)
  const __nv_bfloat162* nqw2 = reinterpret_cast<const __nv_bfloat162*>(
      norm_q_w);
  const __nv_bfloat162* nkw2 = reinterpret_cast<const __nv_bfloat162*>(
      norm_k_w);

  __nv_bfloat16* q_out = q_rope_out + out_row;
  __nv_bfloat16* k_out = k_rope_out + out_row;

  __shared__ float reduce_pad[33];

  // ── Pass 1: accumulate sum(q^2) and sum(k^2) for RMS norm.
  // Float accumulators per thread.
  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];                     // q half
    __nv_bfloat162 kv = qkv2[dim2 + i];              // k half (offset dim2 in bf162 stride)
    const float q0 = __bfloat162float(qv.x), q1 = __bfloat162float(qv.y);
    const float k0 = __bfloat162float(kv.x), k1 = __bfloat162float(kv.y);
    sq += q0 * q0 + q1 * q1;
    sk += k0 * k0 + k1 * k1;
  }
  // Block-wide reduce both (interleaved).
  for (int o = 16; o > 0; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    sk += __shfl_xor_sync(0xffffffffu, sk, o);
  }
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) { reduce_pad[wid] = sq; reduce_pad[16 + wid] = sk; }
  __syncthreads();
  if (!wid) {
    sq = (lane < (blockDim.x >> 5)) ? reduce_pad[lane] : 0.f;
    sk = (lane < (blockDim.x >> 5)) ? reduce_pad[16 + lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    reduce_pad[32] = sq;
    reduce_pad[31] = sk;
  }
  __syncthreads();
  const float inv_q = rsqrtf(reduce_pad[32] / (float)dim + eps);
  const float inv_k = rsqrtf(reduce_pad[31] / (float)dim + eps);

  // ── Pass 2: emit normed Q, K with RoPE applied per (n, c_pair).
  // Iterate c_pair < D_h/2 within each head, n head index.
  const int dh2 = D_h >> 1;        // c_pair count per head
  const bool apply_rope = (t < seq_len);

  // Total elements per row to emit: N * dh2 (each emits 2 fp values).
  const int total_pairs = N * dh2;
  for (int p = threadIdx.x; p < total_pairs; p += blockDim.x) {
    const int n = p / dh2;
    const int c_pair = p - n * dh2;
    const int c_re_in_dim = n * D_h + 2 * c_pair;
    const int c_im_in_dim = c_re_in_dim + 1;

    // Read q, k, norm weights at these two columns.
    const float qre = __bfloat162float(packed_qkv[row_off + c_re_in_dim]);
    const float qim = __bfloat162float(packed_qkv[row_off + c_im_in_dim]);
    const float kre = __bfloat162float(
        packed_qkv[row_off + dim + c_re_in_dim]);
    const float kim = __bfloat162float(
        packed_qkv[row_off + dim + c_im_in_dim]);
    const float wq_re = __bfloat162float(norm_q_w[c_re_in_dim]);
    const float wq_im = __bfloat162float(norm_q_w[c_im_in_dim]);
    const float wk_re = __bfloat162float(norm_k_w[c_re_in_dim]);
    const float wk_im = __bfloat162float(norm_k_w[c_im_in_dim]);

    // Normed q, k (still fp32).
    const float qre_n = qre * inv_q * wq_re;
    const float qim_n = qim * inv_q * wq_im;
    const float kre_n = kre * inv_k * wk_re;
    const float kim_n = kim * inv_k * wk_im;

    float qre_out, qim_out, kre_out, kim_out;
    if (apply_rope) {
      const float fr = freqs_re[fr_row + c_pair];
      const float fi = freqs_im[fr_row + c_pair];
      qre_out = qre_n * fr - qim_n * fi;
      qim_out = qre_n * fi + qim_n * fr;
      kre_out = kre_n * fr - kim_n * fi;
      kim_out = kre_n * fi + kim_n * fr;
    } else {
      qre_out = qre_n; qim_out = qim_n;
      kre_out = kre_n; kim_out = kim_n;
    }

    q_out[c_re_in_dim] = __float2bfloat16(qre_out);
    q_out[c_im_in_dim] = __float2bfloat16(qim_out);
    k_out[c_re_in_dim] = __float2bfloat16(kre_out);
    k_out[c_im_in_dim] = __float2bfloat16(kim_out);
  }
}

__device__ __forceinline__ void qkv_split_norm_one_row(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ norm_q_w,
    const __nv_bfloat16* __restrict__ norm_k_w,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int row, int dim, float eps, float* reduce_pad) {
  const int dim2 = dim >> 1;
  const long long row_off = (long long)row * 3 * dim;
  const long long out_row = (long long)row * dim;
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_off);

  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    const float q0 = __bfloat162float(qv.x), q1 = __bfloat162float(qv.y);
    const float k0 = __bfloat162float(kv.x), k1 = __bfloat162float(kv.y);
    sq += q0 * q0 + q1 * q1;
    sk += k0 * k0 + k1 * k1;
  }
  for (int o = 16; o > 0; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    sk += __shfl_xor_sync(0xffffffffu, sk, o);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) {
    reduce_pad[wid] = sq;
    reduce_pad[16 + wid] = sk;
  }
  __syncthreads();
  if (!wid) {
    sq = (lane < (blockDim.x >> 5)) ? reduce_pad[lane] : 0.f;
    sk = (lane < (blockDim.x >> 5)) ? reduce_pad[16 + lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    reduce_pad[32] = sq;
    reduce_pad[31] = sk;
  }
  __syncthreads();
  const float inv_q = rsqrtf(reduce_pad[32] / (float)dim + eps);
  const float inv_k = rsqrtf(reduce_pad[31] / (float)dim + eps);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    __nv_bfloat162 wq =
        reinterpret_cast<const __nv_bfloat162*>(norm_q_w)[i];
    __nv_bfloat162 wk =
        reinterpret_cast<const __nv_bfloat162*>(norm_k_w)[i];
    const float q0 = __bfloat162float(qv.x) * inv_q * __bfloat162float(wq.x);
    const float q1 = __bfloat162float(qv.y) * inv_q * __bfloat162float(wq.y);
    const float k0 = __bfloat162float(kv.x) * inv_k * __bfloat162float(wk.x);
    const float k1 = __bfloat162float(kv.y) * inv_k * __bfloat162float(wk.y);
    reinterpret_cast<__nv_bfloat162*>(q_out + out_row)[i] =
        __halves2bfloat162(__float2bfloat16(q0), __float2bfloat16(q1));
    reinterpret_cast<__nv_bfloat162*>(k_out + out_row)[i] =
        __halves2bfloat162(__float2bfloat16(k0), __float2bfloat16(k1));
  }
}

__global__ void qkv_split_norm2_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_a,
    const __nv_bfloat16* __restrict__ norm_a_q_w,
    const __nv_bfloat16* __restrict__ norm_a_k_w,
    __nv_bfloat16* __restrict__ q_a_out,
    __nv_bfloat16* __restrict__ k_a_out,
    int L_a, int dim, float eps_a,
    const __nv_bfloat16* __restrict__ packed_u,
    const __nv_bfloat16* __restrict__ norm_u_q_w,
    const __nv_bfloat16* __restrict__ norm_u_k_w,
    __nv_bfloat16* __restrict__ q_u_out,
    __nv_bfloat16* __restrict__ k_u_out,
    int L_u, float eps_u) {
  extern __shared__ float smem[];
  const int row = blockIdx.x;
  if (row < L_a) {
    qkv_split_norm_one_row(packed_a, norm_a_q_w, norm_a_k_w,
                           q_a_out, k_a_out, row, dim, eps_a, smem);
  } else {
    qkv_split_norm_one_row(packed_u, norm_u_q_w, norm_u_k_w,
                           q_u_out, k_u_out, row - L_a, dim, eps_u, smem);
  }
}

__global__ void qkv_split_bias_norm_rope_v_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ qkv_bias,
    const __nv_bfloat16* __restrict__ norm_q_w,
    const __nv_bfloat16* __restrict__ norm_k_w,
    const float* __restrict__ freqs_re,
    const float* __restrict__ freqs_im,
    __nv_bfloat16* __restrict__ q_rope_out,
    __nv_bfloat16* __restrict__ k_rope_out,
    __nv_bfloat16* __restrict__ v_out,
    int L, int N, int D_h, int seq_len, float eps) {
  const int row = blockIdx.x;
  const int t = row % L;
  const int dim = N * D_h;
  const int dim2 = dim >> 1;
  const long long row_off = (long long)row * 3 * dim;
  const long long out_row = (long long)row * dim;
  const long long fr_row = (long long)t * (D_h >> 1);
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_off);
  const __nv_bfloat162* bias2 =
      reinterpret_cast<const __nv_bfloat162*>(qkv_bias);

  __shared__ float reduce_pad[33];
  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    __nv_bfloat162 bq = bias2[i];
    __nv_bfloat162 bk = bias2[dim2 + i];
    const float q0 = __bfloat162float(qv.x) + __bfloat162float(bq.x);
    const float q1 = __bfloat162float(qv.y) + __bfloat162float(bq.y);
    const float k0 = __bfloat162float(kv.x) + __bfloat162float(bk.x);
    const float k1 = __bfloat162float(kv.y) + __bfloat162float(bk.y);
    sq += q0 * q0 + q1 * q1;
    sk += k0 * k0 + k1 * k1;
  }
  for (int o = 16; o > 0; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    sk += __shfl_xor_sync(0xffffffffu, sk, o);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) { reduce_pad[wid] = sq; reduce_pad[16 + wid] = sk; }
  __syncthreads();
  if (!wid) {
    sq = (lane < (blockDim.x >> 5)) ? reduce_pad[lane] : 0.f;
    sk = (lane < (blockDim.x >> 5)) ? reduce_pad[16 + lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    reduce_pad[32] = sq;
    reduce_pad[31] = sk;
  }
  __syncthreads();
  const float inv_q = rsqrtf(reduce_pad[32] / (float)dim + eps);
  const float inv_k = rsqrtf(reduce_pad[31] / (float)dim + eps);
  const int dh2 = D_h >> 1;
  const bool apply_rope = (t < seq_len);

  for (int p = threadIdx.x; p < dim2; p += blockDim.x) {
    const int n = p / dh2;
    const int c_pair = p - n * dh2;
    const int c0 = n * D_h + 2 * c_pair;
    const int c1 = c0 + 1;

    __nv_bfloat162 qv = qkv2[p];
    __nv_bfloat162 kv = qkv2[dim2 + p];
    __nv_bfloat162 vv = qkv2[2 * dim2 + p];
    __nv_bfloat162 bq = bias2[p];
    __nv_bfloat162 bk = bias2[dim2 + p];
    __nv_bfloat162 bv = bias2[2 * dim2 + p];
    __nv_bfloat162 wq = reinterpret_cast<const __nv_bfloat162*>(norm_q_w)[p];
    __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(norm_k_w)[p];

    const float qre_n = (__bfloat162float(qv.x) + __bfloat162float(bq.x))
        * inv_q * __bfloat162float(wq.x);
    const float qim_n = (__bfloat162float(qv.y) + __bfloat162float(bq.y))
        * inv_q * __bfloat162float(wq.y);
    const float kre_n = (__bfloat162float(kv.x) + __bfloat162float(bk.x))
        * inv_k * __bfloat162float(wk.x);
    const float kim_n = (__bfloat162float(kv.y) + __bfloat162float(bk.y))
        * inv_k * __bfloat162float(wk.y);

    float qre_out, qim_out, kre_out, kim_out;
    if (apply_rope) {
      const float fr = freqs_re[fr_row + c_pair];
      const float fi = freqs_im[fr_row + c_pair];
      qre_out = qre_n * fr - qim_n * fi;
      qim_out = qre_n * fi + qim_n * fr;
      kre_out = kre_n * fr - kim_n * fi;
      kim_out = kre_n * fi + kim_n * fr;
    } else {
      qre_out = qre_n; qim_out = qim_n;
      kre_out = kre_n; kim_out = kim_n;
    }

    q_rope_out[out_row + c0] = __float2bfloat16(qre_out);
    q_rope_out[out_row + c1] = __float2bfloat16(qim_out);
    k_rope_out[out_row + c0] = __float2bfloat16(kre_out);
    k_rope_out[out_row + c1] = __float2bfloat16(kim_out);
    reinterpret_cast<__nv_bfloat162*>(v_out + out_row)[p] =
        __halves2bfloat162(
            __float2bfloat16(__bfloat162float(vv.x) + __bfloat162float(bv.x)),
            __float2bfloat16(__bfloat162float(vv.y) + __bfloat162float(bv.y)));
  }
}

__global__ void qkv_split_bias_norm_rope_v_cat_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ qkv_bias,
    const __nv_bfloat16* __restrict__ norm_q_w,
    const __nv_bfloat16* __restrict__ norm_k_w,
    const float* __restrict__ freqs_re,
    const float* __restrict__ freqs_im,
    __nv_bfloat16* __restrict__ q_cat_out,
    __nv_bfloat16* __restrict__ k_cat_out,
    __nv_bfloat16* __restrict__ v_cat_out,
    int total_L, int video_offset, int L, int N, int D_h,
    int seq_len, float eps) {
  const int row = blockIdx.x;
  const int b = blockIdx.y;
  const int t = row % L;
  const int dim = N * D_h;
  const int dim2 = dim >> 1;
  const long long row_off = ((long long)b * L + row) * 3 * dim;
  const long long out_row =
      ((long long)b * total_L + video_offset + row) * dim;
  const long long fr_row = (long long)t * (D_h >> 1);
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_off);
  const __nv_bfloat162* bias2 =
      reinterpret_cast<const __nv_bfloat162*>(qkv_bias);

  __shared__ float reduce_pad[33];
  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    __nv_bfloat162 bq = bias2[i];
    __nv_bfloat162 bk = bias2[dim2 + i];
    const float q0 = __bfloat162float(qv.x) + __bfloat162float(bq.x);
    const float q1 = __bfloat162float(qv.y) + __bfloat162float(bq.y);
    const float k0 = __bfloat162float(kv.x) + __bfloat162float(bk.x);
    const float k1 = __bfloat162float(kv.y) + __bfloat162float(bk.y);
    sq += q0 * q0 + q1 * q1;
    sk += k0 * k0 + k1 * k1;
  }
  for (int o = 16; o > 0; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    sk += __shfl_xor_sync(0xffffffffu, sk, o);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) { reduce_pad[wid] = sq; reduce_pad[16 + wid] = sk; }
  __syncthreads();
  if (!wid) {
    sq = (lane < (blockDim.x >> 5)) ? reduce_pad[lane] : 0.f;
    sk = (lane < (blockDim.x >> 5)) ? reduce_pad[16 + lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    reduce_pad[32] = sq;
    reduce_pad[31] = sk;
  }
  __syncthreads();
  const float inv_q = rsqrtf(reduce_pad[32] / (float)dim + eps);
  const float inv_k = rsqrtf(reduce_pad[31] / (float)dim + eps);
  const int dh2 = D_h >> 1;
  const bool apply_rope = (t < seq_len);

  for (int p = threadIdx.x; p < dim2; p += blockDim.x) {
    const int n = p / dh2;
    const int c_pair = p - n * dh2;
    const int c0 = n * D_h + 2 * c_pair;
    const int c1 = c0 + 1;

    __nv_bfloat162 qv = qkv2[p];
    __nv_bfloat162 kv = qkv2[dim2 + p];
    __nv_bfloat162 vv = qkv2[2 * dim2 + p];
    __nv_bfloat162 bq = bias2[p];
    __nv_bfloat162 bk = bias2[dim2 + p];
    __nv_bfloat162 bv = bias2[2 * dim2 + p];
    __nv_bfloat162 wq = reinterpret_cast<const __nv_bfloat162*>(norm_q_w)[p];
    __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(norm_k_w)[p];

    const float qre_n = (__bfloat162float(qv.x) + __bfloat162float(bq.x))
        * inv_q * __bfloat162float(wq.x);
    const float qim_n = (__bfloat162float(qv.y) + __bfloat162float(bq.y))
        * inv_q * __bfloat162float(wq.y);
    const float kre_n = (__bfloat162float(kv.x) + __bfloat162float(bk.x))
        * inv_k * __bfloat162float(wk.x);
    const float kim_n = (__bfloat162float(kv.y) + __bfloat162float(bk.y))
        * inv_k * __bfloat162float(wk.y);

    float qre_out, qim_out, kre_out, kim_out;
    if (apply_rope) {
      const float fr = freqs_re[fr_row + c_pair];
      const float fi = freqs_im[fr_row + c_pair];
      qre_out = qre_n * fr - qim_n * fi;
      qim_out = qre_n * fi + qim_n * fr;
      kre_out = kre_n * fr - kim_n * fi;
      kim_out = kre_n * fi + kim_n * fr;
    } else {
      qre_out = qre_n; qim_out = qim_n;
      kre_out = kre_n; kim_out = kim_n;
    }

    q_cat_out[out_row + c0] = __float2bfloat16(qre_out);
    q_cat_out[out_row + c1] = __float2bfloat16(qim_out);
    k_cat_out[out_row + c0] = __float2bfloat16(kre_out);
    k_cat_out[out_row + c1] = __float2bfloat16(kim_out);
    reinterpret_cast<__nv_bfloat162*>(v_cat_out + out_row)[p] =
        __halves2bfloat162(
            __float2bfloat16(__bfloat162float(vv.x) + __bfloat162float(bv.x)),
            __float2bfloat16(__bfloat162float(vv.y) + __bfloat162float(bv.y)));
  }
}

__device__ __forceinline__ void qkv_split_norm_cat_one_row(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ norm_q_w,
    const __nv_bfloat16* __restrict__ norm_k_w,
    __nv_bfloat16* __restrict__ q_cat_out,
    __nv_bfloat16* __restrict__ k_cat_out,
    __nv_bfloat16* __restrict__ v_cat_out,
    int in_row, int out_row_idx, int total_L, int dim, float eps,
    float* reduce_pad) {
  const int dim2 = dim >> 1;
  const long long row_off = (long long)in_row * 3 * dim;
  const long long out_row = (long long)out_row_idx * dim;
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_off);

  float sq = 0.f, sk = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    const float q0 = __bfloat162float(qv.x), q1 = __bfloat162float(qv.y);
    const float k0 = __bfloat162float(kv.x), k1 = __bfloat162float(kv.y);
    sq += q0 * q0 + q1 * q1;
    sk += k0 * k0 + k1 * k1;
  }
  for (int o = 16; o > 0; o >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    sk += __shfl_xor_sync(0xffffffffu, sk, o);
  }
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) { reduce_pad[wid] = sq; reduce_pad[16 + wid] = sk; }
  __syncthreads();
  if (!wid) {
    sq = (lane < (blockDim.x >> 5)) ? reduce_pad[lane] : 0.f;
    sk = (lane < (blockDim.x >> 5)) ? reduce_pad[16 + lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    reduce_pad[32] = sq;
    reduce_pad[31] = sk;
  }
  __syncthreads();
  const float inv_q = rsqrtf(reduce_pad[32] / (float)dim + eps);
  const float inv_k = rsqrtf(reduce_pad[31] / (float)dim + eps);

  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 qv = qkv2[i];
    __nv_bfloat162 kv = qkv2[dim2 + i];
    __nv_bfloat162 vv = qkv2[2 * dim2 + i];
    __nv_bfloat162 wq =
        reinterpret_cast<const __nv_bfloat162*>(norm_q_w)[i];
    __nv_bfloat162 wk =
        reinterpret_cast<const __nv_bfloat162*>(norm_k_w)[i];
    const float q0 = __bfloat162float(qv.x) * inv_q * __bfloat162float(wq.x);
    const float q1 = __bfloat162float(qv.y) * inv_q * __bfloat162float(wq.y);
    const float k0 = __bfloat162float(kv.x) * inv_k * __bfloat162float(wk.x);
    const float k1 = __bfloat162float(kv.y) * inv_k * __bfloat162float(wk.y);
    reinterpret_cast<__nv_bfloat162*>(q_cat_out + out_row)[i] =
        __halves2bfloat162(__float2bfloat16(q0), __float2bfloat16(q1));
    reinterpret_cast<__nv_bfloat162*>(k_cat_out + out_row)[i] =
        __halves2bfloat162(__float2bfloat16(k0), __float2bfloat16(k1));
    reinterpret_cast<__nv_bfloat162*>(v_cat_out + out_row)[i] = vv;
  }
}

__global__ void qkv_split_joint3_cat_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_v,
    const __nv_bfloat16* __restrict__ qkv_v_bias,
    const __nv_bfloat16* __restrict__ norm_v_q_w,
    const __nv_bfloat16* __restrict__ norm_v_k_w,
    const float* __restrict__ freqs_re,
    const float* __restrict__ freqs_im,
    const __nv_bfloat16* __restrict__ packed_a,
    const __nv_bfloat16* __restrict__ norm_a_q_w,
    const __nv_bfloat16* __restrict__ norm_a_k_w,
    const __nv_bfloat16* __restrict__ packed_u,
    const __nv_bfloat16* __restrict__ norm_u_q_w,
    const __nv_bfloat16* __restrict__ norm_u_k_w,
    __nv_bfloat16* __restrict__ q_cat_out,
    __nv_bfloat16* __restrict__ k_cat_out,
    __nv_bfloat16* __restrict__ v_cat_out,
    int total_L, int L_v, int L_a, int L_u, int N, int D_h,
    int seq_len, float eps_v, float eps_a, float eps_u) {
  extern __shared__ float smem[];
  const int row = blockIdx.x;
  const int dim = N * D_h;
  if (row < L_v) {
    const int t = row;
    const int dim2 = dim >> 1;
    const long long row_off = (long long)row * 3 * dim;
    const long long out_row = (long long)row * dim;
    const long long fr_row = (long long)t * (D_h >> 1);
    const __nv_bfloat162* qkv2 =
        reinterpret_cast<const __nv_bfloat162*>(packed_v + row_off);
    const __nv_bfloat162* bias2 =
        reinterpret_cast<const __nv_bfloat162*>(qkv_v_bias);

    float sq = 0.f, sk = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
      __nv_bfloat162 qv = qkv2[i];
      __nv_bfloat162 kv = qkv2[dim2 + i];
      __nv_bfloat162 bq = bias2[i];
      __nv_bfloat162 bk = bias2[dim2 + i];
      const float q0 = __bfloat162float(qv.x) + __bfloat162float(bq.x);
      const float q1 = __bfloat162float(qv.y) + __bfloat162float(bq.y);
      const float k0 = __bfloat162float(kv.x) + __bfloat162float(bk.x);
      const float k1 = __bfloat162float(kv.y) + __bfloat162float(bk.y);
      sq += q0 * q0 + q1 * q1;
      sk += k0 * k0 + k1 * k1;
    }
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (!lane) { smem[wid] = sq; smem[16 + wid] = sk; }
    __syncthreads();
    if (!wid) {
      sq = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.f;
      sk = (lane < (blockDim.x >> 5)) ? smem[16 + lane] : 0.f;
      for (int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
        sk += __shfl_xor_sync(0xffffffffu, sk, o);
      }
    }
    __syncthreads();
    if (!threadIdx.x) {
      smem[32] = sq;
      smem[31] = sk;
    }
    __syncthreads();
    const float inv_q = rsqrtf(smem[32] / (float)dim + eps_v);
    const float inv_k = rsqrtf(smem[31] / (float)dim + eps_v);
    const int dh2 = D_h >> 1;
    const bool apply_rope = (t < seq_len);

    for (int p = threadIdx.x; p < dim2; p += blockDim.x) {
      const int n = p / dh2;
      const int c_pair = p - n * dh2;
      const int c0 = n * D_h + 2 * c_pair;
      const int c1 = c0 + 1;

      __nv_bfloat162 qv = qkv2[p];
      __nv_bfloat162 kv = qkv2[dim2 + p];
      __nv_bfloat162 vv = qkv2[2 * dim2 + p];
      __nv_bfloat162 bq = bias2[p];
      __nv_bfloat162 bk = bias2[dim2 + p];
      __nv_bfloat162 bv = bias2[2 * dim2 + p];
      __nv_bfloat162 wq =
          reinterpret_cast<const __nv_bfloat162*>(norm_v_q_w)[p];
      __nv_bfloat162 wk =
          reinterpret_cast<const __nv_bfloat162*>(norm_v_k_w)[p];

      const float qre_n = (__bfloat162float(qv.x) + __bfloat162float(bq.x))
          * inv_q * __bfloat162float(wq.x);
      const float qim_n = (__bfloat162float(qv.y) + __bfloat162float(bq.y))
          * inv_q * __bfloat162float(wq.y);
      const float kre_n = (__bfloat162float(kv.x) + __bfloat162float(bk.x))
          * inv_k * __bfloat162float(wk.x);
      const float kim_n = (__bfloat162float(kv.y) + __bfloat162float(bk.y))
          * inv_k * __bfloat162float(wk.y);

      float qre_out, qim_out, kre_out, kim_out;
      if (apply_rope) {
        const float fr = freqs_re[fr_row + c_pair];
        const float fi = freqs_im[fr_row + c_pair];
        qre_out = qre_n * fr - qim_n * fi;
        qim_out = qre_n * fi + qim_n * fr;
        kre_out = kre_n * fr - kim_n * fi;
        kim_out = kre_n * fi + kim_n * fr;
      } else {
        qre_out = qre_n; qim_out = qim_n;
        kre_out = kre_n; kim_out = kim_n;
      }

      q_cat_out[out_row + c0] = __float2bfloat16(qre_out);
      q_cat_out[out_row + c1] = __float2bfloat16(qim_out);
      k_cat_out[out_row + c0] = __float2bfloat16(kre_out);
      k_cat_out[out_row + c1] = __float2bfloat16(kim_out);
      reinterpret_cast<__nv_bfloat162*>(v_cat_out + out_row)[p] =
          __halves2bfloat162(
              __float2bfloat16(__bfloat162float(vv.x) + __bfloat162float(bv.x)),
              __float2bfloat16(__bfloat162float(vv.y) + __bfloat162float(bv.y)));
    }
  } else if (row < L_v + L_a) {
    const int a_row = row - L_v;
    qkv_split_norm_cat_one_row(
        packed_a, norm_a_q_w, norm_a_k_w, q_cat_out, k_cat_out, v_cat_out,
        a_row, L_v + a_row, total_L, dim, eps_a, smem);
  } else if (row < L_v + L_a + L_u) {
    const int u_row = row - L_v - L_a;
    qkv_split_norm_cat_one_row(
        packed_u, norm_u_q_w, norm_u_k_w, q_cat_out, k_cat_out, v_cat_out,
        u_row, L_v + L_a + u_row, total_L, dim, eps_u, smem);
  }
}

__global__ void qkv_split_norm2_cat_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed_a,
    const __nv_bfloat16* __restrict__ norm_a_q_w,
    const __nv_bfloat16* __restrict__ norm_a_k_w,
    const __nv_bfloat16* __restrict__ packed_u,
    const __nv_bfloat16* __restrict__ norm_u_q_w,
    const __nv_bfloat16* __restrict__ norm_u_k_w,
    __nv_bfloat16* __restrict__ q_cat_out,
    __nv_bfloat16* __restrict__ k_cat_out,
    __nv_bfloat16* __restrict__ v_cat_out,
    int total_L, int L_v, int L_a, int L_u, int dim,
    float eps_a, float eps_u) {
  extern __shared__ float smem[];
  const int row = blockIdx.x;
  if (row < L_a) {
    qkv_split_norm_cat_one_row(
        packed_a, norm_a_q_w, norm_a_k_w,
        q_cat_out, k_cat_out, v_cat_out,
        row, L_v + row, total_L, dim, eps_a, smem);
  } else {
    const int u_row = row - L_a;
    qkv_split_norm_cat_one_row(
        packed_u, norm_u_q_w, norm_u_k_w,
        q_cat_out, k_cat_out, v_cat_out,
        u_row, L_v + L_a + u_row, total_L, dim, eps_u, smem);
  }
}

}  // namespace

void qkv_split_norm_rope_bf16(
    const void*  packed_qkv,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_rope_out,
    void*        k_rope_out,
    int B, int L_v, int N, int D_h, int seq_len, float eps,
    cudaStream_t stream)
{
  if (B <= 0 || L_v <= 0 || N <= 0 || D_h <= 0) return;
  const int blocks = B * L_v;
  qkv_split_norm_rope_bf16_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_qkv),
      reinterpret_cast<const __nv_bfloat16*>(norm_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_k_w),
      freqs_re, freqs_im,
      reinterpret_cast<__nv_bfloat16*>(q_rope_out),
      reinterpret_cast<__nv_bfloat16*>(k_rope_out),
      B, L_v, N, D_h, seq_len, eps);
}

void qkv_split_norm2_bf16(
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    void* q_a_out,
    void* k_a_out,
    int B, int L_a, int N, int D_h, float eps_a,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_u_out,
    void* k_u_out,
    int L_u, float eps_u,
    cudaStream_t stream)
{
  if (B != 1 || L_a <= 0 || L_u <= 0 || N <= 0 || D_h <= 0) return;
  const int dim = N * D_h;
  qkv_split_norm2_bf16_kernel<<<L_a + L_u, 256, 33 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_a),
      reinterpret_cast<const __nv_bfloat16*>(norm_a_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_a_k_w),
      reinterpret_cast<__nv_bfloat16*>(q_a_out),
      reinterpret_cast<__nv_bfloat16*>(k_a_out),
      L_a, dim, eps_a,
      reinterpret_cast<const __nv_bfloat16*>(packed_u),
      reinterpret_cast<const __nv_bfloat16*>(norm_u_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_u_k_w),
      reinterpret_cast<__nv_bfloat16*>(q_u_out),
      reinterpret_cast<__nv_bfloat16*>(k_u_out),
      L_u, eps_u);
}

void qkv_split_bias_norm_rope_v_bf16(
    const void*  packed_qkv,
    const void*  qkv_bias,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_rope_out,
    void*        k_rope_out,
    void*        v_out,
    int B, int L_v, int N, int D_h, int seq_len, float eps,
    cudaStream_t stream)
{
  if (B <= 0 || L_v <= 0 || N <= 0 || D_h <= 0) return;
  qkv_split_bias_norm_rope_v_bf16_kernel<<<B * L_v, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_qkv),
      reinterpret_cast<const __nv_bfloat16*>(qkv_bias),
      reinterpret_cast<const __nv_bfloat16*>(norm_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_k_w),
      freqs_re, freqs_im,
      reinterpret_cast<__nv_bfloat16*>(q_rope_out),
      reinterpret_cast<__nv_bfloat16*>(k_rope_out),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      L_v, N, D_h, seq_len, eps);
}

void qkv_split_bias_norm_rope_v_cat_bf16(
    const void*  packed_qkv,
    const void*  qkv_bias,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_cat_out,
    void*        k_cat_out,
    void*        v_cat_out,
    int B, int total_L, int video_offset, int L_v,
    int N, int D_h, int seq_len, float eps,
    cudaStream_t stream)
{
  if (B <= 0 || total_L <= 0 || L_v <= 0 || N <= 0 || D_h <= 0) return;
  qkv_split_bias_norm_rope_v_cat_bf16_kernel<<<dim3(L_v, B, 1), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_qkv),
      reinterpret_cast<const __nv_bfloat16*>(qkv_bias),
      reinterpret_cast<const __nv_bfloat16*>(norm_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_k_w),
      freqs_re, freqs_im,
      reinterpret_cast<__nv_bfloat16*>(q_cat_out),
      reinterpret_cast<__nv_bfloat16*>(k_cat_out),
      reinterpret_cast<__nv_bfloat16*>(v_cat_out),
      total_L, video_offset, L_v, N, D_h, seq_len, eps);
}

void qkv_split_norm2_cat_bf16(
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out,
    void* k_cat_out,
    void* v_cat_out,
    int B, int total_L, int L_v, int L_a, int L_u,
    int N, int D_h, float eps_a, float eps_u,
    cudaStream_t stream)
{
  if (B != 1 || total_L <= 0 || L_a <= 0 || L_u <= 0 || N <= 0 || D_h <= 0) return;
  const int dim = N * D_h;
  qkv_split_norm2_cat_bf16_kernel<<<L_a + L_u, 256, 33 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_a),
      reinterpret_cast<const __nv_bfloat16*>(norm_a_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_a_k_w),
      reinterpret_cast<const __nv_bfloat16*>(packed_u),
      reinterpret_cast<const __nv_bfloat16*>(norm_u_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_u_k_w),
      reinterpret_cast<__nv_bfloat16*>(q_cat_out),
      reinterpret_cast<__nv_bfloat16*>(k_cat_out),
      reinterpret_cast<__nv_bfloat16*>(v_cat_out),
      total_L, L_v, L_a, L_u, dim, eps_a, eps_u);
}

void qkv_split_joint3_cat_bf16(
    const void* packed_v,
    const void* qkv_v_bias,
    const void* norm_v_q_w,
    const void* norm_v_k_w,
    const float* freqs_re,
    const float* freqs_im,
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out,
    void* k_cat_out,
    void* v_cat_out,
    int B, int total_L, int L_v, int L_a, int L_u,
    int N, int D_h, int seq_len,
    float eps_v, float eps_a, float eps_u,
    cudaStream_t stream)
{
  if (B != 1 || total_L <= 0 || L_v <= 0 || L_a <= 0 || L_u <= 0 ||
      N <= 0 || D_h <= 0) return;
  qkv_split_joint3_cat_bf16_kernel
      <<<L_v + L_a + L_u, 256, 33 * sizeof(float), stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(packed_v),
          reinterpret_cast<const __nv_bfloat16*>(qkv_v_bias),
          reinterpret_cast<const __nv_bfloat16*>(norm_v_q_w),
          reinterpret_cast<const __nv_bfloat16*>(norm_v_k_w),
          freqs_re, freqs_im,
          reinterpret_cast<const __nv_bfloat16*>(packed_a),
          reinterpret_cast<const __nv_bfloat16*>(norm_a_q_w),
          reinterpret_cast<const __nv_bfloat16*>(norm_a_k_w),
          reinterpret_cast<const __nv_bfloat16*>(packed_u),
          reinterpret_cast<const __nv_bfloat16*>(norm_u_q_w),
          reinterpret_cast<const __nv_bfloat16*>(norm_u_k_w),
          reinterpret_cast<__nv_bfloat16*>(q_cat_out),
          reinterpret_cast<__nv_bfloat16*>(k_cat_out),
          reinterpret_cast<__nv_bfloat16*>(v_cat_out),
          total_L, L_v, L_a, L_u, N, D_h, seq_len,
          eps_v, eps_a, eps_u);
}

}  // namespace quantize
}  // namespace flash_rt
