// SPDX-License-Identifier: Apache-2.0
//
// SageAttention2-compatible output variant of the QKV split+norm+RoPE kernel:
// q and k are emitted as signed int8 with caller-supplied static per-tensor
// scales (absmax/127 contract), v as fp16. The arithmetic before the store is
// identical to qkv_split_bias_norm_rope_v_cat2_bf16; only the store format
// changes, because the Sage2 SM89 attention core consumes int8 Q/K and fp16 V.
#include "qkv_split_norm_rope_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

template <int NP, int DH2 = 0>
__global__ void qkv_split_bias_norm_rope_v_cat2_sage_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ qkv_bias,
    const __nv_bfloat16* __restrict__ norm_q_w,
    const __nv_bfloat16* __restrict__ norm_k_w,
    const float* __restrict__ freqs_re,
    const float* __restrict__ freqs_im,
    int8_t* __restrict__ q_cat_out,
    int8_t* __restrict__ k_cat_out,
    __half* __restrict__ v_cat_out,
    int Lq, int off_q, int Lkv, int off_kv, int L, int N, int D_h,
    int seq_len, float eps, float q_scale, float k_scale,
    int Lvo, int off_v,
    const float* __restrict__ q_scale_h, const float* __restrict__ k_scale_h,
    const int* __restrict__ ring_origin, int ring_R) {
  const int row = blockIdx.x;
  const int b = blockIdx.y;
  const int t = row % L;
  const int dim = N * D_h;
  const int dim2 = dim >> 1;
  const long long row_off = ((long long)b * L + row) * 3 * dim;
  // q lands in a [B*L, dim] activation buffer; k and v land directly in the
  // KV cache, whose rows are [B*W, dim] with the write cursor as row offset.
  // Writing the cache from the projection removes the intermediate kh/vh
  // buffers and the copy that moved them into it.
  const long long out_q  = ((long long)b * Lq  + off_q  + row) * dim;
  // KV ring (ring_R > 0): cache row (ring_origin[0] + off_kv + row) mod ring_R,
  // and rows [0, 64) are mirrored at [ring_R, ring_R + 64) for the attention
  // tiles that straddle the end (see qk_int8_sv_f8_bf16_nhd_d128_strided_ring).
  int krow = off_kv + row;
  if (ring_R > 0) krow = (ring_origin[0] + krow) % ring_R;
  const long long out_kv = ((long long)b * Lkv + krow) * dim;
  const long long out_kv_mirror = (ring_R > 0 && krow < 64)
      ? ((long long)b * Lkv + krow + ring_R) * dim : -1;
  // V may go to a different buffer than K (dense fp16 staging ahead of the
  // per-channel fp8 transpose); Lvo/off_v address it when set, else follow K.
  const long long out_v  = ((long long)b * (Lvo > 0 ? Lvo : Lkv)
                            + (off_v >= 0 ? off_v : off_kv) + row) * dim;
  const long long fr_row = (long long)t * (D_h >> 1);
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_off);
  const __nv_bfloat162* bias2 =
      reinterpret_cast<const __nv_bfloat162*>(qkv_bias);

  __shared__ float reduce_pad[33];
  float sq = 0.f, sk = 0.f;
  // NP > 0 (dim2 == NP * 256, 256 threads): the pass-1 q/k sums are kept in
  // registers for pass 2 and every load of a thread is issued together; the
  // arithmetic and the accumulation order are those of the loops below.
  float qs[NP > 0 ? NP : 1][2], ks[NP > 0 ? NP : 1][2];
  if constexpr (NP > 0) {
#pragma unroll
    for (int k = 0; k < NP; ++k) {
      const int i = threadIdx.x + k * 256;
      const __nv_bfloat162 qv = qkv2[i], kv = qkv2[dim2 + i];
      const __nv_bfloat162 bq = bias2[i], bk = bias2[dim2 + i];
      qs[k][0] = __bfloat162float(qv.x) + __bfloat162float(bq.x);
      qs[k][1] = __bfloat162float(qv.y) + __bfloat162float(bq.y);
      ks[k][0] = __bfloat162float(kv.x) + __bfloat162float(bk.x);
      ks[k][1] = __bfloat162float(kv.y) + __bfloat162float(bk.y);
    }
#pragma unroll
    for (int k = 0; k < NP; ++k) {
      sq += qs[k][0] * qs[k][0] + qs[k][1] * qs[k][1];
      sk += ks[k][0] * ks[k][0] + ks[k][1] * ks[k][1];
    }
  } else
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
  // DH2 > 0: the head width is a compile-time constant, so the per-element
  // head/pair split below is a shift and a mask instead of an integer divide.
  const int dh2 = DH2 > 0 ? DH2 : (D_h >> 1);
  const bool apply_rope = (t < seq_len);

#pragma unroll
  for (int it = 0; it < (NP > 0 ? NP : 1); ++it)
  for (int p = (NP > 0 ? threadIdx.x + it * 256 : threadIdx.x); p < dim2;
       p += (NP > 0 ? dim2 : blockDim.x)) {
    const int n = p / dh2;
    const int c_pair = p - n * dh2;
    const int c0 = n * D_h + 2 * c_pair;
    const int c1 = c0 + 1;

    __nv_bfloat162 vv = qkv2[2 * dim2 + p];
    __nv_bfloat162 bv = bias2[2 * dim2 + p];
    __nv_bfloat162 wq = reinterpret_cast<const __nv_bfloat162*>(norm_q_w)[p];
    __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(norm_k_w)[p];
    float qsum0, qsum1, ksum0, ksum1;
    if constexpr (NP > 0) {
      qsum0 = qs[it][0]; qsum1 = qs[it][1]; ksum0 = ks[it][0]; ksum1 = ks[it][1];
    } else {
      __nv_bfloat162 qv = qkv2[p];
      __nv_bfloat162 kv = qkv2[dim2 + p];
      __nv_bfloat162 bq = bias2[p];
      __nv_bfloat162 bk = bias2[dim2 + p];
      qsum0 = __bfloat162float(qv.x) + __bfloat162float(bq.x);
      qsum1 = __bfloat162float(qv.y) + __bfloat162float(bq.y);
      ksum0 = __bfloat162float(kv.x) + __bfloat162float(bk.x);
      ksum1 = __bfloat162float(kv.y) + __bfloat162float(bk.y);
    }

    const float qre_n = qsum0 * inv_q * __bfloat162float(wq.x);
    const float qim_n = qsum1 * inv_q * __bfloat162float(wq.y);
    const float kre_n = ksum0 * inv_k * __bfloat162float(wk.x);
    const float kim_n = ksum1 * inv_k * __bfloat162float(wk.y);

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

    // Signed int8 with the Sage contract (absmax / 127); the static scales are
    // calibrated on real warmup data, so the clamp only guards outliers.
    // q_scale_h / k_scale_h (optional): one static scale per head instead.
    const float iq = 1.0f / (q_scale_h ? q_scale_h[n] : q_scale);
    const float ik = 1.0f / (k_scale_h ? k_scale_h[n] : k_scale);
    int q0 = __float2int_rn(qre_out * iq);
    int q1 = __float2int_rn(qim_out * iq);
    int k0 = __float2int_rn(kre_out * ik);
    int k1 = __float2int_rn(kim_out * ik);
    q0 = max(-127, min(127, q0)); q1 = max(-127, min(127, q1));
    k0 = max(-127, min(127, k0)); k1 = max(-127, min(127, k1));
    // c0 is even: the pair is one aligned 16-bit store
    *reinterpret_cast<char2*>(q_cat_out + out_q + c0) = make_char2((char)q0, (char)q1);
    *reinterpret_cast<char2*>(k_cat_out + out_kv + c0) = make_char2((char)k0, (char)k1);
    if (out_kv_mirror >= 0)
      *reinterpret_cast<char2*>(k_cat_out + out_kv_mirror + c0) = make_char2((char)k0, (char)k1);
    reinterpret_cast<__half2*>(v_cat_out + out_v)[p] =
        __floats2half2_rn(__bfloat162float(vv.x) + __bfloat162float(bv.x),
                          __bfloat162float(vv.y) + __bfloat162float(bv.y));
  }
}

void qkv_split_bias_norm_rope_v_cat2_sage(
    const void*  packed_qkv,
    const void*  qkv_bias,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_out,
    void*        k_out,
    void*        v_out,
    int B, int Lq, int off_q, int Lkv, int off_kv, int L_v,
    int N, int D_h, int seq_len, float eps,
    float q_scale, float k_scale,
    int v_out_rows, int off_v,
    cudaStream_t stream,
    const float* q_scale_h, const float* k_scale_h,
    const int* ring_origin, int ring_R)
{
  if (B <= 0 || Lq <= 0 || Lkv <= 0 || L_v <= 0 || N <= 0 || D_h <= 0) return;
  auto kfn = (N * D_h == 3072 && D_h == 128)
                 ? qkv_split_bias_norm_rope_v_cat2_sage_kernel<6, 64>
                 : (N * D_h == 3072) ? qkv_split_bias_norm_rope_v_cat2_sage_kernel<6>
                                     : qkv_split_bias_norm_rope_v_cat2_sage_kernel<0>;
  kfn<<<dim3(L_v, B, 1), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_qkv),
      reinterpret_cast<const __nv_bfloat16*>(qkv_bias),
      reinterpret_cast<const __nv_bfloat16*>(norm_q_w),
      reinterpret_cast<const __nv_bfloat16*>(norm_k_w),
      freqs_re, freqs_im,
      reinterpret_cast<int8_t*>(q_out),
      reinterpret_cast<int8_t*>(k_out),
      reinterpret_cast<__half*>(v_out),
      Lq, off_q, Lkv, off_kv, L_v, N, D_h, seq_len, eps, q_scale, k_scale,
      v_out_rows, off_v, q_scale_h, k_scale_h, ring_origin, ring_R);
}

// The fp8 PV MMA consumes V's 16-byte k-packs in the bit-swizzled order
// (b0,b3,b1,b2) of the intra-pack index; measured against the f16 path with a
// one-hot-P probe on every k. Store each key at its MMA position so the
// kernel's P and V k indices agree.
__device__ __forceinline__ int v_fp8_mma_pos(int lo) {
    return (lo & 1) | (((lo >> 3) & 1) << 1) | (((lo >> 1) & 3) << 2);
}

// Transpose + quantize an fp16 V block into the SageAttention2 per-channel
// layout [B, D, H, PAD] (fp8 e4m3), writing at KV offset `off`. Reads are
// coalesced along D; each thread performs one 1-byte store.
__global__ void v_transpose_f8_kernel(const __half* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      long total, int H, int D, int LT,
                                      int off, int PAD, float inv_scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int d = (int)(i % D);
    long r = i / D;
    const int h = (int)(r % H);
    r /= H;
    const int t = (int)(r % LT);
    const int b = (int)(r / LT);
    const float v = __half2float(src[i]);
    const __nv_fp8_storage_t byte = __nv_cvt_float_to_fp8(
        v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    const int pos = off + t;
    dst[(((long)b * D + d) * H + h) * PAD + (pos & ~15) + v_fp8_mma_pos(pos & 15)] =
        (uint8_t)byte;
}

// Coalesced variant: one block transposes a 32-token x 32-channel tile through
// shared memory. The plain kernel writes one byte per thread with consecutive
// threads on different `d`, i.e. 58 KB apart; here consecutive threads take
// consecutive token positions, so a warp's 32 stores fill two whole 16-byte
// k-packs. Requires `off + LT` to stay inside the caller's padded row and
// `off` to be 16-aligned, which every KV write in this pipeline is (128- and
// 16-token appends at a 16-aligned cursor).
__global__ void v_transpose_f8_tile_kernel(const __half* __restrict__ src,
                                           uint8_t* __restrict__ dst,
                                           int H, int D, int LT,
                                           int off, int PAD, float inv_scale) {
    __shared__ __half tile[32][33];
    const int d0 = blockIdx.x * 32;
    const int t0 = blockIdx.y * 32;
    const int bh = blockIdx.z;
    const int b = bh / H;
    const int h = bh - b * H;
    const long src_row = (long)(b * LT) * H * D + (long)h * D;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int i = threadIdx.x + it * 256;
        const int rr = i >> 5, cc = i & 31;
        const int t = t0 + rr, d = d0 + cc;
        tile[rr][cc] = (t < LT && d < D)
            ? src[src_row + (long)t * (H * D) + d] : __float2half(0.f);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int i = threadIdx.x + it * 256;
        const int rr = i >> 5, cc = i & 31;
        const int d = d0 + rr, t = t0 + cc;
        if (d >= D || t >= LT) continue;
        const float v = __half2float(tile[cc][rr]);
        const __nv_fp8_storage_t byte = __nv_cvt_float_to_fp8(
            v * inv_scale, __NV_SATFINITE, __NV_E4M3);
        const int pos = off + t;
        dst[(((long)b * D + d) * H + h) * PAD + (pos & ~15) + v_fp8_mma_pos(pos & 15)] =
            (uint8_t)byte;
    }
}

void v_transpose_f8(const void* src, void* dst, long total,
                    int H, int D, int LT, int off, int PAD, float scale,
                    cudaStream_t stream) {
    long blocks = (total + 255) / 256;
    v_transpose_f8_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(src), reinterpret_cast<uint8_t*>(dst),
        total, H, D, LT, off, PAD, 1.0f / scale);
}

void v_transpose_f8_tile(const void* src, void* dst,
                         int B, int H, int D, int LT, int off, int PAD,
                         float scale, cudaStream_t stream) {
    dim3 grid((D + 31) / 32, (LT + 31) / 32, B * H);
    v_transpose_f8_tile_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(src), reinterpret_cast<uint8_t*>(dst),
        H, D, LT, off, PAD, 1.0f / scale);
}

// The K ring's window move: where kv_window_shift_sage moves the surviving K
// rows down by `drop`, the ring advances its origin by `drop` instead
// (origin = (origin + drop) mod ring_R) and no K row moves.
__global__ void kv_ring_advance_kernel(int* origin, int drop, int ring_R) {
    origin[0] = (origin[0] + drop) % ring_R;
}

int kv_ring_advance(void* origin, int drop, int ring_R, cudaStream_t stream) {
    if (!origin || ring_R <= 0 || drop < 0) return -1;
    kv_ring_advance_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int*>(origin), drop, ring_R);
    return static_cast<int>(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// KV window shift for the static prefix-layout cache: the surviving rows
// [drop, drop + keep) move down to [0, keep) in place, in one pass.
//
// K [B, total, H*D] (1-byte elements): a CTA owns one 16-byte column slice of
// one batch element, stages that slice of every surviving row in shared
// memory, then writes it back shifted -- source and destination overlap only
// within the slice, and all of its reads complete before the barrier.
__global__ void kv_shift_k_kernel(uint8_t* __restrict__ k, long long batch_stride,
                                  int row_bytes, int keep, int drop) {
  extern __shared__ uint4 stage[];
  uint8_t* base = k + (long long)blockIdx.y * batch_stride + (long long)blockIdx.x * 16;
  for (int r = threadIdx.x; r < keep; r += blockDim.x)
    stage[r] = *reinterpret_cast<const uint4*>(base + (long long)(r + drop) * row_bytes);
  __syncthreads();
  for (int r = threadIdx.x; r < keep; r += blockDim.x)
    *reinterpret_cast<uint4*>(base + (long long)r * row_bytes) = stage[r];
}

// Transposed V [rows, pad] (1-byte elements, rows = B*D*H): the window runs
// along each row, so one warp moves one row: every lane loads its 16-byte
// chunks of [drop, drop + keep) into registers, then the warp writes them to
// [0, keep). drop and keep are multiples of 16, which keeps each 16-key group
// (and its fp8-MMA position permutation) intact.
template <int MAXC>
__global__ void kv_shift_vt_kernel(uint8_t* __restrict__ v, long long rows, int pad,
                                   int keep, int drop) {
  const long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  uint8_t* r = v + row * pad;
  const int nch = keep >> 4;
  uint4 buf[MAXC];
#pragma unroll
  for (int c = 0; c < MAXC; ++c) {
    const int ch = lane + c * 32;
    if (ch < nch) buf[c] = *reinterpret_cast<const uint4*>(r + drop + ch * 16);
  }
  __syncwarp();
#pragma unroll
  for (int c = 0; c < MAXC; ++c) {
    const int ch = lane + c * 32;
    if (ch < nch) *reinterpret_cast<uint4*>(r + ch * 16) = buf[c];
  }
}

int kv_window_shift_sage(void* k, void* vt, int B, int total, int row_bytes,
                         int pad, int keep, int drop, cudaStream_t stream) {
  if ((row_bytes % 16) || (keep % 16) || (drop % 16) || (pad % 16)) return -1;
  if (keep <= 0) return 0;
  const size_t smem = (size_t)keep * 16;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kv_shift_k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         99 * 1024);
    attr = true;
  }
  if (smem > 99 * 1024) return -2;
  // k == nullptr: V only (the K side is a ring, see kv_ring_advance)
  if (k != nullptr)
    kv_shift_k_kernel<<<dim3(row_bytes / 16, B), 256, smem, stream>>>(
        reinterpret_cast<uint8_t*>(k), (long long)total * row_bytes, row_bytes, keep, drop);
  const long long rows = (long long)B * row_bytes;   // D*H rows per batch element
  const int nch = keep / 16;
  const dim3 grid((unsigned)((rows + 7) / 8));
  if (nch <= 32 * 6)
    kv_shift_vt_kernel<6><<<grid, 256, 0, stream>>>(reinterpret_cast<uint8_t*>(vt), rows, pad, keep, drop);
  else
    return -3;
  return (int)cudaGetLastError();
}

}  // namespace quantize
}  // namespace flash_rt
