// SPDX-License-Identifier: Apache-2.0
//
// FLA-style hand-tuned output_o kernel for the GDN/WY chunked linear
// attention output reconstruction. Mirrors the FLA Triton chunk_fwd_kernel_o
// (fla/ops/common/chunk_o.py) but as a single CTA-resident CUDA kernel
// using raw mma.sync + cp.async pipelining.
//
// Algorithm per CTA = (V_block i_v, chunk i_t, v-head i_h):
//   For i_k in [0, K/BK):
//     load q_pack[i_t,i_h,:,i_k*BK..+BK]
//     load k_pack_hv[i_t,i_h,:,i_k*BK..+BK]    (GQA-expanded)
//     load h[i_t,i_h,i_k*BK..+BK, v_base..+BV]
//     b_o += q @ h     (BT x BV fp32)
//     b_A += q @ k^T   (BT x BT fp32)
//   Apply g_cumsum decay: b_o *= exp(g); b_A *= safe_exp(g_row - g_col)
//   Apply causal mask: b_A[i,j] = 0 if j > i
//   Load v_pack[i_t,i_h,:,v_base..+BV]
//   b_o = b_o * scale + (b_A cast bf16) @ v * scale
//   Store b_o (bf16) to out
//
// Inputs match the cublasLt packed_qkv signature where possible:
//   q_pack:     (NT, num_v_heads, 64, head_dim) bf16  packed-per-chunk
//   k_pack_hv:  (NT, num_v_heads, 64, head_dim) bf16  GQA-expanded packed k
//                or raw k_l2 (S, num_k_heads, head_dim) for the rawk entry
//   v_pack:     (NT, num_v_heads, 64, head_dim) bf16  packed v
//   h:          (NT, num_v_heads, head_dim, head_dim) bf16  chunk-prologue states
//   g_cumsum:   (S, num_v_heads) bf16
//   out:        (S, num_v_heads, head_dim) bf16
//
// Geometry: head_dim = 128 fixed, BT = 64, BV = 64. Grid (V/BV=2, NT, num_v_heads).
// 4 warps per CTA; each warp owns 16 of the 64 BT rows.

#include "gated_delta_wy_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace flash_rt {
namespace kernels {
namespace linear_attention {

namespace output_o_mma_fla {

__device__ __forceinline__ void cp_async_16(
    void* smem_dst, const void* gmem_src)
{
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
               "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit()
{
  asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait()
{
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4_a(
    const __nv_bfloat16* ptr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_int));
}

__device__ __forceinline__ void ldmatrix_x4_a_trans(
    const __nv_bfloat16* ptr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_int));
}

__device__ __forceinline__ void ldmatrix_x2_trans_b(
    const __nv_bfloat16* ptr,
    uint32_t& r0, uint32_t& r1)
{
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
               "{%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_int));
}

// ldmatrix.x2 (no .trans). Loads 2 8x8 b16 matrices in row-major reg layout.
// When the source matrix is stored as (N-axis-rows, K-axis-cols) the per-thread
// reg layout coincides bit-for-bit with the mma B operand requirements, so
// this primitive serves as a B-operand load for "naturally-transposed" source.
__device__ __forceinline__ void ldmatrix_x2_b(
    const __nv_bfloat16* ptr,
    uint32_t& r0, uint32_t& r1)
{
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
               "{%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_int));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                 "r"(b0), "r"(b1));
}

constexpr int kBT = 64;
constexpr int kK  = 128;
constexpr int kV  = 128;
constexpr int kBV = 64;
constexpr int kBK = 64;
constexpr int kThreads = 128;
// sQ + sK + sH + sV + sA + sG (bf16):
//   = (64*64 + 64*64 + 64*64 + 64*64 + 64*64 + 64) * 2 bytes
//   = (5*4096 + 64) * 2 = 41088 bytes
constexpr size_t kSmemBytes = 41088;

__global__ void output_o_kernel(
    const __nv_bfloat16* __restrict__ q_pack,
    const __nv_bfloat16* __restrict__ k_input,
    const __nv_bfloat16* __restrict__ v_pack,
    const __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ g_bf16,
    __nv_bfloat16*       __restrict__ out,
    int S, int H, int NT, float scale,
    int num_k_heads, int qk_group, bool k_input_is_raw)
{
  const int i_v = blockIdx.x;
  const int i_t = blockIdx.y;
  const int i_h = blockIdx.z;
  const int kh = i_h / qk_group;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;          // 0..3
  const int lane = tid & 31;
  const int v_base = i_v * kBV;

  extern __shared__ __nv_bfloat16 smem_raw[];
  __nv_bfloat16* sQ = smem_raw;                       // (BT, BK)
  __nv_bfloat16* sK = sQ + kBT * kBK;                 // (BT, BK)
  __nv_bfloat16* sH = sK + kBT * kBK;                 // (BK, BV)
  __nv_bfloat16* sV = sH + kBK * kBV;                 // (BT, BV)
  __nv_bfloat16* sA = sV + kBT * kBV;                 // (BT, BT) — for cast of b_A
  __nv_bfloat16* sG = sA + kBT * kBT;                 // (BT) bf16
  // Total: 8+8+8+8+8+0.125 = ~40 KB

  // Per-warp accumulators:
  //   b_o: 16 BT rows x BV=64 cols  -> 8 n8 tiles, 4 fp32/tile = 32 fp32/thread
  //   b_A: 16 BT rows x BT=64 cols  -> 8 n8 tiles, 4 fp32/tile = 32 fp32/thread
  float b_o[8][4];
  float b_A[8][4];
  #pragma unroll
  for (int nj = 0; nj < 8; ++nj)
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      b_o[nj][k] = 0.f;
      b_A[nj][k] = 0.f;
    }

  const int m_row_base = warp_id * 16;

  // K-loop: for each BK=64 slice of K=128, accumulate q@h and q@k.
  #pragma unroll
  for (int i_k = 0; i_k < kK / kBK; ++i_k) {
    const int k_base = i_k * kBK;

    // Load sQ: q_pack[i_t, i_h, 0..63, k_base..k_base+64]
    if (tid < kBT) {
      int row = tid;
      const __nv_bfloat16* src = q_pack
          + ((size_t)i_t * H + i_h) * kBT * kK
          + row * kK + k_base;
      #pragma unroll
      for (int q = 0; q < kBK / 8; ++q)
        cp_async_16(&sQ[row * kBK + q * 8], &src[q * 8]);
    }
    // Load sK. The packed entry consumes k_pack_hv[i_t, i_h, row, k].
    // The rawk entry reads k_l2[t, kh, k] directly and avoids the chunk_h
    // GQA-expanded K side-write plus this packed K reread.
    if (tid < kBT) {
      int row = tid;
      const int t_global = i_t * kBT + row;
      const __nv_bfloat16* src = k_input_is_raw
          ? (k_input + ((size_t)t_global * num_k_heads + kh) * kK + k_base)
          : (k_input + ((size_t)i_t * H + i_h) * kBT * kK
             + row * kK + k_base);
      #pragma unroll
      for (int q = 0; q < kBK / 8; ++q)
        if (t_global < S)
          cp_async_16(&sK[row * kBK + q * 8], &src[q * 8]);
        else {
          #pragma unroll
          for (int j = 0; j < 8; ++j)
            sK[row * kBK + q * 8 + j] = __float2bfloat16(0.f);
        }
    }
    // Load sH: h[i_t, i_h, k_base..k_base+64, v_base..v_base+64]
    if (tid < kBK) {
      int row = tid;
      const __nv_bfloat16* src = h
          + ((size_t)i_t * H + i_h) * kK * kV
          + (k_base + row) * kV + v_base;
      #pragma unroll
      for (int q = 0; q < kBV / 8; ++q)
        cp_async_16(&sH[row * kBV + q * 8], &src[q * 8]);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // b_o += sQ @ sH    (mma C += A @ B with A=(BT,BK), B=(BK,BV))
    // K reduction = BK = 64 / 16 = 4 k16 steps
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int k_col_base = kk * 16;
      int row = lane % 16;
      int col_grp = (lane / 16) * 8;
      ldmatrix_x4_a(
          &sQ[(m_row_base + row) * kBK + k_col_base + col_grp],
          af[0], af[1], af[2], af[3]);
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int k_row_base = kk * 16;
        int n_col_base = nj * 8;
        int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
        ldmatrix_x2_trans_b(
            &sH[(k_row_base + b_row) * kBV + n_col_base],
            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(b_o[nj][0], b_o[nj][1], b_o[nj][2], b_o[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }

    // b_A += sQ @ sK^T  (mma: A=(BT,BK), B=sK^T via ldmatrix.x4.trans on (BK,BT) view)
    // sK is stored row-major as (BT, BK). To get (K_red, BT) view for B with N=BT,
    // use x4.trans pattern (same as kv mma in chunk_h).
    // mma C += A(BT,BK) @ B(K=BK, N=BT) where K_reduce = BK = 64.
    // K_reduce slices: BK/16 = 4 k16 steps.
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int k_col_base = kk * 16;
      int row = lane % 16;
      int col_grp = (lane / 16) * 8;
      ldmatrix_x4_a(
          &sQ[(m_row_base + row) * kBK + k_col_base + col_grp],
          af[0], af[1], af[2], af[3]);
      // sK is (BT=N-axis rows, BK=K-axis cols). For mma B operand (K, N) the
      // per-thread reg layout matches an x2 NO-TRANS load of a source in
      // (N, K) layout (= our sK). Lane addressing covers source rows
      // [nj*8, nj*8+8) and 2 K-col tiles at offsets {0, 8}.
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int n_col_base = nj * 8;
        int k_row_base = kk * 16;
        int b_row_in_src = n_col_base + (lane % 8);
        int b_col_in_src = k_row_base + ((lane / 8) % 2) * 8;
        ldmatrix_x2_b(
            &sK[b_row_in_src * kBK + b_col_in_src],
            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(b_A[nj][0], b_A[nj][1], b_A[nj][2], b_A[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }
    __syncthreads();
  }

  // Load g_cumsum for this chunk.
  if (tid < kBT) {
    int row = tid;
    int t_global = i_t * kBT + row;
    sG[row] = (t_global < S)
        ? g_bf16[(size_t)t_global * H + i_h]
        : __float2bfloat16(0.f);
  }
  __syncthreads();

  // Apply g decay and causal mask.
  // b_o[s, v] *= exp(g[s]);  b_A[s, j] *= safe_exp(g[s] - g[j]); mask j > s -> 0
  // Per-thread c-frag layout for tile (mi=warp owns 16 rows starting m_row_base):
  //   reg index kp in {0..3}: (row = m_row_base + lane/4 + (kp/2)*8,
  //                            col = n_col_base + (lane%4)*2 + (kp%2))
  // b_o cols are V cols; b_A cols are BT (chunk-local).
  {
    const int t_global_start = i_t * kBT;
    const int t_count = (t_global_start + kBT <= S) ? kBT : (S - t_global_start);
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int n_col_base = nj * 8;
      #pragma unroll
      for (int kp = 0; kp < 4; ++kp) {
        int row = m_row_base + (lane / 4) + (kp / 2) * 8;
        int col = n_col_base + (lane % 4) * 2 + (kp % 2);
        float gr = (row < t_count)
            ? __bfloat162float(sG[row]) : 0.f;
        // b_o: scale by exp(g[row])
        float scale_o = __expf(gr);
        b_o[nj][kp] *= scale_o;
        // b_A: scale by safe_exp(g[row] - g[col]) and mask
        float gc = (col < t_count)
            ? __bfloat162float(sG[col]) : 0.f;
        float dlt = gr - gc;
        // FlashRT convention: plain exp (no clamp). For monotone log gates this
        // agrees with FLA safe_exp.
        float scale_A = __expf(dlt);
        if (col > row || row >= t_count || col >= t_count) scale_A = 0.f;
        b_A[nj][kp] *= scale_A;
      }
    }
  }

  // Cast b_A fp32 -> bf16 -> sA so the local mma can use it as B operand.
  // sA layout: (BT, BT) row-major. b_A c-frag = (BT row, BT col).
  {
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int n_col_base = nj * 8;
      int r = lane / 4;
      int c = (lane % 4) * 2;
      int r0 = m_row_base + r;
      int r1 = m_row_base + r + 8;
      sA[r0 * kBT + n_col_base + c    ] = __float2bfloat16(b_A[nj][0]);
      sA[r0 * kBT + n_col_base + c + 1] = __float2bfloat16(b_A[nj][1]);
      sA[r1 * kBT + n_col_base + c    ] = __float2bfloat16(b_A[nj][2]);
      sA[r1 * kBT + n_col_base + c + 1] = __float2bfloat16(b_A[nj][3]);
    }
  }

  // Load sV: v_pack[i_t, i_h, 0..63, v_base..v_base+64]
  if (tid < kBT) {
    int row = tid;
    const __nv_bfloat16* src = v_pack
        + ((size_t)i_t * H + i_h) * kBT * kV
        + row * kV + v_base;
    #pragma unroll
    for (int q = 0; q < kBV / 8; ++q)
      cp_async_16(&sV[row * kBV + q * 8], &src[q * 8]);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // local = sA @ sV  (mma: A=(BT,BT), B=(BT,BV), C=(BT,BV) fp32, then add to b_o*scale)
  // We accumulate directly into b_o after scaling.
  // First scale b_o by `scale`.
  #pragma unroll
  for (int nj = 0; nj < 8; ++nj)
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      b_o[nj][kp] *= scale;

  // Compute local in fp32 accumulator then add scaled.
  float b_local[8][4];
  #pragma unroll
  for (int nj = 0; nj < 8; ++nj)
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      b_local[nj][kp] = 0.f;
  // K reduction = BT = 64 / 16 = 4 k16 steps.
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t af[4];
    int k_col_base = kk * 16;
    int row = lane % 16;
    int col_grp = (lane / 16) * 8;
    ldmatrix_x4_a(
        &sA[(m_row_base + row) * kBT + k_col_base + col_grp],
        af[0], af[1], af[2], af[3]);
    uint32_t bf[8][2];
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int k_row_base = kk * 16;
      int n_col_base = nj * 8;
      int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
      ldmatrix_x2_trans_b(
          &sV[(k_row_base + b_row) * kBV + n_col_base],
          bf[nj][0], bf[nj][1]);
    }
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj)
      mma_m16n8k16(b_local[nj][0], b_local[nj][1],
                   b_local[nj][2], b_local[nj][3],
                   af[0], af[1], af[2], af[3],
                   bf[nj][0], bf[nj][1]);
  }

  // Final: out = b_o + b_local * scale
  // Store coalesced to global out (S, H, V) at (i_t*BT + row, i_h, v_base + col)
  {
    const int t_global_start = i_t * kBT;
    const int t_count = (t_global_start + kBT <= S) ? kBT : (S - t_global_start);
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int n_col_base = nj * 8;
      #pragma unroll
      for (int kp = 0; kp < 4; ++kp) {
        int row = m_row_base + (lane / 4) + (kp / 2) * 8;
        int col = n_col_base + (lane % 4) * 2 + (kp % 2);
        if (row >= t_count) continue;
        float val = b_o[nj][kp] + b_local[nj][kp] * scale;
        out[((size_t)(t_global_start + row) * H + i_h) * kV + v_base + col]
            = __float2bfloat16(val);
      }
    }
  }
}

}  // namespace output_o_mma_fla

void gdn_wy_output_o_b64_bf16_mma_fla(
    const void* q_pack,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h,
    const void* g_cumsum,
    void*       out,
    int S,
    int num_v_heads,
    int head_dim,
    float scale,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  if (head_dim != output_o_mma_fla::kK) {
    throw std::runtime_error(
        std::string("gdn_wy_output_o_b64_bf16_mma_fla: head_dim must be ") +
        std::to_string(output_o_mma_fla::kK));
  }
  const int NT = (S + output_o_mma_fla::kBT - 1) / output_o_mma_fla::kBT;

  static bool s_attr_set = false;
  if (!s_attr_set) {
    cudaError_t err = cudaFuncSetAttribute(
        output_o_mma_fla::output_o_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(output_o_mma_fla::kSmemBytes));
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("gdn_wy_output_o_b64_bf16_mma_fla: "
                      "cudaFuncSetAttribute failed: ") +
          cudaGetErrorString(err));
    }
    s_attr_set = true;
  }
  dim3 grid(output_o_mma_fla::kV / output_o_mma_fla::kBV, NT, num_v_heads);
  dim3 block(output_o_mma_fla::kThreads, 1, 1);
  output_o_mma_fla::output_o_kernel<<<grid, block,
                                       output_o_mma_fla::kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pack),
      reinterpret_cast<const __nv_bfloat16*>(k_pack_hv),
      reinterpret_cast<const __nv_bfloat16*>(v_pack),
      reinterpret_cast<const __nv_bfloat16*>(h),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, NT, scale, num_v_heads, 1, false);
}

void gdn_wy_output_o_b64_bf16_mma_fla_rawk(
    const void* q_pack,
    const void* k_l2,
    const void* v_pack,
    const void* h,
    const void* g_cumsum,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    float scale,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || qk_group <= 0) return;
  if (head_dim != output_o_mma_fla::kK) {
    throw std::runtime_error(
        std::string("gdn_wy_output_o_b64_bf16_mma_fla_rawk: head_dim must be ") +
        std::to_string(output_o_mma_fla::kK));
  }
  if (num_v_heads % qk_group != 0 ||
      num_v_heads / qk_group != num_k_heads) {
    throw std::runtime_error(
        "gdn_wy_output_o_b64_bf16_mma_fla_rawk: invalid GQA shape");
  }
  const int NT = (S + output_o_mma_fla::kBT - 1) / output_o_mma_fla::kBT;

  static bool s_attr_set = false;
  if (!s_attr_set) {
    cudaError_t err = cudaFuncSetAttribute(
        output_o_mma_fla::output_o_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(output_o_mma_fla::kSmemBytes));
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("gdn_wy_output_o_b64_bf16_mma_fla_rawk: "
                      "cudaFuncSetAttribute failed: ") +
          cudaGetErrorString(err));
    }
    s_attr_set = true;
  }
  dim3 grid(output_o_mma_fla::kV / output_o_mma_fla::kBV, NT, num_v_heads);
  dim3 block(output_o_mma_fla::kThreads, 1, 1);
  output_o_mma_fla::output_o_kernel<<<grid, block,
                                       output_o_mma_fla::kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pack),
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v_pack),
      reinterpret_cast<const __nv_bfloat16*>(h),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, NT, scale, num_k_heads, qk_group, true);
}

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
