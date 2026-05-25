// SPDX-License-Identifier: Apache-2.0
//
// FLA-style hand-tuned chunk_h kernel for the GDN/WY gated DeltaNet
// recurrence used by Qwen3.6. Replaces the cuBLASLt loop with a single
// CTA-resident kernel using raw mma.sync + cp.async 2-stage pipelining.
//
// Algorithm (per FLA chunk_gated_delta_rule_fwd_kernel_h_blockdim64):
//   b_h kept in fp32 registers across the chunk loop (two 64x64 state blocks
//   per CTA covering K=128). For each chunk:
//     1. store b_h (cast bf16) to h_out
//     2. v_acc = w[:, :64] @ b_h1 + w[:, 64:128] @ b_h2
//     3. write raw v_new = u - v_acc to global (BEFORE decay)
//     4. v_dec = v_new * safe_exp(g_last - g_t)
//     5. b_h *= exp(g_last)
//     6. b_h += k_T @ v_dec
//
// Geometry: K=128, V=128 fixed, BT=64. Grid (H, V/BV=2). 4 warps per CTA.
// 4-warp layout: each warp owns 16 M-rows of the 64-row output. Per warp:
// 1 m16 x 8 n8 = 8 mma tiles per K iter, half the register pressure of a
// 2-warp split, enabling 2 CTAs per SM (matches FLA Triton's winning
// autotune config on sm_120: BV=64 num_warps=4 num_stages=2).

#include "gated_delta_wy_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace flash_rt {
namespace kernels {
namespace linear_attention {

namespace mma_fla {

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
constexpr int kThreads = 128;     // 4 warps
constexpr size_t kSmemBytes = 98816;

__global__ void chunk_h_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ g_bf16,
    __nv_bfloat16*       __restrict__ state,        // in/out
    __nv_bfloat16*       __restrict__ h_out,
    __nv_bfloat16*       __restrict__ v_new_out,    // (S, H, V) raw; nullable
    __nv_bfloat16*       __restrict__ v_new_packed, // (NT, H, BT, V) packed; nullable
    __nv_bfloat16*       __restrict__ k_pack_hv,    // (NT, H, BT, K) packed k with GQA expansion; nullable
    int S, int H, int Hg, int qk_group, int NT)
{
  const int i_h = blockIdx.x;
  const int i_v = blockIdx.y;
  const int kh  = i_h / qk_group;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;     // 0..3
  const int lane = tid & 31;
  const int v_base = i_v * kBV;

  extern __shared__ __nv_bfloat16 smem_raw[];
  __nv_bfloat16* sH1   = smem_raw;
  __nv_bfloat16* sH2   = sH1 + 64 * 64;
  __nv_bfloat16* sW[2] = {sH2 + 64*64,         sH2 + 64*64 + kBT*kK};
  __nv_bfloat16* sV[2] = {sW[1] + kBT*kK,      sW[1] + kBT*kK + kBT*kBV};
  __nv_bfloat16* sK[2] = {sV[1] + kBT*kBV,     sV[1] + kBT*kBV + kBT*kK};
  __nv_bfloat16* sG[2] = {sK[1] + kBT*kK,
                          sK[1] + kBT*kK + kBT};

  // Per warp owns 16 of the 64 output rows. mi dimension collapses to 1.
  // Per thread: 8 (n8 tiles) * 4 (fp32 in 16x8 c-frag) = 32 fp32 per block.
  float b_h1[8][4];
  float b_h2[8][4];

  // Initialize state from state (in-place buffer, also written back at end).
  {
    const __nv_bfloat16* base = state + (size_t)i_h * kK * kV;
    const int m_row_base = warp_id * 16;
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int n_col_base = nj * 8;
      int r = lane / 4;
      int c = (lane % 4) * 2;
      int r0 = m_row_base + r;
      int r1 = m_row_base + r + 8;
      int v0 = v_base + n_col_base + c;
      b_h1[nj][0] = static_cast<float>(base[r0       * kV + v0    ]);
      b_h1[nj][1] = static_cast<float>(base[r0       * kV + v0 + 1]);
      b_h1[nj][2] = static_cast<float>(base[r1       * kV + v0    ]);
      b_h1[nj][3] = static_cast<float>(base[r1       * kV + v0 + 1]);
      b_h2[nj][0] = static_cast<float>(base[(r0 + 64)* kV + v0    ]);
      b_h2[nj][1] = static_cast<float>(base[(r0 + 64)* kV + v0 + 1]);
      b_h2[nj][2] = static_cast<float>(base[(r1 + 64)* kV + v0    ]);
      b_h2[nj][3] = static_cast<float>(base[(r1 + 64)* kV + v0 + 1]);
    }
  }

  // Cooperative chunk loaders, now with 128 threads. Each thread loads one
  // BT row of each (w, u, k). For BT=64 < 128 only the first 64 threads
  // participate.
  auto issue_chunk = [&](int i_t_chunk, int t_count, int stage) {
    if (tid < kBT) {
      int row = tid;
      // w_pack: (NT, H, BT, K) bf16. Contiguous BT rows per (chunk, head).
      if (row < t_count) {
        const __nv_bfloat16* src = w
            + ((size_t)i_t_chunk * H + i_h) * kBT * kK + row * kK;
        #pragma unroll
        for (int q = 0; q < kK / 8; ++q)
          cp_async_16(&sW[stage][row * kK + q * 8], &src[q * 8]);
      } else {
        #pragma unroll
        for (int q = 0; q < kK; ++q)
          sW[stage][row * kK + q] = __float2bfloat16(0.f);
      }
      // u_pack: (NT, H, BT, V) bf16; we only need BV columns.
      if (row < t_count) {
        const __nv_bfloat16* src = u
            + ((size_t)i_t_chunk * H + i_h) * kBT * kV + row * kV + v_base;
        #pragma unroll
        for (int q = 0; q < kBV / 8; ++q)
          cp_async_16(&sV[stage][row * kBV + q * 8], &src[q * 8]);
      } else {
        #pragma unroll
        for (int q = 0; q < kBV; ++q)
          sV[stage][row * kBV + q] = __float2bfloat16(0.f);
      }
      // k_l2 stays in RAW (S, Hg, K) layout because it's not produced by the
      // recompute_wu pipeline; it's the L2-normalized prefill k buffer.
      int t_start = i_t_chunk * kBT;
      if (row < t_count) {
        const __nv_bfloat16* src = k_l2
            + ((size_t)t_start + row) * Hg * kK + kh * kK;
        #pragma unroll
        for (int q = 0; q < kK / 8; ++q)
          cp_async_16(&sK[stage][row * kK + q * 8], &src[q * 8]);
      } else {
        #pragma unroll
        for (int q = 0; q < kK; ++q)
          sK[stage][row * kK + q] = __float2bfloat16(0.f);
      }
      // g_cumsum stays raw (S, H) bf16.
      if (row < t_count) {
        sG[stage][row] =
            g_bf16[(size_t)(t_start + row) * H + i_h];
      } else {
        sG[stage][row] = __float2bfloat16(0.f);
      }
    }
  };

  {
    int t_count = (kBT <= S) ? kBT : S;
    issue_chunk(0, t_count, 0);
    cp_async_commit();
  }
  if (NT > 1) {
    int t_count = (kBT + kBT <= S) ? kBT : (S - kBT);
    issue_chunk(1, t_count, 1);
    cp_async_commit();
  }

  for (int i_t = 0; i_t < NT; ++i_t) {
    const int t_start = i_t * kBT;
    const int t_count = (t_start + kBT <= S) ? kBT : (S - t_start);
    const int stage = i_t & 1;

    cp_async_wait<1>();
    __syncthreads();

    // Optional packed k_pack_hv side output (chunks, H, BT, K). Only needed
    // when the downstream output_o variant consumes the packed K with GQA
    // expansion. Coalesced int4 stores from sK[stage] (which already holds
    // the GQA-expanded k for this v-head).
    if (k_pack_hv != nullptr) {
      __nv_bfloat16* dst = k_pack_hv + ((size_t)i_t * H + i_h) * kBT * kK;
      // 64 * 128 = 8192 bf16 = 1024 int4. 128 threads -> 8 int4 / thread.
      #pragma unroll
      for (int pass = 0; pass < 8; ++pass) {
        int off = (pass * 128 + tid) * 8;
        *reinterpret_cast<int4*>(&dst[off]) =
            *reinterpret_cast<const int4*>(&sK[stage][off]);
      }
    }

    // Cast b_h1, b_h2 -> sH1, sH2 in one pass.
    {
      const int m_row_base = warp_id * 16;
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int n_col_base = nj * 8;
        int r = lane / 4;
        int c = (lane % 4) * 2;
        int r0 = m_row_base + r;
        int r1 = m_row_base + r + 8;
        sH1[r0 * 64 + n_col_base + c    ] = __float2bfloat16(b_h1[nj][0]);
        sH1[r0 * 64 + n_col_base + c + 1] = __float2bfloat16(b_h1[nj][1]);
        sH1[r1 * 64 + n_col_base + c    ] = __float2bfloat16(b_h1[nj][2]);
        sH1[r1 * 64 + n_col_base + c + 1] = __float2bfloat16(b_h1[nj][3]);
        sH2[r0 * 64 + n_col_base + c    ] = __float2bfloat16(b_h2[nj][0]);
        sH2[r0 * 64 + n_col_base + c + 1] = __float2bfloat16(b_h2[nj][1]);
        sH2[r1 * 64 + n_col_base + c    ] = __float2bfloat16(b_h2[nj][2]);
        sH2[r1 * 64 + n_col_base + c + 1] = __float2bfloat16(b_h2[nj][3]);
      }
    }
    __syncthreads();

    // Coalesced h_out write from sH1, sH2. 128 threads * 1 int4/pass * 4 passes.
    {
      __nv_bfloat16* dst_base = h_out + ((size_t)i_t * H + i_h) * kK * kV;
      #pragma unroll
      for (int pass = 0; pass < 4; ++pass) {
        int row = pass * 16 + (tid / 8);
        int col_block = tid % 8;
        int src = row * 64 + col_block * 8;
        int dst0 = row * kV + v_base + col_block * 8;
        *reinterpret_cast<int4*>(&dst_base[dst0]) =
            *reinterpret_cast<const int4*>(&sH1[src]);
        int dst1 = (row + 64) * kV + v_base + col_block * 8;
        *reinterpret_cast<int4*>(&dst_base[dst1]) =
            *reinterpret_cast<const int4*>(&sH2[src]);
      }
    }

    // v_acc = sW[stage][:, 0:64] @ sH1 + sW[stage][:, 64:128] @ sH2.
    float v_acc[8][4];
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj)
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        v_acc[nj][kk] = 0.f;

    const int m_row_base = warp_id * 16;
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int k_col_base = kk * 16;
      int row = lane % 16;
      int col_grp = (lane / 16) * 8;
      ldmatrix_x4_a(
          &sW[stage][(m_row_base + row) * kK + k_col_base + col_grp],
          af[0], af[1], af[2], af[3]);
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int k_row_base = kk * 16;
        int n_col_base = nj * 8;
        int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
        ldmatrix_x2_trans_b(
            &sH1[(k_row_base + b_row) * 64 + n_col_base],
            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(v_acc[nj][0], v_acc[nj][1],
                     v_acc[nj][2], v_acc[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int k_col_base = 64 + kk * 16;
      int row = lane % 16;
      int col_grp = (lane / 16) * 8;
      ldmatrix_x4_a(
          &sW[stage][(m_row_base + row) * kK + k_col_base + col_grp],
          af[0], af[1], af[2], af[3]);
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int k_row_base = kk * 16;
        int n_col_base = nj * 8;
        int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
        ldmatrix_x2_trans_b(
            &sH2[(k_row_base + b_row) * 64 + n_col_base],
            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(v_acc[nj][0], v_acc[nj][1],
                     v_acc[nj][2], v_acc[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }

    // v_new compute: raw -> global; decayed -> sV[stage] alias sVnew.
    __nv_bfloat16* sVnew = sV[stage];
    const float g_last = __bfloat162float(sG[stage][t_count - 1]);
    {
      const int mrb = warp_id * 16;
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int n_col_base = nj * 8;
        #pragma unroll
        for (int kp = 0; kp < 4; ++kp) {
          int row = mrb + (lane / 4) + (kp / 2) * 8;
          int col = n_col_base + (lane % 4) * 2 + (kp % 2);
          float vacc = v_acc[nj][kp];
          float vorig = (row < t_count)
              ? static_cast<float>(sV[stage][row * kBV + col]) : 0.f;
          float vraw = vorig - vacc;
          __nv_bfloat16 vraw_bf16 = __float2bfloat16(vraw);
          if (row < t_count) {
            if (v_new_out != nullptr) {
              v_new_out[((size_t)(t_start + row) * H + i_h) * kV
                        + v_base + col] = vraw_bf16;
            }
            if (v_new_packed != nullptr) {
              // (NT, H, BT, V): packed[i_t, i_h, row, v_base+col]
              v_new_packed[(((size_t)i_t * H + i_h) * kBT + row) * kV
                           + v_base + col] = vraw_bf16;
            }
          }
          float gr = __bfloat162float(sG[stage][row]);
          float scale = __expf(g_last - gr);
          float vdec = vraw * scale;
          if (row >= t_count) vdec = 0.f;
          sVnew[row * kBV + col] = __float2bfloat16(vdec);
        }
      }
    }
    __syncthreads();

    const float decay = __expf(g_last);
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj)
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        b_h1[nj][kk] *= decay;
        b_h2[nj][kk] *= decay;
      }

    // kv mma: b_h += k_T @ v_dec using ldmatrix.x4.trans on raw k.
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int kstate_m_row_base = warp_id * 16;        // K_state offset within block 0
      int k_col_base = kk * 16;                    // BT offset
      int bt_row = k_col_base + (lane % 8) + (lane / 16) * 8;
      int k_col  = kstate_m_row_base + (lane / 8 % 2) * 8;
      ldmatrix_x4_a_trans(&sK[stage][bt_row * kK + k_col],
                          af[0], af[1], af[2], af[3]);
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int k_row_base = kk * 16;
        int n_col_base = nj * 8;
        int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
        ldmatrix_x2_trans_b(&sVnew[(k_row_base + b_row) * kBV + n_col_base],
                            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(b_h1[nj][0], b_h1[nj][1],
                     b_h1[nj][2], b_h1[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4];
      int kstate_m_row_base = 64 + warp_id * 16;   // K_state offset within block 1
      int k_col_base = kk * 16;
      int bt_row = k_col_base + (lane % 8) + (lane / 16) * 8;
      int k_col  = kstate_m_row_base + (lane / 8 % 2) * 8;
      ldmatrix_x4_a_trans(&sK[stage][bt_row * kK + k_col],
                          af[0], af[1], af[2], af[3]);
      uint32_t bf[8][2];
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        int k_row_base = kk * 16;
        int n_col_base = nj * 8;
        int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
        ldmatrix_x2_trans_b(&sVnew[(k_row_base + b_row) * kBV + n_col_base],
                            bf[nj][0], bf[nj][1]);
      }
      #pragma unroll
      for (int nj = 0; nj < 8; ++nj)
        mma_m16n8k16(b_h2[nj][0], b_h2[nj][1],
                     b_h2[nj][2], b_h2[nj][3],
                     af[0], af[1], af[2], af[3],
                     bf[nj][0], bf[nj][1]);
    }
    __syncthreads();

    if (i_t + 2 < NT) {
      int t_start_n = (i_t + 2) * kBT;
      int t_count_n = (t_start_n + kBT <= S) ? kBT : (S - t_start_n);
      issue_chunk(i_t + 2, t_count_n, stage);
      cp_async_commit();
    } else {
      cp_async_commit();
    }
  }

  cp_async_wait<0>();

  // Final state -> write back to the same (H, K, V) bf16 buffer (in-place).
  {
    __nv_bfloat16* base = state + (size_t)i_h * kK * kV;
    const int m_row_base = warp_id * 16;
    #pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      int n_col_base = nj * 8;
      int r = lane / 4;
      int c = (lane % 4) * 2;
      int r0 = m_row_base + r;
      int r1 = m_row_base + r + 8;
      int v0 = v_base + n_col_base + c;
      base[r0       * kV + v0    ] = __float2bfloat16(b_h1[nj][0]);
      base[r0       * kV + v0 + 1] = __float2bfloat16(b_h1[nj][1]);
      base[r1       * kV + v0    ] = __float2bfloat16(b_h1[nj][2]);
      base[r1       * kV + v0 + 1] = __float2bfloat16(b_h1[nj][3]);
      base[(r0 + 64)* kV + v0    ] = __float2bfloat16(b_h2[nj][0]);
      base[(r0 + 64)* kV + v0 + 1] = __float2bfloat16(b_h2[nj][1]);
      base[(r1 + 64)* kV + v0    ] = __float2bfloat16(b_h2[nj][2]);
      base[(r1 + 64)* kV + v0 + 1] = __float2bfloat16(b_h2[nj][3]);
    }
  }
}

}  // namespace mma_fla

void gdn_wy_chunk_h_b64_bf16_mma_fla(
    const void* k_l2,
    const void* w,
    const void* u,
    const void* g_cumsum,
    void*       state,
    void*       h_out,
    void*       v_new,
    void*       v_new_packed,
    void*       k_pack_hv,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || qk_group <= 0) return;
  if (head_dim != mma_fla::kK) {
    throw std::runtime_error(
        std::string("gdn_wy_chunk_h_b64_bf16_mma_fla: head_dim must be ") +
        std::to_string(mma_fla::kK) + ", got " + std::to_string(head_dim));
  }
  if (num_v_heads % qk_group != 0 ||
      num_v_heads / qk_group != num_k_heads) {
    throw std::runtime_error(
        "gdn_wy_chunk_h_b64_bf16_mma_fla: invalid GQA shape");
  }
  const int NT = (S + mma_fla::kBT - 1) / mma_fla::kBT;

  static bool s_attr_set = false;
  if (!s_attr_set) {
    cudaError_t err = cudaFuncSetAttribute(
        mma_fla::chunk_h_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(99 * 1024));
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("gdn_wy_chunk_h_b64_bf16_mma_fla: cudaFuncSetAttribute "
                      "failed: ") + cudaGetErrorString(err));
    }
    s_attr_set = true;
  }
  dim3 grid(num_v_heads, mma_fla::kV / mma_fla::kBV, 1);
  dim3 block(mma_fla::kThreads, 1, 1);
  mma_fla::chunk_h_kernel<<<grid, block, mma_fla::kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(u),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(h_out),
      reinterpret_cast<__nv_bfloat16*>(v_new),
      reinterpret_cast<__nv_bfloat16*>(v_new_packed),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      S, num_v_heads, num_k_heads, qk_group, NT);
}

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
