// SPDX-License-Identifier: Apache-2.0
//
// FLA-style hand-tuned recompute_wu kernel. Fuses pack_recompute_rhs +
// 2 cublasLt matmuls (u_pack = Ai @ rhs_u, w_pack = Ai @ rhs_w) into a
// single CTA-resident CUDA kernel.
//
// Algorithm per CTA = (chunk i_t, v-head i_h):
//   Load Ai_pack[i_t, i_h, :, :]   (BT x BT)
//   Load v_raw[i_t*BT..+BT, i_h, :]   (BT x V)
//   Load k_l2[i_t*BT..+BT, kh, :]      (BT x K)   GQA: kh = i_h / qk_group
//   Load beta[i_t*BT..+BT, i_h]        (BT)
//   Load g_cumsum[i_t*BT..+BT, i_h]    (BT)
//   rhs_u[t, d] = v[t, d] * beta[t]
//   rhs_w[t, d] = k[t, d] * beta[t] * exp(g[t])
//   u_pack = Ai @ rhs_u    (mma)
//   w_pack = Ai @ rhs_w    (mma)
//   Store u_pack, w_pack to global
//
// Geometry: head_dim = K = V = 128, BT = 64, qk_group = 3.
// Grid (NT, num_v_heads). 4 warps per CTA; each warp owns 16 BT rows.

#include "gated_delta_wy_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace flash_rt {
namespace kernels {
namespace linear_attention {

namespace recompute_wu_mma_fla {

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
constexpr int kThreads = 128;
// sAi (8KB) + sV (16KB) + sKL (16KB) + sBetaG (1KB) = 41KB
constexpr size_t kSmemBytes = 41 * 1024;

__global__ void recompute_wu_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    const __nv_bfloat16* __restrict__ Ai_pack,
    __nv_bfloat16*       __restrict__ w_pack,
    __nv_bfloat16*       __restrict__ u_pack,
    int S, int H, int Hg, int qk_group)
{
  const int i_t = blockIdx.x;
  const int i_h = blockIdx.y;
  const int kh = i_h / qk_group;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const int t_start = i_t * kBT;
  const int t_count = (t_start + kBT <= S) ? kBT : (S - t_start);

  extern __shared__ __nv_bfloat16 smem_raw[];
  __nv_bfloat16* sAi   = smem_raw;                       // (BT, BT) = 8KB
  __nv_bfloat16* sV    = sAi + kBT * kBT;                // (BT, V)  = 16KB
  __nv_bfloat16* sKL   = sV  + kBT * kV;                 // (BT, K)  = 16KB
  __nv_bfloat16* sBeta = sKL + kBT * kK;                 // (BT)
  float*         sG    = reinterpret_cast<float*>(sBeta + kBT);  // (BT) fp32 = exp(g)

  // ---- Load all inputs via cp.async ----
  // sAi: (BT, BT) = 4096 bf16 = 512 int4. 128 threads -> 4 int4/thread.
  {
    const __nv_bfloat16* src = Ai_pack
        + ((size_t)i_t * H + i_h) * kBT * kBT;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      int off = (pass * 128 + tid) * 8;
      cp_async_16(&sAi[off], &src[off]);
    }
  }
  // sV: (BT, V). One thread per row, V/8 = 16 int4 per row. 64 threads.
  if (tid < kBT) {
    int row = tid;
    if (row < t_count) {
      const __nv_bfloat16* src = v + ((size_t)t_start + row) * H * kV + i_h * kV;
      #pragma unroll
      for (int q = 0; q < kV / 8; ++q)
        cp_async_16(&sV[row * kV + q * 8], &src[q * 8]);
    } else {
      #pragma unroll
      for (int q = 0; q < kV; ++q) sV[row * kV + q] = __float2bfloat16(0.f);
    }
  }
  // sKL: (BT, K). GQA: kh = i_h / qk_group.
  if (tid < kBT) {
    int row = tid;
    if (row < t_count) {
      const __nv_bfloat16* src = k_l2 + ((size_t)t_start + row) * Hg * kK + kh * kK;
      #pragma unroll
      for (int q = 0; q < kK / 8; ++q)
        cp_async_16(&sKL[row * kK + q * 8], &src[q * 8]);
    } else {
      #pragma unroll
      for (int q = 0; q < kK; ++q) sKL[row * kK + q] = __float2bfloat16(0.f);
    }
  }
  // beta and g: BT scalars each. Sync read; tiny.
  if (tid < kBT) {
    int row = tid;
    int t_global = t_start + row;
    if (t_global < S) {
      sBeta[row] = beta[(size_t)t_global * H + i_h];
      sG[row] = __expf(__bfloat162float(
          g_cumsum[(size_t)t_global * H + i_h]));
    } else {
      sBeta[row] = __float2bfloat16(0.f);
      sG[row] = 0.f;
    }
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // ---- Compute rhs in-place in sV and sKL ----
  // rhs_u[t, d] = v[t, d] * beta[t]
  // rhs_w[t, d] = k[t, d] * beta[t] * exp(g[t])
  // Each thread writes a stripe; 128 threads cover 64*128 = 8192 cells per buf
  // = 64 cells/thread per buf.
  for (int idx = tid; idx < kBT * kV; idx += kThreads) {
    int t = idx / kV;
    int d = idx % kV;
    float bv = __bfloat162float(sBeta[t]);
    float vv = __bfloat162float(sV[idx]);
    sV[idx] = __float2bfloat16(vv * bv);
  }
  for (int idx = tid; idx < kBT * kK; idx += kThreads) {
    int t = idx / kK;
    int d = idx % kK;
    float bv = __bfloat162float(sBeta[t]);
    float gv = sG[t];
    float kv = __bfloat162float(sKL[idx]);
    sKL[idx] = __float2bfloat16(kv * bv * gv);
  }
  __syncthreads();

  // ---- mma: out_pack = sAi @ rhs (both u and w variants) ----
  // A = sAi (BT=M, BT=K_red), B = sRu/sRw (BT=K_red, V=N), C = out (BT, V)
  // 4 warps split M (BT). Per warp: 16 M-rows x 128 N-cols.
  // Per warp per K-iter: 1 m16 x 16 n8 = 16 mma. K_red=BT=64 / 16 = 4 K-iters.
  // Accumulator: 16 N-tiles * 4 fp32 = 64 fp32/thread/output (2 outputs).
  const int m_row_base = warp_id * 16;

  // Compute u_pack = sAi @ sV  (sV now holds rhs_u)
  float c_u[16][4];
  #pragma unroll
  for (int nj = 0; nj < 16; ++nj)
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      c_u[nj][kp] = 0.f;
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t af[4];
    int k_col_base = kk * 16;
    int row = lane % 16;
    int col_grp = (lane / 16) * 8;
    ldmatrix_x4_a(&sAi[(m_row_base + row) * kBT + k_col_base + col_grp],
                  af[0], af[1], af[2], af[3]);
    uint32_t bf[16][2];
    #pragma unroll
    for (int nj = 0; nj < 16; ++nj) {
      int k_row_base = kk * 16;
      int n_col_base = nj * 8;
      int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
      ldmatrix_x2_trans_b(&sV[(k_row_base + b_row) * kV + n_col_base],
                          bf[nj][0], bf[nj][1]);
    }
    #pragma unroll
    for (int nj = 0; nj < 16; ++nj)
      mma_m16n8k16(c_u[nj][0], c_u[nj][1], c_u[nj][2], c_u[nj][3],
                   af[0], af[1], af[2], af[3],
                   bf[nj][0], bf[nj][1]);
  }

  // Compute w_pack = sAi @ sKL (sKL now holds rhs_w)
  float c_w[16][4];
  #pragma unroll
  for (int nj = 0; nj < 16; ++nj)
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      c_w[nj][kp] = 0.f;
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t af[4];
    int k_col_base = kk * 16;
    int row = lane % 16;
    int col_grp = (lane / 16) * 8;
    ldmatrix_x4_a(&sAi[(m_row_base + row) * kBT + k_col_base + col_grp],
                  af[0], af[1], af[2], af[3]);
    uint32_t bf[16][2];
    #pragma unroll
    for (int nj = 0; nj < 16; ++nj) {
      int k_row_base = kk * 16;
      int n_col_base = nj * 8;
      int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
      ldmatrix_x2_trans_b(&sKL[(k_row_base + b_row) * kK + n_col_base],
                          bf[nj][0], bf[nj][1]);
    }
    #pragma unroll
    for (int nj = 0; nj < 16; ++nj)
      mma_m16n8k16(c_w[nj][0], c_w[nj][1], c_w[nj][2], c_w[nj][3],
                   af[0], af[1], af[2], af[3],
                   bf[nj][0], bf[nj][1]);
  }

  // ---- Store u_pack, w_pack to global ----
  // out_pack[i_t, i_h, row, col] = c_u[nj][kp] at position (row, col)
  // row = m_row_base + lane/4 + (kp/2)*8; col = nj*8 + (lane%4)*2 + (kp%2)
  __nv_bfloat16* u_dst = u_pack + ((size_t)i_t * H + i_h) * kBT * kV;
  __nv_bfloat16* w_dst = w_pack + ((size_t)i_t * H + i_h) * kBT * kK;
  #pragma unroll
  for (int nj = 0; nj < 16; ++nj) {
    int n_col_base = nj * 8;
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      int row = m_row_base + (lane / 4) + (kp / 2) * 8;
      int col = n_col_base + (lane % 4) * 2 + (kp % 2);
      __nv_bfloat16 uv = __float2bfloat16(c_u[nj][kp]);
      __nv_bfloat16 wv = __float2bfloat16(c_w[nj][kp]);
      u_dst[row * kV + col] = uv;
      w_dst[row * kK + col] = wv;
    }
  }
}

}  // namespace recompute_wu_mma_fla

void gdn_wy_recompute_wu_b64_bf16_mma_fla(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || qk_group <= 0) return;
  if (head_dim != recompute_wu_mma_fla::kK) {
    throw std::runtime_error(
        std::string("gdn_wy_recompute_wu_b64_bf16_mma_fla: head_dim must be ") +
        std::to_string(recompute_wu_mma_fla::kK));
  }
  if (num_v_heads % qk_group != 0 ||
      num_v_heads / qk_group != num_k_heads) {
    throw std::runtime_error(
        "gdn_wy_recompute_wu_b64_bf16_mma_fla: invalid GQA shape");
  }
  const int NT = (S + recompute_wu_mma_fla::kBT - 1) / recompute_wu_mma_fla::kBT;

  static bool s_attr_set = false;
  if (!s_attr_set) {
    cudaError_t err = cudaFuncSetAttribute(
        recompute_wu_mma_fla::recompute_wu_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(recompute_wu_mma_fla::kSmemBytes));
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("gdn_wy_recompute_wu_b64_bf16_mma_fla: "
                      "cudaFuncSetAttribute failed: ") +
          cudaGetErrorString(err));
    }
    s_attr_set = true;
  }
  dim3 grid(NT, num_v_heads, 1);
  dim3 block(recompute_wu_mma_fla::kThreads, 1, 1);
  recompute_wu_mma_fla::recompute_wu_kernel<<<
      grid, block, recompute_wu_mma_fla::kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<const __nv_bfloat16*>(Ai_pack),
      reinterpret_cast<__nv_bfloat16*>(w_pack),
      reinterpret_cast<__nv_bfloat16*>(u_pack),
      S, num_v_heads, num_k_heads, qk_group);
}

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
