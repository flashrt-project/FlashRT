// ================================================================
// FlashRT — Hy-VLA prefill segment-mask attention (SM120, bf16).
//
// Block-sparse tensor-core (SM80 WMMA m16n16k16 bf16) flash attention
// for the Hy-VLA prefill. The segment mask is block-diagonal vision
// (3 x 49) + causal text, so a large fraction of the full SxS matrix is
// masked. A precomputed per-query-block active-KV-block bitmask skips
// the fully-masked 16-key tiles, so the kernel materializes only the
// tiles that carry any unmasked pair — vs the mem-efficient SDPA
// (fmha_cutlassF sm80) which computes every tile of the full 181x181.
//
// Layouts (bf16, contiguous; H query heads, S tokens, D=128):
//   q/k/v/o : (H, S, D)
//   mask    : (S, S) fp32 additive (0.0 = attend, -inf = masked)
//   active  : (num_q_blocks, H) uint32 bitmask over num_k_blocks
//
// One CTA per (query block of 64, head): 4 warps, each warp owns 16
// query rows and does m16n16k16 WMMA. K/V are staged in shared memory.
// ================================================================

#include "common.cuh"
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

namespace flash_rt {
namespace hyvla {

constexpr int kHeadDim = 128;
constexpr int kQTile = 64;
constexpr int kKTile = 16;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kM = 16, kN = 16, kK = 16;
constexpr int kFrag = kHeadDim / kK;  // 8

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  unsigned a = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(a), "l"(src));
}

__global__ void __launch_bounds__(kThreads, 1)
hyvla_prefill_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const float* __restrict__ mask,
    const uint32_t* __restrict__ active,
    int S, int H, float scale,
    int q_stride_h, int k_stride_h, int v_stride_h)
{
  __shared__ __nv_bfloat16 Qs[kQTile * kHeadDim];
  __shared__ __nv_bfloat16 Ks[2][kKTile * kHeadDim];
  __shared__ __nv_bfloat16 Vs[2][kKTile * kHeadDim];
  __shared__ __nv_bfloat16 Ps[kWarps][kM][kN];

  const int qb = blockIdx.x, head = blockIdx.y;
  const int t = threadIdx.x, warp = t >> 5;
  const int fr = (t & 31) >> 2, fc = (t & 31) & 3;
  const int num_k_blocks = (S + kKTile - 1) / kKTile;
  const int q_base = qb * kQTile;
  const uint32_t active_mask = active[qb * H + head];

  for (int j = t; j < kQTile * kHeadDim / 2; j += kThreads) {
    int r = j / (kHeadDim / 2), c = j % (kHeadDim / 2);
    if (q_base + r < S) {
      Qs[r * kHeadDim + 2 * c] = Q[(size_t)head * q_stride_h + (q_base + r) * kHeadDim + 2 * c];
      Qs[r * kHeadDim + 2 * c + 1] = Q[(size_t)head * q_stride_h + (q_base + r) * kHeadDim + 2 * c + 1];
    } else {
      Qs[r * kHeadDim + 2 * c] = Qs[r * kHeadDim + 2 * c + 1] = __float2bfloat16(0.f);
    }
  }
  __syncthreads();

  const int qrow0 = warp * kM;

  float row_max[kM];
  float row_sum[kM];
  for (int i = 0; i < kM; ++i) { row_max[i] = -1e30f; row_sum[i] = 0.0f; }

  wmma::fragment<wmma::matrix_a, kM, kN, kK, __nv_bfloat16, wmma::row_major> q_frag[kFrag];
  for (int ft = 0; ft < kFrag; ++ft)
    wmma::load_matrix_sync(q_frag[ft], &Qs[qrow0 * kHeadDim + ft * kK], kHeadDim);

  wmma::fragment<wmma::accumulator, kM, kN, kK, float> out_frag[kFrag];
  for (int ft = 0; ft < kFrag; ++ft) wmma::fill_fragment(out_frag[ft], 0.0f);

  auto load_kv_async = [&](int kb, int buf) {
    const int k_col = kb * kKTile;
    for (int j = t; j < kKTile * kHeadDim / 8; j += kThreads) {
      int r = j / (kHeadDim / 8), c = j % (kHeadDim / 8);
      if (k_col + r < S) {
        cp_async16(&Ks[buf][r * kHeadDim + c * 8],
                   &K[(size_t)head * k_stride_h + (k_col + r) * kHeadDim + c * 8]);
        cp_async16(&Vs[buf][r * kHeadDim + c * 8],
                   &V[(size_t)head * v_stride_h + (k_col + r) * kHeadDim + c * 8]);
      }
    }
  };

  load_kv_async(0, 0);
  asm volatile("cp.async.commit_group;\n");

  for (int kb = 0; kb < num_k_blocks; ++kb) {
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    const int k_col = kb * kKTile;
    const int buf = kb & 1;
    if (kb + 1 < num_k_blocks) {
      load_kv_async(kb + 1, (kb + 1) & 1);
      asm volatile("cp.async.commit_group;\n");
    }
    if (!(active_mask & (1u << kb))) { __syncthreads(); continue; }

    wmma::fragment<wmma::accumulator, kM, kN, kK, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);
    for (int ft = 0; ft < kFrag; ++ft) {
      wmma::fragment<wmma::matrix_b, kM, kN, kK, __nv_bfloat16, wmma::col_major> k_frag;
      wmma::load_matrix_sync(k_frag, &Ks[buf][ft * kK], kHeadDim);
      wmma::mma_sync(s_frag, q_frag[ft], k_frag, s_frag);
    }

    // Apply mask + scale in-register; each thread holds 8 elements mapping to
    // row r = fr + ((e&2)?8:0) and col = 2*fc + (e&1) + ((e&4)?8:0).
    float c[8];
    for (int e = 0; e < 8; ++e) {
      const int er = fr + ((e & 2) ? 8 : 0);
      const int ec = 2 * fc + (e & 1) + ((e & 4) ? 8 : 0);
      const int gi = q_base + qrow0 + er, gj = k_col + ec;
      float s = s_frag.x[e];
      if (gi < S && gj < S) s = (s + mask[(size_t)gi * S + gj]) * scale;
      else s = -1e30f;
      c[e] = s;
    }
    // Row max (row r: e 0,1,4,5; row r+8: e 2,3,6,7), shfl-reduced over the quad.
    float m0 = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[4], c[5]));
    float m1 = fmaxf(fmaxf(c[2], c[3]), fmaxf(c[6], c[7]));
    m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 2));
    const float mn0 = fmaxf(row_max[fr], m0), mn1 = fmaxf(row_max[fr + 8], m1);
    const float corr0 = __expf(row_max[fr] - mn0), corr1 = __expf(row_max[fr + 8] - mn1);
    row_sum[fr] *= corr0; row_sum[fr + 8] *= corr1;
    row_max[fr] = mn0; row_max[fr + 8] = mn1;
    for (int ft = 0; ft < kFrag; ++ft) {
      out_frag[ft].x[0] *= corr0; out_frag[ft].x[1] *= corr0;
      out_frag[ft].x[4] *= corr0; out_frag[ft].x[5] *= corr0;
      out_frag[ft].x[2] *= corr1; out_frag[ft].x[3] *= corr1;
      out_frag[ft].x[6] *= corr1; out_frag[ft].x[7] *= corr1;
    }
    float lt0 = 0.f, lt1 = 0.f;
    for (int e = 0; e < 8; ++e) {
      const int er = fr + ((e & 2) ? 8 : 0);
      const int ec = 2 * fc + (e & 1) + ((e & 4) ? 8 : 0);
      const float p = __expf(c[e] - ((e & 2) ? mn1 : mn0));
      Ps[warp][er][ec] = __float2bfloat16(p);
      if (e & 2) lt1 += p; else lt0 += p;
    }
    lt0 += __shfl_xor_sync(~0u, lt0, 1); lt0 += __shfl_xor_sync(~0u, lt0, 2);
    lt1 += __shfl_xor_sync(~0u, lt1, 1); lt1 += __shfl_xor_sync(~0u, lt1, 2);
    row_sum[fr] += lt0; row_sum[fr + 8] += lt1;
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kM, kN, kK, __nv_bfloat16, wmma::row_major> p_frag;
    wmma::load_matrix_sync(p_frag, &Ps[warp][0][0], kN);
    for (int ft = 0; ft < kFrag; ++ft) {
      wmma::fragment<wmma::matrix_b, kM, kN, kK, __nv_bfloat16, wmma::row_major> v_frag;
      wmma::load_matrix_sync(v_frag, &Vs[buf][ft * kK], kHeadDim);
      wmma::mma_sync(out_frag[ft], p_frag, v_frag, out_frag[ft]);
    }
    __syncthreads();
  }

  for (int ft = 0; ft < kFrag; ++ft) {
    for (int e = 0; e < 8; ++e) {
      const int er = fr + ((e & 2) ? 8 : 0);
      const int ec = 2 * fc + (e & 1) + ((e & 4) ? 8 : 0);
      const int gi = q_base + qrow0 + er;
      if (gi >= S) continue;
      const float inv = (row_sum[er] > 0.0f) ? (1.0f / row_sum[er]) : 0.0f;
      O[((size_t)gi * H + head) * kHeadDim + ft * kK + ec] =
          __float2bfloat16(out_frag[ft].x[e] * inv);
    }
  }
}

}  // namespace hyvla
}  // namespace flash_rt

extern "C" void hyvla_prefill_attn_bf16(
    const void* q, const void* k, const void* v, void* o,
    const void* mask, const void* active,
    int S, int H, float scale,
    int q_stride_h, int k_stride_h, int v_stride_h, cudaStream_t stream)
{
  const int num_q_blocks = (S + flash_rt::hyvla::kQTile - 1) / flash_rt::hyvla::kQTile;
  dim3 grid(num_q_blocks, H);
  flash_rt::hyvla::hyvla_prefill_attn_bf16_kernel<<<grid, flash_rt::hyvla::kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      reinterpret_cast<const float*>(mask),
      reinterpret_cast<const uint32_t*>(active),
      S, H, scale, q_stride_h, k_stride_h, v_stride_h);
}
