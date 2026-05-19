// ================================================================
//  motus_dev — Hand NVFP4 Conv3d fprop, sm_120a, **v19 + real SF**
//
//  Adds per-K-tile UE4M3 scale factor inputs to v19_skeleton.cu.
//  All other tile geometry / smem layout / cp.async pipeline /
//  Y-major persistent walk unchanged. v19_skeleton.cu is preserved
//  as-is (SF hardcoded to 1.0); this file is additive.
//
//  Phase 7.5.0 probe-verified SF maps (from sf_probe_test.cu):
//
//    SFA u32 byte k → K-block k (1:1)
//    lane L → M-row provider:
//      L%4==0: M = L/4         (rows 0..7)
//      L%4==1: M = L/4 + 8     (rows 8..15)
//      L%4∈{2,3}: ignored
//
//    SFB u32 byte k → K-block k (1:1)
//    lane L → N-col: N = (L%4)*8 + L/4
//    (atom_id = L%4, col_in_atom = L/4)
//
//  Input SF tensor shapes (matches conv input layout):
//    cache_sfa  [B, T_cache, H, W, Ci/16]  UE4M3 bytes
//    new_sfa    [B, T_new,   H, W, Ci/16]  UE4M3 bytes
//    w_sfb      [Co, 3, 3, 3, Ci/16]       UE4M3 bytes
//  (1 SF per 16 FP4 elements; for Ci=64, Ci/16=4 → 4 SF bytes per row,
//   fitting in one u32 — fits the K-tile load pattern below.)
//
//  Smem layout per K-tile (1 stage shown; 2 stages):
//    A_sf_smem [V19_BLOCK_M * 4]  = 512 bytes  (4 SF bytes per M-row)
//    B_sf_smem [V19_BLOCK_N * 4]  = 512 bytes  (4 SF bytes per N-col)
//
//  Load distribution (per K-tile, 256 threads × 4 bytes):
//    threads 0..127 → SFA[m=t][0..3]
//    threads 128..255 → SFB[n=t-128][0..3]
//  Each load is cp.async.4. Source pointer derived from m/co + (kt,kr,ks)
//  at the current K-iter using the same logic as FP4 data ptrs.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace conv {

// K128 variant: BLOCK_K_ELEM=128 (vs ship 64). Halves outer K-iter count,
// requires 2 cp.async.16 per thread per K-tile and 2x mma per N-group inner.
constexpr int V19SFBK128_BLOCK_M = 128;
constexpr int V19SFBK128_BLOCK_N = 128;
constexpr int V19SFBK128_BLOCK_K_ELEM = 128;
constexpr int V19SFBK128_N_ATOMS  = V19SFBK128_BLOCK_N / 8;   // 16
constexpr int V19SFBK128_N_GROUPS = V19SFBK128_N_ATOMS / 4;   // 4
constexpr int V19SFBK128_NUM_WARPS = 8;
constexpr int V19SFBK128_THREADS = V19SFBK128_NUM_WARPS * 32;
constexpr int V19SFBK128_STAGES = 2;
constexpr int V19SFBK128_SMEM_K_STRIDE = 80;             // 64B FP4 + 16B pad
constexpr int V19SFBK128_SF_K_PER_ROW = 8;               // 8 SF bytes per row (vs ship 4)

__device__ __forceinline__
void v19sfbk128_mma_m16n8k64_e2m1_4x(
    float &d0, float &d1, float &d2, float &d3,
    float &d4, float &d5, float &d6, float &d7,
    float &d8, float &d9, float &d10, float &d11,
    float &d12, float &d13, float &d14, float &d15,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    uint32_t sfa, uint32_t sfb)
{
  constexpr uint16_t bidA = 0, tidA = 0, bidB = 0;
  constexpr uint16_t tidB0 = 0, tidB1 = 1, tidB2 = 2, tidB3 = 3;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d0), "+f"(d1), "+f"(d8), "+f"(d9)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(d0), "f"(d1), "f"(d8), "f"(d9),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB0));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d2), "+f"(d3), "+f"(d10), "+f"(d11)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3),
      "f"(d2), "f"(d3), "f"(d10), "f"(d11),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB1));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d4), "+f"(d5), "+f"(d12), "+f"(d13)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5),
      "f"(d4), "f"(d5), "f"(d12), "f"(d13),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB2));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d6), "+f"(d7), "+f"(d14), "+f"(d15)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7),
      "f"(d6), "f"(d7), "f"(d14), "f"(d15),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB3));
}

// FP4 data ptrs (verbatim from v19_skeleton.cu, k_global in elements)
__device__ __forceinline__
const uint8_t* v19sfbk128_x_byte_ptr(const uint8_t* cache_x_fp4,
                                const uint8_t* new_x_fp4,
                                int m_global, int k_global,
                                int N, int T_cache, int T_new,
                                int H, int W, int Ci) {
  int K_total = 27 * Ci;
  int M_total = N * T_new * H * W;
  if (k_global >= K_total || m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem      -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx_elem = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
    return cache_x_fp4 + (idx_elem >> 1);
  } else {
    int d_new = d_in - T_cache;
    int idx_elem = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
    return new_x_fp4 + (idx_elem >> 1);
  }
}

__device__ __forceinline__
const uint8_t* v19sfbk128_w_byte_ptr(const uint8_t* w_fp4,
                                int co, int k_global, int Co, int Ci) {
  int K_total = 27 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int idx_elem = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
  return w_fp4 + (idx_elem >> 1);
}

// SFA / SFB ptrs: byte = idx_blk (one SF byte per 16 FP4 elem).
// Fixed 2026-05-12: take k_base directly; derive ci_block_off internally so
// Ci > BLOCK_K_ELEM (i.e. Ci > 64) works. See v19_sf_skeleton.cu for context.
__device__ __forceinline__
const uint8_t* v19sfbk128_sfa_byte_ptr(const uint8_t* cache_sfa,
                                  const uint8_t* new_sfa,
                                  int m_global, int k_base,
                                  int N, int T_cache, int T_new,
                                  int H, int W, int Ci) {
  int Ci_blk = Ci >> 4;
  int M_total = N * T_new * H * W;
  if (m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem      -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int kt_iter = k_base / Ci;
  int ci_block_off = (k_base % Ci) >> 4;
  int ks = kt_iter % 3;
  int kr = (kt_iter / 3) % 3;
  int kt = kt_iter / 9;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
    return cache_sfa + idx;
  } else {
    int d_new = d_in - T_cache;
    int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
    return new_sfa + idx;
  }
}

__device__ __forceinline__
const uint8_t* v19sfbk128_sfb_byte_ptr(const uint8_t* w_sfb,
                                  int co, int k_base, int Co, int Ci) {
  int Ci_blk = Ci >> 4;
  if (co >= Co) return nullptr;
  int kt_iter = k_base / Ci;
  int ci_block_off = (k_base % Ci) >> 4;
  int ks = kt_iter % 3;
  int kr = (kt_iter / 3) % 3;
  int kt = kt_iter / 9;
  int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci_blk + ci_block_off;
  return w_sfb + idx;
}

__device__ __forceinline__
void v19sfbk128_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
void v19sfbk128_cp_async_4(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 4;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
void v19sfbk128_cp_async_8(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 8;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
uint32_t v19sfbk128_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __launch_bounds__(V19SFBK128_THREADS, 2)
fp4_conv3d_v19sfbk128_kernel(
    const uint8_t* __restrict__ cache_x, const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sfa, const uint8_t* __restrict__ new_sfa,
    const uint8_t* __restrict__ w_sfb,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem [V19SFBK128_STAGES][V19SFBK128_BLOCK_M * V19SFBK128_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem [V19SFBK128_STAGES][V19SFBK128_BLOCK_N * V19SFBK128_SMEM_K_STRIDE];
  // SF smem: 4 bytes per row, 128 rows. Pad each row to 4 bytes (same).
  __shared__ __align__(16) uint8_t A_sf_smem[V19SFBK128_STAGES][V19SFBK128_BLOCK_M * V19SFBK128_SF_K_PER_ROW];
  __shared__ __align__(16) uint8_t B_sf_smem[V19SFBK128_STAGES][V19SFBK128_BLOCK_N * V19SFBK128_SF_K_PER_ROW];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total = N * T_new * H * W;
  const int K_total = 27 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 32;  // K elements
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 32;

  int tile_idx = blockIdx.x;
  int m_idx  = tile_idx / N_tiles;
  int n_idx  = tile_idx % N_tiles;
  int m_base = m_idx * V19SFBK128_BLOCK_M;
  int co_base = n_idx * V19SFBK128_BLOCK_N;
  if (m_base >= M_total || co_base >= Co) return;

  float dA[V19SFBK128_N_ATOMS] = {0};
  float dB[V19SFBK128_N_ATOMS] = {0};
  float dC[V19SFBK128_N_ATOMS] = {0};
  float dD[V19SFBK128_N_ATOMS] = {0};

  // ── Issue load: K=128 per K-tile = 64 bytes per row. Each thread issues 2
  //    cp.async.16 covering 64 K-elements (32B) each. Total 4 cp.async.16 per
  //    row from 4 threads (was 2 cp.async.16 per row from 2 threads in K=64).
  //    SF: 8 bytes per row (was 4) → cp.async.8 (was cp.async.4).
  auto issue_load = [&](int stage, int k_base) {
    // FP4 A — part 0: K elem [ld_k_off_a, ld_k_off_a + 32)
    // FP4 A — part 1: K elem [ld_k_off_a + 64, ld_k_off_a + 96)
    {
      const uint8_t* src0 = v19sfbk128_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a,
                                            N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int0 = v19sfbk128_to_smem_int(
          &A_smem[stage][ld_row_a * V19SFBK128_SMEM_K_STRIDE + (ld_k_off_a >> 1)]);
      v19sfbk128_cp_async_16(smem_int0, src0);
      const uint8_t* src1 = v19sfbk128_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a + 64,
                                            N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int1 = v19sfbk128_to_smem_int(
          &A_smem[stage][ld_row_a * V19SFBK128_SMEM_K_STRIDE
                         + ((ld_k_off_a + 64) >> 1)]);
      v19sfbk128_cp_async_16(smem_int1, src1);
    }
    // FP4 B — same pattern
    {
      const uint8_t* src0 = v19sfbk128_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b,
                                            Co, Ci);
      uint32_t smem_int0 = v19sfbk128_to_smem_int(
          &B_smem[stage][ld_row_b * V19SFBK128_SMEM_K_STRIDE + (ld_k_off_b >> 1)]);
      v19sfbk128_cp_async_16(smem_int0, src0);
      const uint8_t* src1 = v19sfbk128_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b + 64,
                                            Co, Ci);
      uint32_t smem_int1 = v19sfbk128_to_smem_int(
          &B_smem[stage][ld_row_b * V19SFBK128_SMEM_K_STRIDE
                         + ((ld_k_off_b + 64) >> 1)]);
      v19sfbk128_cp_async_16(smem_int1, src1);
    }
    // SF — 8 bytes per row (was 4) for BLOCK_K=128 = 8 K-blocks × 1 byte.
    if (t < V19SFBK128_BLOCK_M) {
      const uint8_t* src = v19sfbk128_sfa_byte_ptr(cache_sfa, new_sfa,
                                              m_base + t, k_base,
                                              N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = v19sfbk128_to_smem_int(
          &A_sf_smem[stage][t * V19SFBK128_SF_K_PER_ROW]);
      v19sfbk128_cp_async_8(smem_int, src);
    } else {
      int n_idx_t = t - V19SFBK128_BLOCK_M;
      const uint8_t* src = v19sfbk128_sfb_byte_ptr(w_sfb, co_base + n_idx_t,
                                              k_base, Co, Ci);
      uint32_t smem_int = v19sfbk128_to_smem_int(
          &B_sf_smem[stage][n_idx_t * V19SFBK128_SF_K_PER_ROW]);
      v19sfbk128_cp_async_8(smem_int, src);
    }
  };

  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;

  for (int k_base = 0; k_base < K_total; k_base += V19SFBK128_BLOCK_K_ELEM) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + V19SFBK128_BLOCK_K_ELEM;
    if (k_next < K_total) issue_load(next_stage, k_next);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    // Sub-K0: K elements [0..63], smem byte offsets [0..31]
    const int kA0_s0 = 4 * l;
    const int kA2_s0 = 4 * l + 16;
    // Sub-K1: K elements [64..127], smem byte offsets [32..63]
    const int kA0_s1 = 4 * l + 32;
    const int kA2_s1 = 4 * l + 48;

    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0_s0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
    uint32_t A1_s0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
    uint32_t A2_s0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);
    uint32_t A3_s0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);
    uint32_t A0_s1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
    uint32_t A1_s1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
    uint32_t A2_s1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);
    uint32_t A3_s1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);

    // SFA u32 per lane: 4 SF bytes per sub-K. Sub-K0 uses bytes [0..3],
    // sub-K1 uses bytes [4..7] of the row's 8-byte SF.
    int sfa_m_row;
    if ((lane & 3) == 1) {
      sfa_m_row = warp_M_off + (lane >> 2) + 8;
    } else {
      sfa_m_row = warp_M_off + (lane >> 2);
    }
    uint32_t SFA_s0 = *reinterpret_cast<const uint32_t*>(
        &A_sf_smem[compute_stage][sfa_m_row * V19SFBK128_SF_K_PER_ROW + 0]);
    uint32_t SFA_s1 = *reinterpret_cast<const uint32_t*>(
        &A_sf_smem[compute_stage][sfa_m_row * V19SFBK128_SF_K_PER_ROW + 4]);

    // Per N-group: 4 scale_vec::4X calls × 2 sub-K = 8 mma macro calls per warp.
    #pragma unroll
    for (int g = 0; g < V19SFBK128_N_GROUPS; ++g) {
      int base = g * 4;
      int co0 = (base + 0) * 8 + h;
      int co1 = (base + 1) * 8 + h;
      int co2 = (base + 2) * 8 + h;
      int co3 = (base + 3) * 8 + h;

      // B sub-K0 loads
      uint32_t B0_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
      uint32_t B1_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);
      uint32_t B2_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
      uint32_t B3_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);
      uint32_t B4_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
      uint32_t B5_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);
      uint32_t B6_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SFBK128_SMEM_K_STRIDE + kA0_s0]);
      uint32_t B7_s0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SFBK128_SMEM_K_STRIDE + kA2_s0]);

      // SFB sub-K0: bytes [0..3]
      int sfb_n = g * 32 + l * 8 + h;
      uint32_t SFB_s0 = *reinterpret_cast<const uint32_t*>(
          &B_sf_smem[compute_stage][sfb_n * V19SFBK128_SF_K_PER_ROW + 0]);

      v19sfbk128_mma_m16n8k64_e2m1_4x(
          dA[base+0], dB[base+0],
          dA[base+1], dB[base+1],
          dA[base+2], dB[base+2],
          dA[base+3], dB[base+3],
          dC[base+0], dD[base+0],
          dC[base+1], dD[base+1],
          dC[base+2], dD[base+2],
          dC[base+3], dD[base+3],
          A0_s0, A1_s0, A2_s0, A3_s0,
          B0_s0, B1_s0, B2_s0, B3_s0, B4_s0, B5_s0, B6_s0, B7_s0,
          SFA_s0, SFB_s0);

      // B sub-K1 loads
      uint32_t B0_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
      uint32_t B1_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);
      uint32_t B2_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
      uint32_t B3_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);
      uint32_t B4_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
      uint32_t B5_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);
      uint32_t B6_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SFBK128_SMEM_K_STRIDE + kA0_s1]);
      uint32_t B7_s1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SFBK128_SMEM_K_STRIDE + kA2_s1]);

      // SFB sub-K1: bytes [4..7]
      uint32_t SFB_s1 = *reinterpret_cast<const uint32_t*>(
          &B_sf_smem[compute_stage][sfb_n * V19SFBK128_SF_K_PER_ROW + 4]);

      v19sfbk128_mma_m16n8k64_e2m1_4x(
          dA[base+0], dB[base+0],
          dA[base+1], dB[base+1],
          dA[base+2], dB[base+2],
          dA[base+3], dB[base+3],
          dC[base+0], dD[base+0],
          dC[base+1], dD[base+1],
          dC[base+2], dD[base+2],
          dC[base+3], dD[base+3],
          A0_s1, A1_s1, A2_s1, A3_s1,
          B0_s1, B1_s1, B2_s1, B3_s1, B4_s1, B5_s1, B6_s1, B7_s1,
          SFA_s1, SFB_s1);
    }
    compute_stage = next_stage;
  }
  asm volatile("cp.async.wait_all;\n" ::);

  // ── Epilogue: NCDHW out + optional residual add (mirrors v18) ──
  // Output layout y[B, Co, T_new, H, W]. Less coalesced than v19sf's
  // NDHWC bf162 store, but removes the separate NDHWC→NCDHW transpose
  // + residual-add launches downstream.
  const int warp_M_off = warp_id * 16;
  auto ncdhw_idx = [&](int row, int co) -> long long {
    int spatial = T_new * H * W;
    int b_idx = row / spatial;
    int rem = row - b_idx * spatial;
    int t_out = rem / (H * W);
    rem -= t_out * (H * W);
    int h_out = rem / W;
    int w_out = rem - h_out * W;
    return (((long long)b_idx * Co + co) * T_new + t_out)
           * (long long)H * W + (long long)h_out * W + w_out;
  };
  auto add_res = [&](long long idx, float v) -> __nv_bfloat16 {
    __nv_bfloat16 conv_bf16 = __float2bfloat16(v);
    if (residual != nullptr) {
      float summed = __bfloat162float(conv_bf16)
                   + __bfloat162float(residual[idx]);
      return __float2bfloat16(summed);
    }
    return conv_bf16;
  };
  #pragma unroll
  for (int n_atom = 0; n_atom < V19SFBK128_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0    = m_base + warp_M_off + h;
    int row1    = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr && co_pair < Co) {
      b0 = __bfloat162float(bias[co_pair]);
      if (co_pair + 1 < Co) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co) {
      if (row0 < M_total) {
        long long idx0 = ncdhw_idx(row0, co_pair);
        long long idx1 = ncdhw_idx(row0, co_pair + 1);
        y[idx0] = add_res(idx0, dA[n_atom] * alpha + b0);
        y[idx1] = add_res(idx1, dB[n_atom] * alpha + b1);
      }
      if (row1 < M_total) {
        long long idx0 = ncdhw_idx(row1, co_pair);
        long long idx1 = ncdhw_idx(row1, co_pair + 1);
        y[idx0] = add_res(idx0, dC[n_atom] * alpha + b0);
        y[idx1] = add_res(idx1, dD[n_atom] * alpha + b1);
      }
    } else {
      auto store = [&](int row, int co, float v, float bv) {
        if (row < M_total && co < Co) {
          long long idx = ncdhw_idx(row, co);
          y[idx] = add_res(idx, v * alpha + bv);
        }
      };
      store(row0, co_pair + 0, dA[n_atom], b0);
      store(row0, co_pair + 1, dB[n_atom], b1);
      store(row1, co_pair + 0, dC[n_atom], b0);
      store(row1, co_pair + 1, dD[n_atom], b1);
    }
  }
}

extern "C" int motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
    const void* cache_x_fp4,
    const void* new_x_fp4,
    const void* w_fp4,
    const void* cache_sfa, const void* new_sfa, const void* w_sfb,
    void* y_bf16,
    const void* bias_bf16,
    const void* residual_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % V19SFBK128_BLOCK_K_ELEM != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp4_conv3d_v19sfb] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        V19SFBK128_BLOCK_K_ELEM, Ci, Co);
    return -1;
  }
  if (T_cache != 2) {
    std::fprintf(stderr, "[fp4_conv3d_v19sfb] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + V19SFBK128_BLOCK_M - 1) / V19SFBK128_BLOCK_M;
  int N_tiles = (Co + V19SFBK128_BLOCK_N - 1) / V19SFBK128_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(V19SFBK128_THREADS);
  fp4_conv3d_v19sfbk128_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(cache_x_fp4),
      reinterpret_cast<const uint8_t*>(new_x_fp4),
      reinterpret_cast<const uint8_t*>(w_fp4),
      reinterpret_cast<const uint8_t*>(cache_sfa),
      reinterpret_cast<const uint8_t*>(new_sfa),
      reinterpret_cast<const uint8_t*>(w_sfb),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<const __nv_bfloat16*>(residual_bf16),
      N, T_cache, T_new, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp4_conv3d_v19sfb] launch err: %s\n",
                 cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace conv
}  // namespace flash_rt
