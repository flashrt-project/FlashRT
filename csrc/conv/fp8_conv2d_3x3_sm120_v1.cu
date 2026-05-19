// ================================================================
//  flash_rt — Hand FP8 Conv2d 3×3 fprop, sm_120a, v1
//
//  Strip-down of fp8_conv3d_sm120_v17.cu to the 2D case:
//    - no time axis, no causal cache concat
//    - kernel 3×3, padding=1, stride=1 (matches motus VAE resample.1
//      Conv2d sites)
//    - input  : x_fp8  [N, H, W, Ci]  NHWC fp8_e4m3
//    - weight : w_fp8  [Co, 3, 3, Ci] fp8_e4m3  (row-major, kr,ks,ci)
//    - bias   : bias_bf16 [Co]       (or nullptr)
//    - output : y_bf16  [N, H, W, Co] NHWC bf16
//
//  Same tile geometry as v17: BLOCK_M=128 BLOCK_N=128 BLOCK_K=32,
//  8 warps, cp.async 2-stage, persistent Y-major grid, bias-fused
//  bf16x2 epilogue. Same mma.sync.aligned.kind::f8f6f4.m16n8k32.
//
//  Constraints (matching v17):
//    Ci % 32 == 0 (BLOCK_K aligned)
//    Co % 8  == 0 (mma N alignment)
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace conv {

constexpr int C2D_BLOCK_M = 128;
constexpr int C2D_BLOCK_N = 128;
constexpr int C2D_BLOCK_K = 32;
constexpr int C2D_N_ATOMS = C2D_BLOCK_N / 8;
constexpr int C2D_NUM_WARPS = 8;
constexpr int C2D_THREADS = C2D_NUM_WARPS * 32;
constexpr int C2D_STAGES = 2;
constexpr int C2D_SMEM_K_STRIDE = 48;

__device__ __forceinline__
void c2d_mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1));
}

// 2D conv input addressing: m_global decodes (b, h_out, w_out).
// k_global decodes (kr, ks, ci0). Read input at (b, h_in, w_in, ci0)
// where h_in = h_out + kr - 1, w_in = w_out + ks - 1 (pad=1).
__device__ __forceinline__
const uint8_t* c2d_x_byte_ptr(const __nv_fp8_e4m3* x,
                              int m_global, int k_global,
                              int N, int H, int W, int Ci) {
  int K_total = 9 * Ci;
  int M_total = N * H * W;
  if (k_global >= K_total || m_global >= M_total) return nullptr;
  int spatial = H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3;
  int kr  = q / 3;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  int idx = ((b_idx * H + h_in) * W + w_in) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&x[idx]);
}

__device__ __forceinline__
const uint8_t* c2d_w_byte_ptr(const __nv_fp8_e4m3* w,
                              int co, int k_global, int Co, int Ci) {
  int K_total = 9 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3;
  int kr  = q / 3;
  // Weight layout (Co, 3, 3, Ci) row-major.
  int idx = ((co * 3 + kr) * 3 + ks) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&w[idx]);
}

__device__ __forceinline__
void c2d_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__
uint32_t c2d_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __launch_bounds__(C2D_THREADS, 2)
fp8_conv2d_3x3_v1_kernel(
    const __nv_fp8_e4m3* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
          __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    int N, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem[C2D_STAGES][C2D_BLOCK_M * C2D_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem[C2D_STAGES][C2D_BLOCK_N * C2D_SMEM_K_STRIDE];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total = N * H * W;
  const int K_total = 9 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 16;
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 16;

  int tile_idx = blockIdx.x;
  int m_idx  = tile_idx / N_tiles;
  int n_idx  = tile_idx % N_tiles;
  int m_base  = m_idx * C2D_BLOCK_M;
  int co_base = n_idx * C2D_BLOCK_N;

  if (m_base >= M_total || co_base >= Co) return;

  float dA[C2D_N_ATOMS] = {0};
  float dB[C2D_N_ATOMS] = {0};
  float dC[C2D_N_ATOMS] = {0};
  float dD[C2D_N_ATOMS] = {0};

  auto issue_load = [&](int stage, int k_base) {
    {
      const uint8_t* src = c2d_x_byte_ptr(x,
                                          m_base + ld_row_a,
                                          k_base + ld_k_off_a,
                                          N, H, W, Ci);
      uint32_t smem_int = c2d_to_smem_int(
          &A_smem[stage][ld_row_a * C2D_SMEM_K_STRIDE + ld_k_off_a]);
      c2d_cp_async_16(smem_int, src);
    }
    {
      const uint8_t* src = c2d_w_byte_ptr(w, co_base + ld_row_b,
                                          k_base + ld_k_off_b,
                                          Co, Ci);
      uint32_t smem_int = c2d_to_smem_int(
          &B_smem[stage][ld_row_b * C2D_SMEM_K_STRIDE + ld_k_off_b]);
      c2d_cp_async_16(smem_int, src);
    }
  };

  // Prologue
  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;

  for (int k_base = 0; k_base < K_total; k_base += C2D_BLOCK_K) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + C2D_BLOCK_K;

    if (k_next < K_total) {
      issue_load(next_stage, k_next);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    const int kA0 = 4 * l;
    const int kA2 = 4 * l + 16;

    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * C2D_SMEM_K_STRIDE + kA0]);
    uint32_t A1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * C2D_SMEM_K_STRIDE + kA0]);
    uint32_t A2 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * C2D_SMEM_K_STRIDE + kA2]);
    uint32_t A3 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * C2D_SMEM_K_STRIDE + kA2]);

    #pragma unroll
    for (int n_atom = 0; n_atom < C2D_N_ATOMS; ++n_atom) {
      int co_n = n_atom * 8 + h;
      uint32_t B0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * C2D_SMEM_K_STRIDE + kA0]);
      uint32_t B1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * C2D_SMEM_K_STRIDE + kA2]);
      c2d_mma_m16n8k32_e4m3(
          dA[n_atom], dB[n_atom], dC[n_atom], dD[n_atom],
          A0, A1, A2, A3, B0, B1);
    }

    compute_stage = next_stage;
  }

  asm volatile("cp.async.wait_all;\n" ::);

  const int warp_M_off = warp_id * 16;
  #pragma unroll
  for (int n_atom = 0; n_atom < C2D_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0    = m_base + warp_M_off + h;
    int row1    = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr && co_pair < Co) {
      b0 = __bfloat162float(bias[co_pair]);
      if (co_pair + 1 < Co) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co) {
      __nv_bfloat162 packAB;
      packAB.x = __float2bfloat16(dA[n_atom] * alpha + b0);
      packAB.y = __float2bfloat16(dB[n_atom] * alpha + b1);
      __nv_bfloat162 packCD;
      packCD.x = __float2bfloat16(dC[n_atom] * alpha + b0);
      packCD.y = __float2bfloat16(dD[n_atom] * alpha + b1);
      if (row0 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row0 * Co + co_pair]) = packAB;
      }
      if (row1 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row1 * Co + co_pair]) = packCD;
      }
    } else {
      auto store = [&](int row, int co, float v, float bv) {
        if (row < M_total && co < Co) {
          y[row * Co + co] = __float2bfloat16(v * alpha + bv);
        }
      };
      store(row0, co_pair + 0, dA[n_atom], b0);
      store(row0, co_pair + 1, dB[n_atom], b1);
      store(row1, co_pair + 0, dC[n_atom], b0);
      store(row1, co_pair + 1, dD[n_atom], b1);
    }
  }
}

// Inputs:
//   x_fp8     : [N, H, W, Ci] fp8_e4m3 (NHWC)
//   w_fp8     : [Co, 3, 3, Ci] fp8_e4m3 (Co × kR × kS × Ci row-major)
//   bias_bf16 : [Co] bf16 (or nullptr)
// Output:
//   y_bf16    : [N, H, W, Co] bf16 (NHWC)
extern "C" int fp8_conv2d_3x3_v1_nhwc_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % C2D_BLOCK_K != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp8_conv2d_3x3_v1] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        C2D_BLOCK_K, Ci, Co);
    return -1;
  }
  if (N <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) {
    return -2;
  }
  int M = N * H * W;
  int M_tiles = (M + C2D_BLOCK_M - 1) / C2D_BLOCK_M;
  int N_tiles = (Co + C2D_BLOCK_N - 1) / C2D_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(C2D_THREADS);
  fp8_conv2d_3x3_v1_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp8_conv2d_3x3_v1] launch err: %s\n",
                 cudaGetErrorString(e));
    return -3;
  }
  return 0;
}

// v2 keeps the same tile/MMA shape as v1 but removes most of the per-load
// im2col integer decoding from the hot K loop.  The VAE resample shapes have
// Ci as a large multiple of BLOCK_K, so each K tile belongs to exactly one
// (kr, ks) filter position; decode that once and combine it with per-thread
// output row coordinates kept in registers.
__global__ void __launch_bounds__(C2D_THREADS, 2)
fp8_conv2d_3x3_v2_kernel(
    const __nv_fp8_e4m3* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
          __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    int N, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles,
    int out_B, int out_T, int out_ncdhw)
{
  __shared__ __align__(16) uint8_t A_smem[C2D_STAGES][C2D_BLOCK_M * C2D_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem[C2D_STAGES][C2D_BLOCK_N * C2D_SMEM_K_STRIDE];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total = N * H * W;
  const int K_total = 9 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 16;
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 16;

  int tile_idx = blockIdx.x;
  int m_idx  = tile_idx / N_tiles;
  int n_idx  = tile_idx % N_tiles;
  int m_base  = m_idx * C2D_BLOCK_M;
  int co_base = n_idx * C2D_BLOCK_N;

  if (m_base >= M_total || co_base >= Co) return;

  const int m_load = m_base + ld_row_a;
  const bool m_load_valid = m_load < M_total;
  int b_load = 0, h_load = 0, w_load = 0;
  if (m_load_valid) {
    const int spatial = H * W;
    b_load = m_load / spatial;
    const int rem = m_load - b_load * spatial;
    h_load = rem / W;
    w_load = rem - h_load * W;
  }
  const int co_load = co_base + ld_row_b;

  float dA[C2D_N_ATOMS] = {0};
  float dB[C2D_N_ATOMS] = {0};
  float dC[C2D_N_ATOMS] = {0};
  float dD[C2D_N_ATOMS] = {0};

  auto issue_load = [&](int stage, int k_base) {
    {
      const int k_elem = k_base + ld_k_off_a;
      const int q = k_elem / Ci;
      const int ci0 = k_elem - q * Ci;
      const int ks = q % 3;
      const int kr = q / 3;
      const int h_in = h_load + kr - 1;
      const int w_in = w_load + ks - 1;
      const uint8_t* src = nullptr;
      if (m_load_valid && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        const int idx = ((b_load * H + h_in) * W + w_in) * Ci + ci0;
        src = reinterpret_cast<const uint8_t*>(&x[idx]);
      }
      uint32_t smem_int = c2d_to_smem_int(
          &A_smem[stage][ld_row_a * C2D_SMEM_K_STRIDE + ld_k_off_a]);
      c2d_cp_async_16(smem_int, src);
    }
    {
      const int k_elem = k_base + ld_k_off_b;
      const int q = k_elem / Ci;
      const int ci0 = k_elem - q * Ci;
      const int ks = q % 3;
      const int kr = q / 3;
      const uint8_t* src = nullptr;
      if (co_load < Co) {
        const int idx = ((co_load * 3 + kr) * 3 + ks) * Ci + ci0;
        src = reinterpret_cast<const uint8_t*>(&w[idx]);
      }
      uint32_t smem_int = c2d_to_smem_int(
          &B_smem[stage][ld_row_b * C2D_SMEM_K_STRIDE + ld_k_off_b]);
      c2d_cp_async_16(smem_int, src);
    }
  };

  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;

  for (int k_base = 0; k_base < K_total; k_base += C2D_BLOCK_K) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + C2D_BLOCK_K;

    if (k_next < K_total) {
      issue_load(next_stage, k_next);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    const int kA0 = 4 * l;
    const int kA2 = 4 * l + 16;

    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * C2D_SMEM_K_STRIDE + kA0]);
    uint32_t A1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * C2D_SMEM_K_STRIDE + kA0]);
    uint32_t A2 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * C2D_SMEM_K_STRIDE + kA2]);
    uint32_t A3 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * C2D_SMEM_K_STRIDE + kA2]);

    #pragma unroll
    for (int n_atom = 0; n_atom < C2D_N_ATOMS; ++n_atom) {
      int co_n = n_atom * 8 + h;
      uint32_t B0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * C2D_SMEM_K_STRIDE + kA0]);
      uint32_t B1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co_n * C2D_SMEM_K_STRIDE + kA2]);
      c2d_mma_m16n8k32_e4m3(
          dA[n_atom], dB[n_atom], dC[n_atom], dD[n_atom],
          A0, A1, A2, A3, B0, B1);
    }

    compute_stage = next_stage;
  }

  asm volatile("cp.async.wait_all;\n" ::);

  const int warp_M_off = warp_id * 16;
  auto ncdhw_offset = [&](int row, int co) {
    const int spatial = H * W;
    const int n_idx = row / spatial;
    const int rem = row - n_idx * spatial;
    const int ho = rem / W;
    const int wo = rem - ho * W;
    const int b_idx = n_idx / out_T;
    const int t_idx = n_idx - b_idx * out_T;
    return (((b_idx * Co + co) * out_T + t_idx) * H + ho) * W + wo;
  };
  #pragma unroll
  for (int n_atom = 0; n_atom < C2D_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0    = m_base + warp_M_off + h;
    int row1    = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr && co_pair < Co) {
      b0 = __bfloat162float(bias[co_pair]);
      if (co_pair + 1 < Co) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co && out_ncdhw == 0) {
      __nv_bfloat162 packAB;
      packAB.x = __float2bfloat16(dA[n_atom] * alpha + b0);
      packAB.y = __float2bfloat16(dB[n_atom] * alpha + b1);
      __nv_bfloat162 packCD;
      packCD.x = __float2bfloat16(dC[n_atom] * alpha + b0);
      packCD.y = __float2bfloat16(dD[n_atom] * alpha + b1);
      if (row0 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row0 * Co + co_pair]) = packAB;
      }
      if (row1 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row1 * Co + co_pair]) = packCD;
      }
    } else {
      auto store = [&](int row, int co, float v, float bv) {
        if (row < M_total && co < Co) {
          const int out_idx = out_ncdhw ? ncdhw_offset(row, co) : row * Co + co;
          y[out_idx] = __float2bfloat16(v * alpha + bv);
        }
      };
      store(row0, co_pair + 0, dA[n_atom], b0);
      store(row0, co_pair + 1, dB[n_atom], b1);
      store(row1, co_pair + 0, dC[n_atom], b0);
      store(row1, co_pair + 1, dD[n_atom], b1);
    }
  }
}

extern "C" int fp8_conv2d_3x3_v2_nhwc_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % C2D_BLOCK_K != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp8_conv2d_3x3_v2] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        C2D_BLOCK_K, Ci, Co);
    return -1;
  }
  if (N <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) {
    return -2;
  }
  int M = N * H * W;
  int M_tiles = (M + C2D_BLOCK_M - 1) / C2D_BLOCK_M;
  int N_tiles = (Co + C2D_BLOCK_N - 1) / C2D_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  fp8_conv2d_3x3_v2_kernel<<<dim3(total_tiles), dim3(C2D_THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, H, W, Ci, Co, alpha,
      M_tiles, N_tiles, N, 1, 0);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp8_conv2d_3x3_v2] launch err: %s\n",
                 cudaGetErrorString(e));
    return -3;
  }
  return 0;
}

extern "C" int fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int B, int T, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % C2D_BLOCK_K != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp8_conv2d_3x3_v2_ncdhw] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        C2D_BLOCK_K, Ci, Co);
    return -1;
  }
  if (B <= 0 || T <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) {
    return -2;
  }
  int N = B * T;
  int M = N * H * W;
  int M_tiles = (M + C2D_BLOCK_M - 1) / C2D_BLOCK_M;
  int N_tiles = (Co + C2D_BLOCK_N - 1) / C2D_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  fp8_conv2d_3x3_v2_kernel<<<dim3(total_tiles), dim3(C2D_THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, H, W, Ci, Co, alpha,
      M_tiles, N_tiles, B, T, 1);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp8_conv2d_3x3_v2_ncdhw] launch err: %s\n",
                 cudaGetErrorString(e));
    return -3;
  }
  return 0;
}

}  // namespace conv
}  // namespace flash_rt
