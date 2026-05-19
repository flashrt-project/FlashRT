// ================================================================
//  flash_rt — Hand FP8 Conv3d fprop, sm_120a, **v18** (= v16 +
//  virtual cache cat input + direct causal output)
//
//  v18 = v16 with two further wrapper-overhead removals for the
//  motus VAE causal-conv chain:
//
//   1. Virtual time-axis concat. Instead of forcing the wrapper to
//      run torch.cat([cache_fp8, new_fp8], dim=1) and pass the
//      stitched tensor in, v18 takes TWO input pointers:
//        cache_x_fp8 [B, T_cache, H, W, Ci]   (last 2 frames of
//                                              prev chunk's fused
//                                              quant output)
//        new_x_fp8   [B, T_new,   H, W, Ci]   (current chunk)
//      and reads from cache_x for d_in < T_cache, else new_x. Saves
//      ~6 ms aten::cat per motus inference.
//
//   2. Direct causal output. The conv output is now sized
//        y_bf16 [B, T_new, H, W, Co]
//      i.e., the wrapper no longer slices `y_v11[:, 1:T+1]` after
//      the conv runs over T_total = T_cache + T_new. Internally we
//      iterate output positions t_out = 0..T_new-1 and shift the
//      kernel's causal-tap offsets so d_in = t_out + kt
//      (kt ∈ {0,1,2}) reads input at concat positions
//      [t_out, t_out+1, t_out+2]. Saves the slice + a chunk of the
//      output write volume that was being thrown away.
//
//  All other tile geometry / smem layout / cp.async pipelining / Y-
//  major persistent walk / bias-fused epilogue (from v16) are
//  unchanged.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace conv {

// v18 = v16 (BLOCK_M=BLOCK_N=128, BLOCK_K=32, 8 warps, cp.async
// 2-stage, persistent Y-major, bias-fused epilogue) + virtual cache
// cat input + direct causal output. See header.
constexpr int V18_BLOCK_M = 128;
constexpr int V18_BLOCK_N = 128;
constexpr int V18_BLOCK_K = 32;
constexpr int V18_N_ATOMS  = V18_BLOCK_N / 8;  // 16 N-atoms per warp
constexpr int V18_NUM_WARPS = 8;
constexpr int V18_THREADS = V18_NUM_WARPS * 32;
constexpr int V18_STAGES = 2;
constexpr int V18_SMEM_K_STRIDE = 48;

__device__ __forceinline__
void v18_mma_m16n8k32_e4m3(
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

// v18 input addressing: virtual concat of cache_x and new_x along T,
// causal output range. m_global decodes (b, t_out, h_out, w_out)
// where t_out ∈ [0, T_new). Input read uses d_in = t_out + kt for
// kt ∈ {0,1,2}; the d_in coordinate runs over the virtual concat
// of length T_total = T_cache + T_new, so:
//     d_in <  T_cache → cache_x[b, d_in,            h_in, w_in, ci0]
//     d_in >= T_cache → new_x  [b, d_in - T_cache,  h_in, w_in, ci0]
// Spatial pad=1 (h_in, w_in OOB → zero-pad via nullptr).
__device__ __forceinline__
const uint8_t* v18_x_byte_ptr(const __nv_fp8_e4m3* cache_x,
                              const __nv_fp8_e4m3* new_x,
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
  int d_in = t_out + kt;             // causal: kt={0,1,2} → d_in ∈ [t_out, t_out+2]
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  // d_in is always in [0, T_new+1] = [0, T_total-1] by construction
  // (t_out ≤ T_new-1, kt ≤ 2, T_total = T_cache+T_new; with T_cache=2
  // we get d_in ≤ T_new+1 = T_total-1). No d_in OOB check needed.
  if (d_in < T_cache) {
    int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
    return reinterpret_cast<const uint8_t*>(&cache_x[idx]);
  } else {
    int d_new = d_in - T_cache;
    int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
    return reinterpret_cast<const uint8_t*>(&new_x[idx]);
  }
}

__device__ __forceinline__
const uint8_t* v18_w_byte_ptr(const __nv_fp8_e4m3* w,
                              int co, int k_global, int Co, int Ci) {
  int K_total = 27 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
  return reinterpret_cast<const uint8_t*>(&w[idx]);
}

__device__ __forceinline__
void v18_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__
void v18_cp_async_8(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 8;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__
uint32_t v18_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// gridDim.x = total_tiles. Y-major decode: consecutive blockIdx share
// the SAME M (different N). Hardware raster launches CTAs in blockIdx
// order, so a wave of 170*3=510 active CTAs covers ~127 contiguous
// M-tiles × 4 N-tiles = same M inputs reused across N-tile triples.
//
// launch_bounds(256, 2) keeps reg occupancy = v8 = 2-3 CTAs/SM.
__global__ void __launch_bounds__(V18_THREADS, 2)
fp8_conv3d_v18_kernel(
    const __nv_fp8_e4m3* __restrict__ cache_x,  // [B, T_cache, H, W, Ci]
    const __nv_fp8_e4m3* __restrict__ new_x,    // [B, T_new,   H, W, Ci]
    const __nv_fp8_e4m3* __restrict__ w,
          __nv_bfloat16* __restrict__ y,         // [B, Co, T_new, H, W]
    const __nv_bfloat16* __restrict__ residual,  // [B, Co, T_new, H, W] or nullptr
    const __nv_bfloat16* __restrict__ bias,     // [Co] or nullptr
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem[V18_STAGES][V18_BLOCK_M * V18_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem[V18_STAGES][V18_BLOCK_N * V18_SMEM_K_STRIDE];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  // M_total spans the CAUSAL output range (not the virtual concat).
  const int M_total  = N * T_new * H * W;
  const int K_total  = 27 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 16;
  // BLOCK_N=128: B_smem 128 rows × 32 K (48 stride) = 6144 bytes.
  // 256 threads × 16 bytes (1× cp.async.16) = 4096 — not enough.
  // Use 256 threads × 24 bytes? No — go to 2 threads/row × 128 rows
  // × 16 B/thread = 4096 < 6144. Still short.
  // Better: each thread loads 24 bytes via cp.async.16 + cp.async.8.
  // Distribute: ld_row = t/2 (0..127 covers 128 rows),
  //             ld_k_off = (t&1)*16 → 0 or 16 (loads bytes 0..15 and
  //             16..31, covering full 32 K bytes per row).
  // Total: 128 rows × 2 threads/row × 16 bytes/thread = 4096 ≠ 6144.
  // Wait: 128*32 valid bytes = 4096. Padding bytes 32..47 don't need
  // load. So 4096 useful + 2048 padding (uninit). 4096/256 = 16 B/thread.
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 16;

  // Y-major decode: blockIdx.x = m_idx * N_tiles + n_idx
  {
    int tile_idx = blockIdx.x;
    int m_idx  = tile_idx / N_tiles;
    int n_idx  = tile_idx % N_tiles;
    int m_base = m_idx * V18_BLOCK_M;
    int co_base = n_idx * V18_BLOCK_N;

    if (m_base >= M_total || co_base >= Co) return;

    float dA[V18_N_ATOMS] = {0};
    float dB[V18_N_ATOMS] = {0};
    float dC[V18_N_ATOMS] = {0};
    float dD[V18_N_ATOMS] = {0};

    // ── cp.async issue helper for this tile ──
    auto issue_load = [&](int stage, int k_base) {
      {
        const uint8_t* src = v18_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a,
                                            N, T_cache, T_new, H, W, Ci);
        uint32_t smem_int = v18_to_smem_int(
            &A_smem[stage][ld_row_a * V18_SMEM_K_STRIDE + ld_k_off_a]);
        v18_cp_async_16(smem_int, src);
      }
      {
        const uint8_t* src = v18_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b,
                                            Co, Ci);
        uint32_t smem_int = v18_to_smem_int(
            &B_smem[stage][ld_row_b * V18_SMEM_K_STRIDE + ld_k_off_b]);
        v18_cp_async_16(smem_int, src);
      }
    };

    // Prologue
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;

    for (int k_base = 0; k_base < K_total; k_base += V18_BLOCK_K) {
      int next_stage = compute_stage ^ 1;
      int k_next = k_base + V18_BLOCK_K;

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
          &A_smem[compute_stage][rA0 * V18_SMEM_K_STRIDE + kA0]);
      uint32_t A1 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA1 * V18_SMEM_K_STRIDE + kA0]);
      uint32_t A2 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA0 * V18_SMEM_K_STRIDE + kA2]);
      uint32_t A3 = *reinterpret_cast<const uint32_t*>(
          &A_smem[compute_stage][rA1 * V18_SMEM_K_STRIDE + kA2]);

      #pragma unroll
      for (int n_atom = 0; n_atom < V18_N_ATOMS; ++n_atom) {
        int co_n = n_atom * 8 + h;
        uint32_t B0 = *reinterpret_cast<const uint32_t*>(
            &B_smem[compute_stage][co_n * V18_SMEM_K_STRIDE + kA0]);
        uint32_t B1 = *reinterpret_cast<const uint32_t*>(
            &B_smem[compute_stage][co_n * V18_SMEM_K_STRIDE + kA2]);
        v18_mma_m16n8k32_e4m3(
            dA[n_atom], dB[n_atom], dC[n_atom], dD[n_atom],
            A0, A1, A2, A3, B0, B1);
      }

      compute_stage = next_stage;
    }

    asm volatile("cp.async.wait_all;\n" ::);

    // v18 writes NCDHW directly and can fuse ResidualBlock add into
    // the conv epilogue. This removes the separate NDHWC->NCDHW
    // transpose + residual-add launches. Stores are less coalesced
    // than v17's NDHWC epilogue, so this path is measurement-gated.
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
    for (int n_atom = 0; n_atom < V18_N_ATOMS; ++n_atom) {
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
}

// Detect # SMs at runtime for persistent CTA count.
//
// Inputs:
//   cache_x_fp8 : [B, T_cache, H, W, Ci] fp8_e4m3
//   new_x_fp8   : [B, T_new,   H, W, Ci] fp8_e4m3
//   w_fp8       : [Co, 3, 3, 3, Ci]      fp8_e4m3
//   bias_bf16   : [Co]                   bf16  (or nullptr)
// Output:
//   y_bf16      : [B, Co, T_new, H, W]   bf16  (causal window, T_new frames)
extern "C" int fp8_conv3d_v18_ncdhw_res_bf16out(
    const void* cache_x_fp8, const void* new_x_fp8,
    const void* w_fp8, void* y_bf16,
    const void* bias_bf16, const void* residual_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % V18_BLOCK_K != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp8_conv3d_v18] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        V18_BLOCK_K, Ci, Co);
    return -1;
  }
  if (T_cache != 2) {
    // Causal output mapping d_in = t_out + kt assumes T_cache=2 so
    // d_in ∈ [t_out, t_out+2] is always within [0, T_total-1]. For
    // other T_cache values the math still holds but the wrapper
    // contract is fixed to 2; surface it as a hard error rather than
    // silently misbehaving.
    std::fprintf(stderr,
        "[fp8_conv3d_v18] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + V18_BLOCK_M - 1) / V18_BLOCK_M;
  int N_tiles = (Co + V18_BLOCK_N - 1) / V18_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(V18_THREADS);
  fp8_conv3d_v18_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(cache_x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(new_x_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(residual_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, T_cache, T_new, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp8_conv3d_v18] launch err: %s\n",
                 cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace conv
}  // namespace flash_rt
