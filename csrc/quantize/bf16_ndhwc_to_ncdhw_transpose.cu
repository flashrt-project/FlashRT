// SPDX-License-Identifier: Apache-2.0
// G7.23 — fast NDHWC->NCDHW transpose. See header.

#include "bf16_ndhwc_to_ncdhw_transpose.cuh"

#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int kTile = 32;          // tile width on both axes
constexpr int kPad  = 1;           // smem stride pad (33 elts) — avoids bank conflict on transposed read
constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;       // 32x8 = 256 threads/CTA
constexpr int kThreads  = kThreadsX * kThreadsY;

// One CTA owns (b, t, h, w_block_start). Iterates over C in tiles of
// kTile=32. Per tile: load 32(W) x 32(C) NDHWC chunk to smem, sync,
// transposed write 32(C) x 32(W) NCDHW chunk.
//
// Load coalescing: NDHWC inner axis is C, so 32-thread row reads 32
// consecutive bf16 = 64 B = 1 cache line.
// Store coalescing: NCDHW inner axis is W, so 32-thread row writes 32
// consecutive bf16 = 64 B = 1 cache line.
__global__ void bf16_ndhwc_to_ncdhw_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B,T,H,W,C]
    __nv_bfloat16*       __restrict__ y,    // [B,C,T,H,W]
    int B, int C, int T, int H, int W,
    int w_blocks_per_row, int c_tiles)
{
  // smem tile [W=32 rows][C=32+pad cols] bf16.
  __shared__ __nv_bfloat16 sm[kTile][kTile + kPad];

  // Decode CTA id → (b, t, h, w_block).
  int wb   = blockIdx.x % w_blocks_per_row;
  int rest = blockIdx.x / w_blocks_per_row;
  int hwt  = T * H;
  int b    = rest / hwt;
  int rh   = rest - b * hwt;
  int t    = rh / H;
  int h    = rh - t * H;
  if (b >= B) return;

  const int w_start = wb * kTile;
  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;          // 0..7 — 8 warp-rows

  // Strides (in elements).
  const long long x_thw_off =
      (((long long)b * T + t) * H + h) * (long long)W;     // bf16 offset to (b,t,h,0,0)
  const long long y_b_stride = (long long)C * T * H * W;
  const long long y_th_off =
      ((long long)t * H + h) * (long long)W;               // bf16 offset to (b,0,t,h,0)

  for (int c_tile_idx = 0; c_tile_idx < c_tiles; ++c_tile_idx) {
    const int c_base = c_tile_idx * kTile;

    // ── Load 32(W) × 32(C) NDHWC chunk into smem ──
    // Row tx in smem ↔ w = w_start + tx.
    // Col c_local in smem ↔ c = c_base + c_local.
    // 256 threads cover 32 W rows × 32 C cols = 1024 elts; 4 rows per ty
    // (32W / 8 ty-warps = 4 W rows per warp). Each thread loads 1 elt
    // per inner iter; inner unrolls 4 W rows.
    #pragma unroll
    for (int wi = 0; wi < kTile / kThreadsY; ++wi) {
      int w_loc = ty + wi * kThreadsY;       // 0..31
      int w_glob = w_start + w_loc;
      int c_loc = tx;
      int c_glob = c_base + c_loc;
      __nv_bfloat16 v;
      if (w_glob < W && c_glob < C) {
        long long idx = (x_thw_off + (long long)w_glob) * (long long)C + c_glob;
        v = x[idx];
      } else {
        v = __float2bfloat16(0.f);
      }
      sm[w_loc][c_loc] = v;
    }
    __syncthreads();

    // ── Store smem → NCDHW global ──
    // Row tx in smem still ↔ w; col c_local ↔ c. We re-index so threads
    // vary along the W axis (innermost in NCDHW).
    // Layout: thread (tx, ty) writes y[b, c=c_base+ty*4+wi, t, h, w=w_start+tx]
    // — 4 c-rows per thread.
    #pragma unroll
    for (int ci = 0; ci < kTile / kThreadsY; ++ci) {
      int c_loc = ty + ci * kThreadsY;       // 0..31 → c
      int c_glob = c_base + c_loc;
      int w_loc = tx;                         // 0..31 → w
      int w_glob = w_start + w_loc;
      if (w_glob < W && c_glob < C) {
        long long idx = (long long)b * y_b_stride
                       + (long long)c_glob * T * H * W
                       + y_th_off + w_glob;
        y[idx] = sm[w_loc][c_loc];
      }
    }
    __syncthreads();
  }
}

__global__ void bf16_ndhwc_to_ncdhw_add_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B,T,H,W,C]
    const __nv_bfloat16* __restrict__ r,    // [B,C,T,H,W]
    __nv_bfloat16*       __restrict__ y,    // [B,C,T,H,W]
    int B, int C, int T, int H, int W,
    long long rs_b, long long rs_c, long long rs_t,
    long long rs_h, long long rs_w,
    int w_blocks_per_row, int c_tiles)
{
  __shared__ __nv_bfloat16 sm[kTile][kTile + kPad];

  int wb   = blockIdx.x % w_blocks_per_row;
  int rest = blockIdx.x / w_blocks_per_row;
  int hwt  = T * H;
  int b    = rest / hwt;
  int rh   = rest - b * hwt;
  int t    = rh / H;
  int h    = rh - t * H;
  if (b >= B) return;

  const int w_start = wb * kTile;
  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;

  const long long x_thw_off =
      (((long long)b * T + t) * H + h) * (long long)W;
  const long long y_b_stride = (long long)C * T * H * W;
  const long long y_th_off =
      ((long long)t * H + h) * (long long)W;

  for (int c_tile_idx = 0; c_tile_idx < c_tiles; ++c_tile_idx) {
    const int c_base = c_tile_idx * kTile;

    #pragma unroll
    for (int wi = 0; wi < kTile / kThreadsY; ++wi) {
      int w_loc = ty + wi * kThreadsY;
      int w_glob = w_start + w_loc;
      int c_loc = tx;
      int c_glob = c_base + c_loc;
      __nv_bfloat16 v;
      if (w_glob < W && c_glob < C) {
        long long idx = (x_thw_off + (long long)w_glob) * (long long)C + c_glob;
        v = x[idx];
      } else {
        v = __float2bfloat16(0.f);
      }
      sm[w_loc][c_loc] = v;
    }
    __syncthreads();

    #pragma unroll
    for (int ci = 0; ci < kTile / kThreadsY; ++ci) {
      int c_loc = ty + ci * kThreadsY;
      int c_glob = c_base + c_loc;
      int w_loc = tx;
      int w_glob = w_start + w_loc;
      if (w_glob < W && c_glob < C) {
        long long idx = (long long)b * y_b_stride
                       + (long long)c_glob * T * H * W
                       + y_th_off + w_glob;
        long long ridx = (long long)b * rs_b
                        + (long long)c_glob * rs_c
                        + (long long)t * rs_t
                        + (long long)h * rs_h
                        + (long long)w_glob * rs_w;
        float v = __bfloat162float(sm[w_loc][c_loc])
                + __bfloat162float(r[ridx]);
        y[idx] = __float2bfloat16(v);
      }
    }
    __syncthreads();
  }
}

__global__ void bf16_ndhwc_to_ncdhw_bias_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B,T,H,W,C]
    const __nv_bfloat16* __restrict__ bias, // [C]
    __nv_bfloat16*       __restrict__ y,    // [B,C,T,H,W]
    int B, int C, int T, int H, int W,
    int w_blocks_per_row, int c_tiles)
{
  __shared__ __nv_bfloat16 sm[kTile][kTile + kPad];

  int wb   = blockIdx.x % w_blocks_per_row;
  int rest = blockIdx.x / w_blocks_per_row;
  int hwt  = T * H;
  int b    = rest / hwt;
  int rh   = rest - b * hwt;
  int t    = rh / H;
  int h    = rh - t * H;
  if (b >= B) return;

  const int w_start = wb * kTile;
  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;

  const long long x_thw_off =
      (((long long)b * T + t) * H + h) * (long long)W;
  const long long y_b_stride = (long long)C * T * H * W;
  const long long y_th_off =
      ((long long)t * H + h) * (long long)W;

  for (int c_tile_idx = 0; c_tile_idx < c_tiles; ++c_tile_idx) {
    const int c_base = c_tile_idx * kTile;

    #pragma unroll
    for (int wi = 0; wi < kTile / kThreadsY; ++wi) {
      int w_loc = ty + wi * kThreadsY;
      int w_glob = w_start + w_loc;
      int c_loc = tx;
      int c_glob = c_base + c_loc;
      __nv_bfloat16 v;
      if (w_glob < W && c_glob < C) {
        long long idx = (x_thw_off + (long long)w_glob) * (long long)C + c_glob;
        v = x[idx];
      } else {
        v = __float2bfloat16(0.f);
      }
      sm[w_loc][c_loc] = v;
    }
    __syncthreads();

    #pragma unroll
    for (int ci = 0; ci < kTile / kThreadsY; ++ci) {
      int c_loc = ty + ci * kThreadsY;
      int c_glob = c_base + c_loc;
      int w_loc = tx;
      int w_glob = w_start + w_loc;
      if (w_glob < W && c_glob < C) {
        long long idx = (long long)b * y_b_stride
                       + (long long)c_glob * T * H * W
                       + y_th_off + w_glob;
        float v = __bfloat162float(sm[w_loc][c_loc])
                + __bfloat162float(bias[c_glob]);
        y[idx] = __float2bfloat16(v);
      }
    }
    __syncthreads();
  }
}

}  // namespace

int bf16_ndhwc_to_ncdhw_transpose(
    const void* x_NDHWC,
    void*       y_NCDHW,
    int B, int C, int T, int H, int W,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;

  const int w_blocks_per_row = (W + kTile - 1) / kTile;
  const int c_tiles          = (C + kTile - 1) / kTile;
  const long long n_ctas     = (long long)B * T * H * w_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -2;

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  bf16_ndhwc_to_ncdhw_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_NDHWC),
      reinterpret_cast<__nv_bfloat16*>(y_NCDHW),
      B, C, T, H, W, w_blocks_per_row, c_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[ndhwc_to_ncdhw] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

int bf16_ndhwc_to_ncdhw_bias_bf16(
    const void* x_NDHWC,
    const void* bias_C,
    void*       y_NCDHW,
    int B, int C, int T, int H, int W,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if (bias_C == nullptr) return -3;

  const int w_blocks_per_row = (W + kTile - 1) / kTile;
  const int c_tiles          = (C + kTile - 1) / kTile;
  const long long n_ctas     = (long long)B * T * H * w_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -2;

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  bf16_ndhwc_to_ncdhw_bias_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_NDHWC),
      reinterpret_cast<const __nv_bfloat16*>(bias_C),
      reinterpret_cast<__nv_bfloat16*>(y_NCDHW),
      B, C, T, H, W, w_blocks_per_row, c_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[ndhwc_to_ncdhw_bias] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

int bf16_ndhwc_to_ncdhw_add_bf16(
    const void* x_NDHWC,
    const void* residual_NCDHW,
    void*       y_NCDHW,
    int B, int C, int T, int H, int W,
    long long rs_b, long long rs_c, long long rs_t,
    long long rs_h, long long rs_w,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if (residual_NCDHW == nullptr) return -3;

  const int w_blocks_per_row = (W + kTile - 1) / kTile;
  const int c_tiles          = (C + kTile - 1) / kTile;
  const long long n_ctas     = (long long)B * T * H * w_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -2;

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  bf16_ndhwc_to_ncdhw_add_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_NDHWC),
      reinterpret_cast<const __nv_bfloat16*>(residual_NCDHW),
      reinterpret_cast<__nv_bfloat16*>(y_NCDHW),
      B, C, T, H, W, rs_b, rs_c, rs_t, rs_h, rs_w,
      w_blocks_per_row, c_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[ndhwc_to_ncdhw_add] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

}  // namespace quantize
}  // namespace flash_rt
