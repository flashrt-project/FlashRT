// SPDX-License-Identifier: Apache-2.0
// ================================================================
// FlashRT — VLA-JEPA boundary preprocessing (native, graph-capturable)
//
// The delivered VLA-JEPA pipeline's per-decision boundary — raw uint8 camera
// frames and the 8-dim robot state -> the ViT's pixel_values and the
// normalized state token — ran as torch tensor ops: an area resize, an
// antialias bicubic resize, a mean/std normalize, a patchify permute and
// several casts/copies, i.e. 15 ATen kernels inside the captured graph.
// These kernels reproduce the same arithmetic so the boundary is entirely
// repository-native:
//
//   area resize   : adaptive average pooling over the window
//                   [floor(o*in/out), ceil((o+1)*in/out)) per axis, summed in
//                   row-major order and divided by the window size -- the
//                   `interpolate(mode="area")` semantics.
//   bicubic       : align_corners=false, Keys a=-0.75 taps at
//                   center = (o + 0.5) * scale - 0.5, accumulated in the same
//                   order as the antialias kernel it replaces (which, for an
//                   upsampling factor, uses filter_scale 1 and therefore the
//                   plain bicubic weights).
//   patchify      : (x - mean) / std -> temporal tile -> patch grid -> the
//                   processor's permute/reshape order, one launch, bf16 out.
//   state         : (raw - mean) / (std + eps) -> bf16.
//
// The uint8 load folds the /255 rescale into the area resize's first touch.
// ================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

// Adaptive average pooling takes the window [floor(o*in/out),
// ceil((o+1)*in/out)) per axis. The caller passes the bounds as an exact
// integer table (built on the host) instead of recomputing them per element:
// the division is ~20 cycles, and under --use_fast_math the compiler is free
// to reassociate `o * in / out` into `o * (in/out)`, which moves the exact
// boundaries by an ulp and silently changes the window. bounds[o] = {start,
// end} for one axis.
__device__ __forceinline__ int2 aa_bounds(const int2* __restrict__ bounds, int o) {
  return bounds[o];
}

// Keys cubic with the coefficient the *antialias* bicubic resize uses
// (a = -0.5; the non-antialias resize uses -0.75). Verified against the
// reference's own weight matrix: a = -0.5 with unit filter scale reproduces it
// to 4e-7, and the taps are renormalised by their sum (which is how the
// reference handles the taps that fall outside the image).
__device__ __forceinline__ float cubic_coeff(float x) {
  constexpr float A = -0.5f;
  const float ax = fabsf(x);
  if (ax <= 1.0f) {
    return ((A + 2.0f) * ax - (A + 3.0f)) * ax * ax + 1.0f;
  }
  if (ax < 2.0f) {
    return (((ax - 5.0f) * ax + 8.0f) * ax - 4.0f) * A;
  }
  return 0.0f;
}

// Area resize + /255, one thread per output element.
// src: [V, C, H, W] uint8, dst: [V, C, oh, ow] fp32.
// Index is long or int: the int form (when the sizes fit) keeps the index
// decomposition off the 64-bit division path; the arithmetic is the same.
template <typename Index>
__global__ void vla_area_resize_kernel(const uint8_t* __restrict__ src,
                                       float* __restrict__ dst,
                                       const int2* __restrict__ ybounds,
                                       const int2* __restrict__ xbounds,
                                       int V, int H, int W, int C,
                                       int oh, int ow) {
  const Index total = (Index)V * C * oh * ow;
  const Index stride = (Index)gridDim.x * blockDim.x;
  for (Index idx = (Index)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ox = idx % ow;
    const int oy = (idx / ow) % oh;
    const int c = (idx / ((Index)ow * oh)) % C;
    const int v = idx / ((Index)ow * oh * C);
    const int2 yb = aa_bounds(ybounds, oy), xb = aa_bounds(xbounds, ox);
    const int y0 = yb.x, y1 = yb.y, x0 = xb.x, x1 = xb.y;
    const uint8_t* base = src + ((long)v * C + c) * H * W + (long)y0 * W;
    float acc = 0.0f;
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        acc += static_cast<float>(base[(long)(y - y0) * W + x]) / 255.0f;
      }
    }
    dst[idx] = acc / static_cast<float>((y1 - y0) * (x1 - x0));
  }
}

// Bicubic resize (align_corners=false) with the antialias kernel's tap set.
// src: [VC, ih, iw] fp32 contiguous, dst: [VC, oh, ow] fp32.
template <bool RowMajorAcc>
__global__ void vla_bicubic_resize_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          int N, int ih, int iw, int oh, int ow) {
  const long total = (long)N * oh * ow;
  const long stride = (long)gridDim.x * blockDim.x;
  const float sh = (float)ih / (float)oh;
  const float sw = (float)iw / (float)ow;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ox = idx % ow;
    const int oy = (idx / ow) % oh;
    const int n = idx / ((long)ow * oh);
    const float cy = ((float)oy + 0.5f) * sh - 0.5f;
    const float cx = ((float)ox + 0.5f) * sw - 0.5f;
    const int ly = (int)floorf(cy - 2.0f), hy = (int)ceilf(cy + 2.0f);
    const int lx = (int)floorf(cx - 2.0f), hx = (int)ceilf(cx + 2.0f);
    const float* plane = src + (long)n * ih * iw;
    float acc = 0.0f, wsum = 0.0f;
    if (RowMajorAcc) {
      for (int y = ly; y < hy; ++y) {
        if (y < 0 || y >= ih) continue;
        const float wy = cubic_coeff((float)y - cy);
        for (int x = lx; x < hx; ++x) {
          if (x < 0 || x >= iw) continue;
          const float wx = cubic_coeff((float)x - cx);
          acc += (wy * wx) * plane[(long)y * iw + x];
          wsum += wy * wx;
        }
      }
    } else {
      for (int y = ly; y < hy; ++y) {
        if (y < 0 || y >= ih) continue;
        const float wy = cubic_coeff((float)y - cy);
        float row = 0.0f;
        for (int x = lx; x < hx; ++x) {
          if (x < 0 || x >= iw) continue;
          const float wx = cubic_coeff((float)x - cx);
          row += wx * plane[(long)y * iw + x];
          wsum += wy * wx;
        }
        acc += wy * row;
      }
    }
    acc = (wsum != 0.0f) ? acc / wsum : acc;
    dst[idx] = acc;
  }
}

// Same taps, weights and accumulation order as vla_bicubic_resize_kernel<true>
// (bit-identical), with the separable column weights evaluated once per
// output pixel instead of once per tap (the window is at most 5 x 5) and
// 32-bit index math. Needs N * oh * ow < 2^31.
__global__ void vla_bicubic_resize_sep_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int N, int ih, int iw, int oh, int ow) {
  const int total = N * oh * ow;
  const float sh = (float)ih / (float)oh;
  const float sw = (float)iw / (float)ow;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int ox = idx % ow;
    const int oy = (idx / ow) % oh;
    const int n = idx / (ow * oh);
    const float cy = ((float)oy + 0.5f) * sh - 0.5f;
    const float cx = ((float)ox + 0.5f) * sw - 0.5f;
    const int ly = (int)floorf(cy - 2.0f), hy = (int)ceilf(cy + 2.0f);
    const int lx = (int)floorf(cx - 2.0f), hx = (int)ceilf(cx + 2.0f);
    float wxs[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) wxs[k] = cubic_coeff((float)(lx + k) - cx);
    const float* plane = src + (size_t)n * ih * iw;
    float acc = 0.0f, wsum = 0.0f;
    for (int y = ly; y < hy; ++y) {
      if (y < 0 || y >= ih) continue;
      const float wy = cubic_coeff((float)y - cy);
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const int x = lx + k;
        if (x >= hx || x < 0 || x >= iw) continue;
        acc += (wy * wxs[k]) * plane[y * iw + x];
        wsum += wy * wxs[k];
      }
    }
    acc = (wsum != 0.0f) ? acc / wsum : acc;
    dst[idx] = acc;
  }
}

// Bicubic resize straight into the patchified bf16 rows: one thread per
// resized pixel computes the value as vla_bicubic_resize_sep_kernel does and
// stores (v - mean) / std at its TPS temporal copies in the processor's
// patch order, exactly what vla_patchify_kernel would read back from the fp32
// intermediate (bit-identical, one launch and one fp32 round trip fewer).
template <int PS, int TPS, int M>
__global__ void vla_bicubic_patchify_kernel(const float* __restrict__ src,
                                            __nv_bfloat16* __restrict__ out,
                                            float mean, float std_, int V, int C,
                                            int ih, int iw, int oh, int ow) {
  const int total = V * C * oh * ow;
  const float sh = (float)ih / (float)oh;
  const float sw = (float)iw / (float)ow;
  const int GI = oh / (PS * M), GJ = ow / (PS * M);
  const int cols = C * TPS * PS * PS;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int ox = idx % ow;
    const int oy = (idx / ow) % oh;
    const int n = idx / (ow * oh);            // v * C + c
    const float cy = ((float)oy + 0.5f) * sh - 0.5f;
    const float cx = ((float)ox + 0.5f) * sw - 0.5f;
    const int ly = (int)floorf(cy - 2.0f), hy = (int)ceilf(cy + 2.0f);
    const int lx = (int)floorf(cx - 2.0f), hx = (int)ceilf(cx + 2.0f);
    float wxs[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) wxs[k] = cubic_coeff((float)(lx + k) - cx);
    const float* plane = src + (size_t)n * ih * iw;
    float acc = 0.0f, wsum = 0.0f;
    for (int y = ly; y < hy; ++y) {
      if (y < 0 || y >= ih) continue;
      const float wy = cubic_coeff((float)y - cy);
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const int x = lx + k;
        if (x >= hx || x < 0 || x >= iw) continue;
        acc += (wy * wxs[k]) * plane[y * iw + x];
        wsum += wy * wxs[k];
      }
    }
    acc = (wsum != 0.0f) ? acc / wsum : acc;
    const __nv_bfloat16 o = __float2bfloat16((acc - mean) / std_);
    const int v = n / C, c = n % C;
    const int I = oy / (PS * M), A = (oy / PS) % M, p = oy % PS;
    const int J = ox / (PS * M), B = (ox / PS) % M, q = ox % PS;
    const long row = ((((long)v * GI + I) * GJ + J) * M + A) * M + B;
#pragma unroll
    for (int t = 0; t < TPS; ++t)
      out[row * cols + ((c * TPS + t) * PS + p) * PS + q] = o;
  }
}

// Per-axis bicubic tap tables for vla_bicubic_patchify_kernel's geometry:
// for every output coordinate the 5 candidate taps' source index (clamped)
// and weight -- cubic_coeff exactly as that kernel evaluates it, 0 for the
// taps it skips (outside [lo, hi) or outside the image). Built once on the
// device so the weights are the same floats.
__global__ void vla_bicubic_taps_kernel(int in_size, int out_size, float* __restrict__ w,
                                        int* __restrict__ idx) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_size) return;
  const float sc = (float)in_size / (float)out_size;
  const float c = ((float)o + 0.5f) * sc - 0.5f;
  const int lo = (int)floorf(c - 2.0f), hi = (int)ceilf(c + 2.0f);
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int t = lo + k;
    const bool ok = t < hi && t >= 0 && t < in_size;
    w[o * 5 + k] = ok ? cubic_coeff((float)t - c) : 0.0f;
    idx[o * 5 + k] = t < 0 ? 0 : (t >= in_size ? in_size - 1 : t);
  }
}

// vla_bicubic_patchify_kernel with the taps from the tables above: the same
// products and accumulation order (rows ascending, then columns), so the
// same bytes; skipped taps contribute exact zeros (acc and wsum never hold
// -0 here: the source pixels are non-negative). OH / OW / PS / TPS / M are
// compile-time, so the output-row decomposition is shifts.
template <int PS, int TPS, int M, int OH, int OW>
__global__ void __launch_bounds__(256)
vla_bicubic_patchify_tab_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ out,
                                const float* __restrict__ wy_t, const int* __restrict__ iy_t,
                                const float* __restrict__ wx_t, const int* __restrict__ ix_t,
                                float mean, float std_, int V, int C, int ih, int iw) {
  const int total = V * C * OH * OW;
  constexpr int GI = OH / (PS * M), GJ = OW / (PS * M);
  const int cols = C * TPS * PS * PS;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ox = idx % OW;
  const int oy = (idx / OW) % OH;
  const int n = idx / (OW * OH);            // v * C + c
  float wxs[5];
  int xs[5];
#pragma unroll
  for (int k = 0; k < 5; ++k) { wxs[k] = wx_t[ox * 5 + k]; xs[k] = ix_t[ox * 5 + k]; }
  const float* plane = src + (size_t)n * ih * iw;
  float acc = 0.0f, wsum = 0.0f;
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const float wy = wy_t[oy * 5 + j];
    const float* rowp = plane + iy_t[oy * 5 + j] * iw;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      acc += (wy * wxs[k]) * rowp[xs[k]];
      wsum += wy * wxs[k];
    }
  }
  acc = (wsum != 0.0f) ? acc / wsum : acc;
  const __nv_bfloat16 o = __float2bfloat16((acc - mean) / std_);
  const int v = n / C, c = n % C;
  const int I = oy / (PS * M), A = (oy / PS) % M, p = oy % PS;
  const int J = ox / (PS * M), B = (ox / PS) % M, q = ox % PS;
  const long row = ((((long)v * GI + I) * GJ + J) * M + A) * M + B;
#pragma unroll
  for (int t = 0; t < TPS; ++t)
    out[row * cols + ((c * TPS + t) * PS + p) * PS + q] = o;
}

// (x - mean) / std, temporal tile, patchify and bf16 cast in one launch.
// src: [V, C, H, W] fp32, out: [V * (gh/m) * (gw/m) * m * m, C * tps * ps * ps]
// bf16 in the processor's permute/reshape order.
//
// The row/column decompositions are divisions by ps, tps, m and the patch
// grid. With those as template parameters every division folds into a shift,
// which is what makes this kernel bandwidth-bound instead of divider-bound
// (the runtime-divisor form measured 45 us, a 20x-bandwidth-limit outlier).
template <int PS, int TPS, int M, int GI, int GJ>
__global__ void vla_patchify_kernel(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ out,
                                    float mean, float std_,
                                    int V, int C, int H, int W) {
  const long cols = (long)C * TPS * PS * PS;
  const long total = (long)V * GI * GJ * M * M * cols;
  const long stride = (long)gridDim.x * blockDim.x;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const long row = idx / cols;
    const long col = idx % cols;
    const int q = (int)(col & (PS - 1));
    const int p = (int)((col / PS) & (PS - 1));
    const int t = (int)((col / ((long)PS * PS)) & (TPS - 1));
    const int c = (int)(col / ((long)PS * PS * TPS));
    const int B = (int)(row & (M - 1));
    const int A = (int)((row / M) & (M - 1));
    const int J = (int)((row / ((long)M * M)) % GJ);
    const int I = (int)((row / ((long)M * M * GJ)) % GI);
    const int v = (int)(row / ((long)M * M * GJ * GI));
    (void)t;  // the temporal tile replicates the same frame
    const int y = (I * M + A) * PS + p;
    const int x = (J * M + B) * PS + q;
    const float val = src[((long)v * C + c) * H * W + (long)y * W + x];
    out[idx] = __float2bfloat16((val - mean) / std_);
  }
}

// Generic fallback for shapes whose patch geometry is not one of the
// instantiations below (same math, runtime divisors).
__global__ void vla_patchify_generic_kernel(const float* __restrict__ src,
                                            __nv_bfloat16* __restrict__ out,
                                            float mean, float std_, int V, int C,
                                            int H, int W, int tps, int ps,
                                            int m) {
  const int gh = H / ps, gw = W / ps;
  const int gi = gh / m, gj = gw / m;
  const long cols = (long)C * tps * ps * ps;
  const long total = (long)V * gi * gj * m * m * cols;
  const long stride = (long)gridDim.x * blockDim.x;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const long row = idx / cols;
    const long col = idx % cols;
    const int q = (int)(col % ps);
    const int p = (int)((col / ps) % ps);
    const int t = (int)((col / ((long)ps * ps)) % tps);
    const int c = (int)(col / ((long)ps * ps * tps));
    const int B = (int)(row % m);
    const int A = (int)((row / m) % m);
    const int J = (int)((row / ((long)m * m)) % gj);
    const int I = (int)((row / ((long)m * m * gj)) % gi);
    const int v = (int)(row / ((long)m * m * gj * gi));
    (void)t;
    const int y = (I * m + A) * ps + p;
    const int x = (J * m + B) * ps + q;
    const float val = src[((long)v * C + c) * H * W + (long)y * W + x];
    out[idx] = __float2bfloat16((val - mean) / std_);
  }
}

// (raw - mean) / (std + eps) -> bf16, one thread per element.
__global__ void vla_state_norm_kernel(const float* __restrict__ raw,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ std_,
                                      float eps, int n,
                                      __nv_bfloat16* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2bfloat16((raw[i] - mean[i]) / (std_[i] + eps));
}

int grid_for(long total, int threads) {
  const long b = (total + threads - 1) / threads;
  if (b < 1) return 1;
  return static_cast<int>(b > 65535 * 8 ? 65535 * 8 : b);
}

}  // namespace

void vla_frames_area_resize(const uint8_t* frames, float* out,
                            const void* ybounds, const void* xbounds, int views,
                            int H, int W, int C, int oh, int ow,
                            cudaStream_t stream) {
  const long total = (long)views * C * oh * ow;
  const int threads = 256;
  if (total < (1L << 31) && (long)views * C * H * W < (1L << 31)) {
    vla_area_resize_kernel<int><<<grid_for(total, threads), threads, 0, stream>>>(
        frames, out, reinterpret_cast<const int2*>(ybounds),
        reinterpret_cast<const int2*>(xbounds), views, H, W, C, oh, ow);
    return;
  }
  vla_area_resize_kernel<long><<<grid_for(total, threads), threads, 0, stream>>>(
      frames, out, reinterpret_cast<const int2*>(ybounds),
      reinterpret_cast<const int2*>(xbounds), views, H, W, C, oh, ow);
}

void vla_frames_bicubic_resize(const float* src, float* out, int n, int ih,
                               int iw, int oh, int ow, cudaStream_t stream) {
  const long total = (long)n * oh * ow;
  const int threads = 256;
  // Row-major accumulation matches the reference kernel's tap order.
  if (total < (1L << 31) && (long)ih * iw < (1L << 31)) {
    vla_bicubic_resize_sep_kernel<<<grid_for(total, threads), threads, 0, stream>>>(
        src, out, n, ih, iw, oh, ow);
    return;
  }
  vla_bicubic_resize_kernel<true><<<grid_for(total, threads), threads, 0, stream>>>(
      src, out, n, ih, iw, oh, ow);
}

bool vla_frames_bicubic_patchify(const float* src, __nv_bfloat16* out, int views,
                                 int C, int ih, int iw, int oh, int ow, int tps,
                                 int ps, int merge, float mean, float std_,
                                 cudaStream_t stream) {
  if (ps != 16 || tps != 2 || merge != 2 || oh % (ps * merge) || ow % (ps * merge))
    return false;
  const long total = (long)views * C * oh * ow;
  if (total >= (1L << 31) || (long)ih * iw >= (1L << 31)) return false;
  const int threads = 256;
  if (oh == 256 && ow == 256) {
    // tap tables for this (ih, iw) -> 256 x 256 geometry, built once
    static float* wy = nullptr; static int* iy = nullptr;
    static float* wx = nullptr; static int* ix = nullptr;
    static int tab_ih = -1, tab_iw = -1;
    if (tab_ih != ih || tab_iw != iw) {
      if (!wy) {
        cudaMalloc(&wy, 256 * 5 * sizeof(float)); cudaMalloc(&iy, 256 * 5 * sizeof(int));
        cudaMalloc(&wx, 256 * 5 * sizeof(float)); cudaMalloc(&ix, 256 * 5 * sizeof(int));
      }
      cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &cs);
      if (cs != cudaStreamCaptureStatusNone) goto generic;   // tables are built outside capture
      vla_bicubic_taps_kernel<<<1, 256, 0, stream>>>(ih, 256, wy, iy);
      vla_bicubic_taps_kernel<<<1, 256, 0, stream>>>(iw, 256, wx, ix);
      tab_ih = ih; tab_iw = iw;
    }
    vla_bicubic_patchify_tab_kernel<16, 2, 2, 256, 256><<<grid_for(total, threads), threads, 0, stream>>>(
        src, out, wy, iy, wx, ix, mean, std_, views, C, ih, iw);
    return true;
  }
generic:
  vla_bicubic_patchify_kernel<16, 2, 2><<<grid_for(total, threads), threads, 0, stream>>>(
      src, out, mean, std_, views, C, ih, iw, oh, ow);
  return true;
}

void vla_frames_patchify(const float* src, __nv_bfloat16* out, int views, int C,
                         int H, int W, int tps, int ps, int merge, float mean,
                         float std_, cudaStream_t stream) {
  const int gh = H / ps, gw = W / ps;
  const int gi = gh / merge, gj = gw / merge;
  const long rows = (long)views * gi * gj * merge * merge;
  const long total = rows * (long)C * tps * ps * ps;
  const int threads = 256;
  if (ps == 16 && tps == 2 && merge == 2 && gi == 8 && gj == 8) {
    vla_patchify_kernel<16, 2, 2, 8, 8><<<grid_for(total, threads), threads, 0, stream>>>(
        src, out, mean, std_, views, C, H, W);
    return;
  }
  vla_patchify_generic_kernel<<<grid_for(total, threads), threads, 0, stream>>>(
      src, out, mean, std_, views, C, H, W, tps, ps, merge);
}

// DiT step input rows: x[0:head_rows] = head (state token + future tokens,
// constant per decision) and x[head_rows:] += pos (fp32 add, one bf16 rounding
// -- residual_add's math). One PDL-launched kernel instead of a graph memcpy
// node plus a residual add per flow step.
__global__ void vla_dit_step_rows_kernel(__nv_bfloat162* __restrict__ x,
                                         const __nv_bfloat162* __restrict__ head,
                                         const __nv_bfloat162* __restrict__ pos,
                                         int head_pairs, int act_pairs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // head and pos are constant inside the graph: read them before the wait
  __nv_bfloat162 hv = {}, pv = {};
  if (i < head_pairs) hv = head[i];
  else if (i < head_pairs + act_pairs) pv = pos[i - head_pairs];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (i < head_pairs) {
    x[i] = hv;
  } else if (i < head_pairs + act_pairs) {
    const __nv_bfloat162 r = x[i];
    x[i] = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(pv.x),
                                 __bfloat162float(r.y) + __bfloat162float(pv.y));
  }
}

void vla_state_normalize(const float* raw, const float* mean, const float* std_,
                         float eps, int n, __nv_bfloat16* out,
                         cudaStream_t stream) {
  const int threads = 128;
  vla_state_norm_kernel<<<grid_for(n, threads), threads, 0, stream>>>(
      raw, mean, std_, eps, n, out);
}

int vla_dit_step_rows(__nv_bfloat16* x, const __nv_bfloat16* head, const __nv_bfloat16* pos,
                      int head_rows, int act_rows, int dim, cudaStream_t stream) {
  if ((dim & 1) || head_rows < 0 || act_rows < 0) return static_cast<int>(cudaErrorInvalidValue);
  const int head_pairs = head_rows * dim / 2, act_pairs = act_rows * dim / 2;
  const int total = head_pairs + act_pairs, threads = 256;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((total + threads - 1) / threads);
  cfg.blockDim = dim3(threads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return static_cast<int>(cudaLaunchKernelEx(
      &cfg, vla_dit_step_rows_kernel, reinterpret_cast<__nv_bfloat162*>(x),
      reinterpret_cast<const __nv_bfloat162*>(head), reinterpret_cast<const __nv_bfloat162*>(pos),
      head_pairs, act_pairs));
}

// ── Delivered-chunk post-processing (the LeRobot postprocessor) in one launch ──
// clip to [-1, 1] -> gripper pre-snap (>= threshold) -> MIN_MAX inverse
// ((x + 1) / 2 * (max - min) + min) -> gripper binarize (> threshold).
// This is the whole action-side tail of the policy: one kernel instead of the
// eight ATen elementwise launches the per-op chain needed, with identical
// fp32 arithmetic per column (the gripper column carries both snaps).
__global__ void vla_action_postprocess_kernel(
    const __nv_bfloat16* __restrict__ actions, float* __restrict__ out,
    const float* __restrict__ amin, const float* __restrict__ amax,
    int rows, int cols, int gripper_dim, float gripper_threshold) {
  const long total = static_cast<long>(rows) * cols;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const int col = static_cast<int>(i % cols);
    const bool grip = (col == gripper_dim);
    float x = __bfloat162float(actions[i]);
    x = fminf(fmaxf(x, -1.0f), 1.0f);
    if (grip) x = (x >= gripper_threshold) ? 1.0f : 0.0f;
    x = (x + 1.0f) / 2.0f * (amax[col] - amin[col]) + amin[col];
    if (grip) x = 1.0f - 2.0f * ((x > gripper_threshold) ? 1.0f : 0.0f);
    out[i] = x;
  }
}

void vla_action_postprocess(const __nv_bfloat16* actions, float* out,
                            const float* amin, const float* amax, int rows,
                            int cols, int gripper_dim, float gripper_threshold,
                            cudaStream_t stream) {
  const long total = static_cast<long>(rows) * cols;
  vla_action_postprocess_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      actions, out, amin, amax, rows, cols, gripper_dim, gripper_threshold);
}

}  // namespace kernels
}  // namespace flash_rt
