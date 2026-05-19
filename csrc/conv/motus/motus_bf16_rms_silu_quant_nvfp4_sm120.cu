// Phase 8.3 — bf16 NCDHW input → FP4 packed + UE4M3 linear SF NDHWC output,
// fused with RMS norm + SiLU. Templated on bf16_rms_silu_quant_fp8_v4.
//
// Eliminates the FP8 intermediate + dequant/requant detour in
// _vae_fp4_swap.py::_fused_step_fp4. Output layout matches what v19sf
// reads (linear SF, NDHWC).
//
// Built through CMake as part of flash_rt_kernels when Motus SM120 kernels
// are enabled.
//
// Constraints (v1):
//   - C must be multiple of 128 (so each thread-y owns whole SF blocks)
//   - C ≤ 1024 (kMaxBf162 cap, same as v4 FP8)
//   - C % 16 == 0 (SF block size)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kThreads  = kThreadsX * kThreadsY;
constexpr int kWBlock   = kThreadsX;
constexpr int kPadFp4   = 4;       // padding bytes per row in smem (alignment)
constexpr int kMaxBf162 = 64;      // covers c_per_y up to 128 (C=1024)

// FP4 e2m1 magnitude snap (verbatim from bf16_weight_to_nvfp4_swizzled.cu).
__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
  uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
  float a = fabsf(v);
  uint8_t mag;
  if      (a < 0.25f) mag = 0;
  else if (a < 0.75f) mag = 1;
  else if (a < 1.25f) mag = 2;
  else if (a < 1.75f) mag = 3;
  else if (a < 2.5f)  mag = 4;
  else if (a < 3.5f)  mag = 5;
  else if (a < 5.0f)  mag = 6;
  else                mag = 7;
  return sign | mag;
}

// UE4M3 round-toward-ceil (matches bf16_weight_to_nvfp4_swizzled.cu).
__device__ __forceinline__ uint8_t fp32_to_ue4m3_ceil(float v) {
  if (v <= 0.0f) return 0;
  if (v > 240.0f) return 0xFE;
  uint32_t bits = __float_as_uint(v);
  int float_exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t frac = bits & 0x7FFFFF;
  int ue_exp = float_exp + 7;
  if (ue_exp <= 0) {
    float scaled = v * 512.0f;
    int m = (int)ceilf(scaled);
    if (m > 7) return (1 << 3) | 0;
    if (m < 1) m = 1;
    return (uint8_t)m;
  }
  if (ue_exp >= 15) return 0xFE;
  int m = (int)(frac >> 20);
  if (frac & 0xFFFFF) m++;
  if (m >= 8) { m = 0; ue_exp++; }
  if (ue_exp >= 15) return 0xFE;
  return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

__device__ __forceinline__ float silu_f32(float x) {
  return x * (1.0f / (1.0f + __expf(-x)));
}

__global__ void v1_kernel(
    const __nv_bfloat16* __restrict__ x,         // [B,C,T,H,W]
    const __nv_bfloat16* __restrict__ gamma,     // [C]
    const float*         __restrict__ awq_inv_scale,  // [C] or nullptr
    uint8_t*             __restrict__ y_fp4,     // [B,T,H,W,C/2]
    uint8_t*             __restrict__ y_sf,      // [B,T,H,W,C/16]
    int B, int C, int T, int H, int W,
    int W_blocks_per_row, float eps)
{
  extern __shared__ __align__(16) char sm_buf[];
  // sm_out: [kWBlock × (C/2 + pad)] FP4 packed (1 byte per 2 channels)
  // sm_sf:  [kWBlock × (C/16 + pad)] UE4M3 SF bytes
  // sm_red: [kThreadsX × kThreadsY] float for sum_sq reduction
  const int sm_fp4_stride = (C / 2) + kPadFp4;
  const int sm_sf_stride  = (C / 16) + kPadFp4;
  uint8_t* sm_fp4 = reinterpret_cast<uint8_t*>(sm_buf);
  uint8_t* sm_sf  = sm_fp4 + (size_t)kWBlock * sm_fp4_stride;
  float*   sm_red = reinterpret_cast<float*>(sm_sf
                    + (size_t)kWBlock * sm_sf_stride);

  const int wb   = blockIdx.x % W_blocks_per_row;
  const int rest = blockIdx.x / W_blocks_per_row;
  const int hwt  = T * H;
  const int b    = rest / hwt;
  const int rh   = rest - b * hwt;
  const int t    = rh / H;
  const int h    = rh - t * H;
  if (b >= B) return;

  const int w_start = wb * kWBlock;
  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;
  const int my_w = w_start + tx;
  const bool active = (my_w < W);

  const int c_per_y    = (C + kThreadsY - 1) / kThreadsY;
  const int my_c_start = ty * c_per_y;
  const int my_c_end   = min(my_c_start + c_per_y, C);
  const int my_n_c     = my_c_end - my_c_start;
  const int my_n_pair  = (my_n_c + 1) >> 1;            // bf162 pairs

  const long long stride_C = (long long)T * H * W;
  const long long row_off  = (long long)t * H * W + (long long)h * W;
  const long long b_off    = (long long)b * (long long)C * stride_C;

  // ── Pass 1: read x → register cache + sum_sq ──
  __nv_bfloat162 xcache[kMaxBf162];
  float sum_sq = 0.f;
  if (active) {
    #pragma unroll 1
    for (int p = 0; p < my_n_pair; ++p) {
      int c0 = my_c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat16 v0 = x[b_off + (long long)c0 * stride_C + row_off + my_w];
      __nv_bfloat16 v1 = (c1 < my_c_end)
          ? x[b_off + (long long)c1 * stride_C + row_off + my_w]
          : __float2bfloat16(0.f);
      xcache[p] = __nv_bfloat162{v0, v1};
      float f0 = __bfloat162float(v0);
      float f1 = __bfloat162float(v1);
      sum_sq = fmaf(f0, f0, sum_sq);
      if (c1 < my_c_end) sum_sq = fmaf(f1, f1, sum_sq);
    }
  }
  sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.f;
  __syncthreads();
  float total_sum_sq = 0.f;
  #pragma unroll
  for (int yi = 0; yi < kThreadsY; ++yi) {
    total_sum_sq += sm_red[yi * kThreadsX + tx];
  }
  const float invC    = 1.0f / static_cast<float>(C);
  const float inv_rms = active ? rsqrtf(total_sum_sq * invC + eps) : 0.f;

  // ── Pass 2: RMS·γ·SiLU into a per-thread bf16 buffer (size c_per_y),
  //   then per-16-block FP4 quant + UE4M3 SF emit ──
  // Each thread-y owns c_per_y channels = c_per_y/16 SF blocks (we
  // require c_per_y % 16 == 0, i.e. C % 128 == 0).
  __nv_bfloat162 buf[kMaxBf162];
  if (active) {
    #pragma unroll 1
    for (int p = 0; p < my_n_pair; ++p) {
      int c0 = my_c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat162 vp = xcache[p];
      float xv0 = __bfloat162float(vp.x);
      float xv1 = __bfloat162float(vp.y);
      float gv0 = __bfloat162float(gamma[c0]);
      float n0  = xv0 * inv_rms * gv0;
      float s0  = silu_f32(__bfloat162float(__float2bfloat16(n0)));
      if (awq_inv_scale != nullptr) s0 *= awq_inv_scale[c0];   // AWQ per-Ci
      __nv_bfloat16 b0 = __float2bfloat16(s0);
      __nv_bfloat16 b1 = __float2bfloat16(0.f);
      if (c1 < my_c_end) {
        float gv1 = __bfloat162float(gamma[c1]);
        float n1 = xv1 * inv_rms * gv1;
        float s1 = silu_f32(__bfloat162float(__float2bfloat16(n1)));
        if (awq_inv_scale != nullptr) s1 *= awq_inv_scale[c1]; // AWQ per-Ci
        b1 = __float2bfloat16(s1);
      }
      buf[p] = __nv_bfloat162{b0, b1};
    }

    // Per-16-block quantize: walk 16 channels at a time within my range.
    // Each block consumes 8 bf162 pairs.
    const int blocks_per_thread = my_n_c / 16;          // assumed clean
    #pragma unroll 1
    for (int blk = 0; blk < blocks_per_thread; ++blk) {
      int p_base = blk * 8;
      // Find max_abs over 16 elems
      float mx = 0.f;
      #pragma unroll
      for (int p = 0; p < 8; ++p) {
        __nv_bfloat162 vp = buf[p_base + p];
        float a0 = fabsf(__bfloat162float(vp.x));
        float a1 = fabsf(__bfloat162float(vp.y));
        mx = fmaxf(mx, fmaxf(a0, a1));
      }
      float sf_f = mx / 6.0f;
      uint8_t sf_byte = fp32_to_ue4m3_ceil(sf_f);
      float sf_dec = ue4m3_to_fp32(sf_byte);
      float inv_sf = (sf_dec > 0.f) ? (1.0f / sf_dec) : 0.f;
      // Quant 16 elems → 8 packed FP4 bytes
      int sf_idx = (my_c_start / 16) + blk;
      sm_sf[tx * sm_sf_stride + sf_idx] = sf_byte;
      #pragma unroll
      for (int p = 0; p < 8; ++p) {
        __nv_bfloat162 vp = buf[p_base + p];
        float v0 = __bfloat162float(vp.x) * inv_sf;
        float v1 = __bfloat162float(vp.y) * inv_sf;
        uint8_t lo = fp32_to_e2m1(v0);
        uint8_t hi = fp32_to_e2m1(v1);
        int byte_idx = (my_c_start / 2) + (blk * 8) + p;
        sm_fp4[tx * sm_fp4_stride + byte_idx] = (hi << 4) | (lo & 0xF);
      }
    }
  }
  __syncthreads();

  // ── Pass 3: coalesced uint32-vec global writes ──
  // FP4 output: kWBlock rows × (C/2) bytes per row → C/8 u32s per row.
  // SF output:  kWBlock rows × (C/16) bytes per row → C/64 u32s per row (if C ≥ 64).
  const long long y_base_fp4 = ((long long)b * T * H * W
                              + (long long)t * H * W
                              + (long long)h * W
                              + w_start) * (long long)(C / 2);
  const long long y_base_sf  = ((long long)b * T * H * W
                              + (long long)t * H * W
                              + (long long)h * W
                              + w_start) * (long long)(C / 16);

  // FP4 write
  const int fp4_words_per_row = C / 8;
  const int fp4_total_words   = kWBlock * fp4_words_per_row;
  const int tid = threadIdx.x;
  #pragma unroll 1
  for (int idx = tid; idx < fp4_total_words; idx += kThreads) {
    int w_off = idx / fp4_words_per_row;
    int wd    = idx - w_off * fp4_words_per_row;
    if (w_start + w_off < W) {
      uint32_t pack = *reinterpret_cast<const uint32_t*>(
          &sm_fp4[w_off * sm_fp4_stride + (wd << 2)]);
      *reinterpret_cast<uint32_t*>(
          &y_fp4[y_base_fp4 + (long long)w_off * (long long)(C / 2)
                            + (long long)(wd << 2)]) = pack;
    }
  }
  // SF write — for C >= 64, C/16 >= 4 = at least 1 u32 per row
  // For C < 64 we'd need scalar; gate at API.
  if ((C / 16) >= 4) {
    const int sf_words_per_row = (C / 16) / 4;
    const int sf_total_words   = kWBlock * sf_words_per_row;
    #pragma unroll 1
    for (int idx = tid; idx < sf_total_words; idx += kThreads) {
      int w_off = idx / sf_words_per_row;
      int wd    = idx - w_off * sf_words_per_row;
      if (w_start + w_off < W) {
        uint32_t pack = *reinterpret_cast<const uint32_t*>(
            &sm_sf[w_off * sm_sf_stride + (wd << 2)]);
        *reinterpret_cast<uint32_t*>(
            &y_sf[y_base_sf + (long long)w_off * (long long)(C / 16)
                            + (long long)(wd << 2)]) = pack;
      }
    }
  }
}

}  // namespace

extern "C" int motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1(
    const void*  x_bf16,
    const void*  gamma_bf16,
    const void*  awq_inv_scale_fp32,    // [C] float, or NULL
    void*        y_fp4,         // [B,T,H,W,C/2] uint8
    void*        y_sf,          // [B,T,H,W,C/16] uint8 UE4M3 linear
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if ((C & 127) != 0)  return -2;
  if (C > 1024)        return -3;
  if ((C / 16) < 4)    return -4;
  const int W_blocks_per_row = (W + kWBlock - 1) / kWBlock;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas > (long long)INT32_MAX) return -5;

  const size_t sm_fp4_bytes = (size_t)kWBlock * ((C / 2) + kPadFp4);
  const size_t sm_sf_bytes  = (size_t)kWBlock * ((C / 16) + kPadFp4);
  const size_t sm_red_bytes = (size_t)kThreadsX * kThreadsY * 4;
  const size_t smem_bytes = sm_fp4_bytes + sm_sf_bytes + sm_red_bytes;

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  v1_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<const float*>(awq_inv_scale_fp32),
      reinterpret_cast<uint8_t*>(y_fp4),
      reinterpret_cast<uint8_t*>(y_sf),
      B, C, T, H, W, W_blocks_per_row, eps);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[v4q_v1] launch err: %s\n", cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

}  // namespace quantize
}  // namespace flash_rt
