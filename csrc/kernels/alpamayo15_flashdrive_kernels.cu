// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) backend kernels for Thor (sm_110). See
// alpamayo15_flashdrive_kernels.cuh for the cache contract and rounding rules.

#include "alpamayo15_flashdrive_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHeadDim = 128;

inline bool aligned8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 7u) == 0; }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }

struct alignas(8) Bf4 { __nv_bfloat16 v[4]; };

__device__ __forceinline__ long long cache_row(int r, const long long* row_map, long long row0,
                                               const long long* slot_ptr) {
  return (row_map ? row_map[r] : row0 + r) + (slot_ptr ? *slot_ptr : 0LL);
}

// BF16-table RoPE of one lane's 4 values (lane l holds d = 4l..4l+3; partner d +/- 64 = lane l ^ 16).
__device__ __forceinline__ Bf4 rope_bf16_rule(const Bf4& x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
                                              int lane) {
  Bf4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float other = __shfl_xor_sync(0xffffffffu, bf(x.v[i]), 16);
    const int d = 4 * lane + i;
    const float c = bf(cos[d]), s = bf(sin[d]);
    const float rot = lane < 16 ? -other : other;  // negation of a BF16 value is exact
    const float a = bf(to_bf(__fmul_rn(bf(x.v[i]), c)));
    const float b = bf(to_bf(__fmul_rn(rot, s)));
    out.v[i] = to_bf(__fadd_rn(a, b));
  }
  return out;
}

__device__ __forceinline__ Bf4 rope_fp32_rule(const Bf4& x, const float* cos, const float* sin, int lane) {
  Bf4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float other = __shfl_xor_sync(0xffffffffu, bf(x.v[i]), 16);
    const int d = 4 * lane + i;
    const float rot = lane < 16 ? -other : other;
    out.v[i] = to_bf(__fadd_rn(__fmul_rn(bf(x.v[i]), cos[d]), __fmul_rn(rot, sin[d])));
  }
  return out;
}

struct alignas(8) H4 { __half v[4]; };

// Four input values of a projection row: BF16, or FP16 (Marlin output) rounded to BF16 as .to(bfloat16) does.
template <bool kF16In>
__device__ __forceinline__ Bf4 load4(const void* base, size_t off) {
  if (kF16In) {
    const H4 h = *reinterpret_cast<const H4*>(static_cast<const __half*>(base) + off);
    Bf4 r;
#pragma unroll
    for (int i = 0; i < 4; ++i) r.v[i] = to_bf(__half2float(h.v[i]));
    return r;
  }
  return *reinterpret_cast<const Bf4*>(static_cast<const __nv_bfloat16*>(base) + off);
}

// One warp per (row, head) over q | k | v heads.
template <bool kFp32Rule, bool kF16In>
__global__ void fd_qk_prep_kernel(const void* __restrict__ q, const void* __restrict__ k,
                                  const void* __restrict__ v, int q_ld, int k_ld, int v_ld,
                                  const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                                  const void* __restrict__ cos, const void* __restrict__ sin,
                                  __nv_bfloat16* __restrict__ q_out,
                                  __nv_bfloat16* __restrict__ k_raw, long long kr_rs, long long kr_hs,
                                  __nv_bfloat16* __restrict__ k_rope, long long k_rs, long long k_hs,
                                  __nv_bfloat16* __restrict__ v_out, long long v_rs, long long v_hs,
                                  const long long* __restrict__ row_map, const long long* __restrict__ slot_ptr,
                                  int rows, int Hq, int Hk, float eps) {
  const int nh = Hq + 2 * Hk;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= rows * nh) return;
  const int row = gw / nh, head = gw % nh;
  const long long c = cache_row(row, row_map, 0, slot_ptr);
  if (head >= Hq + Hk) {
    const int h = head - Hq - Hk;
    const Bf4 x = load4<kF16In>(v, (size_t)row * v_ld + (size_t)h * kHeadDim + 4 * lane);
    *reinterpret_cast<Bf4*>(v_out + c * v_rs + h * v_hs + 4 * lane) = x;
    return;
  }
  const bool is_q = head < Hq;
  const int h = is_q ? head : head - Hq;
  const Bf4 x = is_q ? load4<kF16In>(q, (size_t)row * q_ld + (size_t)h * kHeadDim + 4 * lane)
                     : load4<kF16In>(k, (size_t)row * k_ld + (size_t)h * kHeadDim + 4 * lane);
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss += __fmul_rn(bf(x.v[i]), bf(x.v[i]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(__fadd_rn(ss / (float)kHeadDim, eps));
  const Bf4 w = *reinterpret_cast<const Bf4*>((is_q ? qw : kw) + 4 * lane);
  Bf4 n;
#pragma unroll
  for (int i = 0; i < 4; ++i) n.v[i] = to_bf(__fmul_rn(bf(w.v[i]), bf(to_bf(__fmul_rn(bf(x.v[i]), inv)))));
  const Bf4 r = kFp32Rule
      ? rope_fp32_rule(n, static_cast<const float*>(cos) + (size_t)row * kHeadDim,
                       static_cast<const float*>(sin) + (size_t)row * kHeadDim, lane)
      : rope_bf16_rule(n, static_cast<const __nv_bfloat16*>(cos) + (size_t)row * kHeadDim,
                       static_cast<const __nv_bfloat16*>(sin) + (size_t)row * kHeadDim, lane);
  if (is_q) {
    *reinterpret_cast<Bf4*>(q_out + ((size_t)row * Hq + h) * kHeadDim + 4 * lane) = r;
  } else {
    *reinterpret_cast<Bf4*>(k_raw + c * kr_rs + h * kr_hs + 4 * lane) = n;
    *reinterpret_cast<Bf4*>(k_rope + c * k_rs + h * k_hs + 4 * lane) = r;
  }
}

template <bool kFp32Rule>
__global__ void fd_rope_rows_kernel(const __nv_bfloat16* __restrict__ src, long long s_rs, long long s_hs,
                                    __nv_bfloat16* __restrict__ dst, long long d_rs, long long d_hs,
                                    const void* __restrict__ cos_v, const void* __restrict__ sin_v,
                                    const long long* __restrict__ row_map, long long row0,
                                    const long long* __restrict__ slot_ptr, const int* __restrict__ n_ptr,
                                    int rows, int Hk) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= rows * Hk) return;
  const int r = gw / Hk, h = gw % Hk;
  if (n_ptr && r >= *n_ptr) return;
  const long long c = cache_row(r, row_map, row0, slot_ptr);
  const Bf4 x = *reinterpret_cast<const Bf4*>(src + c * s_rs + h * s_hs + 4 * lane);
  Bf4 out;
  if (kFp32Rule) {
    out = rope_fp32_rule(x, static_cast<const float*>(cos_v) + c * kHeadDim,
                         static_cast<const float*>(sin_v) + c * kHeadDim, lane);
  } else {
    out = rope_bf16_rule(x, static_cast<const __nv_bfloat16*>(cos_v) + c * kHeadDim,
                         static_cast<const __nv_bfloat16*>(sin_v) + c * kHeadDim, lane);
  }
  *reinterpret_cast<Bf4*>(dst + c * d_rs + h * d_hs + 4 * lane) = out;
}

inline bool stride_ok(long long s) { return s >= 0 && (s % 4) == 0; }

__device__ __forceinline__ float block_sum256(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float t = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

// residual (BF16) += bf16(delta FP16) (BF16 add rounding), out = Qwen3 RMSNorm(residual) * w: the Alpamayo 1.5
// engine's rms_bf16 contract with the delta read straight from a Marlin FP16 output.
__global__ void fd_rms_f16delta_kernel(__nv_bfloat16* __restrict__ res, const __half* __restrict__ delta,
                                       const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                       int dim, float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  __nv_bfloat16* r = res + (size_t)row * dim;
  const __half* dl = delta + (size_t)row * dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 v = to_bf(__fadd_rn(bf(r[i]), bf(to_bf(__half2float(dl[i])))));
    r[i] = v;
    const float f = bf(v);
    ss += __fmul_rn(f, f);
  }
  const float inv = rsqrtf(__fadd_rn(block_sum256(ss, smem) / (float)dim, eps));
  __nv_bfloat16* o = out + (size_t)row * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 n1 = to_bf(__fmul_rn(bf(r[i]), inv));
    o[i] = to_bf(__fmul_rn(bf(w[i]), bf(n1)));
  }
}

// out = bf16(silu(bf16(g))) * bf16(u) with BF16 roundings of the SiLU and of the product (act_fn(gate) * up on
// the BF16 outputs of FlashDrive's W4A8 wrappers), g/u read as Marlin FP16 outputs.
__global__ void fd_silu_mul_f16in_kernel(const __half* __restrict__ g, const __half* __restrict__ u,
                                         __nv_bfloat16* __restrict__ out, long long n) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const float gb = bf(to_bf(__half2float(g[i])));
    const float ub = bf(to_bf(__half2float(u[i])));
    const float sl = bf(to_bf(gb / (1.0f + expf(-gb))));
    out[i] = to_bf(__fmul_rn(sl, ub));
  }
}

// ---------------------------------------------------------------------------
// W4A8 activation preparation (FlashDrive's ParoQuant linears), bit-exact with
//   x.to(float16) -> torch.ops.rotation.rotate(x, pairs, theta, channel_scales, 128)   (paroquant 0.1.15)
//   -> vLLM per_token_quant_int8 (Triton) -> a_scales * input_global_scale,
// for up to 3 rotation sets sharing one input row (q|k|v, gate|up).
// Rotation (paroquant RotateAccessHalf, KROT=8, group 128, CTA_M=4): x_half * scale_half (__hmul), then per
// layer r each of the 64 pairs (i, j) of a group: yi = fma(c, xi, s*xj), yj = fma(c, xj, -s*xi) in FP32 with
// (s, c) = __sincosf(theta), rounded to half per layer. Quantization (Triton PTX): absmax = max(max|x|, 1e-10),
// scale = div.full(absmax, 127), inv = div.full(127, absmax), q = cvt.rzi(add.rz.ftz(x*inv, copysign(0.5, x*inv))).
// ---------------------------------------------------------------------------
constexpr int kRotGroup = 128;
constexpr int kRotLayers = 8;
constexpr int kRotRows = 16;  // rows per rotation CTA (sin/cos and pair indices loaded once per 16 rows)

struct ActPrepSet {
  const __half* theta;      // [8, K/2] fp16
  const int16_t* pairs;     // [8, K] int16 (in-group pair indices, i low / j high per int32)
  const __half* scales;     // [K] fp16 channel scales
  const float* igs;         // input_global_scale (fp32 scalar)
  __half* xr;               // [rows, K] fp16 rotated activations (workspace)
  float* part;              // [rows, K/128] per-group row absmax (workspace)
  int8_t* xq;               // [rows, K] int8
  float* a_scale;           // [rows] fp32 = per-token scale * input_global_scale
};
struct ActPrepSets { ActPrepSet s[3]; };

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float round_half_away_triton(float y) {
  float h = copysignf(0.5f, y), t;
  asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(t) : "f"(y), "f"(h));
  return truncf(t);
}

// Pass 1: grid (row blocks, groups, sets), 64 threads = the 64 rotation pairs of a 128-channel group.
// Per element the sequence is paroquant's: half(x) * half(scale) (__hmul), then 8 layers of
// fma(c, xi, s*xj) / fma(c, xj, -s*xi) in FP32 with __sincosf, rounded to half after each layer.
// Shared tile is row-major [ROWS][128] so the random pair indices of a layer spread over the banks.
template <int ROWS>
__global__ void __launch_bounds__(64) fd_w4a8_rotate_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                             ActPrepSets sets, int rows, int K) {
  __shared__ __half xs[ROWS][kRotGroup];
  __shared__ float wmax[2][ROWS];
  const ActPrepSet& st = sets.s[blockIdx.z];
  const int row0 = blockIdx.x * ROWS, g = blockIdx.y, t = threadIdx.x;
  const int nrow = min(ROWS, rows - row0);
  const int off = kRotGroup * g + 2 * t;
  const __half sc_i = st.scales[off], sc_j = st.scales[off + 1];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (i < nrow) {
      const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)(row0 + i) * ld_x + off);
      xs[i][2 * t] = __hmul(__float2half_rn(__bfloat162float(v.x)), sc_i);
      xs[i][2 * t + 1] = __hmul(__float2half_rn(__bfloat162float(v.y)), sc_j);
    }
  }
  float th[kRotLayers];
  int ij[kRotLayers];
#pragma unroll
  for (int r = 0; r < kRotLayers; ++r) {
    th[r] = __half2float(st.theta[(size_t)r * K / 2 + g * kRotGroup / 2 + t]);
    ij[r] = *reinterpret_cast<const int*>(st.pairs + (size_t)r * K + g * kRotGroup + 2 * t);
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kRotLayers; ++r) {
    const int16_t i = ij[r] & 0xFFFF, j = ij[r] >> 16;
    float s_, c_;
    __sincosf(th[r], &s_, &c_);
#pragma unroll
    for (int m = 0; m < ROWS; ++m) {
      if (m < nrow) {
        const float xi = __half2float(xs[m][i]);
        const float xj = __half2float(xs[m][j]);
        xs[m][i] = __float2half_rn(fmaf(c_, xi, s_ * xj));
        xs[m][j] = __float2half_rn(fmaf(c_, xj, -s_ * xi));
      }
    }
    __syncthreads();
  }
  const int warp = t >> 5, lane = t & 31;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (i < nrow) {                                    // nrow is uniform across the block
      const __half a = xs[i][2 * t], b = xs[i][2 * t + 1];
      __half2 v; v.x = a; v.y = b;
      *reinterpret_cast<__half2*>(st.xr + (size_t)(row0 + i) * K + off) = v;
      float m = fmaxf(fabsf(__half2float(a)), fabsf(__half2float(b)));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (lane == 0) wmax[warp][i] = m;
    }
  }
  __syncthreads();
  if (t < nrow) st.part[(size_t)(row0 + t) * (K / kRotGroup) + g] = fmaxf(wmax[0][t], wmax[1][t]);
}

// Pass 2: grid (rows, sets): row absmax from the group partials, then vLLM's per-token INT8 quantization.
__global__ void __launch_bounds__(256) fd_w4a8_quant_kernel(ActPrepSets sets, int K) {
  __shared__ float red[256];
  const ActPrepSet& st = sets.s[blockIdx.y];
  const int row = blockIdx.x, tid = threadIdx.x, ng = K / kRotGroup;
  float m = 0.0f;
  for (int gi = tid; gi < ng; gi += 256) m = fmaxf(m, st.part[(size_t)row * ng + gi]);
  red[tid] = m;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
    __syncthreads();
  }
  const float absmax = fmaxf(red[0], 1e-10f);
  const float inv = div_full(127.0f, absmax);
  const __half2* xr = reinterpret_cast<const __half2*>(st.xr + (size_t)row * K);
  char2* q = reinterpret_cast<char2*>(st.xq + (size_t)row * K);
  for (int c = tid; c < K / 2; c += 256) {
    const float2 f = __half22float2(xr[c]);
    char2 o;
    o.x = static_cast<signed char>(static_cast<int>(round_half_away_triton(__fmul_rn(inv, f.x))));
    o.y = static_cast<signed char>(static_cast<int>(round_half_away_triton(__fmul_rn(inv, f.y))));
    q[c] = o;
  }
  if (tid == 0) st.a_scale[row] = __fmul_rn(div_full(absmax, 127.0f), *st.igs);
}

}  // namespace

int flashdrive_qk_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                            const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin, int fp32_rule,
                            int f16_in, void* q_out, void* k_raw, long long kr_rs, long long kr_hs,
                            void* k_rope, long long k_rs, long long k_hs,
                            void* v_out, long long v_rs, long long v_hs,
                            const long long* row_map, const long long* slot_ptr,
                            int rows, int Hq, int Hk, float eps, cudaStream_t stream) {
  if (!q || !k || !v || !q_norm_w || !k_norm_w || !cos || !sin || !q_out || !k_raw || !k_rope || !v_out) return 1;
  if (rows <= 0 || Hq <= 0 || Hk <= 0 || q_ld < Hq * kHeadDim || k_ld < Hk * kHeadDim || v_ld < Hk * kHeadDim)
    return 2;
  if (!aligned8(q) || !aligned8(k) || !aligned8(v) || !aligned8(q_norm_w) || !aligned8(k_norm_w) || !aligned8(cos) ||
      !aligned8(sin) || !aligned8(q_out) || !aligned8(k_raw) || !aligned8(k_rope) || !aligned8(v_out) ||
      (q_ld % 4) || (k_ld % 4) || (v_ld % 4) || !stride_ok(kr_rs) || !stride_ok(kr_hs) || !stride_ok(k_rs) ||
      !stride_ok(k_hs) || !stride_ok(v_rs) || !stride_ok(v_hs))
    return 5;
  const long long warps = (long long)rows * (Hq + 2 * Hk);
  const unsigned blocks = (unsigned)((warps + 7) / 8);
#define FD_QK_PREP_ARGS                                                                                        \
  q, k, v, q_ld, k_ld, v_ld, static_cast<const __nv_bfloat16*>(q_norm_w), static_cast<const __nv_bfloat16*>(k_norm_w),  \
      cos, sin, static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_raw), kr_rs, kr_hs,               \
      static_cast<__nv_bfloat16*>(k_rope), k_rs, k_hs, static_cast<__nv_bfloat16*>(v_out), v_rs, v_hs, row_map,       \
      slot_ptr, rows, Hq, Hk, eps
  if (fp32_rule && f16_in) {
    fd_qk_prep_kernel<true, true><<<blocks, 256, 0, stream>>>(FD_QK_PREP_ARGS);
  } else if (fp32_rule) {
    fd_qk_prep_kernel<true, false><<<blocks, 256, 0, stream>>>(FD_QK_PREP_ARGS);
  } else if (f16_in) {
    fd_qk_prep_kernel<false, true><<<blocks, 256, 0, stream>>>(FD_QK_PREP_ARGS);
  } else {
    fd_qk_prep_kernel<false, false><<<blocks, 256, 0, stream>>>(FD_QK_PREP_ARGS);
  }
#undef FD_QK_PREP_ARGS
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_rms_f16delta_bf16(void* residual, const void* delta, const void* weight, void* out, int rows, int dim,
                                 float eps, cudaStream_t stream) {
  if (!residual || !delta || !weight || !out) return 1;
  if (rows <= 0 || dim <= 0) return 2;
  fd_rms_f16delta_kernel<<<rows, 256, 0, stream>>>(static_cast<__nv_bfloat16*>(residual), static_cast<const __half*>(delta),
                                                   static_cast<const __nv_bfloat16*>(weight),
                                                   static_cast<__nv_bfloat16*>(out), dim, eps);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_silu_mul_f16in_bf16(const void* gate, const void* up, void* out, long long n, cudaStream_t stream) {
  if (!gate || !up || !out) return 1;
  if (n <= 0) return 2;
  const long long blocks = (n + 255) / 256;
  fd_silu_mul_f16in_kernel<<<(unsigned)(blocks > 4096 ? 4096 : blocks), 256, 0, stream>>>(
      static_cast<const __half*>(gate), static_cast<const __half*>(up), static_cast<__nv_bfloat16*>(out), n);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_rope_rows(const void* src, long long s_rs, long long s_hs,
                         void* dst, long long d_rs, long long d_hs,
                         const void* cos, const void* sin, int fp32_rule,
                         const long long* row_map, long long row0, const long long* slot_ptr,
                         const int* n_ptr, int rows, int Hk, cudaStream_t stream) {
  if (!src || !dst || !cos || !sin) return 1;
  if (rows <= 0 || Hk <= 0 || row0 < 0) return 2;
  if (!aligned8(src) || !aligned8(dst) || !aligned8(cos) || !aligned8(sin) || !stride_ok(s_rs) ||
      !stride_ok(s_hs) || !stride_ok(d_rs) || !stride_ok(d_hs))
    return 5;
  const long long warps = (long long)rows * Hk;
  const unsigned blocks = (unsigned)((warps + 7) / 8);
  if (fp32_rule) {
    fd_rope_rows_kernel<true><<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src), s_rs, s_hs, static_cast<__nv_bfloat16*>(dst), d_rs, d_hs, cos, sin,
        row_map, row0, slot_ptr, n_ptr, rows, Hk);
  } else {
    fd_rope_rows_kernel<false><<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src), s_rs, s_hs, static_cast<__nv_bfloat16*>(dst), d_rs, d_hs, cos, sin,
        row_map, row0, slot_ptr, n_ptr, rows, Hk);
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

int flashdrive_w4a8_act_prep(const void* x, int ld_x, int rows, int K, int n_sets,
                             const void* const* theta, const void* const* pairs, const void* const* scales,
                             const float* const* igs, void* const* xr, float* const* part, void* const* xq,
                             float* const* a_scale, cudaStream_t stream) {
  if (!x || rows <= 0 || K <= 0 || n_sets < 1 || n_sets > 3) return 2;
  if ((K % kRotGroup) != 0 || ld_x < K || (ld_x % 2) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0 || (ld_x % 2) != 0) return 5;
  ActPrepSets sets{};
  for (int i = 0; i < n_sets; ++i) {
    if (!theta[i] || !pairs[i] || !scales[i] || !igs[i] || !xr[i] || !part[i] || !xq[i] || !a_scale[i]) return 1;
    if ((reinterpret_cast<uintptr_t>(pairs[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(scales[i]) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(xr[i]) & 3u) != 0 || (reinterpret_cast<uintptr_t>(xq[i]) & 1u) != 0)
      return 5;
    sets.s[i] = ActPrepSet{static_cast<const __half*>(theta[i]), static_cast<const int16_t*>(pairs[i]),
                           static_cast<const __half*>(scales[i]), igs[i], static_cast<__half*>(xr[i]), part[i],
                           static_cast<int8_t*>(xq[i]), a_scale[i]};
  }
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  if (rows <= 2) {
    fd_w4a8_rotate_kernel<2><<<dim3((rows + 1) / 2, K / kRotGroup, n_sets), 64, 0, stream>>>(xb, ld_x, sets, rows, K);
  } else if (rows <= 8) {
    fd_w4a8_rotate_kernel<4><<<dim3((rows + 3) / 4, K / kRotGroup, n_sets), 64, 0, stream>>>(xb, ld_x, sets, rows, K);
  } else {
    fd_w4a8_rotate_kernel<kRotRows><<<dim3((rows + kRotRows - 1) / kRotRows, K / kRotGroup, n_sets), 64, 0, stream>>>(
        xb, ld_x, sets, rows, K);
  }
  fd_w4a8_quant_kernel<<<dim3(rows, n_sets), 256, 0, stream>>>(sets, K);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
