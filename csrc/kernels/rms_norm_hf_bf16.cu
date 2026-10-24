// ================================================================
// FlashRT — HF-parity BF16 RMSNorm, q/k RMSNorm + RoPE, SiLU-gate.
//
// What "parity" takes, and where each piece comes from:
//
//   * Rounding points. Qwen3RMSNorm is
//       w * bf16(x32 * rsqrt(mean(x32^2) + eps))
//     -- two roundings, not one -- and HF's rotate-half RoPE rounds each
//     product to bf16 before the add. The decoder residual is a bf16 add,
//     and the norm that follows reduces the *rounded* residual.
//
//   * Reduction order. mean(x32^2) is a torch reduction; a different
//     summation tree moves the fp32 sum by an ulp, which flips a bf16
//     rounding somewhere in the row often enough to show in end-to-end
//     logits. torch maps one row to block_width threads (vectorized by 4
//     when dim > 128), each thread keeps four accumulators over a strided
//     walk, combines them in order, halves across the block through shared
//     memory, then shuffles down within the warp. block_width is
//     512 / min(last_pow2(rows), 16) for a 2560-wide row: 32 once there are
//     16 rows or more, wider below that. Here one warp owns a row and each
//     lane plays block_width / 32 of torch's threads, so the halving steps
//     stay inside the lane.
//
//     PyTorch 2.10 changed two details, so `torch_reduce` names the tree:
//       0  PyTorch <= 2.9: shuffle offsets 1, 2, ..., 16; a 128-wide row
//          (a q/k head) is reduced unvectorized.
//       1  PyTorch >= 2.10: shuffle offsets 16, 8, ..., 1; a 128-wide row is
//          vectorized by 4 like the wider ones.
//
//   * Arithmetic. Every product and sum is an explicit _rn intrinsic so
//     the compiler cannot contract them into FMAs the reference does not
//     have; the mean multiplies by torch's float factor rather than
//     dividing; rsqrtf is the same rsqrt.approx instruction torch uses.
//     SiLU divides with __fdiv_rn because this module builds with
//     --prec-div=false.
// ================================================================

#include "rms_norm_hf_bf16.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace flash_rt::kernels {

namespace {

constexpr int kRowWarps = 4;      // rows per block, one warp each
constexpr int kHeadWarps = 8;     // (row, head) vectors per block
constexpr int kSiluThreads = 256;
constexpr int kTorchMaxThreads = 512;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ float bf_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

union Bf16x4 {
  uint2 u;
  __nv_bfloat16 h[4];
};

union Bf16x8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, bool pdl,
            cudaStream_t stream, const char* name, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err =
      cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(name) + ": launch failed: " +
                             cudaGetErrorString(err));
  }
}

int last_pow2(long long n) {
  int p = 1;
  while (static_cast<long long>(p) * 2 <= n) p *= 2;
  return p;
}

// torch's ReduceConfig::set_block_dimension for a float reduction over the
// innermost dim: returns block_width, and block_height through *bh.
int torch_block_width(long long dim0, long long num_outputs, int* bh) {
  const int dim0_pow2 = dim0 < kTorchMaxThreads ? last_pow2(dim0) : kTorchMaxThreads;
  const int dim1_pow2 =
      num_outputs < kTorchMaxThreads ? last_pow2(num_outputs) : kTorchMaxThreads;
  int bw = dim0_pow2 < 32 ? dim0_pow2 : 32;
  int h = dim1_pow2 < kTorchMaxThreads / bw ? dim1_pow2 : kTorchMaxThreads / bw;
  bw = dim0_pow2 < kTorchMaxThreads / h ? dim0_pow2 : kTorchMaxThreads / h;
  *bh = h;
  return bw;
}

// torch's intra-warp shuffle-down sum, result broadcast to every lane.
template <bool kDescending>
__device__ __forceinline__ float torch_warp_sum(float v) {
  if constexpr (kDescending) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
  } else {
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
  }
  return __shfl_sync(0xffffffffu, v, 0);
}

// ── Row RMSNorm (optionally fused with the residual add) ──
//
// VT = torch block_width / 32. Virtual thread t = lane + 32 * k walks the
// row's 4-element vectors t, t + 32 * VT, ...; vector v is always touched by
// lane v % 32, so the second pass re-reads what the same thread wrote.
template <int VT, bool kResidual, bool kDescending>
__global__ void rms_norm_hf_kernel(__nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ delta,
                                   const __nv_bfloat16* __restrict__ weight,
                                   __nv_bfloat16* __restrict__ out,
                                   int rows, int dim, float factor, float eps) {
  constexpr int kWidth = 32 * VT;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowWarps + (threadIdx.x >> 5);
  pdl_wait();
  if (row >= rows) return;
  const int nvec = dim >> 2;
  __nv_bfloat16* xr = x + static_cast<long long>(row) * dim;
  const __nv_bfloat16* dr = kResidual ? delta + static_cast<long long>(row) * dim : nullptr;

  float part[VT];
#pragma unroll
  for (int k = 0; k < VT; ++k) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int v = lane + 32 * k; v < nvec; v += kWidth) {
      Bf16x4 xv;
      xv.u = *reinterpret_cast<const uint2*>(xr + 4 * v);
      float f[4];
      if constexpr (kResidual) {
        Bf16x4 dv;
        dv.u = *reinterpret_cast<const uint2*>(dr + 4 * v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          xv.h[i] = __float2bfloat16_rn(
              __fadd_rn(__bfloat162float(xv.h[i]), __bfloat162float(dv.h[i])));
        }
        *reinterpret_cast<uint2*>(xr + 4 * v) = xv.u;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) f[i] = __bfloat162float(xv.h[i]);
      a0 = __fadd_rn(a0, __fmul_rn(f[0], f[0]));
      a1 = __fadd_rn(a1, __fmul_rn(f[1], f[1]));
      a2 = __fadd_rn(a2, __fmul_rn(f[2], f[2]));
      a3 = __fadd_rn(a3, __fmul_rn(f[3], f[3]));
    }
    part[k] = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
  }
#pragma unroll
  for (int off = VT / 2; off >= 1; off >>= 1) {
#pragma unroll
    for (int k = 0; k < off; ++k) part[k] = __fadd_rn(part[k], part[k + off]);
  }
  const float sum = torch_warp_sum<kDescending>(part[0]);
  const float inv = rsqrtf(__fadd_rn(__fmul_rn(sum, factor), eps));
  pdl_launch_dependents();

  __nv_bfloat16* orow = out + static_cast<long long>(row) * dim;
  for (int v = lane; v < nvec; v += 32) {
    Bf16x4 xv, wv, ov;
    xv.u = *reinterpret_cast<const uint2*>(xr + 4 * v);
    wv.u = *reinterpret_cast<const uint2*>(weight + 4 * v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float n = bf_round(__fmul_rn(__bfloat162float(xv.h[i]), inv));
      ov.h[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(wv.h[i]), n));
    }
    *reinterpret_cast<uint2*>(orow + 4 * v) = ov.u;
  }
}

// Register-resident variant for dim = 128 * NV: a lane's NV vectors and the
// weight's are loaded up front, so a row costs one round of memory latency
// instead of NV dependent ones. Same arithmetic, in the same order, as
// rms_norm_hf_kernel.
template <int NV, int VT, bool kResidual, bool kDescending>
__global__ void __launch_bounds__(32 * kRowWarps)
rms_norm_hf_reg_kernel(__nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ delta,
                       const __nv_bfloat16* __restrict__ weight,
                       __nv_bfloat16* __restrict__ out,
                       int rows, float factor, float eps) {
  constexpr int kDim = 128 * NV;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowWarps + (threadIdx.x >> 5);
  pdl_wait();
  if (row >= rows) return;
  __nv_bfloat16* xr = x + static_cast<long long>(row) * kDim;
  Bf16x4 xv[NV], wv[NV];
#pragma unroll
  for (int m = 0; m < NV; ++m) {
    xv[m].u = *reinterpret_cast<const uint2*>(xr + 4 * (lane + 32 * m));
    wv[m].u = *reinterpret_cast<const uint2*>(weight + 4 * (lane + 32 * m));
  }
  if constexpr (kResidual) {
    const __nv_bfloat16* dr = delta + static_cast<long long>(row) * kDim;
    Bf16x4 dv[NV];
#pragma unroll
    for (int m = 0; m < NV; ++m) {
      dv[m].u = *reinterpret_cast<const uint2*>(dr + 4 * (lane + 32 * m));
    }
#pragma unroll
    for (int m = 0; m < NV; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xv[m].h[i] = __float2bfloat16_rn(__fadd_rn(
            __bfloat162float(xv[m].h[i]), __bfloat162float(dv[m].h[i])));
      }
      *reinterpret_cast<uint2*>(xr + 4 * (lane + 32 * m)) = xv[m].u;
    }
  }
  // Virtual thread k of the lane owns vectors m = k, k + VT, ...
  float part[VT];
#pragma unroll
  for (int k = 0; k < VT; ++k) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = k; m < NV; m += VT) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float f = __bfloat162float(xv[m].h[i]);
        a[i] = __fadd_rn(a[i], __fmul_rn(f, f));
      }
    }
    part[k] = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
  }
#pragma unroll
  for (int off = VT / 2; off >= 1; off >>= 1) {
#pragma unroll
    for (int k = 0; k < off; ++k) part[k] = __fadd_rn(part[k], part[k + off]);
  }
  const float sum = torch_warp_sum<kDescending>(part[0]);
  const float inv = rsqrtf(__fadd_rn(__fmul_rn(sum, factor), eps));
  pdl_launch_dependents();
  __nv_bfloat16* orow = out + static_cast<long long>(row) * kDim;
#pragma unroll
  for (int m = 0; m < NV; ++m) {
    Bf16x4 ov;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float n = bf_round(__fmul_rn(__bfloat162float(xv[m].h[i]), inv));
      ov.h[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(wv[m].h[i]), n));
    }
    *reinterpret_cast<uint2*>(orow + 4 * (lane + 32 * m)) = ov.u;
  }
}

template <int NV, bool kResidual, bool kDescending>
void launch_rms_norm_hf_reg(int bw, dim3 grid, dim3 block, bool pdl,
                            cudaStream_t stream, const char* name,
                            __nv_bfloat16* x, const __nv_bfloat16* delta,
                            const __nv_bfloat16* weight, __nv_bfloat16* out,
                            int rows, float factor, float eps) {
  switch (bw) {
    case 32:
      launch(rms_norm_hf_reg_kernel<NV, 1, kResidual, kDescending>, grid, block,
             pdl, stream, name, x, delta, weight, out, rows, factor, eps);
      return;
    case 64:
      launch(rms_norm_hf_reg_kernel<NV, 2, kResidual, kDescending>, grid, block,
             pdl, stream, name, x, delta, weight, out, rows, factor, eps);
      return;
    case 128:
      launch(rms_norm_hf_reg_kernel<NV, 4, kResidual, kDescending>, grid, block,
             pdl, stream, name, x, delta, weight, out, rows, factor, eps);
      return;
    case 256:
      launch(rms_norm_hf_reg_kernel<NV, 8, kResidual, kDescending>, grid, block,
             pdl, stream, name, x, delta, weight, out, rows, factor, eps);
      return;
    case 512:
      launch(rms_norm_hf_reg_kernel<NV, 16, kResidual, kDescending>, grid, block,
             pdl, stream, name, x, delta, weight, out, rows, factor, eps);
      return;
    default:
      throw std::runtime_error(std::string(name) + ": unexpected block width " +
                               std::to_string(bw));
  }
}

template <bool kResidual, bool kDescending>
void launch_rms_norm_hf(__nv_bfloat16* x, const __nv_bfloat16* delta,
                        const __nv_bfloat16* weight, __nv_bfloat16* out,
                        int rows, int dim, float eps, bool pdl,
                        cudaStream_t stream, const char* name) {
  if (rows <= 0) return;
  if (!x || !weight || !out || (kResidual && !delta)) {
    throw std::runtime_error(std::string(name) + ": null pointer argument");
  }
  if (dim <= 128 || dim % 4 != 0) {
    throw std::runtime_error(std::string(name) +
                             ": dim must be a multiple of 4 and > 128");
  }
  int bh = 0;
  const int bw = torch_block_width(dim / 4, rows, &bh);
  // torch splits a row across warps once each thread would sum at least
  // min(16 * block_height, 256) values; that tree is not reproduced here.
  const long long per_thread = (dim + bw - 1) / bw;
  const long long split_at = 16LL * bh < 256 ? 16LL * bh : 256;
  if (per_thread >= split_at) {
    throw std::runtime_error(std::string(name) + ": dim " + std::to_string(dim) +
                             " takes torch's cross-warp reduction, unsupported");
  }
  const float factor = static_cast<float>(rows) /
                       static_cast<float>(static_cast<long long>(rows) * dim);
  const dim3 grid((rows + kRowWarps - 1) / kRowWarps);
  const dim3 block(32 * kRowWarps);
  // Hidden sizes of the Qwen3 dense family take the register-resident kernel.
  if (dim % 128 == 0) {
    switch (dim / 128) {
      case 8:
        launch_rms_norm_hf_reg<8, kResidual, kDescending>(
            bw, grid, block, pdl, stream, name, x, delta, weight, out, rows, factor, eps);
        return;
      case 16:
        launch_rms_norm_hf_reg<16, kResidual, kDescending>(
            bw, grid, block, pdl, stream, name, x, delta, weight, out, rows, factor, eps);
        return;
      case 20:
        launch_rms_norm_hf_reg<20, kResidual, kDescending>(
            bw, grid, block, pdl, stream, name, x, delta, weight, out, rows, factor, eps);
        return;
      case 32:
        launch_rms_norm_hf_reg<32, kResidual, kDescending>(
            bw, grid, block, pdl, stream, name, x, delta, weight, out, rows, factor, eps);
        return;
      case 40:
        launch_rms_norm_hf_reg<40, kResidual, kDescending>(
            bw, grid, block, pdl, stream, name, x, delta, weight, out, rows, factor, eps);
        return;
      default:
        break;
    }
  }
  switch (bw) {
    case 32:
      launch(rms_norm_hf_kernel<1, kResidual, kDescending>, grid, block, pdl, stream, name,
             x, delta, weight, out, rows, dim, factor, eps);
      break;
    case 64:
      launch(rms_norm_hf_kernel<2, kResidual, kDescending>, grid, block, pdl, stream, name,
             x, delta, weight, out, rows, dim, factor, eps);
      break;
    case 128:
      launch(rms_norm_hf_kernel<4, kResidual, kDescending>, grid, block, pdl, stream, name,
             x, delta, weight, out, rows, dim, factor, eps);
      break;
    case 256:
      launch(rms_norm_hf_kernel<8, kResidual, kDescending>, grid, block, pdl, stream, name,
             x, delta, weight, out, rows, dim, factor, eps);
      break;
    case 512:
      launch(rms_norm_hf_kernel<16, kResidual, kDescending>, grid, block, pdl, stream, name,
             x, delta, weight, out, rows, dim, factor, eps);
      break;
    default:
      throw std::runtime_error(std::string(name) + ": unexpected block width " +
                               std::to_string(bw));
  }
}

// ── q/k head RMSNorm + rotate-half RoPE, head_dim 128 ──
//
// One warp per (row, head).
//
// Tree 0 (unvectorized): lane l holds elements l, l+32, l+64, l+96, so every
// RoPE partner (j, j+64) is in the same lane. torch reduces a head with 32
// threads when it has 16 or more heads to reduce
// (((x_l + x_{l+32}) + x_{l+64}) + x_{l+96}, then the shuffle tree) and with
// a wider block below that, which works out to
// (x_l + x_{l+64}) + (x_{l+32} + x_{l+96}); `wide_*` selects the latter.
//
// Tree 1 (vectorized by 4): lane l holds elements 4l .. 4l+3, always 32
// threads per head; the RoPE partner of lane l is lane l ^ 16.
template <bool kVectorized>
__global__ void qk_rms_norm_rope_hf_kernel(
    __nv_bfloat16* __restrict__ q, long long q_stride, int nq,
    __nv_bfloat16* __restrict__ k, long long k_stride, int nk,
    const __nv_bfloat16* __restrict__ wq, const __nv_bfloat16* __restrict__ wk,
    const __nv_bfloat16* __restrict__ cos_half,
    const __nv_bfloat16* __restrict__ sin_half,
    int seq, float factor_q, float factor_k, bool wide_q, bool wide_k,
    float eps) {
  constexpr int HD = 128;
  constexpr int HALF = HD / 2;
  const int lane = threadIdx.x & 31;
  const long long vec =
      static_cast<long long>(blockIdx.x) * kHeadWarps + (threadIdx.x >> 5);
  const long long n_q = static_cast<long long>(seq) * nq;
  const long long total = n_q + static_cast<long long>(seq) * nk;
  if (vec >= total) return;
  const bool is_q = vec < n_q;
  const long long v = is_q ? vec : vec - n_q;
  const int heads = is_q ? nq : nk;
  const int s = static_cast<int>(v / heads);
  const int h = static_cast<int>(v % heads);
  const __nv_bfloat16* w = is_q ? wq : wk;

  __nv_bfloat16* x = (is_q ? q + s * q_stride : k + s * k_stride) +
                     static_cast<long long>(h) * HD;
  const __nv_bfloat16* cs = cos_half + static_cast<long long>(s) * HALF;
  const __nv_bfloat16* sl = sin_half + static_cast<long long>(s) * HALF;
  const float factor = is_q ? factor_q : factor_k;

  pdl_wait();
  if constexpr (kVectorized) {
    Bf16x4 wv, cv, sv, xv;
    xv.u = *reinterpret_cast<const uint2*>(x + 4 * lane);
    wv.u = *reinterpret_cast<const uint2*>(w + 4 * lane);
    cv.u = *reinterpret_cast<const uint2*>(cs + 4 * (lane & 15));
    sv.u = *reinterpret_cast<const uint2*>(sl + 4 * (lane & 15));
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __bfloat162float(xv.h[i]);
    const float part = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(f[0], f[0]),
                                                     __fmul_rn(f[1], f[1])),
                                           __fmul_rn(f[2], f[2])),
                                 __fmul_rn(f[3], f[3]));
    const float sum = torch_warp_sum<true>(part);
    const float inv = rsqrtf(__fadd_rn(__fmul_rn(sum, factor), eps));
    pdl_launch_dependents();
    const bool lo = lane < 16;
    Bf16x4 ov;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float y = bf_round(__fmul_rn(__bfloat162float(wv.h[i]),
                                         bf_round(__fmul_rn(f[i], inv))));
      const float yp = __shfl_xor_sync(0xffffffffu, y, 16);
      const float c = __bfloat162float(cv.h[i]), sn = __bfloat162float(sv.h[i]);
      ov.h[i] = __float2bfloat16_rn(__fadd_rn(
          bf_round(__fmul_rn(y, c)), bf_round(__fmul_rn(lo ? -yp : yp, sn))));
    }
    *reinterpret_cast<uint2*>(x + 4 * lane) = ov.u;
  } else {
    float f[4], wf[4], c[2], sn[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[i] = __bfloat162float(x[lane + 32 * i]);
      wf[i] = __bfloat162float(w[lane + 32 * i]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      c[i] = __bfloat162float(cs[lane + 32 * i]);
      sn[i] = __bfloat162float(sl[lane + 32 * i]);
    }
    const float sq0 = __fmul_rn(f[0], f[0]), sq1 = __fmul_rn(f[1], f[1]);
    const float sq2 = __fmul_rn(f[2], f[2]), sq3 = __fmul_rn(f[3], f[3]);
    const bool wide = is_q ? wide_q : wide_k;
    const float part = wide ? __fadd_rn(__fadd_rn(sq0, sq2), __fadd_rn(sq1, sq3))
                            : __fadd_rn(__fadd_rn(__fadd_rn(sq0, sq1), sq2), sq3);
    const float sum = torch_warp_sum<false>(part);
    const float inv = rsqrtf(__fadd_rn(__fmul_rn(sum, factor), eps));
    pdl_launch_dependents();

    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i] = bf_round(__fmul_rn(wf[i], bf_round(__fmul_rn(f[i], inv))));
    }
    // j = lane + 32 i; the partner of j < 64 is j + 64 (lane-local index i + 2).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float lo = y[i], hi = y[i + 2];
      x[lane + 32 * i] = __float2bfloat16_rn(
          __fadd_rn(bf_round(__fmul_rn(lo, c[i])), bf_round(__fmul_rn(-hi, sn[i]))));
      x[lane + 32 * i + HALF] = __float2bfloat16_rn(
          __fadd_rn(bf_round(__fmul_rn(hi, c[i])), bf_round(__fmul_rn(lo, sn[i]))));
    }
  }
}

bool torch_wide_head_reduce(long long num_outputs) {
  int bh = 0;
  return torch_block_width(128, num_outputs, &bh) > 32;
}

// ── SiLU gate ──
__device__ __forceinline__ float silu_hf(float g) {
  return __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
}

__global__ void silu_mul_hf_kernel(const __nv_bfloat16* gate,
                                   const __nv_bfloat16* __restrict__ up,
                                   __nv_bfloat16* out, long long n8) {
  const long long i = static_cast<long long>(blockIdx.x) * kSiluThreads + threadIdx.x;
  pdl_wait();
  if (i >= n8) return;
  Bf16x8 g, u, o;
  g.u = reinterpret_cast<const uint4*>(gate)[i];
  u.u = reinterpret_cast<const uint4*>(up)[i];
  pdl_launch_dependents();
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = bf_round(silu_hf(__bfloat162float(g.h[j])));
    o.h[j] = __float2bfloat16_rn(__fmul_rn(s, __bfloat162float(u.h[j])));
  }
  reinterpret_cast<uint4*>(out)[i] = o.u;
}

}  // namespace

namespace {
void check_torch_reduce(int torch_reduce, const char* name) {
  if (torch_reduce != 0 && torch_reduce != 1) {
    throw std::runtime_error(std::string(name) + ": torch_reduce must be 0 or 1");
  }
}
}  // namespace

void rms_norm_hf_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* out, int rows, int dim, float eps,
                      int torch_reduce, bool pdl, cudaStream_t stream) {
  const char* name = "rms_norm_hf_bf16";
  check_torch_reduce(torch_reduce, name);
  auto* xm = const_cast<__nv_bfloat16*>(x);
  if (torch_reduce == 1) {
    launch_rms_norm_hf<false, true>(xm, nullptr, weight, out, rows, dim, eps,
                                    pdl, stream, name);
  } else {
    launch_rms_norm_hf<false, false>(xm, nullptr, weight, out, rows, dim, eps,
                                     pdl, stream, name);
  }
}

void residual_add_rms_norm_hf_bf16(__nv_bfloat16* residual,
                                   const __nv_bfloat16* delta,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* out, int rows, int dim,
                                   float eps, int torch_reduce, bool pdl,
                                   cudaStream_t stream) {
  const char* name = "residual_add_rms_norm_hf_bf16";
  check_torch_reduce(torch_reduce, name);
  if (torch_reduce == 1) {
    launch_rms_norm_hf<true, true>(residual, delta, weight, out, rows, dim, eps,
                                   pdl, stream, name);
  } else {
    launch_rms_norm_hf<true, false>(residual, delta, weight, out, rows, dim,
                                    eps, pdl, stream, name);
  }
}

void qk_rms_norm_rope_hf_bf16(__nv_bfloat16* q, long long q_row_stride,
                              int q_heads,
                              __nv_bfloat16* k, long long k_row_stride,
                              int k_heads,
                              const __nv_bfloat16* q_weight,
                              const __nv_bfloat16* k_weight,
                              const __nv_bfloat16* cos_half,
                              const __nv_bfloat16* sin_half,
                              int seq, int head_dim, float eps,
                              int torch_reduce, bool pdl, cudaStream_t stream) {
  const char* name = "qk_rms_norm_rope_hf_bf16";
  check_torch_reduce(torch_reduce, name);
  if (seq <= 0) return;
  if (head_dim != 128) {
    throw std::runtime_error(std::string(name) + ": head_dim must be 128");
  }
  if (!q || !k || !q_weight || !k_weight || !cos_half || !sin_half) {
    throw std::runtime_error(std::string(name) + ": null pointer argument");
  }
  if (q_heads <= 0 || k_heads <= 0 ||
      q_row_stride < static_cast<long long>(q_heads) * head_dim ||
      k_row_stride < static_cast<long long>(k_heads) * head_dim) {
    throw std::runtime_error(std::string(name) + ": bad head count or row stride");
  }
  const long long n_q = static_cast<long long>(seq) * q_heads;
  const long long n_k = static_cast<long long>(seq) * k_heads;
  const float factor_q = static_cast<float>(n_q) / static_cast<float>(n_q * head_dim);
  const float factor_k = static_cast<float>(n_k) / static_cast<float>(n_k * head_dim);
  const long long vecs = n_q + n_k;
  const dim3 grid(static_cast<unsigned>((vecs + kHeadWarps - 1) / kHeadWarps));
  if (torch_reduce == 1) {
    launch(qk_rms_norm_rope_hf_kernel<true>, grid, dim3(32 * kHeadWarps), pdl,
           stream, name, q, q_row_stride, q_heads, k, k_row_stride, k_heads,
           q_weight, k_weight, cos_half, sin_half, seq, factor_q, factor_k,
           false, false, eps);
  } else {
    launch(qk_rms_norm_rope_hf_kernel<false>, grid, dim3(32 * kHeadWarps), pdl,
           stream, name, q, q_row_stride, q_heads, k, k_row_stride, k_heads,
           q_weight, k_weight, cos_half, sin_half, seq, factor_q, factor_k,
           torch_wide_head_reduce(n_q), torch_wide_head_reduce(n_k), eps);
  }
}

void silu_mul_hf_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                      __nv_bfloat16* out, long long n, bool pdl,
                      cudaStream_t stream) {
  const char* name = "silu_mul_hf_bf16";
  if (n <= 0) return;
  if (!gate || !up || !out) {
    throw std::runtime_error(std::string(name) + ": null pointer argument");
  }
  if (n % 8 != 0 ||
      (reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) |
       reinterpret_cast<uintptr_t>(out)) % 16 != 0) {
    throw std::runtime_error(std::string(name) +
                             ": n must be a multiple of 8, pointers 16-byte aligned");
  }
  const long long n8 = n / 8;
  const dim3 grid(static_cast<unsigned>((n8 + kSiluThreads - 1) / kSiluThreads));
  launch(silu_mul_hf_kernel, grid, dim3(kSiluThreads), pdl, stream, name,
         gate, up, out, n8);
}

}  // namespace flash_rt::kernels
