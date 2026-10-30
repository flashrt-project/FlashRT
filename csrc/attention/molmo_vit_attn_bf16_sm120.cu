// SPDX-License-Identifier: Apache-2.0
//
// Non-causal BF16 attention for the MolmoAct2 vision tower, SM120a.
// Fixed head geometry: head_dim = 72, 16 query heads (no GQA).
//
// Why a dedicated kernel: the vendored FA2 kernel_traits require
// kHeadDim % 32 == 0, so a 72-deep head is dispatched to the 96-deep
// kernel and its CuTe zero-predicated columns make the QK/PV MMAs do
// 96/72 = 1.33x the useful work. The bf16 mma atom is m16n8k16, so a
// native kernel needs 72 -> 80 (5 k-atoms) for QK and exactly 72
// (9 n-atoms of 8) for PV.
//
// Layout contract (matches the MolmoAct2 vision qkv buffers):
//   q, k, v, out: (batch, seqlen, 16, 72) bf16, contiguous
//   row stride = num_heads * 72, head stride = 72.
//
// One CTA per (head, 128-row query tile, batch). 4 MMA warps, each owning
// two 16-row m-atoms; the query tile is read once into A-fragments in a
// prologue and the transposed-V tile aliases the Q shared region, so the
// CTA needs 42 KB and two CTAs are resident per SM. The wider query tile
// halves the per-row share of KV staging, barriers and load issue.
// V is transposed into a padded (d, key) layout for the P*V B-operand;
// P stays in registers (score atoms 2*kb / 2*kb+1 line up with the
// 16-key A fragment).

#include "molmo_vit_attn_bf16_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace attention {
namespace {

constexpr int kHD = 72, kHDP = 88, kHPV = 72;
constexpr int kBM = 128, kBN = 64, kWarps = 4, kThreads = kWarps * 32;
constexpr int kMI = 2;              // m-atoms per warp
constexpr int kNDP = kBN + 8;
constexpr int kNDI = 9, kNNI = kBN / 8, kNKI = kHDP / 16, kNKK = kBN / 16;

__device__ __forceinline__ void mma_bf16(float c[4], uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// fp16 inputs, fp16 accumulator: twice the f32-accumulate MMA rate on this
// part. Used for the per-KV-tile P*V partial only (see F16PV below).
__device__ __forceinline__ void mma_f16acc(uint32_t c[2], uint32_t a0, uint32_t a1,
                                           uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  unsigned a = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(a), "l"(src));
}
__device__ __forceinline__ void load_rows(__nv_bfloat16* dst,
                                          const __nv_bfloat16* src,
                                          int row0, int nrows, int L,
                                          int row_stride, int dst_stride,
                                          int tid) {
  constexpr int kChunks = kHD * 2 / 16;   // 9
  for (int idx = tid; idx < nrows * kChunks; idx += kThreads) {
    int r = idx / kChunks, c = idx % kChunks;
    int gr = row0 + r;
    int sr = gr < L ? gr : L - 1;
    cp16(dst + r * dst_stride + c * 8, src + (size_t)sr * row_stride + c * 8);
  }
}

// One (batch, head, 128-row query tile) work item; `smem` is the calling
// kernel's dynamic allocation.
//
// F16PV: the P*V product of each 64-key tile runs with fp16 P and V
// (exact bf16 -> fp16 for V in the fp16 normal range; P in [0, 1] gets 3
// more mantissa bits than the bf16 path) into a zeroed fp16 accumulator,
// which is then added to the fp32 running output. QK^T, the online
// softmax and the cross-tile accumulation stay fp32.
template <bool F16PV = false>
__device__ __forceinline__ void vit_item(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
    int L, int H, int qkv_row_stride, int out_row_stride, float scale,
    int z, int head, int qt, __nv_bfloat16* smem,
    __nv_fp8_e4m3* __restrict__ o8 = nullptr,
    const float* __restrict__ o8_scale = nullptr) {
  __nv_bfloat16* Qs = smem;
  __nv_bfloat16* Ks = Qs + kBM * kHDP;
  __nv_bfloat16* Vraw = Ks + kBN * kHDP;
  __nv_bfloat16* Vt = Qs;      // aliases Q after the prologue

  const int q_base = qt * kBM;
  const int row_stride = qkv_row_stride;
  const size_t base = ((size_t)z * L) * row_stride + (size_t)head * kHD;
  const __nv_bfloat16* qp = q + base;
  const __nv_bfloat16* kp = k + base;
  const __nv_bfloat16* vp = v + base;
  __nv_bfloat16* op = o + ((size_t)z * L) * out_row_stride + (size_t)head * kHD;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tt = lane & 3;

  for (int i = tid; i < kBM * (kHDP - kHD); i += kThreads) {
    int r = i / (kHDP - kHD), c = i % (kHDP - kHD);
    Qs[r * kHDP + kHD + c] = __float2bfloat16(0.f);
  }
  for (int i = tid; i < kBN * (kHDP - kHD); i += kThreads) {
    int r = i / (kHDP - kHD), c = i % (kHDP - kHD);
    Ks[r * kHDP + kHD + c] = __float2bfloat16(0.f);
  }
  load_rows(Qs, qp, q_base, kBM, L, row_stride, kHDP, tid);
  asm volatile("cp.async.commit_group;\n");
  const int n_tiles = (L + kBN - 1) / kBN;
  auto load_kv = [&](int t) {
    load_rows(Ks, kp, t * kBN, kBN, L, row_stride, kHDP, tid);
    load_rows(Vraw, vp, t * kBN, kBN, L, row_stride, kHPV, tid);
  };
  auto stage_v = [&]() {
    const int npair = (kBN / 2) * (kHD / 2);
    for (int idx = tid; idx < npair; idx += kThreads) {
      int kp2 = idx / (kHD / 2), dp2 = idx % (kHD / 2);
      int k0 = kp2 * 2, d0 = dp2 * 2;
      uint32_t u0 = *(const uint32_t*)&Vraw[k0 * kHPV + d0];
      uint32_t u1 = *(const uint32_t*)&Vraw[(k0 + 1) * kHPV + d0];
      if constexpr (F16PV) {
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1));
        const __half2 h0 = __floats2half2_rn(f0.x, f0.y);
        const __half2 h1 = __floats2half2_rn(f1.x, f1.y);
        u0 = *reinterpret_cast<const uint32_t*>(&h0);
        u1 = *reinterpret_cast<const uint32_t*>(&h1);
      }
      uint32_t o0 = (u0 & 0xFFFFu) | ((u1 & 0xFFFFu) << 16);
      uint32_t o1 = (u0 >> 16) | (u1 & 0xFFFF0000u);
      *(uint32_t*)&Vt[(d0 + 0) * kNDP + k0] = o0;
      *(uint32_t*)&Vt[(d0 + 1) * kNDP + k0] = o1;
    }
  };

  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();
  const int qrow0 = warp * 32;
  uint32_t qreg[kMI][kNKI][4];
#pragma unroll
  for (int mi = 0; mi < kMI; ++mi) {
#pragma unroll
    for (int ks = 0; ks < kNKI; ++ks) {
      int ko = ks * 16;
      int r0 = qrow0 + mi * 16 + g, r1 = r0 + 8;
      qreg[mi][ks][0] = *(const uint32_t*)&Qs[r0 * kHDP + ko + 2 * tt];
      qreg[mi][ks][1] = *(const uint32_t*)&Qs[r1 * kHDP + ko + 2 * tt];
      qreg[mi][ks][2] = *(const uint32_t*)&Qs[r0 * kHDP + ko + 8 + 2 * tt];
      qreg[mi][ks][3] = *(const uint32_t*)&Qs[r1 * kHDP + ko + 8 + 2 * tt];
    }
  }
  __syncthreads();

  float Oc[kMI][kNDI][4];
  float mi0[kMI], mi1[kMI], li0[kMI], li1[kMI];
#pragma unroll
  for (int mi = 0; mi < kMI; ++mi) {
    mi0[mi] = mi1[mi] = -1e30f; li0[mi] = li1[mi] = 0.f;
#pragma unroll
    for (int nd = 0; nd < kNDI; ++nd)
      Oc[mi][nd][0] = Oc[mi][nd][1] = Oc[mi][nd][2] = Oc[mi][nd][3] = 0.f;
  }

  for (int kv = 0; kv < n_tiles; ++kv) {
    load_kv(kv);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    stage_v();
    __syncthreads();

    float C[kMI][kNNI][4];
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi)
#pragma unroll
      for (int nb = 0; nb < kNNI; ++nb)
        C[mi][nb][0] = C[mi][nb][1] = C[mi][nb][2] = C[mi][nb][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kNKI; ++ks) {
#pragma unroll
      for (int nb = 0; nb < kNNI; ++nb) {
        int n = nb * 8;
        uint32_t b0 = *(const uint32_t*)&Ks[(n + g) * kHDP + ks * 16 + 2 * tt];
        uint32_t b1 = *(const uint32_t*)&Ks[(n + g) * kHDP + ks * 16 + 8 + 2 * tt];
#pragma unroll
        for (int mi = 0; mi < kMI; ++mi)
          mma_bf16(C[mi][nb], qreg[mi][ks][0], qreg[mi][ks][1],
                   qreg[mi][ks][2], qreg[mi][ks][3], b0, b1);
      }
    }
    if (kv == n_tiles - 1 && (L % kBN) != 0) {
      int lim = L - kv * kBN;
#pragma unroll
      for (int nb = 0; nb < kNNI; ++nb) {
        int c0 = nb * 8 + 2 * tt, c1 = c0 + 1;
#pragma unroll
        for (int mi = 0; mi < kMI; ++mi) {
          if (c0 >= lim) { C[mi][nb][0] = -1e30f; C[mi][nb][2] = -1e30f; }
          if (c1 >= lim) { C[mi][nb][1] = -1e30f; C[mi][nb][3] = -1e30f; }
        }
      }
    }
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi) {
      float mt0 = -1e30f, mt1 = -1e30f;
#pragma unroll
      for (int nb = 0; nb < kNNI; ++nb) {
        mt0 = fmaxf(mt0, fmaxf(C[mi][nb][0], C[mi][nb][1]));
        mt1 = fmaxf(mt1, fmaxf(C[mi][nb][2], C[mi][nb][3]));
      }
      mt0 *= scale; mt1 *= scale;
      mt0 = fmaxf(mt0, __shfl_xor_sync(~0u, mt0, 1));
      mt0 = fmaxf(mt0, __shfl_xor_sync(~0u, mt0, 2));
      mt1 = fmaxf(mt1, __shfl_xor_sync(~0u, mt1, 1));
      mt1 = fmaxf(mt1, __shfl_xor_sync(~0u, mt1, 2));
      float mn0 = fmaxf(mi0[mi], mt0), mn1 = fmaxf(mi1[mi], mt1);
      float corr0 = __expf(mi0[mi] - mn0), corr1 = __expf(mi1[mi] - mn1);
      float lt0 = 0.f, lt1 = 0.f;
#pragma unroll
      for (int nb = 0; nb < kNNI; ++nb) {
        C[mi][nb][0] = __expf(C[mi][nb][0] * scale - mn0);
        C[mi][nb][1] = __expf(C[mi][nb][1] * scale - mn0);
        lt0 += C[mi][nb][0] + C[mi][nb][1];
        C[mi][nb][2] = __expf(C[mi][nb][2] * scale - mn1);
        C[mi][nb][3] = __expf(C[mi][nb][3] * scale - mn1);
        lt1 += C[mi][nb][2] + C[mi][nb][3];
      }
      lt0 += __shfl_xor_sync(~0u, lt0, 1); lt0 += __shfl_xor_sync(~0u, lt0, 2);
      lt1 += __shfl_xor_sync(~0u, lt1, 1); lt1 += __shfl_xor_sync(~0u, lt1, 2);
      li0[mi] = li0[mi] * corr0 + lt0; li1[mi] = li1[mi] * corr1 + lt1;
      mi0[mi] = mn0; mi1[mi] = mn1;
#pragma unroll
      for (int nd = 0; nd < kNDI; ++nd) {
        Oc[mi][nd][0] *= corr0; Oc[mi][nd][1] *= corr0;
        Oc[mi][nd][2] *= corr1; Oc[mi][nd][3] *= corr1;
      }
    }
    uint32_t Ch[kMI][kNDI][2];
    if constexpr (F16PV) {
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi)
#pragma unroll
        for (int nd = 0; nd < kNDI; ++nd) Ch[mi][nd][0] = Ch[mi][nd][1] = 0u;
    }
#pragma unroll
    for (int kb = 0; kb < kNKK; ++kb) {
      int ko = kb * 16;
      uint32_t b0[kNDI], b1[kNDI];
#pragma unroll
      for (int nd = 0; nd < kNDI; ++nd) {
        int n = nd * 8 + g;
        b0[nd] = *(const uint32_t*)&Vt[n * kNDP + ko + 2 * tt];
        b1[nd] = *(const uint32_t*)&Vt[n * kNDP + ko + 8 + 2 * tt];
      }
      if constexpr (F16PV) {
#pragma unroll
        for (int mi = 0; mi < kMI; ++mi) {
          __half2 p00 = __floats2half2_rn(C[mi][2 * kb][0], C[mi][2 * kb][1]);
          __half2 p10 = __floats2half2_rn(C[mi][2 * kb][2], C[mi][2 * kb][3]);
          __half2 p01 = __floats2half2_rn(C[mi][2 * kb + 1][0], C[mi][2 * kb + 1][1]);
          __half2 p11 = __floats2half2_rn(C[mi][2 * kb + 1][2], C[mi][2 * kb + 1][3]);
#pragma unroll
          for (int nd = 0; nd < kNDI; ++nd)
            mma_f16acc(Ch[mi][nd], *reinterpret_cast<uint32_t*>(&p00),
                       *reinterpret_cast<uint32_t*>(&p10),
                       *reinterpret_cast<uint32_t*>(&p01),
                       *reinterpret_cast<uint32_t*>(&p11), b0[nd], b1[nd]);
        }
        continue;
      }
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi) {
        __nv_bfloat162 p00 = __floats2bfloat162_rn(C[mi][2 * kb][0], C[mi][2 * kb][1]);
        __nv_bfloat162 p10 = __floats2bfloat162_rn(C[mi][2 * kb][2], C[mi][2 * kb][3]);
        __nv_bfloat162 p01 = __floats2bfloat162_rn(C[mi][2 * kb + 1][0], C[mi][2 * kb + 1][1]);
        __nv_bfloat162 p11 = __floats2bfloat162_rn(C[mi][2 * kb + 1][2], C[mi][2 * kb + 1][3]);
        uint32_t a0 = *reinterpret_cast<uint32_t*>(&p00);
        uint32_t a1 = *reinterpret_cast<uint32_t*>(&p10);
        uint32_t a2 = *reinterpret_cast<uint32_t*>(&p01);
        uint32_t a3 = *reinterpret_cast<uint32_t*>(&p11);
#pragma unroll
        for (int nd = 0; nd < kNDI; ++nd)
          mma_bf16(Oc[mi][nd], a0, a1, a2, a3, b0[nd], b1[nd]);
      }
    }
    if constexpr (F16PV) {
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi)
#pragma unroll
        for (int nd = 0; nd < kNDI; ++nd) {
          const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&Ch[mi][nd][0]));
          const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&Ch[mi][nd][1]));
          Oc[mi][nd][0] += lo.x; Oc[mi][nd][1] += lo.y;
          Oc[mi][nd][2] += hi.x; Oc[mi][nd][3] += hi.y;
        }
    }
    __syncthreads();
  }

  if (o8) {
    // Static-scale FP8 copy of the bf16 output, bit-identical to
    // quantize_fp8_static applied to the bf16 tensor.
    const float inv_s = 1.0f / (*o8_scale);
    __nv_fp8_e4m3* op8 =
        o8 + ((size_t)z * L) * out_row_stride + (size_t)head * kHD;
    auto q8 = [&](__nv_bfloat162 v) -> uint16_t {
      float a = __bfloat162float(v.x) * inv_s;
      float b = __bfloat162float(v.y) * inv_s;
      __nv_fp8_e4m3 qa(fminf(fmaxf(a, -448.0f), 448.0f));
      __nv_fp8_e4m3 qb(fminf(fmaxf(b, -448.0f), 448.0f));
      return (uint16_t)(*reinterpret_cast<uint8_t*>(&qa)) |
             ((uint16_t)(*reinterpret_cast<uint8_t*>(&qb)) << 8);
    };
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi) {
      float ip0 = 1.f / li0[mi], ip1 = 1.f / li1[mi];
#pragma unroll
      for (int nd = 0; nd < kNDI; ++nd) {
        int d = nd * 8 + 2 * tt;
        int r0 = q_base + qrow0 + mi * 16 + g, r1 = r0 + 8;
        if (r0 < L)
          *(uint16_t*)&op8[(size_t)r0 * out_row_stride + d] =
              q8(__floats2bfloat162_rn(Oc[mi][nd][0] * ip0,
                                       Oc[mi][nd][1] * ip0));
        if (r1 < L)
          *(uint16_t*)&op8[(size_t)r1 * out_row_stride + d] =
              q8(__floats2bfloat162_rn(Oc[mi][nd][2] * ip1,
                                       Oc[mi][nd][3] * ip1));
      }
    }
    return;
  }
#pragma unroll
  for (int mi = 0; mi < kMI; ++mi) {
    float ip0 = 1.f / li0[mi], ip1 = 1.f / li1[mi];
#pragma unroll
    for (int nd = 0; nd < kNDI; ++nd) {
      int d = nd * 8 + 2 * tt;
      int r0 = q_base + qrow0 + mi * 16 + g, r1 = r0 + 8;
      if (r0 < L) {
        __nv_bfloat162 v0 = __floats2bfloat162_rn(Oc[mi][nd][0] * ip0,
                                                  Oc[mi][nd][1] * ip0);
        *(uint32_t*)&op[(size_t)r0 * out_row_stride + d] = *(uint32_t*)&v0;
      }
      if (r1 < L) {
        __nv_bfloat162 v1 = __floats2bfloat162_rn(Oc[mi][nd][2] * ip1,
                                                  Oc[mi][nd][3] * ip1);
        *(uint32_t*)&op[(size_t)r1 * out_row_stride + d] = *(uint32_t*)&v1;
      }
    }
  }
}

template <bool F16PV>
__global__ void __launch_bounds__(kThreads, 2)
vit_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
           const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
           int L, int H, int qkv_row_stride, int out_row_stride, float scale,
           __nv_fp8_e4m3* __restrict__ o8, const float* __restrict__ o8_scale) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  vit_item<F16PV>(q, k, v, o, L, H, qkv_row_stride, out_row_stride, scale,
                  blockIdx.z, blockIdx.x, gridDim.y - 1 - blockIdx.y, smem,
                  o8, o8_scale);
}

// Persistent form: a fixed CTA count walks the (batch, head, query-tile)
// work items in launch order. Same per-item tile order and math.
__global__ void __launch_bounds__(kThreads, 2)
vit_persist_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                   const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
                   int L, int H, int qkv_row_stride, int out_row_stride,
                   float scale, int qtiles, int total) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    const int z = item / (H * qtiles);
    const int rem = item - z * (H * qtiles);
    const int head = rem % H;
    const int qt = qtiles - 1 - rem / H;
    vit_item(q, k, v, o, L, H, qkv_row_stride, out_row_stride, scale,
             z, head, qt, smem);
  }
}
}  // namespace

static int vit_launch(const void* q, const void* k, const void* v, void* out,
                      int batch, int seqlen, int num_heads, int head_dim,
                      int qkv_row_stride, int out_row_stride,
                      float softmax_scale, int blocks, cudaStream_t stream,
                      void* out_fp8 = nullptr,
                      const float* out_fp8_scale = nullptr,
                      bool f16pv = false) {
  if (!q || !k || !v || (!out && !out_fp8)) return 1;
  if (out_fp8 && (!out_fp8_scale || blocks > 0)) return 3;
  if (head_dim != kHD || num_heads != 16 || batch <= 0 || seqlen <= 0) {
    return 2;
  }
  constexpr int elems = kBM * kHDP + kBN * kHDP + kBN * kHPV;
  constexpr int bytes = elems * (int)sizeof(__nv_bfloat16);
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(vit_kernel<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaFuncSetAttribute(vit_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaFuncSetAttribute(vit_persist_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    attr_done = true;
  }
  const int qtiles = (seqlen + kBM - 1) / kBM;
  if (blocks <= 0) {
    dim3 grid(num_heads, qtiles, batch);
    (f16pv ? vit_kernel<true> : vit_kernel<false>)<<<grid, kThreads, bytes, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v, (__nv_bfloat16*)out,
        seqlen, num_heads, qkv_row_stride, out_row_stride, softmax_scale,
        (__nv_fp8_e4m3*)out_fp8, out_fp8_scale);
  } else {
    const int total = batch * num_heads * qtiles;
    const int nblk = blocks < total ? blocks : total;
    vit_persist_kernel<<<nblk, kThreads, bytes, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v, (__nv_bfloat16*)out,
        seqlen, num_heads, qkv_row_stride, out_row_stride, softmax_scale,
        qtiles, total);
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : 1;
}

int molmo_vit_attn_bf16(const void* q, const void* k, const void* v, void* out,
                        int batch, int seqlen, int num_heads, int head_dim,
                        int qkv_row_stride, int out_row_stride,
                        float softmax_scale, cudaStream_t stream) {
  return vit_launch(q, k, v, out, batch, seqlen, num_heads, head_dim,
                    qkv_row_stride, out_row_stride, softmax_scale, 0, stream);
}

int molmo_vit_attn_bf16_fp8out(const void* q, const void* k, const void* v,
                               void* out_fp8, const float* out_scale,
                               int batch, int seqlen, int num_heads,
                               int head_dim, int qkv_row_stride,
                               int out_row_stride, float softmax_scale,
                               cudaStream_t stream) {
  return vit_launch(q, k, v, nullptr, batch, seqlen, num_heads, head_dim,
                    qkv_row_stride, out_row_stride, softmax_scale, 0, stream,
                    out_fp8, out_scale);
}

int molmo_vit_attn_f16pv(const void* q, const void* k, const void* v, void* out,
                         int batch, int seqlen, int num_heads, int head_dim,
                         int qkv_row_stride, int out_row_stride,
                         float softmax_scale, cudaStream_t stream) {
  return vit_launch(q, k, v, out, batch, seqlen, num_heads, head_dim,
                    qkv_row_stride, out_row_stride, softmax_scale, 0, stream,
                    nullptr, nullptr, true);
}

int molmo_vit_attn_bf16_persist(const void* q, const void* k, const void* v,
                                void* out, int batch, int seqlen,
                                int num_heads, int head_dim,
                                int qkv_row_stride, int out_row_stride,
                                float softmax_scale, int blocks,
                                cudaStream_t stream) {
  return vit_launch(q, k, v, out, batch, seqlen, num_heads, head_dim,
                    qkv_row_stride, out_row_stride, softmax_scale, blocks,
                    stream);
}


}  // namespace attention
}  // namespace flash_rt
