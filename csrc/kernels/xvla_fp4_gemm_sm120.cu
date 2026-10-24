// FlashRT — X-VLA small-M NVFP4 W4A4 GEMMs (SM120). See the header.
#include "xvla_fp4_gemm_sm120.cuh"
#include "pdl_switch.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace xvla {
namespace {


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_fp4(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  constexpr uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

template <int BKB>
__device__ __forceinline__ int swz(int r, int c) {
  if constexpr (BKB == 128) return c ^ (r & 7);
  else return c ^ ((r >> 1) & 3);
}

template <int BM, int BN, int WM, int WN, int STAGES, int BKB_ = 128>
struct Cfg {
  static constexpr int BKB = BKB_;                // bytes of K per row per stage
  static constexpr int KCH = BKB / 32;            // 64-wide K chunks per stage
  static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
  static constexpr int CWARPS = WARPS_M * WARPS_N;
  static constexpr int THREADS = (CWARPS + 1) * 32;
  static constexpr int MT = WM / 16, NT = WN / 8;
  static constexpr int A_BYTES = BM * BKB, B_BYTES = BN * BKB;
  static constexpr int SFA_BLK = (128 % BM == 0) ? 1 : (BM + 127) / 128 + 1;   // 128-row atom groups an M tile may touch
  static constexpr int SFB_BLK = (BN + 127) / 128;
  static constexpr int SF_GROUP = KCH * 512;               // one 128-row atom group, one stage
  static constexpr int A_OFF = 0;
  static constexpr int B_OFF = A_BYTES;
  static constexpr int SFA_OFF = A_BYTES + B_BYTES;
  static constexpr int SFB_OFF = SFA_OFF + SFA_BLK * SF_GROUP;
  static constexpr int STAGE = ((SFB_OFF + SFB_BLK * SF_GROUP) + 1023) / 1024 * 1024;
  static constexpr int SMEM = STAGE * STAGES + 1024 /*align*/ + 256 /*barriers*/;
};

template <int BM, int BN, int WM, int WN, int STAGES, int BKB_ = 128, int EPI = 0>
__global__ void __launch_bounds__(Cfg<BM, BN, WM, WN, STAGES, BKB_>::THREADS, 1)
fp4_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
             const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
             const __nv_bfloat16* __restrict__ bias, void* __restrict__ Dv, uint8_t* __restrict__ SFD,
             int M, int N, int K, float alpha) {
  using C = Cfg<BM, BN, WM, WN, STAGES, BKB_>;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::STAGE * STAGES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + STAGES);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kchunks = K >> 6;
  const int KT = (K >> 1) / C::BKB;
  const int rb0 = m0 >> 7;
  const int nblkA = ((min(m0 + BM, M) - 1) >> 7) - rb0 + 1;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, C::CWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == C::CWARPS) {
    // Producer.
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      const uint32_t txb = C::B_BYTES + C::SFB_BLK * C::SF_GROUP;
      const uint32_t txa = C::A_BYTES + nblkA * C::SF_GROUP;
      // Weight-side loads of the first stages do not depend on the producer kernel.
      const int pre = KT < STAGES ? KT : STAGES;
      for (int kt = 0; kt < pre; ++kt) {
        uint8_t* st = smem + kt * C::STAGE;
        const uint32_t fb = full0 + 8 * kt;
        mbar_expect_tx(fb, txb + txa);
        tma_2d(smem_u32(st + C::B_OFF), &tmB, kt * C::BKB, n0, fb);
        for (int b = 0; b < C::SFB_BLK; ++b)
          bulk_1d(smem_u32(st + C::SFB_OFF + b * C::SF_GROUP),
                  SFB + ((size_t)((n0 >> 7) + b) * kchunks + kt * C::KCH) * 512, C::SF_GROUP, fb);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int kt = 0; kt < pre; ++kt) {
        uint8_t* st = smem + kt * C::STAGE;
        const uint32_t fb = full0 + 8 * kt;
        tma_2d(smem_u32(st + C::A_OFF), &tmA, kt * C::BKB, m0, fb);
        for (int b = 0; b < nblkA; ++b)
          bulk_1d(smem_u32(st + C::SFA_OFF + b * C::SF_GROUP),
                  SFA + ((size_t)(rb0 + b) * kchunks + kt * C::KCH) * 512, C::SF_GROUP, fb);
      }
      for (int kt = pre; kt < KT; ++kt) {
        const int s = kt % STAGES;
        mbar_wait(empty0 + 8 * s, ((kt / STAGES) - 1) & 1);
        uint8_t* st = smem + s * C::STAGE;
        const uint32_t fb = full0 + 8 * s;
        mbar_expect_tx(fb, txb + txa);
        tma_2d(smem_u32(st + C::B_OFF), &tmB, kt * C::BKB, n0, fb);
        for (int b = 0; b < C::SFB_BLK; ++b)
          bulk_1d(smem_u32(st + C::SFB_OFF + b * C::SF_GROUP),
                  SFB + ((size_t)((n0 >> 7) + b) * kchunks + kt * C::KCH) * 512, C::SF_GROUP, fb);
        tma_2d(smem_u32(st + C::A_OFF), &tmA, kt * C::BKB, m0, fb);
        for (int b = 0; b < nblkA; ++b)
          bulk_1d(smem_u32(st + C::SFA_OFF + b * C::SF_GROUP),
                  SFA + ((size_t)(rb0 + b) * kchunks + kt * C::KCH) * 512, C::SF_GROUP, fb);
      }
    }
    return;
  }

  // Consumers.
  const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;
  const int t0 = lane & 3, t1 = lane >> 2;
  const int sfa_row = (lane & 1) * 8 + t1;
  float acc[C::MT][C::NT][4];
#pragma unroll
  for (int i = 0; i < C::MT; ++i)
#pragma unroll
    for (int j = 0; j < C::NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  // Per-lane byte offsets inside one stage's scale atoms (kc adds 512).
  static_assert(C::WARPS_N == 4 && C::NT % 4 == 0, "n-tiles interleave at 32 columns");
  int sfa_off[C::MT];
  // n-tile j of warp wn covers columns n0 + wn*8 + 32*j: its four scale words
  // for j = 4q..4q+3 are adjacent in the atom (one 16-byte load).
  const int sfb_off = C::SFB_OFF + (wn * 8 + t1) * 16;
#pragma unroll
  for (int i = 0; i < C::MT; ++i) {
    int r = m0 + wm * WM + i * 16 + sfa_row;
    sfa_off[i] = C::SFA_OFF + ((r >> 7) - rb0) * C::SF_GROUP + (r & 31) * 16 + ((r >> 5) & 3) * 4;
  }
  // ldmatrix row addresses (128B rows, 16B chunk ^= row & 7).
  const int ar = wm * WM + (lane & 15);
  const int br = wn * 8 + (lane & 7) + ((lane >> 4) << 5);
  const int ach = lane >> 4, bch = (lane >> 3) & 1;

  uint32_t af[2][C::MT][4], bf[2][C::NT][2], sfa[2][C::MT], sfb[2][C::NT];
  auto load_frag = [&](const uint8_t* st, int kc, int buf) {
    const uint32_t sa = smem_u32(st + C::A_OFF), sb = smem_u32(st + C::B_OFF);
#pragma unroll
    for (int i = 0; i < C::MT; ++i) {
      int r = ar + i * 16;
      int c = swz<C::BKB>(r, kc * 2 + ach);
      ldsm4(af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3], sa + r * C::BKB + c * 16);
      sfa[buf][i] = *reinterpret_cast<const uint32_t*>(st + sfa_off[i] + kc * 512);
    }
#pragma unroll
    for (int j = 0; j < C::NT; j += 2) {
      int r = br + j * 32;
      int c = swz<C::BKB>(r, kc * 2 + bch);
      ldsm4(bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1], sb + r * C::BKB + c * 16);
    }
#pragma unroll
    for (int j = 0; j < C::NT; j += 4) {
      uint4 v = *reinterpret_cast<const uint4*>(st + sfb_off + (j >> 2) * C::SF_GROUP + kc * 512);
      sfb[buf][j] = v.x; sfb[buf][j + 1] = v.y; sfb[buf][j + 2] = v.z; sfb[buf][j + 3] = v.w;
    }
  };
  // Fragments run one 64-wide K chunk ahead of the MMAs. With four consumer
  // warps that carries across stage boundaries (KCH is even, so the register
  // buffer parity carries over); eight warps have no registers to spare for
  // it under the producer warp's allocation and refill at each stage.
  constexpr bool kCross = C::CWARPS < 8;
  if constexpr (kCross) {
    mbar_wait(full0, 0);
    load_frag(smem, 0, 0);
  }
  for (int kt = 0; kt < KT; ++kt) {
    const int s = kt % STAGES;
    const uint8_t* st = smem + s * C::STAGE;
    if constexpr (!kCross) {
      mbar_wait(full0 + 8 * s, (kt / STAGES) & 1);
      load_frag(st, 0, 0);
    }
#pragma unroll
    for (int kc = 0; kc < C::KCH; ++kc) {
      const int cur = kc & 1;
      if (kc + 1 < C::KCH) {
        load_frag(st, kc + 1, cur ^ 1);
      } else if (kCross && kt + 1 < KT) {
        const int s1 = (kt + 1) % STAGES;
        mbar_wait(full0 + 8 * s1, ((kt + 1) / STAGES) & 1);
        load_frag(smem + s1 * C::STAGE, 0, cur ^ 1);
      }
#pragma unroll
      for (int j = 0; j < C::NT; ++j)
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
          mma_fp4(acc[i][j], af[cur][i], bf[cur][j][0], bf[cur][j][1], sfa[cur][i], sfb[cur][j]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if constexpr (EPI == 0) {
    __nv_bfloat16* D = reinterpret_cast<__nv_bfloat16*>(Dv);
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
      int col = n0 + wn * 8 + j * 32 + t0 * 2;
      float b0 = __bfloat162float(bias[col]), b1 = __bfloat162float(bias[col + 1]);
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
        int row = m0 + wm * WM + i * 16 + t1;
        if (row < M)
          *reinterpret_cast<__nv_bfloat162*>(D + (size_t)row * N + col) =
              __floats2bfloat162_rn(fmaf(alpha, acc[i][j][0], b0), fmaf(alpha, acc[i][j][1], b1));
        if (row + 8 < M)
          *reinterpret_cast<__nv_bfloat162*>(D + (size_t)(row + 8) * N + col) =
              __floats2bfloat162_rn(fmaf(alpha, acc[i][j][2], b0), fmaf(alpha, acc[i][j][3], b1));
      }
    }
  } else {
    // bias + GELU(tanh) + NVFP4 with per-16-column UE4M3 scales, matching the
    // cutlass Sm120 LinCombPerColBiasEltActBlockScaleFactor<GELU_taylor> epilogue.
    constexpr int LD = BN + 4;
    float* tile = reinterpret_cast<float*>(smem);
    asm volatile("bar.sync 1, %0;" ::"n"(C::CWARPS * 32) : "memory");   // all stages consumed
    const float k0 = 0.7978845608028654f, k1 = k0 * 0.044715f;
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
      int cl = wn * 8 + j * 32 + t0 * 2;
      float b0 = __bfloat162float(bias[n0 + cl]), b1 = __bfloat162float(bias[n0 + cl + 1]);
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
        int rl = wm * WM + i * 16 + t1;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float z = fmaf(alpha, acc[i][j][2 * h + e], e ? b1 : b0);
            float v1 = fmaf(k1 * z, z, k0);
            float v2 = z * v1, v3;
            asm("tanh.approx.f32 %0, %1;" : "=f"(v3) : "f"(v2));
            tile[(rl + 8 * h) * LD + cl + e] = 0.5f * fmaf(z, v3, z);
          }
        }
      }
    }
    asm volatile("bar.sync 1, %0;" ::"n"(C::CWARPS * 32) : "memory");
    float sd;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sd) : "f"(6.0f));
    uint8_t* D = reinterpret_cast<uint8_t*>(Dv);
    const int nblk = BN / 16;
    for (int b = threadIdx.x; b < BM * nblk; b += C::CWARPS * 32) {
      int rl = b / nblk, g = b % nblk;
      int row = m0 + rl;
      if (row >= M) continue;
      const float* src = tile + rl * LD + g * 16;
      float v[16];
#pragma unroll
      for (int q = 0; q < 16; q += 4) {
        float4 t = *reinterpret_cast<const float4*>(src + q);
        v[q] = t.x; v[q + 1] = t.y; v[q + 2] = t.z; v[q + 3] = t.w;
      }
      float amax = 0.f;
#pragma unroll
      for (int q = 0; q < 16; ++q) asm("max.NaN.f32 %0, %0, %1;" : "+f"(amax) : "f"(fabsf(v[q])));
      float pv = amax * sd;
      uint16_t q8;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q8) : "f"(0.0f), "f"(pv));
      uint8_t sfq = q8 & 0xff;
      uint32_t h2;
      asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)sfq));
      float qf = __half2float(__ushort_as_half((unsigned short)(h2 & 0xffff)));
      float rcp;
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(qf));
      float sc;
      asm("min.NaN.f32 %0, %1, %2;" : "=f"(sc) : "f"(rcp), "f"(3.402823466e38f));
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int q = 0; q < 8; q += 2) {
        uint16_t p;
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }"
            : "=h"(p) : "f"(v[q + 1] * sc), "f"(v[q] * sc));
        lo |= uint32_t(p) << (q * 4);
      }
#pragma unroll
      for (int q = 0; q < 8; q += 2) {
        uint16_t p;
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }"
            : "=h"(p) : "f"(v[8 + q + 1] * sc), "f"(v[8 + q] * sc));
        hi |= uint32_t(p) << (q * 4);
      }
      int col = n0 + g * 16;
      *reinterpret_cast<uint2*>(D + (size_t)row * (N >> 1) + (col >> 1)) = make_uint2(lo, hi);
      int gg = col >> 4;
      SFD[(((row >> 7) * (N >> 6) + (gg >> 2)) << 9) + ((row & 31) << 4) + (((row >> 5) & 3) << 2) + (gg & 3)] = sfq;
    }
  }
}

// ── host side ──────────────────────────────────────────────────────────

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                  const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn encode_tiled() {
  static EncodeTiledFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
      p = nullptr;
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

// Tensor maps depend only on (pointer, shape, box); the pipeline's buffers are
// static, so each is encoded once and reused by every capture or replay.
struct MapKey {
  uintptr_t ptr;
  int rows, row_bytes, box_rows, box_bytes;
  bool operator==(const MapKey& o) const { return std::memcmp(this, &o, sizeof(MapKey)) == 0; }
};
struct MapKeyHash {
  size_t operator()(const MapKey& k) const noexcept {
    return std::hash<uintptr_t>()(k.ptr) ^ (size_t(k.rows) << 1) ^ (size_t(k.row_bytes) << 21) ^
           (size_t(k.box_rows) << 41) ^ (size_t(k.box_bytes) << 51);
  }
};

const CUtensorMap* row_map(const void* p, int rows, int row_bytes, int box_rows, int box_bytes) {
  static std::mutex mu;
  static std::unordered_map<MapKey, CUtensorMap, MapKeyHash> cache;
  MapKey key{reinterpret_cast<uintptr_t>(p), rows, row_bytes, box_rows, box_bytes};
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return &it->second;
  EncodeTiledFn fn = encode_tiled();
  if (!fn) return nullptr;
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {(cuuint32_t)box_bytes, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  if (fn(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(p), dims, strides, box, es,
         CU_TENSOR_MAP_INTERLEAVE_NONE,
         box_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
         CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return nullptr;
  return &cache.emplace(key, m).first->second;
}

template <int BM, int BN, int WM, int WN, int STAGES, int BKB, int EPI>
int launch(const void* A, const void* B, const void* SFA, const void* SFB, const void* bias,
           void* D, void* SFD, int M, int N, int K, float alpha, cudaStream_t stream) {
  using C = Cfg<BM, BN, WM, WN, STAGES, BKB>;
  static_assert(EPI == 0 || BM * (BN + 4) * 4 <= C::STAGE * STAGES, "epilogue tile exceeds smem");
  if (M <= 0) return 0;
  if (N % BN || (K / 2) % BKB) return 1;
  const CUtensorMap* ta = row_map(A, M, K / 2, BM, BKB);
  const CUtensorMap* tb = row_map(B, N, K / 2, BN, BKB);
  if (!ta || !tb) return 3;
  auto k = fp4_gemm_kernel<BM, BN, WM, WN, STAGES, BKB, EPI>;
  static const cudaError_t attr =
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  if (attr != cudaSuccess) return 4;
  return flash_rt::pdl::launch(k, dim3(N / BN, (M + BM - 1) / BM), dim3(C::THREADS), C::SMEM, stream,
                               *ta, *tb, (const uint8_t*)SFA, (const uint8_t*)SFB,
                               (const __nv_bfloat16*)bias, D, (uint8_t*)SFD, M, N, K, alpha) ==
                 cudaSuccess
             ? 0
             : 2;
}

}  // namespace

// Per-shape tiles, measured in-graph on the pipeline's real operands
// (M = 262 / 100 / 30). Small M: 32-row tiles over every N. M ~ 100 with
// N >= 3072: one 112-row tile. M > 128: 80 x 128 at N = 1024 (32 CTAs on 36
// SMs, 288 rows instead of 384), 96 x 256 at N = 3072 (36 CTAs, one wave).
int fp4_gemm_bias_bf16out(const void* A, const void* B, void* D, const void* bias, int M, int N,
                          int K, const void* SFA, const void* SFB, float alpha,
                          cudaStream_t stream) {
  if (M <= 32 || (M <= 128 && N <= 1024))
    return launch<32, 128, 32, 32, 4, 128, 0>(A, B, SFA, SFB, bias, D, nullptr, M, N, K, alpha, stream);
  if (M <= 128)
    return launch<112, 128, 112, 32, 4, 64, 0>(A, B, SFA, SFB, bias, D, nullptr, M, N, K, alpha, stream);
  if (N <= 1024)
    return launch<80, 128, 80, 32, 3, 128, 0>(A, B, SFA, SFB, bias, D, nullptr, M, N, K, alpha, stream);
  if (N % 256 == 0)
    return launch<96, 256, 48, 64, 3, 64, 0>(A, B, SFA, SFB, bias, D, nullptr, M, N, K, alpha, stream);
  return launch<96, 128, 48, 32, 4, 64, 0>(A, B, SFA, SFB, bias, D, nullptr, M, N, K, alpha, stream);
}

int fp4_gemm_bias_gelu_fp4out(const void* A, const void* B, const void* SFA, const void* SFB,
                              const void* bias, void* D, void* SFD, int M, int N, int K, float alpha,
                              cudaStream_t stream) {
  if (M <= 32)
    return launch<32, 128, 32, 32, 4, 128, 1>(A, B, SFA, SFB, bias, D, SFD, M, N, K, alpha, stream);
  if (M <= 128)
    return launch<112, 128, 112, 32, 4, 64, 1>(A, B, SFA, SFB, bias, D, SFD, M, N, K, alpha, stream);
  // fc1 at M = 262: two 128-byte K stages measured 0.1 ms faster equal-scope
  // than four 64-byte ones (21.51 vs 21.61 ms).
  return launch<96, 128, 48, 32, 2, 128, 1>(A, B, SFA, SFB, bias, D, SFD, M, N, K, alpha, stream);
}

}  // namespace xvla
}  // namespace flash_rt
