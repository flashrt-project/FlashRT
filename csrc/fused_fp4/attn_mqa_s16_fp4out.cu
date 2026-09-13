// See attn_mqa_s16_fp4out.cuh.
#include "fused_fp4/attn_mqa_s16_fp4out.cuh"
#include "fused_fp4/pdl.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr int HD = 256;             // head dim
constexpr int TK = 64;              // keys per tile
constexpr int QR = 32;              // query rows per CTA: two heads x 16
constexpr int THREADS = 256;
constexpr int NWARP = THREADS / 32;
constexpr int PITCH = HD + 8;       // halves per K/V/Q smem row (528 B: ldmatrix conflict-free)
constexpr int PPITCH = TK + 8;      // halves per P row
constexpr unsigned FULL = 0xffffffffu;

template <int STAGES>
struct __align__(16) Smem {
  __half q[QR * PITCH];
  __half k[STAGES][TK * PITCH];
  __half v[STAGES][TK * PITCH];
  __half p[QR * PPITCH];
  float rmax[NWARP][QR];
  float rsum[NWARP][QR];
  int last;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Reference e2m1 rounding of quantize_fp4_sfa_vec (ties toward zero).
__device__ __forceinline__ uint8_t fp32_to_e2m1_ref(float x) {
  uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
  float ax = fabsf(x);
  uint8_t mant;
  if      (ax <= 0.25f) mant = 0u;
  else if (ax <= 0.75f) mant = 1u;
  else if (ax <= 1.25f) mant = 2u;
  else if (ax <= 1.75f) mant = 3u;
  else if (ax <= 2.5f)  mant = 4u;
  else if (ax <= 3.5f)  mant = 5u;
  else if (ax <= 5.0f)  mant = 6u;
  else                  mant = 7u;
  return sign | mant;
}

template <int KSPLIT, int STAGES>
__global__ void __launch_bounds__(THREADS, 1)
attn_mqa_s16_fp4out_kernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                           float* __restrict__ ws, int* __restrict__ counters,
                           uint8_t* __restrict__ packed, uint8_t* __restrict__ sfa,
                           int S, int T, int NH, float scale, int per_split, int dbg) {
  flashrt_pdl_wait_and_trigger();
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<STAGES>& sm = *reinterpret_cast<Smem<STAGES>*>(smem_raw);
  const bool skip_mma = dbg & 1, skip_load = dbg & 2;
  const int split = blockIdx.x, hp = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int k_begin = split * per_split;
  const int k_end = min(T, k_begin + per_split);
  const int ntiles = (k_end > k_begin) ? (k_end - k_begin + TK - 1) / TK : 0;
  const int QD = NH * HD;
  constexpr int PART = QR * HD + 2 * QR;   // floats per (hp, split) partial

  // Q rows: [0,S) = head 2hp, [16,16+S) = head 2hp+1, the rest zero.
  for (int i = tid; i < QR * (HD / 8); i += THREADS) {
    const int r = i >> 5, c = i & 31, h = r >> 4, t = r & 15;
    __half* dst = &sm.q[r * PITCH + c * 8];
    if (t < S) cp_async16(dst, Q + static_cast<size_t>(t) * QD + (2 * hp + h) * HD + c * 8, true);
    else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
  }
  auto load_tile = [&](int tile, int stage) {
    const int key0 = k_begin + tile * TK;
    for (int i = tid; i < TK * (HD / 8); i += THREADS) {
      const int r = i >> 5, c = i & 31;
      const int key = key0 + r;
      const bool valid = key < k_end;
      const size_t off = static_cast<size_t>(valid ? key : 0) * HD + c * 8;
      cp_async16(&sm.k[stage][r * PITCH + c * 8], K + off, valid);
      cp_async16(&sm.v[stage][r * PITCH + c * 8], V + off, valid);
    }
  };
  #pragma unroll
  for (int t = 0; t < STAGES; ++t) {
    if (t < ntiles && !skip_load) load_tile(t, t);
    cp_commit();
  }

  const int rows[4] = {lane >> 2, (lane >> 2) + 8, 16 + (lane >> 2), 24 + (lane >> 2)};
  float m_run[4], l_run[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) { m_run[i] = -1e30f; l_run[i] = 0.f; }
  float o_acc[2][4][4];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      #pragma unroll
      for (int c = 0; c < 4; ++c) o_acc[mt][nt][c] = 0.f;

  for (int tile = 0; tile < ntiles; ++tile) {
    const int stage = tile % STAGES;
    // Groups committed after tile t's group: min(STAGES-1, ntiles-1-t).
    if (tile + STAGES - 1 < ntiles) cp_wait<STAGES - 1>();
    else if (STAGES >= 3 && tile + 1 < ntiles) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();
    // S = Q K^T for this warp's 8 keys.
    float s_acc[2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int c = 0; c < 4; ++c) s_acc[mt][c] = 0.f;
    const int kn0 = warp * 8;
    if (!skip_mma)
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ++ks) {
      uint32_t b[2];
      ldsm_x2(b, &sm.k[stage][(kn0 + (lane & 7)) * PITCH + ks * 16 + ((lane >> 3) & 1) * 8]);
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t a[4];
        ldsm_x4(a, &sm.q[(mt * 16 + (lane & 15)) * PITCH + ks * 16 + (lane >> 4) * 8]);
        mma16816(s_acc[mt], a, b);
      }
    }
    const int key0 = k_begin + tile * TK;
    const int kc = key0 + kn0 + 2 * (lane & 3);
    const bool v0 = kc < k_end, v1 = (kc + 1) < k_end;
    float wmax[4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      s_acc[mt][0] = v0 ? s_acc[mt][0] * scale : -1e30f;
      s_acc[mt][1] = v1 ? s_acc[mt][1] * scale : -1e30f;
      s_acc[mt][2] = v0 ? s_acc[mt][2] * scale : -1e30f;
      s_acc[mt][3] = v1 ? s_acc[mt][3] * scale : -1e30f;
      float m0 = fmaxf(s_acc[mt][0], s_acc[mt][1]);
      float m1 = fmaxf(s_acc[mt][2], s_acc[mt][3]);
      m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
      m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
      wmax[mt * 2] = m0; wmax[mt * 2 + 1] = m1;
    }
    if ((lane & 3) == 0) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) sm.rmax[warp][rows[i]] = wmax[i];
    }
    __syncthreads();
    float m_new[4], alpha[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float tm = sm.rmax[0][rows[i]];
      #pragma unroll
      for (int w = 1; w < NWARP; ++w) tm = fmaxf(tm, sm.rmax[w][rows[i]]);
      m_new[i] = fmaxf(m_run[i], tm);
      alpha[i] = __expf(m_run[i] - m_new[i]);
    }
    float psum[4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const float p0 = __expf(s_acc[mt][0] - m_new[mt * 2]);
      const float p1 = __expf(s_acc[mt][1] - m_new[mt * 2]);
      const float p2 = __expf(s_acc[mt][2] - m_new[mt * 2 + 1]);
      const float p3 = __expf(s_acc[mt][3] - m_new[mt * 2 + 1]);
      const int r0 = mt * 16 + (lane >> 2), c = kn0 + 2 * (lane & 3);
      *reinterpret_cast<__half2*>(&sm.p[r0 * PPITCH + c]) = __floats2half2_rn(p0, p1);
      *reinterpret_cast<__half2*>(&sm.p[(r0 + 8) * PPITCH + c]) = __floats2half2_rn(p2, p3);
      float q0 = p0 + p1, q1 = p2 + p3;
      q0 += __shfl_xor_sync(FULL, q0, 1); q0 += __shfl_xor_sync(FULL, q0, 2);
      q1 += __shfl_xor_sync(FULL, q1, 1); q1 += __shfl_xor_sync(FULL, q1, 2);
      psum[mt * 2] = q0; psum[mt * 2 + 1] = q1;
    }
    if ((lane & 3) == 0) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) sm.rsum[warp][rows[i]] = psum[i];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = m_new[i];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        o_acc[mt][nt][0] *= alpha[mt * 2];     o_acc[mt][nt][1] *= alpha[mt * 2];
        o_acc[mt][nt][2] *= alpha[mt * 2 + 1]; o_acc[mt][nt][3] *= alpha[mt * 2 + 1];
      }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float ls = 0.f;
      #pragma unroll
      for (int w = 0; w < NWARP; ++w) ls += sm.rsum[w][rows[i]];
      l_run[i] = l_run[i] * alpha[i] + ls;
    }
    // O += P V (this warp: output columns [32*warp, 32*warp+32)).
    if (!skip_mma)
    #pragma unroll
    for (int ks = 0; ks < TK / 16; ++ks) {
      uint32_t a[2][4];
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        ldsm_x4(a[mt], &sm.p[(mt * 16 + (lane & 15)) * PPITCH + ks * 16 + (lane >> 4) * 8]);
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t b[2];
        ldsm_x2_trans(b, &sm.v[stage][(ks * 16 + (lane & 15)) * PITCH + warp * 32 + nt * 8]);
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma16816(o_acc[mt][nt], a[mt], b);
      }
    }
    __syncthreads();
    if (tile + STAGES < ntiles && !skip_load) load_tile(tile + STAGES, stage);
    cp_commit();
  }

  // Partial (O, m, l) for this split.
  float* wsp = ws + static_cast<size_t>(hp * KSPLIT + split) * PART;
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = mt * 16 + (lane >> 2), c = warp * 32 + nt * 8 + 2 * (lane & 3);
      if ((lane >> 2) < S)
        *reinterpret_cast<float2*>(&wsp[r0 * HD + c]) = make_float2(o_acc[mt][nt][0], o_acc[mt][nt][1]);
      if ((lane >> 2) + 8 < S)
        *reinterpret_cast<float2*>(&wsp[(r0 + 8) * HD + c]) = make_float2(o_acc[mt][nt][2], o_acc[mt][nt][3]);
    }
  if (warp == 0 && (lane & 3) == 0) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) { wsp[QR * HD + rows[i]] = m_run[i]; wsp[QR * HD + QR + rows[i]] = l_run[i]; }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(&counters[hp], 1);
    sm.last = (prev == KSPLIT - 1);
    if (sm.last) counters[hp] = 0;
  }
  __syncthreads();
  if (!sm.last) return;
  __threadfence();

  // Merge the KSPLIT partials, round to fp16 (the chain's PV output precision) and quantize.
  const float* base = ws + static_cast<size_t>(hp * KSPLIT) * PART;
  const int D = NH * HD;
  const int nitems = 2 * S * (HD / 16);
  for (int it = tid; it < nitems; it += THREADS) {
    const int blk = it & 15, t = (it >> 4) % S, h = (it >> 4) / S;
    const int r = h * 16 + t;
    float M = -1e30f;
    #pragma unroll
    for (int s = 0; s < KSPLIT; ++s) M = fmaxf(M, base[s * PART + QR * HD + r]);
    float L = 0.f, vals[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) vals[j] = 0.f;
    #pragma unroll
    for (int s = 0; s < KSPLIT; ++s) {
      const float w = __expf(base[s * PART + QR * HD + r] - M);
      L += w * base[s * PART + QR * HD + QR + r];
      const float4* op = reinterpret_cast<const float4*>(&base[s * PART + r * HD + blk * 16]);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 v = op[j];
        vals[4 * j] += w * v.x; vals[4 * j + 1] += w * v.y; vals[4 * j + 2] += w * v.z; vals[4 * j + 3] += w * v.w;
      }
    }
    const float invL = 1.f / L;
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      vals[j] = __half2float(__float2half_rn(vals[j] * invL));
      amax = fmaxf(amax, fabsf(vals[j]));
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(desired);
    const float inv_bs = 1.f / static_cast<float>(bs_q);
    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint8_t lo = fp32_to_e2m1_ref(vals[2 * p] * inv_bs);
      const uint8_t hi = fp32_to_e2m1_ref(vals[2 * p + 1] * inv_bs);
      ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    const int col0 = (2 * hp + h) * HD + blk * 16;
    *reinterpret_cast<uint2*>(packed + static_cast<size_t>(t) * (D / 2) + (col0 >> 1)) = out;
    const int kb = col0 >> 4;
    const size_t sidx = (static_cast<size_t>(t >> 7) * (D / 64) + (kb >> 2)) * 512 + (t & 31) * 16 + ((t >> 5) & 3) * 4 + (kb & 3);
    sfa[sidx] = *reinterpret_cast<uint8_t*>(&bs_q);
  }
}

}  // namespace

constexpr int KSPLIT_MAX = 16;
size_t attn_mqa_s16_fp4out_ws_bytes(int NH) {
  const size_t part = static_cast<size_t>(QR * HD + 2 * QR) * sizeof(float);
  return static_cast<size_t>(NH / 2) * KSPLIT_MAX * part + static_cast<size_t>(NH / 2) * sizeof(int);
}

template <int KSPLIT, int STAGES>
static int launch_variant(const __half* Q, const __half* K, const __half* V, float* wsf, int* counters,
                          uint8_t* packed, uint8_t* sfa, int S, int T, int NH, float scale, int dbg, cudaStream_t stream) {
  const int per_split = ((T + KSPLIT - 1) / KSPLIT + TK - 1) / TK * TK;
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(attn_mqa_s16_fp4out_kernel<KSPLIT, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(Smem<STAGES>))) != cudaSuccess) return -2;
    attr_set = true;
  }
  launch_maybe_pdl(attn_mqa_s16_fp4out_kernel<KSPLIT, STAGES>, dim3(KSPLIT, NH / 2), dim3(THREADS), sizeof(Smem<STAGES>), stream,
                   Q, K, V, wsf, counters, packed, sfa, S, T, NH, scale, per_split, dbg);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int attn_mqa_s16_fp4out(const void* Q, const void* K, const void* V, void* ws,
                        void* packed, void* sfa, int S, int T, int NH, int HD_,
                        float attn_scale, cudaStream_t stream, int variant, int dbg) {
  if (HD_ != HD || S < 1 || S > 16 || NH < 2 || (NH & 1) || T < 1) return -1;
  if (!Q || !K || !V || !ws || !packed || !sfa) return -1;
  float* wsf = static_cast<float*>(ws);
  int* counters = reinterpret_cast<int*>(wsf + static_cast<size_t>(NH / 2) * KSPLIT_MAX * (QR * HD + 2 * QR));
  const __half* q = static_cast<const __half*>(Q);
  const __half* k = static_cast<const __half*>(K);
  const __half* v = static_cast<const __half*>(V);
  uint8_t* p = static_cast<uint8_t*>(packed);
  uint8_t* sf = static_cast<uint8_t*>(sfa);
  switch (variant) {
    case 0: return launch_variant<4, 2>(q, k, v, wsf, counters, p, sf, S, T, NH, attn_scale, dbg, stream);
    case 1: return launch_variant<8, 2>(q, k, v, wsf, counters, p, sf, S, T, NH, attn_scale, dbg, stream);
    case 2: return launch_variant<16, 2>(q, k, v, wsf, counters, p, sf, S, T, NH, attn_scale, dbg, stream);
    case 3: return launch_variant<8, 3>(q, k, v, wsf, counters, p, sf, S, T, NH, attn_scale, dbg, stream);
    case 4: return launch_variant<4, 3>(q, k, v, wsf, counters, p, sf, S, T, NH, attn_scale, dbg, stream);
    default: return -3;
  }
}

}  // namespace fp4
}  // namespace flash_rt
