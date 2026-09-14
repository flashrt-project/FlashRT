// See attn_mqa_fused_fp4out.cuh.
#include "fused_fp4/attn_mqa_fused_fp4out.cuh"
#include "fused_fp4/pdl.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr int HD = 256;                 // head dim
constexpr int TK = 64;                  // keys per tile
constexpr int PITCH = HD + 8;           // halves per smem row (528 B: ldmatrix conflict-free)
constexpr int KSPLIT = 16;              // CTAs (key-range splits); all co-resident on Thor's 20 SMs
constexpr int RMAX = 128;               // NH*S <= 128 query rows
constexpr size_t PART_BYTES = static_cast<size_t>(RMAX) * HD * 2 + 2 * RMAX * 4;  // per split: fp16 O rows, fp32 m, fp32 l
constexpr unsigned FULL = 0xffffffffu;

template <int MT, int STAGES>
struct __align__(16) Smem {
  __half q[MT * 16 * PITCH];
  __half k[STAGES][TK * PITCH];
  __half v[STAGES][TK * PITCH];
  union {
    float exch[MT * 2][32 * 32];        // per warp [value][lane]: partial scores of one dim-half; merge scratch afterwards
  } u;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(dst)), "l"(src) : "memory");
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
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
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
// Same per-pair arithmetic as qkv_split_rope_kvcache_fp16_vec (rope_vec.cu).
__device__ __forceinline__ int4 rope_rotate4(int4 x_raw, int4 cs_raw) {
  const __half2* x = reinterpret_cast<const __half2*>(&x_raw);
  const __half2* cs = reinterpret_cast<const __half2*>(&cs_raw);
  int4 out_raw;
  __half2* out = reinterpret_cast<__half2*>(&out_raw);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xf = __half22float2(x[i]);
    const float2 cf = __half22float2(cs[i]);
    const float x0 = xf.x, x1 = xf.y, c = cf.x, sn = cf.y;
    out[i] = __floats2half2_rn(x0 * c - x1 * sn, x1 * c + x0 * sn);
  }
  return out_raw;
}

// Query rows are packed r = h*S + s (all heads share the KV head, so one CTA
// covers every head for its key range). Warp w owns m-tile (w % MT) and the
// dim-half (w / MT): QK^T is computed per dim-half and the two halves are
// summed through shared memory; PV keeps the same dim-half split so the
// (16 x 256) accumulator stays at 64 registers per thread.
template <int MT, int STAGES>
__global__ void __launch_bounds__(MT * 64, 1)
attn_mqa_fused_fp4out_kernel(const __half* __restrict__ qkv, const __half* __restrict__ rope,
                             __half* __restrict__ Kc, __half* __restrict__ Vc,
                             float* __restrict__ ws, unsigned* __restrict__ counter,
                             uint8_t* __restrict__ packed, uint8_t* __restrict__ sfa,
                             int S, int enc_seq, int NH, int qkv_stride, float scale, int per_split, int dbg) {
  flashrt_pdl_wait_and_trigger();
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<MT, STAGES>& sm = *reinterpret_cast<Smem<MT, STAGES>*>(smem_raw);
  constexpr int NT = MT * 64;
  constexpr int R = MT * 16;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int mt = warp % MT, dh = warp / MT;
  const int T = enc_seq + S;
  const int Rv = NH * S;
  const int split = blockIdx.x;
  const int k0 = split * per_split;
  const int nkeys = min(per_split, T - k0);
  const int ntiles = (dbg & 4) ? 0 : (nkeys > 0 ? (nkeys + TK - 1) / TK : 0);

  // ---- K/V tile loader: cache rows through cp.async, fresh rows RoPE'd from qkv (and appended to the cache)
  auto load_tile = [&](int tile, int stage) {
    const int kbase = k0 + tile * TK;
    for (int c = tid; c < TK * (HD / 8); c += NT) {
      const int j = c / (HD / 8), c8 = c - j * (HD / 8);
      const int key = kbase + j;
      __half* kd = &sm.k[stage][j * PITCH + c8 * 8];
      __half* vd = &sm.v[stage][j * PITCH + c8 * 8];
      if (key < enc_seq) {
        cp_async16(kd, Kc + static_cast<size_t>(key) * HD + c8 * 8);
        cp_async16(vd, Vc + static_cast<size_t>(key) * HD + c8 * 8);
      } else if (key < T) {
        const int s = key - enc_seq;
        const __half* row = qkv + static_cast<size_t>(s) * qkv_stride + NH * HD;
        const int4 cs = *reinterpret_cast<const int4*>(rope + static_cast<size_t>(s) * HD + c8 * 8);
        const int4 rk = rope_rotate4(*reinterpret_cast<const int4*>(row + c8 * 8), cs);
        const int4 xv = *reinterpret_cast<const int4*>(row + HD + c8 * 8);
        *reinterpret_cast<int4*>(kd) = rk;
        *reinterpret_cast<int4*>(vd) = xv;
        *reinterpret_cast<int4*>(Kc + static_cast<size_t>(key) * HD + c8 * 8) = rk;
        *reinterpret_cast<int4*>(Vc + static_cast<size_t>(key) * HD + c8 * 8) = xv;
      } else {
        *reinterpret_cast<int4*>(kd) = make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(vd) = make_int4(0, 0, 0, 0);
      }
    }
    cp_commit();
  };

  // K/V tiles first so their DRAM latency overlaps the Q staging
  #pragma unroll
  for (int t = 0; t < STAGES; ++t) if (t < ntiles) load_tile(t, t);

  // ---- Q rows (RoPE'd on the fly) -> smem; every load is issued before the first store
  if (!(dbg & 16)) {
    constexpr int NQ = R * (HD / 8);
    constexpr int PER = (NQ + NT - 1) / NT;
    int4 xs[PER], css[PER];
    #pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int c = tid + k * NT;
      const int r = c / (HD / 8), c8 = c - r * (HD / 8);
      xs[k] = make_int4(0, 0, 0, 0); css[k] = make_int4(0, 0, 0, 0);
      if (c < NQ && r < Rv) {
        const int h = r / S, sq = r - h * S;
        xs[k] = *reinterpret_cast<const int4*>(qkv + static_cast<size_t>(sq) * qkv_stride + h * HD + c8 * 8);
        css[k] = *reinterpret_cast<const int4*>(rope + static_cast<size_t>(sq) * HD + c8 * 8);
      }
    }
    #pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int c = tid + k * NT;
      const int r = c / (HD / 8), c8 = c - r * (HD / 8);
      if (c < NQ) {
        const int4 val = (r < Rv) ? rope_rotate4(xs[k], css[k]) : make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(&sm.q[r * PITCH + c8 * 8]) = val;
      }
    }
  }

  float m_run[2] = {-1e30f, -1e30f}, l_run[2] = {0.f, 0.f};
  float o_acc[16][4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { o_acc[i][0] = 0.f; o_acc[i][1] = 0.f; o_acc[i][2] = 0.f; o_acc[i][3] = 0.f; }

  for (int tile = 0; tile < ntiles; ++tile) {
    const int stage = tile % STAGES;
    if (STAGES >= 2 && tile + 1 < ntiles) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    if (dbg & 1) { __syncthreads(); if (tile + STAGES < ntiles) load_tile(tile + STAGES, stage); continue; }

    // ---- partial scores over this warp's 128 dims: 16 rows x 64 keys
    float s_acc[8][4];
    #pragma unroll
    for (int nb = 0; nb < 8; ++nb) { s_acc[nb][0] = 0.f; s_acc[nb][1] = 0.f; s_acc[nb][2] = 0.f; s_acc[nb][3] = 0.f; }
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t a[4];
      ldsm_x4(a, &sm.q[(mt * 16 + (lane & 15)) * PITCH + dh * 128 + kk * 16 + (lane >> 4) * 8]);
      #pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        uint32_t b[2];
        ldsm_x2(b, &sm.k[stage][(nb * 8 + (lane & 7)) * PITCH + dh * 128 + kk * 16 + ((lane >> 3) & 1) * 8]);
        mma16816(s_acc[nb], a, b);
      }
    }
    // ---- sum the two dim-halves
    {
      float* ex = sm.u.exch[warp];
      #pragma unroll
      for (int nb = 0; nb < 8; ++nb)
        #pragma unroll
        for (int i = 0; i < 4; ++i) ex[(nb * 4 + i) * 32 + lane] = s_acc[nb][i];
    }
    __syncthreads();
    {
      const float* ex = sm.u.exch[(dh ^ 1) * MT + mt];
      #pragma unroll
      for (int nb = 0; nb < 8; ++nb)
        #pragma unroll
        for (int i = 0; i < 4; ++i) s_acc[nb][i] += ex[(nb * 4 + i) * 32 + lane];
    }
    // ---- fp16 logits (as the cuBLAS chain stores them), key mask, online softmax
    const int kbase = k0 + tile * TK;
    float mx[2] = {-1e30f, -1e30f};
    #pragma unroll
    for (int nb = 0; nb < 8; ++nb)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int key = kbase + nb * 8 + 2 * (lane & 3) + (i & 1);
        float v = -1e30f;
        if (key < T) v = __half2float(__float2half_rn(s_acc[nb][i] * scale));
        s_acc[nb][i] = v;
        mx[i >> 1] = fmaxf(mx[i >> 1], v);
      }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 2));
    }
    float m_new[2], alpha[2], lsum[2] = {0.f, 0.f};
    #pragma unroll
    for (int h = 0; h < 2; ++h) { m_new[h] = fmaxf(m_run[h], mx[h]); alpha[h] = __expf(m_run[h] - m_new[h]); }
    #pragma unroll
    for (int nb = 0; nb < 8; ++nb)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float p = (s_acc[nb][i] > -1e29f) ? __expf(s_acc[nb][i] - m_new[i >> 1]) : 0.f;
        s_acc[nb][i] = p;
        lsum[i >> 1] += p;
      }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      lsum[h] += __shfl_xor_sync(FULL, lsum[h], 1);
      lsum[h] += __shfl_xor_sync(FULL, lsum[h], 2);
      l_run[h] = l_run[h] * alpha[h] + lsum[h];
      m_run[h] = m_new[h];
    }
    #pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
      o_acc[nd][0] *= alpha[0]; o_acc[nd][1] *= alpha[0]; o_acc[nd][2] *= alpha[1]; o_acc[nd][3] *= alpha[1];
    }
    // ---- O += P V over this warp's 128 dims
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      uint32_t a[4];
      a[0] = pack2(s_acc[2 * kp][0], s_acc[2 * kp][1]);
      a[1] = pack2(s_acc[2 * kp][2], s_acc[2 * kp][3]);
      a[2] = pack2(s_acc[2 * kp + 1][0], s_acc[2 * kp + 1][1]);
      a[3] = pack2(s_acc[2 * kp + 1][2], s_acc[2 * kp + 1][3]);
      #pragma unroll
      for (int nd = 0; nd < 16; ++nd) {
        uint32_t b[2];
        ldsm_x2_trans(b, &sm.v[stage][(kp * 16 + (lane & 15)) * PITCH + dh * 128 + nd * 8]);
        mma16816(o_acc[nd], a, b);
      }
    }
    __syncthreads();   // everyone is done with this stage (and the exchange buffer)
    if (tile + STAGES < ntiles) load_tile(tile + STAGES, stage);
  }

  // ---- split partial (O unnormalized, m, l)
  unsigned char* wsb = reinterpret_cast<unsigned char*>(ws) + static_cast<size_t>(split) * PART_BYTES;
  __half* wo = reinterpret_cast<__half*>(wsb);
  float* wm = reinterpret_cast<float*>(wsb + static_cast<size_t>(RMAX) * HD * 2);
  float* wl = wm + RMAX;
  if (!(dbg & 8))
  #pragma unroll
  for (int nd = 0; nd < 16; ++nd) {
    const int r0 = mt * 16 + (lane >> 2), c = dh * 128 + nd * 8 + 2 * (lane & 3);
    *reinterpret_cast<__half2*>(&wo[r0 * HD + c]) = __floats2half2_rn(o_acc[nd][0], o_acc[nd][1]);
    *reinterpret_cast<__half2*>(&wo[(r0 + 8) * HD + c]) = __floats2half2_rn(o_acc[nd][2], o_acc[nd][3]);
  }
  if (dh == 0 && (lane & 3) == 0) {
    const int r0 = mt * 16 + (lane >> 2);
    wm[r0] = m_run[0]; wm[r0 + 8] = m_run[1];
    wl[r0] = l_run[0]; wl[r0 + 8] = l_run[1];
  }
  __threadfence();
  __syncthreads();
  if (dbg & 2) return;
  // ---- grid barrier over the KSPLIT co-resident CTAs (monotonic generation counter, no reset)
  if (tid == 0) {
    const unsigned a = atomicAdd(counter, 1u);
    const unsigned target = (a / KSPLIT + 1u) * KSPLIT;
    while (ld_acquire(counter) < target) __nanosleep(64);
  }
  __syncthreads();
  __threadfence();

  // ---- merge: this CTA owns rows [r0, r0+nrow); every partial-row load is issued up front,
  // then one reduction pass, then the fp16 rounding (the chain's PV output) and the quantize.
  if (dbg & 32) return;
  constexpr int RPC_MAX = (RMAX + KSPLIT - 1) / KSPLIT;   // 8
  constexpr int NS = (KSPLIT + MT - 1) / MT;               // partials per thread group
  const int RPC = (Rv + KSPLIT - 1) / KSPLIT;
  const int D = NH * HD;
  const int r0 = split * RPC;
  const int nrow = min(RPC, Rv - r0);
  if (nrow <= 0) return;
  float* stats = reinterpret_cast<float*>(sm.q);            // [RPC_MAX][2][KSPLIT] (q is dead now)
  float* row_out = stats + RPC_MAX * 2 * KSPLIT;             // [RPC_MAX][HD]
  float (*red)[64][4] = reinterpret_cast<float (*)[64][4]>(&sm.u.exch[0][0]);   // [RPC_MAX*MT][64][4]
  static_assert(sizeof(sm.u.exch) >= RPC_MAX * MT * 64 * 4 * sizeof(float), "exch too small for the merge");
  static_assert(sizeof(sm.q) >= (RPC_MAX * 2 * KSPLIT + RPC_MAX * HD) * sizeof(float), "q too small for the merge");
  const unsigned char* wsall = reinterpret_cast<const unsigned char*>(ws);
  if (tid < nrow * KSPLIT) {
    const int i = tid / KSPLIT, sp = tid - i * KSPLIT;
    const float* pm = reinterpret_cast<const float*>(wsall + static_cast<size_t>(sp) * PART_BYTES + static_cast<size_t>(RMAX) * HD * 2);
    stats[(i * 2 + 0) * KSPLIT + sp] = pm[r0 + i];
    stats[(i * 2 + 1) * KSPLIT + sp] = pm[RMAX + r0 + i];
  }
  __syncthreads();
  const int d4 = tid & 63, sg = tid >> 6;
  #pragma unroll
  for (int chunk = 0; chunk < RPC_MAX; chunk += 4) {
    float4 v[4][NS];
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii)
      #pragma unroll
      for (int j = 0; j < NS; ++j) {
        const int i = chunk + ii, sp = sg + j * MT;
        v[ii][j] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < nrow && sp < KSPLIT) {
          const __half* po = reinterpret_cast<const __half*>(wsall + static_cast<size_t>(sp) * PART_BYTES);
          const uint2 u = *reinterpret_cast<const uint2*>(&po[(r0 + i) * HD + d4 * 4]);
          const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
          const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
          v[ii][j] = make_float4(f0.x, f0.y, f1.x, f1.y);
        }
      }
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      const int i = chunk + ii;
      if (i < nrow) {
        const float* mr = stats + (i * 2) * KSPLIT;
        float M = -1e30f;
        #pragma unroll
        for (int sp = 0; sp < KSPLIT; ++sp) M = fmaxf(M, mr[sp]);
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int j = 0; j < NS; ++j) {
          const int sp = sg + j * MT;
          if (sp < KSPLIT) {
            const float w = __expf(mr[sp] - M);
            acc.x += w * v[ii][j].x; acc.y += w * v[ii][j].y; acc.z += w * v[ii][j].z; acc.w += w * v[ii][j].w;
          }
        }
        *reinterpret_cast<float4*>(&red[i * MT + sg][d4][0]) = acc;
      }
    }
  }
  __syncthreads();
  for (int i = tid >> 6; i < nrow; i += MT) {
    const float* mr = stats + (i * 2) * KSPLIT;
    const float* lr = mr + KSPLIT;
    float M = -1e30f;
    #pragma unroll
    for (int sp = 0; sp < KSPLIT; ++sp) M = fmaxf(M, mr[sp]);
    float L = 0.f;
    #pragma unroll
    for (int sp = 0; sp < KSPLIT; ++sp) L += __expf(mr[sp] - M) * lr[sp];
    float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int g = 0; g < MT; ++g) {
      const float4 v = *reinterpret_cast<const float4*>(&red[i * MT + g][d4][0]);
      t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w;
    }
    const float invL = 1.f / L;
    row_out[i * HD + d4 * 4 + 0] = __half2float(__float2half_rn(t.x * invL));
    row_out[i * HD + d4 * 4 + 1] = __half2float(__float2half_rn(t.y * invL));
    row_out[i * HD + d4 * 4 + 2] = __half2float(__float2half_rn(t.z * invL));
    row_out[i * HD + d4 * 4 + 3] = __half2float(__float2half_rn(t.w * invL));
  }
  __syncthreads();
  for (int qb = tid; qb < nrow * (HD / 16); qb += NT) {
    const int i = qb >> 4, blk = qb & 15;
    const int r = r0 + i;
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) { vals[j] = row_out[i * HD + blk * 16 + j]; amax = fmaxf(amax, fabsf(vals[j])); }
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
    const int h = r / S, t = r - h * S;
    const int col0 = h * HD + blk * 16;
    *reinterpret_cast<uint2*>(packed + static_cast<size_t>(t) * (D / 2) + (col0 >> 1)) = out;
    const int kb = col0 >> 4;
    const size_t sidx = (static_cast<size_t>(t >> 7) * (D / 64) + (kb >> 2)) * 512 + (t & 31) * 16 + ((t >> 5) & 3) * 4 + (kb & 3);
    sfa[sidx] = *reinterpret_cast<uint8_t*>(&bs_q);
  }
}

template <int MT, int STAGES>
int launch_fused(const __half* qkv, const __half* rope, __half* Kc, __half* Vc, float* wsf, unsigned* counter,
                 uint8_t* packed, uint8_t* sfa, int S, int enc_seq, int NH, int qkv_stride, float scale,
                 int per_split, int dbg, cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(attn_mqa_fused_fp4out_kernel<MT, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(Smem<MT, STAGES>))) != cudaSuccess) return -2;
    attr_set = true;
  }
  launch_maybe_pdl(attn_mqa_fused_fp4out_kernel<MT, STAGES>, dim3(KSPLIT), dim3(MT * 64), sizeof(Smem<MT, STAGES>), stream,
                   qkv, rope, Kc, Vc, wsf, counter, packed, sfa, S, enc_seq, NH, qkv_stride, scale, per_split, dbg);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace

size_t attn_mqa_fused_ws_bytes() {
  return static_cast<size_t>(KSPLIT) * PART_BYTES + 64;
}

int attn_mqa_fused_fp4out(const void* qkv, const void* rope, void* Kc, void* Vc, void* ws,
                          void* packed, void* sfa, int S, int enc_seq, int NH, int HD_,
                          int qkv_stride, float attn_scale, cudaStream_t stream, int dbg) {
  if (HD_ != HD || S < 1 || S > 16 || NH < 1 || NH * S > RMAX || enc_seq < 0) return -1;
  if (qkv_stride < NH * HD + 2 * HD || (qkv_stride & 7)) return -1;
  if (!qkv || !rope || !Kc || !Vc || !ws || !packed || !sfa) return -1;
  const int T = enc_seq + S;
  const int per_split = ((T + KSPLIT - 1) / KSPLIT + TK - 1) / TK * TK;
  float* wsf = static_cast<float*>(ws);
  unsigned* counter = reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(ws) + static_cast<size_t>(KSPLIT) * PART_BYTES);
  const __half* q = static_cast<const __half*>(qkv);
  const __half* rp = static_cast<const __half*>(rope);
  __half* kc = static_cast<__half*>(Kc);
  __half* vc = static_cast<__half*>(Vc);
  uint8_t* p = static_cast<uint8_t*>(packed);
  uint8_t* sf = static_cast<uint8_t*>(sfa);
  const int MT = (NH * S + 15) / 16;
  if (MT <= 5) return launch_fused<5, 2>(q, rp, kc, vc, wsf, counter, p, sf, S, enc_seq, NH, qkv_stride, attn_scale, per_split, dbg, stream);
  return launch_fused<8, 1>(q, rp, kc, vc, wsf, counter, p, sf, S, enc_seq, NH, qkv_stride, attn_scale, per_split, dbg, stream);
}

}  // namespace fp4
}  // namespace flash_rt
