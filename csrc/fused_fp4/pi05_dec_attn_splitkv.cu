// See pi05_dec_attn_splitkv.cuh.
//
// Grid = (key splits, heads). Each CTA scores its head's 10 queries against a
// CHUNK of keys with wmma, keeps the partial (max, sum, unnormalised O) and
// the last CTA to finish a head merges the partials and writes the NVFP4
// activation for that head (threadfence-reduction pattern, one launch).
#include "fused_fp4/pi05_dec_attn_splitkv.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_rt {
namespace fp4 {
namespace {

using namespace nvcuda;
using CfgSF = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int S_Q       = 10;         // query tokens
constexpr int NHEADS    = 8;
constexpr int HD        = 256;
constexpr int CHUNK     = 64;         // keys per CTA
constexpr int MAX_SPLIT = 32;         // S_kv <= 2048
constexpr int LDS       = HD + 8;     // halves
constexpr int LDSF      = CHUNK + 4;  // fp32 logits
constexpr int LDP       = CHUNK + 8;  // fp16 probs
constexpr int LDO       = HD + 4;     // fp32 O staging
constexpr int THREADS   = 128;
constexpr int NWARP     = THREADS / 32;

struct __align__(32) Smem {
  __half q[16 * LDS];
  __half k[CHUNK * LDS];
  __half v[CHUNK * LDS];
  float  s[16 * LDSF];
  __half p[16 * LDP];
};
static_assert(sizeof(float) * 16 * LDO <= sizeof(__half) * CHUNK * LDS,
              "O staging must fit in the K region it aliases");

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool valid) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_size = valid ? 16 : 0;   // src_size 0 => zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(gmem_src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

struct Ws {
  float* o_part;    // [MAX_SPLIT][NHEADS][S_Q][HD]
  float* ml_part;   // [MAX_SPLIT][NHEADS][S_Q][2]
  int*   counters;  // [NHEADS]
};

__device__ __forceinline__ uint8_t fp32_to_e2m1_dk(float x) {
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

template <class LayoutSF>
__global__ void __launch_bounds__(THREADS, 3)
dec_attn_splitkv_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, Ws ws, int S_kv, int nsplit, float scale,
    uint2* __restrict__ dst_packed, uint8_t* __restrict__ dst_sfa,
    LayoutSF layout) {
  extern __shared__ __align__(32) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int key0 = split * CHUNK;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  constexpr int HD8 = HD / 8;
  __shared__ int s_is_last;

  // Q rows of this head (strided by NHEADS*HD in the (S, NH, HD) buffer).
  const int4* q4 = reinterpret_cast<const int4*>(Q);
  const int4* k4 = reinterpret_cast<const int4*>(K);
  const int4* v4 = reinterpret_cast<const int4*>(V);
  #pragma unroll
  for (int i = tid; i < 16 * HD8; i += THREADS) {
    const int r = i / HD8, c = i - r * HD8;
    const bool valid = r < S_Q;
    const int4* src = q4 + (valid ? (static_cast<size_t>(r) * NHEADS + head) * HD8 + c : 0);
    cp_async_16(&sm.q[r * LDS + c * 8], src, valid);
  }
  #pragma unroll 8
  for (int i = tid; i < CHUNK * HD8; i += THREADS) {
    const int r = i / HD8, c = i - r * HD8;
    const int key = key0 + r;
    const bool valid = key < S_kv;
    const size_t off = valid ? static_cast<size_t>(key) * HD8 + c : 0;
    cp_async_16(&sm.k[r * LDS + c * 8], k4 + off, valid);
    cp_async_16(&sm.v[r * LDS + c * 8], v4 + off, valid);
  }
  cp_async_wait_all();
  __syncthreads();

  // S = Q K^T (16 x CHUNK): one n-tile per warp.
  for (int n = warp; n < CHUNK / 16; n += NWARP) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    #pragma unroll 4
    for (int kk = 0; kk < HD; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::load_matrix_sync(a, &sm.q[kk], LDS);
      wmma::load_matrix_sync(b, &sm.k[n * 16 * LDS + kk], LDS);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(&sm.s[n * 16], acc, LDSF, wmma::mem_row_major);
  }
  __syncthreads();

  // Row softmax over the chunk: rows < S_Q real, rows >= S_Q zero.
  for (int r = warp; r < 16; r += NWARP) {
    if (r < S_Q) {
      float v[CHUNK / 32];
      float mx = -CUDART_INF_F;
      #pragma unroll
      for (int j = 0; j < CHUNK / 32; ++j) {
        const int c = lane + 32 * j;
        const bool valid = (key0 + c) < S_kv;
        v[j] = valid ? sm.s[r * LDSF + c] * scale : -CUDART_INF_F;
        mx = fmaxf(mx, v[j]);
      }
      #pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float l = 0.f;
      #pragma unroll
      for (int j = 0; j < CHUNK / 32; ++j) {
        const int c = lane + 32 * j;
        const float p = (mx > -CUDART_INF_F && (key0 + c) < S_kv) ? __expf(v[j] - mx) : 0.f;
        l += p;
        sm.p[r * LDP + c] = __float2half(p);
      }
      #pragma unroll
      for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      if (lane == 0) {
        float* ml = ws.ml_part + ((static_cast<size_t>(split) * NHEADS + head) * S_Q + r) * 2;
        ml[0] = mx; ml[1] = l;
      }
    } else {
      #pragma unroll
      for (int j = 0; j < CHUNK / 32; ++j) sm.p[r * LDP + lane + 32 * j] = __float2half(0.f);
    }
  }
  __syncthreads();

  // O = P V (16 x HD): 16 n-tiles, 4 per warp, CHUNK/16 k-steps each.
  float* o_stage = reinterpret_cast<float*>(sm.k);   // K is dead here
  for (int n = warp; n < HD / 16; n += NWARP) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    #pragma unroll
    for (int kk = 0; kk < CHUNK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
      wmma::load_matrix_sync(a, &sm.p[kk], LDP);
      wmma::load_matrix_sync(b, &sm.v[kk * LDS + n * 16], LDS);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(o_stage + n * 16, acc, LDO, wmma::mem_row_major);
  }
  __syncthreads();

  // Coalesced partial store, rows < S_Q only.
  float* op = ws.o_part + (static_cast<size_t>(split) * NHEADS + head) * S_Q * HD;
  for (int i = tid; i < S_Q * (HD / 4); i += THREADS) {
    const int r = i / (HD / 4), c4 = i - r * (HD / 4);
    reinterpret_cast<float4*>(op)[i] =
        *reinterpret_cast<const float4*>(o_stage + r * LDO + c4 * 4);
  }

  // Last CTA of this head merges.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int old = atomicAdd(&ws.counters[head], 1);
    s_is_last = (old == nsplit - 1);
  }
  __syncthreads();
  if (!s_is_last) return;
  __threadfence();

  const int nblk_head = HD / 16;               // 16 blocks per row per head
  for (int t = tid; t < S_Q * nblk_head; t += THREADS) {
    const int r = t / nblk_head, bh = t - r * nblk_head;
    const int d0 = bh * 16;
    float M = -CUDART_INF_F;
    for (int i = 0; i < nsplit; ++i)
      M = fmaxf(M, ws.ml_part[((static_cast<size_t>(i) * NHEADS + head) * S_Q + r) * 2]);
    float L = 0.f;
    float acc[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.f;
    for (int i = 0; i < nsplit; ++i) {
      const float* ml = ws.ml_part + ((static_cast<size_t>(i) * NHEADS + head) * S_Q + r) * 2;
      const float mi = ml[0], li = ml[1];
      const float w = (mi > -CUDART_INF_F) ? __expf(mi - M) : 0.f;
      L += w * li;
      const float4* o4 = reinterpret_cast<const float4*>(
          ws.o_part + ((static_cast<size_t>(i) * NHEADS + head) * S_Q + r) * HD + d0);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 x = o4[j];
        acc[4 * j + 0] += w * x.x; acc[4 * j + 1] += w * x.y;
        acc[4 * j + 2] += w * x.z; acc[4 * j + 3] += w * x.w;
      }
    }
    const float inv = 1.f / L;
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      vals[j] = __half2float(__float2half(acc[j] * inv));  // chain wrote fp16
      const float a = fabsf(vals[j]);
      if (a > amax) amax = a;
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float bs_dq = static_cast<float>(bs_q);
    const int col = head * HD + d0;
    dst_sfa[layout(r, col, 0)] = *reinterpret_cast<uint8_t*>(&bs_q);
    const float inv_bs = 1.f / bs_dq;
    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint8_t lo = fp32_to_e2m1_dk(vals[2 * p] * inv_bs);
      const uint8_t hi = fp32_to_e2m1_dk(vals[2 * p + 1] * inv_bs);
      ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    dst_packed[static_cast<size_t>(r) * (NHEADS * HD / 16) + col / 16] = out;
  }
  __syncthreads();
  if (tid == 0) ws.counters[head] = 0;   // ready for the next launch
}

}  // namespace

size_t pi05_dec_attn_splitkv_ws_bytes() {
  return static_cast<size_t>(MAX_SPLIT) * NHEADS * S_Q * HD * sizeof(float)
       + static_cast<size_t>(MAX_SPLIT) * NHEADS * S_Q * 2 * sizeof(float)
       + 256;  // counters (zeroed by the owner at allocation)
}

int pi05_dec_attn_splitkv_fp4(
    const void* Q, const void* K, const void* V, void* workspace,
    void* dst_packed, void* dst_sfa,
    int S, int S_kv, int NH, int HDp, float attn_scale,
    cudaStream_t stream) {
  if (HDp != HD || S != S_Q || NH != NHEADS || S_kv <= 0 ||
      S_kv > MAX_SPLIT * CHUNK)
    return -1;
  if ((reinterpret_cast<uintptr_t>(Q) & 15) || (reinterpret_cast<uintptr_t>(K) & 15) ||
      (reinterpret_cast<uintptr_t>(V) & 15) || (reinterpret_cast<uintptr_t>(workspace) & 15) ||
      (reinterpret_cast<uintptr_t>(dst_packed) & 7))
    return -1;
  const int D = NH * HD;
  auto layout = CfgSF::tile_atom_to_shape_SFA(cute::make_shape(S, 1, D, 1));
  using LayoutT = decltype(layout);
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(dec_attn_splitkv_kernel<LayoutT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(Smem))) != cudaSuccess)
      return -3;
    attr_set = true;
  }
  const int nsplit = (S_kv + CHUNK - 1) / CHUNK;
  Ws ws;
  ws.o_part = reinterpret_cast<float*>(workspace);
  ws.ml_part = ws.o_part + static_cast<size_t>(MAX_SPLIT) * NHEADS * S_Q * HD;
  ws.counters = reinterpret_cast<int*>(ws.ml_part + static_cast<size_t>(MAX_SPLIT) * NHEADS * S_Q * 2);

  dec_attn_splitkv_kernel<LayoutT><<<dim3(nsplit, NHEADS), THREADS, sizeof(Smem), stream>>>(
      reinterpret_cast<const __half*>(Q), reinterpret_cast<const __half*>(K),
      reinterpret_cast<const __half*>(V), ws, S_kv, nsplit, attn_scale,
      reinterpret_cast<uint2*>(dst_packed), reinterpret_cast<uint8_t*>(dst_sfa),
      layout);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
