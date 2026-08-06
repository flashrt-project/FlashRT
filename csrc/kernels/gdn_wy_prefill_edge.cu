#include "gdn_wy_prefill_edge.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kChunk = 64;
constexpr float kEps = 1e-6f;    // matches the sequential scan's l2 eps

// Butterfly order, the same summation order the sibling WY normalisation uses.
// Reduction order decides the low bits here, so this is not interchangeable
// with the shuffle-down helper in common.cuh.
template <int kHD>
__device__ __forceinline__ float wy_block_sum(float val, float* smem) {
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (lane < (kHD / 32)) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

// One block per (unique k-head, token). The block reduces both q and k over
// head_dim, writes the unique-head k, and scatters q into the qk_group v-head
// slots of the packed buffer -- so the GQA broadcast never materialises.
//
// The grid covers chunks * 64 tokens rather than S, so the threads past the
// end of the sequence are the ones that zero the packed tail.
template <int kHD>
__global__ void gdn_wy_norm_pack_q_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ k_l2,
    __nv_bfloat16* __restrict__ q_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group)
{
  const int t = threadIdx.x;
  const int h = blockIdx.x;              // unique k-head
  const int s = blockIdx.y;              // token, may run past S into the pad
  if (t >= kHD || h >= num_k_heads) return;

  const int chunk = s / kChunk;
  const int tt = s - chunk * kChunk;

  if (s >= S) {
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int r = 0; r < qk_group; ++r) {
      const int vh = h * qk_group + r;
      q_pack[((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + tt)
             * kHD + t] = zero;
    }
    return;
  }

  // q and k arrive GQA-broadcast, so the group leader carries the value.
  const size_t src = (static_cast<size_t>(s) * num_v_heads + h * qk_group)
                     * kHD + t;
  const float qv = static_cast<float>(q[src]);
  const float kv = static_cast<float>(k[src]);

  __shared__ float scratch[32];
  const float q_sq = wy_block_sum<kHD>(qv * qv, scratch);
  __syncthreads();          // scratch is reused by the second reduction
  const float k_sq = wy_block_sum<kHD>(kv * kv, scratch);
  __syncthreads();

  const __nv_bfloat16 q_norm = __float2bfloat16(qv * rsqrtf(q_sq + kEps));
  const __nv_bfloat16 k_norm = __float2bfloat16(kv * rsqrtf(k_sq + kEps));

  k_l2[(static_cast<size_t>(s) * num_k_heads + h) * kHD + t] = k_norm;

  for (int r = 0; r < qk_group; ++r) {
    const int vh = h * qk_group + r;
    q_pack[((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + tt)
           * kHD + t] = q_norm;
  }
}

// One thread per (chunk, v-head): 64 dependent adds, chunks * num_v_heads of
// them in flight. The sibling path runs one block of num_v_heads threads
// serially over the whole sequence, which is fine for a decode step and two
// orders of magnitude off for a prefill.
__global__ void gdn_wy_cumsum_g_chunk_kernel(
    const __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ g_cumsum,
    int S,
    int num_v_heads,
    int chunks)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunks * num_v_heads) return;
  const int chunk = idx / num_v_heads;
  const int vh = idx - chunk * num_v_heads;

  const int s0 = chunk * kChunk;
  const int s1 = min(s0 + kChunk, S);
  float acc = 0.0f;
  for (int s = s0; s < s1; ++s) {
    const size_t off = static_cast<size_t>(s) * num_v_heads + vh;
    acc += static_cast<float>(g[off]);
    g_cumsum[off] = __float2bfloat16(acc);
  }
}

__global__ void gdn_wy_pack_v_kernel(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ v_pack,
    int S,
    int num_v_heads,
    int head_dim)
{
  const int t = threadIdx.x;
  const int vh = blockIdx.x;
  const int s = blockIdx.y;
  if (t >= head_dim || vh >= num_v_heads) return;

  const int chunk = s / kChunk;
  const int tt = s - chunk * kChunk;
  const size_t dst =
      ((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + tt)
      * head_dim + t;
  v_pack[dst] = (s < S)
      ? v[(static_cast<size_t>(s) * num_v_heads + vh) * head_dim + t]
      : __float2bfloat16(0.0f);
}

}  // namespace

void gdn_wy_norm_pack_q_cumsum_edge_bf16(
    const void* q,
    const void* k,
    const void* g,
    void*       k_l2,
    void*       q_pack,
    void*       g_cumsum,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || head_dim != 128) return;
  const int chunks = (S + kChunk - 1) / kChunk;

  gdn_wy_norm_pack_q_kernel<128>
      <<<dim3(num_k_heads, chunks * kChunk), 128, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q),
          reinterpret_cast<const __nv_bfloat16*>(k),
          reinterpret_cast<__nv_bfloat16*>(k_l2),
          reinterpret_cast<__nv_bfloat16*>(q_pack),
          S, num_k_heads, num_v_heads, qk_group);

  const int total = chunks * num_v_heads;
  gdn_wy_cumsum_g_chunk_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum),
      S, num_v_heads, chunks);
}

void gdn_wy_pack_v_edge_bf16(
    const void* v,
    void*       v_pack,
    int S,
    int num_v_heads,
    int head_dim,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int chunks = (S + kChunk - 1) / kChunk;
  gdn_wy_pack_v_kernel<<<dim3(num_v_heads, chunks * kChunk), head_dim, 0,
                         stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(v_pack),
      S, num_v_heads, head_dim);
}

}  // namespace kernels
}  // namespace flash_rt
