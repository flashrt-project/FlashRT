#include "moe_route_prefill_edge.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kMaxExperts = 1024;
constexpr int kMaxTopK = 32;
constexpr int kSlotsPerBlock = 256;   // slots a scatter/histogram block owns
constexpr int kRouteThreads = 256;

// One warp per token, the whole row held in registers: PER_LANE experts per
// lane, strided so the row loads coalesced. A block-wide version of this cost
// eight barriers per top-k round -- sixty-four per token -- and ran seventy
// times off the bandwidth the row needs; there is no barrier here at all.
//
// Softmax first, then the top-k renormalised over itself. The full denominator
// cancels between the two, but it is kept because it only cancels exactly in
// exact arithmetic and this seeds a decode that has to reproduce.
//
// Ties go to the lower expert index. bf16 logits make the tail probabilities
// tie outright often enough to matter (6% of slots at 256 experts), and the
// tensor top-k this replaces does not define which of two equal experts it
// ranks first -- so the rank order inside a token's top-k can differ from it
// while the selected set, which is what the grouped GEMM reads, does not.
template <int PER_LANE>
__global__ void route_topk_warp_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ ti,
    float* __restrict__ tw,
    int S,
    int n_experts,
    int topk)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int s = blockIdx.x * (blockDim.x >> 5) + warp;
  if (s >= S) return;                    // whole warp, so the shuffles stay put

  const __nv_bfloat16* row = logits + static_cast<size_t>(s) * n_experts;
  float v[PER_LANE];
  #pragma unroll
  for (int i = 0; i < PER_LANE; ++i) v[i] = static_cast<float>(row[i * 32 + lane]);

  float m = -CUDART_INF_F;
  #pragma unroll
  for (int i = 0; i < PER_LANE; ++i) m = fmaxf(m, v[i]);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));

  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < PER_LANE; ++i) { v[i] = __expf(v[i] - m); sum += v[i]; }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, off);
  #pragma unroll
  for (int i = 0; i < PER_LANE; ++i) v[i] /= sum;

  // Lane r keeps rank r, so the results end up spread one per lane and the
  // write below is a single coalesced store.
  float my_val = 0.0f;
  int my_idx = 0;
  for (int r = 0; r < topk; ++r) {
    float best = -CUDART_INF_F;
    int best_i = n_experts;
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
      const int e = i * 32 + lane;
      if (v[i] > best || (v[i] == best && e < best_i)) { best = v[i]; best_i = e; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, best, off);
      const int oi = __shfl_xor_sync(0xffffffff, best_i, off);
      if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
    }
    if (lane == r) { my_val = best; my_idx = best_i; }
    // Compile-time indices: a computed one would push v[] into local memory.
    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i)
      if (i * 32 + lane == best_i) v[i] = -CUDART_INF_F;
  }

  float tsum = (lane < topk) ? my_val : 0.0f;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    tsum += __shfl_xor_sync(0xffffffff, tsum, off);

  if (lane < topk) {
    const size_t base = static_cast<size_t>(s) * topk;
    ti[base + lane] = my_idx;
    tw[base + lane] = my_val / tsum;
  }
}

// Per-block expert histogram over a fixed slice of slots. The warp-level match
// gives each lane its rank among the lanes of its warp holding the same
// expert, which is what makes the scatter's ordering fall out without atomics.
__global__ void slot_hist_kernel(
    const int* __restrict__ ti,
    int* __restrict__ blk_hist,
    int slots,
    int n_experts)
{
  const int blk = blockIdx.x;
  const int t = threadIdx.x;
  const int slot = blk * kSlotsPerBlock + t;

  extern __shared__ int s_hist[];        // n_experts
  for (int e = t; e < n_experts; e += blockDim.x) s_hist[e] = 0;
  __syncthreads();

  if (slot < slots) atomicAdd(&s_hist[ti[slot]], 1);
  __syncthreads();

  int* out = blk_hist + static_cast<size_t>(blk) * n_experts;
  for (int e = t; e < n_experts; e += blockDim.x) out[e] = s_hist[e];
}

// One block per expert: exclusive scan of that expert's per-block counts, so a
// scatter block knows where its own slots for that expert begin.
//
// Scanned across the block in tiles rather than by one thread in a loop. The
// number of slot-blocks grows with the sequence, so a serial walk here is
// O(S) on a single thread -- invisible at two thousand tokens, and the reason
// the prefill rate fell away past four thousand.
template <int kThreads>
__global__ void expert_block_scan_kernel(
    const int* __restrict__ blk_hist,
    int* __restrict__ blk_off,
    int* __restrict__ counts,
    int n_blocks,
    int n_experts)
{
  const int e = blockIdx.x;
  const int t = threadIdx.x;
  __shared__ int s[kThreads];
  __shared__ int s_carry;
  if (t == 0) s_carry = 0;
  __syncthreads();

  for (int base = 0; base < n_blocks; base += kThreads) {
    const int b = base + t;
    const size_t off = static_cast<size_t>(b) * n_experts + e;
    const int own = (b < n_blocks) ? blk_hist[off] : 0;
    s[t] = own;
    __syncthreads();

    // Hillis-Steele inclusive scan; subtracting own value gives the exclusive
    // one without a second pass.
    for (int d = 1; d < kThreads; d <<= 1) {
      const int add = (t >= d) ? s[t - d] : 0;
      __syncthreads();
      s[t] += add;
      __syncthreads();
    }
    const int tile_total = s[kThreads - 1];
    if (b < n_blocks) blk_off[off] = s_carry + s[t] - own;
    __syncthreads();
    if (t == 0) s_carry += tile_total;
    __syncthreads();
  }
  if (t == 0) counts[e] = s_carry;
}

__global__ void group_off_kernel(
    const int* __restrict__ counts,
    int* __restrict__ group_off,
    int n_experts)
{
  if (threadIdx.x != 0) return;
  int acc = 0;
  for (int e = 0; e < n_experts; ++e) {
    group_off[e] = acc;
    acc += counts[e];
  }
  group_off[n_experts] = acc;
}

// Places every slot at group_off[e] + blk_off[blk][e] + its rank within the
// block. Rank comes from the warp match plus the counts of the earlier warps,
// so two runs on the same routing place the same slot in the same row.
__global__ void slot_scatter_kernel(
    const int* __restrict__ ti,
    const int* __restrict__ group_off,
    const int* __restrict__ blk_off,
    int* __restrict__ se,
    long* __restrict__ stok,
    int* __restrict__ inv,
    int slots,
    int n_experts,
    int topk)
{
  const int blk = blockIdx.x;
  const int t = threadIdx.x;
  const int slot = blk * kSlotsPerBlock + t;
  const int warp = t >> 5;
  const int lane = t & 31;
  const int n_warps = blockDim.x >> 5;

  extern __shared__ int s_warp_hist[];   // n_warps * n_experts
  for (int i = t; i < n_warps * n_experts; i += blockDim.x) s_warp_hist[i] = 0;
  __syncthreads();

  const int e = (slot < slots) ? ti[slot] : -1;
  const unsigned active = __ballot_sync(0xffffffff, e >= 0);
  int rank_in_warp = 0;
  if (e >= 0) {
    const unsigned same = __match_any_sync(active, e);
    const unsigned lower = same & ((1u << lane) - 1u);
    rank_in_warp = __popc(lower);
    if (rank_in_warp == 0) {
      s_warp_hist[warp * n_experts + e] = __popc(same);
    }
  }
  __syncthreads();

  if (e >= 0) {
    int before = 0;
    for (int w = 0; w < warp; ++w) before += s_warp_hist[w * n_experts + e];
    const int row = group_off[e]
                  + blk_off[static_cast<size_t>(blk) * n_experts + e]
                  + before + rank_in_warp;
    se[row] = e;
    stok[row] = slot / topk;      // 64-bit: see the note in the header
    inv[slot] = row;
  }
}

__global__ void sfa_offsets_kernel(
    const int* __restrict__ group_off,
    int* __restrict__ sfa_off,
    int n_experts,
    int n_col)
{
  if (threadIdx.x != 0) return;
  int acc = 0;
  for (int e = 0; e < n_experts; ++e) {
    sfa_off[e] = acc;
    const int c = group_off[e + 1] - group_off[e];
    acc += ((c + 127) / 128) * (n_col * 512);
  }
}

int route_blocks(int slots) {
  return (slots + kSlotsPerBlock - 1) / kSlotsPerBlock;
}

}  // namespace

int moe_route_prefill_workspace_bytes(int S, int topk, int n_experts)
{
  const int slots = S * topk;
  const int nblk = route_blocks(slots);
  // blk_hist + blk_off + counts
  return static_cast<int>(
      (2 * static_cast<size_t>(nblk) * n_experts + n_experts) * sizeof(int));
}

int moe_route_prefill_bf16(
    const void* logits,
    void*       ti,
    void*       tw,
    void*       se,
    void*       stok,
    void*       inv,
    void*       group_off,
    void*       ws,
    int         ws_bytes,
    int         S,
    int         n_experts,
    int         topk,
    cudaStream_t stream)
{
  if (S <= 0) return 0;
  if (n_experts <= 0 || n_experts > kMaxExperts || (n_experts % 32) != 0)
    return 1;
  if (topk <= 0 || topk > kMaxTopK || topk > n_experts) return 2;
  if (ws_bytes < moe_route_prefill_workspace_bytes(S, topk, n_experts))
    return 3;

  const int slots = S * topk;
  const int nblk = route_blocks(slots);
  int* blk_hist = reinterpret_cast<int*>(ws);
  int* blk_off = blk_hist + static_cast<size_t>(nblk) * n_experts;
  int* counts = blk_off + static_cast<size_t>(nblk) * n_experts;
  int* ti_i = reinterpret_cast<int*>(ti);

  // One warp per token, four warps a block.
  const int warps = kRouteThreads / 32;
  const int topk_grid = (S + warps - 1) / warps;
  float* tw_f = reinterpret_cast<float*>(tw);
  const __nv_bfloat16* lg = reinterpret_cast<const __nv_bfloat16*>(logits);
  switch (n_experts / 32) {
    case 1:  route_topk_warp_kernel<1><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    case 2:  route_topk_warp_kernel<2><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    case 4:  route_topk_warp_kernel<4><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    case 8:  route_topk_warp_kernel<8><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    case 16: route_topk_warp_kernel<16><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    case 32: route_topk_warp_kernel<32><<<topk_grid, kRouteThreads, 0, stream>>>(
                 lg, ti_i, tw_f, S, n_experts, topk); break;
    default: return 4;      // expert count is not 32 * a power of two
  }

  slot_hist_kernel<<<nblk, kSlotsPerBlock, n_experts * sizeof(int), stream>>>(
      ti_i, blk_hist, slots, n_experts);

  expert_block_scan_kernel<256><<<n_experts, 256, 0, stream>>>(
      blk_hist, blk_off, counts, nblk, n_experts);

  group_off_kernel<<<1, 32, 0, stream>>>(
      counts, reinterpret_cast<int*>(group_off), n_experts);

  const size_t scatter_smem =
      static_cast<size_t>(kSlotsPerBlock / 32) * n_experts * sizeof(int);
  slot_scatter_kernel<<<nblk, kSlotsPerBlock, scatter_smem, stream>>>(
      ti_i, reinterpret_cast<const int*>(group_off), blk_off,
      reinterpret_cast<int*>(se), reinterpret_cast<long*>(stok),
      reinterpret_cast<int*>(inv), slots, n_experts, topk);

  return 0;
}

void moe_route_sfa_offsets(
    const void* group_off,
    void*       sfa_off,
    int         n_experts,
    int         n_col,
    cudaStream_t stream)
{
  sfa_offsets_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<const int*>(group_off),
      reinterpret_cast<int*>(sfa_off), n_experts, n_col);
}

}  // namespace kernels
}  // namespace flash_rt
