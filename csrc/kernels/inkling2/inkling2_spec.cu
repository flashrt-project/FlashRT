// SPDX-License-Identifier: Apache-2.0
//
// Speculative verification bookkeeping: accept (the longest prefix of drafts equal to the target's greedy tokens
// of the preceding rows, plus the target's next token) and commit (the accepted rows' conv inputs shifted into
// every visit's conv states, their staged K / V rows copied into the sliding rings; full-attention rows are in the
// cache already).

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kKv = 8 * 128;

__global__ void accept_kernel(const __grid_constant__ AcceptParams p) {
  pdl_wait();
  pdl_launch();
  if (threadIdx.x != 0) return;
  const int* drafts = after_wait(p.drafts);
  const int* target = after_wait(p.target);
  int n = 0;
  while (n < p.rows - 1 && drafts[n] == target[n]) ++n;
  int* toks = after_wait(p.tokens);
  for (int j = 0; j < n; ++j) toks[j] = drafts[j];
  toks[n] = target[n];
  *after_wait(p.accepted) = n;
  *after_wait(p.next_token) = target[n];
  *after_wait(p.next_pos) = *after_wait(p.pos) + n + 1;
  if (p.log) {
    int* count = after_wait(p.log_count);
    int* log = after_wait(p.log) + *count;
    for (int j = 0; j <= n; ++j) log[j] = toks[j];
    *count += n + 1;
  }
}

// grid (visits, ceil(rows / kCommitRows)): table rows of kCommitFields pointers per visit; the blocks of a visit copy
// kCommitRows ring rows each, the first also shifts the conv states.
// one conv's channels c = t, t + 256, ..: every load of a thread issued before its stores
template <int C>
__device__ __forceinline__ void commit_conv(uint16_t* st, const uint16_t* in, int in_stride, int n) {
  constexpr int kM = (C + 255) / 256;
  uint16_t v[kM][3];
#pragma unroll
  for (int m = 0; m < kM; ++m) {
    const int c = threadIdx.x + m * 256;
    if (c >= C) break;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int i = n + j;                         // item index in [state, inputs]
      v[m][j] = i < 3 ? st[i * C + c] : in[size_t(i - 3) * in_stride + c];
    }
  }
#pragma unroll
  for (int m = 0; m < kM; ++m) {
    const int c = threadIdx.x + m * 256;
    if (c >= C) break;
#pragma unroll
    for (int j = 0; j < 3; ++j) st[j * C + c] = v[m][j];
  }
}

__global__ void __launch_bounds__(256) commit_kernel(const __grid_constant__ CommitParams p) {
  // the visit's table (static pointers) and the position (written many launches back) before the wait
  const uint64_t* rowp = p.table + size_t(blockIdx.x) * kCommitFields;
  uint64_t row[kCommitFields];
#pragma unroll
  for (int f = 0; f < kCommitFields; ++f) row[f] = rowp[f];
  const int pos = *p.pos;
  pdl_wait();
  pdl_launch();
  const int n = *after_wait(p.accepted) + 1;                     // rows to commit
  const int j0 = blockIdx.y * kCommitRows, j1 = min(n, j0 + kCommitRows);
  if (j0 >= n) return;
  if (blockIdx.y == 0) {
    // conv states: last 3 of [state (3 rows), inputs of rows 0 .. n - 1]
    commit_conv<kKv>(reinterpret_cast<uint16_t*>(row[0]), reinterpret_cast<const uint16_t*>(row[4]), 2 * kKv, n);
    commit_conv<kKv>(reinterpret_cast<uint16_t*>(row[1]), reinterpret_cast<const uint16_t*>(row[4]) + kKv, 2 * kKv, n);
    commit_conv<kHidden>(reinterpret_cast<uint16_t*>(row[2]), reinterpret_cast<const uint16_t*>(row[5]), kHidden, n);
    commit_conv<kHidden>(reinterpret_cast<uint16_t*>(row[3]), reinterpret_cast<const uint16_t*>(row[6]), kHidden, n);
  }
  // sliding rings: rows j0 .. j1 - 1 of this block, 16-byte pieces
  if (row[10]) {
    uint4* rk = reinterpret_cast<uint4*>(row[7]);
    uint4* rv = reinterpret_cast<uint4*>(row[8]);
    const uint4* skv = reinterpret_cast<const uint4*>(row[9]);
    constexpr int kPieces = kKv / 8;
    for (int i = threadIdx.x; i < (j1 - j0) * kPieces; i += 256) {
      const int j = j0 + i / kPieces, e = i % kPieces;
      const int slot = (pos + j) % 1024;
      rk[size_t(slot) * kPieces + e] = skv[(size_t(j) * 2) * kPieces + e];
      rv[size_t(slot) * kPieces + e] = skv[(size_t(j) * 2 + 1) * kPieces + e];
    }
  }
}

}  // namespace

void accept(const AcceptParams& p, cudaStream_t stream) {
  if (p.rows < 2 || p.rows > kMaxRows) throw std::invalid_argument("accept: 2..8 rows");
  launch_pdl(accept_kernel, dim3(1), dim3(32), 0, stream, p);
}

void commit(const CommitParams& p, cudaStream_t stream) {
  if (p.visits < 1) throw std::invalid_argument("commit: visits >= 1");
  const int rows = p.rows < 1 ? kMaxRows : p.rows;
  launch_pdl(commit_kernel, dim3(p.visits, (rows + kCommitRows - 1) / kCommitRows), dim3(256), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
