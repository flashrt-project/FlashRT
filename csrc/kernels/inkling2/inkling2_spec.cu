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

// grid (visits): table rows of kCommitFields pointers per visit.
__global__ void __launch_bounds__(256) commit_kernel(const __grid_constant__ CommitParams p) {
  pdl_wait();
  pdl_launch();
  const uint64_t* row = after_wait(p.table) + size_t(blockIdx.x) * kCommitFields;
  const int n = *after_wait(p.accepted) + 1;                     // rows to commit
  const int pos = *after_wait(p.pos);
  // conv states: last 3 of [state (3 rows), inputs of rows 0 .. n - 1]
  struct Conv { int field_state, field_in, channels, in_stride, in_offset; };
  const Conv convs[4] = {{0, 4, kKv, 2 * kKv, 0}, {1, 4, kKv, 2 * kKv, kKv},
                         {2, 5, kHidden, kHidden, 0}, {3, 6, kHidden, kHidden, 0}};
#pragma unroll
  for (int ci = 0; ci < 4; ++ci) {
    uint16_t* st = reinterpret_cast<uint16_t*>(row[convs[ci].field_state]);
    const uint16_t* in = reinterpret_cast<const uint16_t*>(row[convs[ci].field_in]) + convs[ci].in_offset;
    const int C = convs[ci].channels;
    for (int c = threadIdx.x; c < C; c += 256) {
      uint16_t v[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int i = n + j;                       // item index in [state, inputs]
        v[j] = i < 3 ? st[i * C + c] : in[size_t(i - 3) * convs[ci].in_stride + c];
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) st[j * C + c] = v[j];
    }
  }
  // sliding rings
  if (row[10]) {
    uint16_t* rk = reinterpret_cast<uint16_t*>(row[7]);
    uint16_t* rv = reinterpret_cast<uint16_t*>(row[8]);
    const uint16_t* skv = reinterpret_cast<const uint16_t*>(row[9]);
    for (int i = threadIdx.x; i < n * kKv; i += 256) {
      const int j = i / kKv, e = i % kKv;
      const int slot = (pos + j) % 1024;
      rk[size_t(slot) * kKv + e] = skv[(size_t(j) * 2) * kKv + e];
      rv[size_t(slot) * kKv + e] = skv[(size_t(j) * 2 + 1) * kKv + e];
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
  launch_pdl(commit_kernel, dim3(p.visits), dim3(256), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
