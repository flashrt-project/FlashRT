// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini Gated DeltaNet recurrent SEQUENTIAL-SCAN kernel (prefill).
//
// The per-token recurrent (gated_deltanet_recurrent_qwen36_bf16) is launched
// once per prompt token in the prefill GDN loop -> S launches/layer, and the
// HD*HD state is round-tripped through HBM every token. This variant scans the
// whole prompt inside ONE launch per layer: each thread keeps its state column
// in registers across all S timesteps (no per-token state HBM traffic), looping
// q/k/v/g/beta over the sequence and emitting out[t] each step. Bit-equivalent
// math to the per-token kernel (same fp32 ops, bf16 state writeback at the end).
//
// Layout: q/k/v/out (S, num_v_heads, HD); g/beta (S, num_v_heads);
//         state (num_v_heads, HD, HD) -- read as the initial state, overwritten
//         with the final state. HD = 128. All add-only.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

int nexn2_gdn_recurrent_seq_bf16(
    const void*  q,
    const void*  k,
    const void*  v,
    const void*  g,
    const void*  beta,
    void*        state,
    void*        out,
    int          S,
    int          num_v_heads,
    int          head_dim,
    bool         use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
