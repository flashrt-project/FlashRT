// Generic bf16 embedding lookup — model-neutral.
//
//   out[r, :] = embed[token_ids[r], :],   r in [0, rows)
//
// A plain gather of bf16 embedding rows by token id. Historically named
// qwen36_embedding_lookup_bf16, but it is model-neutral and reused by Qwen3,
// the Qwen3-VL SM89 FP8 path and GROOT N1.7. Moved here from qwen36_misc.cu as
// part of the generic-helper ownership cleanup (#112). The legacy name
// qwen36_embedding_lookup_bf16 is kept as a thin wrapper over
// embedding_lookup_bf16 (see qwen36_misc.cu) so existing call sites and the
// existing binding keep working.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void embedding_lookup_bf16(
    const int64_t* token_ids,
    const __nv_bfloat16* embed,
    __nv_bfloat16* out,
    int rows,
    int hidden,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
