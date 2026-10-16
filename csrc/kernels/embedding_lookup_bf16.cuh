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

// Prefix embedding with a multimodal merge: row r is either an image feature
// row or a scaled token row, selected per row by img_index.
//
//   img_index[r] >= 0  -> out[r, :] = img[img_index[r], :]
//   otherwise          -> out[r, :] = bf16(embed[token_ids[r], :] * scale)
//
// img_index is computed on the host (-1 for a text row), so the source of a row
// is a property of the launch rather than a comparison against a sentinel id,
// and there is no device-side scan. `scale` is passed in bf16 by callers that
// pre-round it; the product is formed in fp32 and rounded once.
void embedding_lookup_merge_bf16(
    const int64_t* token_ids,
    const int* img_index,
    const __nv_bfloat16* embed,
    const __nv_bfloat16* img,
    __nv_bfloat16* out,
    int rows,
    int hidden,
    float scale,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
