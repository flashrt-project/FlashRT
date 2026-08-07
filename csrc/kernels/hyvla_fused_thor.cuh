// FlashRT — Hy-VLA fused attention-prep megakernel declaration.
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_rope_qknorm_kvwrite_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);
