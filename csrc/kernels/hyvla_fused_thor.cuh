// FlashRT — Hy-VLA fused attention-prep megakernel declaration.
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_rope_qknorm_kvwrite_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);

// Additive variant: identical math (bit-exact), grid (S, nq + 2*nkv) so the
// heads of a position run in independent blocks (removes the occupancy bound
// at the small denoise suffix). The base kernel above is unchanged.
extern "C" void hyvla_rope_qknorm_kvwrite_parallel_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);

// Variant whose Q output is written directly in the FA2 denoise (2,S,nq,hd)
// packing used by `hyvla_fa2_denoise_attn` (row 0 -> batch 0 pos 0; row s>=1 ->
// batch 1 pos s-1). Replaces the separate (nq,S,hd) write + prepare_q transpose
// with a single fused launch.
extern "C" void hyvla_rope_qknorm_kvwrite_qb_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* qb, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);
