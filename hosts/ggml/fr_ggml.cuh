// ggml-facing interface of the FlashRT NVFP4 path (Thor SM110).
// Included from ggml-cuda.cu under #ifdef GGML_CUDA_FLASHRT.
#pragma once

// Host dependency: ggml-cuda's internal common header. The consuming build
// must put ggml/src/ggml-cuda on the include path (the ggml adapter is
// compiled inside the host's build tree, like the vllm/sglang adapters run
// inside their host's runtime).
#include "common.cuh"

// Called at the start of every backend graph evaluation; invalidates the
// per-evaluation quantized-activation cache.
void ggml_cuda_flashrt_begin_eval();

// True when this mul_mat should be routed to the FlashRT block-scaled NVFP4
// GEMM: NVFP4 weights, fp32 contiguous activations/dst, no batch dims,
// shapes within kernel alignment. cc must already be checked by the caller.
bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// dst = src1 @ src0 via activation quantize + NVFP4 x NVFP4 tcgen05 GEMM.
// Weights are repacked into the CUTLASS wire format on first use and cached
// for the process lifetime.
void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// True when the 4-node FFN subgraph {mul_mat gate, mul_mat up, GEGLU,
// mul_mat down} can run as one fused GeGLU GEMM (interleaved gate/up
// weights, FP4 intermediate) followed by the down GEMM.
bool ggml_cuda_flashrt_should_fuse_geglu(const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, const ggml_tensor * down_mm);

// Execute that fused FFN; writes the down mul_mat's dst.
void ggml_cuda_flashrt_geglu_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, ggml_tensor * down_mm);

// pi0.5 adaLN modulate window: {rms_norm?, mul_mat mod, add bias, view,
// repeat, mul, add, view, repeat, add}. rms is null for the variant whose
// normalized input arrives from a previous graph split. Outputs written by
// the fused execution: the bias add (consumed by the gate view later) and
// the final add.
bool ggml_cuda_flashrt_should_fuse_ada(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * bias_add,
                                       const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
                                       const ggml_tensor * mul, const ggml_tensor * add1,
                                       const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
                                       const ggml_tensor * add2);
void ggml_cuda_flashrt_ada_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mm,
                                ggml_tensor * bias_add, const ggml_tensor * view_scale, const ggml_tensor * mul,
                                const ggml_tensor * view_shift, ggml_tensor * add2);

// LayerNorm + affine window: {NORM, MUL weight, ADD bias} -> one kernel.
bool ggml_cuda_flashrt_should_fuse_ada_cached(
        const ggml_tensor * rms, const ggml_tensor * view_col,
        const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
        const ggml_tensor * mul, const ggml_tensor * add1,
        const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
        const ggml_tensor * add2);
void ggml_cuda_flashrt_ada_norm_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * rms,
                                       const ggml_tensor * view_scale, const ggml_tensor * view_shift,
                                       ggml_tensor * add2);

bool ggml_cuda_flashrt_should_fuse_ln(const ggml_tensor * norm, const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_flashrt_ln_affine(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * mul, ggml_tensor * add);

// SigLIP FFN window: {mul_mat up (NVFP4), add bias, GELU, cont, mul_mat
// down (f16), add bias, cont, add residual} -> fused FP4 Up GEMM (gelu
// epilogue, FP4 hidden) + Down GEMM (bias + residual epilogue).
bool ggml_cuda_flashrt_should_fuse_siglip_ffn(const ggml_tensor * up_mm, const ggml_tensor * bias1, const ggml_tensor * gelu,
                                              const ggml_tensor * cont1, const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                              const ggml_tensor * cont2, const ggml_tensor * res_add);
void ggml_cuda_flashrt_siglip_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * up_mm, const ggml_tensor * bias1,
                                  const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                  const ggml_tensor * cont2, ggml_tensor * res_add);

// pi0.5 AE fused QKV window: {mm k, reshape, rope, view, cpy, mm v,
// reshape, view, cpy, mm q, reshape, rope, scale} -> one fused GEMM over
// row-concatenated [k|v|q] weights + one post kernel (rope/scale/f16 KV
// suffix stores).
bool ggml_cuda_flashrt_should_fuse_qkv(const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                                       const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                                       const ggml_tensor * q_mm, const ggml_tensor * q_rope, const ggml_tensor * q_scale);
void ggml_cuda_flashrt_qkv(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                           const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                           const ggml_tensor * q_mm, const ggml_tensor * q_rope, ggml_tensor * q_scale);

// pi0.5 gated residual window: {view gate, repeat, mul, add}.
// Vision QKV pad window: {mul_mat, add bias, reshape}x3 + {pad}x3 -> three
// padded-weight GEMMs (bias in epilogue) writing the pad buffers directly.
bool ggml_cuda_flashrt_should_fuse_vis_qkv_pad(
        const ggml_tensor * mm_q, const ggml_tensor * add_q, const ggml_tensor * resh_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, const ggml_tensor * resh_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, const ggml_tensor * resh_v,
        const ggml_tensor * pad_q, const ggml_tensor * pad_k, const ggml_tensor * pad_v);
void ggml_cuda_flashrt_vis_qkv_pad(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_q, const ggml_tensor * add_q, ggml_tensor * pad_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, ggml_tensor * pad_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, ggml_tensor * pad_v,
        ggml_tensor * k_cast, ggml_tensor * v_cast);

// GEMM + optional bias + residual add fused into one epilogue.
bool ggml_cuda_flashrt_should_fuse_mm_res(const ggml_tensor * mm, const ggml_tensor * bias_add,
                                          const ggml_tensor * res_add);
bool ggml_cuda_flashrt_mm_res(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
                              const ggml_tensor * bias_add, ggml_tensor * res_add);

bool ggml_cuda_flashrt_should_fuse_gated_res(const ggml_tensor * view, const ggml_tensor * repeat,
                                             const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_flashrt_gated_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * view,
                                      const ggml_tensor * mul, ggml_tensor * add);

// Decomposed tiny-M decode attention: replaces a FLASH_ATTN_EXT node with
// QK-GEMM + masked softmax + PV-GEMM for q_tokens <= 16 over a single f16
// KV head of token rows.
bool ggml_cuda_flashrt_should_fuse_dec_attn(const ggml_tensor * fa);
void ggml_cuda_flashrt_dec_attn(ggml_backend_cuda_context & ctx, ggml_tensor * fa);

// SigLIP vision attention through the AOT FlashAttention-4 module
// (head_dim 80, no mask). Only available when the adapter was built with
// the fa4_aot artifacts; the first use loads the module (falls back if
// that first use happens during CUDA graph capture).
bool ggml_cuda_flashrt_should_fuse_vit_fa4(const ggml_tensor * fa, ggml_backend_cuda_context & ctx);
void ggml_cuda_flashrt_vit_fa4(ggml_backend_cuda_context & ctx, ggml_tensor * fa);

// Variant that also absorbs the {VIEW (head de-pad), CONT} pair that
// follows the vision FA node: the FA4 output converts straight into the
// CONT's packed [(H*d), S, B] f32 destination, skipping the padded f32
// store and the strided copy.
bool ggml_cuda_flashrt_should_fuse_vit_fa4_depad(const ggml_tensor * fa, const ggml_tensor * view,
                                                 const ggml_tensor * cont, ggml_backend_cuda_context & ctx);
void ggml_cuda_flashrt_vit_fa4_depad(ggml_backend_cuda_context & ctx, ggml_tensor * fa,
                                     const ggml_tensor * view, ggml_tensor * cont);

// pi0.5 prefill self-attention (head_dim 256, GQA 1 KV head, full
// attention with a row-uniform pad mask) through the second AOT FA4
// module. The window assumes the prefix-LM mask semantics of the pi0.5
// prefill graph (see the dispatch-side checks).
bool ggml_cuda_flashrt_should_fuse_prefill_fa4(const ggml_tensor * fa, ggml_backend_cuda_context & ctx);
void ggml_cuda_flashrt_prefill_fa4(ggml_backend_cuda_context & ctx, ggml_tensor * fa);

// Prefill fused QKV window: q mm->reshape->rope->scale, k mm->reshape->rope->
// pad, v mm->reshape->pad, each pad permuted+copied into a padded f16 tensor
// of token rows. One fused GEMM + qkv_post + pad-row zeroing.
bool ggml_cuda_flashrt_should_fuse_qkv_prefill(
        const ggml_tensor * q_mm, const ggml_tensor * q_rope, const ggml_tensor * q_scale,
        const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_pad,
        const ggml_tensor * v_mm, const ggml_tensor * v_pad,
        const ggml_tensor * k_cpy, const ggml_tensor * v_cpy);
void ggml_cuda_flashrt_qkv_prefill(ggml_backend_cuda_context & ctx,
        const ggml_tensor * q_mm, const ggml_tensor * q_rope, ggml_tensor * q_scale,
        const ggml_tensor * k_mm, const ggml_tensor * k_rope,
        const ggml_tensor * v_mm,
        ggml_tensor * k_cpy, ggml_tensor * v_cpy);

// Run of terminal f32->f16 row-copy CPY nodes (the persistent encoder-KV
// stores at the prefill graph tail) batched into one kernel launch. A node
// qualifies when it copies [hd, 1, n_rows] contiguous f32 rows into
// contiguous f16 rows and nothing reads the copy back inside the graph;
// all nodes of one batch share hd and n_rows.
bool ggml_cuda_flashrt_kv_tail_cpy_ok(const ggml_tensor * cpy, int64_t * hd, int64_t * n_rows);
bool ggml_cuda_flashrt_kv_tail_cpy(ggml_backend_cuda_context & ctx, ggml_tensor ** cpys, int n);

// {RMS_NORM, MUL(w), ADD(mul, norm)} -> rms_norm(x)*(1+w) in one kernel.
// The execute returns false (run unfused) only when its zero-vector cache
// cannot allocate during graph capture.
bool ggml_cuda_flashrt_should_fuse_rms_gemma(const ggml_tensor * rms, const ggml_tensor * mul,
                                             const ggml_tensor * add);
bool ggml_cuda_flashrt_rms_gemma(ggml_backend_cuda_context & ctx, const ggml_tensor * rms,
                                 const ggml_tensor * mul, ggml_tensor * add);
