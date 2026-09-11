#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// ================================================================
// FlashRT AMD — elementwise launch-geometry tuning probe (CDNA4)
//
// Companion to kernels/stream_probe.h. The production elementwise
// kernels were ported with their CUDA launch geometry (row-per-block
// norms, one packed pair per thread, byte-granular FP8 stores). On
// CDNA that geometry sits well above the measured in-graph dispatch
// floor. This probe instantiates the three archetypes that cover the
// whole elementwise surface — flat quantize transform, fused
// row-reduction norm, and QKV split+RoPE — each as a family of
// geometry variants (access width, FP8 store packing, waves-per-row
// vs block-per-row reduction, ILP, block size), so the production
// geometry is measured, not guessed.
//
// Variant 0 of every family replicates the production kernel's
// current geometry and math exactly (in-file control arm). All
// variants of a family compute the same function; reduction-order
// variants may differ from the control in the last ULP only.
//
// Wide variants require: 16-byte aligned pointers, n (or dim)
// divisible by 8. Wave-per-row variants additionally require
// dim <= 4096. Host wrappers throw std::runtime_error otherwise.
// ================================================================

// ── Family Q: flat FP8 quantize (mirrors quantize_fp8_static) ──
int ew_tune_quant_variant_count();
const char* ew_tune_quant_variant_name(int id);
void ew_tune_quant(int variant, const __hip_bfloat16* in, __hip_fp8_e4m3* out,
                   const float* d_scale, int n, hipStream_t stream);

// ── Family N: fused gate*residual + AdaRMSNorm + style → FP8 ──
// (mirrors gate_residual_ada_norm_fp8 — the fullest fused-norm archetype)
int ew_tune_norm_variant_count();
const char* ew_tune_norm_variant_name(int id);
void ew_tune_norm(int variant, __hip_bfloat16* residual, const __hip_bfloat16* x,
                  const __hip_bfloat16* gate, const __hip_bfloat16* weight,
                  const __hip_bfloat16* style, __hip_fp8_e4m3* out,
                  __hip_bfloat16* gate_out, int seq_len, int dim, float eps,
                  const float* d_scale, hipStream_t stream);

// ── Family R: fused QKV split + RoPE (mirrors qkv_split_rope) ──
int ew_tune_rope_variant_count();
const char* ew_tune_rope_variant_name(int id);
void ew_tune_rope(int variant, const __hip_bfloat16* qkv,
                  const __hip_bfloat16* rope_weights,
                  __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                  int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                  hipStream_t stream);
