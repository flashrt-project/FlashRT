#pragma once

// Grouped NVFP4 activation quantisers for the qwen3_5_moe MoE path.
//
// Built only with the weight-only 4-bit tier
// (-DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON); the matching bindings are guarded on
// FLASHRT_HAVE_QWEN35MOE_W4A16, so a build without that tier contains neither
// these translation units nor their symbols. They live here rather than in
// quantize.cu because the layout they write is the grouped GEMM's, not the
// general quantiser's: scale factors go into the Sm1xx atom layout for each
// group's own row count.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Grouped activation quantiser for the MoE grouped GEMM: every expert's block
// in one launch. Same math as quantize_bf16_to_nvfp4_swizzled; what differs is
// that each group's scale factors go into the Sm1xx atom layout for that
// group's own row count, which is what the block-scaled grouped GEMM reads.
// Quantising per group instead is correct but costs a launch and a host
// iteration per expert -- and a host iteration is what a graph capture cannot
// have.
//
//   A              (slots, K) bf16, rows already sorted by expert
//   expert_of_row  (slots,)  i32
//   group_off      (E + 1,)  i32  prefix sums of the per-expert row counts
//   sfa_off        (E,)      i32  byte offset of each group's SF block
// K must be a multiple of 16. Returns 0 on success, nonzero on arg error.
int moe_grouped_quant_nvfp4_bf16(
    const void* A, const void* expert_of_row, const void* group_off,
    const void* sfa_off, const void* src_row, void* out_packed, void* out_sf,
    int slots, int K, cudaStream_t stream);

// Gate and quantise in one pass: reads the grouped GEMM's merged (slots,
// 2*inter) gate/up output directly, so the strided column halves are never
// copied out. The silu is rounded to bf16 exactly as silu_mul_sm120_bf16 does,
// so the quantiser sees the same value it did when the two were separate.
int moe_grouped_silu_quant_nvfp4_bf16(
    const void* merged, const void* expert_of_row, const void* group_off,
    const void* sfa_off, void* out_packed, void* out_sf,
    int slots, int inter, cudaStream_t stream);

// Warp-per-row form of the above: a lane owns one 16-element scale-factor
// group and keeps it in registers, so there is no shared memory and no
// barrier. Same arithmetic in the same order, so the output is identical.
int moe_grouped_silu_quant_nvfp4_warp_bf16(
    const void* merged, const void* expert_of_row, const void* group_off,
    const void* sfa_off, void* out_packed, void* out_sf,
    int slots, int inter, cudaStream_t stream);
