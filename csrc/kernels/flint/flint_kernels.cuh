// SPDX-License-Identifier: Apache-2.0
//
// Host entry points of the Flint (SM120) kernel module. All pointers are device
// pointers; every launch takes the caller's stream so the sequence can be
// captured into a CUDA graph.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_rt {
namespace flint {

constexpr int kArgmaxBlocks = 128;

// GEMM epilogues (prefill).
constexpr int kEpiBf16 = 0;          // bf16(acc * gs)
constexpr int kEpiF32 = 1;           // acc * gs (fp32)
constexpr int kEpiSwigluDense = 2;   // gate|up interleaved-by-8: bf16(silu(bf16(g)) * bf16(u)) (precise)
constexpr int kEpiSwigluRouted = 3;  // routed experts: bf16(silu_approx(g) * u), fp32 g/u
constexpr int kEpiQkRope = 4;        // QKV projection: Q/K RMSNorm + RoPE + KV-cache write (gemm_w4a16_qkrope)

// Loading.
void repack_nvfp4(const void* src, long long rows, int K, void* codes, void* scales, long long group_rows,
                  long long group_stride, long long row_offset, cudaStream_t s);

// Decode (one token).
void embed_at_pos_bf16(const int* tokens, const int* pos, const void* table, void* out, cudaStream_t s);
// write_res = 0 leaves the residual untouched (final-norm evaluation of all rows).
void add_rmsnorm_bf16(const void* x, void* residual, const float* w, void* out, int rows, float eps,
                      int first, cudaStream_t s, int write_res = 1);
// epilogue: 0 bf16 out, 1 f32 out, 2 gate|up SwiGLU (bf16 out of N/2 rows). Returns non-zero for an
// unsupported (K, epilogue) pair.
int gemv_w4a16(const void* codes, const void* scales, const void* x, void* out, int N, int K, float gs,
               int epilogue, cudaStream_t s);
void qk_norm_rope_kv_bf16(const void* qkv, const float* q_norm_w, const float* k_norm_w,
                          const float* cos_t, const float* sin_t, const int* pos, void* q_out, void* cache,
                          int ring, int rope, float eps, cudaStream_t s);
void decode_attention_bf16(const void* q, const void* cache, const int* pos, int ring, int window,
                           const float* sinks, float scale, int n_split, float* o_part, float* ml_part,
                           void* out, cudaStream_t s);
void router_topk_bf16(const void* xn, const void* wr, float* logits_scratch, unsigned* counter, int* ids,
                      float* tw, cudaStream_t s);
void moe_gate_up_bf16(const void* xn, const int* ids, const void* ecodes, const void* escales,
                      const float* egs, const void* scodes, const void* sscales, float sgs, void* act,
                      cudaStream_t s);
void moe_down_combine_bf16(const void* act, const int* ids, const float* tw, const void* ecodes,
                           const void* escales, const float* egs, const void* scodes, const void* sscales,
                           float sgs, void* out, cudaStream_t s);
void argmax_next_f32(const float* logits, int n, float* cand_v, int* cand_i, unsigned* counter, int* tokens,
                     int* pos, const int* n_prompt, cudaStream_t s);

// Prefill (many tokens per launch). Row r of a chunk is at position *pos + r.
void embed_rows_bf16(const int* tokens, const int* pos, int n, const void* table, void* out, cudaStream_t s);
void qk_norm_rope_kv_rows_bf16(const void* qkv, int n, const float* qw, const float* kw, const float* cos_t,
                               const float* sin_t, const int* pos, void* q_out, void* cache, int ring, int rope,
                               float eps, cudaStream_t s);
// Prefill attention for n query rows at positions *pos.. (causal; window > 0: sliding window over a ring
// cache when ring > 0). Units of 16 tokens x 8 query heads per KV head, each cut into n_split KV ranges
// merged with the sink in-kernel; n_split > 1 needs o_part [units * n_split * 128 * 128], ml_part
// [units * n_split * 128 * 2] and counters [units] (zeroed once; they self-reset), units = ceil(n / 16) * 4.
// variant 0..4 = KV tile / stages 32/2 (default), 32/3, 64/2, 64/3, 32/4. Returns non-zero on bad arguments.
int prefill_attention_bf16(const void* q, int n, const void* cache, const int* pos, int ring, int window,
                            const float* sinks, float scale, void* out, int n_split, float* o_part, float* ml_part,
                            unsigned* counters, int variant, cudaStream_t s);
// Returns non-zero for unsupported shapes / epilogues (K % 64, N % 128).
// W4A16 GEMM launch configuration: BM rows per CTA (16/32/64/128), cp.async stages, split-K CTAs per
// (1, 1, split) cluster (partials reduced in rank order through DSMEM), kb = 64-k sub-tiles per stage (1/2/4).
struct GemmConfig {
  int bm, stages, split, kb;
};
int gemm_w4a16(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs, void* C,
               int ldc, int epilogue, cudaStream_t s);
int gemm_w4a16_cfg(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                   void* C, int ldc, int epilogue, GemmConfig c, cudaStream_t s);
// QKV projection with the qk_norm_rope_kv_rows_bf16 work in the epilogue (each 128-column tile is one head):
// Q heads -> q_out [M, 32 * 128] (normed + RoPE), K heads -> cache (normed + RoPE), V heads -> cache, all at
// positions *pos + row; bit-identical to gemm_w4a16 + qk_norm_rope_kv_rows_bf16. N must be 40 * 128.
int gemm_w4a16_qkrope(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                      const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t, const int* pos,
                      void* q_out, void* cache, int ring, int rope, float eps, cudaStream_t s);
// tile_rows: rows per tile of the tile list (from moe_scan_place); mean_rows: mean routed rows per expert
// (picks the default configuration, BM >= tile_rows).
int gemm_w4a16_grouped(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles, int max_tiles,
                       const void* codes, const void* scales, int N, int K, const float* gs_e, void* C, int ldc,
                       int epilogue, int tile_rows, int mean_rows, cudaStream_t s);
// Search hook: non-zero bm overrides the default dense / grouped configuration.
void set_gemm_override(const GemmConfig& dense, const GemmConfig& grouped);
// Many-token router on tensor cores (wr = router [256, 2048] bf16); adds selections to counts[256].
// pos / n_end (device ints, may be null): rows at positions >= *n_end -- the padding of a chunk launched with
// more rows than tokens -- are selected but neither counted here nor placed by moe_scan_place.
void router_topk_rows_bf16(const void* xn, int n, const void* wr, int* ids, float* tw, int* counts, const int* pos,
                           const int* n_end, cudaStream_t s);
// counts (from router_topk_rows_bf16, re-zeroed) -> expert offsets, grouped-GEMM tile list
// (expert, first sorted slot, rows) x n_tiles, and the (token, slot) scatter: sorted_tok[slot] = token,
// inv[token * 6 + j] = slot (0 for padding rows). Slot order inside an expert is unspecified (results do not
// depend on it).
void moe_scan_place(const int* ids, int n, int tile_rows, int* counts, int* sorted_tok, int* inv, int* tiles,
                    int* n_tiles, const int* pos, const int* n_end, cudaStream_t s);
void moe_combine_rows_bf16(const void* y, const int* inv, const float* tw, const void* shared, void* out, int n,
                           cudaStream_t s);
// moe_combine_rows_bf16 followed by add_rmsnorm_bf16(out, residual, w) in one pass (bit-identical); the combined
// row itself is not stored.
void moe_combine_norm_rows_bf16(const void* y, const int* inv, const float* tw, const void* shared, void* residual,
                                const float* w, void* out, int n, float eps, cudaStream_t s);
void advance_pos(int* pos, int n, cudaStream_t s);

// Launch shapes of the MoE decode kernels: warps per block and rows per warp for the
// router, the gate/up kernel (act rows) and the down/combine kernel (output rows).
struct MoeLaunchConfig {
  int router_warps, router_rows, gu_warps, gu_rows, dn_warps, dn_rows;
};
void set_moe_launch_config(const MoeLaunchConfig& c);
MoeLaunchConfig get_moe_launch_config();

// Launch shapes (rows per warp, warps per block) of the QKV (fused norm) and o_proj decode GEMVs.
struct GemvLaunchConfig {
  int qkv_rows, qkv_warps, o_rows, o_warps;
};
void set_gemv_launch_config(const GemvLaunchConfig& c);

// Fused decode step (PDL). Residual stream ping-pongs: a norm site reads res_in and block 0 writes res_out.
// gemv_fused: norm_w != nullptr fuses residual add + RMSNorm of x (K == 2048) into the GEMV prologue.
int gemv_fused_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                const float* norm_w, int first, float eps, void* out, int N, int K, float gs, int epilogue,
                cudaStream_t s);
void add_rmsnorm_pdl_bf16(const void* x, const void* res_in, const float* w, float eps, void* res_out, void* out,
                       cudaStream_t s);
// Final residual add + RMSNorm + LM head (fp32 logits) + greedy next token in one kernel (K == 2048).
// cand_v/cand_i hold one candidate per block (cand_cap entries); returns non-zero if they are too small or
// the rows-per-warp setting is unsupported. set_lm_head_rows: 2, 4 or 8 rows per warp (8 warps per block).
int lm_head_greedy_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                         const float* norm_w, float eps, float* logits, int N, float gs, float* cand_v, int* cand_i,
                         int cand_cap, unsigned* counter, int* tokens, int* pos, const int* n_prompt, cudaStream_t s);
void set_lm_head_rows(int rows);
void router_norm_bf16(const void* x, const void* res_in, const float* nw, float eps, void* res_out, void* xn_out,
                      const void* wr, float* logits, unsigned* counter, int* ids, float* tw, cudaStream_t s);
void moe_gate_up_pdl_bf16(const void* xn, const int* ids, const void* ecodes, const void* escales, const float* egs,
                       const void* scodes, const void* sscales, float sgs, void* act, cudaStream_t s);
void moe_down_combine_pdl_bf16(const void* act, const int* ids, const float* tw, const void* ecodes, const void* escales,
                            const float* egs, const void* scodes, const void* sscales, float sgs, void* out,
                            cudaStream_t s);
void embed_pdl_bf16(const int* tokens, const int* pos, const void* table, void* out, cudaStream_t s);
// warps in {4, 8, 16}; returns non-zero for an unsupported warp count.
int decode_attention_fused_bf16(const void* qkv, const float* q_norm_w, const float* k_norm_w, const float* cos_t,
                                const float* sin_t, const int* pos, void* cache, int ring, int window, int rope,
                                float eps, const float* sinks, float scale, int n_split, int warps, float* o_part,
                                float* ml_part, unsigned* counters, void* out, cudaStream_t s,
                                const unsigned long long* pf_ranges = nullptr, int n_pf = 0, int kv_slots = 0);

}  // namespace flint
}  // namespace flash_rt
