// SPDX-License-Identifier: Apache-2.0
//
// Host entry points of the Inkling-2-Mini SM120 kernels (module flash_rt_inkling2).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <stdexcept>
#include <string>

namespace flash_rt {
namespace inkling2 {

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Up to kMaxPrefetch byte ranges of weights that a later kernel reads, prefetched into L2 by the blocks of the
// launch that carries them. One SM issues an L2 prefetch piece every ~180 ns, so the ranges are spread over many
// blocks: multi-block kernels share them out; the one-block vector kernels launch kPrefetchBlocks extra blocks
// that only issue prefetches (while the DRAM is otherwise idle) and exit.
constexpr int kMaxPrefetch = 8;
constexpr int kPrefetchBlocks = 64;
struct Prefetch {
  const void* ptr[kMaxPrefetch];
  uint64_t bytes[kMaxPrefetch];
  int count;
};

// ---- BF16 GEMV over several matrices (inkling2_gemv.cu)
constexpr int kMaxGemvTasks = 8;
// Input prologue of a task (applied by every block while it stages the input):
//   kGemvPlain: x as given; kGemvNorm: x = bf16(rms_norm(x) * aux), aux the BF16 norm weight (eps 1e-6);
//   kGemvGate:  x = bf16(x * sigmoid(aux)), aux the BF16 gate (the attention output gate).
enum GemvPrologue { kGemvPlain = 0, kGemvNorm = 1, kGemvGate = 2 };
// rows > 1 (up to kMaxRows): x [rows][k], y [rows][n], a gate aux [rows][k]; every row's arithmetic is the one-row
// kernel's (same staging order, same per-lane accumulation and warp sum), so row r equals a one-row launch on row r.
constexpr int kMaxRows = 8;
struct GemvTask {
  const void* w;     // [n][k] BF16
  const void* x;     // [k] BF16 or FP32 (x_f32)
  void* y;           // [n] BF16 or FP32 (y_f32)
  const void* aux;   // prologue operand (BF16 [k])
  int n, k;
  int x_f32, y_f32;
  int prologue;
  int rows;          // 0 or 1: one row
  int x_stride;      // elements between input rows (0: k)
};
void gemv_multi(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream);

// The kGemvNorm prologue computed once per row: out [rows][k] = bf16(rms_norm(x) * w) in the one-row GEMV's order
// (a kGemvPlain task on out equals a kGemvNorm task on x). One block per (task, row).
constexpr int kMaxNormTasks = 4;
struct NormRowsTask {
  const void* x;     // [rows][x_stride] BF16 or FP32 (x_f32)
  const uint16_t* w; // [k] BF16
  uint16_t* out;     // [rows][k] BF16
  int k, x_f32, rows, x_stride;
};
void norm_rows(const NormRowsTask* tasks, int count, cudaStream_t stream);

// ---- model shape
constexpr int kHidden = 2304;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kInter = 1152;
constexpr int kMaxBlocks = 5;       // attention-residual block states seen by one visit (4) + the new one

// ---- per-visit vector kernels (inkling2_visit.cu), one block each
// Vector kernels over `rows` rows (one block per row; 0 / 1 = one row): per-row buffers are [rows][...] with the
// strides of one row, gate_in [3][rows][2 kHidden]. stage != null: a verification step, the conv state is read
// only, row r's window takes the inputs of rows r - 3 .. r - 1 of this step, and every row's conv input goes to
// stage [rows][kHidden] for the commit.
struct BoundaryParams {
  Prefetch prefetch;                // weights of later kernels (issued by kPrefetchBlocks extra blocks)
  int rows;
  uint16_t* stage;                  // MLP conv inputs [rows][kHidden] (verification)
  // finish the previous visit: hidden_out = bf16(residual + mlp_conv(bf16(sum_s y_s w_s)))
  int finish;
  const uint16_t* expert_out;       // [kTopK][kHidden] BF16
  const float* route_w;             // [kTopK]
  const float* residual;            // [kHidden]
  uint16_t* mlp_conv_state;         // [3][kHidden] BF16, oldest first
  const uint16_t* mlp_conv_w;       // [kHidden][4]
  uint16_t* hidden_out;             // [kHidden] BF16
  // or start the step: hidden_out = bf16(rms_norm(embed[token]) * embed_norm)
  int embed;
  const int* token;
  const uint16_t* embed_table;
  const uint16_t* embed_norm;
  float* bank_partial;              // optional: rms(hidden_out) (the message bank state at the end of a pass)
  // the next visit (none when null): its block residual, normed residual, attention inputs, messages
  const uint16_t* next_hidden;
  float* normed_res;                // [kHidden] rms(next_hidden), kept as a block state at boundaries
  int n_blocks;
  const float* blocks[kMaxBlocks];
  const float* att_query;           // [nq][kHidden] FP32; nq = 0: the input is the residual itself (layer 0)
  int nq;
  float* att_in;                    // [max(nq, 1)][kHidden]
  int messages;                     // loop passes 2 and 3
  const float* bank_block;
  const float* bank_state;
  const float* message_query;       // [4][kHidden]
  float* messages_out;              // [4][kHidden]
  uint16_t* gate_in;                // [3][2 kHidden] BF16: concat(rms(h_g), rms(m_g)) for q, k, v
  unsigned long long* timing;       // debug: globaltimer at the phase boundaries (null: off)
};
void visit_boundary(const BoundaryParams& p, cudaStream_t stream);

struct PostAttentionParams {
  Prefetch prefetch;                // weights of later kernels (issued by kPrefetchBlocks extra blocks)
  int rows;
  uint16_t* stage;                  // attention conv inputs [rows][kHidden] (verification)
  const uint16_t* attn_out;         // o_proj output [kHidden] BF16
  const uint16_t* block_residual;   // the visit input
  uint16_t* attn_conv_state;
  const uint16_t* attn_conv_w;
  int boundary;                     // first layer of an attention-residual block: residual = conv output only
  float* residual;                  // [kHidden] FP32
  int n_blocks;                     // block states including this visit's (at a boundary)
  const float* blocks[kMaxBlocks];
  const float* mlp_query;           // [kHidden]
  float* mlp_in;                    // [kHidden] FP32
  const float* message;             // loop passes 2, 3: the MLP message (row stride 4 kHidden), else null
  uint16_t* gate_in;                // [2 kHidden] BF16
};
void post_attention(const PostAttentionParams& p, cudaStream_t stream);

struct GateMixParams {
  int count;                        // gates (1..3)
  int rows;                         // rows (0 / 1: one); states [rows][64], h / m / out rows at the strides below
  int h_stride, m_stride, out_stride;
  const uint16_t* forget[3];        // [kHidden][64]
  const uint16_t* gain[3];
  const uint16_t* states[3];        // [64] BF16 (down projection output)
  const float* h[3];
  const float* m[3];
  float* out[3];
};
void gate_mix(const GateMixParams& p, cudaStream_t stream);

// ---- attention (inkling2_attention.cu)
struct AttnPrologueParams {
  Prefetch prefetch;                // weights of later kernels (issued by kPrefetchBlocks extra blocks)
  int sliding;
  const int* pos;                   // device position of the new row
  const uint16_t* q_lin;            // [24 * 128] projections (BF16)
  const uint16_t* r_lin;            // [24 * 16]
  const uint16_t* k_lin;            // [8 * 128]
  const uint16_t* v_lin;
  const uint16_t* k_dyn;            // full layers: dynamic conv weights [8 * 4] (weight_proj outputs)
  const uint16_t* v_dyn;
  uint16_t* k_conv_state;           // [3][1024]
  uint16_t* v_conv_state;
  const uint16_t* k_conv_w;         // sliding: [1024][4]; full: static [4][1024]
  const uint16_t* v_conv_w;
  const uint16_t* q_clip;           // [24 * 128]
  const uint16_t* rel_clip;         // [24 * 16]
  const uint16_t* rel_proj;         // [16][1024]
  float qk_threshold, rel_threshold, log_alpha, log_floor;
  uint16_t* q_out;                  // [24 * 128]
  uint16_t* bias;                   // [24][1024]
  uint16_t* k_cache;                // sliding: ring [1024][8][128]; full: [capacity][8][128]
  uint16_t* v_cache;
};
void attn_prologue(const AttnPrologueParams& p, cudaStream_t stream);

struct AttnDecodeParams {
  Prefetch prefetch;                // weights of later kernels, prefetched into L2 before the PDL wait
  int rows;                         // a verification step (rows > 1 with stage_kv set): rows at pos .. pos + rows - 1
  uint16_t* stage_kv;               // verification: the new rows' K / V [rows][2][8][128] (sliding layers)
  uint16_t* stage_in;               // verification: the k / v conv inputs [rows][2][1024] (kv_rows)
  int sliding;
  const int* pos;
  // the projections and the prologue's operands (see AttnPrologueParams): the attention block that holds the new
  // position computes its K / V row; every block its query heads and its keys' relative bias
  const uint16_t* q_lin;
  const uint16_t* r_lin;
  const uint16_t* k_lin;
  const uint16_t* v_lin;
  const uint16_t* k_dyn;
  const uint16_t* v_dyn;
  uint16_t* k_conv_state;
  uint16_t* v_conv_state;
  const uint16_t* k_conv_w;
  const uint16_t* v_conv_w;
  const uint16_t* q_clip;
  const uint16_t* rel_clip;
  const uint16_t* rel_proj;
  float qk_threshold, rel_threshold, log_alpha, log_floor;
  uint16_t* k_cache;
  uint16_t* v_cache;
  const uint16_t* sinks;            // [24]
  float* partial;                   // [8][max_splits][rows][3][130]
  int* counters;                    // [8][rows], zero between launches
  float* q_pre;                     // verification: the prepared queries and relative queries per row and KV head
                                    // ([rows][8][3 * 128 + 3 * 16], written by kv_rows, read by the split blocks)
  int max_splits;
  uint16_t* out;                    // [rows][24 * 128] BF16
  const uint16_t* gate;             // optional [rows][24 * 128]: out = bf16(bf16(attention) * sigmoid(gate)), the
                                    // kGemvGate prologue of the output projection applied once
};
void attn_decode(const AttnDecodeParams& p, cudaStream_t stream);
void kv_rows(const AttnDecodeParams& p, cudaStream_t stream);
int attn_split_keys();

// ---- routed experts (inkling2_moe.cu)
struct RouterParams {
  const float* in;                  // MLP input (FP32) before the norm
  const uint16_t* norm_w;           // mlp_norm
  const uint16_t* weight;           // [128][kHidden]
  const float* bias;                // [128]
  float route_scale;
  int mxfp8;                        // layer's expert format
  float* logits;                    // [128]
  int* counter;                     // zero between launches
  int* ids;                         // [kTopK]
  float* weights;                   // [kTopK]
  uint8_t* x_codes;                 // NVFP4 [kHidden / 2] or MXFP8 [kHidden]
  uint8_t* x_scales;                // [kHidden / 16] or [kHidden / 32]
  float* x_amax;
};
void router(const RouterParams& p, cudaStream_t stream);

struct ExpertParams {
  int mxfp8;
  int rows;                         // token rows (0 / 1: one); slots row * 8 + k grouped by expert
  const int* ids;                   // [kTopK]
  const void* codes;                // [128][rows][k / 2] NVFP4 or [128][rows][k] FP8
  const void* scales;               // [128][rows][k / 16] E4M3 or [128][rows][k / 32] UE8M0
  const float* amax;                // NVFP4: per-expert weight amax [128]
  float denominator;                // NVFP4: outer scale denominator
  float gate_cap, up_cap;
  const uint8_t* x_codes;           // w13: the quantized input (router kernel)
  const uint8_t* x_scales;
  const float* x_amax;
  // w13 with the routing folded in (logits set): every block takes the top-8 of sigmoid(logits) + bias, normalizes
  // and quantizes the MLP input itself; block (0, 0) writes ids_out / weights_out for w2 and the combination
  const float* logits;              // [128] router logits (FP32, gemv_multi)
  const float* bias;                // [128]
  float route_scale;
  const float* x_in;                // MLP input before the norm (FP32)
  const uint16_t* norm_w;           // mlp_norm
  int* ids_out;
  float* weights_out;
  uint16_t* act;                    // [kTopK][kInter] BF16 (w13 output, w2 input)
  uint16_t* out;                    // [kTopK][kHidden] BF16 (w2 output)
  // w13: the routed experts' w2 codes / scales (base, bytes per expert), prefetched into L2 slot by slot
  const uint8_t* w2_base[2];
  uint64_t w2_bytes[2];
  Prefetch prefetch;                // w2: weights of the next visit (host ranges)
  // flat passes: the staged quantized inputs per row (expert_route -> expert_w13_flat) or per slot
  // (expert_act_quant -> expert_w2_flat): NVFP4 codes as f16x2 pairs / MXFP8 values as half ([n][k * 2 bytes]),
  // FP32 block scales ([n][k / 16]), the NVFP4 amax ([n])
  void* xq;
  float* xq_scale;
  float* xq_amax;
};
void expert_w13(const ExpertParams& p, cudaStream_t stream);
void expert_w2(const ExpertParams& p, cudaStream_t stream);
// The same experts as passes over all rows of a step: routing + input quantization once per row, w13 over the flat
// (group, pair) items, the activations quantized once per slot, w2 over the flat (group, rows) items.
void expert_route(const ExpertParams& p, cudaStream_t stream);
void expert_w13_flat(const ExpertParams& p, cudaStream_t stream);
void expert_act_quant(const ExpertParams& p, cudaStream_t stream);
void expert_w2_flat(const ExpertParams& p, cudaStream_t stream);

// ---- LM head + greedy (inkling2_lmhead.cu)
struct LmHeadParams {
  int rows;                         // 0 / 1: one row; in / logits / token / block arrays per row
  int in_stride;                    // elements between rows of `in`
  const float* in;                  // final depth-attention output (FP32)
  const uint16_t* norm_w;           // final norm
  const uint16_t* weight;           // [vocab rows][kHidden]
  const uint8_t* weight_fp8;        // or E4M3 codes [vocab rows][kHidden] with weight_scale [vocab rows] (drafter)
  const float* weight_scale;
  int vocab;                        // unpadded vocabulary
  float softcap;                    // 0: none
  int blocks;
  float* logits;                    // optional [rows][vocab] FP32 (softcapped)
  float* block_best;                // [blocks][rows]
  int* block_index;
  int* counter;                     // zero between launches
  int* token;                       // greedy token
  uint16_t* normed_out;             // optional [rows][kHidden]: the normalized rows (the drafter's final feature)
  Prefetch prefetch;
};
void lm_head(const LmHeadParams& p, cudaStream_t stream);

// ---- DSpark drafter, one sequence (inkling2_mtp.cu); the projections run on gemv_multi
constexpr int kMtpBlock = 8;              // block positions: the anchor and 7 drafts
constexpr int kMtpLayers = 3;
constexpr int kMtpRank = 256;
constexpr int kMtpInter = 9216;
struct MtpParams {
  // context of committed positions: rows j < *count + count_add at pos - count + j
  const uint16_t* feat[4];          // target features [rows][kHidden] (hidden after visits 4 / 22 / 40, final norm)
  const int* count;
  int count_add;
  float alpha[16];                  // fuse mixes [4][4]: layers 0..2, the Markov head
  const uint16_t* norm[4];          // their dflare norms
  const uint16_t* kvc;              // [3][2][8][1024] context k / v projections
  const uint16_t* k_norm[3];        // k_norm_ctx per layer
  // shared
  const int* pos;                   // target position after the step = the block start (the anchor's position)
  uint16_t* kv;                     // [3][2][8][cap][128] drafter K / V
  int cap;
  float theta;
  int layer;                        // block kernels: the layer
  int window;                       // attention: keys from start - (window + 1) (0: all)
  uint16_t* x;                      // normalized rows: context [3][8][kHidden], block [8][kHidden]
  uint16_t* frnn;                   // [kHidden] the Markov feature
  // block
  const int* anchor;
  const uint16_t* embed;
  const uint16_t* embed_norm;
  const uint16_t* mask_embed;
  float* dh;                        // [8][kHidden] the block's residual stream
  const uint16_t* add;              // norm: added to dh first (optional)
  const float* add_part[2];         // norm: or two FP32 partial sums (a projection split along k)
  const uint16_t* norm_w;
  const uint16_t* q_in;             // [8][3072]
  const uint16_t* k_in;             // [8][1024]
  const uint16_t* v_in;             // [8][1024]
  const uint16_t* q_norm;
  const uint16_t* kb_norm;          // the block's k head norm
  uint16_t* q_out;                  // [8][3072] normalized, rotated queries
  uint16_t* attn_out;               // [8][3072]
  const uint16_t* gate;             // [8][9216]
  const uint16_t* up;
  uint16_t* act;
  // Markov walk
  int step;
  int* drafts;                      // [7]
  const uint16_t* markov_w1;        // [vocab][256]
  const uint16_t* hn;               // [8][kHidden] the block's final normalized rows
  uint16_t* rnn_in;                 // [5120]
  const float* proj;                // [768] rnn_proj output (no bias)
  const uint16_t* rnn_bias;         // [768]
  uint16_t* state;                  // [256]
  uint16_t* latent;                 // [256]
  const float* logits;              // [7][vocab_stride] base logits of block rows 1..7
  int vocab_stride;
  int vocab;                        // unpadded
  const uint16_t* fold_w;           // [vocab][256]
  const uint8_t* fold_w8;           // or E4M3 codes [vocab][256] with fold_scale [vocab]
  const float* fold_scale;
  const uint16_t* fold_b;           // [vocab]
  float* pick_best;                 // [mtp_pick_blocks()]
  int* pick_index;
  int* pick_counter;                // zero between launches
};
void mtp_context_mix(const MtpParams& p, cudaStream_t stream);
void mtp_context_write(const MtpParams& p, cudaStream_t stream);
void mtp_block_embed(const MtpParams& p, cudaStream_t stream);
void mtp_norm(const MtpParams& p, cudaStream_t stream);
void mtp_qk(const MtpParams& p, cudaStream_t stream);
void mtp_attention(const MtpParams& p, cudaStream_t stream);
void mtp_silu(const MtpParams& p, cudaStream_t stream);
void mtp_rnn_in(const MtpParams& p, cudaStream_t stream);
void mtp_gates(const MtpParams& p, cudaStream_t stream);
void mtp_pick(const MtpParams& p, cudaStream_t stream);
int mtp_pick_blocks();

// ---- speculative verification bookkeeping (inkling2_spec.cu)
struct AcceptParams {
  int rows;                         // verified rows (anchor + rows - 1 drafts)
  const int* drafts;                // [rows - 1]
  const int* target;                // [rows] greedy tokens of the verification rows
  const int* pos;                   // the step's first position
  int* accepted;                    // drafts accepted (n)
  int* tokens;                      // [rows] the step's tokens: drafts[0 .. n), target[n]
  int* next_token;                  // the next anchor (target[n])
  int* next_pos;                    // pos + n + 1
  int* log;                         // optional: the step's tokens appended at log[*log_count], *log_count += n + 1
  int* log_count;
};
void accept(const AcceptParams& p, cudaStream_t stream);

// per visit: k / v / attention / MLP conv states, staged k-v conv inputs [rows][2][1024], staged attention and MLP
// conv inputs [rows][kHidden], ring K, ring V, staged K / V rows [rows][2][1024], sliding flag
constexpr int kCommitFields = 11;
struct CommitParams {
  int visits;
  const uint64_t* table;            // [visits][kCommitFields]
  const int* accepted;
  const int* pos;
};
void commit(const CommitParams& p, cudaStream_t stream);

}  // namespace inkling2
}  // namespace flash_rt
