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
struct GemvTask {
  const void* w;     // [n][k] BF16
  const void* x;     // [k] BF16 or FP32 (x_f32)
  void* y;           // [n] BF16 or FP32 (y_f32)
  const void* aux;   // prologue operand (BF16 [k])
  int n, k;
  int x_f32, y_f32;
  int prologue;
};
void gemv_multi(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream);

// ---- model shape
constexpr int kHidden = 2304;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kInter = 1152;
constexpr int kMaxBlocks = 5;       // attention-residual block states seen by one visit (4) + the new one

// ---- per-visit vector kernels (inkling2_visit.cu), one block each
struct BoundaryParams {
  Prefetch prefetch;                // weights of later kernels (issued by kPrefetchBlocks extra blocks)
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
  const float* message;             // loop passes 2, 3: the MLP message, else null
  uint16_t* gate_in;                // [2 kHidden] BF16
};
void post_attention(const PostAttentionParams& p, cudaStream_t stream);

struct GateMixParams {
  int count;                        // gates (1..3)
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
  float* partial;                   // [8][max_splits][3][130]
  int* counters;                    // [8], zero between launches
  int max_splits;
  uint16_t* out;                    // [24 * 128] BF16
};
void attn_decode(const AttnDecodeParams& p, cudaStream_t stream);
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
};
void expert_w13(const ExpertParams& p, cudaStream_t stream);
void expert_w2(const ExpertParams& p, cudaStream_t stream);

// ---- LM head + greedy (inkling2_lmhead.cu)
struct LmHeadParams {
  const float* in;                  // final depth-attention output (FP32)
  const uint16_t* norm_w;           // final norm
  const uint16_t* weight;           // [vocab rows][kHidden]
  int vocab;                        // unpadded vocabulary
  float softcap;
  int blocks;
  float* logits;                    // optional [vocab] FP32 softcapped
  float* block_best;                // [blocks]
  int* block_index;
  int* counter;                     // zero between launches
  int* token;                       // greedy token
  Prefetch prefetch;
};
void lm_head(const LmHeadParams& p, cudaStream_t stream);

}  // namespace inkling2
}  // namespace flash_rt
