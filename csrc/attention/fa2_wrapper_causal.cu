// ================================================================
// FlashRT — Raw-pointer entry for vendored FA2 in CAUSAL mode.
//
// Sibling of csrc/attention/fa2_wrapper.cu which hard-codes
// is_causal=false / template <Is_causal=false>. This file does the
// same job but with is_causal=true and template <Is_causal=true>,
// and is exposed to Python as `flash_rt_fa2.fwd_bf16_causal`
// (binding added in csrc/fa2_bindings.cpp).
//
// Build set is intentionally minimal — only (bf16, head_dim=128)
// — because that is the only shape Qwen3-8B prefill needs. Other
// hdims / dtypes can be added later by adding new instantiation
// files under csrc/attention/fa2_causal_inst/ and extending the
// dispatch below.
//
// The non-causal wrapper's helpers (fill_params, splitkv heuristic)
// are duplicated here intentionally to keep this file standalone
// and to satisfy the repo's add-only kernel rule (rule 0.1) — we
// do not refactor / share code with the existing wrapper.
// ================================================================

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "flash_attn_2_src/flash_attn/namespace_config.h"
#include "flash_attn_2_src/flash_attn/flash.h"

namespace FLASH_NAMESPACE {
template<typename elem_type, int kHeadDim, bool Is_causal>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

template<typename elem_type, int kHeadDim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream);
}  // namespace FLASH_NAMESPACE

static inline int round_up_128(int x) { return ((x + 127) / 128) * 128; }

// Ported verbatim from fa2_wrapper.cu: maximise-wave efficiency
// search bounded at 85% of best eligible split.
static int fa2_num_splits_heuristic_causal(int batch_nheads_mblocks, int num_SMs,
                                            int num_n_blocks, int max_splits) {
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    float eff[129]; eff[0] = 0.f;
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&](int s) {
        return s == 1 || ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s - 1);
    };
    for (int s = 1; s <= max_splits; s++) {
        if (!is_split_eligible(s)) { eff[s] = 0.f; continue; }
        float n_waves = float(batch_nheads_mblocks * s) / num_SMs;
        float e = n_waves / ceilf(n_waves);
        if (e > max_efficiency) { max_efficiency = e; }
        eff[s] = e;
    }
    for (int s = 1; s <= max_splits; s++) {
        if (!is_split_eligible(s)) { continue; }
        if (eff[s] >= 0.85f * max_efficiency) { return s; }
    }
    return 1;
}

static void fill_params_causal(
    FLASH_NAMESPACE::Flash_fwd_params& params,
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale)
{
    params = {};
    params.is_bf16 = true;  // this wrapper is bf16-only

    params.q_ptr = const_cast<void*>(q_ptr);
    params.k_ptr = const_cast<void*>(k_ptr);
    params.v_ptr = const_cast<void*>(v_ptr);
    params.o_ptr = o_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = batch;
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = round_up_128(seqlen_q);
    params.seqlen_k_rounded = round_up_128(seqlen_k);
    params.d = head_dim;
    params.d_rounded = (head_dim + 31) & ~31;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
    params.softcap = 0.0f;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;

    // The core difference vs the non-causal wrapper.
    //
    // FA2's mask formula at csrc/attention/flash_attn_2_src/flash_attn/mask.h:173 reads
    //     col_idx_limit_right = min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right)
    // Upstream FA2's flash_api.cpp normalizes (is_causal=true) to
    //     window_size_left  = -1   (unbounded past)
    //     window_size_right =  0   (strict causal — no future)
    // before launching. The non-causal wrapper sets both to -1 because the
    // Is_causal=false template path doesn't read window_size_*; here we're
    // on the Is_causal=true path, so window_size_right MUST be 0 — leaving
    // it at -1 produces an off-by-one mask (row i sees cols 0..i-1 instead
    // of 0..i, and row 0 sees nothing).
    params.is_causal = true;
    params.window_size_left = -1;
    params.window_size_right = 0;

    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;
    params.is_seqlens_k_cumulative = true;
    params.rotary_dim = 0;
}

static int setup_splitkv_causal(FLASH_NAMESPACE::Flash_fwd_params& params,
                                 void* softmax_lse_accum_ptr, void* o_accum_ptr,
                                 int num_sms, int seqlen_q, int seqlen_k,
                                 int head_dim, int batch, int num_heads_q)
{
    int num_splits = 1;
    if (softmax_lse_accum_ptr != nullptr && o_accum_ptr != nullptr && num_sms > 0) {
        const int block_n = head_dim <= 64 ? 256 : (head_dim <= 128 ? 128 : 64);
        const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
        const int num_m_blocks = (seqlen_q + 63) / 64;
        num_splits = fa2_num_splits_heuristic_causal(
            batch * num_heads_q * num_m_blocks, num_sms * 2,
            num_n_blocks, 128);
    }
    params.num_splits = num_splits;
    if (num_splits > 1) {
        params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
        params.oaccum_ptr = o_accum_ptr;
    } else {
        params.softmax_lseaccum_ptr = nullptr;
        params.oaccum_ptr = nullptr;
    }
    return num_splits;
}

extern "C" void fvk_attention_fa2_fwd_bf16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream)
{
    if (head_dim != 128) {
        fprintf(stderr,
            "fvk_attention_fa2_fwd_bf16_causal: head_dim=%d not built. "
            "Only head_dim=128 is currently instantiated for the causal "
            "path. Add a new file under csrc/attention/fa2_causal_inst/ "
            "and extend the dispatch in fa2_wrapper_causal.cu to support "
            "additional shapes.\n", head_dim);
        std::abort();
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params_causal(params,
                       q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
                       batch, seqlen_q, seqlen_k,
                       num_heads_q, num_heads_kv, head_dim,
                       q_batch_stride, q_row_stride, q_head_stride,
                       k_batch_stride, k_row_stride, k_head_stride,
                       v_batch_stride, v_row_stride, v_head_stride,
                       o_batch_stride, o_row_stride, o_head_stride,
                       softmax_scale);

    int num_splits = setup_splitkv_causal(params, softmax_lse_accum_ptr, o_accum_ptr,
                                          num_sms, seqlen_q, seqlen_k,
                                          head_dim, batch, num_heads_q);
    if (num_splits > 1) {
        FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(params, stream);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, true>(params, stream);
    }
}
