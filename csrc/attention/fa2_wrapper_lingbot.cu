// ================================================================
// FlashRT — LingBot-only FA2 entries (slim build).
//
// The generic fa2_wrapper.cu exposes fp16, causal, window, seqused and
// split-KV variants plus every head-dimension/tile combination. LingBot's
// native transformer calls exactly two of them:
//
//   fwd_bf16       S_q > 64,  bf16, head_dim=128, MHA, non-causal,
//                  no window, p_dropout=1 (dropout branch off), no split-KV.
//                  The vendored sm8x dispatch selects
//                  Flash_fwd_kernel_traits<128, 128, 32, 4, false, false>.
//   fwd_bf16_tile  S_q <= 64, bf16, head_dim=128, MHA, non-causal,
//                  no window, no split-KV. The small-query tile is
//                  Flash_fwd_kernel_traits<128, 64, 64, 4, false, false>.
//
// This TU instantiates only those two kernels and the pybind module exposes
// only those two entries. The kernels are byte-for-byte the same template
// instantiations the generic wrapper uses at these shapes, so the output is
// unchanged; the unused kernels never enter the fatbin.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>

#include "flash_attn_2_src/flash_attn/namespace_config.h"
#include "flash_attn_2_src/flash_attn/flash.h"
#include "flash_attn_2_src/flash_attn/flash_fwd_launch_template.h"
#include "attention/fa2_wrapper.h"

static inline int round_up_128(int x) { return ((x + 127) / 128) * 128; }

// Exact copy of the generic wrapper's fill_params (fa2_wrapper.cu). Keep the
// two in sync: the field set is what makes the slim kernel see the same
// Flash_fwd_params as the generic entry.
static void fill_params(
    FLASH_NAMESPACE::Flash_fwd_params& params,
    bool elem_is_bf16,
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
    params.is_bf16 = elem_is_bf16;

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

    // Dropout disabled (inference)
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;

    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;

    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;
    params.is_seqlens_k_cumulative = true;
    params.rotary_dim = 0;
}

extern "C" void fvk_attention_fa2_fwd_bf16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int /*num_sms*/, cudaStream_t stream)
{
    if (head_dim != 128) {
        std::fprintf(stderr,
            "[flash_rt_fa2:slim] fwd_bf16 build supports head_dim=128 only, "
            "got %d\n", head_dim);
        std::abort();
    }
    if (softmax_lse_accum_ptr != nullptr || o_accum_ptr != nullptr) {
        std::fprintf(stderr,
            "[flash_rt_fa2:slim] fwd_bf16 build does not include split-KV; "
            "pass softmax_lse_accum=0 and o_accum=0\n");
        std::abort();
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params, true, q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
                batch, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride, softmax_scale);
    params.num_splits = 1;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;

    // The generic sm8x non-causal dispatch for head_dim=128, dropout off.
    using Traits = ::Flash_fwd_kernel_traits<
        128, 128, 32, 4, false, false, cutlass::bfloat16_t>;
    FLASH_NAMESPACE::run_flash_fwd<Traits, false, false>(params, stream);
}

extern "C" void fvk_attention_fa2_fwd_bf16_tile(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int /*num_sms*/, cudaStream_t stream)
{
    if (head_dim != 128) {
        std::fprintf(stderr,
            "[flash_rt_fa2:slim] fwd_bf16_tile build supports head_dim=128 "
            "only, got %d\n", head_dim);
        std::abort();
    }

    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params, true, q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
                batch, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride, softmax_scale);
    params.seqused_k = reinterpret_cast<int*>(const_cast<void*>(seqused_k_ptr));
    params.num_splits = 1;
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;

    using Traits = ::Flash_fwd_kernel_traits<
        128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    FLASH_NAMESPACE::run_flash_fwd<Traits, false, false>(params, stream);
}
