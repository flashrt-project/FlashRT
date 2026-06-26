// ================================================================
// FlashRT — Raw-pointer entries for vendored Flash-Attention 2
//
// Replaces flash_api.cpp (torch-typed) with two C-linkage entries:
//   fvk_attention_fa2_fwd_fp16   — elem_type = cutlass::half_t
//   fvk_attention_fa2_fwd_bf16   — elem_type = cutlass::bfloat16_t
// Both dispatch to either
//   run_mha_fwd_               — single-call, no KV splits
//   run_mha_fwd_splitkv_dispatch — KV axis split into N pieces
// depending on the num_splits heuristic (ported verbatim from
// upstream flash_api.cpp). Matching the heuristic is essential:
// different kernel paths accumulate FMAs in different orders and the
// 1 fp16/bf16 ULP drift crosses FP8 bucket boundaries, which compounds
// through Pi0's 225+ attention ops into a macroscopic cos regression.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>    // cutlass::half_t, cutlass::bfloat16_t

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
}

static inline int round_up_128(int x) { return ((x + 127) / 128) * 128; }

// Ported from flash_api.cpp:num_splits_heuristic. Maximise-wave
// efficiency search bounded at 85% of best eligible split.
static int fa2_num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
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


// Populate the common portion of Flash_fwd_params. Caller sets is_bf16
// via `elem_is_bf16` because the struct's flag controls some inner
// dispatches; also sets the three alibi/window/rotary disables.
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


// Compute num_splits based on heuristic (or 1 if scratch unavailable)
// and wire the splitkv scratch pointers into params.
static int setup_splitkv(FLASH_NAMESPACE::Flash_fwd_params& params,
                          void* softmax_lse_accum_ptr, void* o_accum_ptr,
                          int num_sms, int seqlen_q, int seqlen_k,
                          int head_dim, int batch, int num_heads_q)
{
    int num_splits = 1;
    if (softmax_lse_accum_ptr != nullptr && o_accum_ptr != nullptr && num_sms > 0) {
        const int block_n = head_dim <= 64 ? 256 : (head_dim <= 128 ? 128 : 64);
        const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
        const int num_m_blocks = (seqlen_q + 63) / 64;   // kBlockM=64 for splitkv
        num_splits = fa2_num_splits_heuristic(
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


// Missing-instantiation trap. Called when the caller requests an
// hdim bucket (or a dtype) that the current build did not include
// via FA2_HDIMS / FA2_DTYPES. Prints a clear message instead of the
// cryptic "undefined reference" you'd get if we linked blindly.
[[noreturn]] static void fa2_not_built(const char* which, int v) {
    fprintf(stderr,
        "fvk_attention_fa2: %s=%d was not compiled into this build. "
        "Rebuild with -DFA2_HDIMS=\"96;128;256\" -DFA2_DTYPES=\"fp16;bf16\" "
        "to enable the full matrix (or add just the value you need).\n",
        which, v);
    std::abort();
}

template<typename elem_t>
static void dispatch_hdim(int head_dim, int num_splits,
                           FLASH_NAMESPACE::Flash_fwd_params& params,
                           cudaStream_t stream)
{
    auto do_dispatch = [&](auto tag) {
        constexpr int kHD = decltype(tag)::value;
        if (num_splits > 1) {
            FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<elem_t, kHD, false>(params, stream);
        } else {
            FLASH_NAMESPACE::run_mha_fwd_<elem_t, kHD, false>(params, stream);
        }
    };
    (void)do_dispatch;  // silence unused warning when all hdims are gated out
    if (head_dim > 0 && head_dim <= 64 && num_splits <= 1) {
        // hd64 (Qwen3-VL 2B vision) builds only the no-split path; any
        // split-KV request falls through to the hd96 bucket below.
#ifdef FA2_HAS_HDIM_64
        do_dispatch(std::integral_constant<int, 64>{});
#else
        fa2_not_built("head_dim<=64", head_dim);
#endif
    } else if (head_dim > 0 && head_dim <= 96) {
#ifdef FA2_HAS_HDIM_96
        do_dispatch(std::integral_constant<int, 96>{});
#else
        fa2_not_built("head_dim<=96", head_dim);
#endif
    } else if (head_dim <= 128) {
#ifdef FA2_HAS_HDIM_128
        do_dispatch(std::integral_constant<int, 128>{});
#else
        fa2_not_built("head_dim<=128", head_dim);
#endif
    } else if (head_dim <= 256) {
#ifdef FA2_HAS_HDIM_256
        do_dispatch(std::integral_constant<int, 256>{});
#else
        fa2_not_built("head_dim<=256", head_dim);
#endif
    } else {
        fprintf(stderr,
            "fvk_attention_fa2_fwd: unsupported head_dim=%d "
            "(supported buckets: <=64, <=96, <=128, <=256).\n", head_dim);
        std::abort();
    }
}


#define DEFINE_FA2_ENTRY(NAME, ELEM_T, IS_BF16)                                  \
extern "C" void NAME(                                                            \
    const void* q_ptr, const void* k_ptr, const void* v_ptr,                     \
    void* o_ptr, void* softmax_lse_ptr,                                          \
    void* softmax_lse_accum_ptr, void* o_accum_ptr,                              \
    int batch, int seqlen_q, int seqlen_k,                                       \
    int num_heads_q, int num_heads_kv, int head_dim,                             \
    int q_batch_stride, int q_row_stride, int q_head_stride,                     \
    int k_batch_stride, int k_row_stride, int k_head_stride,                     \
    int v_batch_stride, int v_row_stride, int v_head_stride,                     \
    int o_batch_stride, int o_row_stride, int o_head_stride,                     \
    float softmax_scale, int num_sms, cudaStream_t stream)                       \
{                                                                                \
    FLASH_NAMESPACE::Flash_fwd_params params;                                    \
    fill_params(params, IS_BF16,                                                 \
                q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,                     \
                batch, seqlen_q, seqlen_k,                                       \
                num_heads_q, num_heads_kv, head_dim,                             \
                q_batch_stride, q_row_stride, q_head_stride,                     \
                k_batch_stride, k_row_stride, k_head_stride,                     \
                v_batch_stride, v_row_stride, v_head_stride,                     \
                o_batch_stride, o_row_stride, o_head_stride,                     \
                softmax_scale);                                                  \
    int num_splits = setup_splitkv(params, softmax_lse_accum_ptr, o_accum_ptr,   \
                                    num_sms, seqlen_q, seqlen_k,                 \
                                    head_dim, batch, num_heads_q);               \
    dispatch_hdim<ELEM_T>(head_dim, num_splits, params, stream);                 \
}

// ──────────────────────────────────────────────────────────────
// Entry symbols. Each dtype is guarded by an FA2_HAS_{FP16,BF16}
// macro set from CMake via FA2_DTYPES. If a dtype was dropped to
// slim the build, we still emit the symbol (so the pybind module
// and Python callers link) but the body aborts with a clear
// "rebuild with this dtype" message.
// ──────────────────────────────────────────────────────────────

#define DEFINE_FA2_STUB(NAME, DTYPE_STR)                                         \
extern "C" void NAME(                                                            \
    const void*, const void*, const void*, void*, void*,                         \
    void*, void*,                                                                \
    int, int, int, int, int, int,                                                \
    int, int, int, int, int, int,                                                \
    int, int, int, int, int, int,                                                \
    float, int, cudaStream_t)                                                    \
{                                                                                \
    fprintf(stderr,                                                              \
        "fvk_attention_fa2: " DTYPE_STR " entry was not compiled. "              \
        "Rebuild with -DFA2_DTYPES=\"fp16;bf16\" to enable it.\n");              \
    std::abort();                                                                \
}

#ifdef FA2_HAS_FP16
DEFINE_FA2_ENTRY(fvk_attention_fa2_fwd_fp16, cutlass::half_t,    false)
#else
DEFINE_FA2_STUB(fvk_attention_fa2_fwd_fp16,  "fp16")
#endif

#ifdef FA2_HAS_BF16
DEFINE_FA2_ENTRY(fvk_attention_fa2_fwd_bf16, cutlass::bfloat16_t, true)

// seqused_k variant: K/V length is read per-batch from device memory
// (seqused_k[b]); the kernel clamps n_block_max to that length and early-exits.
// Forces num_splits=1 (no reduction scratch, CUDA-graph-safe). Lets one captured
// decode graph serve every position — the host writes the current length into
// the seqused_k buffer before each replay (paired with the devpos KV-write).
extern "C" void fvk_attention_fa2_fwd_bf16_seqused(
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
    dispatch_hdim<cutlass::bfloat16_t>(head_dim, 1, params, stream);
}

// seqused_k + split-KV variant (experimental). Caller MUST pre-init
// softmax_lse_accum to -inf each launch (empty splits past seqused_k keep that
// so the combine ignores them) — see standalone graph-replay test.
extern "C" void fvk_attention_fa2_fwd_bf16_seqused_splitkv(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream)
{
    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params, true, q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
                batch, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride, softmax_scale);
    params.seqused_k = reinterpret_cast<int*>(const_cast<void*>(seqused_k_ptr));
    int num_splits = setup_splitkv(params, softmax_lse_accum_ptr, o_accum_ptr,
                                    num_sms, seqlen_q, seqlen_k,
                                    head_dim, batch, num_heads_q);
    dispatch_hdim<cutlass::bfloat16_t>(head_dim, num_splits, params, stream);
}
#else
DEFINE_FA2_STUB(fvk_attention_fa2_fwd_bf16,  "bf16")
extern "C" void fvk_attention_fa2_fwd_bf16_seqused(
    const void*, const void*, const void*, void*, void*, const void*,
    int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, float, int, cudaStream_t)
{ std::abort(); }
extern "C" void fvk_attention_fa2_fwd_bf16_seqused_splitkv(
    const void*, const void*, const void*, void*, void*, const void*,
    void*, void*,
    int, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, float, int, cudaStream_t)
{ std::abort(); }
#endif
