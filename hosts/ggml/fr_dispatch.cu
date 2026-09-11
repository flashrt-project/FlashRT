// Dispatch glue between ggml-cuda's mul_mat and the FlashRT NVFP4 kernels.
// Host-only logic; the device kernels live in the sibling fr_*.cu files.

#include "fr_ggml.cuh"
#include "fr_kernels.h"

// FlashRT kernels consumed directly from the flashrt-public csrc tree
// (GGML_CUDA_FLASHRT_PUBLIC_DIR); no vendored copies.
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_f32out_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_bias_f32b_f16out_sm100.cuh"

#include <mutex>
#include <unordered_map>

namespace {

// Weights repacked into the CUTLASS wire format, keyed by the ggml tensor's
// device pointer. Weight tensors are immutable and live for the process
// lifetime, so entries are never evicted.
struct repacked_weight {
    void * packed = nullptr;
    void * sf     = nullptr;
};

std::unordered_map<const void *, repacked_weight> g_repack_cache;
std::mutex g_repack_mu;

// The repack allocates with cudaMalloc, which is illegal during CUDA graph
// capture. All weights are repacked during the first (uncaptured) warmup
// evaluation of each graph, so a cache miss while capturing indicates a bug.
const repacked_weight * get_repacked(const ggml_tensor * src0, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    auto it = g_repack_cache.find(src0->data);
    if (it != g_repack_cache.end()) {
        return &it->second;
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: weight repack for %s requested during CUDA graph capture", src0->name);
    }

    repacked_weight w;
    CUDA_CHECK(cudaMalloc(&w.packed, ggml_cuda_flashrt::packed_bytes(N, K)));
    CUDA_CHECK(cudaMalloc(&w.sf,     ggml_cuda_flashrt::sf_bytes(N, K)));

    const int rc = ggml_cuda_flashrt::repack_weight(src0->data, w.packed, w.sf, (int) N, (int) K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: weight repack failed for %s (N=%lld K=%lld rc=%d)", src0->name, (long long) N, (long long) K, rc);
    }

    auto res = g_repack_cache.emplace(src0->data, w);
    return &res.first->second;
}

// Interleaved gate/up weight pairs for the fused GeGLU GEMM, keyed by the
// two tensors' device pointers (same immortality caveat as above).
struct pair_key {
    const void * gate;
    const void * up;
    bool operator==(const pair_key & o) const { return gate == o.gate && up == o.up; }
};
struct pair_key_hash {
    size_t operator()(const pair_key & k) const noexcept {
        return std::hash<const void *>()(k.gate) ^ (std::hash<const void *>()(k.up) << 1);
    }
};

std::unordered_map<pair_key, repacked_weight, pair_key_hash> g_pair_cache;

const repacked_weight * get_repacked_pair(const ggml_tensor * gate_w, const ggml_tensor * up_w, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    pair_key key{gate_w->data, up_w->data};
    auto it = g_pair_cache.find(key);
    if (it != g_pair_cache.end()) {
        return &it->second;
    }

    const int64_t K    = gate_w->ne[0];
    const int64_t n_ff = gate_w->ne[1];
    const int64_t N_il = 2 * n_ff;

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: geglu pair repack for %s requested during CUDA graph capture", gate_w->name);
    }

    repacked_weight w;
    CUDA_CHECK(cudaMalloc(&w.packed, ggml_cuda_flashrt::packed_bytes(N_il, K)));
    CUDA_CHECK(cudaMalloc(&w.sf,     ggml_cuda_flashrt::sf_bytes(N_il, K)));

    const int rc = ggml_cuda_flashrt::repack_weight_pair_interleaved(
        gate_w->data, up_w->data, w.packed, w.sf, (int) n_ff, (int) K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu pair repack failed for %s (n_ff=%lld K=%lld rc=%d)",
                   gate_w->name, (long long) n_ff, (long long) K, rc);
    }

    auto res = g_pair_cache.emplace(key, w);
    return &res.first->second;
}

// Per-evaluation quantized-activation cache. Several ops consume the same
// fp32 activation tensor (q/k/v projections, the adaLN conditioning vector
// across all layers); quantizing it once per graph evaluation removes the
// duplicate quantize launches. Keys use the ggml tensor pointer (unique
// within one evaluation) plus an evaluation counter, so recycled device
// addresses across graphs can never alias. Slot buffers are grow-only and
// never freed, which keeps addresses stable for captured CUDA graphs; a
// replayed graph rewrites any slot before its baked consumers read it.
struct act_slot {
    const ggml_tensor * key = nullptr;
    uint64_t eval_id = 0;
    void * packed = nullptr;
    size_t packed_cap = 0;
    void * sf = nullptr;
    size_t sf_cap = 0;
};

act_slot g_act_slots[4];
int      g_act_slot_rr = 0;
uint64_t g_eval_id = 1;

// Returns cached (packed, sf) for src1 quantized as [M, K], quantizing on a
// miss. Returns false when the cache cannot be used (slot growth needed
// while capturing a CUDA graph); the caller must quantize into pool memory.
bool get_quantized_act(const ggml_tensor * src1, int M, int K,
                       const void ** out_packed, const void ** out_sf,
                       cudaStream_t stream) {
    for (auto & s : g_act_slots) {
        if (s.key == src1 && s.eval_id == g_eval_id) {
            *out_packed = s.packed;
            *out_sf     = s.sf;
            return true;
        }
    }

    act_slot & s = g_act_slots[g_act_slot_rr];
    const size_t need_packed = (size_t) ggml_cuda_flashrt::packed_bytes(M, K);
    const size_t need_sf     = (size_t) ggml_cuda_flashrt::sf_bytes(M, K);

    if (need_packed > s.packed_cap || need_sf > s.sf_cap) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            return false;
        }
        if (need_packed > s.packed_cap) {
            if (s.packed != nullptr) { cudaFree(s.packed); }
            CUDA_CHECK(cudaMalloc(&s.packed, need_packed));
            s.packed_cap = need_packed;
        }
        if (need_sf > s.sf_cap) {
            if (s.sf != nullptr) { cudaFree(s.sf); }
            CUDA_CHECK(cudaMalloc(&s.sf, need_sf));
            s.sf_cap = need_sf;
        }
    }
    g_act_slot_rr = (g_act_slot_rr + 1) % 4;

    const int rc = ggml_cuda_flashrt::quantize_act_f32(
        (const float *) src1->data, s.packed, s.sf, M, K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: activation quantize failed (M=%d K=%d rc=%d)", M, K, rc);
    }
    s.key     = src1;
    s.eval_id = g_eval_id;
    *out_packed = s.packed;
    *out_sf     = s.sf;
    return true;
}

// Reserve a cache slot for an activation that a producer kernel will fill
// with already-quantized data (fused quantize). Returns false when slot
// growth would be needed during CUDA graph capture.
bool reserve_quantized_act(const ggml_tensor * out_tensor, int M, int K,
                           void ** out_packed, void ** out_sf,
                           cudaStream_t stream) {
    act_slot & s = g_act_slots[g_act_slot_rr];
    const size_t need_packed = (size_t) ggml_cuda_flashrt::packed_bytes(M, K);
    const size_t need_sf     = (size_t) ggml_cuda_flashrt::sf_bytes(M, K);

    if (need_packed > s.packed_cap || need_sf > s.sf_cap) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            return false;
        }
        if (need_packed > s.packed_cap) {
            if (s.packed != nullptr) { cudaFree(s.packed); }
            CUDA_CHECK(cudaMalloc(&s.packed, need_packed));
            s.packed_cap = need_packed;
        }
        if (need_sf > s.sf_cap) {
            if (s.sf != nullptr) { cudaFree(s.sf); }
            CUDA_CHECK(cudaMalloc(&s.sf, need_sf));
            s.sf_cap = need_sf;
        }
    }
    g_act_slot_rr = (g_act_slot_rr + 1) % 4;

    s.key     = out_tensor;
    s.eval_id = g_eval_id;
    *out_packed = s.packed;
    *out_sf     = s.sf;
    return true;
}

// One-shot f16 Q handoff from the fused decode QKV window to the decomposed
// decode attention: qkv_post writes the rope'd+scaled Q rows as f16 in the
// t-major gather order, and the next attention window consumes them instead
// of running its own gather kernel. A single grow-only slot suffices (the
// producer and consumer alternate strictly within each layer); the key is
// cleared on consumption so a recycled activation address can never alias a
// stale entry.
struct q16_slot {
    const void * key = nullptr;   // data pointer of the Q tensor written for
    uint64_t eval_id = 0;
    void * buf = nullptr;
    size_t cap = 0;
};
q16_slot g_q16;

void * reserve_q16(const void * qdata, size_t bytes, cudaStream_t stream) {
    if (bytes > g_q16.cap) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            return nullptr;
        }
        if (g_q16.buf != nullptr) { cudaFree(g_q16.buf); }
        CUDA_CHECK(cudaMalloc(&g_q16.buf, bytes));
        g_q16.cap = bytes;
    }
    g_q16.key     = qdata;
    g_q16.eval_id = g_eval_id;
    return g_q16.buf;
}

// Grow-only device buffer for the never-written D of the no-D-store GeGLU
// variants (the host-side TMA descriptor still needs a valid allocation).
void * get_dummy_d(size_t bytes) {
    static void * buf = nullptr;
    static size_t cap = 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (bytes > cap) {
        if (buf != nullptr) {
            cudaFree(buf);
        }
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        cap = bytes;
    }
    return buf;
}

} // namespace

bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_FLASHRT_DISABLE") != nullptr;
    if (disabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // Batched src1 with unbatched (broadcast) weights folds into a single
    // GEMM over all rows because src1/dst are fully contiguous.
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (K % 64 != 0 || N % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) ggml_nrows(src1); // batch dims fold into rows (contiguous, broadcast weights)

    cudaStream_t stream = ctx.stream();

    // The persistent repack cache is keyed by the weight tensor's device
    // pointer, which is only sound when weight tensors are immortal (model
    // inference). Tools that create and free tensors at recycled addresses
    // (e.g. test-backend-ops) must set GGML_CUDA_FLASHRT_NO_CACHE=1 to
    // repack into scratch memory on every call instead.
    static const bool no_cache = getenv("GGML_CUDA_FLASHRT_NO_CACHE") != nullptr;

    ggml_cuda_pool_alloc<uint8_t> b_packed_scratch(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> b_sf_scratch    (ctx.pool());

    const void * b_packed = nullptr;
    const void * b_sf     = nullptr;
    if (no_cache) {
        b_packed_scratch.alloc(ggml_cuda_flashrt::packed_bytes(N, K));
        b_sf_scratch.alloc(ggml_cuda_flashrt::sf_bytes(N, K));
        const int rrc = ggml_cuda_flashrt::repack_weight(src0->data, b_packed_scratch.get(), b_sf_scratch.get(), N, K, stream);
        if (rrc != 0) {
            GGML_ABORT("flashrt: weight repack failed (N=%d K=%d rc=%d)", N, K, rrc);
        }
        b_packed = b_packed_scratch.get();
        b_sf     = b_sf_scratch.get();
    } else {
        const repacked_weight * w = get_repacked(src0, stream);
        b_packed = w->packed;
        b_sf     = w->sf;
    }

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        const int qrc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (qrc != 0) {
            GGML_ABORT("flashrt: activation quantize failed (M=%d K=%d rc=%d)", M, K, qrc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    // ggml's block_nvfp4 scale bytes carry standard e4m3 semantics: its
    // dequant table doubles the e2m1 values but its ue4m3 decode halves the
    // scale, so the two cancel and no alpha compensation is needed.
    const float alpha = 1.0f;
    // The 128x128x256 tile beats the wide-N 128x256x128 tile on Thor for
    // every production shape measured (including N=16384 prefill FFN).
    const bool  widen = false;

    const int rc = ggml_cuda_flashrt::gemm_f32out(
        q_packed, q_sf, b_packed, b_sf,
        (float *) dst->data, M, N, K, alpha, widen, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: gemm failed (M=%d N=%d K=%d rc=%d)", M, N, K, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_ada(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * bias_add,
                                       const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
                                       const ggml_tensor * mul, const ggml_tensor * add1,
                                       const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
                                       const ggml_tensor * add2) {
    const ggml_tensor * x = rms != nullptr ? rms->src[0] : mul->src[0];
    const ggml_tensor * normed = rms != nullptr ? rms : x;
    const int64_t C = x->ne[0];
    const int64_t M = x->ne[1];

    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    // modulation projection: [3C, 1] from the shared conditioning vector
    if (!ggml_cuda_flashrt_should_use(mm->src[0], mm->src[1], mm) ||
        mm->ne[0] != 3 * C || mm->ne[1] != 1) {
        return false;
    }
    const ggml_tensor * bias = bias_add->src[1];
    if (bias_add->src[0] != mm || bias->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(bias) || bias->ne[0] != 3 * C || !ggml_is_contiguous(bias_add)) {
        return false;
    }
    // scale = mod[0:C], shift = mod[C:2C]
    if (view_scale->src[0] != bias_add || view_scale->ne[0] != C || view_scale->ne[1] != 1 ||
        view_scale->view_offs != 0) {
        return false;
    }
    if (view_shift->src[0] != bias_add || view_shift->ne[0] != C || view_shift->ne[1] != 1 ||
        view_shift->view_offs != (size_t) C * sizeof(float)) {
        return false;
    }
    if (repeat_scale->src[0] != view_scale || repeat_shift->src[0] != view_shift ||
        repeat_scale->ne[0] != C || repeat_scale->ne[1] != M) {
        return false;
    }
    // t = normed * scale; t2 = normed + t; out = t2 + shift
    if (mul->src[0] != normed || mul->src[1] != repeat_scale ||
        add1->src[0] != normed || add1->src[1] != mul ||
        add2->src[0] != add1 || add2->src[1] != repeat_shift ||
        !ggml_is_contiguous(add2) || add2->ne[0] != C || add2->ne[1] != M) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_ada_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mm,
                                ggml_tensor * bias_add, const ggml_tensor * view_scale, const ggml_tensor * mul,
                                const ggml_tensor * view_shift, ggml_tensor * add2) {
    const ggml_tensor * x    = rms != nullptr ? rms->src[0] : mul->src[0];
    const ggml_tensor * cond = mm->src[1];

    const int C = (int) x->ne[0];
    const int M = (int) x->ne[1];
    const int K = (int) mm->src[0]->ne[0];
    const int N = (int) mm->src[0]->ne[1]; // 3C

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w = get_repacked(mm->src[0], stream);

    ggml_cuda_pool_alloc<uint8_t> c_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> c_sf    (ctx.pool());

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    int rc = 0;
    if (!get_quantized_act(cond, 1, K, &q_packed, &q_sf, stream)) {
        c_packed.alloc(ggml_cuda_flashrt::packed_bytes(1, K));
        c_sf.alloc(ggml_cuda_flashrt::sf_bytes(1, K));
        rc = ggml_cuda_flashrt::quantize_act_f32((const float *) cond->data, c_packed.get(), c_sf.get(), 1, K, stream);
        q_packed = c_packed.get();
        q_sf     = c_sf.get();
    }
    if (rc == 0) {
        // bias applied in the GEMM epilogue, writing the biased modulation
        // vector (still read later through the gate view) directly
        rc = flash_rt::fp4::gemm_bias_f32out(q_packed, q_sf, w->packed, w->sf,
                                                 bias_add->src[1]->data,
                                                 (float *) bias_add->data, 1, N, K, stream);
    }
    if (rc == 0) {
        const float eps = rms != nullptr ? ggml_get_op_params_f32(rms, 0) : 0.0f;
        // emit the quantized form alongside f32 so downstream GEMMs skip
        // their activation quantize (registered in the per-eval cache)
        void * q_out_packed = nullptr;
        void * q_out_sf     = nullptr;
        if (C % 16 == 0 && reserve_quantized_act(add2, M, C, &q_out_packed, &q_out_sf, stream)) {
            rc = ggml_cuda_flashrt::ada_rms_mod_quant((const float *) x->data,
                                                      (const float *) view_scale->data,
                                                      (const float *) view_shift->data,
                                                      (float *) add2->data, q_out_packed, q_out_sf,
                                                      M, C, eps, rms != nullptr, stream);
        } else {
            rc = ggml_cuda_flashrt::ada_rms_mod((const float *) x->data,
                                                (const float *) view_scale->data,
                                                (const float *) view_shift->data,
                                                (float *) add2->data, M, C, eps, rms != nullptr, stream);
        }
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: fused adaLN failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

// Cached-modulation adaLN window: the modulation vector comes from a graph
// input (precomputed per step, see pi0_ae.cpp) instead of a GEMM. Sequence:
// {RMS_NORM, VIEW mod-col, VIEW scale, REPEAT, MUL, ADD, VIEW shift, REPEAT,
// ADD} -> one ada kernel reading scale/shift straight from the input views.
static int g_ada_cached_fail = 0;
#define FR_ADA_FAIL(code) do { g_ada_cached_fail = (code); return false; } while (0)

bool ggml_cuda_flashrt_should_fuse_ada_cached(
        const ggml_tensor * rms, const ggml_tensor * view_col,
        const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
        const ggml_tensor * mul, const ggml_tensor * add1,
        const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
        const ggml_tensor * add2) {
    static const bool disabled = getenv("GGML_CUDA_FLASHRT_DISABLE") != nullptr;
    if (disabled) {
        FR_ADA_FAIL(1);
    }
    const ggml_tensor * x = rms->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        FR_ADA_FAIL(2);
    }
    const int64_t C = x->ne[0];
    const int64_t M = x->ne[1];
    if (x->ne[2] != 1 || x->ne[3] != 1 || C % 4 != 0) {
        FR_ADA_FAIL(3);
    }
    if (view_col->type != GGML_TYPE_F32 || view_col->ne[0] != 3 * C || view_col->ne[1] != 1 ||
        view_col->src[0] == nullptr) {
        FR_ADA_FAIL(4);
    }
    // view_offs accumulates through view-of-view chains down to the ultimate
    // source, so compare offsets relative to the enclosing column view
    if (view_scale->src[0] != view_col || view_scale->ne[0] != C || view_scale->ne[1] != 1 ||
        view_scale->view_offs != view_col->view_offs) {
        if (getenv("GGML_FLASHRT_DEBUG") != nullptr) {
            static int dbg5 = 0;
            if (dbg5++ < 4) {
                fprintf(stderr, "[fr-ada-c5] src_ok=%d ne0=%lld (C=%lld) ne1=%lld offs=%zu col_offs=%zu\n",
                        (int) (view_scale->src[0] == view_col), (long long) view_scale->ne[0],
                        (long long) C, (long long) view_scale->ne[1],
                        view_scale->view_offs, view_col->view_offs);
            }
        }
        FR_ADA_FAIL(5);
    }
    if (view_shift->src[0] != view_col || view_shift->ne[0] != C || view_shift->ne[1] != 1 ||
        view_shift->view_offs != view_col->view_offs + (size_t) C * sizeof(float)) {
        FR_ADA_FAIL(6);
    }
    if (repeat_scale->src[0] != view_scale || repeat_scale->ne[0] != C || repeat_scale->ne[1] != M ||
        repeat_shift->src[0] != view_shift || repeat_shift->ne[0] != C || repeat_shift->ne[1] != M) {
        FR_ADA_FAIL(7);
    }
    if (mul->src[0] != rms || mul->src[1] != repeat_scale ||
        add1->src[0] != rms || add1->src[1] != mul ||
        add2->src[0] != add1 || add2->src[1] != repeat_shift ||
        !ggml_is_contiguous(add2) || add2->ne[0] != C || add2->ne[1] != M) {
        FR_ADA_FAIL(8);
    }
    g_ada_cached_fail = 0;
    return true;
}

int ggml_cuda_flashrt_ada_cached_fail_code() { return g_ada_cached_fail; }

void ggml_cuda_flashrt_ada_norm_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * rms,
                                       const ggml_tensor * view_scale, const ggml_tensor * view_shift,
                                       ggml_tensor * add2) {
    const ggml_tensor * x = rms->src[0];
    const int C = (int) x->ne[0];
    const int M = (int) x->ne[1];
    const float eps = ggml_get_op_params_f32(rms, 0);
    cudaStream_t stream = ctx.stream();

    int rc;
    void * q_out_packed = nullptr;
    void * q_out_sf     = nullptr;
    if (C % 16 == 0 && reserve_quantized_act(add2, M, C, &q_out_packed, &q_out_sf, stream)) {
        rc = ggml_cuda_flashrt::ada_rms_mod_quant((const float *) x->data,
                                                  (const float *) view_scale->data,
                                                  (const float *) view_shift->data,
                                                  (float *) add2->data, q_out_packed, q_out_sf,
                                                  M, C, eps, true, stream);
    } else {
        rc = ggml_cuda_flashrt::ada_rms_mod((const float *) x->data,
                                            (const float *) view_scale->data,
                                            (const float *) view_shift->data,
                                            (float *) add2->data, M, C, eps, true, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: cached adaLN failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_gated_res(const ggml_tensor * view, const ggml_tensor * repeat,
                                             const ggml_tensor * mul, const ggml_tensor * add) {
    const int64_t C = view->ne[0];
    const int64_t M = mul->ne[1];
    if (view->type != GGML_TYPE_F32 || view->ne[1] != 1 ||
        view->src[0] == nullptr || view->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (repeat->src[0] != view || repeat->ne[0] != C || repeat->ne[1] != M) {
        return false;
    }
    const ggml_tensor * branch = mul->src[0];
    if (mul->src[1] != repeat || branch->type != GGML_TYPE_F32 || !ggml_is_contiguous(branch) ||
        branch->ne[0] != C || branch->ne[1] != M) {
        return false;
    }
    const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
    if ((add->src[0] != mul && add->src[1] != mul) || residual == mul ||
        residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
        residual->ne[0] != C || residual->ne[1] != M ||
        !ggml_is_contiguous(add) || add->ne[0] != C || add->ne[1] != M) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_gated_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * view,
                                      const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * branch   = mul->src[0];
    const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
    const int C = (int) view->ne[0];
    const int M = (int) mul->ne[1];

    const int rc = ggml_cuda_flashrt::gated_residual(
        (const float *) residual->data, (const float *) branch->data,
        (const float *) view->data, (float *) add->data, M, C, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: fused gated residual failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

// SigLIP FFN weights prepared for the fused FP4 pair, keyed by the two
// weight pointers. Up rows are padded to a 32 multiple for the FP4 output;
// the f16 Down weight is quantized with its K padded to the same value.
struct siglip_ffn_weights {
    void * up_packed = nullptr;
    void * up_sf     = nullptr;
    void * dn_packed = nullptr;
    void * dn_sf     = nullptr;
    int    h_pad     = 0;
};

std::unordered_map<pair_key, siglip_ffn_weights, pair_key_hash> g_sig_ffn_cache;

const siglip_ffn_weights * get_siglip_ffn_weights(const ggml_tensor * up_w, const ggml_tensor * dn_w, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    pair_key key{up_w->data, dn_w->data};
    auto it = g_sig_ffn_cache.find(key);
    if (it != g_sig_ffn_cache.end()) {
        return &it->second;
    }

    const int64_t K_in  = up_w->ne[0];   // model dim (NVFP4 up weight)
    const int64_t H     = up_w->ne[1];   // hidden dim
    const int64_t H_pad = GGML_PAD(H, 64);
    const int64_t D_out = dn_w->ne[1];   // model dim (f16 down weight, K = H)

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: siglip ffn weight prep for %s requested during CUDA graph capture", up_w->name);
    }

    siglip_ffn_weights w;
    w.h_pad = (int) H_pad;
    CUDA_CHECK(cudaMalloc(&w.up_packed, ggml_cuda_flashrt::packed_bytes(H_pad, K_in)));
    CUDA_CHECK(cudaMalloc(&w.up_sf,     ggml_cuda_flashrt::sf_bytes(H_pad, K_in)));
    CUDA_CHECK(cudaMalloc(&w.dn_packed, ggml_cuda_flashrt::packed_bytes(D_out, H_pad)));
    CUDA_CHECK(cudaMalloc(&w.dn_sf,     ggml_cuda_flashrt::sf_bytes(D_out, H_pad)));

    int rc = ggml_cuda_flashrt::repack_weight_rows_padded(
        up_w->data, w.up_packed, w.up_sf, (int) H, (int) H_pad, (int) K_in, stream);
    if (rc == 0) {
        rc = ggml_cuda_flashrt::quantize_weight_f16_padded(
            dn_w->data, w.dn_packed, w.dn_sf, (int) D_out, (int) H, (int) H_pad, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: siglip ffn weight prep failed for %s (H=%lld rc=%d)", up_w->name, (long long) H, rc);
    }

    auto res = g_sig_ffn_cache.emplace(key, w);
    return &res.first->second;
}

bool ggml_cuda_flashrt_should_fuse_ln(const ggml_tensor * norm, const ggml_tensor * mul, const ggml_tensor * add) {
    const ggml_tensor * x = norm->src[0];
    const int64_t C = norm->ne[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || norm->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if ((mul->src[0] != norm && mul->src[1] != norm) || w == norm ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        w->ne[0] != C || ggml_nelements(w) != C) {
        return false;
    }
    const ggml_tensor * b = add->src[0] == mul ? add->src[1] : add->src[0];
    if ((add->src[0] != mul && add->src[1] != mul) || b == mul ||
        b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) ||
        b->ne[0] != C || ggml_nelements(b) != C) {
        return false;
    }
    if (!ggml_is_contiguous(add) || add->ne[0] != C || ggml_nrows(add) != ggml_nrows(x)) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_ln_affine(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * b = add->src[0] == mul ? add->src[1] : add->src[0];

    const int C = (int) norm->ne[0];
    const int M = (int) ggml_nrows(x);
    const float eps = ggml_get_op_params_f32(norm, 0);

    int rc;
    void * q_out_packed = nullptr;
    void * q_out_sf     = nullptr;
    if (C % 16 == 0 && reserve_quantized_act(add, M, C, &q_out_packed, &q_out_sf, ctx.stream())) {
        rc = ggml_cuda_flashrt::layer_norm_affine_quant(
            (const float *) x->data, (const float *) w->data, (const float *) b->data,
            (float *) add->data, q_out_packed, q_out_sf, M, C, eps, ctx.stream());
    } else {
        rc = ggml_cuda_flashrt::layer_norm_affine(
            (const float *) x->data, (const float *) w->data, (const float *) b->data,
            (float *) add->data, M, C, eps, ctx.stream());
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: fused layer norm failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_geglu(const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, const ggml_tensor * down_mm) {
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_GEGLU) {
        return false;
    }
    // both projections share the same input and shape
    if (gate_mm->src[1] != up_mm->src[1]) {
        return false;
    }
    const ggml_tensor * gate_w = gate_mm->src[0];
    const ggml_tensor * up_w   = up_mm->src[0];
    const ggml_tensor * down_w = down_mm->src[0];
    if (gate_w->ne[0] != up_w->ne[0] || gate_w->ne[1] != up_w->ne[1]) {
        return false;
    }
    if (!ggml_cuda_flashrt_should_use(gate_w, gate_mm->src[1], gate_mm) ||
        !ggml_cuda_flashrt_should_use(up_w,   up_mm->src[1],   up_mm)   ||
        !ggml_cuda_flashrt_should_use(down_w, glu,             down_mm)) {
        return false;
    }
    // the down projection must consume the GLU output with K = n_ff
    if (down_mm->src[1] != glu || down_w->ne[0] != gate_w->ne[1]) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_geglu_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, ggml_tensor * down_mm) {
    GGML_UNUSED(glu);

    const ggml_tensor * src1   = gate_mm->src[1];
    const ggml_tensor * gate_w = gate_mm->src[0];
    const ggml_tensor * down_w = down_mm->src[0];

    const int K    = (int) gate_w->ne[0];
    const int n_ff = (int) gate_w->ne[1];
    const int N_il = 2 * n_ff;
    const int M    = (int) ggml_nrows(src1);
    const int N_out = (int) down_w->ne[1];

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w_il   = get_repacked_pair(gate_w, up_mm->src[0], stream);
    const repacked_weight * w_down = get_repacked(down_w, stream);

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    int rc = 0;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        rc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (rc != 0) {
            GGML_ABORT("flashrt: geglu activation quantize failed (M=%d K=%d rc=%d)", M, K, rc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    ggml_cuda_pool_alloc<uint8_t> compact_packed(ctx.pool(), ggml_cuda_flashrt::packed_bytes(M, n_ff));
    ggml_cuda_pool_alloc<uint8_t> compact_sfa   (ctx.pool(), ggml_cuda_flashrt::sf_bytes(M, n_ff));
    void * dummy_d = get_dummy_d(ggml_cuda_flashrt::packed_bytes(M, N_il));

    // skinny-M tile for decode-sized batches, default tile otherwise
    if (M < 128) {
        rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod_v10(
            q_packed, q_sf, w_il->packed, w_il->sf,
            dummy_d, compact_packed.get(), compact_sfa.get(), M, N_il, K, stream);
    } else {
        rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod(
            q_packed, q_sf, w_il->packed, w_il->sf,
            dummy_d, compact_packed.get(), compact_sfa.get(), M, N_il, K, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu gemm failed (M=%d N_il=%d K=%d rc=%d)", M, N_il, K, rc);
    }

    rc = ggml_cuda_flashrt::gemm_f32out(
        compact_packed.get(), compact_sfa.get(), w_down->packed, w_down->sf,
        (float *) down_mm->data, M, N_out, n_ff, /*alpha=*/1.0f, /*widen=*/false, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu down gemm failed (M=%d N=%d K=%d rc=%d)", M, N_out, n_ff, rc);
    }
}

// Fused QKV weights (row-concat [k | v | q]) keyed by the three pointers.
struct triple_key {
    const void * a; const void * b; const void * c;
    bool operator==(const triple_key & o) const { return a == o.a && b == o.b && c == o.c; }
};
struct triple_key_hash {
    size_t operator()(const triple_key & k) const noexcept {
        return std::hash<const void *>()(k.a) ^ (std::hash<const void *>()(k.b) << 1) ^ (std::hash<const void *>()(k.c) << 2);
    }
};

std::unordered_map<triple_key, repacked_weight, triple_key_hash> g_qkv_cache;

const repacked_weight * get_repacked_qkv(const ggml_tensor * wk, const ggml_tensor * wv, const ggml_tensor * wq, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    triple_key key{wk->data, wv->data, wq->data};
    auto it = g_qkv_cache.find(key);
    if (it != g_qkv_cache.end()) {
        return &it->second;
    }

    const int64_t K = wk->ne[0];
    const int64_t N_tot = wk->ne[1] + wv->ne[1] + wq->ne[1];

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: qkv repack for %s requested during CUDA graph capture", wq->name);
    }

    repacked_weight w;
    CUDA_CHECK(cudaMalloc(&w.packed, ggml_cuda_flashrt::packed_bytes(N_tot, K)));
    CUDA_CHECK(cudaMalloc(&w.sf,     ggml_cuda_flashrt::sf_bytes(N_tot, K)));

    const int rc = ggml_cuda_flashrt::repack_weight_concat3(
        wk->data, (int) wk->ne[1], wv->data, (int) wv->ne[1], wq->data, (int) wq->ne[1],
        w.packed, w.sf, (int) K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: qkv repack failed for %s (rc=%d)", wq->name, rc);
    }

    auto res = g_qkv_cache.emplace(key, w);
    return &res.first->second;
}

bool ggml_cuda_flashrt_should_fuse_siglip_ffn(const ggml_tensor * up_mm, const ggml_tensor * bias1, const ggml_tensor * gelu,
                                              const ggml_tensor * cont1, const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                              const ggml_tensor * cont2, const ggml_tensor * res_add) {
    const ggml_tensor * up_w = up_mm->src[0];
    const ggml_tensor * src1 = up_mm->src[1];
    const ggml_tensor * dn_w = dn_mm->src[0];

    if (!ggml_cuda_flashrt_should_use(up_w, src1, up_mm)) {
        return false;
    }
    if (dn_w->type != GGML_TYPE_F16 || !ggml_is_contiguous(dn_w) ||
        dn_w->ne[0] != up_w->ne[1] || dn_w->ne[0] % 16 != 0 ||
        dn_w->ne[1] % 16 != 0 || dn_w->ne[2] != 1) {
        return false;
    }
    if (ggml_get_unary_op(gelu) != GGML_UNARY_OP_GELU || gelu->src[0] != bias1) {
        return false;
    }
    const ggml_tensor * b1 = bias1->src[0] == up_mm ? bias1->src[1] : bias1->src[0];
    if ((bias1->src[0] != up_mm && bias1->src[1] != up_mm) || b1 == up_mm ||
        b1->type != GGML_TYPE_F32 || !ggml_is_contiguous(b1) || ggml_nelements(b1) != up_w->ne[1]) {
        return false;
    }
    if (cont1->src[0] != gelu || dn_mm->src[1] != cont1) {
        return false;
    }
    const ggml_tensor * b2 = bias2->src[0] == dn_mm ? bias2->src[1] : bias2->src[0];
    if ((bias2->src[0] != dn_mm && bias2->src[1] != dn_mm) || b2 == dn_mm ||
        b2->type != GGML_TYPE_F32 || !ggml_is_contiguous(b2) || ggml_nelements(b2) != dn_w->ne[1]) {
        return false;
    }
    if (cont2->src[0] != bias2) {
        return false;
    }
    const ggml_tensor * residual = res_add->src[0] == cont2 ? res_add->src[1] : res_add->src[0];
    if ((res_add->src[0] != cont2 && res_add->src[1] != cont2) || residual == cont2 ||
        residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
        !ggml_is_contiguous(res_add) || res_add->ne[0] != dn_w->ne[1] ||
        ggml_nrows(res_add) != ggml_nrows(src1) || ggml_nrows(residual) != ggml_nrows(src1)) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_siglip_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * up_mm, const ggml_tensor * bias1,
                                  const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                  const ggml_tensor * cont2, ggml_tensor * res_add) {
    const ggml_tensor * up_w = up_mm->src[0];
    const ggml_tensor * src1 = up_mm->src[1];
    const ggml_tensor * dn_w = dn_mm->src[0];
    const ggml_tensor * b1 = bias1->src[0] == up_mm ? bias1->src[1] : bias1->src[0];
    const ggml_tensor * b2 = bias2->src[0] == dn_mm ? bias2->src[1] : bias2->src[0];
    const ggml_tensor * residual = res_add->src[0] == cont2 ? res_add->src[1] : res_add->src[0];

    const int K_in  = (int) up_w->ne[0];
    const int D_out = (int) dn_w->ne[1];
    const int M     = (int) ggml_nrows(src1);

    cudaStream_t stream = ctx.stream();

    const siglip_ffn_weights * w = get_siglip_ffn_weights(up_w, dn_w, stream);
    const int H_pad = w->h_pad;

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());
    int rc = 0;
    if (!get_quantized_act(src1, M, K_in, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K_in));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K_in));
        rc = ggml_cuda_flashrt::quantize_act_f32((const float *) src1->data, a_packed.get(), a_sf.get(), M, K_in, stream);
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    ggml_cuda_pool_alloc<uint8_t> hid_packed(ctx.pool(), ggml_cuda_flashrt::packed_bytes(M, H_pad));
    ggml_cuda_pool_alloc<uint8_t> hid_sf    (ctx.pool(), ggml_cuda_flashrt::sf_bytes(M, H_pad));

    if (rc == 0) {
        rc = flash_rt::fp4::siglip_ffn_up_gelu_fp4out(
            q_packed, q_sf, w->up_packed, w->up_sf, b1->data,
            hid_packed.get(), hid_sf.get(), M, H_pad, K_in, stream);
    }
    if (rc == 0) {
        rc = flash_rt::fp4::siglip_ffn_down_bias_res_f32(
            hid_packed.get(), hid_sf.get(), w->dn_packed, w->dn_sf, b2->data,
            residual->data, res_add->data, M, D_out, H_pad, stream, 1.0f);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: siglip ffn failed (M=%d H_pad=%d rc=%d)", M, H_pad, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_qkv(const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                                       const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                                       const ggml_tensor * q_mm, const ggml_tensor * q_rope, const ggml_tensor * q_scale) {
    const ggml_tensor * src1 = k_mm->src[1];
    if (v_mm->src[1] != src1 || q_mm->src[1] != src1) {
        return false;
    }
    const ggml_tensor * wk = k_mm->src[0];
    const ggml_tensor * wv = v_mm->src[0];
    const ggml_tensor * wq = q_mm->src[0];
    if (!ggml_cuda_flashrt_should_use(wk, src1, k_mm) ||
        !ggml_cuda_flashrt_should_use(wv, src1, v_mm) ||
        !ggml_cuda_flashrt_should_use(wq, src1, q_mm) ||
        wk->ne[0] != wv->ne[0] || wk->ne[0] != wq->ne[0]) {
        return false;
    }
    // K path: mm -> reshape -> rope -> cpy into an f16 view of the
    // persistent KV buffer; single KV head (Nk == head_dim)
    const int64_t head_dim = k_rope->src[0]->ne[0];
    if (k_rope->src[0]->op != GGML_OP_RESHAPE || k_rope->src[0]->src[0] != k_mm ||
        wk->ne[1] != head_dim || k_rope->src[0]->ne[1] != 1) {
        return false;
    }
    if (k_cpy->src[0] != k_rope || k_cpy->src[1] == nullptr ||
        k_cpy->src[1]->type != GGML_TYPE_F16 || k_cpy->src[1]->op != GGML_OP_VIEW) {
        return false;
    }
    // V path: mm -> reshape -> cpy into f16 view
    if (v_cpy->src[0] == nullptr || v_cpy->src[0]->op != GGML_OP_RESHAPE ||
        v_cpy->src[0]->src[0] != v_mm || wv->ne[1] != head_dim ||
        v_cpy->src[1] == nullptr || v_cpy->src[1]->type != GGML_TYPE_F16 || v_cpy->src[1]->op != GGML_OP_VIEW) {
        return false;
    }
    // Q path: mm -> reshape -> rope -> scale, head_dim x n_head
    if (q_rope->src[0]->op != GGML_OP_RESHAPE || q_rope->src[0]->src[0] != q_mm ||
        q_rope->src[0]->ne[0] != head_dim || wq->ne[1] % head_dim != 0 ||
        q_scale->src[0] != q_rope || !ggml_is_contiguous(q_scale)) {
        return false;
    }
    // both ropes must share positions, freq factors and parameters
    if (k_rope->src[1] != q_rope->src[1] || k_rope->src[2] != q_rope->src[2] ||
        memcmp(k_rope->op_params, q_rope->op_params, sizeof(k_rope->op_params)) != 0) {
        return false;
    }
    // NEOX rope only (matches ggml's rope_neox math replicated in qkv_post)
    const int mode = ((const int32_t *) k_rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }
    // f16 KV views must be row-contiguous (head_dim elements per token row)
    const ggml_tensor * kv = k_cpy->src[1];
    if (kv->nb[0] != sizeof(uint16_t) || kv->nb[1] != head_dim * sizeof(uint16_t)) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_qkv(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                           const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                           const ggml_tensor * q_mm, const ggml_tensor * q_rope, ggml_tensor * q_scale) {
    const ggml_tensor * src1 = k_mm->src[1];
    const ggml_tensor * wk = k_mm->src[0];
    const ggml_tensor * wv = v_mm->src[0];
    const ggml_tensor * wq = q_mm->src[0];

    const int K  = (int) wk->ne[0];
    const int Nk = (int) wk->ne[1];
    const int Nv = (int) wv->ne[1];
    const int Nq = (int) wq->ne[1];
    const int M  = (int) ggml_nrows(src1);
    const int head_dim = Nk;

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w = get_repacked_qkv(wk, wv, wq, stream);

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());
    int rc = 0;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        rc = ggml_cuda_flashrt::quantize_act_f32((const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    const int N_tot = Nk + Nv + Nq;
    ggml_cuda_pool_alloc<float> qkv_cat(ctx.pool(), (int64_t) M * N_tot);

    if (rc == 0) {
        rc = ggml_cuda_flashrt::gemm_f32out(q_packed, q_sf, w->packed, w->sf,
                                            qkv_cat.get(), M, N_tot, K, 1.0f, false, stream);
    }
    if (rc == 0) {
        // rope parameters, mirrored from ggml-cuda's rope host setup
        const int32_t * op = (const int32_t *) k_rope->op_params;
        const int   n_dims     = op[1];
        const int   n_ctx_orig = op[4];
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        memcpy(&freq_base,   op +  5, sizeof(float));
        memcpy(&freq_scale,  op +  6, sizeof(float));
        memcpy(&ext_factor,  op +  7, sizeof(float));
        memcpy(&attn_factor, op +  8, sizeof(float));
        memcpy(&beta_fast,   op +  9, sizeof(float));
        memcpy(&beta_slow,   op + 10, sizeof(float));
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        const float theta_scale = powf(freq_base, -2.0f / n_dims);
        const float scale_f = ggml_get_op_params_f32(q_scale, 0);

        const ggml_tensor * ff = k_rope->src[2];
        void * q16 = reserve_q16(q_scale->data,
                                 (size_t) M * Nq * sizeof(uint16_t), stream);
        rc = ggml_cuda_flashrt::qkv_post(
            qkv_cat.get(), (float *) q_scale->data, q16,
            k_cpy->src[1]->data, v_cpy->src[1]->data,
            (const int32_t *) k_rope->src[1]->data,
            ff != nullptr ? (const float *) ff->data : nullptr,
            M, Nk, Nv, Nq, head_dim, n_dims,
            freq_scale, ext_factor, attn_factor,
            corr_dims[0], corr_dims[1], theta_scale, scale_f, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: fused qkv failed (M=%d N=%d K=%d rc=%d)", M, N_tot, K, rc);
    }
}

// ---------------------------------------------------------------------------
// Vision QKV pad window: {mul_mat, add bias, reshape}x3 + {pad}x3 -> three
// padded-weight GEMMs with the bias in the epilogue, writing the pad buffers
// directly. Widens per-head projections (SigLIP head_dim 72 -> 80) inside the
// repacked weights, so the runtime pad kernels and separate bias adds vanish.

namespace {

struct grouppad_weight {
    void * packed = nullptr;
    void * sf     = nullptr;
    void * bias   = nullptr; // f32 [group_out * n_groups], zero-interleaved
};

std::unordered_map<const void *, grouppad_weight> g_grouppad_cache;

const grouppad_weight * get_repacked_grouppad(
        const ggml_tensor * w, const ggml_tensor * bias,
        int group_in, int group_out, int n_groups, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    auto it = g_grouppad_cache.find(w->data);
    if (it != g_grouppad_cache.end()) {
        return &it->second;
    }

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: grouppad repack for %s requested during CUDA graph capture", w->name);
    }

    const int K     = (int) w->ne[0];
    const int N_pad = group_out * n_groups;

    grouppad_weight g;
    CUDA_CHECK(cudaMalloc(&g.packed, ggml_cuda_flashrt::packed_bytes(N_pad, K)));
    CUDA_CHECK(cudaMalloc(&g.sf,     ggml_cuda_flashrt::sf_bytes(N_pad, K)));
    CUDA_CHECK(cudaMalloc(&g.bias,   (size_t) N_pad * sizeof(float)));

    const int rc = ggml_cuda_flashrt::repack_weight_rows_grouppad(
        w->data, g.packed, g.sf, group_in, group_out, n_groups, K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: grouppad repack failed for %s (gi=%d go=%d ng=%d K=%d rc=%d)",
                   w->name, group_in, group_out, n_groups, K, rc);
    }
    CUDA_CHECK(cudaMemsetAsync(g.bias, 0, (size_t) N_pad * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpy2DAsync(g.bias, (size_t) group_out * sizeof(float),
                                 bias->data, (size_t) group_in * sizeof(float),
                                 (size_t) group_in * sizeof(float), (size_t) n_groups,
                                 cudaMemcpyDeviceToDevice, stream));

    auto res = g_grouppad_cache.emplace(w->data, g);
    return &res.first->second;
}

// One projection triple of the window: mul_mat -> add(bias) -> reshape -> pad.
bool vis_qkv_pad_leg_ok(const ggml_tensor * mm, const ggml_tensor * add,
                        const ggml_tensor * resh, const ggml_tensor * pad,
                        const ggml_tensor * shared_src1) {
    if (mm->op != GGML_OP_MUL_MAT || add->op != GGML_OP_ADD ||
        resh->op != GGML_OP_RESHAPE || pad->op != GGML_OP_PAD) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * b = add->src[1];
    if (w == nullptr || w->type != GGML_TYPE_NVFP4 || mm->src[1] != shared_src1) {
        return false;
    }
    if (add->src[0] != mm || resh->src[0] != add || pad->src[0] != resh) {
        return false;
    }
    if (b == nullptr || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) ||
        b->ne[0] != mm->ne[0] || ggml_nrows(b) != 1) {
        return false;
    }
    const int64_t group_in  = resh->ne[0];
    const int64_t n_groups  = resh->ne[1];
    const int64_t group_out = pad->ne[0];
    if (group_in * n_groups != mm->ne[0] || group_out <= group_in) {
        return false;
    }
    // pad only widens dim0
    if (pad->ne[1] != resh->ne[1] || pad->ne[2] != resh->ne[2] || pad->ne[3] != resh->ne[3]) {
        return false;
    }
    if (pad->type != GGML_TYPE_F32 || !ggml_is_contiguous(pad)) {
        return false;
    }
    const int64_t K     = w->ne[0];
    const int64_t N_pad = group_out * n_groups;
    if (K % 64 != 0 || N_pad % 16 != 0) {
        return false;
    }
    return true;
}

} // namespace

bool ggml_cuda_flashrt_should_fuse_vis_qkv_pad(
        const ggml_tensor * mm_q, const ggml_tensor * add_q, const ggml_tensor * resh_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, const ggml_tensor * resh_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, const ggml_tensor * resh_v,
        const ggml_tensor * pad_q, const ggml_tensor * pad_k, const ggml_tensor * pad_v) {
    static const bool disabled = getenv("GGML_CUDA_FLASHRT_DISABLE") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * src1 = mm_q->src[1];
    if (src1 == nullptr || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (!vis_qkv_pad_leg_ok(mm_q, add_q, resh_q, pad_q, src1) ||
        !vis_qkv_pad_leg_ok(mm_k, add_k, resh_k, pad_k, src1) ||
        !vis_qkv_pad_leg_ok(mm_v, add_v, resh_v, pad_v, src1)) {
        return false;
    }
    // identical geometry across the three legs
    if (mm_k->ne[0] != mm_q->ne[0] || mm_v->ne[0] != mm_q->ne[0] ||
        pad_k->ne[0] != pad_q->ne[0] || pad_v->ne[0] != pad_q->ne[0] ||
        resh_k->ne[0] != resh_q->ne[0] || resh_v->ne[0] != resh_q->ne[0]) {
        return false;
    }
    const int64_t M = ggml_nrows(src1);
    if (M <= 0 || M > INT32_MAX) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_vis_qkv_pad(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_q, const ggml_tensor * add_q, ggml_tensor * pad_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, ggml_tensor * pad_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, ggml_tensor * pad_v,
        ggml_tensor * k_cast, ggml_tensor * v_cast) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * src1 = mm_q->src[1];
    const int K = (int) mm_q->src[0]->ne[0];
    const int M = (int) ggml_nrows(src1);

    const int group_in  = (int) (pad_q->src[0]->ne[0]);
    const int group_out = (int) pad_q->ne[0];
    const int n_groups  = (int) pad_q->ne[1];
    const int N_pad     = group_out * n_groups;

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());
    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        const int qrc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (qrc != 0) {
            GGML_ABORT("flashrt: vis qkv activation quantize failed (M=%d K=%d rc=%d)", M, K, qrc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    // K/V may go straight to their f16 cast tensors (single rounding from
    // the fp32 accumulator, bitwise equal to f32-out + cast); Q stays f32.
    const ggml_tensor * legs[3][4] = {
        { mm_q, add_q, pad_q, nullptr },
        { mm_k, add_k, pad_k, k_cast },
        { mm_v, add_v, pad_v, v_cast },
    };
    for (auto & leg : legs) {
        const grouppad_weight * w = get_repacked_grouppad(
            leg[0]->src[0], leg[1]->src[1], group_in, group_out, n_groups, stream);
        int rc;
        if (leg[3] != nullptr) {
            rc = flash_rt::fp4::gemm_bias_f16out(
                q_packed, q_sf, w->packed, w->sf, w->bias,
                leg[3]->data, M, N_pad, K, stream);
        } else {
            rc = flash_rt::fp4::gemm_bias_f32out(
                q_packed, q_sf, w->packed, w->sf, w->bias,
                (float *) leg[2]->data, M, N_pad, K, stream);
        }
        if (rc != 0) {
            GGML_ABORT("flashrt: vis qkv padded gemm failed (M=%d N=%d K=%d rc=%d)", M, N_pad, K, rc);
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM + (bias) + residual window: {mul_mat, add bias?, add residual} -> one
// GEMM with the bias and the residual in the epilogue (C may alias D; the
// epilogue reads C before writing D). Covers the encoder/vision o- and
// down-projections whose residual adds were separate bandwidth passes.

namespace {

std::unordered_map<int, void *> g_zero_bias_cache; // keyed by N

const void * get_zero_bias(int N, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);
    auto it = g_zero_bias_cache.find(N);
    if (it != g_zero_bias_cache.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        return nullptr; // caller falls back; allocation is capture-illegal
    }
    void * p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t) N * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(p, 0, (size_t) N * sizeof(float), stream));
    g_zero_bias_cache.emplace(N, p);
    return p;
}

const ggml_tensor * mm_res_residual(const ggml_tensor * chain, const ggml_tensor * res_add) {
    const ggml_tensor * r = res_add->src[0] == chain ? res_add->src[1]
                          : res_add->src[1] == chain ? res_add->src[0] : nullptr;
    if (r == nullptr || r == chain || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) ||
        !ggml_are_same_shape(r, res_add)) {
        return nullptr;
    }
    return r;
}

} // namespace

bool ggml_cuda_flashrt_should_fuse_mm_res(const ggml_tensor * mm, const ggml_tensor * bias_add,
                                          const ggml_tensor * res_add) {
    if (!ggml_cuda_flashrt_should_use(mm->src[0], mm->src[1], mm)) {
        return false;
    }
    const int64_t N = mm->ne[0];
    const int64_t K = mm->src[0]->ne[0];
    if (N % 16 != 0 || K % 64 != 0) {
        return false;
    }
    const ggml_tensor * chain = mm;
    if (bias_add != nullptr) {
        const ggml_tensor * b = bias_add->src[1];
        if (bias_add->src[0] != mm || b == nullptr || b->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(b) || b->ne[0] != N || ggml_nrows(b) != 1) {
            return false;
        }
        chain = bias_add;
    }
    if (mm_res_residual(chain, res_add) == nullptr ||
        !ggml_is_contiguous(res_add) || !ggml_are_same_shape(res_add, mm)) {
        return false;
    }
    return true;
}

bool ggml_cuda_flashrt_mm_res(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
                              const ggml_tensor * bias_add, ggml_tensor * res_add) {
    const ggml_tensor * src1 = mm->src[1];
    const int M = (int) ggml_nrows(src1);
    const int N = (int) mm->ne[0];
    const int K = (int) mm->src[0]->ne[0];
    cudaStream_t stream = ctx.stream();

    const void * bias = bias_add != nullptr ? bias_add->src[1]->data : get_zero_bias(N, stream);
    if (bias == nullptr) {
        return false; // zero-bias alloc during capture: run unfused
    }
    const ggml_tensor * residual = mm_res_residual(bias_add != nullptr ? bias_add : mm, res_add);

    const repacked_weight * w = get_repacked(mm->src[0], stream);

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());
    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        const int qrc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (qrc != 0) {
            GGML_ABORT("flashrt: mm+res activation quantize failed (M=%d K=%d rc=%d)", M, K, qrc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    const int rc = flash_rt::fp4::siglip_ffn_down_bias_res_f32(
        q_packed, q_sf, w->packed, w->sf, bias,
        residual->data, (float *) res_add->data, M, N, K, stream, 1.0f);
    if (rc != 0) {
        GGML_ABORT("flashrt: mm+res fused gemm failed (M=%d N=%d K=%d rc=%d)", M, N, K, rc);
    }
    return true;
}

// ── Prefill fused QKV: {q mm→reshape→rope→scale, k mm→reshape→rope→pad,
//    v mm→reshape→pad, permute→cpy ×2} ─────────────────────────────────────
// Same fused GEMM + qkv_post as the decode window, but K/V land in per-eval
// padded f16 tensors (token rows of head_dim, KV length padded to the FA KQ
// stride) instead of the persistent KV suffix; the pad rows are zeroed to
// match the graph's PAD semantics (the FA mask multiplies them out, but
// garbage f16 there would poison the softmax with inf/nan).
bool ggml_cuda_flashrt_should_fuse_qkv_prefill(
        const ggml_tensor * q_mm, const ggml_tensor * q_rope, const ggml_tensor * q_scale,
        const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_pad,
        const ggml_tensor * v_mm, const ggml_tensor * v_pad,
        const ggml_tensor * k_cpy, const ggml_tensor * v_cpy) {
    static const bool disabled = getenv("GGML_FLASHRT_NO_QKV_PREFILL") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * src1 = q_mm->src[1];
    if (k_mm->src[1] != src1 || v_mm->src[1] != src1) {
        return false;
    }
    const ggml_tensor * wq = q_mm->src[0];
    const ggml_tensor * wk = k_mm->src[0];
    const ggml_tensor * wv = v_mm->src[0];
    if (!ggml_cuda_flashrt_should_use(wq, src1, q_mm) ||
        !ggml_cuda_flashrt_should_use(wk, src1, k_mm) ||
        !ggml_cuda_flashrt_should_use(wv, src1, v_mm) ||
        wk->ne[0] != wv->ne[0] || wk->ne[0] != wq->ne[0]) {
        return false;
    }
    const int64_t head_dim = wk->ne[1];
    if (wv->ne[1] != head_dim || wq->ne[1] % head_dim != 0) {
        return false;
    }
    // Q: mm -> reshape [hd, n_head, M] -> rope -> scale (contiguous f32 out)
    if (q_rope->src[0]->op != GGML_OP_RESHAPE || q_rope->src[0]->src[0] != q_mm ||
        q_rope->src[0]->ne[0] != head_dim ||
        q_scale->src[0] != q_rope || !ggml_is_contiguous(q_scale) ||
        q_scale->type != GGML_TYPE_F32) {
        return false;
    }
    // K: mm -> reshape [hd, 1, M] -> rope -> pad along the token dim
    if (k_rope->src[0]->op != GGML_OP_RESHAPE || k_rope->src[0]->src[0] != k_mm ||
        k_rope->src[0]->ne[0] != head_dim || k_rope->src[0]->ne[1] != 1) {
        return false;
    }
    const int64_t M   = k_rope->src[0]->ne[2];
    const int64_t kvp = k_pad->ne[2];
    if (k_pad->src[0] != k_rope || k_pad->ne[0] != head_dim || k_pad->ne[1] != 1 ||
        kvp < M) {
        return false;
    }
    // V: mm -> reshape -> pad, same geometry
    if (v_pad->src[0] == nullptr || v_pad->src[0]->op != GGML_OP_RESHAPE ||
        v_pad->src[0]->src[0] != v_mm || v_pad->src[0]->ne[0] != head_dim ||
        v_pad->src[0]->ne[1] != 1 || v_pad->src[0]->ne[2] != M ||
        v_pad->ne[0] != head_dim || v_pad->ne[1] != 1 || v_pad->ne[2] != kvp) {
        return false;
    }
    // both copies materialize [hd, kvp] f16 token rows
    for (const ggml_tensor * cpy : { k_cpy, v_cpy }) {
        if (cpy->type != GGML_TYPE_F16 || !ggml_is_contiguous(cpy) ||
            cpy->ne[0] != head_dim || cpy->ne[1] != kvp || cpy->ne[2] != 1 ||
            cpy->nb[1] != head_dim * sizeof(uint16_t)) {
            return false;
        }
    }
    // ropes share positions, freq factors and parameters; NEOX math only
    if (k_rope->src[1] != q_rope->src[1] || k_rope->src[2] != q_rope->src[2] ||
        memcmp(k_rope->op_params, q_rope->op_params, sizeof(k_rope->op_params)) != 0) {
        return false;
    }
    const int mode = ((const int32_t *) k_rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_qkv_prefill(ggml_backend_cuda_context & ctx,
        const ggml_tensor * q_mm, const ggml_tensor * q_rope, ggml_tensor * q_scale,
        const ggml_tensor * k_mm, const ggml_tensor * k_rope,
        const ggml_tensor * v_mm,
        ggml_tensor * k_cpy, ggml_tensor * v_cpy) {
    const ggml_tensor * src1 = q_mm->src[1];
    const ggml_tensor * wq = q_mm->src[0];
    const ggml_tensor * wk = k_mm->src[0];
    const ggml_tensor * wv = v_mm->src[0];

    const int K  = (int) wk->ne[0];
    const int Nk = (int) wk->ne[1];
    const int Nv = (int) wv->ne[1];
    const int Nq = (int) wq->ne[1];
    const int M  = (int) ggml_nrows(src1);
    const int head_dim = Nk;
    const int kvp = (int) k_cpy->ne[1];

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w = get_repacked_qkv(wk, wv, wq, stream);

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());
    int rc = 0;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        rc = ggml_cuda_flashrt::quantize_act_f32((const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    const int N_tot = Nk + Nv + Nq;
    ggml_cuda_pool_alloc<float> qkv_cat(ctx.pool(), (int64_t) M * N_tot);

    if (rc == 0) {
        rc = ggml_cuda_flashrt::gemm_f32out(q_packed, q_sf, w->packed, w->sf,
                                            qkv_cat.get(), M, N_tot, K, 1.0f, false, stream);
    }
    if (rc == 0 && kvp > M) {
        // zero the pad rows once per eval; qkv_post then fills rows [0, M)
        const size_t row_bytes = (size_t) head_dim * sizeof(uint16_t);
        cudaMemsetAsync((char *) k_cpy->data + (size_t) M * row_bytes, 0, (size_t) (kvp - M) * row_bytes, stream);
        cudaMemsetAsync((char *) v_cpy->data + (size_t) M * row_bytes, 0, (size_t) (kvp - M) * row_bytes, stream);
    }
    if (rc == 0) {
        const int32_t * op = (const int32_t *) k_rope->op_params;
        const int   n_dims     = op[1];
        const int   n_ctx_orig = op[4];
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        memcpy(&freq_base,   op +  5, sizeof(float));
        memcpy(&freq_scale,  op +  6, sizeof(float));
        memcpy(&ext_factor,  op +  7, sizeof(float));
        memcpy(&attn_factor, op +  8, sizeof(float));
        memcpy(&beta_fast,   op +  9, sizeof(float));
        memcpy(&beta_slow,   op + 10, sizeof(float));
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        const float theta_scale = powf(freq_base, -2.0f / n_dims);
        const float scale_f = ggml_get_op_params_f32(q_scale, 0);

        const ggml_tensor * ff = k_rope->src[2];
        // the rope'd K and plain V rows also land in their graph tensors'
        // f32 buffers, feeding the graph-tail persistent-KV store copies
        rc = ggml_cuda_flashrt::qkv_post_full(
            // no f16 Q handoff on the prefill path (flash attention reads f32 Q)
            qkv_cat.get(), (float *) q_scale->data, nullptr,
            k_cpy->data, v_cpy->data,
            (float *) k_rope->data, (float *) v_mm->data,
            (const int32_t *) k_rope->src[1]->data,
            ff != nullptr ? (const float *) ff->data : nullptr,
            M, Nk, Nv, Nq, head_dim, n_dims,
            freq_scale, ext_factor, attn_factor,
            corr_dims[0], corr_dims[1], theta_scale, scale_f, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: fused prefill qkv failed (M=%d N=%d K=%d rc=%d)", M, N_tot, K, rc);
    }
}

// ── Decomposed tiny-M decode attention ──────────────────────────────────────
// FLASH_ATTN_EXT with q_tokens <= 16, single f16 KV head of token rows, an
// f16 mask and no ALiBi/softcap runs as QK-GEMM + masked softmax + PV-GEMM
// (see fr_decode_attn.cu). Faster than the stream-k fattn + fixup pair at
// these shapes.
bool ggml_cuda_flashrt_should_fuse_dec_attn(const ggml_tensor * fa) {
    static const bool disabled = getenv("GGML_FLASHRT_NO_DEC_ATTN") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * q    = fa->src[0];
    const ggml_tensor * k    = fa->src[1];
    const ggml_tensor * v    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr ||
        fa->src[4] != nullptr) {  // no attention sinks
        return false;
    }
    const int64_t hd     = q->ne[0];
    const int64_t n_tok  = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t n_kv   = k->ne[1];
    if (q->type != GGML_TYPE_F32 || n_tok > 16 || q->ne[3] != 1 ||
        hd % 2 != 0 || n_head < 1) {
        return false;
    }
    for (const ggml_tensor * kv : { k, v }) {
        if (kv->type != GGML_TYPE_F16 || kv->ne[0] != hd || kv->ne[2] != 1 ||
            kv->ne[3] != 1 || kv->nb[0] != sizeof(uint16_t) ||
            (int64_t) kv->nb[1] != hd * (int64_t) sizeof(uint16_t)) {
            return false;
        }
    }
    if (v->ne[1] != n_kv) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] < n_kv || mask->ne[1] < n_tok) {
        return false;
    }
    if (fa->type != GGML_TYPE_F32 || fa->ne[0] != hd || fa->ne[1] != n_head ||
        fa->ne[2] != n_tok || fa->nb[0] != sizeof(float) ||
        (int64_t) fa->nb[1] != hd * (int64_t) sizeof(float) ||
        // the PV GEMM stores its result as one dense column-block
        (int64_t) fa->nb[2] != hd * n_head * (int64_t) sizeof(float)) {
        return false;
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_dec_attn(ggml_backend_cuda_context & ctx, ggml_tensor * fa) {
    const ggml_tensor * q    = fa->src[0];
    const ggml_tensor * k    = fa->src[1];
    const ggml_tensor * v    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    const int hd     = (int) q->ne[0];
    const int n_tok  = (int) q->ne[1];
    const int n_head = (int) q->ne[2];
    const int n_kv   = (int) k->ne[1];
    float scale;
    memcpy(&scale, (const float *) fa->op_params + 0, sizeof(float));
    cudaStream_t stream = ctx.stream();

    const int64_t R = (int64_t) n_head * n_tok;
    ggml_cuda_pool_alloc<uint8_t> scores(ctx.pool(), R * n_kv * sizeof(uint16_t));

    // consume the f16 Q rows the fused QKV window handed off (one-shot);
    // they are valid only when the Q view is the dense t-major layout the
    // handoff was written in
    const bool q16_hit = g_q16.key != nullptr && g_q16.key == q->data &&
        g_q16.eval_id == g_eval_id &&
        q->nb[0] == sizeof(float) &&
        (int64_t) q->nb[2] == (int64_t) hd * sizeof(float) &&
        (int64_t) q->nb[1] == (int64_t) hd * n_head * sizeof(float);
    ggml_cuda_pool_alloc<uint8_t> q16;
    void * q16p;
    if (q16_hit) {
        q16p = g_q16.buf;
        g_q16.key = nullptr;
    } else {
        q16.alloc(ctx.pool(), R * hd * sizeof(uint16_t));
        q16p = q16.get();
    }

    const int rc = ggml_cuda_flashrt::decode_attn_decomposed(
        (void *) ctx.cublas_handle(),
        (const float *) q->data,
        (int64_t) (q->nb[0] / sizeof(float)),
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        k->data, v->data,
        mask->data, (int64_t) (mask->nb[1] / sizeof(uint16_t)),
        (float *) fa->data,
        (int64_t) (fa->nb[2] / sizeof(float)),
        (int64_t) (fa->nb[1] / sizeof(float)),
        q16p, q16_hit ? 1 : 0, scores.get(),
        hd, n_tok, n_head, n_kv, scale, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: decomposed decode attention failed (tok=%d kv=%d rc=%d)", n_tok, n_kv, rc);
    }
}

// ── Batched persistent-KV tail copies ───────────────────────────────────────
// The prefill graph ends with one f32->f16 row-copy per layer and KV tensor
// into the persistent encoder-KV buffers. Each is a tiny kernel; a run of
// them batches into a single launch with identical rounding.
bool ggml_cuda_flashrt_kv_tail_cpy_ok(const ggml_tensor * cpy, int64_t * hd, int64_t * n_rows) {
    static const bool disabled = getenv("GGML_FLASHRT_NO_KV_TAIL") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * src = cpy->src[0];
    if (cpy->op != GGML_OP_CPY || src == nullptr || cpy->src[1] == nullptr) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 || cpy->type != GGML_TYPE_F16 ||
        cpy->src[1]->type != GGML_TYPE_F16) {
        return false;
    }
    const int64_t d = src->ne[0];
    const int64_t r = src->ne[2];
    if (src->ne[1] != 1 || src->ne[3] != 1 || d % 2 != 0 ||
        cpy->ne[0] != d || cpy->ne[1] != 1 || cpy->ne[2] != r || cpy->ne[3] != 1) {
        return false;
    }
    // contiguous rows on both sides (row stride == hd elements)
    if (src->nb[0] != sizeof(float) || (int64_t) src->nb[2] != d * (int64_t) sizeof(float) ||
        cpy->nb[0] != sizeof(uint16_t) || (int64_t) cpy->nb[2] != d * (int64_t) sizeof(uint16_t)) {
        return false;
    }
    if (*hd == 0) {
        *hd     = d;
        *n_rows = r;
    } else if (*hd != d || *n_rows != r) {
        return false;
    }
    return true;
}

bool ggml_cuda_flashrt_kv_tail_cpy(ggml_backend_cuda_context & ctx, ggml_tensor ** cpys, int n) {
    if (n < 1 || n > FR_CPY_ROWS_MAX) {
        return false;
    }
    const float * srcs[FR_CPY_ROWS_MAX];
    void        * dsts[FR_CPY_ROWS_MAX];
    for (int i = 0; i < n; ++i) {
        srcs[i] = (const float *) cpys[i]->src[0]->data;
        dsts[i] = cpys[i]->data;
    }
    const int64_t hd     = cpys[0]->ne[0];
    const int64_t n_rows = cpys[0]->ne[2];
    const int rc = ggml_cuda_flashrt::cpy_rows_f32_f16(
        srcs, dsts, n, (int) hd, (int) n_rows, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: batched kv tail copy failed (n=%d hd=%lld rows=%lld rc=%d)",
                   n, (long long) hd, (long long) n_rows, rc);
    }
    return true;
}

// ── SigLIP vision attention via the AOT FlashAttention-4 module ─────────────
// FLASH_ATTN_EXT with head_dim 80, no mask, f32 Q and f16 K/V whose padded
// buffers all share the dense (B, S, H, D) linear layout. Runs as a dense
// f32->f16 Q convert + the FA4 forward + a dense f16->f32 output convert.
bool ggml_cuda_flashrt_should_fuse_vit_fa4(const ggml_tensor * fa, ggml_backend_cuda_context & ctx) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(fa); GGML_UNUSED(ctx);
    return false;
#else
    static const bool disabled = getenv("GGML_FLASHRT_NO_VIT_FA4") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * q = fa->src[0];
    const ggml_tensor * k = fa->src[1];
    const ggml_tensor * v = fa->src[2];
    if (q == nullptr || k == nullptr || v == nullptr ||
        fa->src[3] != nullptr || fa->src[4] != nullptr) {  // no mask, no sinks
        return false;
    }
    const int64_t hd = q->ne[0];
    const int64_t S  = q->ne[1];
    const int64_t H  = q->ne[2];
    const int64_t B  = q->ne[3];
    if (hd != 80 || S < 32 || H < 1 || B < 1 || q->type != GGML_TYPE_F32) {
        return false;
    }
    // Q: permuted view over the dense (B, S, H, D) f32 buffer
    if (q->nb[0] != sizeof(float) ||
        (int64_t) q->nb[2] != hd * (int64_t) sizeof(float) ||
        (int64_t) q->nb[1] != hd * H * (int64_t) sizeof(float) ||
        (int64_t) q->nb[3] != hd * H * S * (int64_t) sizeof(float)) {
        return false;
    }
    for (const ggml_tensor * kv : { k, v }) {
        if (kv->type != GGML_TYPE_F16 || kv->ne[0] != hd || kv->ne[1] != S ||
            kv->ne[2] != H || kv->ne[3] != B ||
            kv->nb[0] != sizeof(uint16_t) ||
            (int64_t) kv->nb[2] != hd * (int64_t) sizeof(uint16_t) ||
            (int64_t) kv->nb[1] != hd * H * (int64_t) sizeof(uint16_t) ||
            (int64_t) kv->nb[3] != hd * H * S * (int64_t) sizeof(uint16_t)) {
            return false;
        }
    }
    // dst: contiguous [hd, H, S, B] f32 — the same linear layout
    if (fa->type != GGML_TYPE_F32 || fa->ne[0] != hd || fa->ne[1] != H ||
        fa->ne[2] != S || fa->ne[3] != B || fa->nb[0] != sizeof(float) ||
        (int64_t) fa->nb[1] != hd * (int64_t) sizeof(float) ||
        (int64_t) fa->nb[2] != hd * H * (int64_t) sizeof(float) ||
        (int64_t) fa->nb[3] != hd * H * S * (int64_t) sizeof(float)) {
        return false;
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    // module load must happen outside CUDA graph capture; fall back until then
    return ggml_cuda_flashrt::fa4_vit_ensure_loaded(ctx.stream()) == 0;
#endif // GGML_CUDA_FLASHRT_FA4
}

void ggml_cuda_flashrt_vit_fa4(ggml_backend_cuda_context & ctx, ggml_tensor * fa) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(ctx); GGML_UNUSED(fa);
    GGML_ABORT("flashrt: FA4 vision attention not built");
#else
    const ggml_tensor * q = fa->src[0];
    const int B = (int) q->ne[3];
    const int S = (int) q->ne[1];
    const int H = (int) q->ne[2];
    const int D = (int) q->ne[0];
    float scale;
    memcpy(&scale, (const float *) fa->op_params + 0, sizeof(float));

    const int64_t n = (int64_t) B * S * H * D;
    ggml_cuda_pool_alloc<uint8_t> q16(ctx.pool(), n * sizeof(uint16_t));
    ggml_cuda_pool_alloc<uint8_t> o16(ctx.pool(), n * sizeof(uint16_t));

    const int rc = ggml_cuda_flashrt::fa4_vit_attention(
        (const float *) q->data, fa->src[1]->data, fa->src[2]->data,
        (float *) fa->data, D, q16.get(), o16.get(),
        B, S, H, D, scale, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: FA4 vision attention failed (B=%d S=%d H=%d rc=%d)", B, S, H, rc);
    }
#endif // GGML_CUDA_FLASHRT_FA4
}

// Variant absorbing the {VIEW (head de-pad), CONT} pair after the FA node:
// the FA4 output converts directly into the CONT's packed destination.
bool ggml_cuda_flashrt_should_fuse_vit_fa4_depad(const ggml_tensor * fa, const ggml_tensor * view,
                                                 const ggml_tensor * cont, ggml_backend_cuda_context & ctx) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(fa); GGML_UNUSED(view); GGML_UNUSED(cont); GGML_UNUSED(ctx);
    return false;
#else
    if (!ggml_cuda_flashrt_should_fuse_vit_fa4(fa, ctx)) {
        return false;
    }
    const int64_t D = fa->ne[0];
    const int64_t H = fa->ne[1];
    const int64_t S = fa->ne[2];
    const int64_t B = fa->ne[3];
    // view: leading d2 <= D slice of the FA output, no offset
    if (view->src[0] != fa || view->type != GGML_TYPE_F32 ||
        view->data != fa->data ||
        view->ne[0] > D || view->ne[1] != H || view->ne[2] != S || view->ne[3] != B) {
        return false;
    }
    const int64_t D2 = view->ne[0];
    // cont: packed [(H*D2), S, B] contiguous f32
    if (cont->src[0] != view || cont->type != GGML_TYPE_F32 ||
        cont->ne[0] != H * D2 || cont->ne[1] != S || cont->ne[2] != B || cont->ne[3] != 1 ||
        !ggml_is_contiguous(cont)) {
        return false;
    }
    return true;
#endif // GGML_CUDA_FLASHRT_FA4
}

void ggml_cuda_flashrt_vit_fa4_depad(ggml_backend_cuda_context & ctx, ggml_tensor * fa,
                                     const ggml_tensor * view, ggml_tensor * cont) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(ctx); GGML_UNUSED(fa); GGML_UNUSED(view); GGML_UNUSED(cont);
    GGML_ABORT("flashrt: FA4 vision attention not built");
#else
    const ggml_tensor * q = fa->src[0];
    const int B = (int) q->ne[3];
    const int S = (int) q->ne[1];
    const int H = (int) q->ne[2];
    const int D = (int) q->ne[0];
    const int D2 = (int) view->ne[0];
    float scale;
    memcpy(&scale, (const float *) fa->op_params + 0, sizeof(float));

    const int64_t n = (int64_t) B * S * H * D;
    ggml_cuda_pool_alloc<uint8_t> q16(ctx.pool(), n * sizeof(uint16_t));
    ggml_cuda_pool_alloc<uint8_t> o16(ctx.pool(), n * sizeof(uint16_t));

    const int rc = ggml_cuda_flashrt::fa4_vit_attention(
        (const float *) q->data, fa->src[1]->data, fa->src[2]->data,
        (float *) cont->data, D2, q16.get(), o16.get(),
        B, S, H, D, scale, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: FA4 vision attention (depad) failed (B=%d S=%d H=%d rc=%d)", B, S, H, rc);
    }
#endif // GGML_CUDA_FLASHRT_FA4
}

// ── pi0.5 prefill self-attention via the AOT FA4 module ────────────────────
// FLASH_ATTN_EXT with head_dim 256, one f16 KV head of contiguous token
// rows and an f16 mask. The pi0.5 prefill is a prefix-LM: full attention
// with a row-uniform pad-only mask, and the real KV length equals the
// query count (self-attention), so slicing the padded KV to S reproduces
// the mask exactly. The shape gate (hd 256, S >= 64, padded KV) is
// specific to that graph; GGML_FLASHRT_NO_PREFILL_FA4 disables the window.
bool ggml_cuda_flashrt_should_fuse_prefill_fa4(const ggml_tensor * fa, ggml_backend_cuda_context & ctx) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(fa); GGML_UNUSED(ctx);
    return false;
#else
    static const bool disabled = getenv("GGML_FLASHRT_NO_PREFILL_FA4") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * q    = fa->src[0];
    const ggml_tensor * k    = fa->src[1];
    const ggml_tensor * v    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr ||
        fa->src[4] != nullptr) {
        return false;
    }
    const int64_t hd = q->ne[0];
    const int64_t S  = q->ne[1];
    const int64_t H  = q->ne[2];
    if (hd != 256 || S < 64 || H < 2 || q->ne[3] != 1 || q->type != GGML_TYPE_F32) {
        return false;
    }
    // Q: permuted view over the dense (S, H, D) f32 buffer
    if (q->nb[0] != sizeof(float) ||
        (int64_t) q->nb[2] != hd * (int64_t) sizeof(float) ||
        (int64_t) q->nb[1] != hd * H * (int64_t) sizeof(float)) {
        return false;
    }
    // K/V: one head of contiguous f16 token rows, padded to a multiple of
    // 256 covering exactly S (the pi0.5 prefill padding scheme)
    const int64_t SK = k->ne[1];
    if (SK < S || SK % 256 != 0 || SK - S >= 256) {
        return false;
    }
    for (const ggml_tensor * kv : { k, v }) {
        if (kv->type != GGML_TYPE_F16 || kv->ne[0] != hd || kv->ne[1] != SK ||
            kv->ne[2] != 1 || kv->ne[3] != 1 ||
            kv->nb[0] != sizeof(uint16_t) ||
            (int64_t) kv->nb[1] != hd * (int64_t) sizeof(uint16_t)) {
            return false;
        }
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] < SK || mask->ne[1] < S) {
        return false;
    }
    // dst: contiguous [hd, H, S] f32 — the same dense linear layout
    if (fa->type != GGML_TYPE_F32 || fa->ne[0] != hd || fa->ne[1] != H ||
        fa->ne[2] != S || fa->ne[3] != 1 || fa->nb[0] != sizeof(float) ||
        (int64_t) fa->nb[1] != hd * (int64_t) sizeof(float) ||
        (int64_t) fa->nb[2] != hd * H * (int64_t) sizeof(float)) {
        return false;
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    return ggml_cuda_flashrt::fa4_vit_ensure_loaded(ctx.stream()) == 0;
#endif // GGML_CUDA_FLASHRT_FA4
}

void ggml_cuda_flashrt_prefill_fa4(ggml_backend_cuda_context & ctx, ggml_tensor * fa) {
#ifndef GGML_CUDA_FLASHRT_FA4
    GGML_UNUSED(ctx); GGML_UNUSED(fa);
    GGML_ABORT("flashrt: FA4 prefill attention not built");
#else
    const ggml_tensor * q = fa->src[0];
    const int S = (int) q->ne[1];
    const int H = (int) q->ne[2];
    const int D = (int) q->ne[0];
    float scale;
    memcpy(&scale, (const float *) fa->op_params + 0, sizeof(float));

    const int64_t n = (int64_t) S * H * D;
    ggml_cuda_pool_alloc<uint8_t> q16(ctx.pool(), n * sizeof(uint16_t));
    ggml_cuda_pool_alloc<uint8_t> o16(ctx.pool(), n * sizeof(uint16_t));

    const int rc = ggml_cuda_flashrt::fa4_prefill_attention(
        (const float *) q->data, fa->src[1]->data, fa->src[2]->data,
        (float *) fa->data, q16.get(), o16.get(),
        S, H, D, scale, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: FA4 prefill attention failed (S=%d H=%d rc=%d)", S, H, rc);
    }
#endif // GGML_CUDA_FLASHRT_FA4
}

// ── Gemma-style norm fold: {RMS_NORM, MUL(w), ADD(mul, norm)} ────────────────
// out = rms_norm(x)*w + rms_norm(x) == rms_norm(x)*(1 + w): the adaLN
// modulate kernel with scale = w and shift = 0. ggml's own fused rms_norm
// cannot express this form (the add operand is the norm output itself, which
// no longer exists once the chain is fused), so it never fires on it.
bool ggml_cuda_flashrt_should_fuse_rms_gemma(const ggml_tensor * rms, const ggml_tensor * mul,
                                             const ggml_tensor * add) {
    static const bool disabled = getenv("GGML_FLASHRT_NO_RMS_GEMMA") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    if (rms->type != GGML_TYPE_F32 || x == nullptr || x->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(rms) || !ggml_is_contiguous(x) || rms->ne[3] != 1) {
        return false;
    }
    if (mul->src[0] != rms) {
        return false;
    }
    const ggml_tensor * w = mul->src[1];
    if (w == nullptr || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        w->ne[0] != rms->ne[0] || ggml_nrows(w) != 1) {
        return false;
    }
    if (!((add->src[0] == mul && add->src[1] == rms) ||
          (add->src[0] == rms && add->src[1] == mul))) {
        return false;
    }
    if (add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) ||
        !ggml_are_same_shape(add, rms)) {
        return false;
    }
    return true;
}

bool ggml_cuda_flashrt_rms_gemma(ggml_backend_cuda_context & ctx, const ggml_tensor * rms,
                                 const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * x = rms->src[0];
    const int M = (int) ggml_nrows(x);
    const int C = (int) x->ne[0];
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    cudaStream_t stream = ctx.stream();

    const void * zeros = get_zero_bias(C, stream);
    if (zeros == nullptr) {
        return false; // zero-vector alloc during capture: run unfused
    }
    const int rc = ggml_cuda_flashrt::ada_rms_mod((const float *) x->data, (const float *) mul->src[1]->data,
                               (const float *) zeros, (float *) add->data, M, C, eps,
                               /*with_rms=*/true, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: rms_gemma fused kernel failed (M=%d C=%d rc=%d)", M, C, rc);
    }
    return true;
}

void ggml_cuda_flashrt_begin_eval() {
    g_eval_id++;
#ifdef GGML_CUDA_FLASHRT_FA4
    // begin_eval runs before any CUDA graph capture starts, so the AOT
    // module load never has to race a capturing first evaluation
    ggml_cuda_flashrt::fa4_vit_ensure_loaded(nullptr);
#endif // GGML_CUDA_FLASHRT_FA4
}
