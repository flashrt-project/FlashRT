#include "fa4_attention.h"

#include "fa4_aot/fa4_hd256_fwd.h"
#include "fa4_aot/fa4_hd256_q1_fwd.h"
#include "fa4_aot/fa4_hd72_fwd.h"

#include <mutex>

namespace flashrt_trt {
namespace {

std::mutex g_mu;
bool g_loaded = false;
fa4_hd256_fwd_Kernel_Module_t g_hd256;
fa4_hd256_q1_fwd_Kernel_Module_t g_hd256_q1;
fa4_hd72_fwd_Kernel_Module_t g_hd72;

// The FA4 compile key splits on query stages: two when Sq * (query heads per
// KV head) exceeds the 128-row tile, else one (interface_fwd_sm100.py).
constexpr int64_t kTileM = 128;

template <typename T>
void fill(T* t, void* data, int32_t b, int32_t s, int32_t h, int32_t hd, int64_t row_stride) {
    t->data = data;
    t->dynamic_shapes[0] = b;
    t->dynamic_shapes[1] = s;
    t->dynamic_shapes[2] = h;
    t->dynamic_shapes[3] = hd;
    t->dynamic_strides[0] = static_cast<int64_t>(s) * row_stride;
    t->dynamic_strides[1] = row_stride;
    t->dynamic_strides[2] = hd;
}

// Q, K, V, O tensors of one exported module M.
#define FA4_CALL(M, module, b, sq, sk, nh, nkv, hd, q, qrs, k, krs, v, vrs, o, scale, stream) \
    [&]() -> int {                                                                         \
        M##_Tensor_mQ_t tq;                                                                \
        M##_Tensor_mK_t tk;                                                                \
        M##_Tensor_mV_t tv;                                                                \
        M##_Tensor_mO_t to;                                                                \
        fill(&tq, q, b, sq, nh, hd, qrs);                                                  \
        fill(&tk, k, b, sk, nkv, hd, krs);                                                 \
        fill(&tv, v, b, sk, nkv, hd, vrs);                                                 \
        fill(&to, o, b, sq, nh, hd, static_cast<int64_t>(nh) * hd);                        \
        return cute_dsl_##M##_wrapper(module, &tq, &tk, &tv, &to, scale, stream);          \
    }()

}  // namespace

int fa4_load() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_loaded) {
        return 0;
    }
    fa4_hd256_fwd_Kernel_Module_Load(&g_hd256);
    fa4_hd256_q1_fwd_Kernel_Module_Load(&g_hd256_q1);
    fa4_hd72_fwd_Kernel_Module_Load(&g_hd72);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        return -static_cast<int>(e);
    }
    g_loaded = true;
    return 0;
}

int fa4_hd256_gqa(void* q, void* k, void* v, void* o, int32_t sq, int32_t sk, int32_t nh,
                  float scale, cudaStream_t stream) {
    if (!g_loaded) {
        return -1;
    }
    constexpr int32_t hd = 256;
    const int64_t row = static_cast<int64_t>(nh) * hd;
    if (static_cast<int64_t>(sq) * nh > kTileM) {
        return FA4_CALL(fa4_hd256_fwd, &g_hd256, 1, sq, sk, nh, 1, hd, q, row, k, hd, v, hd, o,
                        scale, stream);
    }
    return FA4_CALL(fa4_hd256_q1_fwd, &g_hd256_q1, 1, sq, sk, nh, 1, hd, q, row, k, hd, v, hd, o,
                    scale, stream);
}

int fa4_hd72_mha(void* q, int64_t q_row_stride, void* k, int64_t k_row_stride, void* v,
                 int64_t v_row_stride, void* o, int32_t b, int32_t s, int32_t nh, float scale,
                 cudaStream_t stream) {
    if (!g_loaded) {
        return -1;
    }
    if (s <= kTileM) {
        return -2;  // only the two-stage module is exported for this head size
    }
    return FA4_CALL(fa4_hd72_fwd, &g_hd72, b, s, s, nh, nh, 72, q, q_row_stride, k, k_row_stride,
                    v, v_row_stride, o, scale, stream);
}

}  // namespace flashrt_trt
