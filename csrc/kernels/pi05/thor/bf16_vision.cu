// Specialize FlashRT's vendored FlashAttention 2 to the frozen SigLIP shape.
// Kernel source copyright Tri Dao; retain the vendor license in deployment.
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "flash_fwd_launch_template.h"
namespace py = pybind11;
void vision_scaled(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o, uintptr_t lse,
                   uintptr_t stream, float scale, float scale_log2) {
    using namespace FLASH_NAMESPACE;
    Flash_fwd_params p = {};
    p.q_ptr = (void*)q;
    p.k_ptr = (void*)k;
    p.v_ptr = (void*)v;
    p.o_ptr = (void*)o;
    p.softmax_lse_ptr = (void*)lse;
    p.b = 1;
    p.h = 16;
    p.h_k = 16;
    p.h_h_k_ratio = 1;
    p.seqlen_q = 256;
    p.seqlen_k = 256;
    p.seqlen_q_rounded = 256;
    p.seqlen_k_rounded = 256;
    p.d = 72;
    p.d_rounded = 96;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride = p.o_batch_stride = 256 * 16 * 72;
    p.q_row_stride = p.k_row_stride = p.v_row_stride = p.o_row_stride = 16 * 72;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 72;
    p.is_bf16 = true;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.num_splits = 1;
    p.window_size_left = -1;
    p.window_size_right = -1;
    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale_log2;
    p.p_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    using Traits = Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::bfloat16_t>;
    auto fn = &flash_fwd_kernel<Traits, false, false, false, false, false, false, false, false, false>;
    auto rc =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize);
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    fn<<<dim3(2, 1, 16), Traits::kNThreads, Traits::kSmemSize, (cudaStream_t)stream>>>(p);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
void vision(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o, uintptr_t lse, uintptr_t stream) {
    float scale = (float)(1.0 / std::sqrt(72.0));
    vision_scaled(q, k, v, o, lse, stream, scale, (float)(double(scale) * M_LOG2E));
}
PYBIND11_MODULE(pi05_bf16_fa2_ext, m) {
    m.def("vision", &vision);
    m.def("vision_scaled", &vision_scaled);
    m.attr("scope") =
        "FlashRT vendored FA2, BF16 1x256x16x72, fixed no-dropout/no-mask tile96/128/64/4";
}
