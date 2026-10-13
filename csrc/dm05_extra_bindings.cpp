// SPDX-License-Identifier: Apache-2.0
//
// flash_rt_dm05 — standalone pybind module for DM05-specific kernels.
//
// Kept separate from flash_rt_kernels so the module can be built, installed and
// rebuilt on its own; FLASHRT_ENABLE_DM05 gates the whole thing and nothing in
// the default target references it.
//
// Every entry takes raw device pointers and a stream so it can be captured in a
// CUDA graph and replayed with stable addresses.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "kernels/dm05/dm05_ae_decode_attn.cuh"
#include "kernels/dm05/dm05_prefix_attn.cuh"
#include "kernels/dm05/dm05_vit_attn_hd72.cuh"

namespace py = pybind11;

static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_dm05, m) {
    m.doc() =
        "DM05 fused kernels (raw-pointer, CUDA-graph capturable). Built with "
        "-DFLASHRT_ENABLE_DM05=ON on GPU_ARCH=120.";

    m.def("dm05_vit_attn_hd72",
        [](uintptr_t qkv, long long ld, int D, uintptr_t out, long long o_ld,
           int S, int NH, int nv, float scale, int mt, uintptr_t stream,
           uintptr_t q8, uintptr_t q8_scale) {
            flash_rt::dm05::vit_attn_hd72(
                reinterpret_cast<const __nv_bfloat16*>(qkv), ld, D,
                reinterpret_cast<__nv_bfloat16*>(out), o_ld,
                S, NH, nv, scale, mt,
                reinterpret_cast<__nv_fp8_e4m3*>(q8),
                reinterpret_cast<const float*>(q8_scale),
                to_stream(stream));
        },
        py::arg("qkv"), py::arg("ld"), py::arg("D"), py::arg("out"), py::arg("o_ld"),
        py::arg("S"), py::arg("NH"), py::arg("nv"), py::arg("scale"),
        py::arg("mt") = 2, py::arg("stream") = 0,
        py::arg("q8") = 0, py::arg("q8_scale") = 0,
        "SigLIP self-attention over a fused qkv buffer at head_dim 72. "
        "qkv is [nv*S, 3*D] with Q at +0, K at +D, V at +2*D; D must equal NH*72 "
        "and S must be a multiple of 128. Pass q8 (and q8_scale) to write the "
        "static-FP8 copy of the output that the o_projection reads instead of "
        "the bf16 one: `out` is then left untouched, and the FP8 values are "
        "byte-identical to quantize_fp8_static over the same output.");

    m.def("dm05_vit_attn_hd72_smem_bytes", &flash_rt::dm05::vit_attn_hd72_smem_bytes,
        "Dynamic shared memory the kernel requests per block, in bytes.");

    m.def("dm05_ae_decode_attn",
        [](uintptr_t q, long long q_ld, uintptr_t k, uintptr_t v, long long kv_ld,
           uintptr_t out, long long o_ld,
           uintptr_t ws_o, uintptr_t ws_ml, uintptr_t cnt,
           int T, int G, int NKV, int sk, int ns, float scale, uintptr_t stream) {
            flash_rt::dm05::ae_decode_attn(
                reinterpret_cast<const __nv_bfloat16*>(q), q_ld,
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const __nv_bfloat16*>(v), kv_ld,
                reinterpret_cast<__nv_bfloat16*>(out), o_ld,
                reinterpret_cast<float*>(ws_o), reinterpret_cast<float*>(ws_ml),
                reinterpret_cast<int*>(cnt),
                T, G, NKV, sk, ns, scale, to_stream(stream));
        },
        py::arg("q"), py::arg("q_ld"), py::arg("k"), py::arg("v"), py::arg("kv_ld"),
        py::arg("out"), py::arg("o_ld"),
        py::arg("ws_o"), py::arg("ws_ml"), py::arg("cnt"),
        py::arg("T"), py::arg("G"), py::arg("NKV"), py::arg("sk"), py::arg("ns"),
        py::arg("scale"), py::arg("stream") = 0,
        "Grouped-query decode attention at head_dim 256 over the prefix cache. "
        "T query rows and G query heads per key/value head are packed into one "
        "tile, so G*T must be an instantiated row count (8, 16, 20, 24, 32). "
        "ns > 1 splits the keys and needs the ws_o/ws_ml/cnt workspaces; ns = 1 "
        "writes the output directly and ignores them.");

    m.def("dm05_prefix_attn",
        [](uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, uintptr_t lim,
           int S, int NH, int NKV, float scale, uintptr_t stream) {
            flash_rt::dm05::prefix_attn(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const __nv_bfloat16*>(v),
                reinterpret_cast<__nv_bfloat16*>(out),
                reinterpret_cast<const int*>(lim),
                S, NH, NKV, scale, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("lim"),
        py::arg("S"), py::arg("NH"), py::arg("NKV"), py::arg("scale"),
        py::arg("stream") = 0,
        "Gemma3 prefix attention with a per-row key limit: row i attends to keys "
        "[0, lim[i]), which expresses a mask that is causal over text and "
        "bidirectional within each image block. q/out are [S, NH*256] and k/v are "
        "[S, NKV*256], with contiguous rows.");
    m.def("dm05_prefix_attn_smem_bytes", &flash_rt::dm05::prefix_attn_smem_bytes,
        "Dynamic shared memory the kernel requests per block, in bytes.");

    m.def("dm05_ae_decode_attn_supports_r", &flash_rt::dm05::ae_decode_attn_supports_r,
        py::arg("r"), "Whether this build instantiates the G*T row count r.");
    m.def("dm05_ae_decode_attn_smem_bytes", &flash_rt::dm05::ae_decode_attn_smem_bytes,
        "Dynamic shared memory the kernel requests per block, in bytes.");
    m.def("dm05_ae_decode_attn_ws_o_floats", &flash_rt::dm05::ae_decode_attn_ws_o_floats,
        py::arg("NKV"), py::arg("ns_max"), "Floats to allocate for ws_o.");
    m.def("dm05_ae_decode_attn_ws_ml_floats", &flash_rt::dm05::ae_decode_attn_ws_ml_floats,
        py::arg("NKV"), py::arg("ns_max"), "Floats to allocate for ws_ml.");
}
