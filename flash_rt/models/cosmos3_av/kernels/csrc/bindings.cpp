// cosmos3_av model-local kernel extension — pybind module `cosmos3_av_kernels`.
//
// Bundles the 4 Cosmos3-AV-specific kernels that are NOT in the production
// flash_rt_kernels.so (they are model-specific, so per docs/adding_new_model.md
// §4.3 they live in a model-local object library, not the shared .so):
//   - fp4_silu_aux        cutlass NVFP4 GEMM + silu(aux)*acc -> fp4+SF (FFN up-leg, fp4-direct)
//   - sage_gqa_d128       GQA-native int8-QK / f16-PV SageAttention (late layers)
//   - sage_gqa_f8_d128    GQA-native int8-QK / f8-PV  SageAttention (probe; unused by default)
//   - qk_norm_rope        fused RMS qk-norm + qwen36 partial rope (one launch)
// Byte-for-byte validated Cosmos3 model-local kernels.
#include <pybind11/pybind11.h>
#include <cstdint>

int  fp4_silu_aux(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int, int, int, double, int64_t);
void sage_gqa_d128(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                   int, int, int, int, int, double, int64_t);
void sage_gqa_f8_d128(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                      int, int, int, int, int, double, int64_t);
void qk_norm_rope(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int, int, int, int, double, int64_t);

PYBIND11_MODULE(cosmos3_av_kernels, m) {
  m.doc() = "Cosmos3-Nano AV FP4 model-local kernels (additive; not in flash_rt_kernels.so)";
  m.def("fp4_silu_aux", &fp4_silu_aux,
        "NVFP4 GEMM + silu(aux_gate)*acc -> fp4 packed + swizzled UE4M3 SF");
  m.def("sage_gqa_d128", &sage_gqa_d128,
        "GQA-native d128 SageAttention (int8 QK, f16 PV)");
  m.def("sage_gqa_f8_d128", &sage_gqa_f8_d128,
        "GQA-native d128 SageAttention (int8 QK, f8 PV)");
  m.def("qk_norm_rope", &qk_norm_rope,
        "Fused RMS qk-norm + qwen36 partial rope (q,k in place)");
}
