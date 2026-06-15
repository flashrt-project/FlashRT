// cosmos3_video model-local kernel extension — pybind module `cosmos3_video_kernels`.
//
// The text2video fp8 denoise path needs one model-specific kernel that is NOT in
// the production flash_rt_kernels.so (per docs/adding_new_model.md §4.3 it lives
// in this model-local object library, not the shared .so):
//   - qk_norm_rope    fused RMS qk-norm + qwen36 partial rope (one launch)
#include <pybind11/pybind11.h>
#include <cstdint>

void qk_norm_rope(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int, int, int, int, double, int64_t);

PYBIND11_MODULE(cosmos3_video_kernels, m) {
  m.doc() = "Cosmos3-video model-local kernels (additive; not in flash_rt_kernels.so)";
  m.def("qk_norm_rope", &qk_norm_rope,
        "Fused RMS qk-norm + qwen36 partial rope (q,k in place)");
}
