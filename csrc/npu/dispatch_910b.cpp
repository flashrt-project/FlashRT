// CANN's kernel headers define a translation-unit tiling symbol. Compile
// this kernel family together so the shared library has one definition.
#include "kernels/row_quant_910b.cpp"
#include "kernels/decoder_rope_910b.cpp"
#include "kernels/encoder_rope_910b.cpp"
#include "kernels/euler_update_910b.cpp"
#include "kernels/image_patches_910b.cpp"
#include "kernels/gated_ada_910b.cpp"

#include "kernels/gelu_quant_910b.cpp"
#include "kernels/rms_quant_910b.cpp"
#include "kernels/decoder_int8_910b.cpp"
#include "kernels/decoder_vt_910b.cpp"

extern "C" const char* flashrt_npu_soc_version() {
    return FLASHRT_NPU_SOC_VERSION;
}
