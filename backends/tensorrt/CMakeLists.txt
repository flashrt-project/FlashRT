cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
# Every CUDA target passes its own -gencode (FLASHRT_TRT_GENCODE).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()
project(flashrt_tensorrt LANGUAGES C CXX CUDA)

# FlashRT TensorRT backend: TensorRT plugins for Pi0.5 on Jetson Thor (SM110).
#
#   cmake -S backends/tensorrt -B build/tensorrt
#   cmake --build build/tensorrt -j 2        # build/tensorrt/libflashrt_trt_pi05.so
#
# The plugins compile FlashRT's kernels straight from csrc/ with the flags the
# main build uses for each kernel group, so the engine computes exactly what
# the FlashRT runtime computes.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(FLASHRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CUTLASS_DIR "${FLASHRT_ROOT}/third_party/cutlass" CACHE PATH "NVIDIA/cutlass v4.4.2")
set(FLASHRT_TRT_GENCODE "-gencode=arch=compute_110a,code=sm_110a" CACHE STRING "nvcc target for the kernels")
option(FLASHRT_TRT_BUILD_TESTS "Build the native parity programs" OFF)
set(NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH "nlohmann/json single_include (parity programs)")

if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS v4.4.2 not found at ${CUTLASS_DIR}: git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass (or set CUTLASS_DIR)")
endif()

find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer REQUIRED)
find_path(NVINFER_INCLUDE NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include REQUIRED)

set(C ${FLASHRT_ROOT}/csrc)
set(STAGES ${C}/stages/pi05_thor)
set(CUTLASS_INCLUDES ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)
separate_arguments(GENCODE UNIX_COMMAND "${FLASHRT_TRT_GENCODE}")

# Kernel groups, flags as in the FlashRT main build.
# fp4_kernels_obj
add_library(frt_fp4_kernels OBJECT
  ${C}/fused_fp4/norm_silu_fp4_sfa.cu
  ${C}/fused_fp4/pdl.cu
  ${C}/fused_fp4/pi05_rowops_v2.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_variants_sm100.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_variants.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_variants_persist_a.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_variants_persist_b.cu
  ${C}/gemm/fp4/cutlass_fp4_gemm_variants_swap.cu
  ${C}/quantize/quantize_fp4_sfa.cu
  ${C}/quantize/reshape_scales_sfa.cu)
target_include_directories(frt_fp4_kernels PRIVATE ${C} ${CUTLASS_INCLUDES})
target_compile_definitions(frt_fp4_kernels PRIVATE CUTLASS_ARCH_MMA_SM100_SUPPORTED=1 CUTLASS_ENABLE_GDC_FOR_SM100=1)
target_compile_options(frt_fp4_kernels PRIVATE --expt-relaxed-constexpr -O3 --use_fast_math ${GENCODE})

# cutlass_sm100_obj
add_library(frt_cutlass_sm100 OBJECT ${C}/gemm/cutlass_sm100.cu)
target_include_directories(frt_cutlass_sm100 PRIVATE ${C} ${C}/gemm ${CUTLASS_INCLUDES})
target_compile_definitions(frt_cutlass_sm100 PRIVATE CUTLASS_ENABLE_GDC_FOR_SM100=1)
target_compile_options(frt_cutlass_sm100 PRIVATE --expt-relaxed-constexpr -O3 --use_fast_math ${GENCODE})

# flash_rt_kernels (the subset the Pi0.5 stages use)
add_library(frt_kernels OBJECT
  ${C}/gemm/gemm_runner.cu
  ${C}/kernels/activation.cu
  ${C}/kernels/attention_cublas.cu
  ${C}/kernels/decoder_fused.cu
  ${C}/kernels/norm.cu
  ${C}/kernels/patch_embed.cu
  ${C}/kernels/quantize.cu
  ${C}/kernels/rope_vec.cu
  ${C}/kernels/softmax.cu)
target_include_directories(frt_kernels PRIVATE ${C} ${C}/gemm ${C}/gemm/fp4 ${C}/kernels ${C}/quantize ${CUTLASS_INCLUDES})
target_compile_definitions(frt_kernels PRIVATE
  ENABLE_BF16_CONV3D_V0=1 ENABLE_CUTLASS_SM100_NVFP4_W4A16=1 ENABLE_DECODE_GEMV_M1=1 ENABLE_FP8_CONV3D_V17=1
  ENABLE_FP8_CONV3D_V18=1 ENABLE_LINGBOT=1 ENABLE_QWEN36_FLASHINFER_XQA=1 ENABLE_SM100_CUTLASS=1
  FLASHRT_HAVE_AUDIO_CODEBOOK=1 FLASHRT_HAVE_THOR_VLA_KERNELS=1)
target_compile_options(frt_kernels PRIVATE -Xcompiler=-fvisibility=hidden --expt-relaxed-constexpr -O3
  --ftz=true --prec-div=false --prec-sqrt=false ${GENCODE})

# Model-free operator layer (kernels/frt_ops.h) and the Pi0.5 Thor native
# stages (csrc/stages/pi05_thor); both call the same FlashRT kernels.
add_library(pi05_stages OBJECT
  ${STAGES}/pi05_encoder_layer.cu ${STAGES}/pi05_decoder_step.cu ${STAGES}/pi05_siglip.cu
  ${STAGES}/fa4_attention.cpp ${STAGES}/fa4_shims.c
  kernels/frt_ops.cu)
target_include_directories(pi05_stages PUBLIC ${STAGES} ${C}/attention ${C} ${CMAKE_CURRENT_SOURCE_DIR}/kernels PRIVATE ${CUTLASS_INCLUDES})
target_compile_definitions(pi05_stages PRIVATE CUTLASS_ENABLE_GDC_FOR_SM100=1 CUTLASS_ARCH_MMA_SM100_SUPPORTED=1)
target_compile_options(pi05_stages PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr ${GENCODE}>)
target_link_libraries(pi05_stages PUBLIC CUDA::cudart)

set_target_properties(frt_fp4_kernels frt_cutlass_sm100 frt_kernels pi05_stages PROPERTIES CUDA_ARCHITECTURES OFF)

# FlashAttention-4 ahead-of-time modules (csrc/attention/fa4_aot, from tools/export_fa4_aot.py)
set(FA4_AOT_OBJECTS
  ${C}/attention/fa4_aot/fa4_hd256_fwd.o
  ${C}/attention/fa4_aot/fa4_hd256_q1_fwd.o
  ${C}/attention/fa4_aot/fa4_hd72_fwd.o)
set(KERNEL_OBJECTS
  $<TARGET_OBJECTS:pi05_stages> $<TARGET_OBJECTS:frt_fp4_kernels> $<TARGET_OBJECTS:frt_cutlass_sm100>
  $<TARGET_OBJECTS:frt_kernels> ${FA4_AOT_OBJECTS})
set(KERNEL_LIBS CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)

# TensorRT plugin library
add_library(flashrt_trt_pi05 SHARED
  plugins/pi05_encoder_layer_plugin.cpp plugins/pi05_encoder_stage_plugin.cpp
  plugins/pi05_decoder_step_plugin.cpp plugins/pi05_decoder_plugin.cpp
  plugins/pi05_siglip_plugin.cpp plugins/ops/nvfp4_op_plugins.cpp
  plugins/pi05_plugins.cpp ${KERNEL_OBJECTS})
target_include_directories(flashrt_trt_pi05 PRIVATE ${STAGES} ${C} ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${NVINFER_INCLUDE})
target_link_libraries(flashrt_trt_pi05 PRIVATE ${NVINFER_LIB} ${KERNEL_LIBS})
target_link_options(flashrt_trt_pi05 PRIVATE -Wl,--no-undefined)

if(FLASHRT_TRT_BUILD_TESTS)
  if(NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "Set NLOHMANN_JSON_INCLUDE_DIR for the parity programs")
  endif()
  foreach(t encoder_layer_parity encoder_stage_parity decoder_step_parity siglip_parity)
    add_executable(${t} tests/${t}.cpp ${KERNEL_OBJECTS})
    target_include_directories(${t} PRIVATE ${STAGES} ${C} ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE ${KERNEL_LIBS})
  endforeach()
  add_library(fa4_aot_runner SHARED tests/fa4_aot_runner.c ${STAGES}/fa4_shims.c ${FA4_AOT_OBJECTS})
  target_include_directories(fa4_aot_runner PRIVATE ${C}/attention)
  target_link_libraries(fa4_aot_runner PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()
