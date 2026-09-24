# ================================================================
# FlashRT AMD — standalone HIP build (never wired into root CMake).
#
#   cmake -B build-amd -S csrc/amd -DGPU_ARCH=gfx942|gfx950
#   cmake --build build-amd -j 8
#
# Output: flash_rt/amd/flash_rt_amd_kernels*.so
# The root CMakeLists (CUDA) is untouched; the two builds never mix.
# ================================================================
cmake_minimum_required(VERSION 3.21)
project(flash_rt_amd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ROCm location ──
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# ── GPU_ARCH: gfx string, auto-detected from rocminfo when not given ──
if(NOT DEFINED GPU_ARCH)
  execute_process(
    COMMAND bash -c "${ROCM_PATH}/bin/rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]\\+'"
    OUTPUT_VARIABLE GPU_ARCH_DETECTED
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(GPU_ARCH_DETECTED STREQUAL "")
    message(FATAL_ERROR
      "flash_rt_amd: GPU_ARCH is required when no AMD GPU is visible")
  else()
    set(GPU_ARCH "${GPU_ARCH_DETECTED}")
  endif()
endif()
string(REPLACE ":" ";" GPU_ARCH_PARTS "${GPU_ARCH}")
list(GET GPU_ARCH_PARTS 0 GPU_ARCH_BASE)
if(NOT GPU_ARCH_BASE STREQUAL "gfx942" AND
   NOT GPU_ARCH_BASE STREQUAL "gfx950")
  message(FATAL_ERROR
    "flash_rt_amd supports only gfx942 (CDNA3) and gfx950 (CDNA4); "
    "received GPU_ARCH=${GPU_ARCH}")
endif()
message(STATUS "flash_rt_amd: GPU_ARCH=${GPU_ARCH_BASE}")

enable_language(HIP)
set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH_BASE}")
set(CMAKE_HIP_STANDARD 17)

find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

# ── pybind11 (same resolution as root CMake: ask the interpreter) ──
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

# ── Kernel module ──
set(FLASHRT_AMD_SOURCES
  bindings.cpp
  kernels/norm.hip
  kernels/activation.hip
  kernels/elementwise.hip
  kernels/rope.hip
  kernels/patch_embed.hip
  kernels/fusion.hip
  kernels/quantize_fp8.hip
  kernels/elementwise_fp16.hip
  kernels/norm_fp16.hip
  kernels/adaln_layer_norm.hip
  kernels/stream_probe.hip
  kernels/ew_tune.hip
  attention/decoder_flash.hip
  attention/attn_probe.hip
  gemm/hipblaslt_runner.hip
  gemm/decoder_ffn_fused.hip
  gemm/smallm_mfma.hip
  gemm/smallm_mfma_bf16.hip
)
if(GPU_ARCH_BASE STREQUAL "gfx950")
  list(APPEND FLASHRT_AMD_SOURCES
    attention/encoder_flash.hip
    gemm/smallm_fp8.hip)
endif()
set_source_files_properties(${FLASHRT_AMD_SOURCES} PROPERTIES LANGUAGE HIP)

pybind11_add_module(flash_rt_amd_kernels ${FLASHRT_AMD_SOURCES})

target_compile_definitions(flash_rt_amd_kernels PRIVATE
  FLASHRT_AMD_GPU_ARCH="${GPU_ARCH_BASE}")
if(GPU_ARCH_BASE STREQUAL "gfx942")
  target_compile_definitions(flash_rt_amd_kernels PRIVATE
    FLASHRT_AMD_CDNA3=1)
else()
  target_compile_definitions(flash_rt_amd_kernels PRIVATE
    FLASHRT_AMD_CDNA4=1)
endif()
target_compile_options(flash_rt_amd_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffp-contract=fast>)
target_link_libraries(flash_rt_amd_kernels PRIVATE hip::host roc::hipblaslt)

set_target_properties(flash_rt_amd_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../flash_rt/amd")
