cmake_minimum_required(VERSION 3.24)
project(flash_rt CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)

# ── ccache (opt-in speedup for iterative dev) ──
# If ccache is on PATH, route both C++ and CUDA compiles through it.
# First build is unchanged; subsequent rebuilds with the same flags
# get cache hits (typically 2–10× faster on FA2's templating-heavy
# .cu files). Zero effect on CI's first build and no behaviour change.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  message(STATUS "ccache: ENABLED (${CCACHE_PROGRAM}) — repeat builds will hit cache")
else()
  message(STATUS "ccache: not found (optional; install for faster rebuilds)")
endif()

# ── GPU architecture auto-detection ──
if(NOT DEFINED GPU_ARCH)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE _detected_cc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _detect_result
  )
  if(_detect_result EQUAL 0 AND _detected_cc)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _first_cc "${_detected_cc}")
    string(REPLACE "." "" GPU_ARCH "${_first_cc}")
    message(STATUS "Auto-detected GPU arch: sm_${GPU_ARCH}")
  else()
    set(GPU_ARCH "89")
    message(STATUS "GPU detection failed, defaulting to sm_${GPU_ARCH}")
  endif()
endif()

# ── Feature flags based on SM version ──
# SM100/SM110 (Thor/Hopper): CUTLASS FP8 GEMM templates + FMHA
if(GPU_ARCH STREQUAL "100" OR GPU_ARCH STREQUAL "110")
  set(ENABLE_SM100_CUTLASS ON)
  message(STATUS "SM100 CUTLASS FP8: ENABLED (sm_${GPU_ARCH})")
else()
  set(ENABLE_SM100_CUTLASS OFF)
endif()

# Developer-only SM100 kernels. OFF by default so production flash_rt_kernels
# neither compiles nor exports them; turn on locally for kernel bring-up.
option(FLASHRT_BUILD_SM100_SWEEP
       "Build the transient FP16 CUTLASS tile-sweep bench kernel" OFF)
option(FLASHRT_BUILD_SM100_ENCODER_MLP
       "Build the WIP Path C encoder MLP megakernel scaffold" OFF)

# SM120+ (Blackwell): NVFP4/W4A8 block-scaled GEMM
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "120a" OR
   GPU_ARCH STREQUAL "121" OR GPU_ARCH STREQUAL "121a")
  set(ENABLE_NVFP4 ON)
  message(STATUS "NVFP4/W4A8 support: ENABLED (sm_${GPU_ARCH}a)")
else()
  set(ENABLE_NVFP4 OFF)
  message(STATUS "NVFP4/W4A8 support: DISABLED (requires Blackwell sm_120a/sm_121a, current: sm_${GPU_ARCH})")
endif()

# Only sm_90/100/110/120 ship an 'a' (architecture-accelerated) variant.
# Ada (sm_89) and Ampere (sm_80/86) do not, so emitting -gencode
# arch=compute_89a,code=sm_89a breaks nvcc on 4090. Pick the right
# gencode flag once here and reuse it for every target below.
if(GPU_ARCH STREQUAL "90"  OR GPU_ARCH STREQUAL "100" OR
   GPU_ARCH STREQUAL "110" OR GPU_ARCH STREQUAL "120" OR
   GPU_ARCH STREQUAL "121")
  set(GPU_GENCODE "-gencode=arch=compute_${GPU_ARCH}a,code=sm_${GPU_ARCH}a")
else()
  set(GPU_GENCODE "-gencode=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}")
endif()
message(STATUS "Using gencode flag: ${GPU_GENCODE}")

# Flash-Attention 2 vendor (csrc/attention/flash_attn_2_src/): ships
# SM80-family fp16 fwd kernels. Ampere-family consumer GPUs including
# SM86/87/89 share the same source instantiations; we emit arch-specific
# codegen per target so Jetson Orin (SM87) is not forced through an
# incompatible cubin. SM120 (5090) uses PTX JIT from
# compute_80.
#
# Thor SM110 was excluded on the grounds that it has its own attention path
# (fvk.attention_qkv_fp16, cuBLAS decomposed) and that FA2 would cost ~10 MB
# of .so for nothing. That trade does not survive a long prefill: the
# decomposed path materialises an (S * heads, S_kv) score buffer, which is
# 3.4 GB per layer at ten thousand tokens, and the model that needs it here
# is bf16 at head_dim 256 -- one instantiation, not the twelve the size
# estimate assumed. FA2_HDIMS/FA2_DTYPES are narrowed for this target below.
#
# It stays opt-in all the same: every other Thor model still uses its own
# attention path and would only be paying the compile time and the binary, so
# the target that wants it asks for it. Default OFF keeps a Thor build's
# sources and symbols exactly what they were.
option(FLASHRT_ENABLE_THOR_FA2
       "Build the vendored FA2 attention kernels on Jetson AGX Thor (sm_110)" OFF)
if(FLASHRT_ENABLE_THOR_FA2 AND NOT GPU_ARCH STREQUAL "110")
  message(FATAL_ERROR
    "FLASHRT_ENABLE_THOR_FA2 is the Thor (sm_110) FA2 gate; current "
    "GPU_ARCH=${GPU_ARCH} decides FA2 by architecture and needs no flag.")
endif()
if(GPU_ARCH STREQUAL "80"  OR GPU_ARCH STREQUAL "86" OR
   GPU_ARCH STREQUAL "87"  OR
   GPU_ARCH STREQUAL "89"  OR
   (GPU_ARCH STREQUAL "110" AND FLASHRT_ENABLE_THOR_FA2) OR
   GPU_ARCH STREQUAL "120" OR
   GPU_ARCH STREQUAL "121")
  set(ENABLE_FA2 ON)
  message(STATUS "FA2 in-SO attention: ENABLED (sm_${GPU_ARCH})")
else()
  set(ENABLE_FA2 OFF)
  message(STATUS "FA2 in-SO attention: DISABLED (sm_${GPU_ARCH})")
endif()

# SM80-family CUTLASS INT8 kernels used by the Jetson Orin SM87 Pi0.5 fast
# path. Keep them out of the shared flash_rt_kernels module on other targets
# so Thor/RTX builds do not inherit unrelated compile/link risk.
if(NOT DEFINED ENABLE_SM80_INT8_CUTLASS)
  if(GPU_ARCH STREQUAL "87")
    set(ENABLE_SM80_INT8_CUTLASS ON)
  else()
    set(ENABLE_SM80_INT8_CUTLASS OFF)
  endif()
endif()
if(ENABLE_SM80_INT8_CUTLASS)
  message(STATUS "SM80-family CUTLASS INT8: ENABLED (sm_${GPU_ARCH})")
else()
  message(STATUS "SM80-family CUTLASS INT8: DISABLED (current: sm_${GPU_ARCH})")
endif()

# ── FA2 build slimming flags (opt-in; defaults keep today's behaviour) ──
# FA2's vendored CUTLASS 3.x templates dominate FlashRT's build cost
# (~8–10 min on a warm container for the default matrix of 3 hdims ×
# 2 dtypes × 2 paths × 3 gencode arches = 36 codegens). These flags let
# developers trade binary coverage for iteration speed without touching
# source. Defaults are ON / full set — existing builds are unchanged.
#
#   FA2_ARCH_NATIVE_ONLY=ON    Skip sm_80 + compute_120 PTX AOT; emit only
#                              SASS for the current GPU. Saves ~66% build
#                              time. Don't use if you plan to ship the
#                              resulting .so to a different arch.
#
#   FA2_HDIMS="96;256"         Restrict head_dim instantiations. The
#                              shipped models (Pi0 / Pi0.5 / GROOT) only
#                              need {96, 256}; 128 is reserved for future
#                              DiT-style models. Default "96;128;256".
#
#   FA2_DTYPES="fp16"          Drop bf16 if you only run fp16 models (Pi0).
#                              pi0.5 / groot need bf16. Default "fp16;bf16".
set(FA2_ARCH_NATIVE_ONLY OFF CACHE BOOL
    "Only build FA2 for the detected arch (no sm_80/PTX fallback). ~66% faster build.")
set(FA2_HDIMS "64;96;128;256" CACHE STRING
    "Semicolon-separated FA2 head_dim instantiations to build. Qwen3-VL 2B vision uses 64; other shipped models use 96 and 256.")
set(FA2_DTYPES "fp16;bf16" CACHE STRING
    "Semicolon-separated FA2 dtype instantiations to build. Pi0 uses fp16; pi0.5/groot use bf16.")

# Thor runs one model family through FA2 -- Qwen3.6 full attention, bf16 at
# head_dim 256 -- so the default there is that one instantiation rather than
# the twelve-file matrix the RTX targets distribute. This is why enabling FA2
# on this arch does not cost what the original exclusion assumed. Both remain
# cache variables: an explicit -DFA2_HDIMS on the command line still wins.
if(GPU_ARCH STREQUAL "110" AND ENABLE_FA2)
  if(NOT DEFINED CACHE{FA2_HDIMS} OR FA2_HDIMS STREQUAL "64;96;128;256")
    set(FA2_HDIMS "256" CACHE STRING "" FORCE)
  endif()
  if(NOT DEFINED CACHE{FA2_DTYPES} OR FA2_DTYPES STREQUAL "fp16;bf16")
    set(FA2_DTYPES "bf16" CACHE STRING "" FORCE)
  endif()
endif()

option(FLASHRT_ENABLE_NATIVE_CPP
       "Build Python-free operation libraries for native C++ consumers" OFF)
option(FLASHRT_BUILD_FA2_PYTHON_ADAPTER
       "Build the vendored FA2 Python adapter when FA2 is available" ON)

# Cosmos3 deployment kernels are model- and architecture-specific. Keep them
# out of the shared modules unless the corresponding Thor path is requested.
option(FLASHRT_ENABLE_COSMOS3_EDGE
       "Build Cosmos3-Edge AV kernels and bindings for Jetson AGX Thor" OFF)
option(FLASHRT_ENABLE_COSMOS3_REASONER
       "Build Cosmos3 Reasoner kernels and bindings for Jetson AGX Thor" OFF)
if((FLASHRT_ENABLE_COSMOS3_EDGE OR FLASHRT_ENABLE_COSMOS3_REASONER) AND
   NOT GPU_ARCH STREQUAL "110")
  message(FATAL_ERROR
    "Cosmos3 kernels require GPU_ARCH=110; current GPU_ARCH=${GPU_ARCH}. "
    "Disable FLASHRT_ENABLE_COSMOS3_EDGE/FLASHRT_ENABLE_COSMOS3_REASONER "
    "for non-Thor builds.")
endif()

# Chameleon-7B model kernels: QK Norm/RoPE, SM87 INT8/INT4 GEMM + FHT/QuaRot,
# FA2 FP16 causal instances, and SM100/110 causal FMHA libraries. Off by
# default so unrelated builds pay no compile/link/symbol cost. Model-neutral
# fp16 norm/quant/activation helpers stay in the common layer.
option(FLASHRT_ENABLE_CHAMELEON
       "Build Chameleon-7B model kernels and bindings" OFF)

# Motus beta integration. Motus-specific kernels are additive and must keep
# their symbols prefixed with ``motus_``. Keeping a build tag lets the public
# package compile without Motus kernels when debugging unrelated model paths.
option(FLASHRT_ENABLE_LINGBOT
       "Build LingBot-VLA Thor (sm_110) model kernels and bindings" ON)
option(FLASHRT_ENABLE_MOTUS
       "Build Motus-specific RTX SM120 kernels and bindings" ON)
# qwen3_5_moe family (Nex-N2-mini / Qwen3.5-3.6-35B-A3B) kernels, split into
# three tiers by the hardware each tier actually requires. All are OFF by
# default so they never enter the flash_rt_kernels build for models that do
# not use them.
#
#   _CORE    Architecture-neutral: layout/split, bf16 matvec, router top-k,
#            activation fusion, GDN recurrence, weighted-sum reducer, and the
#            bf16 GEMM. Uses only bf16 intrinsics plus cp.async and
#            mma.m16n8k16.bf16, i.e. SM80 and newer.
#   _W4A16   Weight-only 4-bit: converts packed operands with the
#            __nv_cvt_fp4x2_to_halfraw2 intrinsic and accumulates in bf16, so
#            it needs no block-scaled MMA. SM89 and newer emit the hardware
#            conversion; older targets take the intrinsic's software path.
#   _W4A4    Block-scaled 4-bit MMA through CUTLASS. Requires the
#            CUTE_ARCH_F8F6F4_MMA_ENABLED path, which only sm_120a/sm_121a
#            provide. On other targets CUTLASS still compiles these
#            translation units but replaces the MMA with
#            CUTE_INVALID_CONTROL_PATH, so the gate must be explicit: a
#            successful build would otherwise produce kernels that fail at
#            run time.
#
# FLASHRT_ENABLE_QWEN35MOE remains the single switch for a full Blackwell
# build and turns on all three tiers, so existing configure lines are
# unchanged.
option(FLASHRT_ENABLE_QWEN35MOE
       "Build all qwen3_5_moe (Nex-N2 / Qwen3.6-35B-A3B) kernel tiers" OFF)
option(FLASHRT_ENABLE_QWEN35MOE_CORE
       "Build architecture-neutral qwen3_5_moe kernels" OFF)
option(FLASHRT_ENABLE_QWEN35MOE_W4A16
       "Build weight-only 4-bit qwen3_5_moe kernels" OFF)
option(FLASHRT_ENABLE_QWEN35MOE_W4A4
       "Build block-scaled 4-bit MMA qwen3_5_moe kernels (sm_120a/sm_121a)" OFF)
if(FLASHRT_ENABLE_QWEN35MOE)
  set(FLASHRT_ENABLE_QWEN35MOE_CORE ON)
  set(FLASHRT_ENABLE_QWEN35MOE_W4A16 ON)
  set(FLASHRT_ENABLE_QWEN35MOE_W4A4 ON)
endif()
# The upper tiers reuse the core layout, reducer, and activation kernels.
if(FLASHRT_ENABLE_QWEN35MOE_W4A16 OR FLASHRT_ENABLE_QWEN35MOE_W4A4)
  set(FLASHRT_ENABLE_QWEN35MOE_CORE ON)
endif()
# Qwen3-VL adds a handful of kernels (rotate_half RoPE, ...) that the shared
# binary does not yet need. Built into a SEPARATE flash_rt_qwen3_vl_kernels
# module so flash_rt_kernels.so stays stable; OFF by default.
option(FLASHRT_BUILD_QWEN3_VL
       "Build Qwen3-VL kernels into a separate flash_rt_qwen3_vl_kernels module" OFF)
option(FLASHRT_ENABLE_QWEN3_VL_DEV_KERNELS
       "Expose Qwen3-VL bench/probe pybind symbols for kernel development" OFF)
# MelBandRoformer audio source separation kernels (FP8 fused kernels for
# BF16-in / FP8-out norm, QKV+RoPE, gated attention quant, residual+norm).
# OFF by default; enable with -DFLASHRT_ENABLE_MELBAND_ROFORMER=ON.
option(FLASHRT_ENABLE_MELBAND_ROFORMER
       "Build MelBandRoformer custom fused kernels (BF16/FP8, SM120)" OFF)
# OmniVoice TTS — builds flash_rt_omnivoice.so with OmniVoice-specific
# cfg-logsoftmax and QK-norm/RoPE kernels. The module is opt-in and remains
# separate from the default flash_rt_kernels target.
# Enable explicitly on a Blackwell (sm_120) build.
option(FLASHRT_ENABLE_OMNIVOICE
       "Build OmniVoice TTS RTX SM120 fused kernels in flash_rt_omnivoice" OFF)
# Shared delayed-codebook helper used by discrete-audio paths. Kept behind an
# explicit option so deployment builds that do not serve audio can remove the
# TU and its binding in lockstep.
option(FLASHRT_ENABLE_AUDIO_CODEBOOK
       "Build shared delayed-codebook helper kernels into flash_rt_kernels" ON)
# MiniMax-Remover — builds flash_rt_minimax_remover.so with model-specific
# FP8/NVFP4 transformer and VAE fused kernels. OFF by default; enable with
# -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON on Blackwell only.
option(FLASHRT_ENABLE_MINIMAX_REMOVER
       "Build MiniMax-Remover FP8/NVFP4 kernels in flash_rt_minimax_remover" OFF)
if(FLASHRT_ENABLE_MINIMAX_REMOVER AND NOT ENABLE_NVFP4)
  message(FATAL_ERROR
    "FLASHRT_ENABLE_MINIMAX_REMOVER requires Blackwell NVFP4 support "
    "(GPU_ARCH=120/120a/121/121a). The MiniMax-Remover module contains "
    "SM120 FP8/NVFP4 kernels and must not be enabled for GPU_ARCH=${GPU_ARCH}.")
endif()
option(FLASHRT_ENABLE_SM120_DEV_KERNELS
       "Build SM120 probe/tilesweep tuning kernels and public bindings" OFF)
# ── Slim build for VLA deployments ──
# OFF by default: the standard build keeps the broad, compatibility-oriented
# kernel set it has today. Turn ON (-DFLASHRT_SLIM_BUILD=ON) to drop
# model/architecture-specific translation units that a given hardware/model
# path does not need, reducing flash_rt_kernels compile time on weaker
# deployment machines. This is COMPILE-TIME surface reduction only; it never
# changes kernel math, dispatch, graph capture, or runtime fallback. Each gated
# group is guarded by its own internal FLASHRT_HAVE_* compile-def so the
# corresponding bindings in csrc/bindings.cpp drop in lockstep with the sources
# (a gated source whose binding stayed compiled would fail to link). See
# scripts/build_inventory.py for the measured surface.
option(FLASHRT_SLIM_BUILD
       "Drop model/arch-specific kernels not needed by the target path" OFF)
if(FLASHRT_SLIM_BUILD)
  message(STATUS "Slim VLA build: ENABLED (model/arch-specific kernels gated out)")
else()
  message(STATUS "Slim VLA build: DISABLED (broad compatibility kernel set)")
endif()
if(FLASHRT_ENABLE_MOTUS)
  message(STATUS "Motus beta kernels: ENABLED")
else()
  message(STATUS "Motus beta kernels: DISABLED")
endif()

# ── MSVC / Windows host-compiler flags ──
# Linux is the primary platform; this block only activates under MSVC.
#   /bigobj  — FA2 + CUTLASS template instantiations blow past the
#              default 65k-section limit. Required for the FA2 object lib.
#   /utf-8   — sources contain non-ASCII comments (e.g. "→"). Without
#              this, MSVC interprets them as the system code page and
#              warns/errors depending on locale.
#   /Zc:__cplusplus — make __cplusplus report the real value so feature
#              detection (in CUTLASS, pybind11) doesn't see C++98.
#   /EHsc    — standard C++ exception model (pybind11 expects it).
# We push these globally for both CXX and the nvcc host pass via
# -Xcompiler. nvcc-only flags stay where they are.
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
  )
endif()

# ── Paths ──
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to the CUTLASS source tree")
set(CUTLASS_INCLUDE ${CUTLASS_DIR}/include)

# Fail fast if CUTLASS isn't vendored — the main FP8/FP4 GEMM kernels
# depend on it and silently degrading produces very confusing link errors.
if(NOT EXISTS ${CUTLASS_INCLUDE}/cutlass/cutlass.h)
  message(FATAL_ERROR
    "CUTLASS headers not found at ${CUTLASS_INCLUDE}.\n"
    "Run the following from the repo root before cmake:\n"
    "    git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass\n"
    "(CUTLASS is intentionally not bundled in the repo to keep clone size small.)")
endif()

# ── Find Python & pybind11 ──
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# ── Find cuBLAS / cuBLASLt ──
find_package(CUDAToolkit REQUIRED)

# ── CUTLASS FP8 kernels for SM100/SM110 (Thor) ──
if(ENABLE_SM100_CUTLASS)
  add_library(cutlass_sm100_obj OBJECT csrc/gemm/cutlass_sm100.cu)
  set_target_properties(cutlass_sm100_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cutlass_sm100_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  # SM110a: 'a' suffix enables TMA (Tensor Memory Accelerator) for CUTLASS
  target_compile_options(cutlass_sm100_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  message(STATUS "SM100 CUTLASS FP8 GEMMs: building for sm_${GPU_ARCH}")

  # FP16 CUTLASS GEMMs (encoder/SigLIP FP16 path).
  add_library(cutlass_sm100_fp16_obj OBJECT csrc/gemm/cutlass_sm100_fp16.cu)
  set_target_properties(cutlass_sm100_fp16_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cutlass_sm100_fp16_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(cutlass_sm100_fp16_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  message(STATUS "SM100 CUTLASS FP16 GEMMs: building for sm_${GPU_ARCH}")

  # Path C: vendored production sm100_gemm_tma_warpspecialized.hpp +
  # single-GEMM instantiation for "match production" gate test.  The
  # header is a literal copy of CUTLASS production; the .cu file
  # instantiates it for the split-G7 shape via CollectiveBuilder.
  add_library(flashrt_megakernel_obj OBJECT csrc/gemm/mega/flashrt_megakernel_single.cu)
  set_target_properties(flashrt_megakernel_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(flashrt_megakernel_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/mega
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(flashrt_megakernel_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )

  # Encoder GeGLU megakernel: fuses the gate and up GEMMs (one launch
  # computes hidden = GELU(X @ W_gate) * (X @ W_up)).
  add_library(flashrt_megakernel_geglu_obj OBJECT csrc/gemm/mega/flashrt_megakernel_geglu.cu)
  set_target_properties(flashrt_megakernel_geglu_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  # Include-path priority (mirrors libfmha_fp16_strided rules):
  #   1. csrc/gemm/mega/  — vendored CUTLASS hooks under cutlass/pipeline,
  #      cutlass/epilogue/collective, cutlass/gemm/collective. MUST be
  #      before ${CUTLASS_INCLUDE} so the patched headers shadow the
  #      stock CUTLASS versions only for this translation unit.
  #   2. ${CUTLASS_INCLUDE} — upstream CUTLASS for everything else.
  target_include_directories(flashrt_megakernel_geglu_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/mega
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(flashrt_megakernel_geglu_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )

  # Encoder GeGLU + down-proj megakernel: bundles the GeGLU megakernel and
  # the down GEMM (beta=1 residual) into one C entry.  Uses the same
  # include precedence as the GeGLU object (mega dir before upstream CUTLASS).
  add_library(flashrt_megakernel_geglu_g8_obj OBJECT csrc/gemm/mega/flashrt_megakernel_geglu_g8.cu)
  set_target_properties(flashrt_megakernel_geglu_g8_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(flashrt_megakernel_geglu_g8_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/mega
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(flashrt_megakernel_geglu_g8_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )

  # Path C encoder MLP megakernel scaffold (WIP, unused by the runtime).
  if(FLASHRT_BUILD_SM100_ENCODER_MLP)
    add_library(encoder_mlp_sm100_obj OBJECT csrc/gemm/mega/encoder_mlp_sm100.cu)
    set_target_properties(encoder_mlp_sm100_obj PROPERTIES
      CUDA_STANDARD 17
      POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(encoder_mlp_sm100_obj PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
      ${CUTLASS_INCLUDE}
      ${CUTLASS_DIR}/tools/util/include
    )
    target_compile_options(encoder_mlp_sm100_obj PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3 --use_fast_math
        ${GPU_GENCODE}
      >
    )
  endif()

  # FP16 CUTLASS GEMM tile-sweep bench kernel (transient bring-up tool).
  if(FLASHRT_BUILD_SM100_SWEEP)
    add_library(cutlass_sm100_fp16_sweep_obj OBJECT csrc/gemm/cutlass_sm100_fp16_sweep.cu)
    set_target_properties(cutlass_sm100_fp16_sweep_obj PROPERTIES
      CUDA_STANDARD 17
      POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(cutlass_sm100_fp16_sweep_obj PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
      ${CUTLASS_INCLUDE}
      ${CUTLASS_DIR}/tools/util/include
    )
    target_compile_options(cutlass_sm100_fp16_sweep_obj PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3 --use_fast_math
        ${GPU_GENCODE}
      >
    )
  endif()
endif()

# ── libfmha_fp16_strided.so — CUTLASS SM100 FP16 FMHA for SigLIP ──
# Built as a standalone shared library (not a pybind module) because
# csrc/attention/fmha_dispatch.cu loads it at runtime via dlopen and
# resolves the 'fmha_fp16_strided' symbol with dlsym. Keeping the FMHA
# kernel in its own .so lets flash_rt_kernels.so stay small (~3.6 MB)
# and avoids pulling CUTLASS SM100 FMHA templates into every rebuild.
#
# Include-path priority:
#   1. csrc/attention/             — the repo's vendored collective
#      header (FP8 softmax kPRescale fix for HD=256). MUST be first so
#      it shadows the stock CUTLASS version.
#   2. CUTLASS examples/77_blackwell_fmha — stock device/ + kernel/.
#   3. Standard CUTLASS include + util.
#
# Only built on Thor / Hopper (SM100+); RTX (SM89/SM120) uses in-SO FA2
# instead via flash_rt_fa2.so.
if(ENABLE_SM100_CUTLASS)
  add_library(fmha_fp16_strided SHARED csrc/attention/fmha_fp16_strided.cu)
  set_target_properties(fmha_fp16_strided PROPERTIES
    # Drop the artefact directly into flash_rt/ at build time so a plain
    # `cmake --build` / `ninja` is self-sufficient — no follow-up `cp` or
    # `ninja install` step needed before `import flash_rt` works. The
    # install(TARGETS ...) rule below stays for wheel-packaging users.
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    # CMake defaults CMAKE_CUDA_ARCHITECTURES to "75" on this toolchain;
    # without overriding, the device-link step emits an sm_75 cubin that
    # doesn't cover SM100/SM110 templates → half the FMHA symbols get
    # dropped from the final .so. Pinning the target's architecture with
    # the 'a' suffix matches the compile-time -gencode flag below.
    CUDA_ARCHITECTURES "${GPU_ARCH}a"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    PREFIX "lib"
    OUTPUT_NAME "fmha_fp16_strided"
  )
  target_include_directories(fmha_fp16_strided PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
    ${CUTLASS_DIR}/examples/77_blackwell_fmha
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(fmha_fp16_strided PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr --expt-extended-lambda -O3
      ${GPU_GENCODE}
    >
  )
  target_link_libraries(fmha_fp16_strided PRIVATE CUDA::cudart)
  message(STATUS "libfmha_fp16_strided.so: building for sm_${GPU_ARCH} (Thor FMHA for SigLIP)")

  if(FLASHRT_ENABLE_CHAMELEON)
  # ── libfmha_fp16_causal.so — CUTLASS SM100 FP16 causal FMHA for Chameleon ──
  # Same as fmha_fp16_strided but with CausalMask<true>/<false> and
  # CausalIndividualTileScheduler. Used by Chameleon-7B LLM self-attention
  # (is_causal=True). Loaded at runtime via ctypes in
  # hardware/thor/attn_backend_chameleon.py.
  add_library(fmha_fp16_causal SHARED csrc/attention/fmha_fp16_causal.cu)
  set_target_properties(fmha_fp16_causal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${GPU_ARCH}a"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    PREFIX "lib"
    OUTPUT_NAME "fmha_fp16_causal"
  )
  target_include_directories(fmha_fp16_causal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
    ${CUTLASS_DIR}/examples/77_blackwell_fmha
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(fmha_fp16_causal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr --expt-extended-lambda -O3
      ${GPU_GENCODE}
    >
  )
  target_link_libraries(fmha_fp16_causal PRIVATE CUDA::cudart)
  message(STATUS "libfmha_fp16_causal.so: building for sm_${GPU_ARCH} (Thor causal FMHA for Chameleon)")

  # ── libfmha_fp8_causal.so — CUTLASS SM100 FP8 causal FMHA ──
  # FP8 (E4M3) input variant of fmha_fp16_causal for Chameleon-7B. Inputs
  # Q/K/V are FP8, outputs O FP16, accumulators FP32. CUTLASS Sm100 FMHA
  # mainloop has FP8-aware kPRescale logic that triggers when
  # ``Element == cutlass::float_e4m3_t``. Loaded via dlopen at runtime
  # alongside libfmha_fp16_causal.so as a drop-in alternative.
  add_library(fmha_fp8_causal SHARED csrc/attention/fmha_fp8_causal.cu)
  set_target_properties(fmha_fp8_causal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${GPU_ARCH}a"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    PREFIX "lib"
    OUTPUT_NAME "fmha_fp8_causal"
  )
  target_include_directories(fmha_fp8_causal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
    ${CUTLASS_DIR}/examples/77_blackwell_fmha
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(fmha_fp8_causal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr --expt-extended-lambda -O3
      ${GPU_GENCODE}
    >
  )
  target_link_libraries(fmha_fp8_causal PRIVATE CUDA::cudart)
  message(STATUS "libfmha_fp8_causal.so: building for sm_${GPU_ARCH} (Thor FP8 causal FMHA for Chameleon)")
  endif()  # FLASHRT_ENABLE_CHAMELEON
endif()

# ── CUTLASS SM120a block-128 FP8 GEMM (Path B for Qwen3.6) ──
# DeepSeek-V3 layout: per-token activation x 128x128 weight block scale.
# Built from CUTLASS 4.x example 87b, requires sm_120a (SM120 with the
# 'a' arch-accelerated suffix that enables F8F6F4 MMA + TMA SM120).
if(GPU_ARCH STREQUAL "120")
  add_library(cutlass_sm120_block_obj OBJECT
    csrc/gemm/cutlass_sm120_block128_fp8_gemm.cu)
  set_target_properties(cutlass_sm120_block_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cutlass_sm120_block_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(cutlass_sm120_block_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  message(STATUS "SM120a CUTLASS block-128 FP8: ENABLED")
endif()

# ── Hand-tuned SM120 small-M FP8 GEMMs ──
# These bindings are exposed together with ENABLE_CUTLASS_SM120_BLOCK_FP8, so
# the objects must be linked for every SM120 build, independent of Motus.
if(GPU_ARCH STREQUAL "120")
  add_library(sm120_fp8_smallm_obj OBJECT
    csrc/gemm/fp8_smallM_handtuned_sm120.cu
    csrc/gemm/fp8_smallM_handtuned_splitk_sm120.cu
    csrc/gemm/fp8_smallM_handtuned_ldmatrix_sm120.cu)
  set_target_properties(sm120_fp8_smallm_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(sm120_fp8_smallm_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  )
  target_compile_options(sm120_fp8_smallm_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
endif()

# ── CUTLASS SM120a NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt) ──
# Block-scaled FP4 e2m1 weights × FP4 e2m1 act (post quant) × FP8 ue4m3
# group scales × BF16 output. Ported from NVIDIA unit test
# sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu — sm_120a / sm_121a.
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  add_library(cutlass_nvfp4_w4a16_sm120_obj OBJECT
    csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cu)
  set_target_properties(cutlass_nvfp4_w4a16_sm120_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cutlass_nvfp4_w4a16_sm120_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(cutlass_nvfp4_w4a16_sm120_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  message(STATUS "SM120a CUTLASS NVFP4 W4A16 GEMM: ENABLED")
endif()

# ── SM120a NVFP4 FFN epilogue GEMMs ──
# Shared by Qwen3.6 and Motus. Keep these sources in one object target so a
# Motus-enabled SM120 build does not link the same global symbols twice.
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  add_library(sm120_nvfp4_ffn_obj OBJECT
    csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cu
    csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cu
    csrc/gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cu
    csrc/gemm/fp4/cutlass_nvfp4_dual_gemm_silu_fp4out_sm120.cu)
  set_target_properties(sm120_nvfp4_ffn_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(sm120_nvfp4_ffn_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(sm120_nvfp4_ffn_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
endif()

# ── SM120a NVFP4 development probes ──
# These are tuning/proof kernels, not runtime dependencies. Keep them behind an
# explicit developer option so the public pybind module does not grow research
# symbols by default.
if(FLASHRT_ENABLE_SM120_DEV_KERNELS AND
   (GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121"))
  add_library(sm120_nvfp4_dev_obj OBJECT
    csrc/gemm/fp4/cutlass_nvfp4_w4a16_tilesweep_sm120.cu
    csrc/gemm/fp4/normfold/cutlass_nvfp4_normfold_probe_sm120.cu)
  set_target_properties(sm120_nvfp4_dev_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(sm120_nvfp4_dev_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(sm120_nvfp4_dev_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
endif()

# ── CUTLASS SM100 NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt, Thor SM110) ──
# Sibling of the SM120 W4A16 object above. Same wire format, BF16 output.
# Uses arch::Sm100 + OpClassBlockScaledTensorOp + KernelScheduleAuto,
# the same dispatch already in production on Thor via the Pi0.5 / LingBot
# NVFP4 path (csrc/gemm/fp4/cutlass_fp4_gemm.cu). Compiled with
# -gencode arch=compute_110a,code=sm_110a via GPU_GENCODE.
if(GPU_ARCH STREQUAL "110")
  add_library(cutlass_nvfp4_w4a16_sm100_obj OBJECT
    csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cu)
  set_target_properties(cutlass_nvfp4_w4a16_sm100_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cutlass_nvfp4_w4a16_sm100_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(cutlass_nvfp4_w4a16_sm100_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  message(STATUS "SM100 CUTLASS NVFP4 W4A16 GEMM (Thor): ENABLED")
endif()

# ── Grouped NVFP4 MoE GEMM (qwen3_5_moe weight-only tier, Thor SM110) ──
# One launch per layer over every routed expert, with the per-group shapes read
# from device memory. Only the qwen3_5_moe MoE prefill calls it, so it is its
# own object gated on that model's tier rather than a second source in the
# W4A16 object above: a Thor build that does not ask for this model must not
# pay its CUTLASS grouped-kernel compile time or carry its symbols. The
# matching bindings are guarded on FLASHRT_HAVE_QWEN35MOE_GROUPED_SM100.
if(GPU_ARCH STREQUAL "110" AND FLASHRT_ENABLE_QWEN35MOE_W4A16)
  add_library(qwen35moe_nvfp4_grouped_sm100_obj OBJECT
    csrc/gemm/fp4/cutlass_nvfp4_moe_grouped_sm100.cu)
  set_target_properties(qwen35moe_nvfp4_grouped_sm100_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(qwen35moe_nvfp4_grouped_sm100_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(qwen35moe_nvfp4_grouped_sm100_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  set(ENABLE_QWEN35MOE_GROUPED_SM100 ON)
  message(STATUS "qwen3_5_moe grouped NVFP4 MoE GEMM (Thor): ENABLED")
endif()

# ── Dedicated M=1 decode GEMV (FP8 + BF16) ──
# Origin sm_120a, but neither kernel uses an SM120-specific instruction, so both
# compile for Thor (sm110) and Ada (sm89) too. Independent of
# FLASHRT_ENABLE_MOTUS: the bindings are gated by
# ENABLE_DECODE_GEMV_M1 (set alongside this object below), so the object must
# link whenever it is built — otherwise the binding would reference an unlinked
# symbol when MOTUS is OFF. (FP8 still needs the SM120 prefill GEMM in MOTUS, so
# the FP8 path is SM120-only; the BF16 path runs wherever this object is built.)
if(GPU_ARCH STREQUAL "110" OR GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121" OR
   GPU_ARCH STREQUAL "89")
  add_library(decode_gemv_m1_obj OBJECT
    csrc/gemm/fp8_gemv_m1_sm120.cu
    csrc/gemm/bf16_gemv_m1_sm120.cu)
  set_target_properties(decode_gemv_m1_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(decode_gemv_m1_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  )
  target_compile_options(decode_gemv_m1_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
endif()

# ── Qwen3 SM120 fp8 prefill attention ──
# Shared low-level sage2 fp8 code is compiled from a neutral Qwen3 object so
# Qwen3 does not depend on FLASHRT_ENABLE_MOTUS. Runtime remains env-gated.
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  add_library(qwen3_prefill_sm120_obj OBJECT
    csrc/attention/sage2/sage2_attn_f8_raw.cu
    csrc/attention/fmha_fp8_causal_gqa_sm120.cu)
  set_target_properties(qwen3_prefill_sm120_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(qwen3_prefill_sm120_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/sage2
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/sage2/qattn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize
  )
  target_compile_options(qwen3_prefill_sm120_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
endif()

# ── Motus beta SM120 kernels ──
# These are model-specific RTX kernels and must remain additive. They are
# compiled into flash_rt_kernels so Motus does not depend on separate
# runtime kernel libraries.
if(FLASHRT_ENABLE_MOTUS AND GPU_ARCH STREQUAL "120")
  add_library(motus_aux_sm120_obj OBJECT
    csrc/attention/sage2/sage2_attn_raw.cu
    csrc/conv/fp8_conv2d_3x3_sm120_v1.cu
    csrc/conv/fp8_conv3d_sm120_v17.cu
    csrc/conv/fp8_conv3d_sm120_v18.cu
    csrc/conv/motus/motus_bf16_rms_silu_quant_nvfp4_sm120.cu
    csrc/conv/motus/motus_fp4_conv3d_v19sf_sm120.cu
    csrc/conv/motus/motus_fp4_conv3d_v19sfb_sm120.cu
    csrc/conv/motus/motus_fp4_conv3d_v19sfb_k128_sm120.cu)
  set_target_properties(motus_aux_sm120_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(motus_aux_sm120_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/sage2
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/sage2/qattn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/conv/motus
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize
  )
  target_compile_options(motus_aux_sm120_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )

  add_library(motus_action_ffn_v6t_sm120_obj OBJECT
    csrc/kernels/megakernel/action_ffn_megakernel_v6t_sm120.cu)
  set_target_properties(motus_action_ffn_v6t_sm120_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(motus_action_ffn_v6t_sm120_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  )
  target_compile_options(motus_action_ffn_v6t_sm120_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
      -DV6T_VARIANT_NAME=ku128_sd4
      -DV6T_BLOCK_K_UP=128
      -DV6T_BLOCK_K_DN=256
      -DV6T_NUM_WARPS=4
      -DV6T_STAGES_UP=3
      -DV6T_STAGES_DN=4
      -DV6T_BLOCK_N_UP=32
      -DV6T_BLOCK_N_DN=32
      -DV6T_LAUNCH_MIN=8
    >
  )

  add_library(motus_und_ffn_sm120_obj OBJECT
    csrc/kernels/megakernel/und_ffn_megakernel_v5t_sm120.cu
    csrc/kernels/megakernel/und_ffn_megakernel_v5split_stage3_sm120.cu
    csrc/kernels/megakernel/tinyfp8_kernels_sm120.cu)
  set_target_properties(motus_und_ffn_sm120_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(motus_und_ffn_sm120_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  )
  target_compile_options(motus_und_ffn_sm120_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
      -DV5S_BLOCK_K_UP=64
      -DV5S_BLOCK_K_DN=128
      -DV5S_BLOCK_N_UP=64
      -DV5S_BLOCK_N_DN=32
      -DV5S_LAUNCH_MIN=8
    >
  )
  message(STATUS "Motus beta SM120 kernels: building into flash_rt_kernels")
endif()

# ── CUTLASS block-scaled kernels (sm_120a only) ──
if(ENABLE_NVFP4)
  add_library(gemm_kernels_obj OBJECT csrc/gemm/gemm_kernels.cu)
  set_target_properties(gemm_kernels_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(gemm_kernels_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(gemm_kernels_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      ${GPU_GENCODE}
    >
  )
endif()

# ── Flash-Attention 2 vendored kernels (fp16 + bf16 fwd SM80) ──
# By default the object library is linked directly into the existing
# flash_rt_fa2.so Python module. Native C++ builds explicitly opt into a
# Python-free raw library; an enabled Python adapter then becomes a thin
# consumer of that library. The
# main flash_rt_kernels.so stays small (~3.6 MB) and its rebuild no
# longer waits on FA2's heavy CUTLASS 3.x template codegen (~8 min).
#
# The hdim .cu files are template instantiations for the FA2 forward,
# six per elem_type (three hdim × non-split + splitkv × fp16/bf16).
# fa2_wrapper.cu provides the raw-pointer C entries
# (fvk_attention_fa2_fwd_{fp16,bf16}); fa2_bindings.cpp pybind11-wraps
# them into the flash_rt_fa2 module.
if(ENABLE_FA2 AND
   (FLASHRT_BUILD_FA2_PYTHON_ADAPTER OR FLASHRT_ENABLE_NATIVE_CPP))
  # Dynamic source list: {hdim} x {dtype} x {split, no-split} .cu files,
  # driven by FA2_HDIMS / FA2_DTYPES cache vars. Default is the full
  # 12-file matrix; users can restrict either axis to slim builds.
  set(FA2_SRC_ROOT csrc/attention/flash_attn_2_src/flash_attn)
  set(FA2_SRCS "")
  set(FA2_BUILT_HDIMS "")
  set(FA2_BUILT_DTYPES "")
  foreach(_hd IN LISTS FA2_HDIMS)
    if(NOT _hd MATCHES "^(64|96|128|256)$")
      message(FATAL_ERROR "FA2_HDIMS entry '${_hd}' not supported. Valid: 64, 96, 128, 256.")
    endif()
    list(APPEND FA2_BUILT_HDIMS ${_hd})
    foreach(_dt IN LISTS FA2_DTYPES)
      if(NOT _dt MATCHES "^(fp16|bf16)$")
        message(FATAL_ERROR "FA2_DTYPES entry '${_dt}' not supported. Valid: fp16, bf16.")
      endif()
      list(APPEND FA2_SRCS
        ${FA2_SRC_ROOT}/flash_fwd_hdim${_hd}_${_dt}_sm80.cu
      )
      # hdim64 (Qwen3-VL 2B vision full attention) always takes the no-split
      # path, so skip its split-KV specialization -- that variant is unused
      # and one of the most expensive objects for ptxas.
      if(NOT _hd STREQUAL "64")
        list(APPEND FA2_SRCS
          ${FA2_SRC_ROOT}/flash_fwd_split_hdim${_hd}_${_dt}_sm80.cu
        )
      endif()
    endforeach()
  endforeach()
  foreach(_dt IN LISTS FA2_DTYPES)
    list(APPEND FA2_BUILT_DTYPES ${_dt})
  endforeach()
  list(REMOVE_DUPLICATES FA2_BUILT_HDIMS)
  list(REMOVE_DUPLICATES FA2_BUILT_DTYPES)
  # Always include the wrapper — it ifdef-guards its entries based on
  # which hdim/dtype instantiations were built, so it links cleanly
  # against any subset of the matrix.
  list(APPEND FA2_SRCS csrc/attention/fa2_wrapper.cu)

  # Causal FA2 forward. hdim=128 serves Qwen3-8B prefill; hdim=256
  # serves Qwen3.6 full-attention prefill/verify chunks.
  if("128" IN_LIST FA2_HDIMS AND "bf16" IN_LIST FA2_DTYPES)
    list(APPEND FA2_SRCS
      csrc/attention/fa2_causal_inst/flash_fwd_hdim128_bf16_sm80_causal.cu
      csrc/attention/fa2_causal_inst/flash_fwd_split_hdim128_bf16_sm80_causal.cu
    )
  endif()
  if("256" IN_LIST FA2_HDIMS AND "bf16" IN_LIST FA2_DTYPES)
    list(APPEND FA2_SRCS
      csrc/attention/fa2_causal_inst/flash_fwd_hdim256_bf16_sm80_causal.cu
      csrc/attention/fa2_causal_inst/flash_fwd_split_hdim256_bf16_sm80_causal.cu
    )
  endif()
  # fp16 hdim=128 serves Chameleon-7B causal prefill/decode on Orin SM87.
  if(FLASHRT_ENABLE_CHAMELEON AND "128" IN_LIST FA2_HDIMS AND "fp16" IN_LIST FA2_DTYPES)
    list(APPEND FA2_SRCS
      csrc/attention/fa2_causal_inst/flash_fwd_hdim128_fp16_sm80_causal.cu
      csrc/attention/fa2_causal_inst/flash_fwd_split_hdim128_fp16_sm80_causal.cu
    )
  endif()
  if(FLASHRT_ENABLE_NATIVE_CPP)
    # The native C boundary has a stable five-symbol surface. Its causal
    # wrapper contains native-only fail-fast dispatch for a slim matrix.
    list(APPEND FA2_SRCS csrc/attention/fa2_wrapper_causal.cu)
  elseif(("bf16" IN_LIST FA2_DTYPES OR "fp16" IN_LIST FA2_DTYPES) AND
         ("128" IN_LIST FA2_HDIMS OR "256" IN_LIST FA2_HDIMS))
    # Preserve the existing Python-only source matrix exactly by default.
    list(APPEND FA2_SRCS csrc/attention/fa2_wrapper_causal.cu)
  endif()

  add_library(fa2_vendor_obj OBJECT ${FA2_SRCS})
  set_target_properties(fa2_vendor_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  if(FLASHRT_ENABLE_NATIVE_CPP)
    target_compile_definitions(fa2_vendor_obj PRIVATE
      FLASHRT_FA2_NATIVE_BUILD=1)
    set_target_properties(fa2_vendor_obj PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES)
  endif()
  target_include_directories(fa2_vendor_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/flash_attn_2_src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/flash_attn_2_src/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/flash_attn_2_src/cutlass/include
  )
  target_compile_options(fa2_vendor_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr --expt-extended-lambda
      -O3 --use_fast_math
      # CRITICAL: enable native fp16/bf16 operator overloads — without
      # these -U flags, __half + __half compiles as fp32 round-trip,
      # producing sm_120 SASS that numerically drifts from the pip
      # wheel's SASS by ~1 fp16 ULP per layer. Under the Pi0 FP8
      # quantize pipeline that drift crosses bucket boundaries and
      # accumulates to cos 0.981 (pip baseline 0.998). See commit
      # history for the debug trail. Flag set copied verbatim from
      # upstream FA2 setup.py extra_compile_args["nvcc"].
      -U__CUDA_NO_HALF_OPERATORS__
      -U__CUDA_NO_HALF_CONVERSIONS__
      -U__CUDA_NO_HALF2_OPERATORS__
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__
      -DFLASH_NAMESPACE=fa2_vendor
      # CUTLASS 3.x + CUDA 13 emits noisy deprecation warnings on
      # ulong4/double4 etc. Silence on the host side via -Xcompiler's
      # SHELL: form — paired as one token so CMake doesn't dedupe the
      # Wno flag and leave a dangling -Xcompiler that eats the next
      # arg (seen with make VERBOSE=1: the dangling -Xcompiler would
      # grab the first -gencode and forward it to gcc, which then
      # misreads it as a -g<level> debug flag).
      # MSVC equivalent of -Wno-deprecated-declarations is /wd4996.
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:SHELL:-Xcompiler -Wno-deprecated-declarations>
      $<$<CXX_COMPILER_ID:MSVC>:SHELL:-Xcompiler /wd4996>
      # nvcc's per-file internal parallelism (CUDA 11.2+). Pipes the
      # several codegen stages (front-end, PTXAS, fatbin-pack) through
      # up to 4 threads, cutting single-file compile time ~20% on
      # FA2's templating-heavy sources.
      "SHELL:-t 4"
    >
  )

  # ──────────────────────────────────────────────────────────────
  # gencode selection — default ships multi-arch AOT so one build
  # covers every RTX card (sm_80 SASS runs SM80/86/89 natively; sm_120
  # / sm_121 SASS for Blackwell; compute_121 PTX fallback for future
  # consumer archs).
  # FA2_ARCH_NATIVE_ONLY=ON skips the non-native AOT passes, saving
  # ~66% compile time on single-card developer boxes.
  # ──────────────────────────────────────────────────────────────
  if(FA2_ARCH_NATIVE_ONLY)
    # Just emit native SASS for the detected GPU. Use the real arch
    # (compute_80 for SM80/86/89 since they share Ampere ISA; direct
    # Blackwell arch for SM120/121). Matches upstream FA2 setup.py
    # behaviour where applicable.
    if(GPU_ARCH STREQUAL "80" OR GPU_ARCH STREQUAL "86" OR
       GPU_ARCH STREQUAL "87" OR GPU_ARCH STREQUAL "89")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_80,code=sm_${GPU_ARCH}"
        >
      )
      message(STATUS "FA2 vendor arch: sm_${GPU_ARCH} AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    elseif(GPU_ARCH STREQUAL "110")
      # Native SASS via the same gencode the rest of this build uses, not
      # compute_80 PTX. The 5090 learned that lesson: routing it through
      # compute_80 PTX produced SASS that drifted from the native build by
      # about an fp16 ULP a layer.
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${GPU_GENCODE}>
      )
      message(STATUS "FA2 vendor arch: sm_110a AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    elseif(GPU_ARCH STREQUAL "120")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_120,code=sm_120"
        >
      )
      message(STATUS "FA2 vendor arch: sm_120 AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    elseif(GPU_ARCH STREQUAL "121")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_121,code=sm_121"
        >
      )
      message(STATUS "FA2 vendor arch: sm_121 AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    endif()
  else()
    # Default multi-arch AOT (original behaviour; distributable across
    # the whole RTX / Ampere-through-consumer-Blackwell family from one
    # build). See commit history for the reason each gencode matters —
    # routing 5090 through compute_80 PTX produced a subtle sm_120 SASS
    # drift that accumulated to cos 0.98 under Pi0 FP8.
    # sm_110 is not in the consumer family this multi-arch list distributes
    # for, and compute_80 SASS does not run on it, so it takes its own
    # gencode here as well rather than silently getting no cubin.
    if(GPU_ARCH STREQUAL "110")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${GPU_GENCODE}>
      )
      message(STATUS "FA2 vendor arch: sm_110a AOT (Thor)")
    else()
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_80,code=sm_80"
          "SHELL:-gencode arch=compute_120,code=sm_120"
          "SHELL:-gencode arch=compute_120,code=compute_120"
          "SHELL:-gencode arch=compute_121,code=sm_121"
          "SHELL:-gencode arch=compute_121,code=compute_121"
        >
      )
      message(STATUS "FA2 vendor arch: sm_80 + sm_120/sm_121 AOT + Blackwell PTX fallback (default)")
    endif()
  endif()

  # Macros the wrapper reads to #ifdef-guard optional hdim/dtype
  # entries. If the user restricts FA2_HDIMS="96;256", the wrapper's
  # head_dim <= 128 branch stubs out at compile time so no
  # run_mha_fwd_<..., 128, ...> symbol is referenced from wrapper.cu
  # (which would otherwise break linking).
  foreach(_hd IN LISTS FA2_BUILT_HDIMS)
    target_compile_definitions(fa2_vendor_obj PRIVATE FA2_HAS_HDIM_${_hd}=1)
  endforeach()
  foreach(_dt IN LISTS FA2_BUILT_DTYPES)
    string(TOUPPER ${_dt} _DT_UP)
    target_compile_definitions(fa2_vendor_obj PRIVATE FA2_HAS_${_DT_UP}=1)
  endforeach()
  if(FLASHRT_ENABLE_CHAMELEON)
    target_compile_definitions(fa2_vendor_obj PRIVATE
      FLASHRT_ENABLE_CHAMELEON=1)
  endif()

  string(REPLACE ";" "," _FA2_H "${FA2_BUILT_HDIMS}")
  string(REPLACE ";" "," _FA2_D "${FA2_BUILT_DTYPES}")
  list(LENGTH FA2_SRCS _FA2_NSRC)
  math(EXPR _FA2_NKERN "${_FA2_NSRC} - 1")
  message(STATUS "FA2 vendor instantiations: hdim={${_FA2_H}} x dtype={${_FA2_D}} x {split,no-split} = ${_FA2_NKERN} kernel .cu files")

  if(FLASHRT_ENABLE_NATIVE_CPP)
    # Opt-in Python-free linkage boundary for native C++ runtimes.
    add_library(flashrt_fa2_raw SHARED)
    target_sources(flashrt_fa2_raw PRIVATE
      $<TARGET_OBJECTS:fa2_vendor_obj>)
    target_link_libraries(flashrt_fa2_raw PRIVATE CUDA::cudart)
    if(MSVC)
      target_compile_definitions(flashrt_fa2_raw PRIVATE
        FLASHRT_FA2_NATIVE_BUILD=1)
    else()
      target_link_options(flashrt_fa2_raw PRIVATE
        "LINKER:--no-undefined")
      if(UNIX AND NOT APPLE)
        target_link_options(flashrt_fa2_raw PRIVATE
          "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/fa2_native_exports.map")
      endif()
    endif()
    set_target_properties(flashrt_fa2_raw PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
      BUILD_WITH_INSTALL_RPATH ON)
    install(TARGETS flashrt_fa2_raw
      DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
    message(STATUS "FA2 native C library: flashrt_fa2_raw (opt-in)")
  endif()

  if(FLASHRT_BUILD_FA2_PYTHON_ADAPTER)
    # Independent pybind module. Caller side:
    #     from flash_rt import flash_rt_fa2 as fa2
    #     fa2.fwd_fp16(...)  /  fa2.fwd_bf16(...)
    pybind11_add_module(flash_rt_fa2 csrc/fa2_bindings.cpp)
    set_target_properties(flash_rt_fa2 PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
    target_include_directories(flash_rt_fa2 PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    )
    if(FLASHRT_ENABLE_CHAMELEON)
      target_compile_definitions(flash_rt_fa2 PRIVATE
        FLASHRT_ENABLE_CHAMELEON=1)
    endif()
    if(FLASHRT_ENABLE_NATIVE_CPP)
      target_compile_definitions(flash_rt_fa2 PRIVATE
        FLASHRT_FA2_NATIVE_SHARED=1)
      target_link_libraries(flash_rt_fa2 PRIVATE
        flashrt_fa2_raw CUDA::cudart)
      set_target_properties(flash_rt_fa2 PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH ON)
    else()
      # This is the compatibility path and must remain the default.
      target_sources(flash_rt_fa2 PRIVATE
        $<TARGET_OBJECTS:fa2_vendor_obj>)
      target_link_libraries(flash_rt_fa2 PRIVATE CUDA::cudart)
    endif()
    install(TARGETS flash_rt_fa2
      DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
    message(STATUS "FA2 pybind module: flash_rt_fa2 (separate .so)")
  endif()
endif()

# ── Qwen3-VL kernels (separate module, additive) ──
# Independent pybind module so the shared flash_rt_kernels.so is not rebuilt
# for this model. Caller side:
#     from flash_rt import flash_rt_qwen3_vl_kernels as vlk
#     vlk.rope_neox_qk_bf16(...)
# ── OmniVoice fused kernels (standalone module, independent of flash_rt_kernels) ──
if(FLASHRT_ENABLE_OMNIVOICE)
  if(NOT GPU_ARCH STREQUAL "120")
    message(FATAL_ERROR
      "FLASHRT_ENABLE_OMNIVOICE requires GPU_ARCH=120 (Blackwell / sm_120); "
      "got '${GPU_ARCH}'.")
  endif()
  pybind11_add_module(flash_rt_omnivoice
    csrc/omnivoice_extra_bindings.cpp
    csrc/kernels/omnivoice/omnivoice_cfg_combine.cu
    csrc/kernels/omnivoice/omnivoice_qk_norm_rope.cu)
  set_target_properties(flash_rt_omnivoice PROPERTIES
    CUDA_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  target_include_directories(flash_rt_omnivoice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels)
  target_compile_options(flash_rt_omnivoice PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      -U__CUDA_NO_BFLOAT16_OPERATORS__
      -U__CUDA_NO_BFLOAT162_OPERATORS__
      ${GPU_GENCODE}
    >)
  target_link_libraries(flash_rt_omnivoice PRIVATE CUDA::cudart)
  install(TARGETS flash_rt_omnivoice LIBRARY DESTINATION flash_rt)
  message(STATUS "OmniVoice TTS SM120 kernels: ENABLED")
else()
  message(STATUS "OmniVoice TTS SM120 kernels: DISABLED")
endif()

# ── MiniMax-Remover fused kernels (standalone module, independent
#    of flash_rt_kernels) ──
# Builds flash_rt_minimax_remover.so with MiniMax-specific FP8/NVFP4
# transformer kernels plus fused VAE helpers. The module is opt-in and
# remains separate from the default flash_rt_kernels target.
# Enable explicitly on a Blackwell (sm_120) build:
#   cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
if(FLASHRT_ENABLE_MINIMAX_REMOVER)
  pybind11_add_module(flash_rt_minimax_remover
    csrc/minimax_remover_extra_bindings.cpp
    csrc/kernels/minimax_remover/fp16_rms_norm_ncdhw.cu
    csrc/kernels/minimax_remover/fp16_rms_silu_ncdhw.cu
    csrc/kernels/minimax_remover/fp16_rms_norm_ndhwc.cu
    csrc/kernels/minimax_remover/fp8_conv3d_mm_ndhwc_fp16out.cu
  csrc/kernels/minimax_remover/fp16_quant_fp8_per_tensor.cu
  csrc/kernels/minimax_remover/fp16_rms_silu_fp8_ndhwc.cu
  csrc/kernels/minimax_remover/fp16_bias_gelu_quant_fp8.cu
  csrc/kernels/minimax_remover/fp16_bias_gate_residual.cu
  csrc/kernels/minimax_remover/fp16_ada_layernorm_quant_fp8.cu
  csrc/kernels/minimax_remover/fp16_rmsnorm_rope.cu
  csrc/kernels/minimax_remover/fp16_rmsnorm_rope_quant_int8.cu
  csrc/kernels/minimax_remover/fp16_quant_nvfp4_ndhwc.cu
  csrc/kernels/minimax_remover/nvfp4_conv3d_ndhwc_fp16out.cu)
  set_target_properties(flash_rt_minimax_remover PROPERTIES
    CUDA_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  target_include_directories(flash_rt_minimax_remover PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels)
  target_compile_options(flash_rt_minimax_remover PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      -U__CUDA_NO_HALF_OPERATORS__
      -U__CUDA_NO_HALF2_OPERATORS__
      -U__CUDA_NO_HALF_CONVERSIONS__
      ${GPU_GENCODE}
    >)
  target_link_libraries(flash_rt_minimax_remover PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
  install(TARGETS flash_rt_minimax_remover LIBRARY DESTINATION flash_rt)
  message(STATUS "MiniMax-Remover FP8/NVFP4 kernels: ENABLED")
else()
  message(STATUS "MiniMax-Remover FP8/NVFP4 kernels: DISABLED")
endif()

if(FLASHRT_BUILD_QWEN3_VL)
  # SM120 (RTX 5090): NVFP4/FP8 ViT helpers. SM89 (Ada): native FP8 block-128
  # GEMM/GEMV + fused act/norm-quant + QK norm-rope kernels for the official
  # FP8 Qwen3-VL path. SM87 (Jetson Orin): BF16 ViT helper module only; the
  # language path reuses the generic flash_rt_kernels BF16 Qwen3 helpers.
  # SM110 (Jetson Thor): BF16 ViT tower (SDPA attention, no FP8 block-128
  # path) plus the M=1 decode GEMVs and the batched QK norm-rope prefill.
  # All build into a separate module so neither bloats flash_rt_kernels.so.
  if(NOT (GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "89" OR
          GPU_ARCH STREQUAL "87" OR GPU_ARCH STREQUAL "110"))
    message(FATAL_ERROR
      "FLASHRT_BUILD_QWEN3_VL requires GPU_ARCH=120 (RTX 5090 / sm_120) or "
      "GPU_ARCH=89 (Ada / sm_89) or GPU_ARCH=87 (Jetson Orin / sm_87) or "
      "GPU_ARCH=110 (Jetson Thor / sm_110); got '${GPU_ARCH}'.")
  endif()

  set(QWEN3_VL_SOURCES
    csrc/qwen3_vl_bindings.cpp
    csrc/kernels/rope_neox_qk_bf16.cu
    csrc/kernels/bias_epilogue_bf16.cu)

  if(GPU_ARCH STREQUAL "120")
    list(APPEND QWEN3_VL_SOURCES
      csrc/kernels/norm_act_to_fp8_block128.cu)
  elseif(GPU_ARCH STREQUAL "89")
    # Ada has no TMA / no warp-specialized CUTLASS collective, so these are
    # hand-written cp.async + ldmatrix + block-128 scaling kernels (see
    # docs/qwen3_vl_fp8_sm89.md for why the SM120a CUTLASS path does not port).
    list(APPEND QWEN3_VL_SOURCES
      csrc/kernels/norm_act_to_fp8_block128.cu
      csrc/gemm/fp8_block128_gemm_mma_sm89.cu
      csrc/gemm/fp8_gemv_m1_sm89.cu
      csrc/quantize/fp8_per_token_block_quant.cu
      csrc/kernels/qwen3_qkv_post_proc.cu
      csrc/kernels/bf16_matmul_bf16.cu)
  elseif(GPU_ARCH STREQUAL "87")
    # Weight-only quantized decode GEMVs ship alongside the bf16 one; the
    # int8/int4 pair is the Ampere-friendly tier (hardware I2F dequant),
    # while e4m3/e2m1 need the FP8 conversion sm_87 lacks.
    list(APPEND QWEN3_VL_SOURCES
      csrc/kernels/bf16_matmul_bf16.cu
      csrc/kernels/qwen3_vl_bf16_gemv_m1.cu
      csrc/kernels/qwen3_vl_int8_gemv_m1.cu
      csrc/kernels/qwen3_vl_int4_gemv_m1.cu
      csrc/kernels/qwen3_vl_w8_gemv_m1.cu
      csrc/kernels/qwen3_vl_w4_gemv_m1.cu
      csrc/kernels/qwen3_int8_kv.cu)
  elseif(GPU_ARCH STREQUAL "110")
    # Thor: BF16 ViT tower (SDPA attention — no FP8 block-128 ViT path here),
    # the e4m3/e2m1 decode GEMVs (Thor has hardware FP8 conversion, unlike
    # Orin), plus the batched QK norm-rope kernel the BF16 prefill uses.
    list(APPEND QWEN3_VL_SOURCES
      csrc/kernels/bf16_matmul_bf16.cu
      csrc/kernels/qwen3_vl_bf16_gemv_m1.cu
      csrc/kernels/qwen3_vl_w8_gemv_m1.cu
      csrc/kernels/qwen3_vl_w4_gemv_m1.cu
      csrc/kernels/qwen3_qkv_post_proc.cu)
  endif()

  pybind11_add_module(flash_rt_qwen3_vl_kernels ${QWEN3_VL_SOURCES})
  if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "89")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_FP8_ACT=1)
  endif()
  if(GPU_ARCH STREQUAL "87" OR GPU_ARCH STREQUAL "89" OR
     GPU_ARCH STREQUAL "110")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_BF16_CUBLASLT=1)
  endif()
  if(GPU_ARCH STREQUAL "87" OR GPU_ARCH STREQUAL "110")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_BF16_GEMV_M1=1)
  endif()
  # INT8/INT4 decode GEMVs + INT8 KV cache: Orin only. Thor has hardware FP8
  # conversion so it uses the e4m3/e2m1 tiers, and its attention backend has no
  # INT8 KV path, so building these there would export unused bindings.
  if(GPU_ARCH STREQUAL "87")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_INT_DECODE=1)
  endif()
  # Arches that compile qwen3_qkv_post_proc.cu. Kept separate from
  # ENABLE_SM89_BLOCK_FP8_GEMM so Thor binds these without the FP8 path.
  if(GPU_ARCH STREQUAL "89" OR GPU_ARCH STREQUAL "110")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_QKV_POSTPROC=1)
  endif()
  if(GPU_ARCH STREQUAL "89")
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_SM89_BLOCK_FP8_GEMM=1)
  endif()
  if(FLASHRT_ENABLE_QWEN3_VL_DEV_KERNELS)
    target_compile_definitions(flash_rt_qwen3_vl_kernels PRIVATE
      ENABLE_QWEN3_VL_DEV_KERNELS=1)
  endif()
  set_target_properties(flash_rt_qwen3_vl_kernels PROPERTIES
    CUDA_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  target_include_directories(flash_rt_qwen3_vl_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels)
  target_compile_options(flash_rt_qwen3_vl_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      -U__CUDA_NO_BFLOAT16_OPERATORS__
      -U__CUDA_NO_BFLOAT162_OPERATORS__
      ${GPU_GENCODE}
    >)
  target_link_libraries(flash_rt_qwen3_vl_kernels PRIVATE CUDA::cudart)
  if(GPU_ARCH STREQUAL "87" OR GPU_ARCH STREQUAL "89" OR
     GPU_ARCH STREQUAL "110")
    # bf16_matmul_bf16 (ViT/language bf16 linears) uses cuBLASLt.
    target_link_libraries(flash_rt_qwen3_vl_kernels PRIVATE CUDA::cublasLt)
  endif()
  install(TARGETS flash_rt_qwen3_vl_kernels
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  message(STATUS "Qwen3-VL pybind module: flash_rt_qwen3_vl_kernels (separate .so)")
endif()

# ── Qwen3.6 FlashInfer XQA FP8-KV (SM120/SM121 + SM110/Thor) ──
# Vendored FlashInfer XQA mha.cu handles Blackwell SM120/SM121 and
# Thor SM110 in the same translation unit; the differentiation is
# purely the -gencode flag forwarded via GPU_GENCODE.
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121" OR
   GPU_ARCH STREQUAL "110")
  add_library(qwen36_flashinfer_xqa_obj OBJECT
    csrc/attention/flashinfer_xqa_src/mha.cu
    csrc/kernels/qwen36_flashinfer_xqa.cu)
  set_target_properties(qwen36_flashinfer_xqa_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(qwen36_flashinfer_xqa_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/flashinfer_xqa_src)
  target_compile_definitions(qwen36_flashinfer_xqa_obj PRIVATE
    NDEBUG=1
    BEAM_WIDTH=1
    USE_INPUT_KV=0
    USE_CUSTOM_BARRIER=1
    MLA_WRAPPER=0
    USE_SM90_MHA=0
    INPUT_FP16=0
    DTYPE=__nv_bfloat16
    CACHE_ELEM_ENUM=2
    TOKENS_PER_PAGE=128
    HEAD_ELEMS=256
    HEAD_GRP_SIZE=6
    SLIDING_WINDOW=0
    LOW_PREC_OUTPUT=0
    SPEC_DEC=1)
  target_compile_options(qwen36_flashinfer_xqa_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      --ftz=true --prec-div=false --prec-sqrt=false
      -U__CUDA_NO_BFLOAT16_OPERATORS__
      -U__CUDA_NO_BFLOAT162_OPERATORS__
      ${GPU_GENCODE}
    >)
endif()

# ── Main module ──
# flash_rt_kernels is the shared core pybind module. It currently compiles a
# broad superset of kernels, including several model/architecture-specific
# translation units that not every deployment needs (Qwen3.6/linear-attention,
# SM120/NVFP4-named, Motus/video FP8-history). For VLA deployments on weaker
# machines this is avoidable build latency.
#
# An incremental cleanup gates those model/arch-specific TUs behind explicit
# CMake options so a slim build compiles only what a given hardware/model path
# needs. This is COMPILE-TIME surface reduction only: it does not change kernel
# math, launch params, dtype/dispatch policy, graph capture, or runtime
# fallback. Default builds stay compatibility-oriented (broad set) until the
# slim path is validated across SM89/SM120/Thor. See scripts/build_inventory.py
# and tests/test_build_inventory.py for the measured per-target surface.
pybind11_add_module(flash_rt_kernels
  csrc/gemm/gemm_runner.cu
  csrc/kernels/norm.cu
  csrc/kernels/activation.cu
  csrc/kernels/rope.cu
  csrc/kernels/quantize.cu
  csrc/kernels/elementwise.cu
  csrc/kernels/fusion.cu
  csrc/kernels/patch_embed.cu
  csrc/kernels/softmax.cu
  csrc/kernels/attention_cublas.cu
  csrc/kernels/attention_mha.cu
  csrc/kernels/attention_mha_causal.cu
  csrc/kernels/rope_qwen3.cu
  csrc/kernels/decoder_fused.cu
  csrc/kernels/dit_bf16.cu
  csrc/kernels/attention_dit_bf16.cu
  csrc/quantize/fp8_block128_dequant.cu
  csrc/quantize/fp8_per_token_block_quant.cu
  csrc/quantize/tq_dequant_kv.cu
  csrc/quantize/wmma_probe.cu
  csrc/quantize/tq_dequant_cutlass.cu
  csrc/quantize/ada_layer_norm_fp8.cu
  csrc/quantize/bias_gelu_quantize_fp8.cu
  csrc/quantize/qkv_split_norm_rope_bf16.cu
  csrc/quantize/rope_apply_bf16.cu
  csrc/gemm/fp8_block128_gemm.cu
  csrc/kernels/silu_mul_qwen36.cu
  csrc/kernels/embedding_lookup_bf16.cu
  # qwen3_5_moe-family RTX SM120 kernels are NOT listed here -- they are
  # gated into the module below (FLASHRT_ENABLE_QWEN35MOE), so SM89/87/110
  # and non-Nex-N2 builds never compile them. See the gated block after the
  # target definition.
  csrc/kernels/qwen3_qkv_post_proc.cu
  csrc/kernels/bf16_matvec_qwen36.cu
  csrc/kernels/bf16_matmul_bf16.cu
  csrc/attention/fmha_dispatch.cu
  csrc/bindings.cpp
)
set_target_properties(flash_rt_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)

# ── Thor-class VLA helper kernels (SM100/SM110 only) ──
# The masked-softmax MHA variants, the 16-byte-load norm/rope/quantize
# rewrites, the seqused-fold decoder attention and the vectorized QKV
# split-RoPE exist for the GROOT N1.7 and Pi0.5 Thor paths; their element
# math targets SM100-class memory behaviour and no SM8x frontend calls
# them. Their bindings in csrc/bindings.cpp are guarded by
# FLASHRT_HAVE_THOR_VLA_KERNELS so source and binding drop together on
# other architectures.
if(ENABLE_SM100_CUTLASS)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/attention_mha_masked.cu
    csrc/kernels/vec_fp16_backbone.cu
    csrc/kernels/attention_seqused_fused.cu
    csrc/kernels/rope_vec.cu
    csrc/kernels/qk_norm_rope_rotate_half_bf16.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_THOR_VLA_KERNELS=1)
  message(STATUS "Thor VLA helper kernels: ENABLED (sm_${GPU_ARCH})")
else()
  message(STATUS "Thor VLA helper kernels: DISABLED (SM100-class only)")
endif()

if(FLASHRT_ENABLE_COSMOS3_EDGE)
  target_sources(flash_rt_kernels PRIVATE csrc/kernels/cosmos3_edge_misc.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_COSMOS3_EDGE=1)
  message(STATUS "Cosmos3-Edge AV kernels: ENABLED (sm_110)")
else()
  message(STATUS "Cosmos3-Edge AV kernels: DISABLED")
endif()

if(FLASHRT_ENABLE_COSMOS3_REASONER)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/cosmos3_reasoner_attn.cu
    csrc/kernels/cosmos3_reasoner_gemv.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_COSMOS3_REASONER=1)
  message(STATUS "Cosmos3 Reasoner kernels: ENABLED (sm_110)")
else()
  message(STATUS "Cosmos3 Reasoner kernels: DISABLED")
endif()

if(FLASHRT_ENABLE_AUDIO_CODEBOOK)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/delayed_codebook_kernels.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_AUDIO_CODEBOOK=1)
  message(STATUS "Delayed audio-codebook kernels: ENABLED")
else()
  message(STATUS "Delayed audio-codebook kernels: DISABLED")
endif()

# ── Motus VAE FP8 quantize kernels (slim-gated) ──
# These 5 TUs are used only by the Motus VAE FP8 path (the *_ncdhw* /
# upsample / per-tensor AWQ-FP8 quant helpers; see flash_rt/models/motus/*).
# They compile unconditionally today even on SM89 where no Motus model runs,
# so a slim VLA build drops them. The matching bindings in csrc/bindings.cpp
# are guarded by FLASHRT_HAVE_MOTUS_VAE_FP8 so they drop in lockstep (a gated
# source whose binding stayed compiled would fail to link). The Motus Python
# frontend already probes these with hasattr(), so a missing binding degrades
# clearly instead of crashing on import.
#
# NOTE: ada_layer_norm_fp8.cu and bias_gelu_quantize_fp8.cu are intentionally
# NOT in this group — they are shared (ada_layer_norm_fp8 is used by GROOT N1.7,
# bias_gelu_quantize_fp8_static_bf16 by Melband-RoFormer), so they stay always
# compiled. Do not fold them in without splitting their shared symbols out.
if(NOT FLASHRT_SLIM_BUILD)
  target_sources(flash_rt_kernels PRIVATE
    csrc/quantize/awq_quant_fp8_static_bf16.cu
    csrc/quantize/bf16_ndhwc_to_ncdhw_transpose.cu
    csrc/quantize/bf16_quant_fp8_ncdhw_to_ndhwc.cu
    csrc/quantize/bf16_rms_silu_ncdhw.cu
    csrc/quantize/bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_MOTUS_VAE_FP8=1)
  message(STATUS "Motus VAE FP8 quantize kernels: ENABLED")
else()
  message(STATUS "Motus VAE FP8 quantize kernels: SKIPPED (slim build)")
endif()

# ── Qwen3.6 / linear-attention kernels (slim-gated) ──
# These 10 TUs are used only by the Qwen3.6 family: Qwen3.6 dense/thor/spark
# and Nex-N2 (the Qwen3.6-35B-A3B MoE). They cover causal-conv1d, the gated
# DeltaNet / linear-attention WY stack, the Qwen3.6 fused RMSNorm-gated-SiLU,
# Qwen3.6 misc helpers (partial-rope / argmax / tq-prepare / spec-accept /
# legacy embedding wrapper), and the Qwen3.6 bf16 matmul (AB96 + legacy
# wrapper + Thor MTP). A slim VLA build (Qwen3-VL etc.) needs none of them.
# Their bindings in csrc/bindings.cpp are guarded by FLASHRT_HAVE_QWEN36_KERNELS
# so they drop in lockstep (a gated source whose binding stayed compiled would
# fail to link).
#
# NOTE: silu_mul_qwen36.cu and bf16_matvec_qwen36.cu are intentionally NOT in
# this group — silu_mul_qwen36_bf16 is shared (Qwen3-dense / Higgs / Cosmos3)
# and bf16_matvec_qwen36_bf16 is used by Higgs, so they stay always compiled.
# The neutral helpers bf16_matmul_bf16.cu / embedding_lookup_bf16.cu also stay
# (#112); only the legacy wrappers bf16_matmul_qwen36_bf16 /
# qwen36_embedding_lookup_bf16 live inside the gated TUs. Do not fold the two
# shared files in without splitting their shared symbols out.
if(NOT FLASHRT_SLIM_BUILD)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/causal_conv1d_qwen36.cu
    csrc/kernels/linear_attention/gated_delta_wy_bf16.cu
    csrc/kernels/linear_attention/gated_delta_wy_bf16_mma_fla.cu
    csrc/kernels/linear_attention/gated_delta_wy_output_o_mma_fla.cu
    csrc/kernels/linear_attention/gated_delta_wy_recompute_wu_mma_fla.cu
    csrc/kernels/gated_deltanet_qwen36.cu
    csrc/kernels/rms_norm_gated_silu_qwen36.cu
    csrc/kernels/qwen36_misc.cu
    csrc/kernels/bf16_matmul_qwen36.cu
    csrc/kernels/bf16_matmul_qwen36_thor.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_QWEN36_KERNELS=1)
  message(STATUS "Qwen3.6 / linear-attention kernels: ENABLED")
else()
  message(STATUS "Qwen3.6 / linear-attention kernels: SKIPPED (slim build)")
endif()

# ── SM120 / NVFP4-named kernels (slim-gated, arch-macro keyed) ──
# These TUs are named/built for SM120 NVFP4 but today compile unconditionally,
# wasting build time on SM89 where they are unreachable. Each group compiles
# when "NOT slim build OR <its existing arch macro is ON>": this keeps the
# default SM89 surface broad (compatibility), drops the kernels only in a slim
# SM89 build, and GUARANTEES they stay compiled on any SM120/NVFP4 build (slim
# or not) where the arch macro is ON.
#
# Group A — fp4_w4a4_* SM120 NVFP4 W4A16 GEMM/GEMV. Their bindings are already
# inside #ifdef ENABLE_CUTLASS_SM120_NVFP4_W4A16 in bindings.cpp, so on SM89 the
# Python symbols never exist; only the source was compiling. No bindings change.
# Keyed on the SAME condition that defines that macro (GPU_ARCH 120/121, see the
# ENABLE_CUTLASS_SM120_NVFP4_W4A16 block below) — the macro is a compile-def, not
# a CMake variable, so it must not be tested as one here.
if(NOT FLASHRT_SLIM_BUILD OR GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/fp4_w4a4_matvec_sm120.cu
    csrc/kernels/fp4_w4a4_mma_sm120.cu
    csrc/kernels/fp4_w4a4_mma_warpsplit_sm120.cu
    csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cu)
  message(STATUS "SM120 NVFP4 W4A16 fp4_w4a4 kernels: ENABLED")
else()
  message(STATUS "SM120 NVFP4 W4A16 fp4_w4a4 kernels: SKIPPED (slim build)")
endif()

# Group B — NVFP4 swizzle/quantize helpers. Their bindings are unguarded today
# (exposed on SM89 but runtime-unreachable: every Python loader self-gates on
# has_nvfp4()). Gating the sources requires guarding their bindings + includes
# in lockstep; FLASHRT_HAVE_NVFP4_SWIZZLE drives that guard and is defined on
# the same "NOT slim OR ENABLE_NVFP4" condition.
if(NOT FLASHRT_SLIM_BUILD OR ENABLE_NVFP4)
  target_sources(flash_rt_kernels PRIVATE
    csrc/quantize/fp8_block128_to_nvfp4_swizzled.cu
    csrc/quantize/bf16_weight_to_nvfp4_swizzled.cu
    csrc/kernels/silu_mul_to_nvfp4_swizzled.cu
    csrc/kernels/fp4_swiglu_compact_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_NVFP4_SWIZZLE=1)
  message(STATUS "NVFP4 swizzle/quantize kernels: ENABLED")
else()
  message(STATUS "NVFP4 swizzle/quantize kernels: SKIPPED (slim build)")
endif()

# ── Shared SM120 dense GEMM ──
# Qwen3-VL and qwen3_5_moe both consume this core symbol. Keep its ownership
# independent of the larger, opt-in MoE source group below.
if((FLASHRT_BUILD_QWEN3_VL AND ENABLE_NVFP4) OR
   FLASHRT_ENABLE_QWEN35MOE_CORE)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/w16a16_gemm_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    FLASHRT_HAVE_W16A16_SM120=1)
  message(STATUS "SM120 W16A16 dense GEMM: ENABLED")
endif()

# ── qwen3_5_moe family (Nex-N2 / Qwen3.6-35B-A3B) kernels ──
# Three tiers, each with its own compile definition. The matching bindings in
# csrc/bindings.cpp are guarded on the same macros, so a tier that is off
# contributes neither symbols nor translation units.
if(FLASHRT_ENABLE_QWEN35MOE_CORE)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/qwen35moe_layout.cu
    csrc/kernels/bf16_matvec_sm120.cu
    csrc/kernels/gdn_recurrent_seq_sm120.cu
    csrc/kernels/gdn_wy_prefill_edge.cu
    csrc/kernels/causal_conv1d_rows_edge.cu
    csrc/kernels/act_fuse_sm120.cu
    csrc/kernels/moe_router_topk_sm120.cu
    csrc/kernels/moe_route_prefill_edge.cu
    csrc/kernels/moe_shared_combine_edge.cu
    csrc/kernels/moe_weighted_sum_sm120.cu
    csrc/kernels/qwen35moe_e0m3_dequant.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    FLASHRT_HAVE_QWEN35MOE_CORE=1)
  message(STATUS "qwen3_5_moe core kernels: ENABLED (sm_${GPU_ARCH})")
endif()

if(FLASHRT_ENABLE_QWEN35MOE_W4A16)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/w4a16_matvec_sm120.cu
    csrc/kernels/moe_grouped_w4a16_sm120.cu
    csrc/kernels/w4a16_edge_sm120.cu
    csrc/kernels/w4a16_mrows_edge_sm120.cu
    csrc/kernels/w4a16_gemm_sm120.cu
    csrc/kernels/qwen35moe_grouped_quant.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    FLASHRT_HAVE_QWEN35MOE_W4A16=1)
  # Packed-weight loads in flight per row in the W4A16 GEMVs. The kernels
  # default to 4, which is the value the SM120 path was validated with. Thor
  # measures faster at 2 because the kernel is register-limited there (8 blocks
  # per SM where every other limit allows 24); that is a property of a 20-SM
  # part and is not assumed to transfer, so it is set only for that target.
  # Either value produces bit-identical output; see w4a16_edge_sm120.cu.
  if(GPU_ARCH STREQUAL "110")
    target_compile_definitions(flash_rt_kernels PRIVATE
      FLASHRT_W4A16_EDGE_UNROLL=2)
  endif()
  message(STATUS "qwen3_5_moe weight-only 4-bit kernels: ENABLED (sm_${GPU_ARCH})")
endif()

# Block-scaled 4-bit MMA needs the sm_120a/sm_121a CUTLASS path. Refuse rather
# than build silently broken kernels on other targets.
if(FLASHRT_ENABLE_QWEN35MOE_W4A4)
  if(NOT ENABLE_NVFP4)
    message(FATAL_ERROR
      "FLASHRT_ENABLE_QWEN35MOE_W4A4 requires a block-scaled-MMA target "
      "(current GPU_ARCH=${GPU_ARCH}). CUTLASS compiles these kernels on "
      "other architectures but replaces the MMA with an invalid control "
      "path, so they would fail at run time. Use "
      "FLASHRT_ENABLE_QWEN35MOE_CORE and FLASHRT_ENABLE_QWEN35MOE_W4A16 "
      "instead.")
  endif()
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/moe_grouped_gemv_sm120.cu
    csrc/kernels/moe_m16_mma_sm120.cu
    csrc/kernels/moe_m64_mma_sm120.cu
    csrc/kernels/moe_blocktile_mma_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    FLASHRT_HAVE_QWEN35MOE_W4A4=1)
  message(STATUS "qwen3_5_moe block-scaled 4-bit kernels: ENABLED (sm_${GPU_ARCH})")
endif()

# ── MelBandRoformer custom fused kernels (BF16/FP8, gated) ──
if(FLASHRT_ENABLE_MELBAND_ROFORMER)
  target_sources(flash_rt_kernels PRIVATE csrc/kernels/mbr_kernels.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_MELBAND_ROFORMER=1)
  message(STATUS "MelBandRoformer kernels: ENABLED")
else()
  message(STATUS "MelBandRoformer kernels: DISABLED")
endif()

if(ENABLE_SM100_CUTLASS)
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_sm100_obj>
    $<TARGET_OBJECTS:cutlass_sm100_fp16_obj>
    $<TARGET_OBJECTS:flashrt_megakernel_obj>
    $<TARGET_OBJECTS:flashrt_megakernel_geglu_obj>
    $<TARGET_OBJECTS:flashrt_megakernel_geglu_g8_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_SM100_CUTLASS=1)
  if(FLASHRT_BUILD_SM100_SWEEP)
    target_sources(flash_rt_kernels PRIVATE $<TARGET_OBJECTS:cutlass_sm100_fp16_sweep_obj>)
    target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_SM100_SWEEP=1)
  endif()
  if(FLASHRT_BUILD_SM100_ENCODER_MLP)
    target_sources(flash_rt_kernels PRIVATE $<TARGET_OBJECTS:encoder_mlp_sm100_obj>)
    target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_SM100_ENCODER_MLP=1)
  endif()
endif()

if(ENABLE_NVFP4)
  target_sources(flash_rt_kernels PRIVATE $<TARGET_OBJECTS:gemm_kernels_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_NVFP4=1)
endif()

if(ENABLE_SM80_INT8_CUTLASS)
  target_sources(flash_rt_kernels PRIVATE
    csrc/gemm/cutlass_sm80_int8_rowwise.cu
    csrc/gemm/cutlass_sm80_int8_rowwise_t64x128.cu
    csrc/gemm/cutlass_sm80_int8_silu_gated.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_SM80_INT8_CUTLASS=1)
endif()

# ── Chameleon-7B model kernels (opt-in, see FLASHRT_ENABLE_CHAMELEON) ──
# QK Norm/RoPE fused, AWQ FP16 quant, SM80 INT8/INT4 rowwise GEMM fp16-out +
# FHT/QuaRot rotation. Everything here is Chameleon-specific; model-neutral
# fp16 norm/quant/activation helpers live in the common layer.
if(FLASHRT_ENABLE_CHAMELEON)
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/qk_norm_rope_fused.cu
    csrc/quantize/awq_quant_fp8_static_fp16.cu)
  if(ENABLE_SM80_INT8_CUTLASS)
    target_sources(flash_rt_kernels PRIVATE
      csrc/gemm/cutlass_sm80_int8_rowwise_fp16out.cu
      csrc/gemm/cutlass_sm80_int8_rowwise_fp16out_t64x128.cu
      csrc/gemm/cutlass_sm80_int8_rowwise_fp16out_t256x128.cu
      csrc/gemm/cutlass_sm80_int4_rowwise.cu
      csrc/kernels/fht_int4.cu)
  endif()
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_ENABLE_CHAMELEON=1)
  message(STATUS "Chameleon-7B model kernels: ENABLED")
endif()

# SM120a CUTLASS block-128 FP8 GEMM (Path B for Qwen3.6).
if(GPU_ARCH STREQUAL "120")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_sm120_block_obj>
    $<TARGET_OBJECTS:sm120_fp8_smallm_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_CUTLASS_SM120_BLOCK_FP8=1)
endif()

# Dedicated M=1 decode GEMV (FP8 + BF16); linked + macro defined wherever the
# object is built (sm89 / sm110 / sm120) so the ENABLE_DECODE_GEMV_M1 bindings always
# resolve, independent of FLASHRT_ENABLE_MOTUS.
if(GPU_ARCH STREQUAL "110" OR GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121" OR
   GPU_ARCH STREQUAL "89")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:decode_gemv_m1_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_DECODE_GEMV_M1=1)
endif()

if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:qwen3_prefill_sm120_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_SAGE2_F8_RAW=1
    ENABLE_QWEN3_FP8_PREFILL_ATTN=1)
endif()

# SM120a CUTLASS NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt path).
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_nvfp4_w4a16_sm120_obj>
    $<TARGET_OBJECTS:sm120_nvfp4_ffn_obj>
    csrc/quantize/nvfp4_sf_reshape_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_CUTLASS_SM120_NVFP4_W4A16=1)
  if(FLASHRT_ENABLE_SM120_DEV_KERNELS)
    target_sources(flash_rt_kernels PRIVATE
      $<TARGET_OBJECTS:sm120_nvfp4_dev_obj>)
    target_compile_definitions(flash_rt_kernels PRIVATE
      FLASHRT_HAVE_SM120_NVFP4_DEV=1)
  endif()
endif()

# SM100 CUTLASS NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt on Thor).
# The SF-reshape kernel is a pure memory permutation (no MMA, no
# arch-specific intrinsics), so the same translation unit compiles
# under both gencode flags.
if(GPU_ARCH STREQUAL "110")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_nvfp4_w4a16_sm100_obj>
    csrc/quantize/nvfp4_sf_reshape_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_CUTLASS_SM100_NVFP4_W4A16=1)
endif()

# The grouped MoE GEMM built above, linked and declared only when the model
# tier that calls it is on.
if(ENABLE_QWEN35MOE_GROUPED_SM100)
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:qwen35moe_nvfp4_grouped_sm100_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    FLASHRT_HAVE_QWEN35MOE_GROUPED_SM100=1)
endif()

if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "121" OR
   GPU_ARCH STREQUAL "110")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:qwen36_flashinfer_xqa_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_QWEN36_FLASHINFER_XQA=1)
endif()

if(GPU_ARCH STREQUAL "110")
  target_sources(flash_rt_kernels PRIVATE
    csrc/conv/fp8_conv3d_sm120_v17.cu
    csrc/conv/fp8_conv3d_sm120_v18.cu
    csrc/conv/bf16_conv3d_v0_sm110.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_FP8_CONV3D_V17=1
    ENABLE_FP8_CONV3D_V18=1
    ENABLE_BF16_CONV3D_V0=1)
endif()

if(FLASHRT_ENABLE_MOTUS AND GPU_ARCH STREQUAL "120")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:motus_aux_sm120_obj>
    $<TARGET_OBJECTS:motus_action_ffn_v6t_sm120_obj>
    $<TARGET_OBJECTS:motus_und_ffn_sm120_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_MOTUS=1
    ENABLE_MOTUS_SAGE2_RAW=1
    ENABLE_FP8_CONV3D_V17=1
    ENABLE_FP8_CONV3D_V18=1
    ENABLE_FP8_CONV2D_3X3_V1=1
    ENABLE_ACTION_FFN_MEGAKERNEL_V6T=1
    ENABLE_UND_FFN_MEGAKERNEL_V5T=1
    ENABLE_TINYFP8_KERNELS=1)
endif()

# (ENABLE_FA2 object-lib is linked directly into flash_rt_fa2.so by default.
# Native C++ builds opt into libflashrt_fa2_raw.so and may link the adapter
# against that single implementation. The main
# flash_rt_kernels.so deliberately does NOT pull FA2 in, so rebuilds
# of our hand-written kernels don't re-trigger the FA2 codegen tax.)

target_include_directories(flash_rt_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp4
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/conv
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/conv/motus
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantize
  ${CUTLASS_INCLUDE}
  ${CUTLASS_DIR}/tools/util/include
)

# ── Hy-Embodied-0.5-VLA model kernels (opt-in) ──
# HyVLA sources compile only when FLASHRT_ENABLE_HYVLA=ON together with the
# matching architecture gate. HyVLA OFF leaves the default SM87/SM110 build
# and symbol surface unchanged.
option(FLASHRT_ENABLE_HYVLA "Build HyVLA model kernels" OFF)
if(FLASHRT_ENABLE_HYVLA AND GPU_ARCH STREQUAL "110")
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/hyvla_fused_thor.cu
    csrc/kernels/hyvla_vit_fuse.cu
    csrc/kernels/hyvla_quant_fp8_thor.cu
    csrc/kernels/hyvla_ffn_fp8_thor.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_HYVLA_THOR=1)
  message(STATUS "Hy-VLA Thor SM110 kernels: ENABLED")
endif()

# The fused RoPE+QKNorm+KV-write megakernel is plain CUDA (no SM110-only
# instructions) and removes ~11 launches per attention block; the FP8
# Thor kernels stay SM110-only.
if(FLASHRT_ENABLE_HYVLA AND GPU_ARCH STREQUAL "87")
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/hyvla_fused_thor.cu
    csrc/kernels/hyvla_vit_fuse.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE FLASHRT_HAVE_HYVLA_ORIN=1)
  message(STATUS "Hy-VLA Orin SM87 fused attention-prep kernel: ENABLED")
endif()

target_link_libraries(flash_rt_kernels PRIVATE
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cudart
  ${CMAKE_DL_LIBS}    # dlopen/dlsym for FMHA .so loading
)

target_compile_options(flash_rt_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr -O3
    --ftz=true --prec-div=false --prec-sqrt=false
    ${GPU_GENCODE}
  >
)

install(TARGETS flash_rt_kernels DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)


# ──────────────────────────────────────────────────────────────────────
#  NVFP4 add-on module (separate .so, additive, does NOT touch flash_rt_kernels)
#  Built only when SM100-class GPU present. Thor SM110 via -arch=sm_110a.
# ──────────────────────────────────────────────────────────────────────
if(ENABLE_SM100_CUTLASS)
  add_library(fp4_kernels_obj OBJECT
    csrc/gemm/fp4/cutlass_fp4_gemm.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_variants.cu
    csrc/quantize/quantize_fp4_dynamic.cu
    csrc/quantize/reshape_scales_sfa.cu
    csrc/quantize/quantize_fp4_sfa.cu
    csrc/quantize/quantize_fp4_sfa_vec.cu
    csrc/fused_fp16/rms_norm_noweight_fp16.cu
    csrc/fused_fp4/norm_silu_fp4_sfa.cu
    csrc/fused_fp4/silu_mul_fp4_sfa_v2.cu
    csrc/fused_fp4/silu_mul_fp4_sfa_vec.cu
    csrc/fused_fp4/layer_norm_fp4_sfa.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cu
    csrc/fused_fp4/res_rms_fp4_sfa_v2.cu
    csrc/fused_fp4/per_channel_mul_fp16.cu
    csrc/fused_fp4/res_rms_mul_fp4_sfa.cu
    csrc/fused_fp4/silu_mul_mul_fp4_sfa_v2.cu
    csrc/fused_fp4/silu_mul_two_fp4_to_fp4.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_fp4out.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cu
    csrc/quantize/quantize_fp4_sfa_bf16.cu
    csrc/fused_fp4/dit_norm_fp4_sfa.cu
    csrc/fused_fp4/silu_mul_fp4_sfa_bf16.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cu
    csrc/quantize/quantize_e0m3_sfa.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cu
    csrc/fused_fp4/pi05_e0m3_act.cu
    csrc/fused_fp4/siglip_ln_vec.cu
  )
  if(FLASHRT_ENABLE_COSMOS3_EDGE)
    target_sources(fp4_kernels_obj PRIVATE
      csrc/fused_fp4/cosmos3_edge_fp4.cu
      csrc/gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cu)
    target_compile_definitions(fp4_kernels_obj PRIVATE FLASHRT_HAVE_COSMOS3_EDGE=1)
  endif()
  set_target_properties(fp4_kernels_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(fp4_kernels_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_compile_options(fp4_kernels_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3 --use_fast_math
      ${GPU_GENCODE}
    >
  )
  # CRITICAL: without this define the host-side stub for device kernels
  # does NOT pull in CUTLASS headers, so the stub fails to compile with
  # "::cute has not been declared". Define triggers the CUTLASS-including
  # branch in cutlass_fp4_gemm.cu for BOTH device and host compilation
  # passes of nvcc.
  target_compile_definitions(fp4_kernels_obj PRIVATE CUTLASS_ARCH_MMA_SM100_SUPPORTED=1)
  message(STATUS "NVFP4 add-on module (flash_rt_fp4): building for sm_${GPU_ARCH}a")

  pybind11_add_module(flash_rt_fp4 csrc/fp4_bindings.cpp)
  set_target_properties(flash_rt_fp4 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  target_sources(flash_rt_fp4 PRIVATE $<TARGET_OBJECTS:fp4_kernels_obj>)
  target_include_directories(flash_rt_fp4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  )
  if(FLASHRT_ENABLE_COSMOS3_EDGE)
    target_compile_definitions(flash_rt_fp4 PRIVATE FLASHRT_HAVE_COSMOS3_EDGE=1)
  endif()
  target_link_libraries(flash_rt_fp4 PRIVATE CUDA::cudart)
  install(TARGETS flash_rt_fp4 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
endif()


# ──────────────────────────────────────────────────────────────────────
#  LingBot-VLA model-specific kernels (fused AdaRMSNorm / SwiGLU tail / QKV+RoPE).
#  Compiled INTO flash_rt_kernels (NOT a separate .so) — same pattern as the
#  qwen36 model kernels in csrc/kernels/. lingbot_-prefixed symbols + bindings
#  (csrc/bindings.cpp, #ifdef ENABLE_LINGBOT). Thor-only: gated on
#  FLASHRT_ENABLE_LINGBOT AND GPU_ARCH==110 so RTX (sm_120) / Orin / L40 builds
#  neither compile the lingbot_* sources nor define ENABLE_LINGBOT (the bindings
#  are #ifdef'd the same way). Disable with -DFLASHRT_ENABLE_LINGBOT=OFF.
#  Inherits flash_rt_kernels' CUDA flags (--ftz/--prec-div=false/--prec-sqrt=false),
#  numerically equivalent to these kernels' original --use_fast_math build.
# ──────────────────────────────────────────────────────────────────────
if(FLASHRT_ENABLE_LINGBOT AND GPU_ARCH STREQUAL "110")
  target_sources(flash_rt_kernels PRIVATE
    csrc/kernels/lingbot_norm_fp8.cu
    csrc/kernels/lingbot_rope_qkv.cu
    csrc/kernels/lingbot_silu_mul_fp8.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_LINGBOT=1)
  message(STATUS "LingBot-VLA kernels: compiled into flash_rt_kernels (sm_${GPU_ARCH})")
else()
  message(STATUS "LingBot-VLA kernels: DISABLED (Thor sm_110 only; current sm_${GPU_ARCH})")
endif()


# ──────────────────────────────────────────────────────────────────────
#  JAX FFI add-on (separate .so, additive, does NOT touch flash_rt_kernels)
#
#  Built only when JAX is importable from the configure-time Python and
#  exposes `xla/ffi/api/ffi.h` via `jax.ffi.include_dir()`. The
#  resulting `flash_rt_jax_ffi.so` is consumed by `training/jax/` via
#  ctypes + jax.ffi.register_ffi_target — independent of the PyTorch
#  pybind path, so JAX-only deps never bleed into flash_rt_kernels.
#
#  The handlers delegate to GemmRunner::fp8_nn_dev /
#  quantize_fp8_static (the same kernels the PyTorch path drives), so
#  gemm_runner.cu and quantize.cu are compiled into this .so as well —
#  one-time build cost, no runtime overhead.
# ──────────────────────────────────────────────────────────────────────
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import jax.ffi, sys; sys.stdout.write(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE JAX_FFI_DETECT_RESULT
)
if(JAX_FFI_DETECT_RESULT EQUAL 0
   AND EXISTS "${JAX_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
  message(STATUS "JAX FFI: ENABLED (include=${JAX_FFI_INCLUDE_DIR})")

  add_library(flash_rt_jax_ffi SHARED
    csrc/training/jax_ffi/fp8_gemm_ffi.cu
    csrc/training/jax_ffi/activation_quantize_ffi.cu
    csrc/gemm/gemm_runner.cu
    csrc/kernels/quantize.cu
  )
  set_target_properties(flash_rt_jax_ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""                             # → flash_rt_jax_ffi.so (no "lib" prefix)
    OUTPUT_NAME "flash_rt_jax_ffi"
  )
  target_include_directories(flash_rt_jax_ffi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
    ${JAX_FFI_INCLUDE_DIR}
    ${CUTLASS_INCLUDE}
    ${CUTLASS_DIR}/tools/util/include
  )
  target_link_libraries(flash_rt_jax_ffi PRIVATE
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cudart
  )
  target_compile_options(flash_rt_jax_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      --ftz=true --prec-div=false --prec-sqrt=false
      ${GPU_GENCODE}
    >
  )
  install(TARGETS flash_rt_jax_ffi DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
else()
  message(STATUS "JAX FFI: SKIPPED (jax.ffi not importable or ffi.h missing)")
endif()
