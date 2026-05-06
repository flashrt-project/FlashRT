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

# SM120+ (Blackwell): NVFP4/W4A8 block-scaled GEMM
if(GPU_ARCH STREQUAL "120" OR GPU_ARCH STREQUAL "120a")
  set(ENABLE_NVFP4 ON)
  message(STATUS "NVFP4/W4A8 support: ENABLED (sm_120a)")
else()
  set(ENABLE_NVFP4 OFF)
  message(STATUS "NVFP4/W4A8 support: DISABLED (requires sm_120a, current: sm_${GPU_ARCH})")
endif()

# Only sm_90/100/110/120 ship an 'a' (architecture-accelerated) variant.
# Ada (sm_89) and Ampere (sm_80/86) do not, so emitting -gencode
# arch=compute_89a,code=sm_89a breaks nvcc on 4090. Pick the right
# gencode flag once here and reuse it for every target below.
if(GPU_ARCH STREQUAL "90"  OR GPU_ARCH STREQUAL "100" OR
   GPU_ARCH STREQUAL "110" OR GPU_ARCH STREQUAL "120")
  set(GPU_GENCODE "-gencode=arch=compute_${GPU_ARCH}a,code=sm_${GPU_ARCH}a")
else()
  set(GPU_GENCODE "-gencode=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}")
endif()
message(STATUS "Using gencode flag: ${GPU_GENCODE}")

# Flash-Attention 2 vendor (csrc/attention/flash_attn_2_src/): ships
# SM80-family fp16 fwd kernels. SM89 (4090) runs sm_80 SASS natively
# via Ampere ISA compatibility; SM120 (5090) uses PTX JIT from
# compute_80. Thor SM110 has its own attention path (fvk.attention_qkv_fp16
# cuBLAS decomposed), hand-tuned for unified LPDDR memory — building
# FA2 there would add ~10 MB to the .so and unused compile time.
if(GPU_ARCH STREQUAL "80"  OR GPU_ARCH STREQUAL "86" OR
   GPU_ARCH STREQUAL "89"  OR GPU_ARCH STREQUAL "120")
  set(ENABLE_FA2 ON)
  message(STATUS "FA2 in-SO attention: ENABLED (sm_${GPU_ARCH})")
else()
  set(ENABLE_FA2 OFF)
  message(STATUS "FA2 in-SO attention: DISABLED (Thor SM110 uses fvk.attention_qkv_fp16 cuBLAS path)")
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
set(FA2_HDIMS "96;128;256" CACHE STRING
    "Semicolon-separated FA2 head_dim instantiations to build. Shipped models use 96 and 256.")
set(FA2_DTYPES "fp16;bf16" CACHE STRING
    "Semicolon-separated FA2 dtype instantiations to build. Pi0 uses fp16; pi0.5/groot use bf16.")

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
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
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

# ── CUTLASS SM120a NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt) ──
# Block-scaled FP4 e2m1 weights × FP4 e2m1 act (post quant) × FP8 ue4m3
# group scales × BF16 output. Ported from NVIDIA unit test
# sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu — sm_120a / sm_121a.
if(GPU_ARCH STREQUAL "120")
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
      -gencode arch=compute_120a,code=sm_120a
    >
  )
endif()

# ── Flash-Attention 2 vendored kernels (fp16 + bf16 fwd SM80) ──
# Built as an object library and linked into a SEPARATE pybind module
# flash_rt_fa2.so — same isolation pattern as flash_rt_fp4.so. The
# main flash_rt_kernels.so stays small (~3.6 MB) and its rebuild no
# longer waits on FA2's heavy CUTLASS 3.x template codegen (~8 min).
#
# The hdim .cu files are template instantiations for the FA2 forward,
# six per elem_type (three hdim × non-split + splitkv × fp16/bf16).
# fa2_wrapper.cu provides the raw-pointer C entries
# (fvk_attention_fa2_fwd_{fp16,bf16}); fa2_bindings.cpp pybind11-wraps
# them into the flash_rt_fa2 module.
if(ENABLE_FA2)
  # Dynamic source list: {hdim} x {dtype} x {split, no-split} .cu files,
  # driven by FA2_HDIMS / FA2_DTYPES cache vars. Default is the full
  # 12-file matrix; users can restrict either axis to slim builds.
  set(FA2_SRC_ROOT csrc/attention/flash_attn_2_src/flash_attn)
  set(FA2_SRCS "")
  set(FA2_BUILT_HDIMS "")
  set(FA2_BUILT_DTYPES "")
  foreach(_hd IN LISTS FA2_HDIMS)
    if(NOT _hd MATCHES "^(96|128|256)$")
      message(FATAL_ERROR "FA2_HDIMS entry '${_hd}' not supported. Valid: 96, 128, 256.")
    endif()
    list(APPEND FA2_BUILT_HDIMS ${_hd})
    foreach(_dt IN LISTS FA2_DTYPES)
      if(NOT _dt MATCHES "^(fp16|bf16)$")
        message(FATAL_ERROR "FA2_DTYPES entry '${_dt}' not supported. Valid: fp16, bf16.")
      endif()
      list(APPEND FA2_SRCS
        ${FA2_SRC_ROOT}/flash_fwd_hdim${_hd}_${_dt}_sm80.cu
        ${FA2_SRC_ROOT}/flash_fwd_split_hdim${_hd}_${_dt}_sm80.cu
      )
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

  # P1-a (Qwen3-8B prefill): causal FA2 forward — built only for
  # hdim=128 + bf16 to keep the public .so size in check.
  if("128" IN_LIST FA2_HDIMS AND "bf16" IN_LIST FA2_DTYPES)
    list(APPEND FA2_SRCS
      csrc/attention/fa2_causal_inst/flash_fwd_hdim128_bf16_sm80_causal.cu
      csrc/attention/fa2_causal_inst/flash_fwd_split_hdim128_bf16_sm80_causal.cu
      csrc/attention/fa2_wrapper_causal.cu
    )
  endif()

  add_library(fa2_vendor_obj OBJECT ${FA2_SRCS})
  set_target_properties(fa2_vendor_obj PROPERTIES
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
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
  # SASS for 5090; compute_120 PTX fallback for future consumer archs).
  # FA2_ARCH_NATIVE_ONLY=ON skips the non-native AOT passes, saving
  # ~66% compile time on single-card developer boxes.
  # ──────────────────────────────────────────────────────────────
  if(FA2_ARCH_NATIVE_ONLY)
    # Just emit native SASS for the detected GPU. Use the real arch
    # (compute_80 for SM80/86/89 since they share Ampere ISA; direct
    # compute_120 for SM120). Matches upstream FA2 setup.py behaviour.
    if(GPU_ARCH STREQUAL "80" OR GPU_ARCH STREQUAL "86" OR GPU_ARCH STREQUAL "89")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_80,code=sm_${GPU_ARCH}"
        >
      )
      message(STATUS "FA2 vendor arch: sm_${GPU_ARCH} AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    elseif(GPU_ARCH STREQUAL "120")
      target_compile_options(fa2_vendor_obj PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          "SHELL:-gencode arch=compute_120,code=sm_120"
        >
      )
      message(STATUS "FA2 vendor arch: sm_120 AOT only (FA2_ARCH_NATIVE_ONLY=ON)")
    endif()
  else()
    # Default multi-arch AOT (original behaviour; distributable across
    # the whole RTX / Ampere-through-consumer-Blackwell family from one
    # build). See commit history for the reason each gencode matters —
    # routing 5090 through compute_80 PTX produced a subtle sm_120 SASS
    # drift that accumulated to cos 0.98 under Pi0 FP8.
    target_compile_options(fa2_vendor_obj PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        "SHELL:-gencode arch=compute_80,code=sm_80"
        "SHELL:-gencode arch=compute_120,code=sm_120"
        "SHELL:-gencode arch=compute_120,code=compute_120"
      >
    )
    message(STATUS "FA2 vendor arch: sm_80 + sm_120 AOT + compute_120 PTX fallback (default)")
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

  string(REPLACE ";" "," _FA2_H "${FA2_BUILT_HDIMS}")
  string(REPLACE ";" "," _FA2_D "${FA2_BUILT_DTYPES}")
  list(LENGTH FA2_SRCS _FA2_NSRC)
  math(EXPR _FA2_NKERN "${_FA2_NSRC} - 1")
  message(STATUS "FA2 vendor instantiations: hdim={${_FA2_H}} x dtype={${_FA2_D}} x {split,no-split} = ${_FA2_NKERN} kernel .cu files")

  # Independent pybind module. Caller side:
  #     from flash_rt import flash_rt_fa2 as fa2
  #     fa2.fwd_fp16(...)  /  fa2.fwd_bf16(...)
  pybind11_add_module(flash_rt_fa2 csrc/fa2_bindings.cpp)
  set_target_properties(flash_rt_fa2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  target_sources(flash_rt_fa2 PRIVATE $<TARGET_OBJECTS:fa2_vendor_obj>)
  target_include_directories(flash_rt_fa2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  )
  target_link_libraries(flash_rt_fa2 PRIVATE CUDA::cudart)
  install(TARGETS flash_rt_fa2 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
  message(STATUS "FA2 pybind module: flash_rt_fa2 (separate .so)")
endif()

# ── Main module ──
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
  csrc/quantize/fp8_block128_to_nvfp4_swizzled.cu
  csrc/quantize/bf16_weight_to_nvfp4_swizzled.cu
  csrc/quantize/fp8_per_token_block_quant.cu
  csrc/quantize/tq_dequant_kv.cu
  csrc/quantize/wmma_probe.cu
  csrc/quantize/tq_dequant_cutlass.cu
  csrc/gemm/fp8_block128_gemm.cu
  csrc/kernels/causal_conv1d_qwen36.cu
  csrc/kernels/gated_deltanet_qwen36.cu
  csrc/kernels/rms_norm_gated_silu_qwen36.cu
  csrc/kernels/silu_mul_qwen36.cu
  csrc/kernels/silu_mul_to_nvfp4_swizzled.cu
  csrc/kernels/qwen3_qkv_post_proc.cu
  csrc/kernels/fp4_w4a4_matvec_sm120.cu
  csrc/kernels/fp4_w4a4_mma_sm120.cu
  csrc/kernels/bf16_matvec_qwen36.cu
  csrc/kernels/bf16_matmul_qwen36.cu
  csrc/attention/fmha_dispatch.cu
  csrc/bindings.cpp
)
set_target_properties(flash_rt_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)

if(ENABLE_SM100_CUTLASS)
  target_sources(flash_rt_kernels PRIVATE $<TARGET_OBJECTS:cutlass_sm100_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_SM100_CUTLASS=1)
endif()

if(ENABLE_NVFP4)
  target_sources(flash_rt_kernels PRIVATE $<TARGET_OBJECTS:gemm_kernels_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE ENABLE_NVFP4=1)
endif()

# SM120a CUTLASS block-128 FP8 GEMM (Path B for Qwen3.6).
if(GPU_ARCH STREQUAL "120")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_sm120_block_obj>)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_CUTLASS_SM120_BLOCK_FP8=1)
endif()

# SM120a CUTLASS NVFP4 W4A16 GEMM (Qwen3.6 NVFP4 ckpt path).
if(GPU_ARCH STREQUAL "120")
  target_sources(flash_rt_kernels PRIVATE
    $<TARGET_OBJECTS:cutlass_nvfp4_w4a16_sm120_obj>
    csrc/quantize/nvfp4_sf_reshape_sm120.cu)
  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_CUTLASS_SM120_NVFP4_W4A16=1)
endif()

# (ENABLE_FA2 object-lib is linked into flash_rt_fa2.so only — see
# the dedicated pybind11_add_module(flash_rt_fa2 ...) above. The main
# flash_rt_kernels.so deliberately does NOT pull FA2 in, so rebuilds
# of our hand-written kernels don't re-trigger the FA2 codegen tax.)

target_include_directories(flash_rt_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention
  ${CUTLASS_INCLUDE}
  ${CUTLASS_DIR}/tools/util/include
)

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
    csrc/fused_fp16/rms_norm_noweight_fp16.cu
    csrc/fused_fp4/norm_silu_fp4_sfa.cu
    csrc/fused_fp4/silu_mul_fp4_sfa_v2.cu
    csrc/fused_fp4/res_rms_fp4_sfa_v2.cu
    csrc/fused_fp4/per_channel_mul_fp16.cu
    csrc/fused_fp4/res_rms_mul_fp4_sfa.cu
    csrc/fused_fp4/silu_mul_mul_fp4_sfa_v2.cu
    csrc/fused_fp4/silu_mul_two_fp4_to_fp4.cu
    csrc/gemm/fp4/cutlass_fp4_gemm_fp4out.cu
  )
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
  target_link_libraries(flash_rt_fp4 PRIVATE CUDA::cudart)
  install(TARGETS flash_rt_fp4 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt)
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
