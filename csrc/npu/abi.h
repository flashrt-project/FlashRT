// Identity a shared object carries so the loader can refuse the wrong one.
//
// The Ascend backend ships four shared objects — the vector dispatch unit, the
// gate/up cube unit, the decoder GEMM and the decode attention — because the
// kernel headers define a per-translation-unit tiling symbol and a cube-only
// unit cannot share a file scope with a mixed one. Each is loaded separately
// and each honours its own environment override, so nothing otherwise stops a
// process from mixing a freshly built library with a stale one, or with one
// compiled for a different part. Both are silent: a stale library has the same
// symbol names and the wrong contract behind them.
//
// Every unit therefore exports the same pair, and every loader checks it.
// Include this file exactly once per shared object.
#ifndef FLASHRT_NPU_SOC_VERSION
#error "build the Ascend kernels with scripts/npu/build.sh; FLASHRT_NPU_SOC_VERSION is unset"
#endif
#ifndef FLASHRT_NPU_ABI_VERSION
#error "build the Ascend kernels with scripts/npu/build.sh; FLASHRT_NPU_ABI_VERSION is unset"
#endif

extern "C" const char* flashrt_npu_soc_version() {
    return FLASHRT_NPU_SOC_VERSION;
}

extern "C" int flashrt_npu_abi_version() {
    return FLASHRT_NPU_ABI_VERSION;
}
