#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

// ================================================================
// FvkContext (AMD): per-instance runtime resources.
//
// Owns a single hipBLASLt handle shared by ALL kernel calls.
// Created by Python (via pybind11), passed to every kernel.
// Eliminates static handles — kernels are fully stateless.
//
// Mirrors csrc/context.h (CUDA), which owns a cublasHandle_t.
// The AMD GEMM path is hipBLASLt-only, so the handle here is
// hipblasLtHandle_t rather than a plain hipblasHandle_t.
// ================================================================

struct FvkContext {
    hipblasLtHandle_t hipblaslt_handle;

    FvkContext() : hipblaslt_handle(nullptr) {
        hipblasLtCreate(&hipblaslt_handle);
    }

    ~FvkContext() {
        if (hipblaslt_handle) {
            hipblasLtDestroy(hipblaslt_handle);
            hipblaslt_handle = nullptr;
        }
    }

    // Non-copyable (handle is unique resource)
    FvkContext(const FvkContext&) = delete;
    FvkContext& operator=(const FvkContext&) = delete;

    // Movable
    FvkContext(FvkContext&& other) noexcept : hipblaslt_handle(other.hipblaslt_handle) {
        other.hipblaslt_handle = nullptr;
    }
};
