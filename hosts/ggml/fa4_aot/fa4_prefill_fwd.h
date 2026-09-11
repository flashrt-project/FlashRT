
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} fa4_prefill_fwd_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_fa4_prefill_fwd_cuda_init(void **);
void _mlir_fa4_prefill_fwd_cuda_load_to_device(void **);
static inline void fa4_prefill_fwd_Kernel_Module_Load(fa4_prefill_fwd_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_fa4_prefill_fwd_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_fa4_prefill_fwd_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void fa4_prefill_fwd_Kernel_Module_Unload(fa4_prefill_fwd_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fa4_prefill_fwd_Tensor_mQ_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fa4_prefill_fwd_Tensor_mK_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fa4_prefill_fwd_Tensor_mV_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fa4_prefill_fwd_Tensor_mO_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_fa4_prefill_fwd__mlir_ciface_cutlass___call___flashrt_fa4cutesm100_hd256_2cta_fmha_forwardBlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Te(void **args, int32_t num_args);

static inline int32_t cute_dsl_fa4_prefill_fwd_wrapper(fa4_prefill_fwd_Kernel_Module_t *module, fa4_prefill_fwd_Tensor_mQ_t *mQ, fa4_prefill_fwd_Tensor_mK_t *mK, fa4_prefill_fwd_Tensor_mV_t *mV, fa4_prefill_fwd_Tensor_mO_t *mO, float softmax_scale, cudaStream_t stream) {
    int32_t ret;
    void *args[7] = {
        mQ, mK, mV, mO, &softmax_scale, &stream,
        &ret
    };
    _mlir_fa4_prefill_fwd__mlir_ciface_cutlass___call___flashrt_fa4cutesm100_hd256_2cta_fmha_forwardBlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Te(args, 7);
    return ret;
}
