// ctypes entry points over the exported FA4 AOT modules, for the AOT-vs-JIT
// parity test. Each call takes 4D [B, S, H, HD] tensors with explicit strides.
#include <cuda_runtime.h>

#include "fa4_aot/fa4_hd256_fwd.h"
#include "fa4_aot/fa4_hd256_q1_fwd.h"
#include "fa4_aot/fa4_hd72_fwd.h"

#define FILL(T, name)                                          \
    do {                                                       \
        (T).data = name;                                       \
        for (int i = 0; i < 4; ++i) (T).dynamic_shapes[i] = name##_shape[i]; \
        for (int i = 0; i < 3; ++i) (T).dynamic_strides[i] = name##_stride[i]; \
    } while (0)

#define DEFINE_RUN(M)                                                              \
    static M##_Kernel_Module_t g_##M;                                              \
    static int g_##M##_loaded;                                                     \
    int M##_run(void* q, const int* q_shape, const long long* q_stride,            \
                void* k, const int* k_shape, const long long* k_stride,            \
                void* v, const int* v_shape, const long long* v_stride,            \
                void* o, const int* o_shape, const long long* o_stride,            \
                float scale, void* stream) {                                       \
        if (!g_##M##_loaded) {                                                     \
            M##_Kernel_Module_Load(&g_##M);                                        \
            g_##M##_loaded = 1;                                                    \
        }                                                                          \
        M##_Tensor_mQ_t tq; M##_Tensor_mK_t tk; M##_Tensor_mV_t tv; M##_Tensor_mO_t to; \
        FILL(tq, q); FILL(tk, k); FILL(tv, v); FILL(to, o);                        \
        return cute_dsl_##M##_wrapper(&g_##M, &tq, &tk, &tv, &to, scale,           \
                                      (cudaStream_t)stream);                       \
    }

DEFINE_RUN(fa4_hd256_fwd)
DEFINE_RUN(fa4_hd256_q1_fwd)
DEFINE_RUN(fa4_hd72_fwd)
