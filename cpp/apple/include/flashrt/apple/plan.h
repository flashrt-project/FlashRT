#pragma once
#include <stddef.h>
#include <stdint.h>
#define FRT_APPLE_PLAN_API __attribute__((visibility("default")))
#ifdef __cplusplus
extern "C" {
#endif
// Generic fixed-shape plan builder. Models supply dispatches, never native layer loops.
FRT_APPLE_PLAN_API void* frt_apple_plan_create(void);
FRT_APPLE_PLAN_API void frt_apple_plan_destroy(void*);
FRT_APPLE_PLAN_API const char* frt_apple_plan_error(void);
FRT_APPLE_PLAN_API int64_t frt_apple_plan_buffer(void*, size_t bytes);
FRT_APPLE_PLAN_API int frt_apple_plan_write(void*, uint32_t buffer, size_t offset, const void*, size_t bytes);
FRT_APPLE_PLAN_API int frt_apple_plan_read(void*, uint32_t buffer, size_t offset, void*, size_t bytes);
// Borrowed shared-memory mapping. Caller serializes access with run/destroy.
FRT_APPLE_PLAN_API void* frt_apple_plan_data(void*, uint32_t buffer, size_t offset, size_t bytes);
FRT_APPLE_PLAN_API int frt_apple_plan_copy(void*, uint32_t source, size_t source_offset,
    uint32_t destination, size_t destination_offset, size_t bytes);
FRT_APPLE_PLAN_API int frt_apple_plan_dispatch(void*, const char* kernel, const uint32_t* buffers,
    const size_t* offsets, const uint32_t* access, size_t count,
    const size_t* groups, size_t threads, const char* label);
// Register a fixed-shape Core ML model (single FP16 input/output, both views of plan buffers);
// returns its external task id. compute_units: 0 CPU+Neural Engine, 1 all, 2 CPU+GPU, 3 CPU.
FRT_APPLE_PLAN_API int64_t frt_apple_plan_coreml(void*, const char* model, const char* input,
    uint32_t input_buffer, size_t input_offset, const char* output, uint32_t output_buffer,
    size_t output_offset, const int64_t* shape, size_t rank, uint32_t compute_units);
// Record an external task boundary in the current program: wait=0 signals that operations so far
// completed and the host may run the task; wait=1 blocks later operations until it finished.
// Each task is signalled once and then waited once per program; runs execute tasks in signal order.
FRT_APPLE_PLAN_API int frt_apple_plan_external(void*, uint32_t task, uint32_t wait);
// Start recording an additional program (ids 1, 2, ...); later dispatch/copy calls append to it.
// Programs share the plan's buffers. Operations recorded before the first call form program 0.
FRT_APPLE_PLAN_API int64_t frt_apple_plan_program(void*);
FRT_APPLE_PLAN_API int frt_apple_plan_finish(void*);
FRT_APPLE_PLAN_API int frt_apple_plan_residency(void*, uint32_t enabled);
typedef struct frt_apple_plan_timing {
    double encode_us, gpu_us, wall_us, commit_us, wait_us;
    double commit_to_gpu_us, gpu_to_return_us, kernel_us;
    double submit_host_s, gpu_start_s, gpu_end_s, return_host_s;
} frt_apple_plan_timing;
FRT_APPLE_PLAN_API int frt_apple_plan_run_timed(void*, uint32_t execution, frt_apple_plan_timing*);
// Run one sealed program (0 = operations recorded before frt_apple_plan_program) as one command buffer.
FRT_APPLE_PLAN_API int frt_apple_plan_run_program(void*, uint32_t program, uint32_t execution,
    frt_apple_plan_timing*);
// Diagnostic only: one synchronous command buffer per operation, in plan order.
FRT_APPLE_PLAN_API int frt_apple_plan_profile(void*, frt_apple_plan_timing*, size_t count);
// Diagnostic only: the same for program ids >= 1 (count = operations recorded into that program).
FRT_APPLE_PLAN_API int frt_apple_plan_profile_program(void*, uint32_t program, frt_apple_plan_timing*,
    size_t count);
#ifdef __cplusplus
}
#endif
#undef FRT_APPLE_PLAN_API
