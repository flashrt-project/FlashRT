/* FlashRT exec — CUDA backend.
 *
 * Implements backend/backend.h against the CUDA Runtime API. Uses only the
 * runtime (no kernels), so this is a plain .cpp linked against cudart. Mirrors
 * the stream-level RELAXED capture already used by flash_rt/core/cuda_graph.py.
 */
#include "backend.h"

#include <cuda_runtime.h>

namespace frt {
namespace be {

namespace {
inline cudaStream_t S(void* s) { return static_cast<cudaStream_t>(s); }
inline cudaEvent_t  E(void* e) { return static_cast<cudaEvent_t>(e); }
}  // namespace

void* malloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

void free(void* dptr) {
    if (dptr) cudaFree(dptr);
}

void* stream_create(int priority) {
    cudaStream_t s = nullptr;
    // Clamp requested priority into the device's supported range.
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);  // hi = highest (most negative)
    int p = priority;
    if (p < hi) p = hi;
    if (p > lo) p = lo;
    if (cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, p) != cudaSuccess)
        return nullptr;
    return s;
}

void stream_destroy(void* stream) {
    if (stream) cudaStreamDestroy(S(stream));
}

void stream_sync(void* stream) {
    cudaStreamSynchronize(S(stream));
}

void* event_create() {
    cudaEvent_t e = nullptr;
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess)
        return nullptr;
    return e;
}

void event_destroy(void* event) {
    if (event) cudaEventDestroy(E(event));
}

bool event_record(void* event, void* stream) {
    return cudaEventRecord(E(event), S(stream)) == cudaSuccess;
}

bool stream_wait_event(void* stream, void* event) {
    return cudaStreamWaitEvent(S(stream), E(event), 0) == cudaSuccess;
}

bool memcpy_dtod_async(void* dst, const void* src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, S(stream))
           == cudaSuccess;
}

bool memset_async(void* dptr, int value, size_t bytes, void* stream) {
    return cudaMemsetAsync(dptr, value, bytes, S(stream)) == cudaSuccess;
}

bool capture_begin(void* stream) {
    // Relaxed: capture only ops on THIS stream, leaving other streams (e.g.
    // torch background work) free — same choice as core/cuda_graph.py.
    return cudaStreamBeginCapture(S(stream), cudaStreamCaptureModeRelaxed)
           == cudaSuccess;
}

void* capture_end(void* stream) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(S(stream), &graph) != cudaSuccess) return nullptr;
    cudaGraphExec_t exec = nullptr;
    // CUDA 12 flags form (matches core/cuda_graph.py's ctypes call).
    cudaError_t st = cudaGraphInstantiate(&exec, graph, 0ull);
    cudaGraphDestroy(graph);  // keep only the executable
    if (st != cudaSuccess) return nullptr;
    return exec;
}

void graph_exec_destroy(void* graph_exec) {
    if (graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
}

bool graph_launch(void* graph_exec, void* stream) {
    return cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec), S(stream))
           == cudaSuccess;
}

}  // namespace be
}  // namespace frt
