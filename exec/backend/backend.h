/* FlashRT exec — internal hardware backend interface.
 *
 * The whole hardware surface the execution contract needs, as a small set of
 * free functions. CUDA-free here (all handles are void*) so the core C ABI in
 * src/ never includes a vendor runtime header. One backend translation unit
 * (backend/cuda/cuda_backend.cpp today; backend/hip/... later) implements it.
 *
 * All functions return null / false on failure; the core layer turns that into
 * the public frt_status codes. No exceptions cross this boundary.
 */
#ifndef FLASHRT_EXEC_BACKEND_H
#define FLASHRT_EXEC_BACKEND_H

#include <stddef.h>

namespace frt {
namespace be {

/* --- device memory --- */
void* malloc(size_t bytes);          /* device alloc; null on failure        */
void  free(void* dptr);

/* --- streams (handles owned by ctx) --- */
void* stream_create(int priority);   /* prioritized stream; null on failure   */
void  stream_destroy(void* stream);
void  stream_sync(void* stream);

/* --- events --- */
void* event_create();
void  event_destroy(void* event);
bool  event_record(void* event, void* stream);
bool  stream_wait_event(void* stream, void* event);

/* --- async copies / fills (allocation-free; capture-safe) --- */
bool  memcpy_dtod_async(void* dst, const void* src, size_t bytes, void* stream);
bool  memset_async(void* dptr, int value, size_t bytes, void* stream);

/* --- CUDA-graph-style capture/replay ---
 * capture_begin/end wrap stream-level capture (RELAXED mode); capture_end
 * instantiates and returns an executable graph handle, freeing the transient
 * non-executable graph. graph_exec_destroy frees the executable. */
bool  capture_begin(void* stream);
void* capture_end(void* stream);     /* returns graph-exec handle; null = fail */
void  graph_exec_destroy(void* graph_exec);
bool  graph_launch(void* graph_exec, void* stream);

}  // namespace be
}  // namespace frt

#endif  /* FLASHRT_EXEC_BACKEND_H */
