// ============================================================================
//  FlashRT — fire-and-forget L2 prefetch of weight regions (sm_90+).
//
//  One launch issues cp.async.bulk.prefetch.L2 over up to 8 regions and
//  returns without waiting; the memory system streams the lines into L2
//  while the following kernels on the stream execute. Used to hide the
//  decoder's weight DRAM traffic behind the previous layer's compute.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace fp4 {

constexpr int kL2TouchMaxRegions = 12;   // pi05_l2_prefetch_regions itself still takes at most 8

struct L2PrefetchRegions {
  const void* ptr[kL2TouchMaxRegions];
  unsigned long long bytes[kL2TouchMaxRegions];
  int count;
};

// mode 0: cp.async.bulk.prefetch.L2 (fire-and-forget); mode 1: real ld.global.cg touch (sink: 4-byte scratch).
int pi05_l2_prefetch_regions(const L2PrefetchRegions& regions, cudaStream_t stream, int mode = 0, void* sink = nullptr);

// Real-load L2 touch with a chosen CTA count (256 threads each, 16 loads in flight per thread) over the
// regions in order; hint 1 tags the lines L2::evict_last. `pi05_l2_touch_fork` runs it on a private
// non-blocking side stream forked from `main_stream` with an event (capture-safe, so inside a CUDA
// graph it becomes a parallel branch), `pi05_l2_touch_join` makes `main_stream` wait for the side stream.
// `pi05_l2_touch_init` creates the side stream and the events; call it once before any graph capture.
int pi05_l2_touch_init();
int pi05_l2_touch_regions_ex(const L2PrefetchRegions& regions, cudaStream_t stream, int nctas, int hint, void* sink,
                        int depth = 16, unsigned pace_ns = 0, int nthreads = 256);
int pi05_l2_touch_fork(const L2PrefetchRegions& regions, cudaStream_t main_stream, int nctas, int hint, void* sink,
                  int depth = 16, unsigned pace_ns = 0, int nthreads = 256);
int pi05_l2_touch_join(cudaStream_t main_stream);

}  // namespace fp4
}  // namespace flash_rt
