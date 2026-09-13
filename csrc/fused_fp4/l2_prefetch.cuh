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

struct L2PrefetchRegions {
  const void* ptr[8];
  unsigned long long bytes[8];
  int count;
};

// mode 0: cp.async.bulk.prefetch.L2 (fire-and-forget); mode 1: real ld.global.cg touch (sink: 4-byte scratch).
int l2_prefetch_regions(const L2PrefetchRegions& regions, cudaStream_t stream, int mode = 0, void* sink = nullptr);

}  // namespace fp4
}  // namespace flash_rt
