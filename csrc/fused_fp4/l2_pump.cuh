// ============================================================================
//  FlashRT — persistent L2 weight pump (sm_90+).
//
//  A few resident CTAs (default one) stream a list of weight regions into L2 with
//  cp.async.bulk.prefetch.L2, unit by unit, paced by a progress counter that
//  the consuming pipeline advances (unit i is issued once *progress >= i-ahead).
//  It runs on a forked stream inside the same CUDA graph as the consumer, so
//  the DRAM keeps streaming through the consumer's launch ramps, epilogues and
//  small kernels, and the GEMMs find their weights L2-hot.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

constexpr int kL2PumpMaxChunks = 192;
constexpr int kL2PumpMaxUnits  = 20;

struct L2PumpArgs {
  const void* ptr[kL2PumpMaxChunks];
  unsigned long long bytes[kL2PumpMaxChunks];
  int unit_begin[kL2PumpMaxUnits + 1];   // unit u covers chunks [unit_begin[u], unit_begin[u+1])
  int nunits;                            // distinct units (e.g. decoder layers)
  int total_units;                       // iterations; iteration i prefetches unit (i % nunits)
  int ahead;                             // iteration i waits for *progress >= i - ahead
  const int* progress;                   // nullptr: no pacing
  unsigned chunk_bytes;                  // prefetch granule per lane (multiple of 16)
  unsigned long long spin_limit;         // max wait iterations per unit before giving up (hang guard)
};

// Launch the pump (1 CTA) on `stream`. Returns 0 or -cudaError.
int l2_pump_launch(const L2PumpArgs& args, cudaStream_t stream, int nctas = 1);
// Store `value` to *progress from a 1-thread kernel (PDL wait+trigger at entry so it is stream-ordered).
int l2_pump_progress_store(int* progress, int value, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
