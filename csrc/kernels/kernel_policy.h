// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace flash_rt {
namespace kernels {

// Process-wide opt-in for the producer fast paths and programmatic dependent
// launch. The default is the pre-existing behavior; a model frontend that has
// validated the faster paths calls set_kernel_fast_paths(true, true) before
// its first launch. The FLASHRT_* environment variables remain explicit
// diagnostics/overrides and take precedence over this flag.
inline bool& fast_producers_flag() {
  static bool enabled = false;
  return enabled;
}

inline bool& pdl_flag() {
  static bool enabled = false;
  return enabled;
}

inline void set_kernel_fast_paths(bool fast_producers, bool pdl) {
  fast_producers_flag() = fast_producers;
  pdl_flag() = pdl;
}

inline bool fast_producers() { return fast_producers_flag(); }
inline bool pdl_default() { return pdl_flag(); }

}  // namespace kernels
}  // namespace flash_rt
