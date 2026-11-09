// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace flash_rt {
namespace kernels {

struct KernelPolicy {
  bool fast_producers = false;
  bool pdl = false;
};

// A launch policy lives only for one synchronous model call on this thread.
// CUDA graph nodes retain their own launch attributes after capture.
inline KernelPolicy& current_kernel_policy() {
  static thread_local KernelPolicy policy;
  return policy;
}

class ScopedKernelPolicy {
 public:
  explicit ScopedKernelPolicy(KernelPolicy policy)
      : previous_(current_kernel_policy()) { current_kernel_policy() = policy; }
  ~ScopedKernelPolicy() { current_kernel_policy() = previous_; }
  ScopedKernelPolicy(const ScopedKernelPolicy&) = delete;
  ScopedKernelPolicy& operator=(const ScopedKernelPolicy&) = delete;
 private:
  KernelPolicy previous_;
};

inline bool fast_producers() { return current_kernel_policy().fast_producers; }
inline bool pdl_default() { return current_kernel_policy().pdl; }

}  // namespace kernels
}  // namespace flash_rt
