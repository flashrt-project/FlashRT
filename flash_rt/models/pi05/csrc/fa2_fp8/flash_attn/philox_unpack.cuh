// Vendor patch: replace torch's ATen/cuda/detail/UnpackRaw.cuh with a
// self-contained inline stub for at::cuda::philox::unpack(). Upstream
// FA2 only needs this to read {seed, offset} out of at::PhiloxCudaState
// during dropout sampling; in inference we run with p_dropout=0 so the
// state is inert, but the kernel still calls unpack() as part of
// boilerplate codegen, so we preserve the same signature + semantics.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <utility>
#include "flash.h"  // brings our POD at::PhiloxCudaState stub into scope

namespace at { namespace cuda { namespace philox {

// Same signature as upstream ATen at::cuda::philox::unpack().
// Returns {seed, offset} either read directly from the POD fields
// (non-captured case) or dereferenced from device pointers (CUDA graph
// capture case). Inference never takes the captured branch.
inline __device__ std::pair<uint64_t, uint64_t>
unpack(const at::PhiloxCudaState& st) {
    if (st.captured_) {
        return {*(st.seed_ptr_), *(st.offset_ptr_)};
    }
    return {st.seed_, st.offset_};
}

}}}  // namespace at::cuda::philox
