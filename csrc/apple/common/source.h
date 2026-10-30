#pragma once
#include <string>

namespace flashrt::apple::shaders {
// Source fragments are embedded so an installed dylib has no shader-file or
// offline compiler dependency. Only TensorOps kernels include the MPP headers.
inline constexpr const char* metal_preamble = R"MSL(
#include <metal_stdlib>
using namespace metal;
)MSL";
inline constexpr const char* tensor_ops_preamble = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
)MSL";
}  // namespace flashrt::apple::shaders
