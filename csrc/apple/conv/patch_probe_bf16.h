#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string patch_probe_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Rows { uint rows,dim; };
// Diagnostic-only NCHW layout used by the shared torch patch probe.
kernel void tr_patch_probe(device const bfloat* x [[buffer(0)]],device bfloat* y [[buffer(1)]],
    constant Rows& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p.rows*p.dim){uint col=i%p.dim,row=i/p.dim;y[(row/256*p.dim+col)*256+row%256]=x[i];}
}
)MSL";
}
}  // namespace flashrt::apple::shaders
