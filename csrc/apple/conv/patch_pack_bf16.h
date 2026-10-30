#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string patch_pack_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
// Nonoverlapping 14x14 patches, NHWC images -> [views*256, 3*14*14].
kernel void tr_patch_pack(device const bfloat* x [[buffer(0)]],device bfloat* y [[buffer(1)]],
    constant uint& views [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=views*256*588)return;
    uint k=i%588, row=i/588, view=row/256, patch=row%256;
    uint c=k/196, ky=(k%196)/14, kx=k%14;
    uint py=(patch/16)*14+ky, px=(patch%16)*14+kx;
    y[i]=x[((view*224+py)*224+px)*3+c];
}
)MSL";
}
}  // namespace flashrt::apple::shaders
