#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
// Row-major BF16 activations [rows, channels] -> channel-major FP16 [channels, capacity],
// the (1, C, H, W) layout consumed by Core ML convolutions (capacity = H * W). Rows in
// [rows, capacity) are written as zero so a fixed-shape consumer never reads stale data.
// 32x32 tiles are transposed through threadgroup memory. BF16 -> FP16 is exact inside the
// FP16 normal range; larger magnitudes become +/-inf and tiny ones FP16 subnormals.
inline std::string channel_pack_f16_source() {
    return std::string(metal_preamble) + R"MSL(
struct ChannelPack { uint rows, channels, capacity; };
// Threadgroups of 256 threads; grid (ceil(capacity/32), ceil(channels/32)).
kernel void pack_channels_f16(device const bfloat* x [[buffer(0)]], device half* y [[buffer(1)]],
    constant ChannelPack& p [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[32][33];
    uint lx = tid % 32, ly = tid / 32;
    uint row0 = group.x * 32, channel0 = group.y * 32;
    for (uint i = ly; i < 32; i += 8) {
        uint row = row0 + i, channel = channel0 + lx;
        tile[i][lx] = (row < p.rows && channel < p.channels) ? half(float(x[row * p.channels + channel])) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ly; i < 32; i += 8) {
        uint channel = channel0 + i, row = row0 + lx;
        if (channel < p.channels && row < p.capacity) y[channel * p.capacity + row] = tile[lx][i];
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
