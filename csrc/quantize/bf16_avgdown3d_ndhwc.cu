// SPDX-License-Identifier: Apache-2.0
//
// The Wan VAE encoder's AvgDown3D shortcut on a channels-last activation.
//
// The reference (diffusers AvgDown3D.forward) views x [B, C, T, H, W] as
// [B, C, T/ft, ft, H/fs, fs, W/fs, fs], permutes the three factor axes next to
// the channel, materialises that copy (.contiguous()), regroups the C * ft *
// fs * fs channels into `out` groups of G and averages each group (ATen mean,
// fp32 accumulation, times 1/G, one bf16 rounding). Output channel o averages
// the G source elements whose permuted channel index c' = o * G + g decomposes
// as c' = ((c * ft + it) * fs + ih) * fs + iw. This computes each output from
// its G sources directly -- no permuted copy, no separate reduction pass -- and
// writes channels-last. `order` selects the fp32 summation order (0: sequential
// in g, 1: pairwise (g0 + g1) + (g2 + g3)), so the one ATen uses can be pinned
// by bit comparison on real activations; with G a power of two the scaling is
// exact either way.

#include "bf16_avgdown3d_ndhwc.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {
namespace {

__global__ void avgdown3d_ndhwc_kernel(const __nv_bfloat16* __restrict__ x,
                                       __nv_bfloat16* __restrict__ y,
                                       long long n_out, int C, int T, int H, int W,
                                       int ft, int fs, int out_ch, int G, float inv_g,
                                       int order) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    const int To = T / ft, Ho = H / fs, Wo = W / fs;
    // y is channels-last: [B, To, Ho, Wo, out]
    long long r = i;
    const int o = (int)(r % out_ch); r /= out_ch;
    const int wo = (int)(r % Wo); r /= Wo;
    const int ho = (int)(r % Ho); r /= Ho;
    const int to = (int)(r % To);
    const long long b = r / To;
    float v[16];
    for (int g = 0; g < G; ++g) {
        int cp = o * G + g;
        const int iw = cp % fs; cp /= fs;
        const int ih = cp % fs; cp /= fs;
        const int it = cp % ft;
        const int c = cp / ft;
        const long long src =
            ((((b * T + (to * ft + it)) * H + (ho * fs + ih)) * W + (wo * fs + iw)) * C + c);
        v[g] = __bfloat162float(x[src]);
    }
    float s;
    if (order == 1 && G == 4) {
        s = __fadd_rn(__fadd_rn(v[0], v[1]), __fadd_rn(v[2], v[3]));
    } else {
        s = v[0];
        for (int g = 1; g < G; ++g) s = __fadd_rn(s, v[g]);
    }
    y[i] = __float2bfloat16_rn(__fmul_rn(s, inv_g));
}

}  // namespace

int bf16_avgdown3d_ndhwc(const void* x, void* y, int B, int C, int T, int H, int W,
                         int ft, int fs, int out_ch, int order, cudaStream_t stream) {
    if (!x || !y || B <= 0 || C <= 0 || ft <= 0 || fs <= 0 || out_ch <= 0) return -1;
    if ((T % ft) || (H % fs) || (W % fs)) return -2;
    const int factor = ft * fs * fs;
    if ((C * factor) % out_ch) return -3;
    const int G = C * factor / out_ch;
    if (G > 16 || (G & (G - 1))) return -4;   // power of two: 1/G is exact
    const long long n_out = (long long)B * (T / ft) * (H / fs) * (W / fs) * out_ch;
    const int block = 256;
    const long long grid = (n_out + block - 1) / block;
    avgdown3d_ndhwc_kernel<<<(unsigned)grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
        n_out, C, T, H, W, ft, fs, out_ch, G, 1.0f / (float)G, order);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace quantize
}  // namespace flash_rt
