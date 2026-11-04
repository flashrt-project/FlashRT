// SPDX-License-Identifier: Apache-2.0
//
// One sweep's state update for a sliding-window Picard solve of a
// flow-matching Euler sampler (bf16 latents).
//
// xs holds the n + 1 trajectory points x_0..x_n, each E elements. The sweep
// evaluated the drift v_j at points ws .. ws + r - 1; the window base s
// (ws <= s) is exact. This kernel re-runs the Euler recurrence from x_s,
//   x_{i+1} = bf16(x_i + bf16(v_{i-ws} * dsig_i)),   i = s .. ws + r - 1,
// with the same two roundings as `x + v * dsig` on bf16 tensors, then fills
// the points past the window -- either by the same recurrence with the newest
// drift v_{r-1} (extrap) or by copying x_{ws+r} -- and, when ws_next >= 0,
// writes the next sweep's masked inputs nl[j] = xs[ws_next + j] * mask[c].
// The latents are channel-major ([C, T] per point, e = c * T + t); v may be
// the transformer's token-major output ([T, C] per point, v_ch = C, token
// stride v_ld >= C, point stride v_pt) or share the latent layout (v_ch = 0,
// points E apart).
#include "flow_euler_window.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

__global__ void flow_euler_window_kernel(__nv_bfloat16* __restrict__ xs,
                                         const __nv_bfloat16* __restrict__ v,
                                         const float* __restrict__ dsig,
                                         int s, int ws, int r, int n, int E,
                                         int v_ch, int v_ld, int v_pt, int extrap,
                                         __nv_bfloat16* __restrict__ nl,
                                         const __nv_bfloat16* __restrict__ mask,
                                         int ch_stride, int ws_next) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    // offset of element e inside one drift point
    const int ve = v_ch > 0 ? (e % (E / v_ch)) * v_ld + e / (E / v_ch) : e;
    const size_t vp = v_ch > 0 ? (size_t)v_pt : (size_t)E;
    __nv_bfloat16 x = xs[(size_t)s * E + e];
    for (int i = s; i < ws + r; ++i) {
        const __nv_bfloat16 t =
            __float2bfloat16_rn(__bfloat162float(v[(size_t)(i - ws) * vp + ve]) * dsig[i]);
        x = __float2bfloat16_rn(__bfloat162float(x) + __bfloat162float(t));
        xs[(size_t)(i + 1) * E + e] = x;
    }
    if (ws + r < n) {
        if (extrap) {
            const float vl = __bfloat162float(v[(size_t)(r - 1) * vp + ve]);
            for (int i = ws + r; i < n; ++i) {
                const __nv_bfloat16 t = __float2bfloat16_rn(vl * dsig[i]);
                x = __float2bfloat16_rn(__bfloat162float(x) + __bfloat162float(t));
                xs[(size_t)(i + 1) * E + e] = x;
            }
        } else {
            for (int i = ws + r + 1; i <= n; ++i) xs[(size_t)i * E + e] = x;
        }
    }
    if (ws_next >= 0) {
        const __nv_bfloat16 mk = mask[e / ch_stride];
        for (int j = 0; j < r; ++j)
            nl[(size_t)j * E + e] = __hmul(xs[(size_t)(ws_next + j) * E + e], mk);
    }
}

// CFG blend + Euler step straight from the transformer's patch sequence.
// pred is [2, L, ld] bf16 (element 0 conditional, 1 unconditional; each token
// row holds ph * pw * C values ordered (ph, pw, C), row stride ld); lat is the
// [C, F, H, W] latent. Per element, with the roundings of the bf16 tensor
// expressions `u + g * (c - u)` and `x + v * dsig`:
//   v = bf16(u + bf16(g * bf16(c - u))),  out = bf16(x + bf16(v * dsig)).
__global__ void flow_cfg_euler_patch_kernel(const __nv_bfloat16* __restrict__ pred,
                                            const __nv_bfloat16* __restrict__ lat,
                                            __nv_bfloat16* __restrict__ out,
                                            int C, int F, int H, int W, int ph, int pw,
                                            int ld, float g, float dsig) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W, total = C * F * HW;
    if (e >= total) return;
    const int c = e / (F * HW), rem = e % (F * HW);
    const int f = rem / HW, y = (rem % HW) / W, x = rem % W;
    const int Hp = H / ph, Wp = W / pw, L = F * Hp * Wp;
    const int l = (f * Hp + y / ph) * Wp + x / pw;
    const int j = ((y % ph) * pw + (x % pw)) * C + c;
    const float cv = __bfloat162float(pred[(size_t)l * ld + j]);
    const float uv = __bfloat162float(pred[((size_t)L + l) * ld + j]);
    const float d = __bfloat162float(__float2bfloat16_rn(cv - uv));
    const float gd = __bfloat162float(__float2bfloat16_rn(g * d));
    const float v = __bfloat162float(__float2bfloat16_rn(uv + gd));
    const float t = __bfloat162float(__float2bfloat16_rn(v * dsig));
    out[e] = __float2bfloat16_rn(__bfloat162float(lat[e]) + t);
}

}  // namespace

int flow_euler_window_bf16(void* xs, const void* v, const float* dsig, int s, int ws,
                           int r, int n, int E, int v_ch, int v_ld, int v_pt,
                           bool extrap, void* nl, const void* mask, int ch_stride,
                           int ws_next, cudaStream_t stream) {
    if (!xs || !v || !dsig || E <= 0 || r <= 0 || ws > s || s >= ws + r || ws + r > n)
        return -1;
    if (v_ch < 0 || (v_ch > 0 && (E % v_ch || v_ld < v_ch || v_pt < (E / v_ch) * v_ld)))
        return -3;
    if (ws_next >= 0 && (!nl || !mask || ch_stride <= 0 || ws_next + r > n)) return -2;
    const int threads = 256;
    flow_euler_window_kernel<<<(E + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(xs), reinterpret_cast<const __nv_bfloat16*>(v),
        dsig, s, ws, r, n, E, v_ch, v_ld, v_pt, extrap ? 1 : 0, reinterpret_cast<__nv_bfloat16*>(nl),
        reinterpret_cast<const __nv_bfloat16*>(mask), ch_stride, ws_next);
    return (int)cudaGetLastError();
}

}  // namespace kernels
}  // namespace flash_rt

namespace flash_rt {
namespace kernels {

int flow_cfg_euler_patch_bf16(const void* pred, const void* lat, void* out, int C, int F,
                              int H, int W, int ph, int pw, int ld, float g, float dsig,
                              cudaStream_t stream) {
    if (!pred || !lat || !out || C <= 0 || ph <= 0 || pw <= 0 || H % ph || W % pw ||
        ld < ph * pw * C)
        return -1;
    const int total = C * F * H * W, threads = 256;
    flow_cfg_euler_patch_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(pred),
        reinterpret_cast<const __nv_bfloat16*>(lat), reinterpret_cast<__nv_bfloat16*>(out),
        C, F, H, W, ph, pw, ld, g, dsig);
    return (int)cudaGetLastError();
}

}  // namespace kernels
}  // namespace flash_rt
