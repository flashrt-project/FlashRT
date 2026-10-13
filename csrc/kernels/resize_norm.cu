// Generic uint8 image resize + rescale-to-bf16 preprocessing. See resize_norm.cuh.

#include "resize_norm.cuh"

#include <stdexcept>

namespace flash_rt {
namespace kernels {
namespace {

struct LinCoef { int s0; short a0, a1; };

// OpenCV's coefficient computation. It clamps the fractional weight only along x;
// along y it keeps the raw fraction and clamps only the source *row* index, which
// rounds differently at the first and last output rows. Reproducing that
// asymmetry is part of matching cv2.
__device__ __forceinline__ LinCoef lin_coef(int d, int ssize, double scale, bool clamp_coef) {
    float fx = (float)((d + 0.5) * scale - 0.5);
    int sx = (int)floorf(fx);
    fx -= (float)sx;
    if (clamp_coef) {
        if (sx < 0) { fx = 0.f; sx = 0; }
        if (sx >= ssize - 1) { fx = 0.f; sx = ssize - 1; }
    }
    LinCoef c;
    c.s0 = sx;
    c.a0 = (short)__float2int_rn((1.f - fx) * 2048.f);
    c.a1 = (short)__float2int_rn(fx * 2048.f);
    return c;
}

__global__ void resize_norm_kernel(const uint8_t* __restrict__ in,
                                   __nv_bfloat16* __restrict__ out,
                                   int nv, int H, int W, int OH, int OW) {
    const long long total = (long long)nv * OH * OW;
    const double sy = (double)H / OH, sx = (double)W / OW;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int ox = (int)(t % OW), oy = (int)((t / OW) % OH), v = (int)(t / ((long long)OW * OH));
        const uint8_t* img = in + (long long)v * H * W * 3;
        const LinCoef cx = lin_coef(ox, W, sx, true), cy = lin_coef(oy, H, sy, false);
        const int x1 = min(cx.s0 + 1, W - 1);
        const int y0 = min(max(cy.s0, 0), H - 1), y1 = min(max(cy.s0 + 1, 0), H - 1);
        for (int c = 0; c < 3; ++c) {
            const int r0 = img[(y0 * W + cx.s0) * 3 + c] * cx.a0 + img[(y0 * W + x1) * 3 + c] * cx.a1;
            const int r1 = img[(y1 * W + cx.s0) * 3 + c] * cx.a0 + img[(y1 * W + x1) * 3 + c] * cx.a1;
            // OpenCV's SIMD vertical combine, not the scalar FixedPtCast
            int val = (((cy.a0 * (r0 >> 4)) >> 16) + ((cy.a1 * (r1 >> 4)) >> 16) + 2) >> 2;
            val = min(max(val, 0), 255);
            out[(((long long)v * 3 + c) * OH + oy) * OW + ox] =
                __float2bfloat16(__fdiv_rn((float)val - 127.5f, 127.5f));
        }
    }
}

}  // namespace

void resize_norm_u8_hwc_to_bf16_chw(const uint8_t* in, __nv_bfloat16* out,
                                    int nv, int H, int W, int OH, int OW,
                                    cudaStream_t stream) {
    if (nv <= 0 || H <= 0 || W <= 0 || OH <= 0 || OW <= 0) return;
    if (!in || !out)
        throw std::runtime_error("resize_norm_u8_hwc_to_bf16_chw: null pointer argument");
    const long long total = (long long)nv * OH * OW;
    int grid = (int)((total + 255) / 256);
    if (grid > 65535) grid = 65535;
    resize_norm_kernel<<<grid, 256, 0, stream>>>(in, out, nv, H, W, OH, OW);
}

}  // namespace kernels
}  // namespace flash_rt
