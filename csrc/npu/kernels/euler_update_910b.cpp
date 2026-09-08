#include "kernel_operator.h"
using namespace AscendC;

// The biased action GEMM rounds to BF16. The reference promotes its result
// before multiplying by dt; do not introduce another BF16 rounding or an FMA.
__global__ __aicore__ void euler_update_kernel(GM_ADDR x, GM_ADDR velocity,
                                              GM_ADDR out, int n, float dt) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xi, vi;
    TQue<QuePosition::VECOUT, 1> yo;
    TBuf<QuePosition::VECCALC> temporary;
    pipe.InitBuffer(xi, 1, 256 * 4);
    pipe.InitBuffer(vi, 1, 256 * 2);
    pipe.InitBuffer(yo, 1, 256 * 4);
    pipe.InitBuffer(temporary, 256 * 4);
    GlobalTensor<float> xg, yg;
    GlobalTensor<bfloat16_t> vg;
    xg.SetGlobalBuffer((__gm__ float*)x);
    yg.SetGlobalBuffer((__gm__ float*)out);
    vg.SetGlobalBuffer((__gm__ bfloat16_t*)velocity);
    for(int offset = GetBlockIdx() * 256; offset < n; offset += GetBlockNum() * 256) {
        int count = n - offset < 256 ? n - offset : 256;
        auto a = xi.AllocTensor<float>();
        auto v = vi.AllocTensor<bfloat16_t>();
        DataCopy(a, xg[offset], count);
        DataCopy(v, vg[offset], count);
        xi.EnQue(a); vi.EnQue(v);
        a = xi.DeQue<float>(); v = vi.DeQue<bfloat16_t>();
        auto f = temporary.Get<float>();
        auto y = yo.AllocTensor<float>();
        Cast(f, v, RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
        Muls(f, f, dt, count); PipeBarrier<PIPE_V>();
        Sub(y, a, f, count); yo.EnQue(y);
        y = yo.DeQue<float>(); DataCopy(yg[offset], y, count);
        xi.FreeTensor(a); vi.FreeTensor(v); yo.FreeTensor(y);
    }
}

extern "C" int flashrt_npu_euler_update(void* stream, void* x, void* velocity,
                                        void* out, int n, float dt) {
    if(!x || !velocity || !out || n <= 0 || n % 32 || n > 2147483647 / 4
       || !(dt > 0.0f && dt <= 1.0f)) return 1;
    int blocks = (n + 255) / 256;
    if(blocks > 40) blocks = 40;
    euler_update_kernel<<<blocks, nullptr, stream>>>((uint8_t*)x,
        (uint8_t*)velocity, (uint8_t*)out, n, dt);
    return 0;
}
