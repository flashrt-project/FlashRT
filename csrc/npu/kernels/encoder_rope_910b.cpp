#include "kernel_operator.h"
using namespace AscendC;

// Half-split RoPE tables repeat their first 128 columns in the second half.
// Process all eight Q heads and the K head with strided vector instructions;
// both products and their sum remain FP32 before one final BF16 rounding.
__global__ __aicore__ void encoder_rope_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                               GM_ADDR qo, GM_ADDR ko, int rows) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> iq, cq, sq;
    TQue<QuePosition::VECOUT, 1> oq;
    TBuf<QuePosition::VECCALC> xb, yb, tb;
    pipe.InitBuffer(iq, 1, 2304 * 2); pipe.InitBuffer(cq, 1, 128 * 4);
    pipe.InitBuffer(sq, 1, 128 * 4); pipe.InitBuffer(oq, 1, 2304 * 2);
    pipe.InitBuffer(xb, 2304 * 4); pipe.InitBuffer(yb, 2304 * 4); pipe.InitBuffer(tb, 2304 * 4);
    GlobalTensor<bfloat16_t> qg, kg, qog, kog; GlobalTensor<float> cg, sg;
    qg.SetGlobalBuffer((__gm__ bfloat16_t*)q); kg.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    qog.SetGlobalBuffer((__gm__ bfloat16_t*)qo); kog.SetGlobalBuffer((__gm__ bfloat16_t*)ko);
    cg.SetGlobalBuffer((__gm__ float*)cos); sg.SetGlobalBuffer((__gm__ float*)sin);
    const BinaryRepeatParams broadcast{1, 1, 1, 32, 32, 0};
    const BinaryRepeatParams aligned{1, 1, 1, 32, 32, 32};
    for(int row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        auto input = iq.AllocTensor<bfloat16_t>(); DataCopy(input, qg[row * 2048], 2048);
        DataCopy(input[2048], kg[row * 256], 256); iq.EnQue(input);
        auto c = cq.AllocTensor<float>(); auto s = sq.AllocTensor<float>();
        DataCopy(c, cg[row * 256], 128); DataCopy(s, sg[row * 256], 128); cq.EnQue(c); sq.EnQue(s);
        input = iq.DeQue<bfloat16_t>(); c = cq.DeQue<float>(); s = sq.DeQue<float>();
        auto x = xb.Get<float>(); auto y = yb.Get<float>(); auto t = tb.Get<float>();
        auto output = oq.AllocTensor<bfloat16_t>();
        Cast(x, input, RoundMode::CAST_NONE, 2304); PipeBarrier<PIPE_V>();
        for(int offset = 0; offset < 128; offset += 64) {
            Mul(y[offset], x[offset], c[offset], uint64_t(64), 9, broadcast); PipeBarrier<PIPE_V>();
            Mul(t[offset], x[128 + offset], s[offset], uint64_t(64), 9,
                broadcast); PipeBarrier<PIPE_V>();
            Sub(y[offset], y[offset], t[offset], uint64_t(64), 9, aligned); PipeBarrier<PIPE_V>();
            Mul(y[128 + offset], x[128 + offset], c[offset], uint64_t(64), 9, broadcast);
            PipeBarrier<PIPE_V>();
            Mul(t[offset], x[offset], s[offset], uint64_t(64), 9, broadcast); PipeBarrier<PIPE_V>();
            Add(y[128 + offset], y[128 + offset], t[offset], uint64_t(64), 9,
                aligned); PipeBarrier<PIPE_V>();
        }
        Cast(output, y, RoundMode::CAST_RINT, 2304); oq.EnQue(output);
        output = oq.DeQue<bfloat16_t>();
        DataCopy(qog[row * 2048], output, 2048); DataCopy(kog[row * 256], output[2048], 256);
        iq.FreeTensor(input); cq.FreeTensor(c); sq.FreeTensor(s); oq.FreeTensor(output);
    }
}
extern "C" int flashrt_npu_encoder_rope(void* stream, void* q, void* k, void* cos, void* sin,
                                        void* qo, void* ko, int rows) {
    if(!q || !k || !cos || !sin || !qo || !ko || rows <= 0 || rows > 2147483647 / 2304)return 1;
    encoder_rope_kernel <<< rows < 40?rows:40, nullptr,
        stream >>> ((uint8_t*)q, (uint8_t*)k, (uint8_t*)cos, (uint8_t*)sin, (uint8_t*)qo,
                     (uint8_t*)ko,
                     rows); return 0;
}
