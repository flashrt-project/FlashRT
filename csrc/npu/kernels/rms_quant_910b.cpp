#include "kernel_operator.h"
#include "lib/quantization/ascend_quant.h"
using namespace AscendC;

// Preserve the encoder's BF16 residual and normalized-value rounding boundaries.
// Variance, reciprocal RMS, gamma multiplication and scale multiplication use FP32.
// The final AscendQuant conversion writes INT8 directly into the GEMM operand.
__global__ __aicore__ void rms_quant_kernel(GM_ADDR input, GM_ADDR other, GM_ADDR gamma,
                                            GM_ADDR inverse, GM_ADDR quantized, GM_ADDR residual,
                                            int rows, int add) {
    constexpr int D = 2048;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xq, oq, gq;
    TQue<QuePosition::VECOUT, 1> qq, rq;

    TBuf<QuePosition::VECCALC> xb, tb, gb, bb, rb, work, qtemp;

    pipe.InitBuffer(xq, 1, 2 * D);
    pipe.InitBuffer(oq, 1, 2 * D);
    pipe.InitBuffer(gq, 1, 2 * D);
    pipe.InitBuffer(qq, 1, D);
    pipe.InitBuffer(rq, 1, 2 * D);

    pipe.InitBuffer(xb, 4 * D);
    pipe.InitBuffer(tb, 4 * D);
    pipe.InitBuffer(gb, 4 * D);
    pipe.InitBuffer(bb, 2 * D);
    pipe.InitBuffer(rb, 256);
    pipe.InitBuffer(work, 4 * D);
    pipe.InitBuffer(qtemp, 8 * D);

    GlobalTensor<bfloat16_t> xg, og, gg, rg;
    GlobalTensor<float> ig;
    GlobalTensor<int8_t> qg;

    xg.SetGlobalBuffer((__gm__ bfloat16_t*)input);
    og.SetGlobalBuffer((__gm__ bfloat16_t*)other);
    gg.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
    rg.SetGlobalBuffer((__gm__ bfloat16_t*)residual);
    ig.SetGlobalBuffer((__gm__ float*)inverse);
    qg.SetGlobalBuffer((__gm__ int8_t*)quantized);

    auto gl = gq.AllocTensor<bfloat16_t>();
    DataCopy(gl, gg, D);
    gq.EnQue(gl);
    gl = gq.DeQue<bfloat16_t>();
    auto gf = gb.Get<float>();
    Cast(gf, gl, RoundMode::CAST_NONE, D);
    PipeBarrier<PIPE_V>();
    gq.FreeTensor(gl);

    for (int row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        auto xl = xq.AllocTensor<bfloat16_t>();
        DataCopy(xl, xg[row * D], D);
        xq.EnQue(xl);
        xl = xq.DeQue<bfloat16_t>();

        auto x = xb.Get<float>();
        auto t = tb.Get<float>();
        auto bf = bb.Get<bfloat16_t>();
        auto q = qq.AllocTensor<int8_t>();
        auto r = rq.AllocTensor<bfloat16_t>();

        Cast(x, xl, RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();

        if (add) {
            auto o = oq.AllocTensor<bfloat16_t>();
            DataCopy(o, og[row * D], D);
            oq.EnQue(o);
            o = oq.DeQue<bfloat16_t>();
            Cast(t, o, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            Add(t, x, t, D);
            PipeBarrier<PIPE_V>();
            Cast(r, t, RoundMode::CAST_RINT, D);
            PipeBarrier<PIPE_V>();
            Cast(x, r, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            oq.FreeTensor(o);

        }
        Mul(t, x, x, D);
        PipeBarrier<PIPE_V>();
        auto rs = rb.Get<float>();
        ReduceSum(rs, t, work.Get<float>(), D);
        PipeBarrier<PIPE_V>();
        Muls(rs, rs, 1.0f / D, 1);
        PipeBarrier<PIPE_V>();
        Adds(rs, rs, 1e-6f, 1);
        PipeBarrier<PIPE_V>();
        Rsqrt(rs, rs, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float rms_inverse = rs.GetValue(0);

        Muls(t, x, rms_inverse, D);
        PipeBarrier<PIPE_V>();
        Mul(t, t, gf, D);
        PipeBarrier<PIPE_V>();
        Cast(bf, t, RoundMode::CAST_RINT, D);
        PipeBarrier<PIPE_V>();
        Cast(t, bf, RoundMode::CAST_NONE, D);
        PipeBarrier<PIPE_V>();
        Muls(t, t, ig.GetValue(row), D);
        PipeBarrier<PIPE_V>();
        AscendQuant(q, t, qtemp.Get<uint8_t>(), 1.0f, 0.0f, D);

        qq.EnQue(q);
        rq.EnQue(r);
        xq.FreeTensor(xl);
        q = qq.DeQue<int8_t>();
        r = rq.DeQue<bfloat16_t>();
        DataCopy(qg[row * D], q, D);
        if (add) DataCopy(rg[row * D], r, D);
        qq.FreeTensor(q);
        rq.FreeTensor(r);

    }
}
extern "C" int flashrt_npu_rms_row_quant(void* stream, void* input, void* other, void* gamma,
                                         void* inverse, void* quantized, void* residual, int rows,
                                         int add) {
    if (!input || !gamma || !inverse || !quantized || !residual || (add != 0 && add != 1) ||
        rows <= 0 || rows > 2147483647 / 2048 || (add && !other)) return 1;

    rms_quant_kernel<<<rows < 40 ? rows : 40, nullptr, stream>>>((uint8_t*)input, (uint8_t*)other, (uint8_t*)gamma, (uint8_t*)inverse,
                     (uint8_t*)quantized, (uint8_t*)residual, rows, add);
    return 0;

}
