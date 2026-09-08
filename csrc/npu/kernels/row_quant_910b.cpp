#include "kernel_operator.h"
#include "lib/quantization/ascend_quant.h"
using namespace AscendC;
template<int tile>
__global__ __aicore__ void row_quant_kernel(GM_ADDR input, GM_ADDR inv_scale,
                                           GM_ADDR output, int rows, int columns) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> in_queue;
    TQue<QuePosition::VECOUT, 1> out_queue;
    TBuf<QuePosition::VECCALC> float_buffer, temporary;
    pipe.InitBuffer(in_queue, 1, tile * sizeof(bfloat16_t));
    pipe.InitBuffer(out_queue, 1, tile * sizeof(int8_t));
    pipe.InitBuffer(float_buffer, tile * sizeof(float));
    pipe.InitBuffer(temporary, tile * 8);
    GlobalTensor<bfloat16_t> src;
    GlobalTensor<int8_t> dst;
    GlobalTensor<float> scales;
    src.SetGlobalBuffer((__gm__ bfloat16_t*)input);
    dst.SetGlobalBuffer((__gm__ int8_t*)output);
    scales.SetGlobalBuffer((__gm__ float*)inv_scale);
    for (int row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        const float scale = scales.GetValue(row);
        for (int col = 0; col < columns; col += tile) {
            const int count = columns - col < tile ? columns - col : tile;
            auto x = in_queue.AllocTensor<bfloat16_t>();
            DataCopy(x, src[row * columns + col], count);
            in_queue.EnQue(x);
            x = in_queue.DeQue<bfloat16_t>();
            auto f = float_buffer.Get<float>();
            auto y = out_queue.AllocTensor<int8_t>();
            Cast(f, x, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Muls(f, f, scale, count);
            PipeBarrier<PIPE_V>();
            AscendQuant(y, f, temporary.Get<uint8_t>(), 1.0f, 0.0f, count);
            out_queue.EnQue(y);
            in_queue.FreeTensor(x);
            y = out_queue.DeQue<int8_t>();
            DataCopy(dst[row * columns + col], y, count);
            out_queue.FreeTensor(y);
        }
    }
}
// Checked pointer/stream ABI. Submission is asynchronous; the caller owns
// buffer lifetime and observes completion through its stream or graph.
extern "C" int flashrt_npu_quantize_rows(void* stream, void* input, void* inv_scale,
                                        void* output, int rows, int columns) {
    if (!input || !inv_scale || !output || rows <= 0 || columns <= 0 || columns % 32)
        return 1;
    const int blocks = rows < 40 ? rows : 40;
    if (columns >= 8192) {
        row_quant_kernel<8192><<<blocks, nullptr, stream>>>((uint8_t*)input,
            (uint8_t*)inv_scale, (uint8_t*)output, rows, columns);
    } else {
        row_quant_kernel<4096><<<blocks, nullptr, stream>>>((uint8_t*)input,
            (uint8_t*)inv_scale, (uint8_t*)output, rows, columns);
    }
    return 0;
}
