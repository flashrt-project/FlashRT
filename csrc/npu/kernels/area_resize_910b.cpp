// The evaluation image transform's resize, on the die, bit for bit.
//
// The reference's evaluation transform is a smallest-edge resize to 256, a
// centre crop to 95 percent of each side, and the same resize again -- all of
// it cv2.INTER_AREA on uint8. On this box it is 7.0 ms of host a frame, more
// than a tenth of the whole served frame, and every step of it is a gather and
// a weighted sum.
//
// Two reasons this is a kernel rather than a handful of torch calls. A torch
// version of the same arithmetic measures 8.9 ms -- slower than the host it
// replaces -- because an integer right shift on this part runs at 15 GB/s where
// a multiply runs at 151, and the pipeline needs ten of them over five-megabyte
// tensors. And the shifts cannot be turned into float multiplies outside a
// kernel: the vertical pass forms a 26-bit product, and FP32 has 24 bits. In UB
// the shifts are free, and the whole transform is one pass over each output row.
//
// What it reproduces is OpenCV's enlarging path. True area interpolation exists
// only for shrinking; when the image is enlarged OpenCV emulates it with a
// two-tap interpolation that keeps the area source coordinate, and both of this
// transform's resizes enlarge. The tap tables are built on the host, where the
// three details that decide exactness live (see preprocess.py). What lives here
// is the other half: the 8-bit vertical pass is not the fixed-point cast its own
// generic template advertises but a specialisation that truncates three separate
// times, and rounding the accumulator once instead is off by one least
// significant bit on about eight percent of the pixels.
#include "kernel_operator.h"
using namespace AscendC;

namespace flashrt_area_resize {
constexpr int MAX_ROW = 2048;    // source samples in a row, channels included
constexpr int MAX_OUT = 1536;    // destination samples in a row, channels included
}

__global__ __aicore__ void area_resize_kernel(GM_ADDR src, GM_ADDR dst, GM_ADDR offsets,
                                              GM_ADDR weights, GM_ADDR rows,
                                              GM_ADDR rowWeights, int images,
                                              int srcPlane, int srcRowStride, int rowOffset,
                                              int srcSamples, int dstPlane,
                                              int dstRowStride, int dstHeight,
                                              int dstSamples, int tableStride) {
    using namespace flashrt_area_resize;
    TPipe pipe;
    TBuf<TPosition::VECCALC> ubuf, hbuf, fbuf, g0buf, g1buf, a0buf, a1buf, i0buf, i1buf,
        o0buf, o1buf, w0buf, w1buf;
    pipe.InitBuffer(ubuf, MAX_ROW);
    pipe.InitBuffer(hbuf, MAX_ROW * 2);
    pipe.InitBuffer(fbuf, MAX_ROW * 4);
    pipe.InitBuffer(g0buf, MAX_OUT * 4);
    pipe.InitBuffer(g1buf, MAX_OUT * 4);
    pipe.InitBuffer(a0buf, MAX_OUT * 4);
    pipe.InitBuffer(a1buf, MAX_OUT * 4);
    pipe.InitBuffer(i0buf, MAX_OUT * 4);
    pipe.InitBuffer(i1buf, MAX_OUT * 4);
    pipe.InitBuffer(o0buf, MAX_OUT * 4);
    pipe.InitBuffer(o1buf, MAX_OUT * 4);
    pipe.InitBuffer(w0buf, MAX_OUT * 4);
    pipe.InitBuffer(w1buf, MAX_OUT * 4);

    GlobalTensor<uint8_t> sg, dg;
    GlobalTensor<uint32_t> og;
    GlobalTensor<float> wg;
    GlobalTensor<int32_t> rg, bg;
    sg.SetGlobalBuffer((__gm__ uint8_t*)src);
    dg.SetGlobalBuffer((__gm__ uint8_t*)dst);
    og.SetGlobalBuffer((__gm__ uint32_t*)offsets);
    wg.SetGlobalBuffer((__gm__ float*)weights);
    rg.SetGlobalBuffer((__gm__ int32_t*)rows);
    bg.SetGlobalBuffer((__gm__ int32_t*)rowWeights);

    auto bytes = ubuf.Get<uint8_t>();
    auto wide = hbuf.Get<half>();
    auto row = fbuf.Get<float>();
    auto gathered0 = g0buf.Get<float>();
    auto gathered1 = g1buf.Get<float>();
    auto acc0 = a0buf.Get<float>();
    auto acc1 = a1buf.Get<float>();
    auto part0 = i0buf.Get<int32_t>();
    auto part1 = i1buf.Get<int32_t>();
    auto column0 = o0buf.Get<uint32_t>();
    auto column1 = o1buf.Get<uint32_t>();
    auto weight0 = w0buf.Get<float>();
    auto weight1 = w1buf.Get<float>();

    const int span = (dstSamples + 7) / 8 * 8;
    const int sourceSpan = (srcSamples + 7) / 8 * 8;
    // The column tables are the same for every row of every image. The second
    // tap's table starts a whole number of 32-byte blocks in, because a copy
    // out of global memory has to start on one -- at 1365 samples a row the
    // natural packing puts it at byte 5460 and the second tap reads garbage.
    DataCopy(column0, og, span);
    DataCopy(column1, og[tableStride], span);
    DataCopy(weight0, wg, span);
    DataCopy(weight1, wg[tableStride], span);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    // A source row rounded up to whole 32-byte blocks. The tail it drags in is
    // never gathered: every offset in the table is inside the live extent.
    const int copyBytes = (srcSamples + 31) / 32 * 32;
    // Vector work runs over a whole number of 32-byte blocks. Gather writes
    // blocks, so a span of 1365 leaves the last five samples of every row
    // holding whatever was there before -- a fifth of a percent of the pixels,
    // and invisible in anything but a per-pixel check. The tap tables are
    // padded to the same span with a zero offset and a zero weight, so the
    // extra samples compute zero and land in the destination row's padding.
    const int units = images * dstHeight;
    for (int unit = GetBlockIdx(); unit < units; unit += GetBlockNum()) {
        const int image = unit / dstHeight;
        const int line = unit - image * dstHeight;
        for (int tap = 0; tap < 2; ++tap) {
            const int source = rg.GetValue(tap * dstHeight + line);
            // The row buffer is about to be overwritten on MTE2 while the
            // previous tap is still reading it on V, and the two pipes run
            // concurrently. Without this the first repeat of each row is right
            // and the rest is whatever arrived first -- a mismatch that starts
            // at element 64 and looks like a cast with the wrong stride.
            PipeBarrier<PIPE_ALL>();
            DataCopy(bytes, sg[(uint32_t)image * srcPlane
                               + (uint32_t)(rowOffset + source) * srcRowStride],
                     copyBytes);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Cast(wide, bytes, RoundMode::CAST_NONE, sourceSpan);
            PipeBarrier<PIPE_V>();
            Cast(row, wide, RoundMode::CAST_NONE, sourceSpan);
            PipeBarrier<PIPE_V>();
            // The horizontal pass is exact in FP32: both taps are at most
            // 255 * 2048 and their sum is at most 2048 * 255, well inside 24
            // bits. It is the vertical pass that needs integers.
            // Gather's offsets are bytes from the start of unified buffer, and
            // a LocalTensor's own address is already in them, so the base is
            // zero. Passing the tensor's address instead double counts it.
            Gather(gathered0, row, column0, 0, (uint32_t)span);
            PipeBarrier<PIPE_V>();
            Gather(gathered1, row, column1, 0, (uint32_t)span);
            PipeBarrier<PIPE_V>();
            Mul(gathered0, gathered0, weight0, span);
            PipeBarrier<PIPE_V>();
            Mul(gathered1, gathered1, weight1, span);
            PipeBarrier<PIPE_V>();
            auto sum = tap == 0 ? acc0 : acc1;
            Add(sum, gathered0, gathered1, span);
            PipeBarrier<PIPE_V>();
            // >> 4, as a float multiply and a flooring cast: the operand is
            // under 2^24 here, so this is exact and it is not a shift.
            Muls(sum, sum, 0.0625f, span);
            PipeBarrier<PIPE_V>();
            auto part = tap == 0 ? part0 : part1;
            Cast(part, sum, RoundMode::CAST_FLOOR, span);
            PipeBarrier<PIPE_V>();
            Muls(part, part, bg.GetValue(tap * dstHeight + line), span);
            PipeBarrier<PIPE_V>();
            ShiftRight(part, part, (int32_t)16, span);
            PipeBarrier<PIPE_V>();
        }
        Add(part0, part0, part1, span);
        PipeBarrier<PIPE_V>();
        Adds(part0, part0, (int32_t)2, span);
        PipeBarrier<PIPE_V>();
        ShiftRight(part0, part0, (int32_t)2, span);
        PipeBarrier<PIPE_V>();
        Cast(acc0, part0, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Cast(wide, acc0, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Cast(bytes, wide, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // The destination row is padded to whole blocks by its caller, so the
        // tail this writes past the live samples lands inside the row.
        DataCopy(dg[(uint32_t)image * dstPlane + (uint32_t)line * dstRowStride], bytes,
                 (dstSamples + 31) / 32 * 32);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int flashrt_npu_area_resize(void* stream, void* src, void* dst, void* offsets,
                                       void* weights, void* rows, void* rowWeights,
                                       int images, int srcPlane, int srcRowStride,
                                       int rowOffset, int srcSamples, int dstPlane,
                                       int dstRowStride, int dstHeight, int dstSamples,
                                       int tableStride, int cores) {
    using namespace flashrt_area_resize;
    if (!stream || !src || !dst || !offsets || !weights || !rows || !rowWeights) {
        return 1;
    }
    if (images <= 0 || dstHeight <= 0 || dstSamples <= 0 || srcSamples <= 0) { return 2; }
    if (srcSamples > MAX_ROW || dstSamples > MAX_OUT) { return 3; }
    if (srcRowStride % 32 || dstRowStride % 32) { return 4; }
    if (cores <= 0 || cores > 48) { return 5; }
    if (tableStride < dstSamples || tableStride % 8) { return 6; }
    area_resize_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)src, (uint8_t*)dst, (uint8_t*)offsets, (uint8_t*)weights,
        (uint8_t*)rows, (uint8_t*)rowWeights, images, srcPlane, srcRowStride, rowOffset,
        srcSamples, dstPlane, dstRowStride, dstHeight, dstSamples, tableStride);
    return 0;
}

#include "../abi.h"
