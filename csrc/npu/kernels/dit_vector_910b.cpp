// The action head's fused add-and-normalise.
//
// This is a draw on time and it ships for its arithmetic. The vendor's fused
// form charges 7.69 us at the DiT's 41 rows for 504 KB of traffic, which is
// 0.71 us at this part's read roof; this one charges the same. That cost is
// not bandwidth and it is not instructions -- binding the two DiT norm sites
// to different implementations moves it from one site to the other, so it
// belongs to the site, and the only thing that removes it is not launching a
// kernel there at all, which needs the GEMM epilogue to own it.
//
// What it does buy is one rounding instead of two. The residual sum is rounded
// to BF16 -- that value is what the next block carries forward, so it has to
// be -- and the normalisation then runs in FP32 from it, where the vendor's
// pair rounds again in between. End to end that is 0.9999471 to 0.9999601
// combined, and the gripper modality 0.9956 to 0.9981.
//
// A row to a core, every read of the launch issued before any of it is waited
// on, and the two reductions broadcast back through Brcb rather than read on
// the scalar unit.
//
// It also carries two things that belong to the projection in front of it. The
// branch's bias is added here rather than by that projection, because aclnn's
// biased matmul casts its bias on every call -- 1536 numbers that never change,
// once a call, as its own kernel launch -- while this one already has the row in
// UB and reads the affine pair once a launch anyway. And the normalised row is
// written at a pitch, so it can end in a constant one and let *its* consumer
// carry a bias as one more input channel for the same reason.
#include "kernel_operator.h"
using namespace AscendC;

namespace flashrt_dit_vector {
constexpr int MAX_NORM_COLS = 2048;
// Broadcast one reduced value across a whole row without a scalar read: Brcb
// fills a 32-byte block with it, and a repeat stride of zero makes every
// repeat of the elementwise op read that same block.
__aicore__ inline void BroadcastRow(const LocalTensor<float>& dst,
                                    const LocalTensor<float>& scalarSrc) {
    Brcb(dst, scalarSrc, 1, BrcbRepeatParams(1, 8));
}
}  // namespace flashrt_dit_vector

__global__ __aicore__ void dit_add_layer_norm_kernel(GM_ADDR residual, GM_ADDR branch,
                                                     GM_ADDR gamma, GM_ADDR beta,
                                                     GM_ADDR norm, GM_ADDR total,
                                                     GM_ADDR branchBias, int rows,
                                                     int cols, int pitch, float eps) {
    using namespace flashrt_dit_vector;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> rq, bq, gq, cq, pq;
    TQue<TPosition::VECOUT, 1> nq, tq;
    TBuf<TPosition::VECCALC> xbuf, sbuf, gbuf, cbuf, wbuf, rbuf, mbuf, pbuf, hbuf;
    pipe.InitBuffer(rq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(bq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(gq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(cq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(nq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(tq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(xbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(sbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(gbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(cbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(wbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(rbuf, 256);
    pipe.InitBuffer(mbuf, 256);
    pipe.InitBuffer(pq, 1, MAX_NORM_COLS * 2);
    pipe.InitBuffer(pbuf, MAX_NORM_COLS * 4);
    pipe.InitBuffer(hbuf, MAX_NORM_COLS * 2);

    GlobalTensor<bfloat16_t> rg, bg, gg, cg, ng, tg, pg;
    if (branchBias) { pg.SetGlobalBuffer((__gm__ bfloat16_t*)branchBias); }
    rg.SetGlobalBuffer((__gm__ bfloat16_t*)residual);
    bg.SetGlobalBuffer((__gm__ bfloat16_t*)branch);
    gg.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
    cg.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
    ng.SetGlobalBuffer((__gm__ bfloat16_t*)norm);
    tg.SetGlobalBuffer((__gm__ bfloat16_t*)total);

    auto x = xbuf.Get<float>();
    auto sq = sbuf.Get<float>();
    auto gf = gbuf.Get<float>();
    auto cf = cbuf.Get<float>();
    auto work = wbuf.Get<float>();
    auto red = rbuf.Get<float>();
    auto bcast = mbuf.Get<float>();

    // Every read of the launch is issued before any of it is waited on. The
    // affine pair is 3 KB and the row is 3 KB, and at 41 rows a core owns one
    // row: waiting for the pair and then asking for the row is two cold round
    // trips in series with nothing else to do, and in the frame that is the
    // difference between four microseconds and fourteen. The pair is also read
    // once for the whole launch rather than once a row.
    const float inverse = 1.0f / (float)cols;
    const int repeats = cols / 64;
    const int first = GetBlockIdx();
    auto gl = gq.AllocTensor<bfloat16_t>();
    DataCopy(gl, gg, cols);
    gq.EnQue(gl);
    auto cl = cq.AllocTensor<bfloat16_t>();
    DataCopy(cl, cg, cols);
    cq.EnQue(cl);
    if (first < rows) {
        auto r0 = rq.AllocTensor<bfloat16_t>();
        DataCopy(r0, rg[(uint32_t)first * cols], cols);
        rq.EnQue(r0);
        auto b0 = bq.AllocTensor<bfloat16_t>();
        DataCopy(b0, bg[(uint32_t)first * cols], cols);
        bq.EnQue(b0);
    }
    // The branch's bias, read once for the launch beside the affine pair.
    auto bias = pbuf.Get<float>();
    if (branchBias) {
        auto bl = pq.AllocTensor<bfloat16_t>();
        DataCopy(bl, pg, cols);
        pq.EnQue(bl);
        bl = pq.DeQue<bfloat16_t>();
        Cast(bias, bl, RoundMode::CAST_NONE, cols);
        PipeBarrier<PIPE_V>();
        pq.FreeTensor(bl);
    }
    gl = gq.DeQue<bfloat16_t>();
    Cast(gf, gl, RoundMode::CAST_NONE, cols);
    PipeBarrier<PIPE_V>();
    gq.FreeTensor(gl);
    cl = cq.DeQue<bfloat16_t>();
    Cast(cf, cl, RoundMode::CAST_NONE, cols);
    PipeBarrier<PIPE_V>();
    cq.FreeTensor(cl);

    for (int row = first; row < rows; row += GetBlockNum()) {
        const uint32_t base = (uint32_t)row * cols;
        if (row != first) {
            auto rn = rq.AllocTensor<bfloat16_t>();
            DataCopy(rn, rg[base], cols);
            rq.EnQue(rn);
            auto bn = bq.AllocTensor<bfloat16_t>();
            DataCopy(bn, bg[base], cols);
            bq.EnQue(bn);
        }
        auto r = rq.DeQue<bfloat16_t>();
        auto b = bq.DeQue<bfloat16_t>();
        Cast(x, r, RoundMode::CAST_NONE, cols);
        PipeBarrier<PIPE_V>();
        Cast(sq, b, RoundMode::CAST_NONE, cols);
        PipeBarrier<PIPE_V>();
        rq.FreeTensor(r);
        bq.FreeTensor(b);
        if (branchBias) {
            // The bias goes onto the branch and the branch is then rounded to
            // BF16, which is exactly what the biased matmul this replaced did.
            // Adding it straight into the FP32 sum instead is *better*
            // arithmetic and a worse answer: the reference rounds here, so the
            // end-to-end cosine reads the missing rounding as drift -- it cost
            // the combined figure 0.9999614 -> 0.9999531 and the gripper
            // modality 0.9946 -> 0.9910 before this was put back.
            auto rounded = hbuf.Get<bfloat16_t>();
            Add(sq, sq, bias, cols);
            PipeBarrier<PIPE_V>();
            Cast(rounded, sq, RoundMode::CAST_RINT, cols);
            PipeBarrier<PIPE_V>();
            Cast(sq, rounded, RoundMode::CAST_NONE, cols);
            PipeBarrier<PIPE_V>();
        }
        Add(x, x, sq, cols);
        PipeBarrier<PIPE_V>();

        // Round to BF16 first: this is the value the next block sees as its
        // residual, and the reference normalises exactly it.
        auto t = tq.AllocTensor<bfloat16_t>();
        Cast(t, x, RoundMode::CAST_RINT, cols);
        PipeBarrier<PIPE_V>();
        Cast(x, t, RoundMode::CAST_NONE, cols);
        PipeBarrier<PIPE_V>();
        tq.EnQue(t);
        t = tq.DeQue<bfloat16_t>();
        DataCopy(tg[base], t, cols);
        tq.FreeTensor(t);

        ReduceSum<float>(red, x, work, cols);
        PipeBarrier<PIPE_V>();
        Muls(red, red, inverse, 1);
        PipeBarrier<PIPE_V>();
        BroadcastRow(bcast, red);
        PipeBarrier<PIPE_V>();
        Sub(x, x, bcast, 64, (uint8_t)repeats, BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        PipeBarrier<PIPE_V>();
        Mul(sq, x, x, cols);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(red, sq, work, cols);
        PipeBarrier<PIPE_V>();
        Muls(red, red, inverse, 1);
        PipeBarrier<PIPE_V>();
        Adds(red, red, eps, 1);
        PipeBarrier<PIPE_V>();
        Rsqrt(red, red, 1);
        PipeBarrier<PIPE_V>();
        BroadcastRow(bcast, red);
        PipeBarrier<PIPE_V>();
        Mul(x, x, bcast, 64, (uint8_t)repeats, BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        PipeBarrier<PIPE_V>();
        Mul(x, x, gf, cols);
        PipeBarrier<PIPE_V>();
        Add(x, x, cf, cols);
        PipeBarrier<PIPE_V>();
        auto n = nq.AllocTensor<bfloat16_t>();
        Cast(n, x, RoundMode::CAST_RINT, cols);
        PipeBarrier<PIPE_V>();
        nq.EnQue(n);
        n = nq.DeQue<bfloat16_t>();
        DataCopy(ng[(uint32_t)row * pitch], n, cols);
        nq.FreeTensor(n);
    }
}

extern "C" int flashrt_npu_dit_add_layer_norm(void* stream, void* residual, void* branch,
                                              void* gamma, void* beta, void* norm,
                                              void* total, void* branchBias, int rows,
                                              int cols, int pitch, float eps) {
    using namespace flashrt_dit_vector;
    if (!stream || !residual || !branch || !gamma || !beta || !norm || !total) { return 1; }
    // A 64-element repeat is what carries the broadcast of the mean and the
    // reciprocal square root, so the row has to be a whole number of them.
    if (rows <= 0 || cols <= 0 || cols % 64 || cols > MAX_NORM_COLS) { return 2; }
    // The normalised row is written at `pitch` and the sum at `cols`: the sum is
    // a residual and nothing projects from it. A 16-element pitch step keeps the
    // store block aligned.
    if (pitch < cols || (pitch - cols) % 16) { return 3; }
    const int blocks = rows < 40 ? rows : 40;
    dit_add_layer_norm_kernel<<<blocks, nullptr, stream>>>(
        (uint8_t*)residual, (uint8_t*)branch, (uint8_t*)gamma, (uint8_t*)beta,
        (uint8_t*)norm, (uint8_t*)total, (uint8_t*)branchBias, rows, cols, pitch, eps);
    return 0;
}

#include "../abi.h"
