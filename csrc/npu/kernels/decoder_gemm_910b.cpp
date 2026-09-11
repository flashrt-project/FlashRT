// Hand-pipelined INT8 GEMM for the Pi0.5 decoder projections.
//
// At ten action rows every projection is weight-bandwidth bound, so the whole
// job is to keep the weight stream saturated. A bare load loop on these cores
// fills L1 at 706 GB/s and the vendor BF16 matmul runs within 2.6 us of that
// floor, while the Matmul API's INT8 path leaves 5.2 us on the table at the
// gate/up shape. This owns the pipeline directly: nd2nz straight into L1, one
// 2D load into L0, Mmad, and the per-output-channel dequant applied by fixpipe
// on the way out as FP16. Nothing touches the vector cores.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
using namespace AscendC;

// Fixpipe writes the result rows in NT-wide pieces. Two cores writing pieces
// that fall in the same aligned region of GM is not safe: the frame's replays
// then disagree with each other, while the same kernel run on an idle device
// does not. Tiles are handed out in groups whose combined row write covers one
// whole region, so a region belongs to exactly one core.
constexpr int WRITE_GRAIN = 512;

__global__ __aicore__ void decoder_gemm_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR deq, GM_ADDR c,
                                           int M, int N, int K, int NT, int KT,
                                           int srcStride, int grain, int nzout) {
    if ASCEND_IS_AIV { return; }
    const int M16 = (M + 15) / 16 * 16;
    const int ksteps = K / KT;
    TPipe pipe;
    TQue<TPosition::A1, 1> a1q;
    TQue<TPosition::A2, 1> a2q;
    TQue<TPosition::B1, 2> b1q;
    TQue<TPosition::B2, 2> b2q;
    TQue<TPosition::CO1, 1> coq;
    TBuf<TPosition::C1> dbuf;
    pipe.InitBuffer(a1q, 1, M16 * K);
    pipe.InitBuffer(a2q, 1, M16 * K);
    pipe.InitBuffer(b1q, 2, NT * KT);
    pipe.InitBuffer(b2q, 2, NT * KT);
    pipe.InitBuffer(coq, 1, M16 * NT * 4);
    pipe.InitBuffer(dbuf, N * 8);

    GlobalTensor<int8_t> ag, bg;
    GlobalTensor<half> cg;
    GlobalTensor<uint64_t> dg;
    ag.SetGlobalBuffer((__gm__ int8_t*)a);
    bg.SetGlobalBuffer((__gm__ int8_t*)b);
    cg.SetGlobalBuffer((__gm__ half*)c);
    dg.SetGlobalBuffer((__gm__ uint64_t*)deq);

    // A and the dequant vector are both small, both cold on the first call of a
    // step, and both only needed after the first weight tile is already moving.
    // Issue them, then let the B stream start before waiting on either: in the
    // frame these two round trips sat in front of the whole weight load.
    auto a1 = a1q.AllocTensor<int8_t>();
    DataCopy(a1, ag, Nd2NzParams{1, (uint16_t)M, (uint16_t)K, 0, (uint16_t)K,
                                 (uint16_t)M16, 1, 0});
    a1q.EnQue(a1);
    // The whole dequant vector is staged once: a per-tile reload would race the
    // fixpipe still reading the previous tile's scales, which is silent and
    // shows up as a moderate error spread across two thirds of the columns.
    auto dl = dbuf.Get<uint64_t>();
    DataCopy(dl, dg, N);
    SetFlag<HardEvent::MTE2_FIX>(EVENT_ID0);
    LocalTensor<int8_t> a2;
    bool primed = false;
    const int tiles = (N + NT - 1) / NT;
    // In NZ the whole tile lands as one contiguous aligned block, so a core
    // never shares a region with another and every tile can be handed out.
    const int width = nzout ? M16 * NT * 2 : NT * 2;
    int group = grain > width ? grain / width : 1;
    if (group < 1) { group = 1; }
    const int groups = (tiles + group - 1) / group;
    for (int gi = GetBlockIdx(); gi < groups; gi += GetBlockNum()) {
    const int gbase = gi * group;
    const int gend = gbase + group < tiles ? gbase + group : tiles;
    for (int tile = gbase; tile < gend; ++tile) {
        const int n0 = tile * NT;
        auto co = coq.AllocTensor<int32_t>();
        for (int j = 0; j < ksteps; ++j) {
            auto b1 = b1q.AllocTensor<int8_t>();
            DataCopy(b1, bg[n0 * K + j * KT],
                     Nd2NzParams{1, (uint16_t)NT, (uint16_t)KT, 0, (uint16_t)K,
                                 (uint16_t)NT, 1, 0});
            b1q.EnQue(b1);
            if (!primed) {
                a1 = a1q.DeQue<int8_t>();
                a2 = a2q.AllocTensor<int8_t>();
                LoadData(a2, a1, LoadData2DParams{0, (uint8_t)((M16 / 16) * (K / 32)), 1, 0,
                                                  0, false, 0});
                a2q.EnQue(a2);
                a2 = a2q.DeQue<int8_t>();
                a1q.FreeTensor(a1);
                WaitFlag<HardEvent::MTE2_FIX>(EVENT_ID0);
                primed = true;
            }
            b1 = b1q.DeQue<int8_t>();
            auto b2 = b2q.AllocTensor<int8_t>();
            LoadData(b2, b1, LoadData2DParams{0, (uint8_t)((NT / 16) * (KT / 32)), 1, 0, 0,
                                              false, 0});
            b2q.EnQue(b2);
            b2 = b2q.DeQue<int8_t>();
            b1q.FreeTensor(b1);
            MmadParams mp;
            mp.m = (uint16_t)M;
            mp.n = (uint16_t)NT;
            mp.k = (uint16_t)KT;
            mp.cmatrixInitVal = (j == 0);
            Mmad(co, a2[j * KT * M16], b2, mp);
            b2q.FreeTensor(b2);
        }
        coq.EnQue(co);
        co = coq.DeQue<int32_t>();
        FixpipeParamsV220 fp;
        fp.nSize = (uint16_t)NT;
        fp.mSize = (uint16_t)M;
        fp.srcStride = (uint16_t)srcStride;
        fp.quantPre = QuantMode_t::VDEQF16;
        fp.ndNum = 1;
        if (nzout) {
            // A fractal-NZ block holds sixteen rows interleaved at 32-byte
            // granularity, so writing only the M live ones threads gaps through
            // it. That partial write is not safe between cores: with every
            // projection running, a captured frame stops replaying to one answer
            // -- 38 of 300 replays disagree, in 19 different ways. Writing the
            // whole block is exact 300 of 300 and costs nothing measurable. The
            // rows past M come from an A operand that was never loaded and are
            // never gathered by the consumer, which walks rows 0..M-1.
            fp.mSize = (uint16_t)M16;
            fp.dstStride = (uint32_t)M16;
            Fixpipe<half, int32_t, CFG_NZ>(cg[(uint32_t)tile * M16 * NT], co, dl[n0], fp);
        } else {
            fp.dstStride = (uint32_t)N;
            Fixpipe<half, int32_t, CFG_ROW_MAJOR>(cg[n0], co, dl[n0], fp);
        }
        coq.FreeTensor(co);
        // The accumulator is single buffered, so the next tile's first Mmad has
        // to see the fixpipe drain. Without this the results are launch-order
        // dependent, and a narrower column tile faults outright.
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
    }
    // Drain the fixpipe before the kernel returns. Standalone the host sync
    // hides an undrained write; in a captured frame the next kernel reads the
    // buffer straight away and two replays of the same graph disagree.
    SetFlag<HardEvent::FIX_S>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
    if (primed) {
        a2q.FreeTensor(a2);
    } else {
        auto drain = a1q.DeQue<int8_t>();
        a1q.FreeTensor(drain);
        WaitFlag<HardEvent::MTE2_FIX>(EVENT_ID0);
    }
}

extern "C" int flashrt_npu_decoder_gemm(void* stream, void* a, void* b, void* deq, void* c,
                                    int M, int N, int K, int NT, int KT, int srcStride,
                                    int cores, int grain, int nzout) {
    if (!stream || !a || !b || !deq || !c) { return 1; }
    if (M <= 0 || M > 256 || N <= 0 || N % 16 || K <= 0 || K % 32) { return 2; }
    if (NT <= 0 || NT % 16 || NT > N || KT <= 0 || KT % 32 || K % KT) { return 3; }
    if ((long long)NT * KT > 32768 || cores <= 0 || cores > 20) { return 4; }
    // A K-split only accumulates correctly when the L0B slot is exactly half the
    // buffer; a smaller slot silently drops part of the second step.
    if (K / KT > 1 && NT * KT != 32768) { return 5; }
    decoder_gemm_kernel<<<cores, nullptr, stream>>>((uint8_t*)a, (uint8_t*)b, (uint8_t*)deq,
        (uint8_t*)c, M, N, K, NT, KT, srcStride,
        grain > 0 ? grain : WRITE_GRAIN, nzout);
    return 0;
}

#include "../abi.h"
