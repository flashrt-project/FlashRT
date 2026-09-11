// Multi-head attention for the GR00T N1.7 DiT, on the raw cube path.
//
// The DiT's three attention geometries are all tiny: 41 query rows against 41,
// 13 or 448 keys, over 32 heads of 48 channels. The vendor prompt
// flash-attention operator charges 42 to 49 us for every one of them -- the
// cost barely moves between 13 keys and 448 -- while the 41x41 case moves
// 378 KB and does 10 MFLOP, which is about 10 us of launch and a microsecond of
// work. That gap, 128 times a frame, is what this kernel exists to close.
//
// One head per iteration, and the whole head fits in L0 without tiling:
//
//     S = Q(Sq,HD) * K(Skv,HD)^T ;  P = softmax(S*scale) ;  O = P * V(Skv,HD)
//
// Both GEMMs drive Mmad directly. A raw B operand takes its L1 source in (N,K)
// form, which K already is as the projection writes it, and which V is only if
// V arrives transposed -- so the caller hands V in as (heads*HD, Skv). That
// costs the caller nothing where V is a frame constant, which is every
// cross-attention layer.
//
// Every extent the cube sees is 16-aligned and every byte of that alignment is
// a real zero the caller wrote. That is not fussiness: Nd2Nz fills only the
// rows it is given, so a buffer padded to the fractal by the *kernel* leaves
// uninitialised L1 in the tail, which the Mmad then reads. So the caller hands
// in Q padded to Sq16 rows, K padded to Skv16 rows and V^T padded to Skv16
// columns, all zero past the live extent, and the probability plane is
// allocated zero and only ever written over the live keys.
//
// Padding is then arithmetic rather than masking: a zero key scores zero, takes
// a finite share of the softmax, and multiplies a zero value column into the
// output. The row max and row sum run over the live keys only, so the live
// probabilities are the correct ones -- the same argument the Pi0.5 decode
// attention makes for the gap in its cache.
//
// The two phases are separated by one cross-core handoff each, not one per
// head: a core computes every score plane it owns, hands them all to the
// vector cores at once, and takes back every probability plane at once. With
// two heads to a core that halves the number of round trips, and neither head
// waits on the other's softmax.
//
// The softmax takes a whole head's score plane at once and never reads a
// reduction on the scalar unit. Doing it a row at a time is the obvious way and
// it costs 13 us a head: each row pays two reduction-then-GetValue round trips,
// each of which has to be guarded by a vector-to-scalar flag, and 41 rows of
// that is the entire kernel. The plane form replaces 41 of those with one
// WholeReduce per reduction, one Brcb, and a broadcast whose operand advances a
// block a row -- about a dozen instructions for the head instead of four
// hundred.
#include "kernel_operator.h"
using namespace AscendC;

namespace flashrt_dit_attn {
constexpr int MBLK = 16;          // fractal row block
constexpr int MAX_HD = 128;       // head width ceiling, for the L0A/L0B budget
constexpr int MAX_SKV = 512;      // key ceiling: Sq16 * MAX_SKV * 4 fits L0C
constexpr int LANES = 64;         // FP32 elements one vector repeat can touch
constexpr int MAX_PLANE = 24576;  // score plane a vector core holds, in elements
constexpr int MAX_ROWS = 64;      // query rows, so one repeat argument covers them

// The row softmax, over a whole (rows, width) plane, with no scalar reads.
//
// The fractal padding is carried through the arithmetic rather than masked out
// of it. Masking is what a per-row loop can afford; a fold reads whole repeats,
// and writing a sentinel into the tail is not even expressible because the live
// width is rarely on a 32-byte boundary. A padded column holds a zero score, so
// after the shift it holds exp(-rowmax) exactly, and its share of the row sum
// is the closed form subtracted below -- the probabilities it leaves are
// finite and multiply a zero value column, which is the same argument the
// padding already relies on.
//
//   plane  (rows, width) FP32, row stride `width`, live columns [0, live)
//   work   (rows, 64)    FP32 scratch, used only when a row spans repeats
//   red    4 * 64        FP32 scratch
//   bcast  rows * 8      FP32 scratch
__aicore__ inline int ChunkWidth(int width, int c) {
    const int rest = width - c * LANES;
    return rest < LANES ? rest : LANES;
}

__aicore__ inline void RowSoftmax(const LocalTensor<float>& plane,
                                  const LocalTensor<float>& work,
                                  const LocalTensor<float>& red,
                                  const LocalTensor<float>& bcast,
                                  int rows, int width, int live) {
    const uint8_t blocks = (uint8_t)(width / 8);
    const int chunks = (width + LANES - 1) / LANES;
    const uint8_t rep = (uint8_t)rows;
    const uint8_t brep = (uint8_t)((rows + 7) / 8);

    if (chunks == 1) {
        WholeReduceMax<float>(red, plane, width, rep, 1, 1, blocks,
                              ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        // Fold the row to one repeat first: a repeat is 64 FP32 lanes and the
        // mask cannot reach past it, so a 448-wide row has no single-shot
        // reduction. A narrower tail chunk leaves the fold's upper lanes
        // holding chunk zero, which is live data, so the fold stays correct.
        Adds<float>(work, plane, 0.0f, LANES, rep, UnaryRepeatParams(1, 1, 8, blocks));
        PipeBarrier<PIPE_V>();
        for (int c = 1; c < chunks; ++c) {
            Max<float>(work, work, plane[c * LANES], ChunkWidth(width, c), rep,
                       BinaryRepeatParams(1, 1, 1, 8, 8, blocks));
            PipeBarrier<PIPE_V>();
        }
        WholeReduceMax<float>(red, work, LANES, rep, 1, 1, 8,
                              ReduceOrder::ORDER_ONLY_VALUE);
    }
    PipeBarrier<PIPE_V>();
    Brcb(bcast, red, brep, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    for (int c = 0; c < chunks; ++c) {
        Sub<float>(plane[c * LANES], plane[c * LANES], bcast, ChunkWidth(width, c), rep,
                   BinaryRepeatParams(1, 1, 0, blocks, blocks, 1));
    }
    PipeBarrier<PIPE_V>();
    Exp<float>(plane, plane, (int32_t)(rows * width));
    PipeBarrier<PIPE_V>();
    if (chunks == 1) {
        WholeReduceSum<float>(red[192], plane, width, rep, 1, 1, blocks);
    } else {
        Adds<float>(work, plane, 0.0f, LANES, rep, UnaryRepeatParams(1, 1, 8, blocks));
        PipeBarrier<PIPE_V>();
        for (int c = 1; c < chunks; ++c) {
            Add<float>(work, work, plane[c * LANES], ChunkWidth(width, c), rep,
                       BinaryRepeatParams(1, 1, 1, 8, 8, blocks));
            PipeBarrier<PIPE_V>();
        }
        WholeReduceSum<float>(red[192], work, LANES, rep, 1, 1, 8);
    }
    PipeBarrier<PIPE_V>();
    if (live < width) {
        Muls<float>(red[64], red, -1.0f, rows);
        PipeBarrier<PIPE_V>();
        Exp<float>(red[64], red[64], rows);
        PipeBarrier<PIPE_V>();
        Muls<float>(red[64], red[64], -(float)(width - live), rows);
        PipeBarrier<PIPE_V>();
        Add<float>(red[192], red[192], red[64], rows);
        PipeBarrier<PIPE_V>();
    }
    // Reciprocal is the hardware approximation, about 1e-3 relative, which is
    // coarser than the BF16 the probabilities are about to be rounded to. One
    // Newton step over `rows` elements costs nothing and removes the question.
    Reciprocal<float>(red[64], red[192], rows);
    PipeBarrier<PIPE_V>();
    Mul<float>(red[128], red[192], red[64], rows);
    PipeBarrier<PIPE_V>();
    Muls<float>(red[128], red[128], -1.0f, rows);
    PipeBarrier<PIPE_V>();
    Adds<float>(red[128], red[128], 2.0f, rows);
    PipeBarrier<PIPE_V>();
    Mul<float>(red[64], red[64], red[128], rows);
    PipeBarrier<PIPE_V>();
    Brcb(bcast, red[64], brep, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    for (int c = 0; c < chunks; ++c) {
        Mul<float>(plane[c * LANES], plane[c * LANES], bcast, ChunkWidth(width, c), rep,
                   BinaryRepeatParams(1, 1, 0, blocks, blocks, 1));
    }
    PipeBarrier<PIPE_V>();
}
}  // namespace flashrt_dit_attn

// One L1 -> L0A load per 16-row block of A.
//
// A fractal-NZ buffer is ordered column-block first: fractal (k, m) sits at
// index k * (M / 16) + m. L0A wants an A operand ordered row-block first. The
// two coincide only when M is exactly 16, which is why a decode kernel whose
// query block is one fractal can load the whole operand with a single call and
// this one cannot: a flat load of every fractal here delivers a permuted A, and
// the result is wrong in a way that still looks like attention.
template <typename T>
__aicore__ inline void LoadA(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                             int m16, int k16, int k) {
    for (int m = 0; m < m16; ++m) {
        LoadData(dst[(uint32_t)m * 16 * k], src,
                 LoadData2DParams{(uint16_t)m, (uint8_t)k16, (uint16_t)m16, 0, 0,
                                  false, 0});
    }
}

__global__ __aicore__ void dit_attn_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR vt, GM_ADDR out,
                                           GM_ADDR scores, GM_ADDR probs, GM_ADDR ctx,
                                           int heads, int sq, int skv, int hd,
                                           int stride, float scale, uint64_t sync) {
    using namespace flashrt_dit_attn;
    SetSyncBaseAddr(sync);
    TPipe pipe;
    const int cores = GetBlockNum();
    const int blk = GetBlockIdx() / GetTaskRation();
    const int sq16 = (sq + MBLK - 1) / MBLK * MBLK;
    const int skv16 = (skv + MBLK - 1) / MBLK * MBLK;
    const int width = heads * hd;

    GlobalTensor<bfloat16_t> qg, kg, vg, pg, og;
    GlobalTensor<float> sg, cg;
    qg.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    kg.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    vg.SetGlobalBuffer((__gm__ bfloat16_t*)vt);
    pg.SetGlobalBuffer((__gm__ bfloat16_t*)probs);
    og.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    sg.SetGlobalBuffer((__gm__ float*)scores);
    cg.SetGlobalBuffer((__gm__ float*)ctx);

    if ASCEND_IS_AIC {
        TQue<TPosition::A1, 1> a1q;
        TQue<TPosition::A2, 1> a2q;
        TQue<TPosition::B1, 1> b1q;
        TQue<TPosition::B2, 1> b2q;
        TQue<TPosition::CO1, 1> coq;
        pipe.InitBuffer(a1q, 1, MAX_SKV * MAX_HD * 2);
        pipe.InitBuffer(a2q, 1, MAX_SKV * MAX_HD * 2);
        pipe.InitBuffer(b1q, 1, MAX_SKV * MAX_HD * 2);
        pipe.InitBuffer(b2q, 1, MAX_SKV * MAX_HD * 2);
        pipe.InitBuffer(coq, 1, MBLK * 3 * MAX_SKV * 4);

        // ---- S = Q * K^T, for every head this core owns ---------------
        for (int h = blk; h < heads; h += cores) {
            const uint32_t sbase = (uint32_t)h * sq16 * skv16;
            auto a1 = a1q.AllocTensor<bfloat16_t>();
            // The query and key rows may be a slice of a wider buffer: three
            // projections that share an activation are one GEMM, and one GEMM
            // of 4608 columns costs 26 us where three of 1536 cost 34.
            DataCopy(a1, qg[(uint32_t)h * hd],
                     Nd2NzParams{1, (uint16_t)sq16, (uint16_t)hd, 0, (uint16_t)stride,
                                 (uint16_t)sq16, 1, 0});
            a1q.EnQue(a1);
            a1 = a1q.DeQue<bfloat16_t>();
            auto a2 = a2q.AllocTensor<bfloat16_t>();
            LoadA(a2, a1, sq16 / 16, hd / 16, hd);
            a2q.EnQue(a2);
            a2 = a2q.DeQue<bfloat16_t>();
            a1q.FreeTensor(a1);

            auto b1 = b1q.AllocTensor<bfloat16_t>();
            DataCopy(b1, kg[(uint32_t)h * hd],
                     Nd2NzParams{1, (uint16_t)skv16, (uint16_t)hd, 0, (uint16_t)stride,
                                 (uint16_t)skv16, 1, 0});
            b1q.EnQue(b1);
            b1 = b1q.DeQue<bfloat16_t>();
            auto b2 = b2q.AllocTensor<bfloat16_t>();
            LoadData(b2, b1, LoadData2DParams{0, (uint8_t)((skv16 / 16) * (hd / 16)), 1, 0,
                                              0, false, 0});
            b2q.EnQue(b2);
            b2 = b2q.DeQue<bfloat16_t>();
            b1q.FreeTensor(b1);

            auto co = coq.AllocTensor<float>();
            MmadParams mp;
            mp.m = (uint16_t)sq16;
            mp.n = (uint16_t)skv16;
            mp.k = (uint16_t)hd;
            mp.cmatrixInitVal = true;
            Mmad(co, a2, b2, mp);
            a2q.FreeTensor(a2);
            b2q.FreeTensor(b2);
            coq.EnQue(co);
            co = coq.DeQue<float>();
            FixpipeParamsV220 fp;
            fp.nSize = (uint16_t)skv16;
            fp.mSize = (uint16_t)sq16;
            fp.srcStride = (uint16_t)sq16;
            fp.dstStride = (uint32_t)skv16;
            fp.quantPre = QuantMode_t::NoQuant;
            fp.ndNum = 1;
            Fixpipe<float, float, CFG_ROW_MAJOR>(sg[sbase], co, fp);
            coq.FreeTensor(co);
        }
        SetFlag<HardEvent::FIX_S>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
        NotifyEvent<PIPE_FIX>(4);

        // ---- O = P * V, for every head this core owns -----------------
        WaitEvent(5);
        for (int h = blk; h < heads; h += cores) {
            const uint32_t sbase = (uint32_t)h * sq16 * skv16;
            auto p1 = a1q.AllocTensor<bfloat16_t>();
            DataCopy(p1, pg[sbase],
                     Nd2NzParams{1, (uint16_t)sq16, (uint16_t)skv16, 0, (uint16_t)skv16,
                                 (uint16_t)sq16, 1, 0});
            a1q.EnQue(p1);
            p1 = a1q.DeQue<bfloat16_t>();
            auto p2 = a2q.AllocTensor<bfloat16_t>();
            LoadA(p2, p1, sq16 / 16, skv16 / 16, skv16);
            a2q.EnQue(p2);
            p2 = a2q.DeQue<bfloat16_t>();
            a1q.FreeTensor(p1);

            // V arrives transposed, so this tile is the (N, K) source the B
            // operand wants with nothing to convert.
            auto v1 = b1q.AllocTensor<bfloat16_t>();
            DataCopy(v1, vg[(uint32_t)h * hd * skv16],
                     Nd2NzParams{1, (uint16_t)hd, (uint16_t)skv16, 0, (uint16_t)skv16,
                                 (uint16_t)hd, 1, 0});
            b1q.EnQue(v1);
            v1 = b1q.DeQue<bfloat16_t>();
            auto v2 = b2q.AllocTensor<bfloat16_t>();
            LoadData(v2, v1, LoadData2DParams{0, (uint8_t)((hd / 16) * (skv16 / 16)), 1, 0,
                                              0, false, 0});
            b2q.EnQue(v2);
            v2 = b2q.DeQue<bfloat16_t>();
            b1q.FreeTensor(v1);

            auto oc = coq.AllocTensor<float>();
            MmadParams mo;
            mo.m = (uint16_t)sq16;
            mo.n = (uint16_t)hd;
            mo.k = (uint16_t)skv16;
            mo.cmatrixInitVal = true;
            Mmad(oc, p2, v2, mo);
            a2q.FreeTensor(p2);
            b2q.FreeTensor(v2);
            coq.EnQue(oc);
            oc = coq.DeQue<float>();
            FixpipeParamsV220 fo;
            fo.nSize = (uint16_t)hd;
            fo.mSize = (uint16_t)sq16;
            fo.srcStride = (uint16_t)sq16;
            fo.dstStride = (uint32_t)hd;
            fo.quantPre = QuantMode_t::NoQuant;
            fo.ndNum = 1;
            Fixpipe<float, float, CFG_ROW_MAJOR>(cg[(uint32_t)h * sq16 * hd], oc, fo);
            coq.FreeTensor(oc);
        }
        SetFlag<HardEvent::FIX_S>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
        NotifyEvent<PIPE_FIX>(6);
        WaitEvent(7);
    }

    if ASCEND_IS_AIV {
        // One head to a vector core rather than one row: 32 heads over 16 cube
        // cores is exactly the 32 vector cores, and a whole plane is what makes
        // the reductions vector work instead of scalar traffic.
        TQue<TPosition::VECIN, 1> inq;
        TQue<TPosition::VECOUT, 1> outq;
        TBuf<TPosition::VECCALC> wbuf, rbuf, bbuf;
        pipe.InitBuffer(inq, 1, MAX_PLANE * 4);
        pipe.InitBuffer(outq, 1, MAX_PLANE * 2);
        pipe.InitBuffer(wbuf, MAX_ROWS * LANES * 4);
        pipe.InitBuffer(rbuf, 4 * LANES * 4);
        pipe.InitBuffer(bbuf, MAX_ROWS * 8 * 4);
        auto work = wbuf.Get<float>();
        auto red = rbuf.Get<float>();
        auto bcast = bbuf.Get<float>();
        // The handoff is a core and its own two vector cores, not a barrier
        // across the die, so a vector core may only read the planes the cube
        // core it sits on wrote. Reindexing the heads across all vector cores
        // reads another core's scores before they exist -- which looks like
        // attention, scores 0.5 to 0.9 against the vendor, and changes between
        // runs. So the head map here is the cube's, and the two vector cores of
        // a core divide what that core owns: a head each where it owns two,
        // and the rows of the one head where it owns one.
        const int sub = GetSubBlockIdx();
        const int mine = (heads - blk + cores - 1) / cores;
        const bool byhead = mine >= 2;
        const int half = (sq + 1) / 2;
        const int r0 = byhead ? 0 : sub * half;
        const int rn = byhead ? sq : (sq - r0 < half ? sq - r0 : half);

        WaitEvent(4);
        for (int h = blk, idx = 0; h < heads && rn > 0; h += cores, ++idx) {
            if (byhead && (idx & 1) != sub) { continue; }
            const uint32_t sbase = (uint32_t)h * sq16 * skv16 + (uint32_t)r0 * skv16;
            const uint32_t count = (uint32_t)rn * skv16;
            auto plane = inq.AllocTensor<float>();
            DataCopy(plane, sg[sbase], count);
            inq.EnQue(plane);
            plane = inq.DeQue<float>();
            Muls(plane, plane, scale, (int32_t)count);
            PipeBarrier<PIPE_V>();
            RowSoftmax(plane, work, red, bcast, rn, skv16, skv);
            auto p = outq.AllocTensor<bfloat16_t>();
            Cast(p, plane, RoundMode::CAST_RINT, (int32_t)count);
            PipeBarrier<PIPE_V>();
            outq.EnQue(p);
            inq.FreeTensor(plane);
            p = outq.DeQue<bfloat16_t>();
            // Rows past the live query extent are never written, so the plane
            // the caller allocated zero stays zero there and the second Mmad
            // reads zeros rather than whatever the last geometry left behind.
            DataCopy(pg[sbase], p, count);
            outq.FreeTensor(p);
        }
        NotifyEvent<PIPE_MTE3>(5);

        WaitEvent(6);
        for (int h = blk, idx = 0; h < heads && rn > 0; h += cores, ++idx) {
            if (byhead && (idx & 1) != sub) { continue; }
            const uint32_t count = (uint32_t)rn * hd;
            auto ctx = inq.AllocTensor<float>();
            DataCopy(ctx, cg[(uint32_t)h * sq16 * hd + (uint32_t)r0 * hd], count);
            inq.EnQue(ctx);
            ctx = inq.DeQue<float>();
            auto o = outq.AllocTensor<bfloat16_t>();
            Cast(o, ctx, RoundMode::CAST_RINT, (int32_t)count);
            PipeBarrier<PIPE_V>();
            outq.EnQue(o);
            inq.FreeTensor(ctx);
            o = outq.DeQue<bfloat16_t>();
            // The head's slice of every output row in one strided store: the
            // rows are contiguous here and a head apart there.
            DataCopy(og[(uint32_t)r0 * width + (uint32_t)h * hd], o,
                     DataCopyParams{(uint16_t)rn, (uint16_t)(hd / 16), 0,
                                    (uint16_t)((width - hd) / 16)});
            outq.FreeTensor(o);
        }
        NotifyEvent<PIPE_MTE3>(7);
    }
}

extern "C" int rtGetC2cCtrlAddr(uint64_t*, uint32_t*);

extern "C" int flashrt_npu_dit_attn(void* stream, void* q, void* k, void* vt, void* out,
                                    void* scores, void* probs, void* ctx,
                                    int heads, int sq, int skv, int hd, int cores,
                                    int stride, float scale) {
    using namespace flashrt_dit_attn;
    if (!stream || !q || !k || !vt || !out || !scores || !probs || !ctx) { return 1; }
    if (heads <= 0 || heads > 128 || hd <= 0 || hd % MBLK || hd > MAX_HD) { return 2; }
    if (sq <= 0 || sq > 512 || skv <= 0 || skv > MAX_SKV) { return 3; }
    if (cores <= 0 || cores > 20 || cores > heads) { return 4; }
    if (stride < heads * hd) { return 11; }
    // The whole head lives in L0 untiled, which is what makes this kernel worth
    // writing; a geometry that does not fit is refused rather than silently
    // producing a result the accumulator could not have held.
    const int sq16 = (sq + MBLK - 1) / MBLK * MBLK;
    const int skv16 = (skv + MBLK - 1) / MBLK * MBLK;
    if ((long long)sq16 * skv16 * 4 > 131072) { return 5; }          // L0C
    if ((long long)sq16 * (hd > skv16 ? hd : skv16) * 2 > 65536) { return 6; }  // L0A
    if ((long long)skv16 * hd * 2 > 65536) { return 7; }             // L0B
    // The softmax holds a whole head's score plane in UB and drives its
    // reductions with one repeat argument per row, so both of those have a
    // ceiling too. Every DiT site is 41 rows; a geometry that is not gets a
    // refusal rather than a plane that runs off the end of the buffer.
    if (sq > MAX_ROWS) { return 8; }
    if ((long long)sq * skv16 > MAX_PLANE) { return 9; }               // UB
    uint64_t sync = 0; uint32_t len = 0;
    int rc = rtGetC2cCtrlAddr(&sync, &len);
    if (rc) { return rc; }
    dit_attn_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)q, (uint8_t*)k, (uint8_t*)vt, (uint8_t*)out, (uint8_t*)scores,
        (uint8_t*)probs, (uint8_t*)ctx, heads, sq, skv, hd, stride, scale, sync);
    return 0;
}

#include "../abi.h"
