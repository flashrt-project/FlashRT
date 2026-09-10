// Shared-KV decode attention for the Pi0.5 action expert, on the raw cube path.
//
// The action expert is MQA with one KV head and every query row in the chunk
// attends over the same key range, so a layer-step is two dense GEMMs with a
// softmax between them:
//
//     S = Q(Mq,HD) * K(KV,HD)^T ;  P = softmax(S*scale) ;  O = P * V(KV,HD)
//
// Mq is chunk*heads, five 16-row fractal blocks at the shipped geometry, so the
// work splits over query rows and a core owns its rows end to end.
//
// Both GEMMs drive Mmad directly. A raw B operand wants its GM source in (N,K)
// form -- measured, not assumed -- which K already is as the cache holds it,
// and which V is only if V is stored transposed. The cache keeps that
// transpose in fractal NZ (see decoder_vt_910b.cpp), which makes the value
// tile for a key block plain contiguous bytes: no Nd2Nz conversion on the
// operand that dominates this kernel's reads.
//
// Because the transposed cache seats the action suffix first, the live keys
// are two ranges -- [0, lo) for the suffix and [hi, end) for the encoder
// prefix -- with the gap between them and the tail past `end` padding. The
// padded columns are not masked. Both cache halves are zero there, so a padded
// column scores zero, takes a finite share of the softmax, and then multiplies
// a zero value column into the output. The row max and the row sum are taken
// over the live ranges only, so the live probabilities are the correct ones.
#include "kernel_operator.h"
using namespace AscendC;

namespace flashrt_attn {
constexpr int MBLK = 16;   // fractal row block
constexpr int HD = 256;    // head dimension
// One L0B slot must be exactly half the buffer for a K-split to accumulate:
// a smaller slot silently drops part of a later step. At bf16 that fixes the
// key tile, since HD * NT * 2 == 32768.
constexpr int NT = 64;
constexpr int SLOT = HD * NT * 2;
}

__global__ __aicore__ void decode_attn_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR vnz, GM_ADDR out,
                                              GM_ADDR scores, GM_ADDR probs, GM_ADDR ctx,
                                              int lo, int hi, int end, int kvp,
                                              float scale, uint64_t sync) {
    using namespace flashrt_attn;
    SetSyncBaseAddr(sync);
    TPipe pipe;
    const int blk = GetBlockIdx() / GetTaskRation();
    const int srow = blk * MBLK;
    const int tiles = kvp / NT;

    GlobalTensor<bfloat16_t> qg, kg, vg, pg, og;
    GlobalTensor<float> sg, cg;
    qg.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    kg.SetGlobalBuffer((__gm__ bfloat16_t*)k);
    vg.SetGlobalBuffer((__gm__ bfloat16_t*)vnz);
    pg.SetGlobalBuffer((__gm__ bfloat16_t*)probs);
    og.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    sg.SetGlobalBuffer((__gm__ float*)scores);
    cg.SetGlobalBuffer((__gm__ float*)ctx);

    if ASCEND_IS_AIC {
        TQue<TPosition::A1, 1> qa1;
        TQue<TPosition::A2, 1> qa2;
        TQue<TPosition::A1, 2> pa1;
        TQue<TPosition::A2, 2> pa2;
        TQue<TPosition::B1, 2> b1q;
        TQue<TPosition::B2, 2> b2q;
        TQue<TPosition::CO1, 1> coq;
        pipe.InitBuffer(qa1, 1, MBLK * HD * 2);
        pipe.InitBuffer(qa2, 1, MBLK * HD * 2);
        pipe.InitBuffer(pa1, 2, MBLK * NT * 2);
        pipe.InitBuffer(pa2, 2, MBLK * NT * 2);
        pipe.InitBuffer(b1q, 2, SLOT);
        pipe.InitBuffer(b2q, 2, SLOT);
        pipe.InitBuffer(coq, 1, MBLK * HD * 4);

        // Q is (MBLK, HD) row major and stays in L0A for the whole score pass.
        auto q1 = qa1.AllocTensor<bfloat16_t>();
        DataCopy(q1, qg[srow * HD], Nd2NzParams{1, (uint16_t)MBLK, (uint16_t)HD, 0,
                                                (uint16_t)HD, (uint16_t)MBLK, 1, 0});
        qa1.EnQue(q1);
        q1 = qa1.DeQue<bfloat16_t>();
        auto q2 = qa2.AllocTensor<bfloat16_t>();
        LoadData(q2, q1, LoadData2DParams{0, (uint8_t)(HD / 16), 1, 0, 0, false, 0});
        qa2.EnQue(q2);
        q2 = qa2.DeQue<bfloat16_t>();
        qa1.FreeTensor(q1);

        for (int t = 0; t < tiles; ++t) {
            auto b1 = b1q.AllocTensor<bfloat16_t>();
            DataCopy(b1, kg[(uint32_t)t * NT * HD],
                     Nd2NzParams{1, (uint16_t)NT, (uint16_t)HD, 0, (uint16_t)HD,
                                 (uint16_t)NT, 1, 0});
            b1q.EnQue(b1);
            b1 = b1q.DeQue<bfloat16_t>();
            auto b2 = b2q.AllocTensor<bfloat16_t>();
            LoadData(b2, b1, LoadData2DParams{0, (uint8_t)((NT / 16) * (HD / 16)), 1, 0,
                                              0, false, 0});
            b2q.EnQue(b2);
            b2 = b2q.DeQue<bfloat16_t>();
            b1q.FreeTensor(b1);
            auto co = coq.AllocTensor<float>();
            MmadParams mp;
            mp.m = (uint16_t)MBLK;
            mp.n = (uint16_t)NT;
            mp.k = (uint16_t)HD;
            mp.cmatrixInitVal = true;
            Mmad(co, q2, b2, mp);
            b2q.FreeTensor(b2);
            coq.EnQue(co);
            co = coq.DeQue<float>();
            FixpipeParamsV220 fp;
            fp.nSize = (uint16_t)NT;
            fp.mSize = (uint16_t)MBLK;
            fp.srcStride = (uint16_t)MBLK;
            fp.dstStride = (uint32_t)kvp;
            fp.quantPre = QuantMode_t::NoQuant;
            fp.ndNum = 1;
            Fixpipe<float, float, CFG_ROW_MAJOR>(sg[(uint32_t)srow * kvp + t * NT], co, fp);
            coq.FreeTensor(co);
            // The accumulator is single buffered: the next tile's Mmad has to
            // see this fixpipe drain first.
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        qa2.FreeTensor(q2);
        SetFlag<HardEvent::FIX_S>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
        NotifyEvent<PIPE_FIX>(4);

        WaitEvent(5);
        auto oc = coq.AllocTensor<float>();
        for (int t = 0; t < tiles; ++t) {
            auto p1 = pa1.AllocTensor<bfloat16_t>();
            DataCopy(p1, pg[(uint32_t)srow * kvp + t * NT],
                     Nd2NzParams{1, (uint16_t)MBLK, (uint16_t)NT, 0, (uint16_t)kvp,
                                 (uint16_t)MBLK, 1, 0});
            pa1.EnQue(p1);
            p1 = pa1.DeQue<bfloat16_t>();
            auto p2 = pa2.AllocTensor<bfloat16_t>();
            LoadData(p2, p1, LoadData2DParams{0, (uint8_t)(NT / 16), 1, 0, 0, false, 0});
            pa2.EnQue(p2);
            p2 = pa2.DeQue<bfloat16_t>();
            pa1.FreeTensor(p1);
            // The cache already holds the NZ transpose of V, and NT is four
            // whole 16-column blocks, so this tile is exactly the bytes the
            // B operand wants, in order, with nothing to convert.
            auto b1 = b1q.AllocTensor<bfloat16_t>();
            DataCopy(b1, vg[(uint32_t)t * NT * HD], NT * HD);
            b1q.EnQue(b1);
            b1 = b1q.DeQue<bfloat16_t>();
            auto b2 = b2q.AllocTensor<bfloat16_t>();
            LoadData(b2, b1, LoadData2DParams{0, (uint8_t)((HD / 16) * (NT / 16)), 1, 0,
                                              0, false, 0});
            b2q.EnQue(b2);
            b2 = b2q.DeQue<bfloat16_t>();
            b1q.FreeTensor(b1);
            MmadParams mp;
            mp.m = (uint16_t)MBLK;
            mp.n = (uint16_t)HD;
            mp.k = (uint16_t)NT;
            mp.cmatrixInitVal = (t == 0);
            Mmad(oc, p2, b2, mp);
            pa2.FreeTensor(p2);
            b2q.FreeTensor(b2);
        }
        coq.EnQue(oc);
        oc = coq.DeQue<float>();
        FixpipeParamsV220 fo;
        fo.nSize = (uint16_t)HD;
        fo.mSize = (uint16_t)MBLK;
        fo.srcStride = (uint16_t)MBLK;
        fo.dstStride = (uint32_t)HD;
        fo.quantPre = QuantMode_t::NoQuant;
        fo.ndNum = 1;
        Fixpipe<float, float, CFG_ROW_MAJOR>(cg[(uint32_t)srow * HD], oc, fo);
        coq.FreeTensor(oc);
        SetFlag<HardEvent::FIX_S>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
        NotifyEvent<PIPE_FIX>(6);
        WaitEvent(7);
    }

    if ASCEND_IS_AIV {
        TQue<TPosition::VECIN, 1> inq;
        TQue<TPosition::VECOUT, 1> outq;
        TBuf<TPosition::VECCALC> wbuf, rbuf;
        pipe.InitBuffer(inq, 1, 8192 * 4);
        pipe.InitBuffer(outq, 1, 8192 * 2);
        pipe.InitBuffer(wbuf, 4096);
        pipe.InitBuffer(rbuf, 256);
        const int sub = GetSubBlockIdx();
        auto work = wbuf.Get<float>();
        auto red = rbuf.Get<float>();
        const auto evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const int pref = end - hi;

        WaitEvent(4);
        for (int r = sub; r < MBLK; r += 2) {
            auto in = inq.AllocTensor<float>();
            DataCopy(in, sg[(uint32_t)(srow + r) * kvp], kvp);
            inq.EnQue(in);
            in = inq.DeQue<float>();
            Muls(in, in, scale, kvp);
            PipeBarrier<PIPE_V>();
            // Reading a reduction on the scalar unit is a vector-to-scalar
            // hazard that PipeBarrier does not cover: without the flag the max
            // comes back stale, the exponential overflows and the row is NaN.
            // Both reductions run over the two live ranges only, so neither
            // the gap between them nor the padded tail enters max or sum.
            // The two results land in separate 32-byte slots so one flag
            // covers both reads.
            ReduceMax<float>(red, in, work, lo, false);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(red[8], in[hi], work, pref, false);
            SetFlag<HardEvent::V_S>(evVS);
            WaitFlag<HardEvent::V_S>(evVS);
            const float m0 = red.GetValue(0), m1 = red.GetValue(8);
            const float rowmax = m0 > m1 ? m0 : m1;
            Adds(in, in, -rowmax, kvp);
            PipeBarrier<PIPE_V>();
            Exp(in, in, kvp);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(red[16], in, work, lo);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(red[24], in[hi], work, pref);
            SetFlag<HardEvent::V_S>(evVS);
            WaitFlag<HardEvent::V_S>(evVS);
            const float rowsum = red.GetValue(16) + red.GetValue(24);
            Muls(in, in, 1.0f / rowsum, kvp);
            PipeBarrier<PIPE_V>();
            auto p = outq.AllocTensor<bfloat16_t>();
            Cast(p, in, RoundMode::CAST_RINT, kvp);
            PipeBarrier<PIPE_V>();
            outq.EnQue(p);
            inq.FreeTensor(in);
            p = outq.DeQue<bfloat16_t>();
            DataCopy(pg[(uint32_t)(srow + r) * kvp], p, kvp);
            outq.FreeTensor(p);
        }
        NotifyEvent<PIPE_MTE3>(5);

        WaitEvent(6);
        const int n = (MBLK / 2) * HD;
        auto in = inq.AllocTensor<float>();
        DataCopy(in, cg[(uint32_t)srow * HD + sub * n], n);
        inq.EnQue(in);
        in = inq.DeQue<float>();
        auto o = outq.AllocTensor<bfloat16_t>();
        Cast(o, in, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        outq.EnQue(o);
        inq.FreeTensor(in);
        o = outq.DeQue<bfloat16_t>();
        DataCopy(og[(uint32_t)srow * HD + sub * n], o, n);
        outq.FreeTensor(o);
        NotifyEvent<PIPE_MTE3>(7);
    }
}

extern "C" int rtGetC2cCtrlAddr(uint64_t*, uint32_t*);

extern "C" int flashrt_npu_decode_attn(void* stream, void* q, void* k, void* vnz, void* out,
                                       void* scores, void* probs, void* ctx,
                                       int mq, int lo, int hi, int end, int kvp, float scale) {
    using namespace flashrt_attn;
    if (!stream || !q || !k || !vnz || !out || !scores || !probs || !ctx) { return 1; }
    if (mq <= 0 || mq % MBLK || mq > 2048) { return 2; }
    if (kvp <= 0 || kvp % NT || kvp > 8192) { return 3; }
    if (lo <= 0 || hi < lo || hi % 16 || end <= hi || end > kvp) { return 4; }
    uint64_t sync = 0; uint32_t len = 0;
    int rc = rtGetC2cCtrlAddr(&sync, &len);
    if (rc) { return rc; }
    decode_attn_kernel<<<mq / MBLK, nullptr, stream>>>(
        (uint8_t*)q, (uint8_t*)k, (uint8_t*)vnz, (uint8_t*)out, (uint8_t*)scores,
        (uint8_t*)probs, (uint8_t*)ctx, lo, hi, end, kvp, scale, sync);
    return 0;
}
