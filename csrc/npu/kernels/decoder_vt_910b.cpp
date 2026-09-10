// Fractal-NZ transposed value cache for the action decoder's cross attention.
//
// A raw Mmad B operand wants its GM source in (N, K) form. For O = P * V that
// is V as (HD, KV), which the cache does not hold. Storing the transpose as
// plain (HD, KV) would make each step's ten new value columns 256 scattered
// two-byte writes; storing it in fractal NZ instead -- element (hd, pos) at
// (pos/16)*HD*16 + hd*16 + (pos%16) -- turns one sixteen-position block into
// 8 KB of contiguous bytes, and makes the attention kernel's B tile a plain
// contiguous copy with no Nd2Nz conversion at all.
//
// Sixteen positions per block is also why the cache keeps the action suffix
// first and the encoder prefix from column 16: the prefix is 572 rows at the
// shipped geometry, so a suffix written after it would straddle two blocks and
// have to thread its ten columns through twelve prefix ones at 32-byte
// granularity. That is the partial fractal write this campaign already paid
// for once. Attention does not care in what order the keys arrive, so the
// suffix takes block zero whole and no core ever writes part of a block.
//
// `Transpose` accepts int16_t, uint16_t and half only. A transpose moves bits,
// so bfloat16 rides through it reinterpreted as int16_t, bit for bit.
#include "kernel_operator.h"
using namespace AscendC;

namespace flashrt_vt {
constexpr int PB = 16;              // positions in a fractal block
constexpr int HD = 256;             // head dimension
constexpr int GRP = HD / PB;        // 16 hd groups of 16
constexpr uint16_t ROWGAP = (HD * 2 - 32) / 32;   // 32-byte units
}

// One layer of encoder prefix values, (rows, HD) bf16, into the NZ transpose
// starting at column `cb`. Runs once a frame per layer, in place of the
// straight copy the row-major cache used to take.
__global__ __aicore__ void vt_prefix_kernel(GM_ADDR src, GM_ADDR dst, int rows, int cb) {
    using namespace flashrt_vt;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inq;
    TBuf<QuePosition::VECCALC> tbuf, obuf;
    pipe.InitBuffer(inq, 1, PB * HD * 2);
    pipe.InitBuffer(tbuf, PB * PB * 2);
    pipe.InitBuffer(obuf, PB * HD * 2);
    GlobalTensor<int16_t> sg, dg;
    sg.SetGlobalBuffer((__gm__ int16_t*)src);
    dg.SetGlobalBuffer((__gm__ int16_t*)dst);
    auto tile = tbuf.Get<int16_t>();
    auto out = obuf.Get<int16_t>();
    const int nblk = (rows + PB - 1) / PB;
    for (int j = GetBlockIdx(); j < nblk; j += GetBlockNum()) {
        const int rest = rows - j * PB;
        const int live = rest < PB ? rest : PB;
        auto in = inq.AllocTensor<int16_t>();
        DataCopy(in, sg[(uint32_t)j * PB * HD], live * HD);
        inq.EnQue(in);
        in = inq.DeQue<int16_t>();
        // The tail of a short last block has to be zero: it becomes padded
        // columns, which score zero and must then contribute zero value.
        if (live < PB) { Duplicate<int16_t>(in[live * HD], 0, (PB - live) * HD); }
        PipeBarrier<PIPE_ALL>();
        for (int g = 0; g < GRP; ++g) {
            DataCopy(tile, in[g * PB], DataCopyParams{(uint16_t)PB, 1, ROWGAP, 0});
            PipeBarrier<PIPE_ALL>();
            Transpose(out[g * PB * PB], tile);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(dg[(uint32_t)(cb / PB + j) * HD * PB], out, HD * PB);
        inq.FreeTensor(in);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int flashrt_npu_vt_prefix(void* stream, void* src, void* dst, int rows, int cb) {
    using namespace flashrt_vt;
    if (!stream || !src || !dst || rows <= 0 || rows > 65536 || cb < 0 || cb % PB) { return 1; }
    vt_prefix_kernel<<<40, nullptr, stream>>>((uint8_t*)src, (uint8_t*)dst, rows, cb);
    return 0;
}

// Decoder rotary for the transposed cache. Same rotation and the same FP16
// input slab as the shipped kernel; what changes is where the two cache
// halves land. Keys go to rows [0, rows) because the suffix now leads the
// cache, and values are transposed into fractal block zero. The rotary
// position is still the real one, `pos_base + row`, so the numbers are
// unchanged -- only the seat they take in the cache is.
__global__ __aicore__ void decoder_rope_vt_kernel(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR keys, GM_ADDR vnz, int pos_base, int rows) {
    using namespace flashrt_vt;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> iq, cq, sq;
    TQue<QuePosition::VECOUT, 1> oq;
    TBuf<QuePosition::VECCALC> xfbuf, yfbuf, tmpbuf, tbuf, vbuf;
    pipe.InitBuffer(iq, 1, 512); pipe.InitBuffer(cq, 1, 1024);
    pipe.InitBuffer(sq, 1, 1024); pipe.InitBuffer(oq, 1, 512);
    pipe.InitBuffer(xfbuf, 1024); pipe.InitBuffer(yfbuf, 1024); pipe.InitBuffer(tmpbuf, 1024);
    pipe.InitBuffer(tbuf, PB * PB * 2); pipe.InitBuffer(vbuf, PB * PB * 2);
    GlobalTensor<half> xg;
    GlobalTensor<bfloat16_t> qg, kg;
    GlobalTensor<int16_t> vg;
    GlobalTensor<float> cg, sg;
    xg.SetGlobalBuffer((__gm__ half*)qkv);
    qg.SetGlobalBuffer((__gm__ bfloat16_t*)query);
    kg.SetGlobalBuffer((__gm__ bfloat16_t*)keys);
    vg.SetGlobalBuffer((__gm__ int16_t*)vnz);
    cg.SetGlobalBuffer((__gm__ float*)cos); sg.SetGlobalBuffer((__gm__ float*)sin);
    const int nqk = rows * 9;
    for (int unit = GetBlockIdx(); unit < nqk + GRP; unit += GetBlockNum()) {
        if (unit < nqk) {
            const int row = unit / 9, head = unit % 9;
            auto x = iq.AllocTensor<half>();
            DataCopy(x, xg[row * 2560 + head * 256], 256);
            iq.EnQue(x);
            x = iq.DeQue<half>();
            auto y = oq.AllocTensor<bfloat16_t>();
            auto xf = xfbuf.Get<float>(); auto yf = yfbuf.Get<float>();
            auto tmp = tmpbuf.Get<float>();
            Cast(xf, x, RoundMode::CAST_NONE, 256); PipeBarrier<PIPE_V>();
            auto c = cq.AllocTensor<float>(); auto s = sq.AllocTensor<float>();
            DataCopy(c, cg[(pos_base + row) * 256], 256);
            DataCopy(s, sg[(pos_base + row) * 256], 256);
            cq.EnQue(c); sq.EnQue(s); c = cq.DeQue<float>(); s = sq.DeQue<float>();
            Mul(yf, xf, c, 256); PipeBarrier<PIPE_V>();
            Mul(tmp, xf[128], s, 128); PipeBarrier<PIPE_V>();
            Sub(yf, yf, tmp, 128); PipeBarrier<PIPE_V>();
            Mul(tmp, xf, s[128], 128); PipeBarrier<PIPE_V>();
            Add(yf[128], yf[128], tmp, 128); PipeBarrier<PIPE_V>();
            Cast(y, yf, RoundMode::CAST_RINT, 256);
            cq.FreeTensor(c); sq.FreeTensor(s);
            oq.EnQue(y); iq.FreeTensor(x); y = oq.DeQue<bfloat16_t>();
            if (head < 8) { DataCopy(qg[row * 2048 + head * 256], y, 256); }
            else { DataCopy(kg[row * 256], y, 256); }
            oq.FreeTensor(y);
        } else {
            // One hd group of sixteen: gather this group's column out of every
            // action row, transpose it, and write the 512 bytes it owns of
            // block zero. Sixteen cores tile the block and each writes a whole
            // aligned 512-byte run, so no core writes part of another's.
            const int g = unit - nqk;
            auto x = iq.AllocTensor<half>();
            DataCopy(x, xg[2304 + g * PB],
                     DataCopyParams{(uint16_t)rows, 1, (uint16_t)((2560 * 2 - 32) / 32), 0});
            iq.EnQue(x);
            x = iq.DeQue<half>();
            if (rows < PB) { Duplicate<half>(x[rows * PB], (half)0, (PB - rows) * PB); }
            auto xf = xfbuf.Get<float>();
            auto v = vbuf.Get<bfloat16_t>();
            auto tile = tbuf.Get<int16_t>();
            PipeBarrier<PIPE_V>();
            Cast(xf, x, RoundMode::CAST_NONE, PB * PB); PipeBarrier<PIPE_V>();
            Cast(v, xf, RoundMode::CAST_RINT, PB * PB); PipeBarrier<PIPE_V>();
            Transpose(tile, v.template ReinterpretCast<int16_t>());
            iq.FreeTensor(x);
            PipeBarrier<PIPE_ALL>();
            DataCopy(vg[g * PB * PB], tile, PB * PB);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

extern "C" int flashrt_npu_decoder_rope_vt(void* stream, void* qkv, void* cos, void* sin,
    void* query, void* keys, void* vnz, int pos_base, int rows) {
    using namespace flashrt_vt;
    if (!stream || !qkv || !cos || !sin || !query || !keys || !vnz) { return 1; }
    if (pos_base < 0 || rows <= 0 || rows > PB) { return 2; }
    if (pos_base > 2147483647 / 256 - rows) { return 3; }
    decoder_rope_vt_kernel<<<40, nullptr, stream>>>((uint8_t*)qkv, (uint8_t*)cos, (uint8_t*)sin,
        (uint8_t*)query, (uint8_t*)keys, (uint8_t*)vnz, pos_base, rows);
    return 0;
}
