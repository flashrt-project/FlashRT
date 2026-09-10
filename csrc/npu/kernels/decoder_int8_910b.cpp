// Producer- and consumer-side kernels for the INT8 decoder projections.
//
// An INT8 projection only pays if nothing is added to the dependent path, so
// the quantise rides in the kernel that already writes the activation and the
// dequantised FP16 the cube emits is consumed where it lands. These are
// siblings of the shipped vector kernels, which stay as they are: the gated
// AdaRMS gains an INT8 norm output and an FP16 branch input, the gated GELU
// reads the FP16 gate/up slab and emits the INT8 the down projection consumes,
// and the decoder rotary reads the FP16 QKV slab.
#include "kernel_operator.h"
#include "lib/quantization/ascend_quant.h"
using namespace AscendC;

namespace {
constexpr int ADA_D = 1024;

template <typename BR, typename NM>
__aicore__ inline void gated_ada_quant_impl(GM_ADDR residual, GM_ADDR branch, GM_ADDR gate,
                                  GM_ADDR gamma, GM_ADDR shift, GM_ADDR norm,
                                  GM_ADDR updated, GM_ADDR inv, int rows, int has_branch,
                                  int branch_nz) {
    constexpr int D = ADA_D;
    constexpr bool QUANT = IsSameType<NM, int8_t>::value;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> rq, bq, gq, cq, sq;
    TQue<QuePosition::VECOUT, 1> nq, uq;
    TBuf<QuePosition::VECCALC> xbuf, tbuf, gbuf, bbuf, rbuf, workbuf, qbuf;
    pipe.InitBuffer(rq, 1, 4 * D); pipe.InitBuffer(bq, 1, 2 * D); pipe.InitBuffer(gq, 1, 2 * D);
    pipe.InitBuffer(cq, 1, 2 * D); pipe.InitBuffer(sq, 1, 4 * D);
    pipe.InitBuffer(nq, 1, QUANT ? D : 2 * D); pipe.InitBuffer(uq, 1, 4 * D);
    pipe.InitBuffer(xbuf, 4 * D); pipe.InitBuffer(tbuf, 4 * D); pipe.InitBuffer(gbuf, 4 * D);
    pipe.InitBuffer(bbuf, 2 * D); pipe.InitBuffer(rbuf, 256); pipe.InitBuffer(workbuf, 4 * D);
    pipe.InitBuffer(qbuf, 8 * D);
    GlobalTensor<float> rg, sg, ug, ivg;
    GlobalTensor<bfloat16_t> gg, cg;
    GlobalTensor<BR> bg;
    GlobalTensor<NM> ng;
    rg.SetGlobalBuffer((__gm__ float*)residual); sg.SetGlobalBuffer((__gm__ float*)shift);
    ug.SetGlobalBuffer((__gm__ float*)updated); cg.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
    ng.SetGlobalBuffer((__gm__ NM*)norm);
    if (has_branch) {
        bg.SetGlobalBuffer((__gm__ BR*)branch);
        gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);
    }
    float qs = 1.0f;
    if (QUANT) {
        ivg.SetGlobalBuffer((__gm__ float*)inv);
        qs = ivg.GetValue(0);
    }
    for (int row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        auto r = rq.AllocTensor<float>(); auto c = cq.AllocTensor<bfloat16_t>();
        auto s = sq.AllocTensor<float>();
        DataCopy(r, rg[row * D], D); DataCopy(c, cg, D); DataCopy(s, sg, D);
        rq.EnQue(r); cq.EnQue(c); sq.EnQue(s);
        r = rq.DeQue<float>(); c = cq.DeQue<bfloat16_t>(); s = sq.DeQue<float>();
        auto x = xbuf.Get<float>(); auto tmp = tbuf.Get<float>(); auto gf = gbuf.Get<float>();
        auto bf = bbuf.Get<bfloat16_t>();
        auto u = uq.AllocTensor<float>(); auto n = nq.AllocTensor<NM>();
        if (has_branch) {
            auto b = bq.AllocTensor<BR>(); auto g = gq.AllocTensor<bfloat16_t>();
            // A cube kernel whose output row is narrower than one write region
            // cannot hand out its column tiles to every core safely, so the two
            // narrow projections write fractal NZ instead and the row is
            // gathered here: sixteen values every sixteen rows.
            if (branch_nz) {
                DataCopyParams bp{(uint16_t)(D / 16), 1, 15, 0};
                DataCopy(b, bg[row * 16], bp);
            } else {
                DataCopy(b, bg[row * D], D);
            }
            DataCopy(g, gg, D); bq.EnQue(b); gq.EnQue(g);
            b = bq.DeQue<BR>(); g = gq.DeQue<bfloat16_t>();
            Cast(x, b, RoundMode::CAST_NONE, D); Cast(gf, g, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x, gf, D); PipeBarrier<PIPE_V>();
            Cast(bf, tmp, RoundMode::CAST_RINT, D); PipeBarrier<PIPE_V>();
            Cast(tmp, bf, RoundMode::CAST_NONE, D); PipeBarrier<PIPE_V>();
            Add(u, r, tmp, D); PipeBarrier<PIPE_V>();
            bq.FreeTensor(b); gq.FreeTensor(g);
            Cast(bf, u, RoundMode::CAST_RINT, D);
        } else {
            Cast(bf, r, RoundMode::CAST_RINT, D);
        }
        PipeBarrier<PIPE_V>(); Cast(x, bf, RoundMode::CAST_NONE, D); PipeBarrier<PIPE_V>();
        Mul(tmp, x, x, D); PipeBarrier<PIPE_V>();
        auto rs = rbuf.Get<float>();
        ReduceSum(rs, tmp, workbuf.Get<float>(), D); PipeBarrier<PIPE_V>();
        Muls(rs, rs, 1.0f / D, 1); PipeBarrier<PIPE_V>();
        Adds(rs, rs, 1e-6f, 1); PipeBarrier<PIPE_V>();
        Rsqrt(rs, rs, 1); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float inverse = rs.GetValue(0);
        Cast(gf, c, RoundMode::CAST_NONE, D); Muls(x, x, inverse, D); PipeBarrier<PIPE_V>();
        Mul(tmp, x, gf, D); PipeBarrier<PIPE_V>();
        Cast(bf, tmp, RoundMode::CAST_RINT, D); PipeBarrier<PIPE_V>();
        Cast(tmp, bf, RoundMode::CAST_NONE, D); PipeBarrier<PIPE_V>();
        Add(tmp, tmp, s, D); PipeBarrier<PIPE_V>();
        if (QUANT) {
            // Round to the BF16 the parent would have written before scaling, so
            // the only difference from the shipped kernel is the quantiser.
            Cast(bf, tmp, RoundMode::CAST_RINT, D); PipeBarrier<PIPE_V>();
            Cast(tmp, bf, RoundMode::CAST_NONE, D); PipeBarrier<PIPE_V>();
            Muls(tmp, tmp, qs, D); PipeBarrier<PIPE_V>();
            AscendQuant(n.template ReinterpretCast<int8_t>(), tmp, qbuf.Get<uint8_t>(),
                        1.0f, 0.0f, D);
        } else {
            Cast(n.template ReinterpretCast<bfloat16_t>(), tmp, RoundMode::CAST_RINT, D);
        }
        nq.EnQue(n); uq.EnQue(u);
        rq.FreeTensor(r); cq.FreeTensor(c); sq.FreeTensor(s);
        n = nq.DeQue<NM>(); u = uq.DeQue<float>();
        DataCopy(ng[row * D], n, D);
        if (has_branch) DataCopy(ug[row * D], u, D);
        nq.FreeTensor(n); uq.FreeTensor(u);
    }
}
}  // namespace

#define GATED_ADA_QUANT_ENTRY(name, BR, NM)                                                        \
    __global__ __aicore__ void name(GM_ADDR residual, GM_ADDR branch, GM_ADDR gate,      \
        GM_ADDR gamma, GM_ADDR shift, GM_ADDR norm, GM_ADDR updated, GM_ADDR inv,        \
        int rows, int has_branch, int branch_nz) {                                                      \
        gated_ada_quant_impl<BR, NM>(residual, branch, gate, gamma, shift, norm, updated, inv,     \
                           rows, has_branch, branch_nz);                                 \
    }
GATED_ADA_QUANT_ENTRY(gated_ada_quant_bb, bfloat16_t, bfloat16_t)
GATED_ADA_QUANT_ENTRY(gated_ada_quant_bq, bfloat16_t, int8_t)
GATED_ADA_QUANT_ENTRY(gated_ada_quant_hb, half, bfloat16_t)
GATED_ADA_QUANT_ENTRY(gated_ada_quant_hq, half, int8_t)

extern "C" int flashrt_npu_gated_ada_quant(void* stream, void* residual, void* branch, void* gate,
    void* gamma, void* shift, void* norm, void* updated, void* inv,
    int rows, int has_branch, int branch_fp16, int norm_int8, int branch_nz) {
    if (!residual || !gamma || !shift || !norm || !updated || rows <= 0 ||
        rows > 2147483647 / ADA_D || (has_branch && (!branch || !gate)) ||
        (norm_int8 && !inv)) { return 1; }
    const int blocks = rows < 40 ? rows : 40;
    if (branch_fp16) {
        if (norm_int8) {
            gated_ada_quant_hq<<<blocks, nullptr, stream>>>((uint8_t*)residual, (uint8_t*)branch,
                (uint8_t*)gate, (uint8_t*)gamma, (uint8_t*)shift, (uint8_t*)norm,
                (uint8_t*)updated, (uint8_t*)inv, rows, has_branch, branch_nz);
        } else {
            gated_ada_quant_hb<<<blocks, nullptr, stream>>>((uint8_t*)residual, (uint8_t*)branch,
                (uint8_t*)gate, (uint8_t*)gamma, (uint8_t*)shift, (uint8_t*)norm,
                (uint8_t*)updated, (uint8_t*)inv, rows, has_branch, branch_nz);
        }
    } else if (norm_int8) {
        gated_ada_quant_bq<<<blocks, nullptr, stream>>>((uint8_t*)residual, (uint8_t*)branch,
            (uint8_t*)gate, (uint8_t*)gamma, (uint8_t*)shift, (uint8_t*)norm,
            (uint8_t*)updated, (uint8_t*)inv, rows, has_branch, branch_nz);
    } else {
        gated_ada_quant_bb<<<blocks, nullptr, stream>>>((uint8_t*)residual, (uint8_t*)branch,
            (uint8_t*)gate, (uint8_t*)gamma, (uint8_t*)shift, (uint8_t*)norm,
            (uint8_t*)updated, (uint8_t*)inv, rows, has_branch, branch_nz);
    }
    return 0;
}

// Gated GELU over the FP16 slab the gate/up cube wrote, quantised for the down
// projection. Columns [0,h) are the activated half, [h,2h) the multiplicand.
__global__ __aicore__ void geglu_quant_kernel(GM_ADDR gu, GM_ADDR inv, GM_ADDR out,
                                          int rows, int h) {
    constexpr int TILE = 2048;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gq, uq;
    TQue<QuePosition::VECOUT, 1> oq;
    TBuf<QuePosition::VECCALC> xb, tb, bb, scratch;
    pipe.InitBuffer(gq, 1, 2 * TILE); pipe.InitBuffer(uq, 1, 2 * TILE);
    pipe.InitBuffer(oq, 1, TILE);
    pipe.InitBuffer(xb, 4 * TILE); pipe.InitBuffer(tb, 4 * TILE);
    pipe.InitBuffer(bb, 2 * TILE); pipe.InitBuffer(scratch, 8 * TILE);
    GlobalTensor<half> gug;
    GlobalTensor<float> ivg;
    GlobalTensor<int8_t> og;
    gug.SetGlobalBuffer((__gm__ half*)gu);
    ivg.SetGlobalBuffer((__gm__ float*)inv);
    og.SetGlobalBuffer((__gm__ int8_t*)out);
    const float scale = ivg.GetValue(0);
    for (int row = GetBlockIdx(); row < rows; row += GetBlockNum()) {
        const int base = row * 2 * h;
        for (int col = 0; col < h; col += TILE) {
            const int n = h - col < TILE ? h - col : TILE;
            auto g = gq.AllocTensor<half>(); auto u = uq.AllocTensor<half>();
            DataCopy(g, gug[base + col], n); DataCopy(u, gug[base + h + col], n);
            gq.EnQue(g); uq.EnQue(u);
            g = gq.DeQue<half>(); u = uq.DeQue<half>();
            auto x = xb.Get<float>(); auto t = tb.Get<float>(); auto bf = bb.Get<bfloat16_t>();
            auto o = oq.AllocTensor<int8_t>();
            Cast(x, g, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            Mul(t, x, x, n); PipeBarrier<PIPE_V>();
            Muls(t, t, 0.044715f, n); PipeBarrier<PIPE_V>();
            Adds(t, t, 1.0f, n); PipeBarrier<PIPE_V>();
            Mul(t, t, x, n); PipeBarrier<PIPE_V>();
            Muls(t, t, -1.5957691216057308f, n); PipeBarrier<PIPE_V>();
            Exp(t, t, n); PipeBarrier<PIPE_V>();
            Adds(t, t, 1.0f, n); PipeBarrier<PIPE_V>();
            Div(t, x, t, n); PipeBarrier<PIPE_V>();
            Cast(bf, t, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
            Cast(t, bf, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            Cast(x, u, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            Mul(t, t, x, n); PipeBarrier<PIPE_V>();
            Cast(bf, t, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
            Cast(t, bf, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            Muls(t, t, scale, n); PipeBarrier<PIPE_V>();
            AscendQuant(o, t, scratch.Get<uint8_t>(), 1.0f, 0.0f, n);
            oq.EnQue(o); gq.FreeTensor(g); uq.FreeTensor(u);
            o = oq.DeQue<int8_t>();
            DataCopy(og[row * h + col], o, n);
            oq.FreeTensor(o);
        }
    }
}

extern "C" int flashrt_npu_geglu_quant(void* stream, void* gu, void* inv, void* out,
                                       int rows, int h) {
    if (!gu || !inv || !out || rows <= 0 || h <= 0 || h % 32 ||
        rows > 2147483647 / (2 * h)) { return 1; }
    geglu_quant_kernel<<<rows < 40 ? rows : 40, nullptr, stream>>>((uint8_t*)gu, (uint8_t*)inv,
        (uint8_t*)out, rows, h);
    return 0;
}

// Decoder rotary over the FP16 slab the QKV cube wrote. Identical rotation to
// the shipped kernel; only the input container changes, because fixpipe cannot
// dequantise an INT32 accumulator straight to BF16 but does reach FP16, and a
// separate cast would cost a launch on the dependent path.
__global__ __aicore__ void decoder_rope_fp16_kernel(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR keys, GM_ADDR values, int prefix, int rows) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> iq, cq, sq;
    TQue<QuePosition::VECOUT, 1> oq;
    TBuf<QuePosition::VECCALC> xfbuf, yfbuf, tmpbuf;
    pipe.InitBuffer(iq, 1, 512); pipe.InitBuffer(cq, 1, 1024);
    pipe.InitBuffer(sq, 1, 1024); pipe.InitBuffer(oq, 1, 512);
    pipe.InitBuffer(xfbuf, 1024); pipe.InitBuffer(yfbuf, 1024); pipe.InitBuffer(tmpbuf, 1024);
    GlobalTensor<half> xg;
    GlobalTensor<bfloat16_t> qg, kg, vg;
    GlobalTensor<float> cg, sg;
    xg.SetGlobalBuffer((__gm__ half*)qkv);
    qg.SetGlobalBuffer((__gm__ bfloat16_t*)query);
    kg.SetGlobalBuffer((__gm__ bfloat16_t*)keys);
    vg.SetGlobalBuffer((__gm__ bfloat16_t*)values);
    cg.SetGlobalBuffer((__gm__ float*)cos); sg.SetGlobalBuffer((__gm__ float*)sin);
    for (int unit = GetBlockIdx(); unit < rows * 10; unit += GetBlockNum()) {
        const int row = unit / 10, head = unit % 10;
        auto x = iq.AllocTensor<half>();
        DataCopy(x, xg[row * 2560 + head * 256], 256);
        iq.EnQue(x);
        x = iq.DeQue<half>();
        auto y = oq.AllocTensor<bfloat16_t>();
        auto xf = xfbuf.Get<float>(); auto yf = yfbuf.Get<float>(); auto tmp = tmpbuf.Get<float>();
        Cast(xf, x, RoundMode::CAST_NONE, 256); PipeBarrier<PIPE_V>();
        if (head < 9) {
            auto c = cq.AllocTensor<float>(); auto s = sq.AllocTensor<float>();
            DataCopy(c, cg[(prefix + row) * 256], 256);
            DataCopy(s, sg[(prefix + row) * 256], 256);
            cq.EnQue(c); sq.EnQue(s); c = cq.DeQue<float>(); s = sq.DeQue<float>();
            Mul(yf, xf, c, 256); PipeBarrier<PIPE_V>();
            Mul(tmp, xf[128], s, 128); PipeBarrier<PIPE_V>();
            Sub(yf, yf, tmp, 128); PipeBarrier<PIPE_V>();
            Mul(tmp, xf, s[128], 128); PipeBarrier<PIPE_V>();
            Add(yf[128], yf[128], tmp, 128); PipeBarrier<PIPE_V>();
            Cast(y, yf, RoundMode::CAST_RINT, 256);
            cq.FreeTensor(c); sq.FreeTensor(s);
        } else {
            Cast(y, xf, RoundMode::CAST_RINT, 256);
        }
        oq.EnQue(y); iq.FreeTensor(x); y = oq.DeQue<bfloat16_t>();
        if (head < 8) DataCopy(qg[row * 2048 + head * 256], y, 256);
        else if (head == 8) DataCopy(kg[(prefix + row) * 256], y, 256);
        else DataCopy(vg[(prefix + row) * 256], y, 256);
        oq.FreeTensor(y);
    }
}

extern "C" int flashrt_npu_decoder_rope_fp16(void* stream, void* qkv, void* cos, void* sin,
    void* query, void* keys, void* values, int prefix, int rows) {
    if (!qkv || !cos || !sin || !query || !keys || !values || prefix < 0 || rows <= 0 ||
        rows > 2147483647 / 2560 || prefix > 2147483647 / 256 - rows) { return 1; }
    decoder_rope_fp16_kernel<<<40, nullptr, stream>>>((uint8_t*)qkv, (uint8_t*)cos, (uint8_t*)sin,
        (uint8_t*)query, (uint8_t*)keys, (uint8_t*)values, prefix, rows);
    return 0;
}
