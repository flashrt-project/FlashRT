#include "kernel_operator.h"
using namespace AscendC;

__global__ __aicore__ void decoder_rope_kernel(GM_ADDR qkv, GM_ADDR cos,
    GM_ADDR sin, GM_ADDR query, GM_ADDR keys, GM_ADDR values, int prefix, int rows) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> iq, cq, sq;
    TQue<QuePosition::VECOUT, 1> oq;
    TBuf<QuePosition::VECCALC> xfbuf, yfbuf, tmpbuf;
    pipe.InitBuffer(iq, 1, 512); pipe.InitBuffer(cq, 1, 1024);
    pipe.InitBuffer(sq, 1, 1024); pipe.InitBuffer(oq, 1, 512);
    pipe.InitBuffer(xfbuf, 1024); pipe.InitBuffer(yfbuf, 1024); pipe.InitBuffer(tmpbuf, 1024);
    GlobalTensor<bfloat16_t> xg, qg, kg, vg;
    GlobalTensor<float> cg, sg;
    xg.SetGlobalBuffer((__gm__ bfloat16_t*)qkv);qg.SetGlobalBuffer((__gm__ bfloat16_t*)query);
    kg.SetGlobalBuffer((__gm__ bfloat16_t*)keys);vg.SetGlobalBuffer((__gm__ bfloat16_t*)values);
    cg.SetGlobalBuffer((__gm__ float*)cos);sg.SetGlobalBuffer((__gm__ float*)sin);
    for (int unit=GetBlockIdx(); unit<rows*10; unit+=GetBlockNum()) {
        const int row=unit/10, head=unit%10;
        auto x=iq.AllocTensor<bfloat16_t>(); DataCopy(x,xg[row*2560+head*256],256);iq.EnQue(x);
        x=iq.DeQue<bfloat16_t>();
        auto y=oq.AllocTensor<bfloat16_t>();
        auto xf=xfbuf.Get<float>();auto yf=yfbuf.Get<float>();auto tmp=tmpbuf.Get<float>();
        Cast(xf,x,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
        if (head<9) {
            auto c=cq.AllocTensor<float>();auto s=sq.AllocTensor<float>();
            DataCopy(c,cg[(prefix+row)*256],256);DataCopy(s,sg[(prefix+row)*256],256);
            cq.EnQue(c);sq.EnQue(s);c=cq.DeQue<float>();s=sq.DeQue<float>();
            Mul(yf,xf,c,256);PipeBarrier<PIPE_V>();
            Mul(tmp,xf[128],s,128);PipeBarrier<PIPE_V>();
            Sub(yf,yf,tmp,128);PipeBarrier<PIPE_V>();
            Mul(tmp,xf,s[128],128);PipeBarrier<PIPE_V>();
            Add(yf[128],yf[128],tmp,128);PipeBarrier<PIPE_V>();
            Cast(y,yf,RoundMode::CAST_RINT,256);
            cq.FreeTensor(c);sq.FreeTensor(s);
        } else { Cast(y,xf,RoundMode::CAST_RINT,256); }
        oq.EnQue(y);iq.FreeTensor(x);y=oq.DeQue<bfloat16_t>();
        if(head<8)DataCopy(qg[row*2048+head*256],y,256);
        else if(head==8)DataCopy(kg[(prefix+row)*256],y,256);
        else DataCopy(vg[(prefix+row)*256],y,256);
        oq.FreeTensor(y);
    }
}
extern "C" int flashrt_npu_decoder_rope(void* stream,void* qkv,void* cos,void* sin,
                                   void* query,void* keys,void* values,int prefix,int rows) {
    if(!qkv||!cos||!sin||!query||!keys||!values||prefix<0||rows<=0||rows>2147483647/2560||prefix>2147483647/256-rows)return 1;
    decoder_rope_kernel<<<40,nullptr,stream>>>((uint8_t*)qkv,(uint8_t*)cos,(uint8_t*)sin,
        (uint8_t*)query,(uint8_t*)keys,(uint8_t*)values,prefix,rows);return 0;
}
