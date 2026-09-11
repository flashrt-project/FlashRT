#include "kernel_operator.h"
using namespace AscendC;
__global__ __aicore__ void gated_ada_kernel(GM_ADDR residual,GM_ADDR branch,
    GM_ADDR gate,GM_ADDR gamma,GM_ADDR shift,GM_ADDR norm,GM_ADDR updated,
    int rows,int has_branch) {
    constexpr int D=1024;
    TPipe pipe;
    TQue<QuePosition::VECIN,1> rq,bq,gq,cq,sq;
    TQue<QuePosition::VECOUT,1> nq,uq;
    TBuf<QuePosition::VECCALC> xbuf,tbuf,gbuf,bbuf,rbuf,workbuf;
    pipe.InitBuffer(rq,1,4*D);pipe.InitBuffer(bq,1,2*D);pipe.InitBuffer(gq,1,2*D);
    pipe.InitBuffer(cq,1,2*D);pipe.InitBuffer(sq,1,4*D);
    pipe.InitBuffer(nq,1,2*D);pipe.InitBuffer(uq,1,4*D);
    pipe.InitBuffer(xbuf,4*D);pipe.InitBuffer(tbuf,4*D);pipe.InitBuffer(gbuf,4*D);
    pipe.InitBuffer(bbuf,2*D);pipe.InitBuffer(rbuf,256);pipe.InitBuffer(workbuf,4*D);
    GlobalTensor<float> rg,sg,ug;GlobalTensor<bfloat16_t> bg,gg,cg,ng;
    rg.SetGlobalBuffer((__gm__ float*)residual);sg.SetGlobalBuffer((__gm__ float*)shift);
    ug.SetGlobalBuffer((__gm__ float*)updated);cg.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
    ng.SetGlobalBuffer((__gm__ bfloat16_t*)norm);
    if(has_branch){bg.SetGlobalBuffer((__gm__ bfloat16_t*)branch);gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);}
    for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        auto r=rq.AllocTensor<float>();auto c=cq.AllocTensor<bfloat16_t>();auto s=sq.AllocTensor<float>();
        DataCopy(r,rg[row*D],D);DataCopy(c,cg,D);DataCopy(s,sg,D);
        rq.EnQue(r);cq.EnQue(c);sq.EnQue(s);r=rq.DeQue<float>();c=cq.DeQue<bfloat16_t>();s=sq.DeQue<float>();
        auto x=xbuf.Get<float>();auto tmp=tbuf.Get<float>();auto gf=gbuf.Get<float>();auto bf=bbuf.Get<bfloat16_t>();
        auto u=uq.AllocTensor<float>();auto n=nq.AllocTensor<bfloat16_t>();
        if(has_branch) {
            auto b=bq.AllocTensor<bfloat16_t>();auto g=gq.AllocTensor<bfloat16_t>();
            DataCopy(b,bg[row*D],D);DataCopy(g,gg,D);bq.EnQue(b);gq.EnQue(g);
            b=bq.DeQue<bfloat16_t>();g=gq.DeQue<bfloat16_t>();
            Cast(x,b,RoundMode::CAST_NONE,D);Cast(gf,g,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
            Mul(tmp,x,gf,D);PipeBarrier<PIPE_V>();Cast(bf,tmp,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
            Cast(tmp,bf,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();Add(u,r,tmp,D);PipeBarrier<PIPE_V>();
            bq.FreeTensor(b);gq.FreeTensor(g);
            Cast(bf,u,RoundMode::CAST_RINT,D);
        } else { Cast(bf,r,RoundMode::CAST_RINT,D); }
        PipeBarrier<PIPE_V>();Cast(x,bf,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Mul(tmp,x,x,D);PipeBarrier<PIPE_V>();auto rs=rbuf.Get<float>();
        ReduceSum(rs,tmp,workbuf.Get<float>(),D);PipeBarrier<PIPE_V>();
        Muls(rs,rs,1.0f/D,1);PipeBarrier<PIPE_V>();Adds(rs,rs,1e-6f,1);PipeBarrier<PIPE_V>();
        Rsqrt(rs,rs,1);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float inverse=rs.GetValue(0);
        Cast(gf,c,RoundMode::CAST_NONE,D);Muls(x,x,inverse,D);PipeBarrier<PIPE_V>();
        Mul(tmp,x,gf,D);PipeBarrier<PIPE_V>();Cast(bf,tmp,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(tmp,bf,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();Add(tmp,tmp,s,D);PipeBarrier<PIPE_V>();
        Cast(n,tmp,RoundMode::CAST_RINT,D);
        nq.EnQue(n);uq.EnQue(u);rq.FreeTensor(r);cq.FreeTensor(c);sq.FreeTensor(s);
        n=nq.DeQue<bfloat16_t>();u=uq.DeQue<float>();DataCopy(ng[row*D],n,D);
        if(has_branch)DataCopy(ug[row*D],u,D);
        nq.FreeTensor(n);uq.FreeTensor(u);
    }
}
extern "C" int flashrt_npu_gated_ada(void* stream,void* residual,void* branch,void* gate,
    void* gamma,void* shift,void* norm,void* updated,int rows,int has_branch) {
    if(!residual||!gamma||!shift||!norm||!updated||rows<=0||rows>2147483647/1024||
       (has_branch && (!branch||!gate)))return 1;
    gated_ada_kernel<<<rows<40?rows:40,nullptr,stream>>>((uint8_t*)residual,(uint8_t*)branch,
        (uint8_t*)gate,(uint8_t*)gamma,(uint8_t*)shift,(uint8_t*)norm,(uint8_t*)updated,rows,has_branch);return 0;
}
