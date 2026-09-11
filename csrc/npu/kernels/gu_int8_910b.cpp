// INT8 gate/up GEMM with a BF16-rounded GELU product and static INT8 output.
// Cube/Vector exchange uses two bounded GM slots per Cube core.
// The caller owns scratch exclusively for the duration of execution.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/quantization/ascend_quant.h"
using namespace AscendC;
using A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t,false>;
using B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t,true>;
using O=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
__global__ __aicore__ void flashrt_gu_int8_kernel(GM_ADDR aa,GM_ADDR bb,GM_ADDR ws,GM_ADDR acts,GM_ADDR inverse,GM_ADDR indices,GM_ADDR output,GM_ADDR scratch,GM_ADDR tt,int M,int H,int K,uint64_t sync){
 SetSyncBaseAddr(sync);
 constexpr int SM=128,SN=1024,R=8,L=4096,GH=SN/2;
 TPipe pipe;TCubeTiling t;
 for(int i=0;i<sizeof(TCubeTiling)/4;++i)reinterpret_cast<uint32_t*>(&t)[i]=reinterpret_cast<__gm__ uint32_t*>(tt)[i];
 Matmul<A,B,O,O,CFG_MDL> mm;
 GlobalTensor<int8_t> ag,bg,og;GlobalTensor<int32_t> work;GlobalTensor<float> wg,actg,invg;GlobalTensor<uint32_t> ixg;
 ag.SetGlobalBuffer((__gm__ int8_t*)aa);bg.SetGlobalBuffer((__gm__ int8_t*)bb);og.SetGlobalBuffer((__gm__ int8_t*)output);work.SetGlobalBuffer((__gm__ int32_t*)scratch);
 wg.SetGlobalBuffer((__gm__ float*)ws);actg.SetGlobalBuffer((__gm__ float*)acts);invg.SetGlobalBuffer((__gm__ float*)inverse);ixg.SetGlobalBuffer((__gm__ uint32_t*)indices);
 TQue<QuePosition::VECIN,1> iq;TQue<QuePosition::VECOUT,1> oq;
 TBuf<QuePosition::VECCALC> fb,gb,ub,tb,bbuf,ib,sb,ab,ob,qb;
 if ASCEND_IS_AIC {mm.Init(&t,&pipe);}
 if ASCEND_IS_AIV {
  pipe.InitBuffer(iq,1,R*SN*4);pipe.InitBuffer(oq,1,L);
  pipe.InitBuffer(fb,R*SN*4);pipe.InitBuffer(gb,L*4);pipe.InitBuffer(ub,L*4);pipe.InitBuffer(tb,L*4);
  pipe.InitBuffer(bbuf,R*SN*2);pipe.InitBuffer(ib,L*4);pipe.InitBuffer(sb,SN*4);pipe.InitBuffer(ab,SM*4);pipe.InitBuffer(ob,SM*4);pipe.InitBuffer(qb,L*8);
  DataCopy(ib.Get<uint32_t>(),ixg,L);PipeBarrier<PIPE_ALL>();
 }
 int core=GetBlockIdx()/GetTaskRation(),nt=H/GH,mt=(M+SM-1)/SM,count=0;
 for(int tile=core;tile<mt*nt;tile+=20,++count){
  // Column index outermost. Tiles go out as tile += 20, so consecutive indices
  // are the ones running at the same time; putting the mt tiles that share a
  // column block of B on consecutive indices means all but the first read that
  // block from L2. With the row index outermost each column block is fetched
  // once per M tile and the kernel sits MTE2 bound at 0.74. Each tile
  // accumulates on its own, so the walk order cannot change the arithmetic.
  int ni=tile/mt,mi=tile%mt,m=mi*SM,n=ni*SN,cm=M-m<SM?M-m:SM,slot=count%2;
  int off=(core*2+slot)*SM*SN;
  if ASCEND_IS_AIC {
   if(count>=2)WaitEvent(6+slot);
   mm.SetTensorA(ag[m*K]);mm.SetTensorB(bg[n*K],true);mm.SetTail(cm,SN);
   mm.IterateAll(work[off],0,true);mm.End();NotifyEvent<PIPE_FIX>(4+slot);
  }
  if ASCEND_IS_AIV {
   WaitEvent(4+slot);
   auto scales=sb.Get<float>();DataCopy(scales,wg[n],SN);
   auto as=ab.Get<float>();auto os=ob.Get<float>();
   DataCopyExtParams sp{1,(uint32_t)(cm*4),0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
   DataCopyPad(as,actg[m],sp,pad);DataCopyPad(os,invg[m],sp,pad);PipeBarrier<PIPE_ALL>();
   for(int r0=GetSubBlockIdx()*R;r0<cm;r0+=2*R){
    int rows=cm-r0<R?cm-r0:R,len=rows*GH,total=rows*SN;
    auto in=iq.AllocTensor<int32_t>();DataCopy(in,work[off+r0*SN],total);iq.EnQue(in);in=iq.DeQue<int32_t>();
    auto f=fb.Get<float>();auto g=gb.Get<float>();auto u=ub.Get<float>();auto tmp=tb.Get<float>();auto bf=bbuf.Get<bfloat16_t>();auto mult=in.ReinterpretCast<float>();
    Cast(f,in,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();
    for(int r=0;r<rows;++r){
     Muls(mult[r*SN],scales,as.GetValue(r0+r),SN);
    }
    PipeBarrier<PIPE_V>();Mul(f,f,mult,total);PipeBarrier<PIPE_V>();
    Cast(bf,f,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(f,bf,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    Gather(g,f,ib.Get<uint32_t>(),0,len);Gather(u,f[GH],ib.Get<uint32_t>(),0,len);PipeBarrier<PIPE_V>();
    Mul(tmp,g,g,len);PipeBarrier<PIPE_V>();Muls(tmp,tmp,0.044715f,len);PipeBarrier<PIPE_V>();Adds(tmp,tmp,1.0f,len);PipeBarrier<PIPE_V>();
    Mul(tmp,tmp,g,len);PipeBarrier<PIPE_V>();Muls(tmp,tmp,-1.5957691216057308f,len);PipeBarrier<PIPE_V>();Exp(tmp,tmp,len);PipeBarrier<PIPE_V>();
    Adds(tmp,tmp,1.0f,len);PipeBarrier<PIPE_V>();Div(tmp,g,tmp,len);PipeBarrier<PIPE_V>();Cast(bf,tmp,RoundMode::CAST_RINT,len);PipeBarrier<PIPE_V>();
    Cast(tmp,bf,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_V>();Mul(tmp,tmp,u,len);PipeBarrier<PIPE_V>();Cast(bf,tmp,RoundMode::CAST_RINT,len);PipeBarrier<PIPE_V>();Cast(tmp,bf,RoundMode::CAST_NONE,len);PipeBarrier<PIPE_V>();
    for(int r=0;r<rows;++r){Muls(tmp[r*GH],tmp[r*GH],os.GetValue(r0+r),GH);}
    PipeBarrier<PIPE_V>();auto q=oq.AllocTensor<int8_t>();AscendQuant(q,tmp,qb.Get<uint8_t>(),1.0f,0.0f,len);oq.EnQue(q);iq.FreeTensor(in);
    q=oq.DeQue<int8_t>();DataCopyParams params{(uint16_t)rows,(uint16_t)(GH/32),0,(uint16_t)((H-GH)/32)};DataCopy(og[(m+r0)*H+ni*GH],q,params);oq.FreeTensor(q);
   }
   NotifyEvent<PIPE_MTE3>(6+slot);
  }
 }
 if ASCEND_IS_AIC {for(int i=count>1?count-2:0;i<count;++i)WaitEvent(6+i%2);}
}
extern "C" int rtGetC2cCtrlAddr(uint64_t*,uint32_t*);
extern "C" int flashrt_npu_gu_int8(void* stream,void* a,void* b,void* ws,void* acts,void* inv,void* ix,void* out,void* scratch,void* tiling,int M,int H,int K){
 if(!a||!b||!ws||!acts||!inv||!ix||!out||!scratch||!tiling||M<=0||H%512||H<=0||K<=0||K%32||M>4096||H>65536||K>65536||2LL*H*K>2147483647LL)return 1;
 uint64_t sync=0;uint32_t len=0;int rc=rtGetC2cCtrlAddr(&sync,&len);if(rc)return rc;
 flashrt_gu_int8_kernel<<<20,nullptr,stream>>>((uint8_t*)a,(uint8_t*)b,(uint8_t*)ws,(uint8_t*)acts,(uint8_t*)inv,(uint8_t*)ix,(uint8_t*)out,(uint8_t*)scratch,(uint8_t*)tiling,M,H,K,sync);return 0;
}

#include "../abi.h"
