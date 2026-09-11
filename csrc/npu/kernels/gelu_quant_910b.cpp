#include "kernel_operator.h"
#include "lib/quantization/ascend_quant.h"
using namespace AscendC;
__global__ __aicore__ void gelu_quant_kernel(GM_ADDR gate,GM_ADDR up,GM_ADDR inverse,GM_ADDR output,int rows,int columns) {
 constexpr int tile=8192;
 TPipe pipe;TQue<QuePosition::VECIN,1> gq,uq;TQue<QuePosition::VECOUT,1> oq;
 TBuf<QuePosition::VECCALC> xb,tb,bb,scratch;
 pipe.InitBuffer(gq,1,2*tile);pipe.InitBuffer(uq,1,2*tile);pipe.InitBuffer(oq,1,tile);
 pipe.InitBuffer(xb,4*tile);pipe.InitBuffer(tb,4*tile);pipe.InitBuffer(bb,2*tile);pipe.InitBuffer(scratch,8*tile);
 GlobalTensor<bfloat16_t> gg,ug;GlobalTensor<float> ig;GlobalTensor<int8_t> og;
 gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);ug.SetGlobalBuffer((__gm__ bfloat16_t*)up);
 ig.SetGlobalBuffer((__gm__ float*)inverse);og.SetGlobalBuffer((__gm__ int8_t*)output);
 for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
  const float scale=ig.GetValue(row);
  for(int col=0;col<columns;col+=tile) {
   int n=columns-col<tile?columns-col:tile;auto g=gq.AllocTensor<bfloat16_t>();auto u=uq.AllocTensor<bfloat16_t>();
   DataCopy(g,gg[row*columns+col],n);DataCopy(u,ug[row*columns+col],n);gq.EnQue(g);uq.EnQue(u);
   g=gq.DeQue<bfloat16_t>();u=uq.DeQue<bfloat16_t>();auto x=xb.Get<float>();auto t=tb.Get<float>();auto bf=bb.Get<bfloat16_t>();auto out=oq.AllocTensor<int8_t>();
   Cast(x,g,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
   Mul(t,x,x,n);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,n);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,n);PipeBarrier<PIPE_V>();
   Mul(t,t,x,n);PipeBarrier<PIPE_V>();Muls(t,t,-1.5957691216057308f,n);PipeBarrier<PIPE_V>();Exp(t,t,n);PipeBarrier<PIPE_V>();
   Adds(t,t,1.0f,n);PipeBarrier<PIPE_V>();Div(t,x,t,n);PipeBarrier<PIPE_V>();Cast(bf,t,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
   Cast(t,bf,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();Cast(x,u,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();Mul(t,t,x,n);PipeBarrier<PIPE_V>();Cast(bf,t,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
   Cast(t,bf,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();Muls(t,t,scale,n);PipeBarrier<PIPE_V>();
   AscendQuant(out,t,scratch.Get<uint8_t>(),1.0f,0.0f,n);oq.EnQue(out);gq.FreeTensor(g);uq.FreeTensor(u);
   out=oq.DeQue<int8_t>();DataCopy(og[row*columns+col],out,n);oq.FreeTensor(out);
  }
 }
}
extern "C" int flashrt_npu_gelu_mul_quant(void* stream,void* gate,void* up,void* inverse,void* output,int rows,int columns) {
 if(!gate||!up||!inverse||!output||rows<=0||columns<=0||columns%32||rows>2147483647/columns)return 1;
 gelu_quant_kernel<<<rows<40?rows:40,nullptr,stream>>>((uint8_t*)gate,(uint8_t*)up,(uint8_t*)inverse,(uint8_t*)output,rows,columns);return 0;
}
