// Exact uint8 normalization lookup followed by CHW patch gathering.
#include "kernel_operator.h"
using namespace AscendC;
__global__ __aicore__ void image_patches_kernel(GM_ADDR raw,GM_ADDR indices,GM_ADDR lut,GM_ADDR output,int views) {
 TPipe pipe;TQue<QuePosition::VECIN,1> iq;TQue<QuePosition::VECOUT,1> oq;
 TBuf<QuePosition::VECCALC> hb,fb,bb,ib,lb;
 pipe.InitBuffer(iq,1,896);pipe.InitBuffer(oq,1,608*4);
 pipe.InitBuffer(hb,896*2);pipe.InitBuffer(fb,896*4);pipe.InitBuffer(bb,896*4);pipe.InitBuffer(lb,256*4);pipe.InitBuffer(ib,608*4);
 GlobalTensor<uint8_t> ig;GlobalTensor<uint32_t> ix;GlobalTensor<float> og,lg;
 ig.SetGlobalBuffer((__gm__ uint8_t*)raw);ix.SetGlobalBuffer((__gm__ uint32_t*)indices);og.SetGlobalBuffer((__gm__ float*)output);lg.SetGlobalBuffer((__gm__ float*)lut);
 auto values=lb.Get<float>();DataCopy(values,lg,256);auto index=ib.Get<uint32_t>();DataCopy(index,ix,608);PipeBarrier<PIPE_ALL>();
 for(int patch=GetBlockIdx();patch<views*256;patch+=GetBlockNum()) {
  int v=patch/256,ph=(patch%256)/16,pw=patch%16;
  auto in=iq.AllocTensor<uint8_t>();
  DataCopyExtParams input_params{14,42,630,0,0};DataCopyPadExtParams<uint8_t> padding{true,0,22,0};
  DataCopyPad(in,ig[v*224*224*3+ph*14*224*3+pw*14*3],input_params,padding);iq.EnQue(in);
  in=iq.DeQue<uint8_t>();auto halfs=hb.Get<half>();auto floats=fb.Get<float>();auto offsets=bb.Get<int32_t>();
  Cast(halfs,in,RoundMode::CAST_NONE,896);PipeBarrier<PIPE_V>();Cast(offsets,halfs,RoundMode::CAST_RINT,896);PipeBarrier<PIPE_V>();
  auto byte_offsets=offsets.ReinterpretCast<uint32_t>();ShiftLeft(byte_offsets,byte_offsets,uint32_t(2),896);PipeBarrier<PIPE_V>();
  Gather(floats,values,byte_offsets,0,896);PipeBarrier<PIPE_V>();
  auto out=oq.AllocTensor<float>();Gather(out,floats,index,0,608);oq.EnQue(out);iq.FreeTensor(in);
  out=oq.DeQue<float>();DataCopyExtParams output_params{1,588*4,0,0,0};DataCopyPad(og[patch*588],out,output_params);oq.FreeTensor(out);
 }
}
extern "C" int flashrt_npu_image_patches(void* stream,void* raw,void* indices,void* lut,void* output,int views) {
 if(!raw||!indices||!lut||!output||views<=0||views>3)return 1;
 image_patches_kernel<<<40,nullptr,stream>>>((uint8_t*)raw,(uint8_t*)indices,(uint8_t*)lut,(uint8_t*)output,views);return 0;
}
