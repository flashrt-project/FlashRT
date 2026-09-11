#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
using namespace matmul_tiling;
extern "C" int flashrt_npu_gu_tiling(void* out,int capacity,int M,int N,int K,int sm,int sn,int bm,int bn,int bk) {
 if(!out||capacity<=0||M<=0||M>4096||N!=1024||K<=0||K%32||K>65536||sm!=128||sn!=1024||bm!=128||bn!=128||bk!=256)return -1;
 auto platform=platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B4");
 if(!platform)return -2;
 MatmulApiTiling api(*platform);
 api.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,false);
 api.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_INT8,true);
 api.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32);
 api.SetOrgShape(M,N,K);api.SetShape(sm,sn,K);api.SetBias(false);
 api.SetFixSplit(bm,bn,bk);api.SetBufferSpace(-1,-1,-1);
 optiling::TCubeTiling t;
 if(api.GetTiling(t)!=0)return -3;
 int size=t.GetDataSize();if(size>capacity)return -4;t.SaveToBuffer(out,capacity);return size;
}
