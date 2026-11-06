#pragma once
#include "common/source.h"
#include <stdexcept>

namespace flashrt::apple::shaders {
// A[M,Kp], W[N,Kp], Kp=ceil(K/128)*128; signed symmetric [-127,127].
// INT32 accumulation, FP32 row/channel descale, BF16 output. The optional
// bias preserves the existing GEMM -> BF16 -> bias -> BF16 boundary.
// Caller must bound K*127*127 <= INT32_MAX and zero all padded columns.
inline std::string gemm_int8_source(int tm=32,int tn=64,int sg=2,bool bias=false) {
    if(tm<=0 || tn<=0 || sg<=0)
        throw std::invalid_argument("INT8 tiles and SIMD group count must be positive");
    return std::string(tensor_ops_preamble)+
        "constant constexpr int TM="+std::to_string(tm)+",TN="+std::to_string(tn)+",SG="+std::to_string(sg)+";\n"+R"MSL(
struct Int8Gemm { int rows,columns,inner,stride; };
kernel void gemm_int8_bf16(device int8_t* a [[buffer(0)]],
                          device int8_t* w [[buffer(1)]],
                          device bfloat* c [[buffer(2)]],
                          device const float* activation_scales [[buffer(3)]],
                          device const float* weight_scales [[buffer(4)]],
)MSL"+(bias?std::string("device const bfloat* bias [[buffer(5)]],\n"):std::string())+
        "constant Int8Gemm& p [[buffer("+std::to_string(bias?6:5)+")]],\n"+R"MSL(
                          uint2 group [[threadgroup_position_in_grid]]) {
    if(group.y*TM>=uint(p.rows) || group.x*TN>=uint(p.columns))return;
    using E=dextents<int,2>;
    tensor<device int8_t,E,tensor_inline> left(a,E(p.stride,p.rows),array<int,2>{1,p.stride});
    tensor<device int8_t,E,tensor_inline> right(w,E(p.stride,p.columns),array<int,2>{1,p.stride});
    auto l=left.slice(0,int(group.y)*TM);
    auto r=right.slice(0,int(group.x)*TN);
    constexpr auto desc=matmul2d_descriptor(TM,TN,int(dynamic_extent),false,true,false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto accum=op.get_destination_cooperative_tensor<decltype(l),decltype(r),int>();
    op.run(l,r,accum);
    for(uint i=0;i<accum.get_capacity();++i) if(accum.is_valid_element(i)) {
        auto coord=accum.get_multidimensional_index(i);
        uint n=group.x*TN+coord[0],m=group.y*TM+coord[1];
        if(m<uint(p.rows) && n<uint(p.columns)) {
            float alpha=activation_scales[m]*weight_scales[n];
            bfloat value=bfloat(float(accum[i])*alpha);
)MSL"+(bias?std::string("value=bfloat(float(value)+float(bias[n]));\n"):std::string())+R"MSL(
            c[m*p.columns+n]=value;
        }
    }
}
)MSL";
}
} // namespace flashrt::apple::shaders
