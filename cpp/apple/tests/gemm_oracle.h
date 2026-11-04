#pragma once
#include <flashrt/apple/pipeline.h>
#include "gemm/bf16_tensor_ops.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace flashrt::apple;

namespace {
uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits+0x7fff+((bits>>16)&1))>>16);
}

void check(Context& context, const Kernel& gemm, int rows, int columns, int inner,
           int tile_m=64, int tile_n=64, int simdgroups=2, bool fused_bias=false) {
    // A[m,k]=a[m]+u[k], B[k,n]=b[n]+v[k]. The dot product has an
    // independent O(1) oracle. These binary fractions sum exactly in FP32
    // for the tested K, so the expected BF16 bits do not depend on order.
    auto a_value=[](int m, int variant) { return float(m%7-3)/8+float(variant)/4; };
    auto b_value=[](int n) { return float(n%9-4)/8; };
    float sum_u=0, sum_v=0, sum_uv=0;
    std::vector<uint16_t> a(rows*inner), b(inner*columns);
    for (int k=0;k<inner;++k) {
        float u=float(k%13-6)/16, v=float(k%11-5)/32;
        sum_u+=u; sum_v+=v; sum_uv+=u*v;
        for (int n=0;n<columns;++n) b[k*columns+n]=bf16(b_value(n)+v);
    }
    const std::size_t elements=std::size_t(rows)*columns, guard=16;
    const uint16_t sentinel=0x4b53;
    std::vector<uint16_t> output(elements+guard*2), reference(elements);
    auto left=context.buffer(a.size()*2), right=context.buffer(b.size()*2);
    auto result=context.buffer(output.size()*2), params=context.buffer(3*sizeof(int32_t));
    std::vector<uint16_t> bias(columns);
    for (int n=0;n<columns;++n) bias[n]=bf16(float(n%17-8)/128);
    auto bias_buffer=context.buffer(columns*2); bias_buffer.write(bias.data(),columns*2);
    int32_t shape[3]={rows,columns,inner};
    params.write(shape,sizeof(shape));
    right.write(b.data(),b.size()*2);
    std::vector<Binding> bindings={{left},{right},{result,guard*2,Access::write},{params}};
    if (fused_bias) bindings.push_back({bias_buffer});
    auto plan=context.pipeline({{gemm,bindings,
        {std::size_t(columns+tile_n-1)/tile_n,std::size_t(rows+tile_m-1)/tile_m},
        {std::size_t(simdgroups*32)},"BF16 GEMM preflight"}});
    for (int variant=0;variant<2;++variant) {
        for (int m=0;m<rows;++m) {
            float am=a_value(m,variant);
            for (int k=0;k<inner;++k) a[m*inner+k]=bf16(am+float(k%13-6)/16);
            for (int n=0;n<columns;++n) {
                uint16_t value=bf16(float(inner)*am*b_value(n)+am*sum_v+b_value(n)*sum_u+sum_uv);
                if (fused_bias) {
                    uint32_t bits=uint32_t(value)<<16, bias_bits=uint32_t(bias[n])<<16;
                    float rounded, bias_value;
                    std::memcpy(&rounded,&bits,4); std::memcpy(&bias_value,&bias_bits,4);
                    value=bf16(rounded+bias_value);
                }
                reference[m*columns+n]=value;
            }
        }
        left.write(a.data(),a.size()*2);
        for (auto mode : {Execution::direct, Execution::icb}) {
            std::fill(output.begin(),output.end(),sentinel);
            result.write(output.data(),output.size()*2);
            plan.run(mode);
            result.read(output.data(),output.size()*2);
            for (std::size_t i=0;i<guard;++i)
                if (output[i]!=sentinel || output[guard+elements+i]!=sentinel)
                    throw std::runtime_error("BF16 GEMM wrote outside its output bounds");
            for (std::size_t i=0;i<elements;++i)
                if (output[guard+i]!=reference[i])
                    throw std::runtime_error("BF16 GEMM oracle mismatch at element "+std::to_string(i)+
                        " for "+std::to_string(rows)+"x"+std::to_string(columns)+"x"+std::to_string(inner));
        }
    }
    std::cout << "PASS BF16 GEMM" << (fused_bias ? "+bias " : " ") << rows << "x" << columns << "x" << inner
              << " direct/ICB, changed inputs, exact oracle and output guards\n";
}
}
