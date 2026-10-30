#include <flashrt/apple/pipeline.h>
#include "attention/attention_d72_bf16.h"
#include "attention/attention_simd_bf16.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace flashrt::apple;

namespace {
uint16_t pack(float x) {
    uint32_t bits;
    std::memcpy(&bits,&x,4);
    return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
float unpack(uint16_t x) {
    uint32_t bits=uint32_t(x)<<16;
    float value;
    std::memcpy(&value,&bits,4);
    return value;
}
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return (values[(values.size()-1)/2]+values[values.size()/2])/2;
}

void check(Context& context, const Kernel& tiled, const Kernel& scalar,
           uint32_t views, uint32_t seq, uint32_t heads, bool benchmark) {
    const uint32_t dim=heads*72;
    const std::size_t count=std::size_t(views)*seq*dim, guard=16;
    uint32_t shape[6]={views,seq,dim,dim,heads,1};
    auto input=context.buffer(count*3*2), output=context.buffer((count+guard*2)*2);
    auto params=context.buffer(sizeof(shape)); params.write(shape,sizeof(shape));
    auto plan=context.pipeline({{tiled,{{input},{output,guard*2,Access::write},{params}},
        {(seq+15)/16,heads,views},{128},"D72 tiled attention"}});
    std::mt19937 generator(24);
    std::uniform_real_distribution<float> random(-1.0f,1.0f);
    std::vector<uint16_t> qkv(count*3), result(count+guard*2), direct;
    const uint16_t sentinel=0x4b53;
    for (int variant=0;variant<3;++variant) {
        // Uniform softmax, ordinary logits, and concentrated logits. Reuse
        // the same direct/ICB plans with changed inputs and multiple views.
        for (std::size_t i=0;i<qkv.size();++i) {
            bool is_value=(i%(3*dim))>=2*dim;
            qkv[i]=pack(random(generator)*(is_value ? 1.0f : (variant==0 ? 0.0f : variant==1 ? 1.0f : 8.0f)));
        }
        input.write(qkv.data(),qkv.size()*2);
        std::vector<float> reference(count);
        std::vector<double> probability(seq);
        for (uint32_t view=0;view<views;++view)
            for (uint32_t head=0;head<heads;++head)
                for (uint32_t row=0;row<seq;++row) {
                    auto offset=[&](uint32_t r,uint32_t part,uint32_t channel) {
                        return ((view*seq+r)*3+part)*dim+head*72+channel;
                    };
                    double maximum=-INFINITY;
                    for (uint32_t k=0;k<seq;++k) {
                        double dot=0;
                        for (uint32_t c=0;c<72;++c)
                            dot+=double(unpack(qkv[offset(row,0,c)]))*unpack(qkv[offset(k,1,c)]);
                        probability[k]=dot/std::sqrt(72.0);
                        maximum=std::max(maximum,probability[k]);
                    }
                    double denom=0;
                    for (auto& p:probability) { p=std::exp(p-maximum); denom+=p; }
                    for (uint32_t c=0;c<72;++c) {
                        double value=0;
                        for (uint32_t k=0;k<seq;++k)
                            value+=probability[k]/denom*unpack(qkv[offset(k,2,c)]);
                        reference[(view*seq+row)*dim+head*72+c]=unpack(pack(float(value)));
                    }
                }
        for (auto mode:{Execution::direct,Execution::icb}) {
            std::fill(result.begin(),result.end(),sentinel);
            output.write(result.data(),result.size()*2);
            plan.run(mode);
            output.read(result.data(),result.size()*2);
            for (std::size_t i=0;i<guard;++i)
                if (result[i]!=sentinel || result[guard+count+i]!=sentinel)
                    throw std::runtime_error("D72 attention output guard overwritten");
            double squared=0, norm=0, max_error=0;
            for (std::size_t i=0;i<count;++i) {
                double actual=unpack(result[guard+i]), delta=actual-reference[i];
                if (!std::isfinite(actual)) throw std::runtime_error("Nonfinite D72 output");
                squared+=delta*delta; norm+=double(reference[i])*reference[i];
                max_error=std::max(max_error,std::abs(delta));
            }
            if (max_error>.008 || std::sqrt(squared/std::max(norm,1e-20))>.003)
                throw std::runtime_error("D72 attention disagrees with FP64 CPU oracle");
            if (mode==Execution::direct) direct=result;
            else if (direct!=result) throw std::runtime_error("D72 direct/ICB mismatch");
        }
    }
    std::cout << "PASS D72 attention views=" << views << " seq=" << seq << " heads=" << heads
              << ": FP64 oracle, uniform/sharp logits, changed inputs, guards, direct/ICB\n";
    if (benchmark) {
        auto baseline=context.pipeline({{scalar,{{input},{output,guard*2,Access::write},{params}},
            {(views*seq*heads+3)/4},{128},"D72 scalar attention"}});
        for (int i=0;i<10;++i) { baseline.run(); plan.run(); }
        std::vector<double> old_gpu,new_gpu;
        for (int i=0;i<100;++i) {
            if (i%2) { new_gpu.push_back(plan.run().gpu_us); old_gpu.push_back(baseline.run().gpu_us); }
            else { old_gpu.push_back(baseline.run().gpu_us); new_gpu.push_back(plan.run().gpu_us); }
        }
        std::cout << "D72 attention GPU p50 us: scalar=" << median(old_gpu)
                  << " tiled=" << median(new_gpu) << " speedup=" << median(old_gpu)/median(new_gpu) << '\n';
    }
}
}

int main(int argc,char** argv) {
    try {
        bool benchmark=argc==2 && std::string(argv[1])=="--benchmark";
        if (argc>1 && !benchmark) throw std::invalid_argument("Usage: attention_smoke [--benchmark]");
        if (benchmark)
            for (const char* name:{"MTL_DEBUG_LAYER","MTL_SHADER_VALIDATION","MTL_CAPTURE_ENABLED"}) {
                const char* value=std::getenv(name);
                if (value && value[0] && std::string(value)!="0")
                    throw std::invalid_argument(std::string("Disable ")+name+" for timing");
            }
        Context context;
        auto tiled=context.kernel(shaders::attention_d72_bf16_source(),"attention_d72_bf16",true);
        auto scalar=context.kernel(shaders::attention_simd_bf16_source(),"attention_bf16",true);
        for (auto shape:{std::array<uint32_t,3>{1,1,1},{2,7,3},{3,33,2},{2,255,16},{2,256,16}})
            check(context,tiled,scalar,shape[0],shape[1],shape[2],benchmark && shape[1]==256);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL D72 attention: " << e.what() << '\n';
        return 1;
    }
}
