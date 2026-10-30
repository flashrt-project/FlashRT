// Standalone, opt-in tuning experiment. Keeps the production shader unchanged.
#include "gemm_oracle.h"
#include <cstdlib>
#include <iomanip>

namespace {
std::array<double,2> measure(Context& context,const Kernel& kernel,const Kernel& baseline,
                            int m,int n,int sg,int rows,int columns,int inner) {
    std::vector<uint16_t> a(rows*inner),b(inner*columns);
    for (std::size_t i=0;i<a.size();++i) a[i]=bf16(float(int(i%31)-15)/32);
    for (std::size_t i=0;i<b.size();++i) b[i]=bf16(float(int(i%29)-14)/32);
    auto left=context.buffer(a.size()*2),right=context.buffer(b.size()*2);
    auto result=context.buffer(std::size_t(rows)*columns*2),params=context.buffer(12);
    int dims[3]={rows,columns,inner}; params.write(dims,12);
    left.write(a.data(),a.size()*2);right.write(b.data(),b.size()*2);
    auto plan=context.pipeline({{kernel,{{left},{right},{result,0,Access::write},{params}},
        {std::size_t(columns+n-1)/n,std::size_t(rows+m-1)/m},{std::size_t(sg*32)},"GEMM tile benchmark"}});
    auto base=context.pipeline({{baseline,{{left},{right},{result,0,Access::write},{params}},
        {std::size_t(columns+63)/64,std::size_t(rows+(rows<=16?15:63))/(rows<=16?16:64)},{64},"Baseline GEMM"}});
    plan.set_resident(true); base.set_resident(true);
    for (int i=0;i<5;++i) { base.run(Execution::icb);plan.run(Execution::icb); }
    std::vector<double> samples,controls;
    for (int i=0;i<20;++i) {
        if (i%2) { samples.push_back(plan.run(Execution::icb).gpu_us);controls.push_back(base.run(Execution::icb).gpu_us); }
        else { controls.push_back(base.run(Execution::icb).gpu_us);samples.push_back(plan.run(Execution::icb).gpu_us); }
    }
    std::sort(samples.begin(),samples.end());std::sort(controls.begin(),controls.end());
    return {(controls[9]+controls[10])/2,(samples[9]+samples[10])/2};
}
}
int main(int argc,char** argv) {
    try {
        for (const char* name:{"MTL_DEBUG_LAYER","MTL_SHADER_VALIDATION","MTL_CAPTURE_ENABLED"}) {
            const char* value=std::getenv(name);
            if (value && value[0] && std::string(value)!="0") throw std::runtime_error(std::string("Disable ")+name);
        }
        Context context;
        std::cout << "Device: " << context.device_name() << std::endl;
        std::string stage=argc>1 ? argv[1] : "prefix";
        if(stage!="prefix" && stage!="decode" && stage!="vision")
            throw std::invalid_argument("expected prefix, decode or vision");
        bool small=stage=="decode";
        auto baseline=context.kernel(shaders::gemm_bf16_source(small?16:64,64,2),"gemm_bf16",true);
        std::vector<std::array<int,3>> configs=small ?
            std::vector<std::array<int,3>>{{16,32,1},{16,64,1},{16,128,1},{16,64,2},
                {16,128,2},{16,256,2},{16,128,4},{8,64,1},{8,128,2}} :
            std::vector<std::array<int,3>>{{64,64,2},{64,128,2},{128,64,2},{128,64,4},
                {128,128,4},{64,128,4},{64,256,4},{32,128,2}};
        std::vector<std::array<int,3>> shapes;
        if(stage=="prefix") shapes={{522,32768,2048},{522,2048,16384},{522,2560,2048},{522,2048,2048}};
        else if(small) shapes={{10,1024,4096},{10,8192,1024},{10,1024,2048},{10,2560,1024}};
        else shapes={{512,3456,1152},{512,1152,1152},{512,4304,1152},{512,1152,4304}};
        for(auto config:configs) {
            int m=config[0],n=config[1],sg=config[2];
            auto kernel=context.kernel(shaders::gemm_bf16_source(m,n,sg),"gemm_bf16",true);
            check(context,kernel,13,20,9,m,n,sg);
            check(context,kernel,17,133,129,m,n,sg);
            for(auto shape:shapes) {
                auto us=measure(context,kernel,baseline,m,n,sg,shape[0],shape[1],shape[2]);
                std::cout << "RESULT " << m << ' ' << n << ' ' << sg << ' '
                          << shape[0] << ' ' << shape[1] << ' ' << shape[2] << ' '
                          << std::fixed << std::setprecision(3) << us[0] << ' ' << us[1] << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';return 1;
    }
}
