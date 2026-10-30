#include "gemm_oracle.h"

int main() {
    try {
        Context context;
        // Compile the BF16 GEMM family; registry_smoke covers the other sources.
        std::cout << "Compiling BF16 shaders on " << context.device_name() << std::endl;
        auto gemm=context.kernel(shaders::gemm_bf16_source(),"gemm_bf16",true);
        auto fused=context.kernel(shaders::gemm_bf16_source(),"gemm_bias_bf16",true);
        for (auto shape : {std::array<int,3>{13,20,9}, {512,3456,1152},
                           {512,1152,1152}, {512,4304,1152}, {512,1152,4304}}) {
            check(context,gemm,shape[0],shape[1],shape[2]);
            check(context,fused,shape[0],shape[1],shape[2],64,64,2,true);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL BF16 shader/GEMM preflight: " << e.what() << '\n';
        return 1;
    }
}
