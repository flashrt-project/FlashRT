#include <flashrt/apple/pipeline.h>
#include "kernel_registry.h"
#include <iostream>
#include <set>

using namespace flashrt::apple;

int main() {
    try {
        Context context;
        std::set<std::string> names;
        // Compile every registered entry, including aliases and diagnostic
        // kernels that a particular complete model plan may not exercise.
        for (const auto& source : shaders::kernel_registry) {
            if (!names.emplace(source.name).second)
                throw std::runtime_error("Duplicate Apple kernel: " + std::string(source.name));
            const auto& resolved=shaders::find_kernel_source(source.name);
            auto kernel=context.kernel(resolved.build_source(),resolved.entry,true);
        }
        for (const char* unknown : {"tr_unknown", "gemm_fp8", "tr_copy"}) {
            bool rejected=false;
            try { (void)shaders::find_kernel_source(unknown); }
            catch (const std::invalid_argument&) { rejected=true; }
            if (!rejected) throw std::runtime_error("Unknown kernel was accepted");
        }
        std::cout << "PASS " << names.size()
                  << " registered Metal entries and unknown-kernel rejection\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL kernel registry: " << e.what() << '\n';
        return 1;
    }
}
