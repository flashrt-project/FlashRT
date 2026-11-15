#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flashrt::apple {
// One fixed-shape Core ML model with a single FP16 input and output, both views of
// caller-owned host memory (Metal shared buffers). Predictions read the input view and
// leave the output in the output view; one caller at a time.
class CoreMLTask {
public:
    // compute_units: 0 CPU + Neural Engine, 1 all, 2 CPU + GPU, 3 CPU only.
    CoreMLTask(const std::string& model_path, const std::string& input, void* input_data,
               const std::string& output, void* output_data, const std::vector<std::int64_t>& shape,
               int compute_units);
    ~CoreMLTask();
    bool run(std::string& error) const;
    bool output_backed() const;   // last prediction wrote the output view directly
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace flashrt::apple
